import asyncio
import gc
import http
import json
import logging
import threading
import time
import traceback
from typing import Any, Callable

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol.

    Supports hot-reload of policy weights via GET /reload endpoint.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        policy_factory: Callable[..., tuple[_base_policy.BasePolicy, dict[str, Any] | None]] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._policy_factory = policy_factory
        self._policy_lock = threading.RLock()
        self._reloading = threading.Event()
        self._reloading.set()  # "not reloading" — waiters pass through
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
            open_timeout=300,
        ) as server:
            await server.serve_forever()

    # ------------------------------------------------------------------
    # HTTP request routing (before WebSocket upgrade)
    # ------------------------------------------------------------------

    async def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if request.path.startswith("/reload"):
            return await self._handle_reload(connection, request)
        return None

    async def _handle_reload(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response:
        if self._policy_factory is None:
            body = json.dumps({"status": "error", "message": "reload not supported (no policy_factory)"})
            return connection.respond(http.HTTPStatus(501), body + "\n")

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(request.path)
        params = parse_qs(parsed.query)

        checkpoint_dir = _first_param(params, "checkpoint_dir")
        config_name = _first_param(params, "config_name")
        config_yaml = _first_param(params, "config_yaml")
        pytorch_device = _first_param(params, "pytorch_device")
        default_prompt = _first_param(params, "default_prompt")

        if not checkpoint_dir:
            body = json.dumps({"status": "error", "message": "checkpoint_dir is required"})
            return connection.respond(http.HTTPStatus.BAD_REQUEST, body + "\n")

        logger.info("Reloading policy: checkpoint_dir=%s config_name=%s config_yaml=%s", checkpoint_dir, config_name, config_yaml)

        try:
            new_policy, new_metadata = await asyncio.to_thread(
                self._reload_sync,
                checkpoint_dir=checkpoint_dir,
                config_name=config_name,
                config_yaml=config_yaml,
                pytorch_device=pytorch_device,
                default_prompt=default_prompt,
            )
        except Exception as exc:
            logger.error("Reload failed: %s", exc, exc_info=True)
            body = json.dumps({"status": "error", "message": str(exc)})
            return connection.respond(http.HTTPStatus.INTERNAL_SERVER_ERROR, body + "\n")

        with self._policy_lock:
            self._policy = new_policy
            if new_metadata:
                self._metadata.update(new_metadata)

        logger.info("Policy reloaded successfully")
        body = json.dumps({"status": "ok", "metadata": self._metadata})
        return connection.respond(http.HTTPStatus.OK, body + "\n")

    # ------------------------------------------------------------------
    # Synchronous reload (runs in thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _reload_sync(
        self,
        *,
        checkpoint_dir: str,
        config_name: str | None,
        config_yaml: str | None,
        pytorch_device: str | None,
        default_prompt: str | None,
    ) -> tuple[_base_policy.BasePolicy, dict[str, Any] | None]:
        assert self._policy_factory is not None

        self._reloading.clear()
        old_policy = None
        try:
            # Swap out the current policy so its GPU memory can be freed.
            with self._policy_lock:
                old_policy = self._policy
                self._policy = _NullPolicy()

            result = self._load_new_policy(
                old_policy=old_policy,
                checkpoint_dir=checkpoint_dir,
                config_name=config_name,
                config_yaml=config_yaml,
                pytorch_device=pytorch_device,
                default_prompt=default_prompt,
            )
            old_policy = None  # ownership transferred; don't restore on error
            return result

        except Exception:
            # Restore old policy so the server stays functional.
            if old_policy is not None:
                logger.warning("Reload failed — restoring previous model")
                with self._policy_lock:
                    self._policy = old_policy
            raise
        finally:
            self._reloading.set()

    def _load_new_policy(
        self,
        *,
        old_policy: Any,
        checkpoint_dir: str,
        config_name: str | None,
        config_yaml: str | None,
        pytorch_device: str | None,
        default_prompt: str | None,
    ) -> tuple[_base_policy.BasePolicy, dict[str, Any] | None]:
        # --- Phase 1: Free GPU memory from old model ---
        for attr_name in list(vars(old_policy).keys()):
            if attr_name not in ("metadata", "_metadata"):
                try:
                    delattr(old_policy, attr_name)
                except Exception:
                    pass
        del old_policy
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # --- Phase 2: Try loading (JIT caches may survive) ---
        try:
            logger.info("Old model unloaded, loading new model...")
            return self._policy_factory(
                checkpoint_dir=checkpoint_dir,
                config_name=config_name,
                config_yaml=config_yaml,
                pytorch_device=pytorch_device,
                default_prompt=default_prompt,
            )
        except Exception as first_err:
            if "RESOURCE_EXHAUSTED" not in str(first_err):
                raise

        # --- Phase 3: Aggressive cleanup and retry ---
        logger.warning("OOM on first attempt, clearing all GPU buffers and retrying...")
        try:
            import jax
            jax.clear_caches()
            for dev in jax.devices():
                for buf in dev.live_buffers():
                    buf.delete()
        except Exception as exc:
            logger.debug("JAX aggressive cleanup: %s", exc)

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        return self._policy_factory(
            checkpoint_dir=checkpoint_dir,
            config_name=config_name,
            config_yaml=config_yaml,
            pytorch_device=pytorch_device,
            default_prompt=default_prompt,
        )

    # ------------------------------------------------------------------
    # WebSocket inference handler
    # ------------------------------------------------------------------

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        with self._policy_lock:
            metadata = dict(self._metadata)
        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        loop = asyncio.get_running_loop()
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Wait for reload to finish without blocking the event loop.
                if not self._reloading.is_set():
                    ok = await loop.run_in_executor(None, self._reloading.wait, 300)
                    if not ok:
                        raise RuntimeError("Timed out waiting for model reload")

                infer_time = time.monotonic()
                with self._policy_lock:
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as exc:
                logger.error("Handler error: %s", exc, exc_info=True)
                try:
                    await websocket.send(traceback.format_exc())
                except Exception:
                    pass
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                except Exception:
                    pass
                break


class _NullPolicy:
    """Placeholder policy used while reloading to free the real model's memory."""

    @property
    def metadata(self) -> dict:
        return {}

    def infer(self, obs):
        raise RuntimeError("Policy is being reloaded, please retry.")


def _first_param(params: dict[str, list[str]], key: str) -> str | None:
    vals = params.get(key)
    if vals:
        return vals[0]
    return None
