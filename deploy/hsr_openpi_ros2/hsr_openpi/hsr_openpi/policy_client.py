"""Policy backends for the ROS 2 openpi HSR deployment.

Two backends are provided:

``websocket``
    Talks to ``server/serve_hsr_policy_ws.py`` (``openpi.serving.websocket_policy_server``)
    over a websocket. This is the recommended setup because openpi needs Python
    3.11 + JAX while ROS 2 Humble ships Python 3.10, so the two live in separate
    containers.

``local``
    Imports openpi in-process. Only usable when the ROS 2 node runs inside an
    environment that has openpi installed.

The websocket wire format (msgpack with a small ndarray extension) is
re-implemented here instead of importing ``openpi_client`` so that the ROS 2
package only depends on ``msgpack`` + ``websockets``. It is byte compatible with
``packages/openpi-client/src/openpi_client/msgpack_numpy.py``.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# msgpack <-> numpy (wire compatible with openpi_client.msgpack_numpy)
# --------------------------------------------------------------------------- #
def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


class BasePolicy:
    """Minimal policy interface: ``infer(obs) -> dict``."""

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}


class WebsocketPolicyClient(BasePolicy):
    """Client for ``openpi.serving.websocket_policy_server.WebsocketPolicyServer``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 8010,
        api_key: Optional[str] = None,
        *,
        connect_timeout_s: float = -1.0,
        loginfo: Callable[[str], None] = logger.info,
    ) -> None:
        import msgpack  # noqa: PLC0415  (imported lazily so `local` mode does not need it)

        self._packer_factory = functools.partial(msgpack.Packer, default=_pack_array)
        self._unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._loginfo = loginfo
        self._packer = self._packer_factory()
        self._ws, self._server_metadata = self._wait_for_server(connect_timeout_s)

    def _wait_for_server(self, timeout_s: float):
        import websockets.sync.client  # noqa: PLC0415

        self._loginfo(f"Waiting for policy server at {self._uri} ...")
        deadline = None if timeout_s is None or timeout_s < 0 else time.time() + timeout_s
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = self._unpackb(conn.recv())
                self._loginfo(f"Connected to policy server: {metadata}")
                return conn, metadata
            except (ConnectionRefusedError, OSError) as e:
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError(f"Policy server at {self._uri} did not become ready: {e}") from e
                self._loginfo("Still waiting for policy server ...")
                time.sleep(2.0)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._server_metadata or {})

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            # We are expecting bytes; a string means the server raised.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpackb(response)


class LocalPolicy(BasePolicy):
    """In-process openpi policy (requires openpi to be importable)."""

    def __init__(
        self,
        *,
        config_name: str,
        config_yaml: str,
        checkpoint_dir: str,
        loginfo: Callable[[str], None] = logger.info,
    ) -> None:
        import pathlib  # noqa: PLC0415

        from openpi.policies import policy_config  # noqa: PLC0415
        from openpi.training import config as openpi_config  # noqa: PLC0415
        from openpi.training import experiment_config  # noqa: PLC0415

        requested_yaml = (config_yaml or "").strip()
        if not requested_yaml and config_name.lower().endswith((".yaml", ".yml")):
            requested_yaml = config_name

        # A checkpoint produced by scripts/train.py embeds the merged experiment
        # YAML; prefer it so that the client does not have to know the config.
        if not requested_yaml and not config_name:
            embedded = pathlib.Path(checkpoint_dir) / "experiment_config" / "experiment_config.yaml"
            if embedded.exists():
                requested_yaml = str(embedded)

        if requested_yaml:
            loginfo(f"Load config from YAML: {requested_yaml}")
            train_config = experiment_config.load_experiment_config(pathlib.Path(requested_yaml))
        else:
            loginfo(f"Load config from registered config.py name: {config_name}")
            train_config = openpi_config.get_config(config_name)

        self._policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
        self._metadata = {"config_name": getattr(train_config, "name", config_name)}

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._policy.infer(obs)


def create_policy(
    backend: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8010,
    api_key: str = "",
    config_name: str = "",
    config_yaml: str = "",
    checkpoint_dir: str = "",
    connect_timeout_s: float = -1.0,
    loginfo: Callable[[str], None] = logger.info,
) -> BasePolicy:
    backend = (backend or "websocket").strip().lower()
    if backend == "websocket":
        return WebsocketPolicyClient(
            host=host,
            port=port,
            api_key=api_key or None,
            connect_timeout_s=connect_timeout_s,
            loginfo=loginfo,
        )
    if backend == "local":
        return LocalPolicy(
            config_name=config_name,
            config_yaml=config_yaml,
            checkpoint_dir=checkpoint_dir,
            loginfo=loginfo,
        )
    raise ValueError(f"Unknown policy backend '{backend}'. Available: websocket, local")
