#!/usr/bin/env python3
"""Mock openpi policy server used to smoke test the ROS 2 client without a GPU.

It speaks exactly the same websocket protocol as
``openpi.serving.websocket_policy_server.WebsocketPolicyServer``:

1. right after the handshake the server sends its msgpack encoded metadata,
2. for every received observation it replies with ``{"actions": (T, 11) float32}``.

The returned chunk is *relative* for the arm / head dimensions (the ROS 2 node
adds the measured joint state back), absolute for the gripper and a velocity for
the base — matching what the real pi0.5 HSR policy emits.

Usage::

    python mock_policy_server.py --port 8010 --pattern wiggle
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server

logger = logging.getLogger("mock_policy_server")

ACTION_DIM = 11
ACTION_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_theta",
]


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class MockPolicy:
    def __init__(self, pattern: str, horizon: int, amplitude: float, period_s: float, action_hz: float):
        self.pattern = pattern
        self.horizon = horizon
        self.amplitude = amplitude
        self.period_s = period_s
        self.action_hz = action_hz
        self.calls = 0

    def infer(self, obs: dict) -> np.ndarray:
        self.calls += 1
        actions = np.zeros((self.horizon, ACTION_DIM), dtype=np.float32)

        # Gripper is absolute: keep it open.
        actions[:, 5] = 1.0

        if self.pattern == "zero":
            return actions

        t0 = (self.calls - 1) * self.horizon / self.action_hz
        t = t0 + np.arange(self.horizon, dtype=np.float32) / self.action_hz
        phase = 2.0 * np.pi * t / self.period_s
        step = self.amplitude / self.action_hz

        if self.pattern in ("wiggle", "arm"):
            # Relative arm / head targets: a slow sine wave per control step.
            actions[:, 1] = -step * np.cos(phase)          # arm_flex
            actions[:, 3] = step * np.cos(phase)           # wrist_flex
            actions[:, 6] = 0.5 * step * np.cos(phase)     # head_pan
            actions[:, 0] = 0.2 * step * np.cos(phase)     # arm_lift
        if self.pattern in ("wiggle", "base"):
            # Base actions are velocities.
            actions[:, 10] = 0.3 * np.cos(phase)           # base_theta [rad/s]
            actions[:, 8] = 0.05 * np.sin(phase)           # base_x [m/s]
        return actions


async def _handler(websocket, policy: MockPolicy, metadata: dict) -> None:
    peer = getattr(websocket, "remote_address", "?")
    logger.info("client connected: %s", peer)
    await websocket.send(packb(metadata))
    try:
        async for message in websocket:
            obs = unpackb(message)
            shapes = {
                k: (getattr(v, "shape", None) or type(v).__name__) for k, v in obs.items()
            }
            logger.info("obs #%d %s", policy.calls + 1, shapes)
            actions = policy.infer(obs)
            await websocket.send(packb({"actions": actions, "policy_timing": {"infer_ms": 0.0}}))
    except Exception as e:  # noqa: BLE001
        logger.info("client disconnected (%s): %s", type(e).__name__, e)


async def _serve(args) -> None:
    policy = MockPolicy(args.pattern, args.horizon, args.amplitude, args.period, args.action_hz)
    metadata = {
        "config_name": "mock_policy",
        "checkpoint_dir": "<mock>",
        "action_dim": ACTION_DIM,
        "action_names": ACTION_NAMES,
        "action_horizon": args.horizon,
    }
    logger.info("serving mock policy on ws://%s:%d (pattern=%s)", args.host, args.port, args.pattern)
    async with ws_server.serve(
        functools.partial(_handler, policy=policy, metadata=metadata),
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--horizon", type=int, default=16, help="Action chunk length.")
    parser.add_argument(
        "--pattern", default="wiggle", choices=["zero", "wiggle", "arm", "base"], help="Motion pattern."
    )
    parser.add_argument("--amplitude", type=float, default=0.35, help="Joint speed scale [rad/s].")
    parser.add_argument("--period", type=float, default=8.0, help="Sine period [s].")
    parser.add_argument("--action-hz", type=float, default=10.0, help="Rate the chunk is meant to be played at.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
