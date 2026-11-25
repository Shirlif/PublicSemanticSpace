"""ZMQ obs/gt buses (T14).

The simulator publishes on two independent endpoints:

* ``obs`` -- perception frames only; the PSS process may connect to this.
* ``gt``  -- ground truth only; a separate GT writer process connects.

The split is a process-level firewall, not a code-review convention: a PSS
process that is given only the obs endpoint cannot read GT even by accident.
Messages are JSON (small, human-inspectable); images stay on disk and are
referenced by path.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

DEFAULT_OBS_ADDR = "tcp://127.0.0.1:5555"
DEFAULT_GT_ADDR = "tcp://127.0.0.1:5556"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class BusPublisher:
    def __init__(
        self,
        obs_addr: str = DEFAULT_OBS_ADDR,
        gt_addr: str = DEFAULT_GT_ADDR,
        *,
        context=None,
    ):
        import zmq

        self._zmq = zmq
        self.context = context or zmq.Context.instance()
        self.obs = self.context.socket(zmq.PUB)
        self.obs_addr = self._bind(self.obs, obs_addr)
        # GT must not be lossy: PUB/SUB drops on a slow joiner, which is exactly
        # the failure a GT store cannot tolerate.  PUSH/PULL queues instead.
        self.gt = self.context.socket(zmq.PUSH)
        self.gt_addr = self._bind(self.gt, gt_addr)

    @staticmethod
    def _bind(socket, addr: str) -> str:
        if not addr.endswith(":*"):
            socket.bind(addr)
            return addr
        port = socket.bind_to_random_port(addr[:-2])
        return f"{addr[:-2]}:{port}"

    def publish_obs(self, payload: dict[str, Any]) -> None:
        self.obs.send_json(jsonable(payload))

    def publish_gt(self, payload: dict[str, Any]) -> None:
        self.gt.send_json(jsonable(payload))

    def close(self) -> None:
        for socket in (self.obs, self.gt):
            try:
                socket.close(linger=0)
            except Exception:
                pass


class BusSubscriber:
    def __init__(self, addr: str, *, kind: str = "sub", context=None):
        import zmq

        self._zmq = zmq
        self.context = context or zmq.Context.instance()
        if kind not in ("sub", "pull"):
            raise ValueError("kind must be 'sub' or 'pull'")
        self.kind = kind
        self.socket = self.context.socket(zmq.SUB if kind == "sub" else zmq.PULL)
        self.socket.connect(addr)
        if kind == "sub":
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.addr = addr

    def recv(self, timeout_ms: int = 1000) -> dict[str, Any] | None:
        if self.socket.poll(timeout_ms) == 0:
            return None
        return json.loads(self.socket.recv_string())

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
