"""rl/vec_adapter.py: the Python client of the in-engine environment.

EngineEnvServer speaks `PROTO|`-prefixed NDJSON (reset/step/close) over TCP to
env-ts/engine-env.ts, running inside the game engine. One connection is one
clientless Player; scripts/stack.py boots one world per agent and names its port.
"""
from __future__ import annotations

import json
import os
import select
import socket
import time
from typing import Any

ENGINE_ENV_HOST = os.environ.get("RL_ENV_HOST", "127.0.0.1")


class EnvServerError(RuntimeError):
    pass


def reset_payload(seed: int, cfg: dict | None) -> dict:
    payload = {"cmd": "reset", "seed": seed}
    if cfg:
        payload["cfg"] = cfg
    return payload


class EngineEnvServer:
    """Synchronous reset/step client for the in-engine environment over TCP.

    The engine assigns the agent; `bot_name` only labels logs.
    """

    def __init__(self, port: int, bot_name: str = "env0", boot_timeout_s: float = 180.0):
        self.bot_name = bot_name
        self.port = port
        self._buf = b""
        self.sock = self._connect(boot_timeout_s)
        self._wait_ready(boot_timeout_s)

    def _connect(self, timeout_s: float) -> socket.socket:
        # The engine needs ~15s to load the game map before it listens, and a
        # trainer launched in the same breath as the server would otherwise die on
        # a refused connection.
        deadline = time.monotonic() + timeout_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.create_connection((ENGINE_ENV_HOST, self.port), timeout=10.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                return s
            except OSError as e:  # refused while the world boots
                last = e
                time.sleep(1.0)
        raise EnvServerError(
            f"could not reach engine-env at {ENGINE_ENV_HOST}:{self.port} in {timeout_s}s "
            f"({last}). Start it with: cd vendor/rs-sdk/server/engine && "
            f"RL_ENV_PORT={self.port} RL_WORLD_ID=<n> bun run ../../../../env-ts/engine-env.ts"
        )

    # -------- protocol plumbing --------

    def _readline(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = self._buf[:nl]
                self._buf = self._buf[nl + 1 :]
                text = line.decode("utf-8", "replace")
                if text.startswith("PROTO|"):
                    return text[len("PROTO|") :]
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EnvServerError(f"timeout ({timeout_s}s) waiting for engine-env {self.bot_name}")
            r, _, _ = select.select([self.sock], [], [], remaining)
            if not r:
                continue
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EnvServerError(f"engine-env closed the connection for {self.bot_name}")
            self._buf += chunk

    def _rpc(self, payload: dict, timeout_s: float) -> dict:
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        msg = json.loads(self._readline(timeout_s))
        if msg.get("type") == "error":
            raise EnvServerError(msg.get("message", "unknown error"))
        return msg

    def _wait_ready(self, timeout_s: float) -> None:
        msg = json.loads(self._readline(timeout_s))
        if msg.get("type") != "ready":
            raise EnvServerError(f"unexpected boot message: {msg}")
        self.agent_name = msg.get("bot", self.bot_name)

    # -------- public API --------

    def reset(self, seed: int = 0, cfg: dict | None = None) -> dict:
        # No cheat round trips and no teleport verification here: reset writes
        # fields directly, so it costs a couple of ticks rather than ~45s.
        obs = self._rpc(reset_payload(seed, cfg), timeout_s=60.0)
        if obs.get("type") != "obs":
            raise EnvServerError(f"bad reset response: {obs}")
        return obs

    def step(self, action: int) -> dict:
        obs = self._rpc({"cmd": "step", "action": int(action)}, timeout_s=60.0)
        if obs.get("type") != "obs":
            raise EnvServerError(f"bad step response: {obs}")
        return obs

    def close(self) -> None:
        try:
            self._rpc({"cmd": "close"}, timeout_s=10.0)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self) -> "EngineEnvServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
