"""Bounded supervisor checks use Python HTTP stubs, never game services."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location("stack", Path(__file__).parents[2] / "scripts/stack.py")
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


def test_stop_file_interrupts_startup_without_network_contact(tmp_path, monkeypatch):
    stop = tmp_path / "stop.requested"
    stop.touch()
    class Opener:
        def open(self, *args, **kwargs):
            pytest.fail("made readiness request after cancellation")
    monkeypatch.setattr(stack.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(stack.StackCancelled):
        stack.wait_ready("http://unused", lambda data: True, [], time.monotonic() + 600, stop_file=stop)


def test_occupied_port_refused_without_connection():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="refusing to adopt"):
            stack.require_free_ports([listener.getsockname()[1]])
        listener.settimeout(0.05)
        with pytest.raises(TimeoutError):
            listener.accept()


def test_existing_lock_is_preserved(tmp_path):
    path = tmp_path / "owner.lock"
    path.write_text("another owner")
    with pytest.raises(RuntimeError, match="existing stack lock"):
        with stack.run_lock(path, tmp_path):
            pytest.fail("adopted lock")
    assert path.read_text() == "another owner"


@pytest.mark.parametrize("child_code,timeout,expected", [
    ("raise SystemExit(7)", 5, 7),
    ("import time; time.sleep(120)", 1.5, "timeout"),
    ("raise SystemExit(0)", 5, "wrong_identity"),
    # a demo world: the command sees the declared spectator and pace, as the world does
    ("import os; raise SystemExit(3 if (os.environ['RL_DEMO_SPECTATOR'], os.environ['RL_DEMO_PACE_MS']) == "
     "('viewer', '450') else 4)", 5, "demo"),
    # a shell's stray demo settings reach neither the world nor the command of a stack without the demo flags
    ("import os; raise SystemExit(5 if not {'RL_DEMO_SPECTATOR', 'RL_DEMO_PACE_MS'} & set(os.environ) else 6)", 5, "stray"),
])
def test_stub_stack_ready_exit_and_timeout_reap_all(tmp_path, monkeypatch, child_code, timeout, expected):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    stub = tmp_path / "http_stub.py"
    stub.write_text(
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
        "import json, os\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  self.send_response(200); self.end_headers()\n"
        "  self.wfile.write(json.dumps(dict(status='running',schema=1,bootId=os.environ['RL_DEMO_BOOT_ID'],tickRateMs=48,xpRate=25)).encode())\n"
        f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n")
    real_owned, real_wait = stack.OwnedProcess, stack.wait_ready
    owned, environments = [], []
    monkeypatch.setenv("RL_DEMO_SPECTATOR", "stray")  # a shell's copies: the stack's flags alone decide
    monkeypatch.setenv("RL_DEMO_PACE_MS", "9")
    def launch(argv, **kwargs):
        environments.append(kwargs["env"])
        if len(owned) == 0:  # the engine; the command follows
            argv = [sys.executable, str(stub)]
            if expected == "wrong_identity":
                kwargs["env"] = {**kwargs["env"], "RL_DEMO_BOOT_ID": "another-launch"}
        kwargs["cwd"] = str(tmp_path)
        obj = real_owned(argv, **kwargs)
        owned.append(obj)
        return obj
    def ready(url, *args):
        return real_wait(f"http://127.0.0.1:{port}/status", *args)
    monkeypatch.setattr(stack, "ROOT", tmp_path)
    monkeypatch.setattr(stack, "prerequisites", lambda bun: None)
    monkeypatch.setattr(stack, "require_free_ports", lambda *args, **kwargs: None)
    monkeypatch.setattr(stack, "OwnedProcess", launch)
    monkeypatch.setattr(stack, "wait_ready", ready)
    output = tmp_path / "evidence"
    if expected == "wrong_identity":
        with pytest.raises(RuntimeError, match="boot identity does not match"):
            stack.run(Path(sys.executable), output, timeout, [sys.executable, "-c", child_code])
    elif expected == "timeout":
        with pytest.raises(TimeoutError):
            stack.run(Path(sys.executable), output, timeout, [sys.executable, "-c", child_code])
    elif expected == "demo":
        assert stack.run(Path(sys.executable), output, timeout, [sys.executable, "-c", child_code], lockstep=True,
                         demo=dict(spectator="viewer", pace_ms=450)) == 3
    elif expected == "stray":
        assert stack.run(Path(sys.executable), output, timeout, [sys.executable, "-c", child_code], lockstep=True) == 5
    else:
        assert stack.run(Path(sys.executable), output, timeout, [sys.executable, "-c", child_code]) == expected
    count = 1 if expected == "wrong_identity" else 2
    assert len(owned) == count
    assert all(obj.proc.poll() is not None and obj.proc.stdin.closed for obj in owned)
    assert not (tmp_path / "logs/stack.lock").exists()
    events = [json.loads(line) for line in (output / "supervisor.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "stopped" for event in events) == count
    assert any(event["event"] == "ready" for event in events) is (expected != "wrong_identity")
    start = events[0]
    assert start["settings"][0]["RL_NPC_RADIUS"] == "64"
    assert start["settings"][0]["RL_DEMO_BOOT_ID"]
    demo = [event for event in events if event["event"] == "demo"]
    assert (start["settings"][0].get("RL_DEMO_SPECTATOR"), len(demo)) == (("viewer", 1) if expected == "demo" else (None, 0))
    assert start["demo"] == (dict(spectator="viewer", pace_ms=450) if expected == "demo" else None)
    shown = [(e.get("RL_DEMO_SPECTATOR"), e.get("RL_DEMO_PACE_MS")) for e in environments]
    assert shown == [("viewer", "450") if expected == "demo" else (None, None)] * len(environments)


@pytest.mark.parametrize("options", [dict(), dict(lockstep=True, worlds=2), dict(lockstep=True, worlds=0, engines={"pinned": Path(".")}, blocks=1)])
def test_a_demo_is_one_fixed_lockstep_world(tmp_path, monkeypatch, options):
    monkeypatch.setattr(stack, "prerequisites", lambda bun: None)
    with pytest.raises(ValueError, match="one fixed lockstep world"):
        stack.run(Path(sys.executable), tmp_path / "evidence", 5, ["true"], demo=dict(spectator="viewer", pace_ms=450),
                  **options)
    assert not (tmp_path / "evidence").exists()


def test_worlds_get_distinct_ports_and_identities():
    first, second = stack.world_settings(0), stack.world_settings(1)
    assert first["RL_ENV_PORT"] == "43595" and first["RL_TELEMETRY_PORT"] == "43610"
    ports = lambda world: {world[name] for name in stack.BASE_PORTS}
    assert not ports(first) & ports(second)
    assert first["RL_DEMO_BOOT_ID"] != second["RL_DEMO_BOOT_ID"] and second["RL_WORLD_ID"] == "d2"
    # port blocks for alternating runs (--first-world): twelve worlds, no port shared by two of them
    blocks = [ports(stack.world_settings(index)) for index in range(12)]
    assert len(set().union(*blocks)) == 12 * len(stack.BASE_PORTS)
