"""Boot game worlds (engines), run one command against them, stop them all.

    python scripts/stack.py --bun "$(which bun)" --output logs/run1/stack -- \
        .venv/bin/python -m baseline.train --output logs/run1

--worlds N boots N independent worlds on offset ports, one per training agent.
The command gets them in RL_WORLDS (JSON); world 0 is also the classic single
world (RL_DEMO_BOOT_ID, default ports). --first-world K gives them the ports and
ids of worlds K..K+N-1 instead: consecutive short runs can alternate port blocks,
since a port a run has just closed refuses binds for about 30 s (TIME_WAIT).

--engine NAME=CHECKOUT (repeatable) starts a world service instead: the command
asks this supervisor for a fresh world of a named engine build (an rs-sdk
checkout), plays one episode in it and releases it, which stops it (WorldService;
the command finds it in RL_WORLD_SERVICE). No world serves a second episode, so
nothing an episode changes in the world reaches the next one.

Works on macOS, Linux and Windows. Occupied ports or an existing lock abort the
run. Only the processes this invocation launched are stopped; no installation,
adoption, PID-file killing or retries.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
from owned_process import OwnedProcess

BASE_PORTS = dict(NODE_PORT=43594, RL_ENV_PORT=43595, WEB_PORT=8888, WEB_MANAGEMENT_PORT=8898,
                  HISCORES_WEB_PORT=8889, AGENT_PORT=7780, RL_TELEMETRY_PORT=43610)
WORLD_PORT_STRIDE = 100
PORTS = tuple(BASE_PORTS.values())
MAX_BLOCKS = (65535 - max(PORTS)) // WORLD_PORT_STRIDE + 1  # the highest port block that still fits
SETTINGS = {
    "RL_SPEED_MS": "48", "NODE_TICKRATE": "48", "NODE_XPRATE": "25",
    "RL_WORLD_ID": "d1", "RL_DEMO_SERVICES": "1", "RL_TELEMETRY_PORT": "43610",
    "BUILD_LIVE_RELOAD": "false", "EASY_STARTUP": "false",
    "LOGIN_SERVER": "false", "FRIEND_SERVER": "false", "LOGGER_SERVER": "false",
    "NODE_PORT": "43594", "WEB_PORT": "8888", "WEB_MANAGEMENT_PORT": "8898",
    "AGENT_PORT": "7780", "RL_ENV_PORT": "43595", "RL_ENV_PORTS": "43595",
    "RL_ENV_HOST": "127.0.0.1",
    "RL_NPC_RADIUS": "64",
    # Engine transport, native resource capture on, experimental lifecycles off.
    "RL_TEST_HOOKS": "1", "RL_RESOURCE_CAPTURE_EXPERIMENT": "1",
    "RL_CONTINUITY_EXPERIMENT": "0", "RL_DROP_LINEAGE_EXPERIMENT": "0",
}


def _probe(port, reuse=False):
    for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        if family == socket.AF_INET6 and not socket.has_ipv6:
            continue
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            elif reuse:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            probe.bind((host, port))


def require_free_ports(ports=PORTS, settle_seconds=0.0, reuse=False):
    """Bind probes create no connections (an RL connection would admit a player).

    A port closed moments ago refuses binds while its sockets sit in TIME_WAIT
    (about 30 s on macOS, 60 s on Linux); settle_seconds waits that out. A live
    service never frees its port, so it is still refused. reuse probes as the
    engine's servers bind (SO_REUSEADDR): only a live listener is refused.
    """
    deadline = time.monotonic() + settle_seconds
    for port in ports:
        while True:
            try:
                _probe(port, reuse)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"port {port} is unavailable; refusing to adopt or stop another service") from error
                time.sleep(1)


@contextmanager
def run_lock(path, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"existing stack lock: {path}; inspect its owner, never overwrite it automatically") from None
    try:
        with handle:
            json.dump({"supervisor_pid": os.getpid(), "output": str(output)}, handle)
        yield
    finally:
        path.unlink()


class StackCancelled(Exception):
    """Explicit stop requested by the recording panel."""


def wait_ready(url, predicate, owned, deadline, stop_file=None):
    # Avoid inherited HTTP proxy settings for loopback readiness.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if stop_file is not None and stop_file.exists():
            raise StackCancelled("stop requested during startup")
        for name, process in owned:
            if process.proc.poll() is not None:
                raise RuntimeError(f"{name} exited before readiness ({process.proc.returncode}); inspect its log")
        try:
            with opener.open(url, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                data = json.load(response)
            if predicate(data):
                return data
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"stack deadline expired waiting for {url}")


def prerequisites(bun):
    if not bun.is_absolute() or not bun.is_file():
        raise RuntimeError("--bun must be the full path to an existing Bun executable")
    for relative in ("env-ts/engine-env.ts", "env-ts/demo-services.ts",
                     "vendor/rs-sdk/node_modules", "vendor/rs-sdk/server/engine/data/pack/server",
                     "vendor/rs-sdk/server/engine/data/pack/client"):
        if not (ROOT / relative).exists():
            raise RuntimeError(f"missing prerequisite {relative}; prepare it separately, no auto-install")


def world_settings(index):
    """Settings for world `index`: its own ports, world id, profile and boot identity."""
    ports = {name: str(port + WORLD_PORT_STRIDE * index) for name, port in BASE_PORTS.items()}
    return {**SETTINGS, **ports, "RL_ENV_PORTS": ports["RL_ENV_PORT"], "RL_WORLD_ID": f"d{index + 1}",
            "NODE_PROFILE": "demo" + uuid.uuid4().hex[:8], "RL_DEMO_BOOT_ID": str(uuid.uuid4())}


def engine_environment(values, bun):
    environment = {**os.environ, **values}
    environment["PATH"] = str(bun.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "rl"), str(ROOT), environment.get("PYTHONPATH")) if p)
    return environment


# What decides an engine build's behaviour at run time, relative to its rs-sdk checkout: the engine's sources,
# packed game data, the SDK planner map and the vendored path finder. Installed dependencies (node_modules)
# are not included; a container image keeps its own.
IDENTITY_PATHS = ("server/engine/src", "server/engine/data/pack", "sdk", "server/vendor", "package.json",
                  "server/engine/package.json")
SKIPPED_FILES = (".DS_Store",)


def digest(base, paths):
    """sha256 over the files under `paths` (relative to `base`), in path order, names included."""
    total = hashlib.sha256()
    for relative in paths:
        top = base / relative
        if top.is_file():
            files = [top]
        elif top.is_dir():
            files = sorted(Path(d) / f for d, dirs, names in os.walk(top) for f in names if f not in SKIPPED_FILES)
        else:
            raise RuntimeError(f"missing engine file {top}")
        for path in files:
            total.update(path.relative_to(base).as_posix().encode() + b"\0")
            total.update(hashlib.sha256(path.read_bytes()).digest())
    return total.hexdigest()


def engine_identity(checkout):
    """The build at an rs-sdk checkout: the git tree of a checkout without tracked changes (no declared build has
    any) or, in a copy without git (a container image), the tree its ENGINE_IDENTITY.json declares, accepted only
    when the runtime files still have the digest recorded there from the verified checkout."""
    found = digest(checkout, IDENTITY_PATHS)
    git = ["git", "-C", str(checkout)]
    try:
        top = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout
        if Path(top.strip()).resolve() != checkout.resolve():
            raise subprocess.CalledProcessError(1, git)  # inside another repository, not a checkout of its own
        changed = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        if changed:
            raise RuntimeError(f"engine checkout {checkout} has tracked changes; no declared build has them")
        tree = subprocess.run([*git, "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        declared = json.loads((checkout / "ENGINE_IDENTITY.json").read_text(encoding="utf-8"))
        if declared.get("digest") != found:
            raise RuntimeError(f"engine copy {checkout} differs from its declared build ({declared.get('tree')})") from None
        tree = declared["tree"]
    return dict(tree=tree, digest=found)


def engine_root(name, checkout, output):
    """A project root for one engine build. env-ts reaches the engine and its planner map through
    ../vendor/rs-sdk, so a build other than this checkout's own gets a root of its own: a copy of this env-ts
    beside a link to its rs-sdk checkout."""
    checkout = checkout.resolve()
    if checkout == (ROOT / "vendor/rs-sdk").resolve():
        return ROOT
    root = output / "engines" / name
    shutil.copytree(ROOT / "env-ts", root / "env-ts", ignore=shutil.ignore_patterns(*SKIPPED_FILES))
    (root / "vendor").mkdir()
    (root / "vendor/rs-sdk").symlink_to(checkout, target_is_directory=True)
    return root


class WorldService:
    """Fresh worlds on request while the command runs, each used for one episode and then stopped.

    The command connects to `address` (AF_UNIX, `authkey`) with one request per connection: acquire(engine)
    returns a booted world of that build, release(id) stops it, target(engine, count) keeps that many booted
    ahead, identity returns each build's tree and digest. Worlds use port blocks first_block..first_block +
    blocks - 1; a released block waits `cooldown` seconds before reuse, and a block whose ports another service
    listens on is skipped. A world that fails to boot or exits while booted or in use stops the service; nothing
    is retried."""

    def __init__(self, bun, engines, output, record, *, first_block, blocks, lockstep, xp_rate, deadline,
                 cooldown=2.0, boot_seconds=120.0):
        self.bun, self.output, self.record = bun, output, record
        self.lockstep, self.xp_rate, self.deadline = lockstep, xp_rate, deadline
        self.cooldown, self.boot_seconds = cooldown, boot_seconds
        self.engines = {}
        for name, checkout in engines.items():
            for relative in ("node_modules", "server/engine/data/pack/server", "server/engine/data/pack/client"):
                if not (checkout / relative).exists():
                    raise RuntimeError(f"engine {name}: missing {checkout / relative}; prepare it separately")
            root = engine_root(name, checkout, output)
            if digest(root, ("env-ts",)) != digest(ROOT, ("env-ts",)):
                raise RuntimeError(f"engine {name}: its env-ts differs from this checkout's")
            self.engines[name] = dict(root=root, identity=engine_identity(checkout))
        record("world_service", engines={n: dict(root=str(e["root"]), **e["identity"]) for n, e in self.engines.items()},
               first_block=first_block, blocks=blocks, cooldown=cooldown)
        (output / "engine-logs").mkdir()
        self.lock = threading.Condition()
        self.free, self.cooling = deque(range(first_block, first_block + blocks)), deque()
        self.ready = {name: deque() for name in engines}
        self.target = {name: 0 for name in engines}
        self.booting = {name: 0 for name in engines}
        self.waiting = {name: 0 for name in engines}
        self.live, self.serial, self.error, self.closed = {}, 0, None, False
        self.counts = dict(booted=0, released=0)
        self.socket_dir = Path(tempfile.mkdtemp(prefix="rlw-"))
        self.address, self.authkey = str(self.socket_dir / "s"), os.urandom(32)
        self.listener = Listener(self.address, family="AF_UNIX", backlog=256, authkey=self.authkey)
        threading.Thread(target=self._serve, daemon=True).start()

    def spec(self):
        return json.dumps(dict(address=self.address, authkey=self.authkey.hex()))

    def _serve(self):
        while not self.closed:
            try:
                connection = self.listener.accept()
            except Exception:
                continue  # a closed listener ends the loop; a failed handshake is that client's error
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection):
        with connection:
            try:
                request = connection.recv()
                op = request.get("op")
                if op == "acquire":
                    reply = dict(world=self.acquire(request["engine"], request.get("timeout", 300.0)))
                elif op == "release":
                    reply = dict(released=self.release(request["id"]))
                elif op == "target":
                    reply = dict(target=self.set_target(request["engine"], request["count"]))
                elif op == "identity":
                    reply = dict(engines={n: e["identity"] for n, e in self.engines.items()}, counts=dict(self.counts))
                else:
                    raise ValueError(f"unknown world service request {op!r}")
                reply["ok"] = True
            except Exception as error:
                reply = dict(ok=False, error=f"{type(error).__name__}: {error}")
            try:
                connection.send(reply)
            except OSError:
                pass

    def _fill(self):
        """Start boots until each build has its target (and every waiting request) booted or booting.
        Called with the lock held."""
        now = time.monotonic()
        while self.cooling and self.cooling[0][0] <= now:
            self.free.append(self.cooling.popleft()[1])
        for name in self.engines:
            while (self.free and not self.closed and self.error is None and
                   len(self.ready[name]) + self.booting[name] < self.target[name] + self.waiting[name]):
                block = self.free.popleft()
                try:
                    require_free_ports([port + WORLD_PORT_STRIDE * block for port in PORTS], reuse=True)
                except RuntimeError as busy:
                    # Another service listens on one of its ports (a host's own server): never used, never touched.
                    self.record("block_skipped", block=block, reason=str(busy))
                    continue
                self.booting[name] += 1
                self.serial += 1
                threading.Thread(target=self._boot, args=(name, block, self.serial), daemon=True).start()

    def _boot(self, name, block, serial):
        world = None
        try:
            values = dict(world_settings(block), RL_LOCKSTEP="1" if self.lockstep else "0",
                          NODE_XPRATE=str(self.xp_rate), RL_XP_RATE=str(self.xp_rate))
            root = self.engines[name]["root"]
            log = (self.output / "engine-logs" / f"b{block}.log").open("a", encoding="utf-8")
            log.write(f"--- world {serial} ({name}) boot {values['RL_DEMO_BOOT_ID']}\n")
            log.flush()
            world = dict(id=serial, engine=name, block=block, values=values, log=log, process=None,
                         started=time.monotonic())
            world["process"] = OwnedProcess([str(self.bun), str(root / "env-ts/engine-env.ts")],
                                            cwd=str(root / "vendor/rs-sdk/server/engine"),
                                            env=engine_environment(values, self.bun), stdin=subprocess.PIPE,
                                            stdout=log, stderr=subprocess.STDOUT, text=True)
            self.record("world_launched", id=serial, engine=name, block=block, owner_pid=world["process"].proc.pid)
            deadline = min(self.deadline, time.monotonic() + self.boot_seconds)
            owned = [(f"world {serial}", world["process"])]
            wait_ready(f"http://127.0.0.1:{values['WEB_PORT']}/engine-status", lambda d: d.get("status") == "running",
                       owned, deadline)
            telemetry = wait_ready(f"http://127.0.0.1:{values['RL_TELEMETRY_PORT']}/status",
                                   lambda d: d.get("schema") == 1 and bool(d.get("bootId")), owned, deadline)
            if telemetry.get("bootId") != values["RL_DEMO_BOOT_ID"]:
                raise RuntimeError("telemetry boot identity does not match this owned launch")
            if telemetry.get("tickRateMs") != 48 or telemetry.get("xpRate") != self.xp_rate:
                raise RuntimeError(f"resolved runtime does not match 48 ms / {self.xp_rate} XP: {telemetry}")
            self.record("world_ready", id=serial, boot_seconds=round(time.monotonic() - world["started"], 3))
            with self.lock:
                self.booting[name] -= 1
                if not self.closed:
                    self.ready[name].append(world)
                    self.counts["booted"] += 1
                    self.lock.notify_all()
                    world = None
        except BaseException as error:
            with self.lock:
                self.booting[name] -= 1
                if not self.closed and self.error is None:
                    self.error = f"world {serial} ({name}, block {block}) failed to boot: {type(error).__name__}: {error}"
                self.lock.notify_all()
            self.record("world_failed", id=serial, engine=name, block=block, error=f"{type(error).__name__}: {error}")
            if world is None:
                with self.lock:
                    self.cooling.append((time.monotonic() + self.cooldown, block))
        finally:
            if world is not None:
                self._stop(world)

    def _stop(self, world, reason="released"):
        process = world["process"]
        try:
            if process is not None:
                process.kill()
                process.proc.wait(timeout=10)
                if process.proc.stdin:
                    try:
                        process.proc.stdin.close()
                    except OSError:
                        pass
        finally:
            world["log"].close()
            with self.lock:
                self.cooling.append((time.monotonic() + self.cooldown, world["block"]))
                self._fill()
            self.record("world_stopped", id=world["id"], reason=reason, seconds=round(time.monotonic() - world["started"], 3),
                        returncode=None if process is None else process.proc.returncode)

    def acquire(self, name, timeout=300.0):
        """A booted world of build `name`, from the ready pool or the next to boot."""
        if name not in self.engines:
            raise ValueError(f"unknown engine build {name!r}; the service has {sorted(self.engines)}")
        deadline = time.monotonic() + timeout
        with self.lock:
            self.waiting[name] += 1
            try:
                self._fill()
                while not self.ready[name]:
                    if self.error is not None or self.closed:
                        raise RuntimeError(self.error or "the world service has stopped")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"no {name} world booted within {timeout} s")
                    self.lock.wait(min(0.25, max(0.01, deadline - time.monotonic())))
                    self._fill()  # a cooled block may now be free
                world = self.ready[name].popleft()
            finally:
                self.waiting[name] -= 1
            self.live[world["id"]] = world
            self._fill()
        values = world["values"]
        return dict(id=world["id"], engine=name, tree=self.engines[name]["identity"]["tree"], block=world["block"],
                    env_port=int(values["RL_ENV_PORT"]), boot_id=values["RL_DEMO_BOOT_ID"],
                    status_url=f"http://127.0.0.1:{values['RL_TELEMETRY_PORT']}/status")

    def release(self, world_id):
        with self.lock:
            world = self.live.pop(world_id, None)
            if world is None:
                raise KeyError(f"world {world_id} is not in use")
            self.counts["released"] += 1
        self._stop(world)
        return world_id

    def set_target(self, name, count):
        if name not in self.engines or type(count) is not int or count < 0:
            raise ValueError(f"invalid target {name!r} {count!r}")
        with self.lock:
            self.target[name] = count
            self._fill()
        return count

    def check(self):
        """Raise if a world failed to boot, or one booted or in use has exited."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(self.error)
            for world in [*self.live.values(), *(w for pool in self.ready.values() for w in pool)]:
                if world["process"].proc.poll() is not None:
                    self.error = f"world {world['id']} ({world['engine']}) exited while booted or in use"
                    raise RuntimeError(self.error)

    def close(self):
        """Stop every world this service started; booting worlds stop as their boot ends."""
        with self.lock:
            self.closed = True
            worlds = [*self.live.values(), *(w for pool in self.ready.values() for w in pool)]
            self.live.clear()
            for pool in self.ready.values():
                pool.clear()
            self.lock.notify_all()
        try:
            self.listener.close()
        except OSError:
            pass
        errors = []
        for world in worlds:
            try:
                self._stop(world, reason="service closed")
            except BaseException as error:
                errors.append(f"world {world['id']}: {error}")
        deadline = time.monotonic() + self.boot_seconds
        while any(self.booting.values()) and time.monotonic() < deadline:
            time.sleep(0.1)
        if any(self.booting.values()):
            errors.append(f"boots still running at close: {self.booting}")
        shutil.rmtree(self.socket_dir, ignore_errors=True)
        if errors:
            raise RuntimeError(f"world service cleanup failed: {errors}")


def run(bun, output, timeout_seconds, command, stop_file=None, worlds=1, lockstep=False, xp_rate=25, first_world=0,
        engines=None, blocks=0, demo=None):
    """engines ({name: rs-sdk checkout}) and blocks start a WorldService on port blocks after the fixed worlds';
    with it, worlds may be 0. demo ({spectator, pace_ms}): one fixed lockstep world that a declared spectator
    login watches, each tick paced (env-ts/core/spectator.ts); its outputs are demos, never evaluation evidence."""
    prerequisites(bun)
    if timeout_seconds <= 0 or not command or worlds < 0 or first_world < 0 or (worlds == 0 and not engines):
        raise ValueError("a positive timeout, a world or a world service, a first world >= 0 and one child command "
                         "are required")
    if engines and (blocks < 1 or first_world + worlds + blocks > MAX_BLOCKS):
        raise ValueError(f"the world service needs 1..{MAX_BLOCKS - first_world - worlds} port blocks")
    if demo and (not lockstep or worlds != 1 or engines):
        raise ValueError("a demo is one fixed lockstep world (--lockstep --worlds 1), never the world service")
    # lockstep: each world ticks only when its one agent steps (engine transport only).
    # xp_rate: rs-sdk's bot default is 25; the real game is 1. The command reads it as RL_XP_RATE.
    per_world = [dict(world_settings(index), RL_LOCKSTEP="1" if lockstep else "0", NODE_XPRATE=str(xp_rate),
                      RL_XP_RATE=str(xp_rate),
                      **(dict(RL_DEMO_SPECTATOR=demo["spectator"], RL_DEMO_PACE_MS=str(demo["pace_ms"])) if demo else {}))
                 for index in range(first_world, first_world + worlds)]
    settings = per_world[0] if per_world else dict(RL_LOCKSTEP="1" if lockstep else "0", RL_XP_RATE=str(xp_rate))
    def environment_for(values):
        return engine_environment(values, bun)
    command_environment = environment_for({**settings, "RL_WORLDS": json.dumps([
        dict(env_port=int(w["RL_ENV_PORT"]), status_url=f"http://127.0.0.1:{w['RL_TELEMETRY_PORT']}/status",
             boot_id=w["RL_DEMO_BOOT_ID"]) for w in per_world])})
    service = None
    owned, logs = [], []
    started = time.monotonic()
    deadline = started + timeout_seconds
    with run_lock(ROOT / "logs/stack.lock", output):
        require_free_ports([port + WORLD_PORT_STRIDE * index for index in range(first_world, first_world + worlds)
                            for port in PORTS], settle_seconds=90)
        output.mkdir(parents=True, exist_ok=False)
        events = (output / "supervisor.jsonl").open("x", encoding="utf-8", buffering=1)
        written = threading.Lock()  # the world service records from its own threads
        def record(kind, **data):
            line = json.dumps({"event": kind, "elapsed_s": time.monotonic() - started, **data}) + "\n"
            with written:
                events.write(line)
        def launch(name, argv, cwd, environment):
            log = (output / f"{name}.log").open("x", encoding="utf-8")
            logs.append(log)
            process = OwnedProcess(argv, cwd=str(cwd), env=environment,
                                   stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
            owned.append((name, process))
            record("launched", name=name, owner_pid=process.proc.pid, argv=argv, cwd=str(cwd))
            return process
        try:
            record("start", supervisor_pid=os.getpid(), settings=per_world, timeout_seconds=timeout_seconds)
            if stop_file is not None and stop_file.exists():
                raise StackCancelled("stop requested before startup")
            readiness_options = {"stop_file": stop_file} if stop_file is not None else {}
            for index, values in enumerate(per_world):
                suffix = "" if index == 0 else str(index)
                launch("engine" + suffix, [str(bun), str(ROOT / "env-ts/engine-env.ts")],
                       ROOT / "vendor/rs-sdk/server/engine", environment_for(values))
            for index, values in enumerate(per_world):
                engine_status = wait_ready(f"http://127.0.0.1:{values['WEB_PORT']}/engine-status",
                                           lambda d: d.get("status") == "running", owned, deadline, **readiness_options)
                telemetry = wait_ready(f"http://127.0.0.1:{values['RL_TELEMETRY_PORT']}/status",
                                       lambda d: d.get("schema") == 1 and bool(d.get("bootId")), owned, deadline,
                                       **readiness_options)
                if telemetry.get("bootId") != values["RL_DEMO_BOOT_ID"]:
                    raise RuntimeError("telemetry boot identity does not match this owned launch; refusing command")
                if telemetry.get("tickRateMs") != 48 or telemetry.get("xpRate") != xp_rate:
                    raise RuntimeError(f"resolved runtime does not match 48 ms / {xp_rate} XP: {telemetry}")
                record("ready", world=index, engine=engine_status, telemetry=telemetry)
            if demo:
                record("demo", label="demo: not evaluation evidence", spectator=demo["spectator"], pace_ms=demo["pace_ms"],
                       client=f"http://127.0.0.1:{per_world[0]['WEB_PORT']}/")
            if engines:
                service = WorldService(bun, engines, output, record, first_block=first_world + worlds, blocks=blocks,
                                       lockstep=lockstep, xp_rate=xp_rate, deadline=deadline)
                command_environment["RL_WORLD_SERVICE"] = service.spec()
            child = launch("command", command, ROOT, command_environment)
            while child.proc.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("total stack/command deadline expired")
                for name, process in owned[:-1]:
                    if process.proc.poll() is not None:
                        raise RuntimeError(f"{name} exited while command was running")
                if service is not None:
                    service.check()
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            result = child.proc.returncode
            record("command_exit", returncode=result)
            return result
        except StackCancelled as error:
            record("cancelled", reason=str(error))
            return 0
        except BaseException as error:
            record("failed", error=str(error), error_type=type(error).__name__)
            raise
        finally:
            cleanup_errors = []
            if service is not None:
                try:
                    service.close()
                    record("world_service_closed", **service.counts)
                except BaseException as error:
                    cleanup_errors.append(f"world service: {error}")
            for name, process in reversed(owned):
                try:
                    process.kill()
                    process.proc.wait(timeout=10)
                    record("stopped", name=name, owner_pid=process.proc.pid, returncode=process.proc.returncode)
                except BaseException as error:
                    cleanup_errors.append(f"{name}: {error}")
                finally:
                    if process.proc.stdin:
                        try:
                            process.proc.stdin.close()
                        except OSError:
                            pass
            if cleanup_errors:
                record("cleanup_failed", errors=cleanup_errors)
            for log in logs:
                log.close()
            events.close()
            if cleanup_errors:
                raise RuntimeError(f"owned process cleanup failed: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--worlds", type=int, default=1, help="independent worlds to boot, one per agent")
    parser.add_argument("--first-world", type=int, default=0,
                        help="the first world's index: its ports and id (alternate blocks between consecutive runs)")
    parser.add_argument("--lockstep", action="store_true",
                        help="worlds tick only when their agent steps; engine transport only")
    parser.add_argument("--xp-rate", type=int, default=25,
                        help="XP multiplier; every training run used 25, the real game is 1")
    parser.add_argument("--engine", action="append", default=[], metavar="NAME=CHECKOUT",
                        help="an engine build (rs-sdk checkout) the world service boots fresh worlds of; repeatable")
    parser.add_argument("--blocks", type=int, default=0,
                        help="port blocks the world service may use at once (booting, ready, in use and cooling)")
    parser.add_argument("--demo-spectator", metavar="NAME",
                        help="a demo: the one network login the world admits, hidden, inert, kept on the agent's tile "
                             "(with --demo-pace-ms, --lockstep and --worlds 1)")
    parser.add_argument("--demo-pace-ms", type=int, metavar="MS",
                        help="a demo: milliseconds the world waits after each tick (1..500; the client walks a tile "
                             "in about 450)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (args.demo_spectator is None) != (args.demo_pace_ms is None):
        parser.error("--demo-spectator and --demo-pace-ms go together")
    demo = dict(spectator=args.demo_spectator, pace_ms=args.demo_pace_ms) if args.demo_spectator else None
    engines = {}
    for spec in args.engine:
        name, _, checkout = spec.partition("=")
        if not name or not checkout or name in engines:
            parser.error(f"--engine takes distinct NAME=CHECKOUT, not {spec!r}")
        engines[name] = Path(checkout).resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args.bun, args.output.resolve(), args.timeout_seconds, command, worlds=args.worlds,
                   lockstep=args.lockstep, xp_rate=args.xp_rate, first_world=args.first_world, engines=engines,
                   blocks=args.blocks, demo=demo)
    except (RuntimeError, ValueError, OSError, TimeoutError) as error:
        print(f"stack: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
