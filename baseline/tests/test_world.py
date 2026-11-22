"""Tick delivery decides which runs are comparable; run logs stay readable; a world's control calls keep one connection
(baseline/world.py)."""
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import socket
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hypothesis import given, settings, strategies as st
import pytest

from baseline import world
from baseline.world import JsonlEvidence, ResourceService, delivery_report, read_jsonl


def sample(tick, ms, **extra):
    return dict(schema=1, bootId='b', tick=tick, tickRateMs=48, monotonicMs=ms, xpRate=25, **extra)


def test_real_time_needs_measured_windows_and_lockstep_is_exact_by_construction():
    slow = [sample(i * 52, i * 5000.0) for i in range(6)]  # 52 ticks per 5 s: half the promised rate
    assert not delivery_report(SimpleNamespace(samples=slow), min_windows=3)['ok']
    fast = [sample(i * 5000, i * 5000.0, lockstep=True) for i in range(6)]  # wall clock is meaningless here
    assert delivery_report(SimpleNamespace(samples=fast), min_windows=3) == dict(ok=True, mode='lockstep', windows=[])


def test_a_compressed_log_reads_back_and_a_killed_runs_log_ends_at_its_last_line(tmp_path):
    log = JsonlEvidence(tmp_path / 'run.jsonl.gz')
    for i in range(3):
        log('step', i=i)
    # What a run killed now leaves: flushed lines, no gzip trailer.
    (tmp_path / 'killed.jsonl.gz').write_bytes((tmp_path / 'run.jsonl.gz').read_bytes())
    log.close()
    assert [row['i'] for row in read_jsonl(tmp_path / 'run.jsonl.gz')] == [0, 1, 2]
    assert [row['i'] for row in read_jsonl(tmp_path / 'killed.jsonl.gz')] == [0, 1, 2]


def test_a_demo_waits_for_its_declared_spectator(monkeypatch):
    rows = []
    def status(url, timeout):
        return io.BytesIO(json.dumps(rows.pop(0)).encode())
    monkeypatch.setattr(world, 'urlopen', status)
    monkeypatch.setattr(world.time, 'sleep', lambda seconds: None)
    demo = lambda *present, declared='viewer': dict(demo=dict(spectator=dict(declared=declared, present=[dict(name=n) for n in present])))
    said = []
    rows += [demo(), demo('wd1a0'), demo('viewer')]
    assert world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60, say=lambda *a, **k: said.append(a)) >= 0
    assert not rows and len(said) == 1  # the login instructions, once in the first 15 s
    rows += [demo('viewer', declared='other')]
    with pytest.raises(ValueError, match='does not declare'):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60)
    rows += [dict(schema=1)]  # an ordinary world declares no spectator
    with pytest.raises(ValueError, match='does not declare'):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60)
    rows += [demo()]
    with pytest.raises(TimeoutError):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 0, say=lambda *a, **k: None)


@settings(max_examples=200, deadline=None)
@given(actors=st.integers(0, 3), watching=st.integers(0, 1), declared=st.booleans(), v2=st.booleans())
def test_only_a_demo_worlds_declared_spectator_is_left_out_of_its_population(actors, watching, declared, v2):
    # The status an engine gives of a world with these players: actors (the agent, or any other player) and
    # the declared spectator 'viewer' when it watches. Only a population-v2 world declares a spectator.
    declared = declared and v2
    watching = watching if declared else 0
    row = sample(7, 1.0, players=actors + watching)
    if v2:
        row.update(population='population-v2', actors=actors)
    if declared:
        row['demo'] = dict(paceMs=450, spectator=dict(declared='viewer', present=[dict(name='viewer')] * watching))
    def accepted(**options):
        try:
            return world.check_status(row, actors=1, **options) is row
        except ValueError:
            return False
    assert accepted() == (actors == 1 and not watching)  # population-v1: every player acts
    assert accepted(spectator='viewer') == (actors == 1 and declared)
    assert not accepted(spectator='guest')
    if watching:
        row['demo']['spectator']['present'] = [dict(name='guest')]
        assert not accepted(spectator='viewer')


class ControlServer:
    """A world's control port on loopback, kept open between requests as the engine's Bun.serve keeps it (HTTP/1.1).
    answer(method, path, body) gives (status, reply), or None to close the connection without a reply. A connection
    is closed without notice after `drop_after` replies, as a server closes an idle one. `connections` counts the TCP
    connections it accepted, `requests` the requests it read and `closed` the connections it closed: the server's
    counts, not the client's."""

    def __init__(self, answer, drop_after=None, delay=0.0):
        self.answer, self.drop_after, self.delay = answer, drop_after, delay
        self.connections = self.requests = self.closed = 0
        lock, server = threading.Lock(), self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def setup(self):
                super().setup()
                self.replies = 0
                with lock:
                    server.connections += 1

            def reply(self):
                body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
                with lock:
                    server.requests += 1
                answer = server.answer(self.command, self.path, json.loads(body) if body else None)
                if answer is None:
                    self.close_connection = True
                    return
                time.sleep(server.delay)
                data = json.dumps(answer[1]).encode()
                self.send_response(answer[0])
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.replies += 1
                if server.drop_after is not None and self.replies >= server.drop_after:
                    self.close_connection = True  # no Connection: close header; the client finds out at its next call
            do_GET = do_POST = reply

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def shutdown_request(self, request):
                super().shutdown_request(request)
                with lock:
                    server.closed += 1

        self.httpd = Server(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/status'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def wait_closed(self, n):
        """Until the server has closed n connections, and a moment more for its close to reach the client's end."""
        end = time.monotonic() + 5
        while self.closed < n:
            assert time.monotonic() < end, 'the server did not close the connection'
            time.sleep(0.01)
        time.sleep(0.05)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


ID = dict(worldBootId='b', accountId='agent', accountEpoch=1, incarnationId='p1')
STATUS = sample(7, 1.0, players=1, actors=1)


def engine(method, path, body):
    """The answers of demo-services as the client reads them: a status row, the control replies, and the 409 that a
    refused operation gets (env-ts/core/resource-control.ts)."""
    if method == 'GET':
        return 200, STATUS
    if body['operation'] == 'identity':
        return 200, dict(type='resource_identity', identity=ID)
    if body['operation'] == 'budget':
        return 200, dict(type='resource_budget', identity=ID, budget=dict(remainingTicks=40, status='active'))
    if body['operation'] == 'retire':
        return 200, dict(type='resource_budget', identity=ID)  # the wrong type: the client refuses it
    return 409, dict(error='resource incarnation mismatch')


def service(url):
    return ResourceService(url, 'b', 'agent', lambda kind, **data: None, speed_ms=48, xp_rate=25)


def before(url, operation):
    """The control call as baseline/world.py made it before it kept its connection: a new urlopen connection a call."""
    payload = dict(operation=operation, bootId='b', accountName='agent', **({} if operation == 'identity' else dict(identity=ID)))
    request = Request(url.replace('/status', '/resource-capture'), data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def test_the_control_calls_share_one_kept_connection_and_answer_as_the_urlopen_calls_did():
    with ControlServer(engine) as server, ControlServer(engine) as reference:
        control = service(server.url)
        assert control('identity') == before(reference.url, 'identity')
        for _ in range(300):
            assert control('budget', identity=ID) == before(reference.url, 'budget')
        with urlopen(reference.url, timeout=5) as response:
            assert control.status() == json.load(response)
        with pytest.raises(HTTPError) as new:
            control('read', identity=ID)
        with pytest.raises(HTTPError) as old:
            before(reference.url, 'read')
        assert (new.value.code, str(new.value), new.value.read()) == (old.value.code, str(old.value), old.value.read())
        with pytest.raises(ValueError, match='reply mismatch'):
            control('retire', identity=ID)
        control('budget', identity=ID)
        assert server.connections == 1 and server.requests >= 304  # the error replies kept the connection too
        assert reference.connections == reference.requests == 303  # a new connection a call, as before


def test_a_kept_connection_the_server_closed_is_replaced_before_sending_and_a_sent_call_is_never_sent_again():
    with ControlServer(engine, drop_after=3) as server:
        control = service(server.url)
        control('identity')  # a status row and the identity: two replies on the first connection
        control('budget', identity=ID)  # the third reply, after which the server closes it
        server.wait_closed(1)
        assert control('budget', identity=ID) == engine('POST', '', dict(operation='budget'))[1]
        assert server.connections == 2 and server.requests == 4  # a new connection; nothing went into the closed one
        server.answer = lambda *request: None  # from now on the server reads each request and closes without a reply
        with pytest.raises(RemoteDisconnected):
            control('budget', identity=ID)
        assert server.connections == 2 and server.requests == 5  # read once and not sent again
        with pytest.raises(RemoteDisconnected):
            before(server.url, 'budget')  # the error the urlopen call raised
    with ControlServer(lambda *request: None) as server:  # a new connection that drops is not opened again
        with pytest.raises(RemoteDisconnected):
            service(server.url)('identity')
        assert server.connections == server.requests == 1
    closed = socket.socket()
    closed.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{closed.getsockname()[1]}/status'
    closed.close()
    with pytest.raises(URLError) as new:
        service(url)('identity')
    with pytest.raises(URLError) as old:
        before(url, 'identity')
    assert type(new.value.reason) is type(old.value.reason) is ConnectionRefusedError


def test_a_reply_that_does_not_come_in_time_raises_without_a_new_connection(monkeypatch):
    monkeypatch.setattr(ResourceService, 'TIMEOUT', 0.2)
    with ControlServer(engine) as server:
        control = service(server.url)
        control('identity')
        server.delay = 1.0
        with pytest.raises(TimeoutError) as error:
            control('budget', identity=ID)
        assert not isinstance(error.value, URLError) and server.connections == 1
