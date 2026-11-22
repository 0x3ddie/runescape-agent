"""Connect to the world booted by scripts/stack.py and build task episodes.

The world's native evaluator (demo-services, loopback only) owns the task clock
and the resource evidence. Every raw exchange is appended to a JSONL log.
"""
import gzip
from http.client import HTTPConnection
import io
import json
import math
import os
from pathlib import Path
import re
import select
import threading
import time
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit,urlunsplit
from urllib.request import urlopen

from baseline import config
from baseline.resource_task import ResourceEpisode
from baseline.scripted import SKILLS, prepare_capacity, prepare_declared
from vec_adapter import EngineEnvServer


class JsonlEvidence:
    """Append-only JSONL log; safe to call from the agents' worker threads.

    A path ending in .gz is written compressed: an evaluation logs every observation, about
    300 MB a run uncompressed, and a full disk lost eight runs. Each line is flushed,
    so a killed run's log still reads up to its last line (read_jsonl)."""
    def __init__(self,path,kinds=None):
        path=Path(path)
        self.stream=(gzip.open(path,'xt',encoding='utf-8',compresslevel=6) if path.suffix=='.gz'
                     else path.open('x',encoding='utf-8'))
        self.kinds=kinds;self.lock=threading.Lock()
    def __call__(self,kind,**data):
        if self.kinds is not None and kind not in self.kinds:return
        line=json.dumps(dict(kind=kind,**data),allow_nan=False)+'\n'
        with self.lock:
            self.stream.write(line);self.stream.flush()
    def close(self):
        self.stream.close()


def read_jsonl(path):
    """The rows of a JsonlEvidence log, compressed or not; a compressed log cut off by a
    killed run ends at its last whole line."""
    path=Path(path)
    stream=gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')
    with stream:
        try:
            for line in stream:
                if line.endswith('\n'):
                    yield json.loads(line)
        except EOFError:
            return


def check_status(row, *, boot_id=None, speed_ms=48, xp_rate=25, actors=None, spectator=None):
    """One /status row of an owned world. actors: the players that may act in it; every player is one unless
    spectator names a demo world's declared spectator (population-v2), which may also be present."""
    if row.get('schema') != 1 or not row.get('bootId'):
        raise ValueError('missing server identity')
    if boot_id is not None and row['bootId'] != boot_id:
        raise ValueError('world restarted')
    if row.get('tickRateMs') != speed_ms or row.get('xpRate') != xp_rate:
        raise ValueError('world speed or XP rate differs from configuration')
    if type(row.get('tick')) is not int or row['tick'] < 0:
        raise ValueError('invalid authoritative world tick')
    if not isinstance(row.get('monotonicMs'), (int, float)) or not math.isfinite(row['monotonicMs']):
        raise ValueError('invalid server clock')
    if spectator is None:
        # every player acts; a population-v1 world does not report actors
        if actors is not None and (row.get('players') != actors or row.get('actors', actors) != actors):
            raise ValueError('unexpected world population')
        return row
    watch = (row.get('demo') or {}).get('spectator') or {}
    present = watch.get('present')
    if row.get('population') != 'population-v2' or watch.get('declared') != spectator or type(present) is not list:
        raise ValueError(f'the world does not count its declared spectator {spectator} (population-v2)')
    if (any(type(p) is not dict or p.get('name') != spectator for p in present) or row.get('actors') != actors
            or row.get('players') != actors + len(present)):
        raise ValueError('unexpected world population')
    return row


def tick_windows(rows, speed_ms, seconds=5):
    """Disjoint server-clock windows during a rollout; the first is warmup."""
    if not rows:
        return []
    start = rows[0]
    previous = start
    result = []
    for row in rows[1:]:
        if row['bootId'] != start['bootId'] or row['tick'] < previous['tick'] or row['monotonicMs'] <= previous['monotonicMs']:
            raise ValueError('world counter/clock changed or regressed')
        previous = row
        elapsed = (row['monotonicMs'] - start['monotonicMs']) / 1000
        if elapsed >= seconds:
            ticks = row['tick'] - start['tick']
            result.append({'seconds': elapsed, 'ticks': ticks,
                           'delivery_fraction': ticks / elapsed / (1000 / speed_ms),
                           'warmup': len(result) == 0})
            start = row
    return result


class ResourceService:
    """A world's control client: demo-services' /status and /resource-capture on its loopback port.

    Its calls share one kept-open HTTP/1.1 connection. With a new loopback connection for each call, a watch's
    four worlds made about 100 a second; a VPN's packet filter kept state for each one and then stalled every new local
    connect for seconds. Before a request, a kept connection that the server has closed since its last reply is
    replaced by a new one, so nothing is sent into it. A request once sent is never sent again: a resent arm or release
    whose first copy took effect would be refused by the engine, and the lost reply of a release is the capture's
    only receipt. Every failure raises as urlopen raised: URLError for a connect or send that failed, HTTPError for an
    error status, TimeoutError for a reply that does not come in TIMEOUT seconds, the drop itself for a connection
    closed without a reply."""
    TIMEOUT=5

    def __init__(self,status_url,boot_id,account_name,record,*,speed_ms,xp_rate,spectator=None):
        url=urlsplit(status_url)
        if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost','::1') or url.username or url.password or url.path!='/status' or url.query or url.fragment:
            raise ValueError('owned loopback status URL required')
        if not isinstance(boot_id,str) or not boot_id or not isinstance(account_name,str) or not re.fullmatch('[a-z0-9]{1,12}',account_name):
            raise ValueError('explicit owned boot and native account required')
        self.status_url=status_url;self.netloc=url.netloc
        self.boot=boot_id;self.account=account_name;self.record=record
        self.speed_ms=speed_ms;self.xp_rate=xp_rate;self.spectator=spectator;self.samples=[];self.last_status=None
        self.connection=HTTPConnection(url.hostname,url.port,timeout=self.TIMEOUT)

    def _exchange(self,method,path,body=None):
        """One request on the kept connection; returns the reply's JSON."""
        sock=self.connection.sock
        if sock is not None:  # a kept connection reads as ready only when the server closed it (no reply is due)
            ready=select.poll();ready.register(sock,select.POLLIN)
            if ready.poll(0):self.connection.close()  # request() opens a new one
        try:
            try:self.connection.request(method,path,body=body,
                                        headers={'Content-Type':'application/json'} if body is not None else {})
            except OSError as error:raise URLError(error)  # as urlopen: a connect or send that failed
            response=self.connection.getresponse();data=response.read()
        except Exception:
            self.connection.close()
            raise
        if not 200<=response.status<300:
            raise HTTPError(urlunsplit(('http',self.netloc,path,'','')),response.status,response.reason,response.headers,
                            io.BytesIO(data))
        return json.loads(data)

    def status(self):
        payload=self._exchange('GET','/status')
        payload=check_status(payload,boot_id=self.boot,actors=1,spectator=self.spectator,speed_ms=self.speed_ms,
                             xp_rate=self.xp_rate)
        self.record('server_clock',server=payload)
        self.samples.append(payload);self.last_status=time.monotonic()
        return payload

    def __call__(self,operation,*,identity=None,horizon=None,targets=None):
        if self.last_status is None or time.monotonic()-self.last_status>=1:self.status()
        if operation not in ('identity','arm','budget','read','retire','release'):
            raise ValueError('invalid resource operation')
        payload=dict(operation=operation,bootId=self.boot,accountName=self.account)
        if operation!='identity':payload['identity']=identity
        if operation=='arm':payload['horizon']=horizon
        if operation=='arm' and targets is not None:payload['targets']=targets  # certified item scope
        reply=self._exchange('POST','/resource-capture',json.dumps(payload).encode())
        wanted='resource_identity' if operation=='identity' else 'resource_budget' if operation=='budget' else 'resource_capture'
        if reply.get('type')!=wanted or (identity is not None and reply.get('identity')!=identity):
            raise ValueError('resource control reply mismatch')
        return reply


def wait_for_spectator(status_url, name, timeout, say=print):
    """Poll a demo world (scripts/stack.py --demo-spectator) until its declared spectator is in it; returns the
    seconds waited."""
    started = time.monotonic()
    next_say = started
    while True:
        with urlopen(status_url, timeout=5) as response:
            spectator = (json.load(response).get('demo') or {}).get('spectator') or {}
        if spectator.get('declared') != name:
            raise ValueError(f'the world does not declare the spectator {name}')
        if any(p['name'] == name for p in spectator['present']):
            return time.monotonic() - started
        now = time.monotonic()
        if now - started >= timeout:
            raise TimeoutError(f'the spectator {name} did not log in within {timeout} s')
        if now >= next_say:
            say(f"[demo] open http://127.0.0.1:{os.environ.get('WEB_PORT', '8888')}/ and log in as {name} "
                f"(any password); the trial starts when {name} is in the world", flush=True)
            next_say = now + 15
        time.sleep(.5)


def worlds():
    """Worlds booted by scripts/stack.py: [{env_port, status_url, boot_id}, ...]."""
    if os.environ.get('RL_WORLDS'):
        return json.loads(os.environ['RL_WORLDS'])
    boot_id=os.environ.get('RL_DEMO_BOOT_ID')
    if not boot_id:raise RuntimeError('RL_DEMO_BOOT_ID unset; run inside scripts/stack.py')
    return [dict(env_port=None,status_url=config.STATUS_URL,boot_id=boot_id)]


def join(world, record, index=0, spectator=None):
    """One agent in one world ({env_port, status_url, boot_id}); returns (transport, service). spectator: a demo
    world's declared spectator, the one other player its status may count."""
    transport=EngineEnvServer(world['env_port'],config.AGENT_NAME)
    try:
        return transport,ResourceService(world['status_url'],world['boot_id'],transport.agent_name,
                                         lambda kind,**data:record(kind,agent=index,**data),
                                         speed_ms=config.SPEED_MS,xp_rate=config.XP_RATE,spectator=spectator)
    except Exception:
        transport.close()
        raise


def connect(record, agents=1, spectator=None):
    """Join one agent per world; returns [(transport, service), ...].

    The native resource capture follows exactly one actor per world, so agents
    never share a world.
    """
    available=worlds()
    if agents>len(available):
        raise RuntimeError(f'{agents} agents need {agents} worlds; run scripts/stack.py --worlds {agents}')
    links=[]
    try:
        for index,world in enumerate(available[:agents]):
            links.append(join(world,record,index,spectator))
        return links
    except Exception:
        for transport,_ in links:transport.close()
        raise


class FreshWorlds:
    """Client of scripts/stack.py's world service (RL_WORLD_SERVICE): a booted world of an engine build for
    each episode, stopped when released. One connection per request, so agent threads can share it."""
    def __init__(self, spec=None):
        spec=json.loads(spec or os.environ.get('RL_WORLD_SERVICE') or 'null')
        if not spec:raise RuntimeError('RL_WORLD_SERVICE unset; run inside scripts/stack.py --engine NAME=CHECKOUT')
        self.address,self.authkey=spec['address'],bytes.fromhex(spec['authkey'])

    def _call(self, **request):
        from multiprocessing.connection import Client
        with Client(self.address,family='AF_UNIX',authkey=self.authkey) as connection:
            connection.send(request)
            reply=connection.recv()
        if not reply.pop('ok'):
            raise RuntimeError(f"world service: {reply['error']}")
        return reply

    def acquire(self, engine, timeout=300.):
        return self._call(op='acquire',engine=engine,timeout=timeout)['world']

    def release(self, world_id):
        self._call(op='release',id=world_id)

    def target(self, engine, count):
        self._call(op='target',engine=engine,count=count)

    def identity(self):
        return self._call(op='identity')


def make_episode(transport, service, fixture, record):
    cfg=dict(config.TASK_CFG,**{**config.VILLAGES,**config.DROP_IN,**config.SEARCH,**config.HIDDEN}[fixture.get('village','draynor')])
    if 'spawnTele' in fixture:cfg['spawnTele']=fixture['spawnTele']  # a fixture's own spawn (edgeville_oak)
    if 'spawn' in fixture:cfg['spawnTele']=config.SPAWNS[fixture['spawn']]
    if 'spawn_steps' in fixture:cfg['spawnSteps']=list(fixture['spawn_steps'])
    if fixture['capacity']=='declared':
        # A declared start (four-goal task): no starter kit or bank setup; these items and base levels.
        cfg.update(kit='none',seedItems=fixture['seed_items'],setStats=fixture['set_stats'])
        def prepare(env, obs, write):
            return prepare_declared(env, obs, write, fixture['seed_items'], fixture['set_stats'])
    else:
        def prepare(env, obs, write):
            return prepare_capacity(env, obs, write, fixture['capacity'], SKILLS[fixture.get('skill', 'woodcutting')]['tool'])
    return ResourceEpisode(transport, service, cfg, fixture['horizon'], config.REWARD_SCALE,
                           record, prepare, goal=fixture.get('goal'))


def delivery_report(service, min_windows):
    """Did the world deliver its promised tick rate while we were acting in it?"""
    if service.samples and all(s.get('lockstep') for s in service.samples):
        # A lockstep world advances exactly the ticks each step asks for.
        return dict(ok=True,mode='lockstep',windows=[])
    windows=tick_windows(service.samples,config.SPEED_MS)
    measured=[w for w in windows if not w['warmup']]
    ok=len(measured)>=min_windows and all(.95<=w['delivery_fraction']<=1.05 for w in measured)
    return dict(ok=ok,windows=windows)
