"""free-play-v1 in the real engine: many agents in each world, one shared policy. The worlds are the vendor engine's,
booted by scripts/stack.py in lockstep with RL_SHARED_AGENTS=N (shared-lockstep-v1, env-ts/engine/batch.ts);
`sim.train train --env real` trains the free-play learner on them, from a simulator-trained policy.

The worlds. W worlds of N agents each (REAL: 8 x 128), one controller connection a world (WorldClient, the only code that
knows the wire protocol). Agent j is slot j mod N of world j // N for the whole run. The number of agents, W x N, is a
setting of its own beside the learner's n_envs (the segments of an update). The real engine's world excludes nothing: no
excluded-pickup rule and no unsupported rule; every mechanic is the engine's own. Any engine exit, a closed connection or
an error result stops the run: it never goes on with fewer worlds or agents.

Lives as in free-play-v1. Each life takes the agent's next start record as sim/watch.py draws them (Records of the
agent's own stream, life_cfg and world_rules of the run's `starts` world: kit, levels, worn weapon, reset seed; the
engine runs sim/watch.py setup's steps inside the reset: the declared start, the style step, the Wield), a start site (a
keyed draw over the run's `sites`, sites of its world: sim/watch.py world_sites, all of them unless the run names some)
and a tile near it (a keyed draw over the engine's `places` within the site's radius of its tile: the site's own,
sim/watch.py SITE_RADIUS, or the batch's `radius`; the reset seed then picks a connected tile within 3 of it), and a
length as training draws it (sim/task.py draw_life over the run's `life`). A life starts at its reset's result, after
the setup (the engine's own decision and tick counters include the setup's steps; a life's ticks, decisions and length
here count from its start, as the simulator's free-play episode counts from its arm), and ends at a death or at the
first decision point at or past its length (a cut, whose last observation the learner values); then that agent alone starts its next life.
Agent j draws its records, sites, tiles and lengths from stream j + (leg - 1) x LEG_STREAMS: a run is leg 1 (stream j,
the draws of every run before legs), and a declared continuation (sim/train.py --leg-from, --leg N) starts every agent
on a new life from streams that no earlier leg drew from.
Pay is free-play-v1's: the change in fractional level over its seven skills (sim/audit.py fractional_level, the
simulator's own function) at each decision, from the XP in tenths each result reads from the player (`pay.xp`).
free-play-v3 (all_skills, sim/workers.py FREE_PLAY_V3): the same change over every stat the engine's ledger reports,
with the version's extra item (a tinderbox) in each reset's
kit; its checks are free-play-v1's, whose running and closing
ledger checks already cover every stat, and a v3 life row keeps `paid` (every stat) beside `paid_seven`, `gain_all` and
`xp_all`, as the counts and the logger's real/levels_seven and real/levels_all keep the seven and every stat apart.
free-play-v2 (death_price D, sim/workers.py FREE_PLAY_V2): the decision whose result ends the life by the engine's death
(`dead`) also pays -D (death_charge); nothing else changes, and a free-play-v1 batch (D = 0) runs exactly as before. Its
checks: at every reply the price is D exactly when the engine's own death record (facts.death, its tick and killers) is
in that reply, and the life's prices are D only once that record is; at a life's end its rewards (pay less price) equal
its ledger level gain less D exactly when the engine ended it by a death; at each hand-over to the learner every
segment's stored rows hold the price at exactly the rows where a life ended with the engine's death record, and sum to
the segment's level gain less D for each (_check_rows, in every run, --validate or not); at the run's end the prices in
the stored rows handed over equal D times the engine deaths of the lives they ended, and every death of the life records
is either among those or in rows the learner never received. A difference stops the run as a pay difference does.
The level gain and the price stay apart: a life row's `paid` is its level gain alone, beside `price` and
`return_with_price`; each update's record, the counts and the logger's real/levels_paid and real/prices_paid hold them
apart too. A batch pays one version: a death price with every stat paid is refused.

The independent check (`ledger`; a play run may turn it off, training may not). At every result the XP the agent's
results show it gained since its life started, by stat, must equal the engine's running ledger total (`pay.ledger`: every
XP grant inside Player.addXp since the life started). The reset or leave that closes a life returns the engine's ledger
of it: its total must equal the XP the results paid, its start and end must be the life's own ticks, and the life's pay
must equal its fractional level gain. At the run's end every live agent leaves and its ledger is checked before the
connection closes (finish). A difference stops the run (RealStop, evidence in real-stop.json); so do an engine error or
fatal code, a closed connection, a stalled world (no reply in REAL['stall'] seconds), an item that has not come back
within REAL['stuck_ticks'] world ticks, an engine limit that ends a life and any reply the protocol does not allow.
Nothing is retried inside a run, with one declared rule: a reset that fails before its life is armed (one of the engine's
reset error codes, REAL['reset_failures']: a death during the setup, a spawn that did not land, ...) gives no row and no
pay; its closing ledger, if it closed a life, is checked as any; the agent is reset again with its next start record; the
failure is written to failed-resets.jsonl (agent, world, slot, site, tile, life, record, code, message, tick) and counted
(counts['failed_resets'] of counts['resets'], in each update's record and the summary). The run stops when one agent
fails REAL['failures_in_row'] resets in a row, or when the failed resets pass REAL['failure_share'] of all resets after
the first REAL['failure_floor']. While the learner works between collections (its update, validation, checkpoint), a
thread pings every world each REAL['ping_seconds'] seconds, so that the engine's answer limit is not the only guard.

The collector (asynchronous, a pool of finished segments, no hold). The engine
stops a world at the first tick where one of its agents' items is over; every due agent is answered in the next request,
so no agent waits for another and none stands in a live world without a decision. A round sends each world with an agent
inside an item one batch request, reads every world's reply, then makes one policy call for the agents due in all worlds,
each from its own recurrent state (a world waits for the slowest reply of its round: `wait_seconds`). Each agent fills
segments of n_steps decisions in a row; a segment is finished when the observation after its last decision has been
valued, or when its last decision ended its life. Finished segments join a pool in a fixed order (the round they
finished in, then the agent's index); when it holds n_envs whose ended lives have all passed the ledger check, the first
n_envs are one rollout buffer of the learner's shape (a column a segment, with its stored recurrent states) and the
learner updates on it, unchanged. During an update no request is sent, so every world stands still (the answer limit,
REAL['answer_limit_ms'], covers it); the agents left due then decide with the new weights. With `workers`, worker
processes (Workers, each a group of worlds) send the requests, read and decode the replies and project the observations,
and the learner's process keeps the lives, the checks, the policy and the segments: the rows and lives are the same.
Devices: the update runs on the model's device; the policy calls of the collection run on it too, or with
`collect_device` (sim.train --collect-device, for a model on CUDA) on a copy of the policy there that takes the model's
weights at the start of each collection, after the update and before its first round (HANDOVER; each update's record
holds the digest it took, which must be the model's). The stage, the recurrent states and the kept weights of each
version live on the collection's device; the rollout is copied to the buffer on the model's device. A run without
collect_device keeps the processor path and its records as before.

Rows of several ages. Each row records the version of the weights that collected it (the updates before it). An agent
whose decisions are long (a chop, a fish, a wait) fills its segment over several updates, and the pool keeps finished
segments that did not fit an update, so a batch holds rows collected under the weights of several updates before its
own; the stored behaviour log-probabilities keep PPO's ratio right for each. The weights of each version are kept while
rows of that version may still reach a batch (REAL['max_age'] updates), and --validate checks each row with its own
version's weights (sim/train.py validate_free_play: buffer.segments, buffer.version_weights). A finished segment whose
oldest row is more than REAL['max_age'] updates old is dropped (counted): younger ones stay, being the long actions'
only data. Each update records its rows by age (0, 1, 2, 3, 4 and more updates), PPO's approximate KL and clip fraction
for each age before and after the update (lag_stats; age 0 apart for segments with older rows, whose recurrent states
PPO recomputes through those rows), the dropped segments, the least and the most segments one agent gave, the world ticks
that passed and the collector's times (`updates`).

lives.jsonl (in the run's folder) holds a row a life, written at its ledger check: agent, world, slot, start site and
tile, record, length, ticks, decisions, gain and XP by skill, its end (death, horizon, or left at the run's end), the
killers, eats (the engine's) and eats at full hitpoints, the share of decisions with a weapon worn, the pay, the ledger's
grants and the weights versions (with free-play-v2's price fields and free-play-v3's every-stat fields). lives-xp.jsonl
beside it holds, for the same lives of every version, every stat's XP gained (tenths) and whole levels gained, so a
free-play-v1 run is readable for every stat too while its lives.jsonl rows stay as they were. When every world's hello
names the engine's facts food and food_drops, each life row of every version also holds four reads that nothing pays
(life_facts): half_hp_decisions (decisions whose observation's hitpoints were half the base or less, 2 * hp <= base, by
the facts before the decision), half_hp_food_decisions (of those, the ones whose pack then held an item that offers Eat:
the engine's facts.food, from the item's own options), half_hp_eats (the engine's Eats at those decisions) and food_drops
(the engine's Drops in the life that took an item that offers Eat); a world whose hello names neither gets none of them.
With them, when the hello also names tool_drops, a fifth read: tool_drops (the engine's Drops in the life that took a
tool, an item env-ts TOOL_PROTECT_RE names; under mask-v6 a drop of one is legal).

free-play-v12 (sim/workers.py FREE_PLAY_V12): free-play-v10 with each life's reset naming its start record's drawn
hitpoints (startHp). When every world's hello names hurt, each life row of every version also holds hurt_ticks (the
life's full-deficit ticks, exactly from facts.hurt) and start_hp (facts.hp at the arm): reads that nothing pays.

free-play-v14 (sim/workers.py FREE_PLAY_V14): free-play-v12 with the start rule old-life-v1 (OLD_LIFE). Every agent of odd
index j plays every life from its record as sim/watch.py profiled gives it with the old-life profile (parse_levels of
OLD_LIFE's levels): the profile's stats at its levels, every other draw the record's own and the hitpoints at the arm
the profile's hitpoints level less the record's own deficit, never under 1; the agents of even index play their records
as drawn. The record streams are not touched, so an even agent's resets are a free-play-v12 batch's same agent's. Each
world holds an even number of agents, so half of them in each world, and in the run, are of each kind. At each
profiled life's arm the engine's base level of each stat of the profile must be the profile's, and its hitpoints are
checked by start_hp_most against the profiled record's own drawn hitpoints and the engine's base. Each life row and
each lives-xp.jsonl row of a v14 batch holds `profile` (whether the life played the profile) and `start_levels` (the
levels its reset set, the profiled ones for a profiled life), so that every read can be split by kind; the pay and the
checks are v12's.

free-play-v15 (sim/workers.py FREE_PLAY_V15): free-play-v14 with the pay rare-xp-v2 (RareXp), an inner pay: each decision
whose XP rose in a stat s pays 10 / (10 + R) levels for s, R the run's decisions with XP in s among its last 131,072
before it. Its rows (rare-xp.jsonl) are replayed before each update by RareXp.replay, apart from the pay's own window,
and the stored rows' inner part checked against them; at each life's close RareXp.check_life recounts from the
engine's grants alone which decisions had XP in each stat, and a difference stops the run there. A checkpoint saves the
window (rare_xp_window), which a leg restores. Its life rows add rare_xp (by stat) and rare_xp_all, its lives-xp rows
rare_xp; the levels' pay and its reads are v14's.

free-play-v11 (sim/workers.py FREE_PLAY_V11): free-play-v10 on the wide contract (rl/wide_contract.py): every reset names
item-v4, every observation must stamp it and mask-v6, and projects to obs-v7 (project_free); the learner's action space
is act-v4's 128 rows.

The study arms free-play-v4, v5, v7, v8, v9 and v13 and their pays (first-item-v1, health-drive-v1, first-use-v1,
health-cost-v1) were removed after their studies closed (sim/workers.py REMOVED).

    NODE_RANDOM_EVENTS=false RL_SHARED_AGENTS=N PYTHONPATH=.:rl .venv/bin/python scripts/stack.py --bun BUN \\
        --output OUT/stack --timeout-seconds T --lockstep --xp-rate 1 --worlds W -- \\
        .venv-sim/bin/python -m sim.real play --output OUT/play --starts village-world-v2 \\
        --life '["uniform",6000,36000]' --agents N --key K --seed S --seconds T   # random legal play, no learner

A demo (sim.real demo): N agents of one real-engine checkpoint in one shared world, sampled, at a
pace a person can watch in the web client, through this batch and its checks. Never evidence: every record it writes says
so (DEMO['label']), it writes only inside logs/demo/, and it runs only in a demo world (scripts/stack.py --demo-spectator
NAME --demo-pace-ms MS, one world), which `play` and training refuse.

    NODE_RANDOM_EVENTS=false RL_SHARED_AGENTS=N PYTHONPATH=.:rl .venv/bin/python scripts/stack.py --bun BUN \\
        --output logs/demo/NAME/stack --timeout-seconds T --lockstep --xp-rate 1 --worlds 1 \\
        --demo-spectator viewer --demo-pace-ms 500 -- \\
        .venv-sim/bin/python -m sim.real demo --output logs/demo/NAME/play --real-checkpoint CKPT --launch LAUNCH \\
        --starts village-world-v2 --agents N --sites SITE [SITE] --key K --seed S --seconds T \\
        [--start-levels name=level,...] [--trace]
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.real view --busiest --every 30      # meanwhile: follow the crowd
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.real demo-reads logs/demo/NAME/play # after a --trace demo: reads.json
"""
import argparse
from bisect import bisect_right
from collections import Counter, deque
from copy import deepcopy
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import select
import threading
import time

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sim.audit import fractional_level
from sim.spec import INFO_FIELDS, TERMINAL_DEATH, TERMINAL_HORIZON, dtype
from sim.task import FREE_PLAY, check_life, draw_life, free_play_cfg
from item_contract import ITEM_CONTRACTS, ITEM_V2, ITEM_V3
from wide_contract import ITEM_V4, project_free
from sim.workers import (FREE_PLAY_V2, FREE_PLAY_V3, FREE_PLAY_V10, FREE_PLAY_V11, FREE_PLAY_V12, FREE_PLAY_V14,
                         FREE_PLAY_V15, FREE_PLAY_VERSIONS, OLD_LIFE, observation_table)
from sim import watch
from sim.agent import A_INV_NEXT, A_INV_PREV

PAGE_TURNS = (A_INV_PREV, A_INV_NEXT)  # the pack's page turns (rows 67, 68): a life row's page_turns counts them

ROOT = Path(__file__).resolve().parents[1]
REAL = dict(
    engine='real', task=FREE_PLAY['version'], observation=FREE_PLAY['observation'], protocol='shared-lockstep-v1',
    worlds=8, agents=128,           # 1,024 agents, the learner's frozen n_envs
    radius=16,                      # start tiles within 16 tiles of a site (the engine's places)
    tiles=256,                      # start tiles fetched a site (the engine deals a site's usable tiles again past them)
    stall=120.0,                    # seconds a world may take to answer one batch request
    stuck_ticks=500,                # world ticks an item may take (a decision's waits end near 100)
    max_age=8,                      # updates a row may be old in a batch; a segment with an older row is dropped
    answer_limit_ms=3_600_000,      # the engine's limit on a held world: longer than any update (the protocol's most)
    ping_seconds=60.0,              # a keep-alive to every world this often while the learner works between collections
    reset_failures=('spawn-not-landed', 'declared-start', 'setup', 'left-world', 'reset'),  # the engine's reset errors
    failures_in_row=5,              # failed resets of one agent in a row that stop the run
    failure_share=0.02,             # the share of failed resets that stops the run, counted after failure_floor resets
    failure_floor=200,
    xp_rate=1,
)
DEMO = dict(
    label='demo: not evaluation evidence',
    root='logs/demo',               # a demo writes only inside this folder (demo_checks)
    agents=48,                      # agents in the one world
    radius=8,                       # start tiles within 8 tiles of a site, so a start is inside the client's view
    view_tiles=15,                  # the client draws players within 15 tiles (rsbuf BuildArea.PREFERRED_VIEW_DISTANCE)
    view_every=5.0,                 # wall seconds between two looks at the world (views.jsonl)
    wait=600.0,                     # seconds the demo waits for its spectator to log in
)
S = FREE_PLAY['skills']
R_SITE = 41  # the purpose of a life's site and tile draw (sim/task.py R_LIFE is 40)
LEG_STREAMS = 1 << 20  # the draw streams of a leg: leg N's agent j draws stream j + (N - 1) x LEG_STREAMS
OPS = ('reset', 'step', 'leave')  # the phases of an agent inside an item
KIND = dict(reset='reset', step='step', leave='left')  # each item's result kind


class ProtocolError(RuntimeError):
    """A reply the protocol does not allow, or the engine's error or fatal message; `world` and `slot` name the agent
    and `result` holds its result when one is known."""

    def __init__(self, message, world=None, slot=None, result=None):
        super().__init__(message)
        self.world, self.slot, self.result = world, slot, result


class RealStop(RuntimeError):
    """The run stops before the next update (module docstring): its `reason`, and the evidence written to `path`."""

    def __init__(self, reason, path=None):
        super().__init__(reason if path is None else f'{reason} (evidence {path})')
        self.reason, self.path = reason, path


# ---- the wire protocol -----------------------------------------------------------------------------------------------
class WorldClient:
    """One world's shared-lockstep-v1 controller connection (the header of env-ts/engine/batch.ts is the protocol): the
    `ready` line, `hello` (the stat names of every pay and ledger array, the XP rate, the capacity, the answer limit, the
    tick), `places` (start tiles within a radius of a centre), `batch` (items for exactly the due slots: reset with the
    setup, step, leave; each result with its kind, tick, ticks, obs, done, dead, pay with the running ledger, facts and,
    on a reset or leave, the closed life's ledger) and `close`. An error or fatal message, or a closed connection, raises
    ProtocolError. transcript: a file object that gets every request and reply (the contract test's fixture)."""

    def __init__(self, port, agents, name='real', transcript=None):
        from vec_adapter import EngineEnvServer
        transport = EngineEnvServer(port, name)  # its ready line read
        self.sock, self.buf, self.port, self.transcript = transport.sock, transport._buf, port, transcript
        hello = self.call(dict(cmd='hello', protocol=REAL['protocol'], answerLimitMs=REAL['answer_limit_ms']))
        if hello.get('protocol') != REAL['protocol'] or hello.get('capacity', 0) < agents:
            raise ProtocolError(f"a world of {REAL['protocol']} for {agents} agents, not {hello}")
        self.skills = [s.lower() for s in hello['skills']]
        self.xp_rate, self.world_id, self.answer_limit = hello.get('xpRate'), hello.get('worldId'), hello['answerLimitMs']
        self.tick, self.random_events = hello['tick'], hello.get('randomEvents')
        self.fact_fields = hello.get('factFields')  # the fields of every result's facts (env-ts FACT_FIELDS), if named

    def call(self, request, timeout=60.0):
        self.send(request)
        deadline = time.monotonic() + timeout
        while (reply := self.poll()) is None:
            if not select.select([self.sock], [], [], max(0.0, deadline - time.monotonic()))[0]:
                raise ProtocolError(f'no reply to {request["cmd"]} in {timeout} s')
            self.read()
        return reply

    def send(self, request):
        if self.transcript is not None:
            self.transcript.write(json.dumps(dict(port=self.port, request=request)) + '\n')
        self.sock.sendall((json.dumps(request) + '\n').encode())

    def read(self):
        chunk = self.sock.recv(1 << 20)
        if not chunk:
            raise ProtocolError(f'the engine on port {self.port} closed the connection (any engine exit ends its world)')
        self.buf += chunk

    def poll(self):
        """The next whole reply in the buffer, or None; an error or fatal reply raises."""
        while (nl := self.buf.find(b'\n')) >= 0:
            line, self.buf = self.buf[:nl].decode('utf-8', 'replace'), self.buf[nl + 1:]
            if line.startswith('PROTO|'):
                reply = json.loads(line[len('PROTO|'):])
                if self.transcript is not None:
                    self.transcript.write(json.dumps(dict(port=self.port, reply=reply)) + '\n')
                if reply.get('type') in ('error', 'fatal'):
                    raise ProtocolError(f"engine {reply['type']} {reply.get('code')} on port {self.port}: "
                                        f"{reply.get('message')}")
                return reply
        return None

    def places(self, count, seed, centre, radius):
        tiles = self.call(dict(cmd='places', count=count, seed=seed, centres=[list(centre)], radius=radius))['tiles']
        return list(dict.fromkeys((t['x'], t['z']) for t in tiles))  # each usable tile once

    def facts(self):
        """The world's facts from its hello."""
        return dict(skills=self.skills, xp_rate=self.xp_rate, world_id=self.world_id, answer_limit=self.answer_limit,
                    tick=self.tick, random_events=self.random_events, fact_fields=self.fact_fields)

    def close(self):
        try:
            self.send(dict(cmd='close'))
        except OSError:
            pass
        self.sock.close()


def pay_of(result, skills):
    """({stat: XP in tenths}, {stat: base level}, the running ledger [stat order] or None) of a result, by the stat names
    of the world's hello."""
    pay = result['pay']
    return dict(zip(skills, pay['xp'])), dict(zip(skills, pay['base'])), pay.get('ledger')


def gather(clients, requests, stall):
    """Send each world of `requests` ({world: items}) its batch request and read every reply (`clients`: {world:
    WorldClient}). Returns ({world: reply}, the replies' arrival times, by time.perf_counter). A world silent for `stall`
    seconds raises ProtocolError."""
    for w, items in requests.items():
        clients[w].send(dict(cmd='batch', items=items))
    replies, arrived = {}, []
    deadline = time.monotonic() + stall
    while True:
        for w in requests:
            if w not in replies and (reply := clients[w].poll()) is not None:
                replies[w] = reply
                arrived.append(time.perf_counter())
        silent = [w for w in requests if w not in replies]
        if not silent:
            break
        ready = select.select([clients[w].sock for w in silent], [], [], max(0.0, deadline - time.monotonic()))[0]
        if not ready:
            raise ProtocolError(f'worlds {silent} stalled: no reply in {stall} s')
        for w in silent:
            if clients[w].sock in ready:
                clients[w].read()
    return replies, arrived


def project(replies):
    """Each result's observation projected (its `projected`: obs-v6, or obs-v7 under item-v4, by its stamp:
    rl/wide_contract.py project_free); one that does not project raises ProtocolError naming its world and slot, with
    the result."""
    for w, reply in replies.items():
        for result in reply.get('results') or []:
            if result.get('obs') is not None:
                result['contract'] = result['obs'].get('contractVersion')  # the item contract the engine stamped
                try:
                    result['projected'] = project_free(result['obs'])
                except Exception as error:
                    raise ProtocolError(f"world {w} slot {result.get('slot')}: an observation that does not project: "
                                        f'{type(error).__name__}: {error}', world=w, slot=result.get('slot'),
                                        result=result) from error


def ping(clients):
    """A keep-alive to every world of `clients` ({world: WorldClient}), each answered with a pong."""
    for client in clients.values():
        if client.call(dict(cmd='ping'), timeout=30.0).get('type') != 'pong':
            raise ProtocolError(f'a ping to port {client.port} got no pong')


class Local:
    """The worlds' clients in this process (no workers): the batch's transport, as Workers."""
    count = 0

    def __init__(self, ports, agents, transcript=None):
        self.clients = {}
        try:
            for w, port in enumerate(ports):
                self.clients[w] = WorldClient(port, agents, name=f'real{w}', transcript=transcript)
        except BaseException:
            self.close()
            raise
        self.hellos = [c.facts() for c in self.clients.values()]

    def exchange(self, requests, stall):
        replies, arrived = gather(self.clients, requests, stall)
        project(replies)
        return replies, arrived

    def places(self, *args):
        return self.clients[0].places(*args)

    def ping(self):
        ping(self.clients)

    def last_raw(self, world, slot):
        return None  # the agent holds its own (Agent.raw), the result its reply's

    def close(self):
        for client in self.clients.values():
            try:
                client.close()
            except OSError:
                pass
        self.clients = {}


def _serve(conn, worlds, ports, agents):
    """A worker process: the clients of `worlds` (at `ports`). Each command comes numbered and its answer carries the
    number: ('ok', n, value) or ('error', n, reason, world, slot, result). A batch command sends the requests, reads and
    decodes the replies (gather), keeps each slot's last two raw observations (the stop's evidence: the one a decision
    used and its reply's), projects them (project) and sends the replies without the raw observations. The hello facts
    carry the digest of the name table this process loaded (rl/names.py), which the projection reads."""
    from collections import deque
    from names import name_table
    clients, last = {}, {}
    try:
        for w, port in zip(worlds, ports):
            clients[w] = WorldClient(port, agents, name=f'real{w}')
        conn.send(('ok', 0, {w: dict(c.facts(), names=name_table().digest) for w, c in clients.items()}))
    except BaseException as error:
        conn.send(('error', 0, f'{type(error).__name__}: {error}', None, None, None))
        return
    while True:
        command, n, *args = conn.recv()
        try:
            if command == 'batch':
                replies, arrived = gather(clients, *args)
                for w, reply in replies.items():
                    for result in reply.get('results') or []:
                        if result.get('obs') is not None:
                            last.setdefault((w, result['slot']), deque(maxlen=2)).append(result['obs'])
                project(replies)
                for reply in replies.values():
                    for result in reply.get('results') or []:
                        result.pop('obs', None)
                conn.send(('ok', n, (replies, arrived)))
            elif command == 'places':
                conn.send(('ok', n, clients[min(clients)].places(*args)))
            elif command == 'raw':
                conn.send(('ok', n, list(last.get(tuple(args), ()))))
            elif command == 'ping':
                ping(clients)
                conn.send(('ok', n, None))
            else:
                for client in clients.values():
                    try:
                        client.close()
                    except OSError:
                        pass
                conn.send(('ok', n, None))
                return
        except BaseException as error:
            conn.send(('error', n, f'{type(error).__name__}: {error}', getattr(error, 'world', None),
                       getattr(error, 'slot', None), getattr(error, 'result', None)))


class Workers:
    """The worlds' clients in worker processes, min(count, worlds) of them, each serving a contiguous group of worlds
    (_serve): the decoding and the projection of every reply leave the learner's process, which keeps the lives, the
    checks, the policy and the segments. The same transport as Local: the replies come back with their projected
    observations, in world order. Every command is numbered and its answer must carry its number; a worker that ends
    or answers out of turn raises ProtocolError; a worker whose name table differs from this process's is refused."""

    def __init__(self, ports, agents, count):
        import multiprocessing as mp
        from names import name_table
        ctx = mp.get_context('spawn')
        groups = [list(g) for g in np.array_split(np.arange(len(ports)), min(count, len(ports))) if len(g)]
        self.count, self.of, self.sent = len(groups), {int(w): k for k, g in enumerate(groups) for w in g}, 0
        self.conns, self.processes = [], []
        try:
            for g in groups:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child, [int(w) for w in g], [ports[w] for w in g], agents),
                                      daemon=True)
                process.start()
                child.close()
                self.conns.append(parent)
                self.processes.append(process)
            hellos = {}
            for conn in self.conns:
                hellos.update(self._answer(conn, 0, 600.0))
            mine = name_table().digest
            if any(h['names'] != mine for h in hellos.values()):
                raise ProtocolError(f"a worker loaded another name table than this process's ({mine}): "
                                    f"{sorted({h['names'] for h in hellos.values()})}")
            self.hellos = [hellos[w] for w in range(len(ports))]
        except BaseException:
            self.close()
            raise

    def _send(self, k, command, *args):
        self.sent += 1
        try:
            self.conns[k].send((command, self.sent, *args))
        except (OSError, EOFError) as error:
            raise ProtocolError(f'worker {k} ended: {type(error).__name__}: {error}') from error
        return self.sent

    def _answer(self, conn, n, timeout):
        """The answer to command `n` on `conn`: its value, or ProtocolError (an error answer, one of another number, no
        answer within `timeout` s, or a worker that ended)."""
        try:
            if not conn.poll(timeout):
                raise ProtocolError(f'a worker gave no answer in {timeout} s')
            status, number, value, *where = conn.recv()
        except (EOFError, OSError) as error:
            raise ProtocolError(f'a worker ended: {type(error).__name__}: {error}') from error
        if number != n:
            raise ProtocolError(f'a worker answered command {number} in place of {n}')
        if status != 'ok':
            world, slot, result = where
            raise ProtocolError(f'a worker: {value}', world=world, slot=slot, result=result)
        return value

    def exchange(self, requests, stall):
        parts = {}
        for w, items in requests.items():
            parts.setdefault(self.of[w], {})[w] = items
        numbers = {k: self._send(k, 'batch', part, stall) for k, part in parts.items()}
        replies, arrived = {}, []
        for k, n in numbers.items():
            got, times = self._answer(self.conns[k], n, stall + 60.0)
            replies.update(got)
            arrived += times
        return replies, sorted(arrived)

    def places(self, *args):
        return self._answer(self.conns[0], self._send(0, 'places', *args), 60.0)

    def ping(self):
        numbers = [self._send(k, 'ping') for k in range(len(self.conns))]
        for conn, n in zip(self.conns, numbers):
            self._answer(conn, n, 60.0)

    def last_raw(self, world, slot):
        k = self.of[world]
        return self._answer(self.conns[k], self._send(k, 'raw', world, slot), 10.0)

    def close(self, seconds=30.0):
        """close to every worker first, then their answers and their ends against one deadline; a worker still alive
        at it is killed."""
        deadline, numbers = time.monotonic() + seconds, {}
        for k in range(len(self.conns)):
            try:
                numbers[k] = self._send(k, 'close')
            except ProtocolError:
                pass
        for k, n in numbers.items():
            try:
                self._answer(self.conns[k], n, max(0.0, deadline - time.monotonic()))
            except ProtocolError:
                pass
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.conns, self.processes = [], []


# ---- the inner pay, rare-xp-v2 (free-play-v15) ----------------------------------------------------------------------
class RareXp:
    """rare-xp-v2, free-play-v15's pay (sim/workers.py FREE_PLAY_V15), for directed exploration: a pay for XP of a rare
    kind. The run's decisions are numbered in the order the batch takes their results (rounds in turn, a round in agent
    order: world, then slot; a leg goes on from its checkpoint's number). A decision whose XP rose in a stat s (its
    result's XP against the result before, the vector Agent.stepped has checked against the engine's running ledger; one
    event a stat a decision, however many grants) pays 1 / (1 + R / 10) = 10 / (10 + R) levels for s (one correctly
    rounded division of whole numbers), R the decisions of the run with XP in s among the WINDOW decisions before it,
    counted exactly; a decision's stats in the engine's stat order, its pay their sum. Each event is a row of
    rare-xp.jsonl (round, tick, agent, world, slot, life, decision: its index in the life, n: its number in the run,
    stat, r, pay). The window lives here as a queue of (n, stat) with a count by stat; replay() counts the rows again by
    its own lists of numbers by stat. A checkpoint saves the window (rare_xp_window: the next number, and by stat the
    numbers in the window and their count), which a leg restores. At a life's close, check_life recounts from the
    engine's grants alone which of its decisions had XP in each stat. A life row and its lives-xp row add rare_xp, the
    life's pay by stat, and the life row rare_xp_all, its sum."""
    version = FREE_PLAY_V15['inner']
    record, row_kind, saved_as, life_sum = 'rare-xp.jsonl', 'xp', 'rare_xp_window', 'rare_xp_all'
    order = ("the run's decisions in turn (rounds in turn; a round in agent order, world then slot); a decision's stats "
             "in the engine's order")
    WINDOW, K = 131_072, 10  # the run's last 131,072 decisions (two updates of 65,536); 1 / (1 + R / K)
    params, every_keys = dict(inner_window=WINDOW, inner_k=K), ('rare_xp',)

    def __init__(self, write, path, saved=None):
        self.write, self.path, self.offset, self.round = write, Path(path), 0, 0
        self.restore(saved or dict(decisions=0, events={}))

    @classmethod
    def pay(cls, r):
        return cls.K / (cls.K + r)

    def restore(self, saved):
        """The window from its saved form (saved()): refused unless every number lies in the window of the next."""
        self.decisions = int(saved['decisions'])
        found = sorted((int(n), str(stat)) for stat, ns in saved['events'].items() for n in ns)
        if any(not self.decisions - self.WINDOW <= n < self.decisions for n, _ in found) or len(set(found)) != len(found):
            raise ValueError(f'a rare-xp-v2 window whose events lie outside the {self.WINDOW} decisions before '
                             f'{self.decisions}, or repeat')
        self.events, self.counts = deque(found), Counter(stat for _, stat in found)

    def saved(self):
        """The window as a checkpoint saves it: the next decision's number and, by stat, the numbers of the decisions in
        its window with XP in it (ascending) and their count."""
        events = {}
        for n, stat in self.events:
            if n >= self.decisions - self.WINDOW:
                events.setdefault(stat, []).append(n)
        return dict(decisions=self.decisions, counts={k: len(v) for k, v in sorted(events.items())},
                    events=dict(sorted(events.items())))

    def start(self, agent, result):
        """A life's arm: nothing counted; the ticks of its decisions start at the arm's (check_life)."""
        agent.rare_ticks = [result['tick']]

    def fresh(self, agent, result):
        """A decision's result: the decision takes its number; each stat whose XP it raised (agent.rose) is paid from the
        window before it, then joins the window. Its rows."""
        n = self.decisions
        self.decisions += 1
        agent.rare_ticks.append(result['tick'])
        while self.events and self.events[0][0] < n - self.WINDOW:
            self.counts[self.events.popleft()[1]] -= 1
        rows = []
        for stat in agent.rose:
            r = self.counts[stat]
            rows.append(dict(round=self.round, tick=result['tick'], agent=agent.j, world=agent.world, slot=agent.slot,
                             life=agent.life, decision=agent.decisions, n=n, stat=stat, r=r, pay=self.pay(r)))
            self.write(self.row_kind, **rows[-1])
        for stat in agent.rose:
            self.events.append((n, stat))
            self.counts[stat] += 1
        return rows

    @staticmethod
    def kept(row):
        return {k: row[k] for k in ('decision', 'stat', 'r', 'pay')}

    @staticmethod
    def life_fields(paid, kept):
        by_stat = {}
        for row in kept:
            by_stat[row['stat']] = by_stat.get(row['stat'], 0.0) + row['pay']
        return dict(rare_xp=by_stat, rare_xp_all=paid)

    def closing(self, agent):
        return dict(rare_ticks=list(agent.rare_ticks))

    def read_new(self):
        return read_rows(self, self.row_kind)

    def replay_state(self):
        """What the recount replays from: the window as it stands, by stat (a list of numbers each)."""
        return dict(events={stat: deque(ns) for stat, ns in self.saved()['events'].items()}, first=self.decisions)

    @classmethod
    def replay(cls, rows, state, last=None):
        """The rows in their order from `state` (replay_state), apart from the pay's own queue: each row's number at or
        after the one before (`last`: the number and the stats of its decision so far) and at or after the window's
        start, no stat twice in a decision, and its r and pay those its order gives. Returns ({(agent, life, decision):
        that decision's pay}, the last, the first problem or None)."""
        paid = {}
        for row in rows:
            n, stat = row['n'], row['stat']
            if n < state['first'] or (last is not None and (n < last[0] or n == last[0] and stat in last[1])):
                return paid, last, f'the row of {stat} at decision {n} for agent {row["agent"]} is out of order'
            numbers = state['events'].setdefault(stat, deque())
            while numbers and numbers[0] < n - cls.WINDOW:
                numbers.popleft()
            r, pay = len(numbers), cls.pay(len(numbers))
            if row['r'] != r or row['pay'] != pay:
                return paid, last, (f'agent {row["agent"]} life {row["life"]} decision {row["decision"]}: {stat} '
                                    f'recorded with r {row["r"]} and pay {row["pay"]!r}, its order gives r {r} and pay '
                                    f'{pay!r}')
            numbers.append(n)
            last = (n, (last[1] if last is not None and last[0] == n else frozenset()) | {stat})
            decision = (row['agent'], row['life'], row['decision'])
            paid[decision] = paid.get(decision, 0.0) + pay
        return paid, last, None

    def same(self, state):
        """The replayed window equals the pay's own (both cut at the next decision's window)."""
        start = self.decisions - self.WINDOW
        replayed = {stat: [n for n in ns if n >= start] for stat, ns in state['events'].items()}
        return {stat: ns for stat, ns in replayed.items() if ns} == self.saved()['events']

    @staticmethod
    def check_life(p, ledger, skills):
        """The independent check at a life's close: from the engine's grants alone ([stat, xp10, tick]: the life's
        closing ledger), which of its decisions had XP in each stat, against its paid rows (p['kinds']). The engine
        stamps a grant with World.currentTick of the cycle that makes it, and increments the tick after the cycle, so a
        result at tick t holds the grants stamped before t: decision k (from 1) owns the grants stamped in [t(k-1),
        t(k)), t(0) the arm's tick and t(k) its result's (p['rare_ticks']); a grant at a result's own tick is the next
        decision's (its action is dispatched at that tick, and its first cycle runs at it). The first problem, or None.
        A difference stops the run at this close, which comes after the life's last result: rows of the life that an
        update has already used are not taken back."""
        ticks = p['rare_ticks']
        found = set()
        for stat, xp10, tick in ledger['grants']:
            k = bisect_right(ticks, tick)
            if not 1 <= k < len(ticks):
                return (f'a grant of {xp10} tenths of {skills[stat]} at tick {tick}, outside the life\'s decisions '
                        f'(ticks {ticks[0]} to {ticks[-1]})')
            if xp10 > 0:
                found.add((k, skills[stat]))
        paid = {(row['decision'], row['stat']) for row in p['kinds']}
        if found != paid:
            return (f'the engine\'s grants give XP at (decision, stat) {sorted(found - paid)[:5]} that the pay rows lack, '
                    f'and the rows pay {sorted(paid - found)[:5]} that the grants lack')
        return None


INNER_PAYS = {RareXp.version: RareXp}


def read_rows(record, kind):
    """The rows of `kind` that `record` (its path, its offset so far) wrote since its last read, in their order."""
    with open(record.path, 'rb') as stream:
        stream.seek(record.offset)
        text = stream.read()
    whole = text[:text.rfind(b'\n') + 1]
    record.offset += len(whole)
    return [row for row in map(json.loads, whole.decode().splitlines()) if row.get('kind') == kind]


# ---- the engine's hitpoint facts -------------------------------------------------------------------------------------
def check_hurt(result):
    """The engine's facts.hurt of a result (env-ts/engine/hurt.ts: [[base, n], ...], bases ascending and distinct, base 1
    or more, n 0 or more, whole numbers), which the life rows' hurt_ticks read; refused otherwise."""
    hurt = (result.get('facts') or {}).get('hurt')
    if not isinstance(hurt, list) or any(not isinstance(e, list) or len(e) != 2 or any(type(x) is not int for x in e)
                                         or e[0] < 1 or e[1] < 0 for e in hurt) \
            or [e[0] for e in hurt] != sorted({e[0] for e in hurt}):
        raise ProtocolError(f"a result without the engine's hurt count (facts.hurt: [[base, missing hitpoint ticks], "
                            f'...]), which the life rows read: {hurt!r}')
    return hurt


def full_deficit(hurt):
    """The full-deficit ticks of a facts.hurt count, exactly: the sum over its bases of n / base."""
    return sum((Fraction(n, base) for base, n in hurt), Fraction(0))


def check_hp(result):
    """The engine's hitpoints of a result (facts.hp [current, base], whole numbers, base 1 or more), which the start check
    of free-play-v12, v14 and v15 and the life rows' start_hp read; refused otherwise."""
    hp = (result.get('facts') or {}).get('hp')
    if not isinstance(hp, list) or len(hp) != 2 or any(type(h) is not int for h in hp) or hp[0] < 0 or hp[1] < 1:
        raise ProtocolError(f"a result without the engine's hitpoints (facts.hp: [current, base]), which the start "
                            f'check reads: {hp!r}')
    return hp


# ---- one agent's lives -----------------------------------------------------------------------------------------------
HEAL_TICKS = 100  # the game's healing (content stat.rs2 [timer,health_regen], settimer(health_regen, 100) at each login)


def start_hp_most(drawn, base, reset_ticks):
    """The most hitpoints a free-play-v12, v14 or v15 life may arm at, its reset having applied `drawn` (startHp): drawn, and
    one more for each whole HEAL_TICKS of the reset item, never above the base. Each reset admits a new Player, whose
    login sets the healing timer again, so its first heal comes HEAL_TICKS after that login, inside the reset item; a
    reset item lasts some 6 to 14 ticks, so a life arms at drawn at most. An engine that ignored startHp arms at the base
    and is caught. Below drawn, down to 1, is allowed: the world is live during the reset item, and an npc that attacks
    first can hit the new player before its arm (a Highwayman at falador_farm_west, 20 November 2026, named by the
    arm's observation as its attacker; 19 of the 3,610 lives of test 1 armed 1 under their drawn hitpoints, 18 of them
    there)."""
    return min(base, drawn + reset_ticks // HEAL_TICKS)


def death_charge(result, price):
    """The price a decision's result pays (free-play-v2): `price` when the engine ended the life by a death, else 0."""
    return price if result['dead'] else 0.0


class Agent:
    """Agent j: slot `slot` of world `world`, its record stream and its life. phase: 'reset', 'step' or 'leave' (an item
    in flight), 'due' (a result that waits for its decision), 'left'. start: the next decision's episode start. pending:
    the life that ended and waits for its closing ledger (the next reset's or the leave's result). sent: the world tick
    its item in flight started at. price: the death price D (0 for free-play-v1); charge: the last decision's price.
    paid: the stats whose fractional levels are paid (S, or every stat of the ledger for free-play-v3, all_skills);
    extra: (item, count) pairs the start kit holds beside the record's (free-play-v3's tinderbox). first: the inner pay,
    the batch's RareXp (free-play-v15; None otherwise): the stats whose XP its last decision raised (rose), that
    decision's inner pay (inner), the life's (inner_life) and what its rows keep (kinds). life_facts: the four reads of
    the module docstring, kept when the engine reports food and food_drops. start_rule: free-play-v14's old-life-v1 is
    played (its rows' profile and start_levels); profile: this agent's levels of it (parse_levels), None for an agent
    that plays its records as drawn."""

    def __init__(self, j, world, slot, key, group, stream, price=0.0, paid=S, extra=(), all_skills=False, first=None,
                 life_facts=False, items=ITEM_V2, start_hp=False, hurt_facts=False, start_rule=False, profile=None):
        self.j, self.world, self.slot, self.name, self.stream = j, world, slot, f'w{world}s{slot}', stream
        self.start_hp = start_hp  # free-play-v12 to v14: each reset names the record's drawn hitpoints (startHp)
        self.start_rule, self.profile = start_rule, profile  # free-play-v14: old-life-v1, and this agent's levels of it
        self.hurt_facts = hurt_facts  # the engine reports facts.hurt: the life rows' hurt_ticks and start_hp reads
        self.price, self.charge, self.paid_skills, self.extra = price, 0.0, tuple(paid), tuple(extra)
        self.items = items  # the item contract each life's reset names (free-play-v10: item-v3)
        self.all_skills = all_skills  # free-play-v3: every stat paid (RealBatch's flag, the row's v3 fields with it)
        self.first, self.inner = first, 0.0
        self.life_facts = life_facts
        self.records, self.life, self.checked = watch.Records(key, stream, group), -1, 0
        self.phase, self.start, self.pending, self.versions, self.armed, self.failed_in_row = None, 1.0, None, [], False, 0
        self.action = None  # the action of its decision in flight (RealBatch.answer)

    def begin(self, key, sites, tiles, life):
        """The next life's draws and its reset item, with the setup the engine runs (a Wield when the weapon starts worn)."""
        self.life += 1
        self.record = watch.profiled(self.records.next(), self.profile)  # the record itself when profile is None
        rng = np.random.default_rng(np.random.SeedSequence([key, self.stream, self.life, R_SITE]))
        self.site = sites[int(rng.integers(len(sites)))]
        self.tile = tiles[self.site][int(rng.integers(len(tiles[self.site])))]
        self.length = draw_life(key, self.stream, self.life, life)
        self.phase, self.armed = 'reset', False
        return dict(op='reset', slot=self.slot, seed=self.record['reset_seed'],
                    cfg=watch.life_cfg(self.tile, self.record, self.extra, self.items,
                                       hp=self.record['hp_drawn'] if self.start_hp else None),
                    setup=dict(wield=self.record['weapon'] if self.record['worn'] else None))

    def leave(self):
        """The leave item at the run's end; a live life ends as 'left' and waits for the leave's ledger."""
        if self.armed and self.pending is None:
            self.close_life('left', self.base)
        self.phase = 'leave'
        return dict(op='leave', slot=self.slot)

    def arm(self, result, xp, base, running, skills):
        """The reset's result: the life starts at its tick, its first decision due; its running ledger starts at zero."""
        if running is None or any(running):
            raise ProtocolError(f'a life started with a running ledger of {running}')
        self.raw, self.obs, self.phase, self.start, self.armed = result.get('obs'), result['projected'], 'due', 1.0, True
        self.tick = self.arm_tick = result['tick']
        self.setup_ticks, self.xp0, self.xp, self.base0, self.base = result['ticks'], xp, xp, base, base
        self.level = [fractional_level(xp[s]) for s in self.paid_skills]
        self.paid, self.decisions, self.eats_full, self.worn, self.versions = 0.0, 0, 0, 0, []
        self.page_turns = 0  # the life's decisions that played a pack page turn (PAGE_TURNS): a read, paid nothing
        self.facts, self.priced, self.rewarded = result['facts'], 0.0, 0.0
        if self.first is not None:  # rare-xp-v2: nothing paid at an arm
            self.inner_life, self.kinds = 0.0, []
            self.first.start(self, result)
        if self.life_facts:
            self.half_hp, self.half_hp_food, self.half_hp_eats = 0, 0, 0
        if self.hurt_facts:  # the hitpoints at the arm: a read, paid nothing
            self.arm_hp = list(check_hp(result))
        if self.profile is not None and any(base[s] != level for s, level in self.profile.items()):
            raise RealStop(f"agent {self.j} life {self.life}: the engine armed the life at levels "
                           f"{ {s: base[s] for s in self.profile} }, not the profile's {self.profile}")
        if self.start_hp:  # free-play-v12 to v14: the engine applied the drawn hitpoints (start_hp_most)
            hp = check_hp(result)
            most = start_hp_most(self.record['hp_drawn'], hp[1], result['ticks'])
            if not 1 <= hp[0] <= most:
                raise RealStop(f"agent {self.j} life {self.life}: the engine started the life at hitpoints {hp}, none or "
                               f"above its start record's drawn {self.record['hp_drawn']} (at most {most} after a reset "
                               f"item of {result['ticks']} ticks)")

    def stepped(self, result, xp, base, running, skills):
        """A decision's result: (its pay, its ticks, the life's end: 'death', 'horizon' or None). The XP gained since
        the life started must equal the engine's running ledger (the pay check at every reply). With a death price the
        decision's price (charge, death_charge) must be D exactly when the reply holds the engine's death record, and
        the life's prices D only once (the price check at every reply)."""
        ticks = result['tick'] - self.tick
        if ticks < 1 or ticks != result['ticks']:
            raise ProtocolError(f"a decision of {ticks} ticks, the engine's {result['ticks']}")
        if result['done'] and not result['dead']:
            raise ProtocolError('the engine\'s agent limits ended a life (free_play_cfg)')
        gained = [xp[s] - self.xp0[s] for s in skills]
        if running != gained:
            raise RealStop(f'agent {self.j} life {self.life}: the pay check failed at tick {result["tick"]}: the results '
                           f'gained {dict(zip(skills, gained))}, the engine\'s running ledger {running}')
        level = [fractional_level(xp[s]) for s in self.paid_skills]
        pay = sum(b - a for a, b in zip(self.level, level))
        before, facts = self.facts, result['facts']
        if self.price:
            charge, owed = death_charge(result, self.price), self.price if facts['death'] is not None else 0.0
            if charge != owed or self.priced + charge != owed:
                raise RealStop(f'agent {self.j} life {self.life}: the price check failed at tick {result["tick"]}: a price '
                               f'of {charge!r} ({self.priced!r} before it) at a reply whose engine death record is '
                               f'{facts["death"]}, the price {self.price!r}')
            self.charge, self.priced, self.rewarded = charge, self.priced + charge, self.rewarded + pay - charge
        if self.first is not None:  # the stats whose XP this decision raised, in the engine's order (rare-xp-v2)
            self.rose = tuple(s for s in skills if xp[s] > self.xp[s])
        self.worn += before['weapon'] is not None  # the weapon worn at the decision's observation
        self.page_turns += self.action in PAGE_TURNS
        self.eats_full += facts['eats'] > before['eats'] and before['hp'][0] >= before['hp'][1]
        self.level, self.paid, self.decisions, self.xp, self.base, self.facts = level, self.paid + pay, \
            self.decisions + 1, xp, base, facts
        self.raw, self.obs, self.tick = result.get('obs'), result['projected'], result['tick']
        if self.first is not None:  # the rare-XP rows of this decision's stats: its inner pay
            rows = self.first.fresh(self, result)
            self.inner = sum(r['pay'] for r in rows)  # in the rows' order, as recount() adds them
            self.inner_life += self.inner
            self.kinds += [self.first.kept(r) for r in rows]
        if self.life_facts:  # at half hitpoints or less before the decision: with food in the pack, and its Eats
            hurt = 2 * before['hp'][0] <= before['hp'][1]
            self.half_hp += hurt
            self.half_hp_food += hurt and before['food'] > 0
            self.half_hp_eats += (facts['eats'] - before['eats']) if hurt else 0
        end = 'death' if result['dead'] else 'horizon' if self.tick - self.arm_tick >= self.length else None
        if end is not None:
            self.close_life(end, base)
        return pay, ticks, end

    def close_life(self, end, base):
        gain = {s: base[s] - self.base0[s] for s in S if base[s] != self.base0[s]}
        paid = {s: base[s] - self.base0[s] for s in self.paid_skills if base[s] != self.base0[s]}
        every = {s: base[s] - self.base0[s] for s in base if base[s] != self.base0[s]}  # lives-xp.jsonl's
        self.pending = dict(life=self.life, end=end, tick=self.tick, xp=self.xp, gain=gain, gain_paid=paid, facts=self.facts,
                            gain_every=every,
                            record=self.record['index'], reset_seed=self.record['reset_seed'], site=self.site,
                            tile=list(self.tile), length=self.length, arm_tick=self.arm_tick, setup_ticks=self.setup_ticks,
                            decisions=self.decisions, paid=self.paid, xp0=self.xp0, eats_full=self.eats_full,
                            worn=self.worn, versions=self.versions, price=self.priced, rewarded=self.rewarded,
                            page_turns=self.page_turns,
                            **({} if self.first is None else dict(inner=self.inner_life, kinds=list(self.kinds),
                                                                   **self.first.closing(self))),
                            **({} if not self.life_facts else dict(
                                half_hp_decisions=self.half_hp, half_hp_food_decisions=self.half_hp_food,
                                half_hp_eats=self.half_hp_eats, food_drops=self.facts['food_drops'],
                                **({'tool_drops': self.facts['tool_drops']} if 'tool_drops' in self.facts else {}))),
                            **({} if not self.hurt_facts else dict(
                                hurt_ticks=float(full_deficit(check_hurt(dict(facts=self.facts)))), start_hp=self.arm_hp)),
                            **({} if not self.start_rule else dict(profile='profile' in self.record,
                                                                   start_levels=dict(self.record['levels']))))
        self.armed = False

    def settle(self, ledger, skills, check):
        """The ended life's row (module docstring) at its closing ledger, and the pay check's problem (None when it
        holds). check False: the ledger is not compared (a play run without the check)."""
        p, self.pending = self.pending, None
        paid = [p['xp'][s] - p['xp0'][s] for s in skills]
        fractional = sum(fractional_level(p['xp'][s]) - fractional_level(p['xp0'][s]) for s in self.paid_skills)
        problems = []
        if abs(p['paid'] - fractional) > 1e-9 * max(1.0, abs(fractional)):
            problems.append(f"paid {p['paid']!r}, the fractional level gain {fractional!r}")
        if self.price:  # the rewards: the level gain less D exactly when the engine ended the life by a death
            owed = self.price if p['facts']['death'] is not None else 0.0
            if p['price'] != owed or abs(p['rewarded'] - (fractional - owed)) > 1e-9 * max(1.0, abs(fractional - owed)):
                problems.append(f"rewards {p['rewarded']!r} with a price of {p['price']!r}; the ledger's level gain "
                                f"{fractional!r} less {owed!r} (the engine's death record {p['facts']['death']})")
        if check:
            if ledger is None:
                problems.append('the item that closed the life returned no ledger')
            elif ledger['total'] != paid or (ledger['start'], ledger['end']) != (p['arm_tick'], p['tick']):
                problems.append(f"the results paid {dict(zip(skills, paid))} over ticks {p['arm_tick']}..{p['tick']}, the "
                                f"engine's ledger {dict(zip(skills, ledger['total']))} over {ledger['start']}..{ledger['end']}")
            elif self.first is not None:  # rare-xp-v2: its events from the engine's grants
                problem = self.first.check_life(p, ledger, skills)
                if problem is not None:
                    problems.append(f'{self.first.version} against the engine\'s grants: {problem}')
        death, ticks = p['facts']['death'], p['tick'] - p['arm_tick']
        row = dict(agent=self.j, world=self.world, slot=self.slot, life=p['life'], record=p['record'],
                   reset_seed=p['reset_seed'], site=p['site'], tile=p['tile'], length=p['length'], arm_tick=p['arm_tick'],
                   setup_ticks=p['setup_ticks'], ticks=ticks, overrun=ticks - p['length'] if p['end'] == 'horizon' else None,
                   decisions=p['decisions'], terminal=p['end'], death_tick=death['tick'] if death else None,
                   killers=death['killers'] if death else None, gain=p['gain'], gain_s=sum(p['gain'].values()),
                   xp={s: v for s, v in zip(skills, paid) if v and s in S}, paid=p['paid'], eats=p['facts']['eats'],
                   eats_full=p['eats_full'], worn_share=round(p['worn'] / p['decisions'], 4) if p['decisions'] else None,
                   page_turns=p['page_turns'],  # a read that nothing pays: the life's decisions that turned the pack page
                   ledger='not read' if not check else 'differs' if problems else 'match',
                   grants=len(ledger['grants']) if ledger else None,
                   versions=[min(p['versions']), max(p['versions'])] if p['versions'] else None,
                   **(dict(price=p['price'], return_with_price=p['rewarded']) if self.price else {}),
                   **({} if not self.all_skills else dict(  # free-play-v3: paid is every stat's; the seven's apart
                       paid_seven=sum(fractional_level(p['xp'][s]) - fractional_level(p['xp0'][s]) for s in S),
                       gain_all=p['gain_paid'], gain_all_s=sum(p['gain_paid'].values()),
                       xp_all={s: v for s, v in zip(skills, paid) if v})),
                   # the inner pay apart from `paid` (the levels alone): free-play-v15's rare_xp and rare_xp_all
                   **({} if self.first is None else self.first.life_fields(p['inner'], p['kinds'])),
                   **{k: p[k] for k in (*LIFE_FACTS, TOOL_FACT, *HURT_FACTS) if k in p},  # the reads, when reported
                   **{k: p[k] for k in START_FIELDS if k in p})  # free-play-v14: the life's kind and start levels
        self.checked = p['life'] + 1
        self.every = dict(agent=self.j, world=self.world, slot=self.slot, life=p['life'], terminal=p['end'],
                          xp={s: v for s, v in zip(skills, paid) if v}, levels=p['gain_every'],  # lives-xp.jsonl's row
                          **{k: p[k] for k in START_FIELDS if k in p},
                          **({} if self.first is None else {k: row[k] for k in self.first.every_keys}))
        return row, '; '.join(problems) or None


# ---- the batch -------------------------------------------------------------------------------------------------------
# the reads of a life row that nothing pays (module docstring), kept when every world's hello names ENGINE_LIFE_FACTS
LIFE_FACTS = ('half_hp_decisions', 'half_hp_food_decisions', 'half_hp_eats', 'food_drops')
ENGINE_LIFE_FACTS = ('food', 'food_drops')
TOOL_FACT = 'tool_drops'  # the fifth read, beside the four when every world's hello also names it (module docstring)
HURT_FACTS = ('hurt_ticks', 'start_hp')  # the reads of facts.hurt and the arm's hitpoints, when every hello names hurt
START_FIELDS = ('profile', 'start_levels')  # free-play-v14's: whether a life played old-life-v1, and its start levels


def env_ts_commit():
    """The last commit that changed env-ts/ and whether env-ts/ differs from it (a local git read)."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    run = lambda *argv: subprocess.run(['git', '-C', str(root), *argv], capture_output=True, text=True, check=True).stdout
    try:
        return dict(commit=run('log', '-1', '--format=%H', '--', 'env-ts').strip(),
                    changed=bool(run('status', '--porcelain', '--', 'env-ts').strip()))
    except (subprocess.CalledProcessError, OSError) as error:
        return dict(commit=None, error=f'{type(error).__name__}: {error}')


def check_settings(*, starts, life, worlds, agents, sites, radius):
    """The batch's settings, refused before any world is joined: the world whose start group the records draw from
    (sim/watch.py village: a world of sim/combat.py VILLAGES or REAL_WORLDS, named with no default), a life-length
    distribution (sim/task.py check_life), whole numbers of worlds and agents, distinct sites of that world (sim/watch.py
    world_sites: village-world-v2 refuses a shore site) and a radius of 0 or more. Returns (its rules, the life)."""
    rules, life = watch.world_rules(starts), check_life(life)
    if any(type(v) is not int for v in (worlds, agents, radius)) or worlds < 1 or agents < 1 or radius < 0:
        raise ValueError(f'whole numbers of worlds and agents a world, and a radius of 0 or more: {worlds!r}, {agents!r}, '
                         f'{radius!r}')
    own = watch.world_sites(starts)
    if not sites or len(set(sites)) != len(sites) or not set(sites) <= set(own):
        raise ValueError(f'distinct sites of {starts} (sim/watch.py world_sites: {list(own)}): {sites!r}')
    return rules, life


def check_stack(environ, agents, demo=False):
    """The stack's facts a run records and refuses otherwise: XP rate 1, the shared mode for `agents` agents a world
    (RL_SHARED_AGENTS, at least as many), the pinned engine checkout (sim/watch.py check_engine, a local git read of
    vendor/rs-sdk) and a demo world (RL_DEMO_SPECTATOR) exactly when `demo`: a demo world serves only sim.real demo.
    Random events are read from each engine's hello (RealBatch); the stack's NODE_RANDOM_EVENTS is recorded, and stands
    in only while the engines' hello does not report them."""
    from baseline import config
    if bool(environ.get('RL_DEMO_SPECTATOR')) != demo:
        raise ValueError('a demo world (scripts/stack.py --demo-spectator) serves only sim.real demo' if not demo else
                         'sim.real demo runs inside scripts/stack.py --demo-spectator NAME --demo-pace-ms MS')
    if config.XP_RATE != REAL['xp_rate']:
        raise ValueError(f"the real engine plays at XP rate {REAL['xp_rate']} (scripts/stack.py --xp-rate 1)")
    shared = environ.get('RL_SHARED_AGENTS', '')
    if not shared.isdigit() or int(shared) < agents:
        raise ValueError(f'launch the stack with RL_SHARED_AGENTS={agents} (shared-lockstep-v1), not {shared!r}')
    tree = watch.checkout_tree()
    watch.check_engine(tree)
    return dict(stack_random_events=environ.get('NODE_RANDOM_EVENTS'), shared_agents=int(shared),
                engine_build=dict(name=watch.WATCH['engine'], tree=tree), env_ts=env_ts_commit())


class RealBatch:
    """W worlds of N agents (module docstring) for the free-play learner (collect) or a play run (play). ports: each
    world's env port (scripts/stack.py's worlds, or a test's fake engine). n_envs: the learner's segments an update
    (default W x N). starts: the world whose start group the records draw from (sim/watch.py world_rules); life: the
    life-length distribution; sites and radius: the start areas (sites: every site of the world, sim/watch.py
    world_sites, unless given; radius: the batch's, for every site without its own, sim/watch.py SITE_RADIUS); key: the
    records', sites', tiles' and lengths' key;
    output: the folder of lives.jsonl and real-stop.json; ledger: the pay check against the engine's ledger; stall:
    seconds a world may take to reply; facts: the stack's facts to record (check_stack); transcript: a path that records
    every request and reply (gzip JSONL; no workers); workers: the processes that decode and project the replies (0:
    this process, Local; else Workers, each a group of worlds; the rows and lives are the same with or without workers);
    policy_threads: torch's threads for the collection's rounds (its policy and value calls), the earlier count restored
    before the rollout is filled, checked and updated on (None: unchanged); leg: the run's leg (module docstring: 1, or a
    declared continuation's number), whose agents draw from streams offset by (leg - 1) x LEG_STREAMS; death_price: the
    levels a death takes (module docstring; 0 for free-play-v1, sim/workers.py death_price); all_skills, kit_extra:
    free-play-v3's every stat of the ledger paid and its (item, count) pairs beside each record's kit (sim/workers.py
    task_batch; False and none for free-play-v1); collect_device: where the
    collection's policy calls run when not on the model's own device (None: the model's own policy, on its device; a
    device: a copy of the policy there that takes the model's weights at the start of each collection, HANDOVER);
    label: a demo's label (DEMO), written into its settings and into every row and stop record it writes; profile: a
    demo's start levels for every agent (sim.real demo --start-levels: parse_levels), each life played as sim/watch.py
    profiled gives its record and armed under old-life-v1's checks, never with a start rule."""
    goals = ()  # free play has none (sim/train.py reads a batch's goals)

    def __init__(self, ports, *, agents, starts, life, key, output, n_envs=None, sites=None, radius=REAL['radius'],
                 ledger=True, stall=REAL['stall'], facts=None, transcript=None, workers=0, policy_threads=None, leg=1,
                 death_price=0.0, collect_device=None, all_skills=False, kit_extra=(), inner=None, label=None,
                 items=ITEM_V2, start_hp=False, start_rule=None, profile=None):
        sites = list(watch.world_sites(starts)) if sites is None else sites
        if profile is not None and (label is None or start_rule is not None or not profile):
            raise ValueError('start levels for every agent are a demo\'s alone (sim.real demo --start-levels), never with '
                             f'a start rule: profile {profile!r}, label {label!r}, start rule {start_rule!r}')
        self.demo_profile = None if profile is None else dict(profile)
        self.labelled = {} if label is None else dict(demo=label)
        rules, self.life = check_settings(starts=starts, life=life, worlds=len(ports), agents=agents, sites=list(sites),
                                          radius=radius)
        if type(key) is not int or key < 0:
            raise ValueError(f'the record key is a whole number of 0 or more: {key!r}')
        if policy_threads is not None and (type(policy_threads) is not int or policy_threads < 1):
            raise ValueError(f'policy_threads: a whole number of 1 or more, or none: {policy_threads!r}')
        if type(leg) is not int or leg < 1 or len(ports) * agents > LEG_STREAMS:
            raise ValueError(f'a leg of 1 or more, and at most {LEG_STREAMS} agents: leg {leg!r}, {len(ports) * agents} agents')
        if death_price not in (0.0, FREE_PLAY_V2['death_price']):
            raise ValueError(f"a death price of 0 (free-play-v1) or {FREE_PLAY_V2['death_price']:g} (free-play-v2), not "
                             f'{death_price!r}')
        self.price = float(death_price)
        self.collect_setting = None if collect_device is None else str(collect_device)
        self.all_skills, self.kit_extra = bool(all_skills), tuple((str(n), int(c)) for n, c in kit_extra)
        if self.all_skills and self.price:
            raise ValueError('a batch pays one version: free-play-v2\'s death price or free-play-v3\'s every stat, not both')
        if self.kit_extra not in ((), tuple(FREE_PLAY_V3['kit_extra'])):
            raise ValueError(f"a start kit of none or free-play-v3's tinderbox, not {self.kit_extra!r}")
        if inner not in (None, *INNER_PAYS) or (inner and not self.all_skills):
            raise ValueError(f"an inner pay of free-play-v15 ({RareXp.version!r}), with free-play-v3's every stat and kit, "
                             f'or none, not {inner!r}')
        self.inner = inner
        if start_hp not in (False, True) or (start_hp and items != ITEM_V3):
            raise ValueError(f'the drawn start hitpoints go with the page of kinds (free-play-v12, v14, v15), not start_hp '
                             f'{start_hp!r}, item contract {items!r}')
        self.start_hp = start_hp
        if start_rule not in (None, OLD_LIFE) or (start_rule and (not start_hp or agents % 2)):
            raise ValueError(f"the start rule {OLD_LIFE['name']} goes with free-play-v12 alone (the drawn start hitpoints "
                             f'on the page of kinds; free-play-v14, v15) and an even number of agents a world, not start '
                             f'rule {start_rule!r} with start_hp {start_hp!r}, {agents} agents a world')
        if inner == RareXp.version and (items != ITEM_V3 or not start_hp or start_rule is None):
            raise ValueError(f'{RareXp.version} goes with free-play-v14 alone (free-play-v15: the page of kinds, the drawn '
                             f'start hitpoints and {OLD_LIFE["name"]}), not item contract {items!r}, start_hp '
                             f'{start_hp!r}, start rule {start_rule!r}')
        self.start_rule = None if start_rule is None else deepcopy(start_rule)
        # old-life-v1's levels by the watch's own parser (its stat names and order); every agent of odd index plays them
        self.profile = None if start_rule is None else watch.parse_levels(
            ','.join(f'{stat}={level}' for stat, level in start_rule['levels'].items()))
        if items not in (*ITEM_CONTRACTS, ITEM_V4) or (items != ITEM_V2 and (
                not self.all_skills or self.kit_extra != tuple(FREE_PLAY_V3['kit_extra']))):
            raise ValueError(f'an item contract of {(*ITEM_CONTRACTS, ITEM_V4)}, and {ITEM_V3} or {ITEM_V4} with '
                             f"free-play-v3's every stat and kit alone (free-play-v10, v11, v12, v14, v15), not {items!r}")
        self.item_contract = items  # every life's reset names it, and every observation must stamp it
        self.policy_threads = policy_threads
        self.ports, self.per_world, self.sites, self.key = list(ports), agents, list(sites), key
        self.n_agents = len(ports) * agents
        self.n_envs = self.n_agents if n_envs is None else n_envs
        self.ledger, self.stall, self.output = ledger, stall, Path(output)
        self.observation, self.shapes = observation_table(items)  # obs-v7 under item-v4, else REAL's (obs-v6)
        if type(workers) is not int or workers < 0 or (workers and transcript is not None):
            raise ValueError(f'workers: a whole number of 0 or more, and a transcript only without them: {workers!r}')
        from baseline.world import JsonlEvidence
        self.output.mkdir(parents=True, exist_ok=True)
        self.transcript = None if transcript is None else gzip.open(transcript, 'xt', encoding='utf-8')
        self.leg = leg  # a leg above 1 names itself in each row of lives.jsonl and lives-xp.jsonl (LegRows)
        self.lives, self.transport = LegRows(JsonlEvidence(self.output / 'lives.jsonl'), leg), None
        self.lives_xp = LegRows(JsonlEvidence(self.output / 'lives-xp.jsonl'), leg)  # every stat's XP and levels
        self.failures = JsonlEvidence(self.output / 'failed-resets.jsonl')
        record = None if not inner else INNER_PAYS[inner].record  # rare-xp.jsonl
        self.first = None if not inner else INNER_PAYS[inner](LegRows(JsonlEvidence(self.output / record), leg),
                                                              self.output / record)
        self.recounted, self.recount_last, self.recount_paid = self.first.replay_state() if self.first else None, None, {}
        try:
            self.transport = Workers(self.ports, agents, workers) if workers else Local(self.ports, agents, self.transcript)
            hellos = self.transport.hellos
            for w, h in enumerate(hellos):
                if h['xp_rate'] != REAL['xp_rate'] or h['skills'] != hellos[0]['skills'] or h['random_events'] not in (None, False):
                    raise ValueError(f"world {w} plays at XP rate {h['xp_rate']} with random events {h['random_events']} "
                                     f"over stats {h['skills']}: not at {REAL['xp_rate']} without random events, as world 0")
            if not set(S) <= set(hellos[0]['skills']):
                raise ValueError(f"the engine's stats {hellos[0]['skills']} lack the paid skills {S}")
            # the life rows' four reads (LIFE_FACTS) when every world's engine reports food and food_drops
            self.life_facts = all(set(ENGINE_LIFE_FACTS) <= set(h.get('fact_fields') or ()) for h in hellos)
            self.tool_facts = self.life_facts and all(TOOL_FACT in (h.get('fact_fields') or ()) for h in hellos)
            self.hurt_facts = all('hurt' in (h.get('fact_fields') or ()) for h in hellos)
            if self.start_hp and not self.hurt_facts:
                raise ValueError('the drawn start hitpoints need the engine that applies them (startHp, the commit that '
                                 'added facts.hurt): a world whose hello names no hurt field (factFields '
                                 f"{[h.get('fact_fields') for h in hellos]})")
            events = {h['random_events'] for h in hellos}
            if None in events and (facts or {}).get('stack_random_events', 'false') != 'false':
                raise ValueError('an engine\'s hello does not say whether random events are off, and the stack ran '
                                 'without NODE_RANDOM_EVENTS=false')
            self.radii = {site: watch.SITE_RADIUS.get(site, radius) for site in self.sites}  # each site's start radius
            self.tiles = {site: self.transport.places(REAL['tiles'], key, watch.SITES[site], self.radii[site])
                          for site in self.sites}
        except BaseException:
            self.close()
            raise
        self.skills = hellos[0]['skills']
        own_radius = {site: r for site, r in self.radii.items() if site in watch.SITE_RADIUS}
        self.paid_skills = tuple(self.skills) if self.all_skills else S  # free-play-v3: every stat the ledger reports
        offset = (leg - 1) * LEG_STREAMS
        self.agents = [Agent(w * agents + k, w, k, key, rules['start_group'], offset + w * agents + k, self.price,
                             self.paid_skills, self.kit_extra, self.all_skills, self.first, self.life_facts,
                             items, self.start_hp, self.hurt_facts,
                             self.start_rule is not None or self.demo_profile is not None,
                             # a demo's profile: every agent; old-life-v1: the agents of odd index
                             self.demo_profile or (self.profile if (w * agents + k) % 2 else None))
                       for w in range(len(ports)) for k in range(agents)]
        self.settings_in_effect = json.loads(json.dumps(dict(
            engine=REAL['engine'], protocol=REAL['protocol'], observation=self.observation, xp_rate=REAL['xp_rate'],
            random_events=dict(off=True, source='the engines\' hello' if events == {False} else
                               'the stack\'s environment (an engine\'s hello does not report it)'),
            starts=starts, start_group=rules['start_group'], life=list(self.life), worlds=len(ports),
            agents_per_world=agents, agents=self.n_agents, n_envs=self.n_envs,
            sites={name: list(watch.SITES[name]) for name in self.sites}, radius=radius,
            **({} if not own_radius else dict(site_radius=own_radius)),  # a run whose sites have none records none
            tiles={name: len(t) for name, t in self.tiles.items()}, cfg=free_play_cfg(),
            excluded='nothing: no excluded-pickup rule and no unsupported rule', ledger=ledger,
            answer_limit_ms=hellos[0]['answer_limit'], world_ids=[h['world_id'] for h in hellos],
            workers=dict(asked=workers, running=self.transport.count),
            stuck_ticks=REAL['stuck_ticks'], max_age=REAL['max_age'], collector='segments, no hold (sim/real.py)',
            policy_threads=policy_threads, leg=leg, stream_offset=offset,
            **({} if items == ITEM_V2 else dict(item_contract=items)),  # an item-v2 run records none, as before
            **(dict(task=FREE_PLAY_V2['version'], death_price=self.price) if self.price else {}),
            **({} if self.collect_setting is None else dict(collect_device=self.collect_setting, handover=HANDOVER)),
            **(dict(task=self.task_version(), paid='every stat of the ledger',
                    paid_skills=list(self.paid_skills), kit_extra=[list(x) for x in self.kit_extra])
               if self.all_skills else {}),
            **(dict(inner_pay=self.inner, inner_order=self.first.order, inner_record=self.first.record,
                    **self.first.params) if self.inner else {}),
            **(dict(start_hp='drawn') if self.start_hp else {}),
            **(dict(start_rule=self.start_rule, profile=self.profile) if self.start_rule else {}),
            **(dict(profile=self.demo_profile, profile_agents='every agent (a demo)') if self.demo_profile else {}),
            **(dict(life_facts=list(LIFE_FACTS) + [TOOL_FACT] * self.tool_facts) if self.life_facts else {}),
            **(dict(hurt_facts=list(HURT_FACTS)) if self.hurt_facts else {}),
            **(facts or {}), **self.labelled)))
        self.queued, self.world_tick = [[] for _ in self.ports], [h['tick'] for h in hellos]
        self.counts = dict(rounds=0, decisions=0, lives=0, deaths=0, cut=0, left=0, resets=0, failed_resets=0,
                           ledger_checks=0, ledger_grants=0, running_checks=0, dropped_segments=0, engine_seconds=0.0,
                           wait_seconds=0.0, policy_seconds=0.0,
                           **(dict(levels_paid=0.0, prices=0.0, priced_deaths=0, row_prices=0.0, row_deaths=0)
                              if self.price else {}), **(dict(levels_seven=0.0, levels_all=0.0) if self.all_skills else {}),
                           **(dict(inner_paid=0.0, inner_lives=0.0) if self.inner else {}))
        # free-play-v2: each settled life's engine death record by (agent, life), until a handed-over segment's check reads
        # it (_check_rows), and the run's tolerance of the stored rows' prices (finish)
        self.recorded, self.row_tolerance = {}, 0.0
        self.lock, self.closing, self.pinger, self.ping_error, self.pings = threading.Lock(), threading.Event(), None, None, 0
        self.started, self.updates = False, []

    def task_version(self):
        """The name of the every-stat version this batch pays: free-play-v10, v12, v14 and v15 by their start hitpoints,
        start rule and inner pay on the page of kinds, free-play-v11 on item-v4, free-play-v3 on item-v2."""
        if self.item_contract == ITEM_V3:
            return (FREE_PLAY_V15 if self.inner == RareXp.version else FREE_PLAY_V14 if self.start_rule else
                    FREE_PLAY_V12 if self.start_hp else FREE_PLAY_V10)['version']
        if self.item_contract == ITEM_V4:
            return FREE_PLAY_V11['version']
        return FREE_PLAY_V3['version']

    # -- rounds --------------------------------------------------------------------------------------------------------
    def start(self):
        for a in self.agents:
            self.queued[a.world].append(a.begin(self.key, self.sites, self.tiles, self.life))
        self.started = True

    def exchange(self):
        """One round: each world with an agent inside an item gets its queued items in one request, then every reply is
        read (a world silent for `stall` seconds, or an item not back within REAL['stuck_ticks'] world ticks, stops the
        run). Returns the results as (agent, result), in agent order."""
        span = lambda w: self.agents[w * self.per_world:(w + 1) * self.per_world]
        busy = [w for w in range(len(self.ports)) if any(a.phase in OPS for a in span(w))]
        t0 = time.perf_counter()
        requests = {}
        for w in range(len(self.ports)):
            if w in busy:
                for item in self.queued[w]:
                    self.agents[w * self.per_world + item['slot']].sent = self.world_tick[w]
                requests[w] = self.queued[w]
            elif self.queued[w]:
                self.stop(f'world {w}: items for agents that are inside no item')
            self.queued[w] = []
        try:
            replies, arrived = self.transport.exchange(requests, self.stall)
        except (ProtocolError, OSError, EOFError, ValueError) as error:  # a reply line that is no JSON too
            w, slot = getattr(error, 'world', None), getattr(error, 'slot', None)
            agent = self.agents[w * self.per_world + slot] if type(w) is int and type(slot) is int \
                and 0 <= w < len(self.ports) and 0 <= slot < self.per_world else None
            self.stop(f'{type(error).__name__}: {error}', agent, result=getattr(error, 'result', None))
        if arrived:
            self.counts['engine_seconds'] += arrived[0] - t0
            self.counts['wait_seconds'] += arrived[-1] - arrived[0]
        self.counts['rounds'] += 1
        results = []
        for w, reply in sorted(replies.items()):
            if reply.get('type') != 'batch' or type(reply.get('tick')) is not int or not reply.get('results'):
                self.stop(f'world {w}: a reply {reply.get("type")!r} at tick {reply.get("tick")!r} with '
                          f'{len(reply.get("results") or [])} results to a batch request')
            self.world_tick[w] = reply['tick']
            replied = set()
            for result in reply['results']:
                slot = result.get('slot')
                agent = self.agents[w * self.per_world + slot] if type(slot) is int and 0 <= slot < self.per_world else None
                tick = result.get('tick')  # a leave's result carries the tick it left at, which a later stop may return
                if agent is None or agent.phase not in OPS or not (tick == reply['tick'] or result.get('kind') == 'left'
                                                                   and type(tick) is int and tick <= reply['tick']):
                    self.stop(f'world {w}: a result for slot {slot!r}, which is inside no item, or at another tick')
                failed_reset = agent.phase == 'reset' and result.get('kind') == 'error' \
                    and result.get('code') in REAL['reset_failures']
                if result.get('kind') != KIND[agent.phase] and not failed_reset:
                    self.stop(f"agent {agent.j}: its {agent.phase} ended as {result.get('kind')} "
                              f"{result.get('code')}: {result.get('message')}", agent, result=result)
                results.append((agent, result))
                replied.add(agent.j)
            stuck = [a.j for a in span(w) if a.phase in OPS and a.j not in replied
                     and reply['tick'] - a.sent > REAL['stuck_ticks']]
            if stuck:
                self.stop(f"world {w}: the items of agents {stuck} have not come back within {REAL['stuck_ticks']} ticks")
        return sorted(results, key=lambda r: r[0].j)

    def advance(self, results, completed=None):
        """Each result's agent moves on: a reset's or leave's result settles the life it closed (its ledger check and its
        row in lives.jsonl); a reset's arms the next life; a decision's result goes to completed(agent, pay, ticks, end),
        then to the agent's next decision or, at its life's end, its next life's reset. Returns the agents due, in agent
        order (the order free-play-v15's rare-XP pay numbers the decisions in: RareXp)."""
        if self.first is not None:
            self.first.round = self.counts['rounds']
        for agent, result in results:
            try:
                kind = result['kind']
                if 'contract' in result and result['contract'] != self.item_contract:
                    raise ProtocolError(f"an observation of the item contract {result['contract']!r}; the run plays "
                                        f'{self.item_contract!r}')
                if kind in ('reset', 'left', 'error'):
                    if agent.pending is not None:
                        self._settle(agent, result)
                    elif result.get('ledger') is not None:
                        raise ProtocolError('an item that closed no life returned a ledger')
                    if kind == 'left':
                        agent.phase = 'left'
                    elif kind == 'error':  # a reset that failed before its life was armed: the next record (docstring)
                        self._failed_reset(agent, result)
                        self.queued[agent.world].append(agent.begin(self.key, self.sites, self.tiles, self.life))
                    else:
                        self.counts['resets'] += 1
                        agent.failed_in_row = 0
                        agent.arm(result, *pay_of(result, self.skills), self.skills)
                    continue
                pay, ticks, end = agent.stepped(result, *pay_of(result, self.skills), self.skills)
                self.counts['running_checks'] += 1
            except RealStop as stop:
                if stop.path is not None:
                    raise
                self.stop(stop.reason, agent, result=result)
            except (ProtocolError, KeyError, TypeError, ValueError) as error:
                self.stop(f'agent {agent.j}: {type(error).__name__}: {error}', agent, result=result)
            self.counts['decisions'] += 1
            if completed is not None:
                completed(agent, pay, ticks, end)
            if end is None:
                agent.phase, agent.start = 'due', 0.0
                continue
            self.counts['lives'] += 1
            self.counts['deaths' if end == 'death' else 'cut'] += 1
            self.queued[agent.world].append(agent.begin(self.key, self.sites, self.tiles, self.life))
        return [a for a in self.agents if a.phase == 'due']

    def _failed_reset(self, agent, result):
        """A reset that failed before its life was armed (module docstring): its record and count, and the stop rules."""
        c = self.counts
        c['resets'] += 1
        c['failed_resets'] += 1
        agent.failed_in_row += 1
        self.failures('failed_reset', agent=agent.j, world=agent.world, slot=agent.slot, site=agent.site, tile=list(agent.tile),
                      life=agent.life, record=agent.record['index'], code=result.get('code'), message=result.get('message'),
                      tick=result.get('tick'), **self.labelled)
        if agent.failed_in_row >= REAL['failures_in_row']:
            self.stop(f"agent {agent.j}: {agent.failed_in_row} resets in a row failed (the last {result.get('code')}: "
                      f"{result.get('message')})", agent, result=result)
        if c['resets'] > REAL['failure_floor'] and c['failed_resets'] > REAL['failure_share'] * c['resets']:
            self.stop(f"{c['failed_resets']} of {c['resets']} resets failed, more than {REAL['failure_share']:.0%}",
                      agent, result=result)

    def _settle(self, agent, result):
        row, problem = agent.settle(result.get('ledger'), self.skills, self.ledger)
        self.counts['ledger_checks'] += self.ledger
        self.counts['ledger_grants'] += row['grants'] or 0
        self.counts['left'] += row['terminal'] == 'left'
        if self.price:  # settled lives: their level gain, their prices and the engine deaths those prices answer
            self.counts['levels_paid'] += row['paid']
            self.counts['prices'] += row['price']
            self.counts['priced_deaths'] += row['death_tick'] is not None
            self.recorded[agent.j, row['life']] = row['death_tick'] is not None
        if self.all_skills:  # settled lives: the seven old skills' level gain and every stat's, apart
            self.counts['levels_seven'] += row['paid_seven']
            self.counts['levels_all'] += row['paid']
        if self.first is not None:  # and their inner pay, apart from the levels
            self.counts['inner_lives'] += row[self.first.life_sum]
        self.lives('life', **row, **self.labelled)
        self.lives_xp('life', **agent.every, **self.labelled)
        if problem is not None:
            self.stop(f'agent {agent.j} life {row["life"]}: the pay check failed: {problem}', agent, row, result=result)

    def answer(self, agents, actions):
        for agent, action in zip(agents, actions):
            agent.action, agent.phase = int(action), 'step'
            self.queued[agent.world].append(dict(op='step', slot=agent.slot, action=int(action)))

    def _keep_alive(self):
        """The keep-alive thread (module docstring): while no collection holds the lock, a ping to every world each
        REAL['ping_seconds']; a failure is kept and stops the next collection."""
        while not self.closing.wait(REAL['ping_seconds']):
            with self.lock:
                if self.closing.is_set():
                    return
                try:
                    self.transport.ping()
                    self.pings += 1
                except (ProtocolError, OSError, EOFError, ValueError) as error:
                    self.ping_error = f'{type(error).__name__}: {error}'
                    return

    def _stop_pinger(self):
        self.closing.set()
        if self.pinger is not None:
            self.pinger.join(timeout=60)
            self.pinger = None

    def finish(self, on_results=None):
        """The run's end: every agent leaves as soon as it is due (an item in flight first comes back and is settled as
        any), each life a leave ends is checked against the ledger it returns, and only then do the connections close.
        on_results: play's, called with the results that come back here too. Returns the counts."""
        self._stop_pinger()
        if self.ping_error is not None:  # a ping's late answer must not become a leave's reply
            self.stop(f'a keep-alive ping before the run\'s end failed: {self.ping_error}')
        for w, items in enumerate(self.queued):  # a reset not yet sent: the agent leaves in its place
            self.queued[w] = [self.agents[w * self.per_world + item['slot']].leave() for item in items]
        for a in self.agents:
            if a.phase == 'due':
                self.queued[a.world].append(a.leave())
        while any(a.phase in OPS for a in self.agents):
            results = self.exchange()
            if on_results is not None:
                on_results(results)
            for agent, result in results:
                self.advance([(agent, result)])
                if result['kind'] != 'left':  # a decision or a reset came back: its pay and checks, then the leave
                    self.queued[agent.world] = [i for i in self.queued[agent.world] if i['slot'] != agent.slot]
                    self.queued[agent.world].append(agent.leave())
        c = self.counts
        if self.first is not None:  # the inner pay: the rows of the run's last rounds recounted too
            _, self.recount_last, problem = self.first.replay(self.first.read_new(), self.recounted, self.recount_last)
            if problem is not None or not self.first.same(self.recounted):
                self.stop(f"the inner pay recount failed at the run's end: {problem or 'the counts differ'}")
        if self.price:  # the stored rows handed to the learner against the engine's death records (module docstring)
            unhanded = sum(self.recorded.values())  # deaths whose rows never reached the learner
            if abs(c['row_prices'] - self.price * c['row_deaths']) > self.row_tolerance or \
                    c['priced_deaths'] != c['row_deaths'] + unhanded or c['priced_deaths'] != c['deaths']:
                self.stop(f"the stored rows handed to the learner hold prices of {c['row_prices']!r} for {c['row_deaths']} "
                          f"engine death records (D {self.price!r}); the life records hold {c['priced_deaths']} deaths, "
                          f"{unhanded} of them in rows not handed over; the replies {c['deaths']} deaths")
        self.close()
        return dict(self.counts)

    def stop(self, reason, agent=None, row=None, result=None):
        """Keep the evidence (real-stop.json) and stop the run (RealStop)."""
        evidence = dict(reason=reason, counts=self.counts, settings=getattr(self, 'settings_in_effect', None),
                        world_ticks=self.world_tick, **getattr(self, 'labelled', {}))
        if agent is not None:
            raw = getattr(agent, 'raw', None)
            if raw is None and self.transport is not None:  # a worker keeps the last two raw observations of a slot
                try:
                    evidence['observations'] = self.transport.last_raw(agent.world, agent.slot)
                except (ProtocolError, OSError, EOFError):
                    pass
            evidence.update(agent=agent.j, world=agent.world, slot=agent.slot, phase=agent.phase, life=agent.life,
                            record=getattr(agent, 'record', None), row=row, result=result, observation=raw)
        path = self.output / 'real-stop.json'
        path.write_text(json.dumps(evidence, indent=1, default=str) + '\n', encoding='utf-8')
        raise RealStop(reason, path)

    # -- a play run (no learner) ---------------------------------------------------------------------------------------
    def play(self, seed, actor='random', seconds=None, rounds=None, until_death=False, on_round=None, on_results=None):
        """Random legal play, WAIT or, actor 'policy', the policy bound by bind_play (a demo) through the batch for
        `seconds` of wall time or `rounds` rounds (with until_death, to the first round where a death and a cut have both
        passed their ledger check, a function of the replies, so a replay of a transcript stops there too), then finish:
        the batch's own path, no learner. on_round: called after each round (a demo's looks at its world); on_results:
        called with each round's (agent, result) pairs before the agents move on (a demo's trace)."""
        if actor == 'policy' and getattr(self, 'play_model', None) is None:
            raise ValueError('a policy plays once bind_play has bound it')
        rng = np.random.default_rng(seed)
        self.start()
        started, due, c = time.perf_counter(), [], self.counts
        done = lambda: until_death and min(c['deaths'], c['cut']) >= 1 and c['ledger_checks'] >= c['lives']
        while (seconds is None or time.perf_counter() - started < seconds) and (rounds is None or c['rounds'] < rounds) \
                and not done():
            if actor == 'policy':
                self.answer(due, self._sampled(due) if due else [])
            else:
                self.answer(due, [0 if actor == 'wait' else int(rng.choice(np.flatnonzero(a.obs['mask']))) for a in due])
            results = self.exchange()
            if on_results is not None:
                on_results(results)
            due = self.advance(results)
            if on_round is not None:
                on_round()
        played = dict(self.counts, seconds=time.perf_counter() - started)
        return dict(played, final=self.finish(on_results))

    def bind_play(self, model):
        """A play run's policy (a demo): `model` (load_free_policy's) decides for every due agent in one call a round,
        sampled, each agent from its own recurrent state, zeroed at its life's first decision (the episode start), as the
        collection's rounds do (_forward); no rollout storage, no learner and no keep-alive thread."""
        import torch
        lstm = model.policy.lstm_actor
        self.play_model = self.model = model
        self.device = model.device
        self.policy, self.collect_device = model.policy, model.device  # _forward's policy and device (bind sets them too)
        self.state = [torch.zeros(lstm.num_layers, self.n_agents, lstm.hidden_size, device=self.device) for _ in range(4)]

    def _sampled(self, agents):
        """The bound policy's sampled actions for `agents`; their recurrent states move on."""
        t0 = time.perf_counter()
        actions, _, _, new, J, _ = self._forward(agents)
        for state, value in zip(self.state, new):
            state[:, J] = value
        self.counts['policy_seconds'] += time.perf_counter() - t0
        return actions.tolist()

    # -- the learner's collection ----------------------------------------------------------------------------------------
    def bind(self, model):
        """The collector's storage for `model` (a DeviceRecurrentPPO of this batch): each agent's recurrent state after its
        last decision, its segment in flight as [n_steps, agents] rows laid out as the learner's buffer, and the weights of
        each version whose rows may still reach a batch. The policy calls run on the collection's device: the model's own
        policy on its device, or with a collect_device a copy of it there (HANDOVER); the stage, the recurrent states,
        the kept weights and the spare policy live there too, and the rollout goes to the buffer on the model's device."""
        import torch
        b, lstm = model.rollout_buffer, model.policy.lstm_actor
        self.model, self.n_steps, self.device = model, model.n_steps, model.device
        if self.collect_setting is None:
            self.policy, self.collect_device = model.policy, model.device
        else:
            self.collect_device = torch.device(self.collect_setting)
            self.policy = deepcopy(model.policy).to(self.collect_device)
        A, T, L, H = self.n_agents, self.n_steps, lstm.num_layers, lstm.hidden_size
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype, device=self.collect_device)
        self.state = [zeros(L, A, H) for _ in range(4)]  # pi h, pi c, vf h, vf c
        self.stage = dict(obs={k: zeros(T, A, *v.shape[2:], dtype=v.dtype) for k, v in b.observations.items()},
                          actions=zeros(T, A, dtype=torch.int64), values=zeros(T, A), log_probs=zeros(T, A),
                          starts=zeros(T, A), rewards=zeros(T, A), cut=zeros(T, A), version=zeros(T, A, dtype=torch.int64),
                          states=[zeros(T, A, L, H) for _ in range(4)],
                          infos={k: zeros(T, A, dtype=getattr(torch, np.dtype(v).name)) for k, v in INFO_FIELDS.items()},
                          # free-play-v15: each row's inner pay and the (life, decision) it pays, for the recount
                          **({} if self.first is None else dict(
                              inner=zeros(T, A), life=zeros(T, A, dtype=torch.int64),
                              decision=zeros(T, A, dtype=torch.int64))))
        self.pos, self.meta = [0] * A, [None] * A
        self.pool, self.fresh, self.version, self.weights = [], [], 0, {}
        self.ticks_at = list(self.world_tick)
        self.spare = deepcopy(self.policy)  # loaded with a version's weights to check its rows (--validate)
        self.pinger = threading.Thread(target=self._keep_alive, name='real-keep-alive', daemon=True)
        self.pinger.start()

    def _forward(self, agents, values_only=False):
        """One policy call for `agents`, each from its own recurrent state and episode start: (actions, values, log
        probabilities, new states, their rows, the observations), or with values_only the values alone."""
        import torch
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        from sim.learn import MODEL_DTYPES
        device = self.collect_device
        J = torch.as_tensor([a.j for a in agents], dtype=torch.int64, device=device)
        space = self.model.observation_space
        obs = {k: torch.from_numpy(np.stack([np.asarray(a.obs[k], dtype(k)) for a in agents])).to(
            device, MODEL_DTYPES[np.dtype(space[k].dtype)]) for k in self.shapes}
        starts = torch.as_tensor([a.start for a in agents], dtype=torch.float32, device=device)
        state = [s.index_select(1, J).contiguous() for s in self.state]
        with torch.no_grad():
            if values_only:
                return self.policy.predict_values(obs, (state[2], state[3]), starts).flatten()
            actions, values, log_probs, new = self.policy(obs, RNNStates((state[0], state[1]), (state[2], state[3])), starts)
        return actions, values.flatten(), log_probs, (*new.pi, *new.vf), J, obs

    def _part(self, agent):
        """Close the segment's part of the agent's life: its ticks and its fractional level gain, both from the part's
        first decision to the agent's last result (independent of the rows' decision_ticks and rewards)."""
        meta = self.meta[agent.j]
        meta['ticks'] += agent.tick - meta['part'][0]
        meta['gain'] += sum(agent.level) - meta['part'][1]
        meta['part'] = None

    def _finish(self, agent, last_value, done):
        """Agent j's segment, finished: its rows (cloned from the stage) and facts join this round's segments."""
        j, st, meta = agent.j, self.stage, self.meta[agent.j]
        if meta['part'] is not None:
            self._part(agent)
        rows = lambda v: v[:, j].clone()
        self.fresh.append(dict(obs={k: rows(v) for k, v in st['obs'].items()}, states=[rows(s) for s in st['states']],
                               infos={k: rows(v) for k, v in st['infos'].items()},
                               **{k: rows(st[k]) for k in ('actions', 'values', 'log_probs', 'starts', 'rewards', 'cut',
                                                           'version', 'inner', 'life', 'decision') if k in st},
                               agent=j, first_start=meta['first_start'], gain=meta['gain'], ticks=meta['ticks'],
                               ended=meta['ended'], last_value=float(last_value), done=done))
        self.pos[j], self.meta[j] = 0, None

    def _young(self, segments):
        """`segments` in their order without those whose oldest row is more than REAL['max_age'] updates old (counted
        as dropped)."""
        kept = [s for s in segments if self.version - int(s['version'].min()) <= REAL['max_age']]
        self.counts['dropped_segments'] += len(segments) - len(kept)
        return kept

    def _pool(self):
        """This round's finished segments join the pool, in agent order (the too old ones dropped: _young)."""
        self.pool += self._young(sorted(self.fresh, key=lambda s: s['agent']))
        self.fresh = []

    def _checked(self, segments):
        """Every life that ended inside `segments` has passed its ledger check."""
        return all(self.agents[j].checked > life for s in segments for j, life in s['ended'])

    def _decide(self, due, waiting):
        """The round's policy call for every due agent: the waiting agents' segments finish with their values from it,
        then each due agent's decision starts a row of its segment and is answered."""
        import torch
        t0 = time.perf_counter()
        actions, values, log_probs, new, J, obs = self._forward(due)
        position = {a.j: k for k, a in enumerate(due)}
        for agent in waiting:
            self._finish(agent, values[position[agent.j]], False)
        st = self.stage
        P = torch.as_tensor([self.pos[a.j] for a in due], dtype=torch.int64, device=self.collect_device)
        for k, v in obs.items():
            st['obs'][k][P, J] = v.to(st['obs'][k].dtype)
        st['actions'][P, J], st['values'][P, J], st['log_probs'][P, J] = actions, values, log_probs
        st['starts'][P, J] = torch.as_tensor([a.start for a in due], dtype=torch.float32, device=self.collect_device)
        st['version'][P, J] = self.version
        for k in ('rewards', 'cut', 'inner'):
            if k in st:
                st[k][P, J] = 0
        for v in st['infos'].values():
            v[P, J] = 0
        for store, state, value in zip(st['states'], self.state, new):
            store[P, J] = state.index_select(1, J).transpose(0, 1)
            state[:, J] = value
        for agent in due:
            meta = self.meta[agent.j]
            if meta is None:
                meta = self.meta[agent.j] = dict(first_start=agent.start, gain=0.0, ticks=0, part=None, ended=[])
            if meta['part'] is None:
                meta['part'] = (agent.tick, sum(agent.level))
            self.pos[agent.j] += 1
            agent.versions.append(self.version)
        self.answer(due, actions.tolist())
        self.counts['policy_seconds'] += time.perf_counter() - t0

    def _completed(self, done, cuts):
        """completed() of advance in a collection: a decision's pay, ticks and end into its row. A segment whose last row
        ended its life goes to `done` (finished once its cut value is in); each cut life to `cuts`."""
        def completed(agent, pay, ticks, end):
            j, t = agent.j, self.pos[agent.j] - 1
            st, info = self.stage, self.stage['infos']
            reward = pay - agent.charge  # free-play-v2: a death's row pays -D too
            if self.first is not None:  # free-play-v15: and the inner pay of its decision (RareXp)
                reward += agent.inner
                st['inner'][t, j] = agent.inner
            if 'life' in st:
                st['life'][t, j], st['decision'][t, j] = agent.life, agent.decisions
            st['rewards'][t, j] = float(np.float32(reward))
            info['decision_ticks'][t, j] = ticks
            info['goal'][t, j] = -1
            if end is None:
                return
            p = agent.pending
            info['terminal'][t, j] = TERMINAL_DEATH if end == 'death' else TERMINAL_HORIZON
            info['episode_return'][t, j] = float(np.float32(agent.paid))
            info['episode_decisions'][t, j], info['episode_ticks'][t, j] = agent.decisions, p['tick'] - p['arm_tick']
            info['outcomes'][t, j] = sum(p['gain_paid'].values())  # the paid stats' whole levels (S, or every stat)
            self.meta[j]['ended'].append((j, p['life']))
            self._part(agent)
            if end == 'horizon':
                cuts.append((agent, t))
            if t == self.n_steps - 1:
                done.append(agent)
        return completed

    def _cut_values(self, cuts):
        """Each cut life's bootstrap: the value of its last observation from the recurrent state its last decision left,
        with no episode start (sim/learn.py DeviceRecurrentPPO._cut_values), into that decision's row."""
        agents = [a for a, _ in cuts]
        starts = [a.start for a in agents]
        for a in agents:
            a.start = 0.0
        values = self._forward(agents, values_only=True)
        for a, s in zip(agents, starts):
            a.start = s
        for (agent, t), value in zip(cuts, values.tolist()):
            self.stage['cut'][t, agent.j] = value

    def collect(self, model, callback, buffer, n_rollout_steps):
        """sim/learn.py DeviceRecurrentPPO.collect_rollouts for this batch (module docstring): rounds until the pool holds
        n_envs finished segments whose ended lives have passed the ledger check; the first n_envs become the rollout
        buffer, then the callbacks' steps, the returns and advantages, the rollout's record and its end. Returns True.
        It holds the lock, so the keep-alive thread pings only between collections."""
        with self.lock:
            if self.ping_error is not None:
                self.stop(f'a keep-alive ping between collections failed: {self.ping_error}')
            return self._collect(model, callback, buffer, n_rollout_steps)

    def _collect(self, model, callback, buffer, n_rollout_steps):
        import torch
        if not self.started:
            self.bind(model)
            self.start()
        if n_rollout_steps != self.n_steps or model.n_envs != self.n_envs:
            raise ValueError('the collector was bound to another rollout shape')
        if buffer.full and self.updates:  # the last update's PPO statistics on the rollout it trained on
            self.updates[-1]['threads_after'] = thread_state()
            self.updates[-1]['after'] = lag_stats(model, buffer.segments['version'], buffer.segments['current'])
        model.policy.set_training_mode(False)
        buffer.reset()
        callback.on_rollout_start()  # ends the last update's clock (sim/train.py): the hand-over is not in its seconds
        t0 = time.perf_counter()
        handover = self._hand_over(model)  # before the first policy call of this collection
        handover_seconds = time.perf_counter() - t0
        start, t_start, before = model.num_timesteps, time.perf_counter(), dict(self.counts)
        self.pool = self._young(self.pool)  # the pool's segments are a version older now
        threads = torch.get_num_threads()  # the rounds' policy calls on policy_threads, everything after on all
        if self.policy_threads is not None:
            torch.set_num_threads(self.policy_threads)
        try:
            while True:
                due = [a for a in self.agents if a.phase == 'due']
                waiting = [a for a in due if self.pos[a.j] == self.n_steps]
                if len(self.pool) + len(self.fresh) + len(waiting) >= self.n_envs:
                    if waiting:  # their last values under the weights that collected them
                        t0 = time.perf_counter()
                        for agent, value in zip(waiting, self._forward(waiting, values_only=True).tolist()):
                            self._finish(agent, value, False)
                        self.counts['policy_seconds'] += time.perf_counter() - t0
                        waiting = []
                    self._pool()
                    if len(self.pool) >= self.n_envs and self._checked(self.pool[:self.n_envs]):
                        break  # no agent is answered: the worlds stand still during the update
                if due:
                    self._decide(due, waiting)
                self._pool()
                done, cuts = [], []
                self.advance(self.exchange(), self._completed(done, cuts))
                if cuts:
                    self._cut_values(cuts)
                for agent in done:
                    self._finish(agent, 0.0, True)
        finally:
            torch.set_num_threads(threads)
        chosen, self.pool = self.pool[:self.n_envs], self.pool[self.n_envs:]
        t_buffer = time.perf_counter()
        self._fill(buffer, chosen)
        rows = self._check_rows(chosen) if self.price else None
        inner = self._recount(chosen) if self.first is not None else None  # free-play-v15: before the update
        given = np.bincount([s['agent'] for s in chosen], minlength=self.n_agents)
        ages = self.version - buffer.segments['version']
        record = dict(update=self.version,
                      **({} if handover is None else dict(handover=handover, handover_seconds=handover_seconds)),
                      rows_by_age=age_histogram(ages), lagged_share=float((ages > 0).float().mean()),
                      segments_per_agent=[int(given.min()), int(given.max())],
                      world_ticks=[None if a is None else t - a for t, a in zip(self.world_tick, self.ticks_at)],
                      pool_left=len(self.pool), collect_seconds=time.perf_counter() - t_start,
                      **{k: self.counts[k] - before[k] for k in self.counts})
        if rows is not None:  # the rollout's stored rows: their level gain and their prices, apart (_check_rows)
            record.update(rows_levels_paid=rows[0], rows_prices=rows[1])
        if inner is not None:  # the rollout's stored rows: their level gain and their inner pay, apart (_recount)
            record.update(rows_levels_paid=inner[0], rows_inner_paid=inner[1], kinds_counted=inner[2])
        self.ticks_at = list(self.world_tick)
        self.updates.append(record)
        for _ in range(self.n_steps):  # the callbacks see the rollout's steps, n_envs decisions each
            model.num_timesteps += self.n_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        model._check_supported(buffer, start)
        buffer.compute_returns_and_advantage(
            last_values=torch.as_tensor([s['last_value'] for s in chosen], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor([s['done'] for s in chosen], device=self.device))
        model._record_rollout(buffer, start)
        if model.step_timing is not None:
            for name, value in (('policy', record['policy_seconds']),
                                ('environment', record['engine_seconds'] + record['wait_seconds']),
                                ('buffer', time.perf_counter() - t_buffer)):
                model.step_timing[name] = model.step_timing.get(name, 0.0) + value
        callback.on_rollout_end()
        record['before'] = lag_stats(model, buffer.segments['version'], self.version)  # the weights the update starts from
        record['threads_before'] = thread_state()
        for name, value in (('lagged_share', record['lagged_share']), ('segments_min', record['segments_per_agent'][0]),
                            ('segments_max', record['segments_per_agent'][1]), ('wait_seconds', record['wait_seconds']),
                            ('dropped_segments', record['dropped_segments'])):
            model.logger.record(f'real/{name}', value)
        if self.price:
            model.logger.record('real/levels_paid', record['rows_levels_paid'])
            model.logger.record('real/prices_paid', record['rows_prices'])
        if self.all_skills:  # the lives settled in this collection: the seven old skills' gain and every stat's
            model.logger.record('real/levels_seven', record['levels_seven'])
            model.logger.record('real/levels_all', record['levels_all'])
        if inner is not None:
            model.logger.record('real/inner_paid', record['rows_inner_paid'])
        # the weights that collected this version's rows (the model's: a collection copy holds them, HANDOVER), kept on
        # the collection's device while its rows may still reach a batch
        self.weights[self.version] = {k: v.detach().to(self.collect_device, copy=True)
                                      for k, v in model.policy.state_dict().items()}
        self.weights = {v: w for v, w in self.weights.items() if v >= self.version - REAL['max_age']}
        self.version += 1
        return True

    def _hand_over(self, model):
        """HANDOVER: a collection copy takes the model's weights at the start of each collection (no request goes out
        during an update, so its first round already plays them); returns their digest, which must be the model's
        (else the run stops), or None when the collection uses the model's own policy."""
        if self.policy is model.policy:
            return None
        from sim.train import policy_digest
        self.policy.load_state_dict(model.policy.state_dict())
        self.policy.set_training_mode(False)
        digest = policy_digest(self.policy)
        if digest != policy_digest(model.policy):
            self.stop(f'the collection copy on {self.collect_device} does not hold the model\'s weights after the hand-over')
        return digest

    def restore_inner(self, saved):
        """A leg's start (sim/train.py): the window its checkpoint saved (model.rare_xp_window, rare-xp-v2), which the
        pay and the recount go on from; refused unless it is this batch's inner pay's, before the batch starts."""
        if self.first is None or self.started or not isinstance(saved, dict) or saved.get('version') != self.inner:
            raise ValueError(f'a leg goes on from its checkpoint\'s counts of its own inner pay ({self.inner}), before its '
                             f'batch starts: not {saved!r}')
        self.first.restore(saved)
        self.recounted = self.first.replay_state()  # what the recount replays from

    def _recount(self, chosen):
        """free-play-v15, before each update: rare-xp.jsonl's rows written since the last recount, replayed in their
        order (RareXp.replay) from the window so far, must hold the r and the pay their order gives, leave the window the
        pay used, and give each stored row of the chosen segments its inner pay (the row's (life, decision) as the replay
        pays it, float32 as stored). A difference stops the run (RealStop) before the update. Returns (the rollout's level
        gain, its inner pay, the rows replayed); the model saves the window with its next checkpoint."""
        rows = self.first.read_new()
        paid, self.recount_last, problem = self.first.replay(rows, self.recounted, self.recount_last)
        if problem is None and not self.first.same(self.recounted):
            problem = 'the recounted counts differ from the counts the pay used'
        for decision, pay in paid.items():
            self.recount_paid[decision] = self.recount_paid.get(decision, 0.0) + pay
        if problem is not None:
            self.stop(f'the inner pay recount failed: {problem}')
        levels = inner = 0.0
        for s in chosen:
            want = [self.recount_paid.pop((s['agent'], life, decision), 0.0)
                    for life, decision in zip(s['life'].tolist(), s['decision'].tolist())]
            if [float(np.float32(w)) for w in want] != s['inner'].tolist():
                self.stop(f"agent {s['agent']}'s segment: its stored rows' inner pay {s['inner'].tolist()} is not the "
                          f'recount\'s {want}')
            levels, inner = levels + s['gain'], inner + sum(want)
        self.counts['inner_paid'] += inner
        setattr(self.model, self.first.saved_as, dict(version=self.inner, **self.first.saved()))
        return levels, inner, len(rows)

    def _check_rows(self, chosen):
        """free-play-v2: the stored rows of each segment handed to the learner, against its agent's pay and the engine's
        life records (module docstring). Its death rows are exactly the rows at which a life ended with the engine's
        death record (that life's row, settled before the hand-over: _checked); every other row's reward is a level gain,
        never negative, and a death row's at least -D; its rewards sum to the segment's own level gain (from the agent's
        levels, apart from the rewards) less D for each such death, within the float32 rounding of its rows. Returns the
        rollout's level gain and the prices its rows hold; counts them for the run's end (finish)."""
        levels = prices = 0.0
        for s in chosen:
            rewards, death = s['rewards'].double(), s['infos']['terminal'] == TERMINAL_DEATH
            died = sum(self.recorded.pop(key) for key in s['ended'])
            price, tolerance = s['gain'] - float(rewards.sum()), 1e-5 + 2 ** -23 * (abs(s['gain']) + self.price * died)
            if int(death.sum()) != died or bool((rewards[~death] < 0).any()) or \
                    bool((rewards[death] < -self.price).any()) or abs(price - self.price * died) > tolerance:
                self.stop(f"agent {s['agent']}'s segment: its stored rows hold a price of {price!r} at {int(death.sum())} "
                          f'death rows; its lives\' records hold {died} engine deaths at {self.price!r} each')
            levels, prices = levels + s['gain'], prices + price
            self.counts['row_prices'] += price
            self.counts['row_deaths'] += died
            self.row_tolerance += tolerance
        return levels, prices

    def _fill(self, buffer, chosen):
        """The chosen segments as the learner's rollout buffer, column c segment c, with the segments' own facts for
        --validate: buffer.segments (each row's version, and the current one) and, for the older versions,
        buffer.version_weights with buffer.spare_policy to load them into."""
        import torch
        stack = lambda values: torch.stack(values, dim=1)
        for k, store in buffer.observations.items():
            store.copy_(stack([s['obs'][k] for s in chosen]))
        for store, k in ((buffer.rewards, 'rewards'), (buffer.episode_starts, 'starts'), (buffer._values, 'values'),
                         (buffer.log_probs, 'log_probs'), (buffer.cut_values, 'cut')):
            store.copy_(stack([s[k] for s in chosen]))
        buffer.actions.copy_(stack([s['actions'] for s in chosen])[..., None])
        for k, store in enumerate((buffer.hidden_states_pi, buffer.cell_states_pi, buffer.hidden_states_vf,
                                   buffer.cell_states_vf)):
            store.copy_(stack([s['states'][k] for s in chosen]))
        buffer.infos = {k: stack([s['infos'][k] for s in chosen]).to(self.device) for k in INFO_FIELDS}
        buffer.pos, buffer.full = self.n_steps, True
        tensor = lambda key, kind: torch.as_tensor([s[key] for s in chosen], dtype=kind, device=self.device)
        buffer.segments = dict(agent=tensor('agent', torch.int64), first_start=tensor('first_start', torch.float32),
                               gain=tensor('gain', torch.float64), ticks=tensor('ticks', torch.int64),
                               version=stack([s['version'] for s in chosen]).to(self.device), current=self.version,
                               **({} if self.first is None else dict(  # free-play-v15: the rows' inner pay (--validate)
                                   inner=stack([s['inner'] for s in chosen]).to(self.device, torch.float64))))
        buffer.version_weights, buffer.spare_policy, buffer.collection_policy = self.weights, self.spare, self.policy

    def report(self):
        """The batch's record for the run's summary: its settings in effect, counts, each update's collector record and
        the device note."""
        return dict(settings=self.settings_in_effect, counts=self.counts, updates=self.updates, pings=self.pings,
                    timing_note=TIMING_NOTE,
                    lives=str(self.output / 'lives.jsonl'), failed_resets=str(self.output / 'failed-resets.jsonl'),
                    device_note=DEVICE_NOTE_GPU if str(getattr(self, 'device', 'cpu')).startswith('cuda') else DEVICE_NOTE)

    def close(self):
        if getattr(self, 'closing', None) is not None:
            self._stop_pinger()
        if getattr(self, 'transport', None) is not None:
            self.transport.close()
            self.transport = None
        for name in ('lives', 'lives_xp', 'failures', 'transcript'):
            if getattr(self, name, None) is not None:
                getattr(self, name).close()
                setattr(self, name, None)
        if getattr(self, 'first', None) is not None:  # rare-xp.jsonl (free-play-v15); the window stays readable
            self.first.write.close()


TIMING_NOTE = ('engine_seconds: from a round\'s requests to its first reply; wait_seconds: from the first reply to the last. '
               'Without workers both are read in this process as each reply arrives; with workers, in each worker as its '
               'replies arrive, before their decoding and projection, which then fall in the main process\'s other '
               'wall time (its wait for the workers\' answers).')
DEVICE_NOTE = ('The collection makes one policy call a round over the agents due in it (a batch of varying size), while '
               '--validate recomputes the stored rows in other shapes. On processor cores the two agree within its tolerance '
               '(sim/train.py ATOL + RTOL x the magnitude each output is computed from); on '
               'CUDA, cuDNN may run the LSTM in TF32, whose rounding depends on the shape, so a GPU run first needs the '
               'check matched to the collection\'s shapes or TF32 off for the LSTM (torch.backends.cudnn.allow_tf32), '
               'measured before its tolerance is trusted.')
# a run whose model is on CUDA records this note in place of DEVICE_NOTE, which a processor run keeps as before
DEVICE_NOTE_GPU = ('The collection makes one policy call a round over the agents due in it (a batch of varying size), '
                   'while --validate recomputes the stored rows in other shapes, on the collection\'s device with the '
                   'weights of each row\'s version. On processor cores the two agree within its tolerance (sim/train.py '
                   'ATOL + RTOL x the magnitude each output is computed from). On CUDA, cuDNN may run '
                   'the LSTM in TF32, whose rounding depends on the shape: sim.train refuses --validate with the '
                   'collection\'s calls on the GPU and TF32 on (--collect-device cuda --cudnn-tf32 on); with the calls on '
                   'the processor (--collect-device cpu) the check is the processor\'s.')
HANDOVER = ('the collection copy takes the model\'s weights at the start of each collection: after the update, before its '
            'first round (no request goes out during an update); each version\'s rows are checked with the model\'s '
            'weights of that version')


def thread_state():
    """torch's intra-op and inter-op thread counts and, on Linux, the process's CPU affinity, its OS threads and their
    distinct CPU masks: recorded before and after each update, so a run shows whether the threads the update has are the
    threads it had (policy_threads switches torch's count around the rounds)."""
    import torch
    out = dict(torch=torch.get_num_threads(), interop=torch.get_num_interop_threads())
    if hasattr(os, 'sched_getaffinity'):
        out['affinity'] = len(os.sched_getaffinity(0))
    if os.path.isdir('/proc/self/task'):
        masks = {}
        for tid in os.listdir('/proc/self/task'):
            try:
                with open(f'/proc/self/task/{tid}/status') as status:
                    mask = next(line for line in status if line.startswith('Cpus_allowed_list')).split(':', 1)[1].strip()
            except (OSError, StopIteration):
                continue
            masks[mask] = masks.get(mask, 0) + 1
        out.update(os_threads=sum(masks.values()), masks=masks)
    return out


def age_histogram(ages):
    """Rows by age in updates: 0, 1, 2, 3, 4 and more."""
    return {'0': int((ages == 0).sum()), '1': int((ages == 1).sum()), '2': int((ages == 2).sum()),
            '3': int((ages == 3).sum()), '4+': int((ages >= 4).sum())}


def lag_stats(model, versions, current):
    """PPO's approximate KL, mean((r - 1) - log r), and clip fraction, mean(|r - 1| > clip), of the rollout in the buffer
    under the policy's weights now, for each age of row in updates (between the row's version, [n_steps, n_envs], and
    `current`: 0, 1, 2, 3, 4 and more; age 0 apart in segments that also hold older rows, '0_mixed'). Each sequence runs
    from its stored state as the update evaluates it (evaluate_compact over the buffer's sequences in its own order,
    minibatches of batch_size; no draw from numpy's generator, so the update's draws are the same with or without it), so
    the recurrent states of a row after older rows of its segment are recomputed under weights that did not collect those
    rows: before an update only age '0' is the behaviour policy itself. r = exp(log-probability now - the stored
    behaviour log-probability)."""
    import torch
    b, policy = model.rollout_buffer, model.policy
    clip = model.clip_range(model._current_progress_remaining)
    ages = current - versions
    mixed = (ages > 0).any(0, keepdim=True).expand_as(ages)
    kinds = {'0': (ages == 0) & ~mixed, '0_mixed': (ages == 0) & mixed, '1': ages == 1, '2': ages == 2, '3': ages == 3,
             '4+': ages >= 4}
    kinds = {k: v.reshape(-1) for k, v in kinds.items()}
    total, size = b.buffer_size * b.n_envs, model.batch_size
    indices = torch.arange(total, device=b.device)
    rows = b._rows(indices)
    starts = (b.episode_starts.reshape(-1)[rows] != 0) | (indices % b.buffer_size == 0)
    starts[::size] = True
    first = starts.nonzero().squeeze(1)
    on_host = first.cpu().numpy()
    sums = {k: torch.zeros(3, dtype=torch.float64, device=b.device) for k in kinds}
    policy.set_training_mode(False)
    with torch.no_grad():
        for s in range(0, total, size):
            e = min(total, s + size)
            a, z = np.searchsorted(on_host, s), np.searchsorted(on_host, e)
            data = b.compact(rows[s:e], first[a:z] - s, int(np.diff(on_host[a:z], append=e).max()))
            _, log_prob, _ = policy.evaluate_compact(data.observations, data.actions, data.lstm_states, data.keep,
                                                     data.pos, data.n_seq, data.max_length)
            log_ratio = (log_prob - data.old_log_prob).double()
            kl, clipped = (torch.exp(log_ratio) - 1) - log_ratio, ((torch.exp(log_ratio) - 1).abs() > clip).double()
            for name, kind in kinds.items():
                keep = kind[rows[s:e]].double()
                sums[name] += torch.stack([keep.sum(), (kl * keep).sum(), (clipped * keep).sum()])
    out = {}
    for name, (n, kl, clipped) in ((k, v.tolist()) for k, v in sums.items()):
        out[name] = dict(rows=int(n), approx_kl=kl / n if n else None, clip_fraction=clipped / n if n else None)
    return out


class RealEnv(VecEnv):
    """A RealBatch behind SB3's VecEnv for sim/learn.py's DeviceRecurrentPPO: n_envs columns of obs-v6 observations;
    collect_rollouts hands the collection to the batch (asynchronous). Nothing steps it synchronously."""
    render_mode = None
    asynchronous, last_obs = True, True

    def __init__(self, batch, device='cpu'):
        from baseline.adapter import observation_space
        self.env, self.batch, self.device = batch, batch, device
        self.observation, self.shapes = batch.observation, batch.shapes
        self.item_contract = batch.item_contract  # the policy records it (sim/learn.py build)
        self.settings_in_effect = batch.settings_in_effect
        self._zeros = {k: np.zeros((batch.n_envs, *v), dtype(k)) for k, v in self.shapes.items()}
        super().__init__(batch.n_envs, observation_space({k: v[0] for k, v in self._zeros.items()}),
                         spaces.Discrete(self.shapes['mask'][0]))  # a row for each mask entry: 96, or act-v4's 128

    def collect(self, model, callback, buffer, n_rollout_steps):
        return self.batch.collect(model, callback, buffer, n_rollout_steps)

    def reset(self):
        return self._zeros  # SB3's _setup_learn; the collector keeps each agent's own observations

    def step_async(self, actions):
        raise NotImplementedError('the real batch collects asynchronously (RealBatch.collect)')

    def step_wait(self):
        raise NotImplementedError('the real batch collects asynchronously (RealBatch.collect)')

    def get_attr(self, attr_name, indices=None):
        if attr_name != 'render_mode':
            raise AttributeError(attr_name)
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError('the real batch is fixed when it is made')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('the real batch has no per-world methods')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.batch.close()


SETTINGS = ('starts', 'life', 'worlds', 'agents', 'sites', 'radius', 'workers', 'policy_threads')  # --batch-setting names


class LegRows:
    """An evidence file (baseline/world.py JsonlEvidence) whose rows carry the leg's number from leg 2 on; leg 1's rows
    are written as before."""

    def __init__(self, evidence, leg):
        self.evidence, self.leg = evidence, leg

    def __call__(self, kind, **data):
        self.evidence(kind, **data, **({} if self.leg == 1 else dict(leg=self.leg)))

    def close(self):
        self.evidence.close()


def make_env(n_envs, key, output, device, settings, environ=None, ports=None, leg=1, death_price=0.0,
             collect_device=None, all_skills=False, kit_extra=(), inner=None, items=ITEM_V2, start_hp=False,
             start_rule=None):
    """sim/train.py make_env's real engine: the stack's worlds (baseline/world.py worlds) after check_stack, the batch's
    settings (starts and life with no default; worlds, agents a world, sites and radius REAL's and the world's unless given;
    the agents, worlds x agents, apart from the learner's n_envs), the run's leg, its death price, the device of the
    collection's policy calls and its task's paid stats, extra kit, inner pay, item contract, drawn start hitpoints and
    start rule (RealBatch)."""
    settings = dict(settings or {})
    missing, unknown = sorted({'starts', 'life'} - set(settings)), sorted(set(settings) - set(SETTINGS))
    if missing or unknown:
        raise ValueError(f'the real engine\'s batch names starts and life and takes {SETTINGS} (--batch-setting): '
                         f'missing {missing}, unknown {unknown}')
    worlds, agents = settings.get('worlds', REAL['worlds']), settings.get('agents', REAL['agents'])
    sites = settings['sites'] if 'sites' in settings else list(watch.world_sites(settings['starts']))
    radius, workers = settings.get('radius', REAL['radius']), settings.get('workers', 0)
    check_settings(starts=settings['starts'], life=settings['life'], worlds=worlds, agents=agents, sites=sites, radius=radius)
    facts = check_stack(os.environ if environ is None else environ, agents)
    if ports is None:
        from baseline.world import worlds as stack_worlds
        ports = [w['env_port'] for w in stack_worlds()]
    if len(ports) != worlds:
        raise ValueError(f'{worlds} worlds named, the stack booted {len(ports)} (scripts/stack.py --worlds {worlds})')
    return RealEnv(RealBatch(ports, agents=agents, starts=settings['starts'], life=settings['life'], key=key,
                             output=output, n_envs=n_envs, sites=sites, radius=radius, facts=facts, workers=workers,
                             policy_threads=settings.get('policy_threads'), leg=leg, death_price=death_price,
                             collect_device=collect_device, all_skills=all_skills, kit_extra=kit_extra, inner=inner,
                             items=items, start_hp=start_hp, start_rule=start_rule),
                   device)


# ---- a demo (module docstring) ---------------------------------------------------------------------------------------
TUTORIAL_ISLAND = (3094, 3107)  # where a new login stands (env-ts/core/spectator.ts TUTORIAL_ISLAND)


def island_gap(x, z):
    """The 64-tile chunks between (x, z) and a Tutorial Island login, the larger of the two axes."""
    return max(abs((x >> 6) - (TUTORIAL_ISLAND[0] >> 6)), abs((z >> 6) - (TUTORIAL_ISLAND[1] >> 6)))


def wakes_island_npcs(x, z):
    """A player at (x, z) wakes npcs that a login on Tutorial Island wakes: their chunks are within 2, since a player
    wakes its chunk and the chunks beside it (RL_NPC_RADIUS 64): the four Rimmington sites. The one-agent demo
    (baseline.evaluate --demo), whose spectator waits on the island, never plays them; a shared
    demo's spectator never stands there at an npc turn (demo), and a view at such a tile is recorded
    (env-ts/core/spectator.ts wakesIslandNpcs, the same rule)."""
    return island_gap(x, z) <= 2


def in_island_chunks(x, z):
    """(x, z) is in the chunks a Tutorial Island login wakes (47-49 on both axes), where the world refuses a
    view to stand (env-ts/core/spectator.ts inIslandChunks, the same rule)."""
    return island_gap(x, z) <= 1


def spectator_audit(status):
    """The spectator's counts in a demo world's /status (env-ts/core/spectator.ts status: moves, npcs only it saw by
    tick, refused logins, dropped packets, collision restores, views, the click marker's), without where it is."""
    s = dict((status.get('demo') or {}).get('spectator') or {})
    return {k: v for k, v in s.items() if k not in ('present', 'view', 'declared')}


def demo_checks(output, environ):
    """A demo's refusals before anything loads or joins: a new folder inside logs/demo/ (never a run's, an evaluation's or
    the results' folder), and a demo world of scripts/stack.py (RL_DEMO_SPECTATOR and RL_DEMO_PACE_MS) of one world.
    Returns (the spectator's name, the pace in ms a tick, the world's {env_port, status_url})."""
    output, root = Path(output).resolve(), (ROOT / DEMO['root']).resolve()
    if root not in output.parents:
        raise ValueError(f'a demo writes inside {root}, never elsewhere: not {output}')
    if output.exists():
        raise ValueError(f'{output} exists: a demo writes a new folder')
    spectator, pace = environ.get('RL_DEMO_SPECTATOR'), environ.get('RL_DEMO_PACE_MS')
    if not spectator or not pace:
        raise ValueError('sim.real demo runs inside scripts/stack.py --demo-spectator NAME --demo-pace-ms MS')
    worlds = json.loads(environ.get('RL_WORLDS') or '[]')
    if len(worlds) != 1 or environ.get('RL_WORLD_SERVICE'):
        raise ValueError(f'a demo is one world of its own (scripts/stack.py --worlds 1), not {len(worlds)} or a service')
    return spectator, int(pace), worlds[0]


def set_view(status_url, view):
    """POST /demo-view on the world's local telemetry port (env-ts/demo-services.ts): the spectator's view from now on,
    {tile: [x, z]}, {follow: slot} or both (env-ts/core/spectator.ts parseView). Returns the world's reply; a refused view
    raises ValueError with the world's reason."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    request = Request(status_url.rsplit('/', 1)[0] + '/demo-view', data=json.dumps(view).encode(), method='POST',
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(f'the world refused the view {view}: {error.read().decode(errors="replace")}') from None


def view_sample(status, spectator, sites, reach=DEMO['view_tiles']):
    """One look at a demo world (its /status): the agents in it, where the spectator stands, the agents it can see (the
    client draws players within `reach` tiles, the larger of |dx| and |dz|, on its level), the agents within `reach`
    of each site's tile, the view, whether the spectator wakes npcs a Tutorial Island login wakes (wakes_island_npcs) and
    its counts (spectator_audit)."""
    players = [p for p in status.get('playerStates') or [] if 'x' in p]
    me = next((p for p in players if p['name'] == spectator), None)
    agents = [p for p in players if p['name'] != spectator]
    near = lambda x, z, level: sum(max(abs(p['x'] - x), abs(p['z'] - z)) <= reach and p['level'] == level for p in agents)
    return dict(tick=status.get('tick'), agents=len(agents), spectator=me and [me['x'], me['z'], me['level']],
                in_view=me and near(me['x'], me['z'], me['level']),
                near_site={name: near(*watch.SITES[name], 0) for name in sites},
                view=((status.get('demo') or {}).get('spectator') or {}).get('view'),
                wakes_island_npcs=me and wakes_island_npcs(me['x'], me['z']), audit=spectator_audit(status))


def demo(output, *, checkpoint, launch, starts, sites, key, seed, seconds, agents=DEMO['agents'], radius=DEMO['radius'],
         life=None, tile=None, follow=None, wait=DEMO['wait'], start_levels=None, trace=False, environ=None, say=print):
    """A demo (module docstring) inside scripts/stack.py's demo world: demo_checks; the spectator's first view (`tile`,
    else the first site's tile, and the agent of slot `follow` when given), set before the spectator logs in so that it
    never waits on Tutorial Island; one checkpoint of a real-engine run under its own checks (sim/watch.py load_actor:
    a free-play checkpoint, its run's launch.json in the world `starts`); the stack's facts (check_stack); then, once the
    spectator is in the world, `agents` agents in the one world from `sites` within `radius` tiles, sampled, for `seconds`
    of wall time, every agent leaving at the end through the batch's checks. `start_levels` (the text of the watch's
    --start-levels: sim/watch.py parse_levels) starts every life of every agent at that profile (profiled), armed under
    old-life-v1's checks. A Rimmington site is played: the first view is set before the spectator logs in, and the engine
    turns its npcs before it takes a login (World.cycle: processNpcs, then processLogins and processInfo, where the
    spectator moves to its view), so no npc turn runs with the spectator on Tutorial Island; a view whose tile wakes a
    chunk a login there would wake is recorded (wakes_island_npcs). Writes demo.json (the first view's reply and
    the spectator's counts at the end; a demo that fails after it is written says so, with its cause), lives.jsonl,
    failed-resets.jsonl, views.jsonl (a look at the world every DEMO['view_every'] s with the spectator's counts:
    view_sample), xp.jsonl (each rise of a stat's XP by agent and world tick: rose) and, with `trace`,
    observations.jsonl.gz (every result's raw observation with its agent, life, world tick and the action that led to it,
    for demo_reads), each labelled a demo, and no evaluation row. Each view change is in the engine's log with the counts
    at that tick. Returns demo.json's record."""
    from urllib.request import urlopen
    from baseline.world import JsonlEvidence, wait_for_spectator
    environ = os.environ if environ is None else environ
    spectator, pace, world = demo_checks(output, environ)
    output = Path(output)
    if not sites or len(set(sites)) != len(sites) or not set(sites) <= set(watch.SITES):
        raise ValueError(f'distinct sites of sim/watch.py SITES: {sites!r}')
    levels = None if start_levels is None else watch.parse_levels(start_levels)
    if follow is not None and (type(follow) is not int or not 0 <= follow < agents):
        raise ValueError(f'the agent to follow is a slot in 0..{agents - 1}: {follow!r}')
    first = dict(tile=list(tile) if tile is not None else list(watch.SITES[sites[0]]),
                 **({} if follow is None else dict(follow=follow)))
    first_reply = set_view(world['status_url'], first)
    models, facts = watch.load_actor('policy', None, None, None, starts, real_checkpoint=checkpoint, launch=launch)
    life = life if life is not None else (facts['run'].get('settings') or {}).get('life')
    # the version's own batch settings, as the checkpoint saved them (sim/watch.py VERSION_SAVES): a free-play-v3 to v10
    # policy plays with its kit_extra (the tinderbox) and every stat paid, a free-play-v2 one with its death price; a
    # demo pays no inner pay (none of its records is evidence)
    run = facts['run']
    version = dict(death_price=float(run.get('death_price') or 0.0), all_skills=run.get('task') in watch.EVERY_STAT,
                   kit_extra=tuple(tuple(x) for x in run.get('kit_extra') or ()), items=watch.run_items(run.get('task')),
                   # free-play-v12, v14 and v15: each life at its record's drawn hitpoints, as the checkpoint trained
                   **({'start_hp': True} if FREE_PLAY_VERSIONS.get(run.get('task'), {}).get('start_hp') == 'drawn' else {}))
    stack = check_stack(environ, agents, demo=True)
    output.mkdir(parents=True)
    record = dict(label=DEMO['label'], status='waiting', spectator=spectator, pace_ms=pace,
                  client=f"http://127.0.0.1:{environ.get('WEB_PORT', '8888')}/", first_view=first, first_view_reply=first_reply,
                  settings=dict(starts=starts, sites=list(sites), radius=radius, agents=agents, life=life, key=key,
                                seed=seed, seconds=seconds, decoding='sampled', task=run.get('task') or FREE_PLAY['version'],
                                kit_extra=[list(x) for x in version['kit_extra']],
                                **({'start_hp': 'drawn'} if version.get('start_hp') else {}),
                                **({} if levels is None else dict(start_levels=levels)),
                                **(dict(trace='observations.jsonl.gz') if trace else {})),
                  policy=facts, stack=stack)
    write = lambda: (output / 'demo.json').write_text(json.dumps(record, indent=1, default=str) + '\n', encoding='utf-8')
    write()
    batch = views = rises = observed = tick0 = None
    xp_seen = {}  # agent -> (its life, its XP in tenths by stat at its last result)
    started, next_look = time.perf_counter(), 0.0

    def status():
        with urlopen(world['status_url'], timeout=5) as response:
            return json.load(response)

    def look():
        nonlocal next_look
        if time.perf_counter() < next_look:
            return
        next_look = time.perf_counter() + DEMO['view_every']
        wall = round(time.perf_counter() - started, 2)
        try:
            views('view', demo=DEMO['label'], wall=wall, **view_sample(status(), spectator, sites))
        except (OSError, ValueError) as error:  # a look that fails is recorded; the demo goes on
            views('view_error', demo=DEMO['label'], wall=wall, error=f'{type(error).__name__}: {error}')

    def rose():
        """Each rise of a stat's XP since an agent's last result, at the world tick of that result (xp.jsonl: when a
        clip shows a bone buried or a fire lit, read with the looks' tiles)."""
        for agent in batch.agents:
            xp = getattr(agent, 'xp', None) if agent.armed else None
            if xp is None:
                continue
            before = xp_seen.get(agent.j)
            if before is not None and before[0] == agent.life:
                for stat in batch.skills:
                    if xp[stat] > before[1][stat]:
                        rises('xp', demo=DEMO['label'], tick=agent.tick, agent=agent.j, slot=agent.slot, life=agent.life,
                              stat=stat, tenths=xp[stat] - before[1][stat])
            xp_seen[agent.j] = (agent.life, dict(xp))

    def traced(results):
        """Each result's raw observation before its agent moves on, with the life it belongs to: a reset's arms the
        agent's new life (its record's index and drawn hitpoints beside it), a step's ends the decision of the agent's
        action (observations.jsonl.gz, read by demo_reads)."""
        for agent, result in results:
            kind = result.get('kind')
            if kind in ('left', 'error'):
                continue
            arm = kind == 'reset'
            observed('observed', demo=DEMO['label'], agent=agent.j, life=agent.life, result=kind, tick=result.get('tick'),
                     ticks=result.get('ticks'), action=None if arm else agent.action,
                     **(dict(record=agent.record['index'], hp_drawn=agent.record['hp_drawn']) if arm else {}),
                     obs=result.get('obs'))

    try:
        record['waited_seconds'] = round(wait_for_spectator(world['status_url'], spectator, wait, say), 1)
        import torch
        torch.manual_seed(seed)
        started = time.perf_counter()
        batch = RealBatch([world['env_port']], agents=agents, starts=starts, life=life, key=key, output=output,
                          sites=sites, radius=radius, facts=stack, label=DEMO['label'], profile=levels, **version)
        views = JsonlEvidence(output / 'views.jsonl')
        rises = JsonlEvidence(output / 'xp.jsonl')
        observed = JsonlEvidence(output / 'observations.jsonl.gz') if trace else None
        record.update(status='playing', settings_in_effect=batch.settings_in_effect)
        write()
        tick0 = batch.world_tick[0]
        say(f"[demo] {agents} agents join at {', '.join(sites)}; the world runs {seconds:.0f} s", flush=True)
        batch.bind_play(models['policy'])
        record.update(batch.play(seed, 'policy', seconds=seconds, on_round=lambda: (rose(), look()),
                                 on_results=traced if trace else None), status='complete')
    except RealStop as stop:
        record.update(batch.counts, status='stopped', cause=str(stop), seconds=time.perf_counter() - started)
    except BaseException as error:  # the spectator's wait, a world's hello, an interrupt, any other error
        record.update(batch.counts if batch is not None else {}, status='failed', cause=f'{type(error).__name__}: {error}',
                      seconds=time.perf_counter() - started)
        raise
    finally:
        if batch is not None:
            batch.close()
        for evidence in (views, rises, observed):
            if evidence is not None:
                evidence.close()
        try:  # the spectator's counts at the end; the world is still up until this command ends
            record['spectator_audit'] = spectator_audit(status())
        except (OSError, ValueError) as error:
            record['spectator_audit'] = dict(error=f'{type(error).__name__}: {error}')
        if tick0 is not None:  # the world's ticks from the first join to the last leave, over that wall time
            ticks, wall = batch.world_tick[0] - tick0, time.perf_counter() - started
            record.update(world_ticks=ticks, wall_seconds=wall, ticks_per_second=round(ticks / wall, 3))
        write()
    return record


def demo_reads(output):
    """The watch's monitors (sim/watch.py life_reads) of each life a demo traced (demo with `trace`), from its
    observations.jsonl.gz and lives.jsonl: {"agent:life": reads}, written to reads.json with the demo's label. Each life
    is given to life_reads as a watch life's rows: the reset, the arm's observation and tick, then each decision's action
    and observation, and its end: a horizon at the life's length, a death at the engine's death tick, a life the demo's end
    cut ('left') at its last result. A demo records no capture and no executed action, so production, kills, damage and
    bank deposits and withdrawals are not read (None) and a fight is 'fight' with no npc. A life the trace does not hold
    from its arm, or that no life row closed, is not read. Never evidence: a demo's reads describe a demo."""
    from baseline.world import read_jsonl
    output = Path(output)
    lives, steps = {}, {}
    for row in read_jsonl(output / 'observations.jsonl.gz'):
        steps.setdefault((row['agent'], row['life']), []).append(row)
    for row in read_jsonl(output / 'lives.jsonl'):
        if row.get('kind') == 'life':
            lives[row['agent'], row['life']] = row
    out = {}
    for (agent, life), end in sorted(lives.items()):
        rows = steps.get((agent, life)) or []
        if not rows or rows[0]['result'] != 'reset' or rows[0]['tick'] != end['arm_tick']:
            continue
        arm, ticks = rows[0], end['ticks']
        if end['terminal'] == 'horizon':
            ticks = end['length']
        elif end['death_tick'] is not None:
            ticks = end['death_tick'] - end['arm_tick']
        given = [dict(kind='watch_life', life=life, record=dict(index=arm['record'], hp_drawn=arm['hp_drawn']), seed=None,
                      horizon=end['length']),
                 dict(kind='reset', observation=dict(tick=arm['tick'] - arm['ticks'])),
                 dict(kind='episode_start', observation=arm['obs']),
                 dict(kind='resource_control', operation='arm', response=dict(audit=dict(startTick=arm['tick'])))]
        for step in rows[1:]:
            given += [dict(kind='requested', action=step['action']), dict(kind='observed', observation=step['obs'])]
        given.append(dict(kind='watch_life_end', ticks=ticks, reason=end['terminal'], status='complete'))
        reads = watch.life_reads(given)
        reads['bank'].update(deposits=None, withdrawals=None)
        out[f'{agent}:{life}'] = dict(reads, site=end['site'], terminal=end['terminal'], decisions_traced=len(rows) - 1)
    (output / 'reads.json').write_text(json.dumps(dict(demo=DEMO['label'], reads='sim/watch.py life_reads', lives=out),
                                                  indent=1, default=str) + '\n', encoding='utf-8')
    return out


def busiest(status, spectator, reach=DEMO['view_tiles']):
    """The tile (level 0) whose view, `reach` tiles the larger of |dx| and |dz|, holds the most agents of a demo world
    now (its /status), on a 2-tile grid over the agents' extent, with that count; among such tiles the one nearest the
    agents it sees (the least sum of their distances), so the group stands in the middle of the view, then the lowest x
    and z; never a tile the world refuses as a view (in_island_chunks). (None, 0) when no agent is on level 0."""
    agents = [(p['x'], p['z']) for p in status.get('playerStates') or [] if 'x' in p and p['name'] != spectator
              and p['level'] == 0]
    xs, zs = [x for x, _ in agents], [z for _, z in agents]

    def score(x, z):
        d = [max(abs(ax - x), abs(az - z)) for ax, az in agents]
        seen = [v for v in d if v <= reach]
        return len(seen), -sum(seen), -x, -z
    tiles = [(x, z) for x in range(min(xs, default=0), max(xs, default=-1) + 1, 2)
             for z in range(min(zs, default=0), max(zs, default=-1) + 1, 2) if not in_island_chunks(x, z)]
    if not tiles:
        return None, 0
    best = max(score(x, z) for x, z in tiles)
    return (-best[2], -best[3]), best[0]


def view(status_url, *, tile=None, follow=None, crowd=False, every=None, say=print):
    """Manual control of a running demo's spectator (sim.real view): stand at `tile`, follow the agent of slot
    `follow`, or, `crowd`, stand where the view holds the most agents now (busiest), then again every `every` seconds,
    moving only when that view holds more agents than the spectator's own view, until the world stops."""
    from urllib.error import URLError
    from urllib.request import urlopen
    if sum((tile is not None, follow is not None, crowd)) != 1 or (every is not None and (not crowd or every <= 0)):
        raise ValueError('a view is one of a tile, an agent to follow or the busiest view (with --every seconds)')
    if not crowd:
        return set_view(status_url, dict(tile=list(tile)) if tile is not None else dict(follow=follow))
    while True:
        try:
            with urlopen(status_url, timeout=5) as response:
                status = json.load(response)
        except (URLError, OSError):
            say('[view] the world has stopped', flush=True)
            return None
        name = ((status.get('demo') or {}).get('spectator') or {}).get('declared')
        (at, count), now = busiest(status, name), view_sample(status, name, ())
        if at is not None and (now['in_view'] is None or count > now['in_view']):
            reply = set_view(status_url, dict(tile=list(at)))
            say(f"[view] {count} agents in view at {at[0]},{at[1]} (tick {reply['tick']})", flush=True)
        if every is None:
            return at, count
        time.sleep(every)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    play = commands.add_parser('play', help='random legal play through the batch, inside scripts/stack.py: no learner')
    play.add_argument('--output', type=Path, required=True, help='a new folder: play.json, lives.jsonl')
    play.add_argument('--starts', required=True, help='the world whose start group the records draw from')
    play.add_argument('--life', type=json.loads, required=True, help='the life-length distribution, JSON')
    play.add_argument('--agents', type=int, default=REAL['agents'], help='agents a world (RL_SHARED_AGENTS or fewer)')
    play.add_argument('--sites', nargs='+', help='sites of the world (default: all of them, sim/watch.py world_sites)')
    play.add_argument('--radius', type=int, default=REAL['radius'])
    play.add_argument('--key', type=int, required=True)
    play.add_argument('--seed', type=int, required=True)
    length = play.add_mutually_exclusive_group(required=True)
    length.add_argument('--seconds', type=float, help='wall seconds of play')
    length.add_argument('--rounds', type=int, help='rounds of play (a transcript\'s replay plays as many)')
    play.add_argument('--actor', choices=('random', 'wait'), default='random')
    play.add_argument('--no-ledger', action='store_true', help='do not compare the engine\'s XP ledger (no pay check)')
    play.add_argument('--transcript', type=Path, help='record every request and reply here (gzip JSONL; no workers)')
    play.add_argument('--workers', type=int, default=0, help='processes that decode and project the replies (0: none)')
    play.add_argument('--until-death', action='store_true', help='stop once a death and a cut have passed their ledger '
                      'check (a transcript for the contract test)')
    show = commands.add_parser('demo', help='a paced demo of one real-engine checkpoint, N agents in one demo world '
                               '(scripts/stack.py --demo-spectator): never evidence')
    show.add_argument('--output', type=Path, required=True, help='a new folder inside logs/demo/')
    show.add_argument('--real-checkpoint', type=Path, required=True, help='one checkpoint of a real-engine run')
    show.add_argument('--launch', type=Path, required=True, help='that run\'s launch.json')
    show.add_argument('--starts', required=True, help='the world whose start group the records draw from (the run\'s)')
    show.add_argument('--sites', nargs='+', required=True, help='one or two names of sim/watch.py SITES')
    show.add_argument('--agents', type=int, default=DEMO['agents'], help='agents in the world (RL_SHARED_AGENTS or fewer)')
    show.add_argument('--radius', type=int, default=DEMO['radius'], help='start tiles within this many tiles of a site')
    show.add_argument('--life', type=json.loads, help='the life-length distribution, JSON (default: the run\'s)')
    show.add_argument('--key', type=int, required=True)
    show.add_argument('--seed', type=int, required=True)
    show.add_argument('--seconds', type=float, required=True, help='wall seconds of play once the agents join')
    show.add_argument('--view-tile', type=int, nargs=2, metavar=('X', 'Z'), help='where the spectator stands first '
                      '(default: the first site\'s tile)')
    show.add_argument('--follow', type=int, metavar='SLOT', help='follow this agent from the start')
    show.add_argument('--wait-seconds', type=float, default=DEMO['wait'], help='how long to wait for the spectator')
    show.add_argument('--start-levels', help='every life at this profile, as sim.watch --start-levels '
                      '(name=level,...; every other draw as drawn)')
    show.add_argument('--trace', action='store_true', help='write every observation (observations.jsonl.gz) for '
                      'demo-reads')
    reads = commands.add_parser('demo-reads', help='the watch\'s monitors of each life a demo traced (reads.json)')
    reads.add_argument('output', type=Path, help='the demo\'s folder (its observations.jsonl.gz and lives.jsonl)')
    look = commands.add_parser('view', help='move a running demo\'s spectator: a tile, an agent or the busiest view')
    choice = look.add_mutually_exclusive_group(required=True)
    choice.add_argument('--tile', type=int, nargs=2, metavar=('X', 'Z'), help='stand at this tile (level 0)')
    choice.add_argument('--follow', type=int, metavar='SLOT', help='follow this agent (0 .. agents - 1)')
    choice.add_argument('--busiest', action='store_true', help='stand where the view holds the most agents now')
    look.add_argument('--every', type=float, metavar='S', help='with --busiest: look again every S seconds until the '
                      'world stops, moving only to a view with more agents')
    look.add_argument('--status', default='http://127.0.0.1:43610/status', help='the demo world\'s status address')
    args = parser.parse_args()
    if args.command == 'demo-reads':
        for name, r in demo_reads(args.output).items():
            print(f"[demo-reads] {name}: {r['ticks']} ticks, {r['terminal']}, gain {r['gain']}, time {r['time']}")
        raise SystemExit(0)
    if args.command == 'view':
        try:
            view(args.status, tile=args.tile, follow=args.follow, crowd=args.busiest, every=args.every)
        except (ValueError, OSError) as error:
            raise SystemExit(f'[view] {error}')
        except KeyboardInterrupt:
            pass
        raise SystemExit(0)
    if args.command == 'demo':
        try:
            record = demo(args.output, checkpoint=args.real_checkpoint, launch=args.launch, starts=args.starts,
                          sites=args.sites, key=args.key, seed=args.seed, seconds=args.seconds, agents=args.agents,
                          radius=args.radius, life=args.life, tile=args.view_tile, follow=args.follow,
                          wait=args.wait_seconds, start_levels=args.start_levels, trace=args.trace)
        except (ValueError, RuntimeError, TimeoutError) as error:
            raise SystemExit(f'[demo] {error}')
        print(f"[demo] {record['status']}: {record.get('decisions')} decisions, {record.get('lives')} lives ended "
              f"({record.get('deaths')} deaths), {record.get('world_ticks')} world ticks in {record.get('seconds', 0):.0f} s "
              f"({record.get('ticks_per_second')} a second)" + (f"; {record['cause']}" if record.get('cause') else ''),
              flush=True)
        raise SystemExit(0 if record['status'] == 'complete' else 1)
    if args.output.exists():
        raise SystemExit(f'[real] {args.output} exists: a run writes a new folder')
    from baseline.world import worlds as stack_worlds
    try:
        facts = check_stack(os.environ, args.agents)
        ports = [w['env_port'] for w in stack_worlds()]
        batch = RealBatch(ports, agents=args.agents, starts=args.starts, life=args.life, key=args.key, output=args.output,
                          sites=args.sites, radius=args.radius, ledger=not args.no_ledger, facts=facts,
                          transcript=args.transcript, workers=args.workers)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f'[real] {error}')
    result = dict(actor=args.actor, seed=args.seed, settings=batch.settings_in_effect, status='running')
    started = time.perf_counter()
    try:
        result.update(batch.play(args.seed, args.actor, seconds=args.seconds, rounds=args.rounds,
                                 until_death=args.until_death), status='complete')
    except RealStop as stop:
        result.update(batch.counts, status='stopped', cause=str(stop), seconds=time.perf_counter() - started)
    finally:
        batch.close()
    result['decisions_per_second'] = round(result['decisions'] / result['seconds'], 2) if result.get('seconds') else None
    (args.output / 'play.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    final = result.get('final', result)
    print(f"[real] {result['status']}: {result['decisions']} decisions in {result['seconds']:.1f} s "
          f"({result['decisions_per_second']}/s) and {result['rounds']} rounds, {result['lives']} lives ended "
          f"({result['deaths']} deaths, {result['cut']} cut); at the end {final['left']} live lives left; "
          f"{final['ledger_checks']} life checks over {final['ledger_grants']} grants, {final['running_checks']} "
          f"running checks" + (f"; {result['cause']}" if result.get('cause') else ''), flush=True)
    raise SystemExit(0 if result['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
