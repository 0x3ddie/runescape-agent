"""sim/real.py, free-play-v1 in the real engine. A fake engine speaks shared-lockstep-v1 (env-ts/engine/batch.ts) over
TCP and keeps its rules, not the batch's: ready, hello with the stat names, places around a centre, slots that join at
their first reset, the setup inside the reset, the stop rule (a request carries items for exactly the due slots, else it
is refused; the reply holds every agent whose item ends at the first such tick), the XP of every stat read from the
player, life facts, and at the reset that closes a life that life's ledger of XP grants. Its agents are
sim/tests/test_watch.py's fake native agent (the declared start, items, levels), with XP that every decision gains, a
death at a chosen decision and, for the pay check, results that read more woodcutting XP than was granted. The batch
plays it as it plays scripts/stack.py's worlds: lives drawn as the watch draws them, pay from the player's tenths, a death
and a cut end only that agent's life, a cut life bootstraps from its last observation, a rollout holds only lives whose
ledger check passed, the pool does not depend on the order the worlds reply in, a ledger difference stops the run before
the next update, and sim.train trains on it from a checkpoint of its own contract and refuses one of another."""
import json
import math
import re
import socket
import threading
import time
import types

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from baseline.policy import POPART
from baseline.tests.test_resource_task import raw as plain_raw
from obs import SKILL_ORDER, project_v6
from sim import real as R
from sim.agent import A_HELD0, A_STYLE0
from wide_contract import page_entry, page_row
from sim.audit import fractional_level
from sim.learn import MODEL_DTYPES, build
from sim.spec import TERMINAL_DEATH, TERMINAL_HORIZON, TERMINAL_RUNNING
from sim.tests.test_watch import NAMES, Fake, level_of
from sim.workers import (FREE_PLAY_TASK, FREE_PLAY_TASKS, FREE_PLAY_V2, FREE_PLAY_V3, FREE_PLAY_V10, FREE_PLAY_V11,
                         FREE_PLAY_V12, FREE_PLAY_V14, FREE_PLAY_V15, OLD_LIFE)

V2, WORLD_V3 = 'village-world-v2', 'village-world-v3'
GAIN = dict(Woodcutting=253, Attack=31, Hitpoints=7)  # tenths a decision gains: never whole XP


class Body(Fake):
    """One agent's player in a fake world: Fake's declared start, items, levels and held options; each decision grants
    GAIN times one more than the life's index (so its lives differ), a death at decision `death` of its first life, and
    from decision `corrupt` of its first life on a pay that reads ten tenths of woodcutting more than was granted.
    hits[(life, decision)]: hitpoints lost (or, below 0, gained) at that decision, within 0 and the base. report_food (its
    world's `food`): its facts hold the pack's food and the life's drops of food, as the engine's facts.food and
    facts.food_drops do."""

    def __init__(self, ticks, death=None, corrupt=None, gain=GAIN, hits=None):
        super().__init__(step_ticks=ticks)
        self.death, self.corrupt, self.grants, self.decisions, self.extra, self.eats = death, corrupt, [], 0, 0, 0
        self.gain = gain  # tenths a decision grants by stat (GAIN, or a test's with a stat outside the seven)
        self.hits, self.report_food, self.food_drops = dict(hits or {}), False, 0
        self.report_hurt, self.hurt = False, {}  # facts.hurt: {base: missing hitpoint ticks} of the armed life

    def reset(self, seed, cfg):
        obs = super().reset(seed, cfg)
        self.tenths = {n: 10 * xp for n, xp in self.xp.items()}
        self.grants, self.decisions, self.extra, self.eats, self.food_drops = [], 0, 0, 0, 0
        self.hurt = {}
        return obs

    def food(self):
        return [k for k, i in enumerate(self.items) if i and i['name'] in ('Shrimps', 'Bread')]

    def step(self, action, setup=False):
        if not setup:
            self.decisions += 1
            for name, tenths in self.gain.items():
                self.tenths[name] += tenths * (1 + self.life)
                # stamped as the engine stamps a grant: the tick of the cycle that makes it, here the decision's first
                self.grants.append([SKILL_ORDER.index(name), tenths * (1 + self.life), self.tick])
            self.xp = {n: t // 10 for n, t in self.tenths.items()}
            if (self.life, self.decisions) == (0, self.death):
                self.dead, self.hp = True, 0
            if (self.life, self.decisions) == (0, self.corrupt):
                self.extra = 10
            top = level_of(self.tenths['Hitpoints'] // 10)
            if not self.dead and (self.life, self.decisions) in self.hits:
                self.hp = max(0, min(top, self.hp - self.hits[self.life, self.decisions]))
        held, entry = self.held(), page_entry(action)
        if entry is not None and entry[0] == A_HELD0 and entry[1] < len(held) and held[entry[1]]['option'] == 'Eat':
            self.eats += 1
        obs = super().step(action)
        if not setup and not self.dead:  # each of the step's ticks at the hitpoints after it (the engine counts tick by tick)
            base = level_of(self.tenths['Hitpoints'] // 10)
            self.hurt[base] = self.hurt.get(base, 0) + self.step_ticks * (base - min(self.hp, base))
        return obs

    def pay(self):
        running = [0] * len(SKILL_ORDER)
        for stat, xp10, _ in self.grants:
            running[stat] += xp10
        return dict(xp=[self.tenths[n] + (self.extra if n == 'Woodcutting' else 0) for n in SKILL_ORDER],
                    base=[level_of(self.tenths[n] // 10) for n in SKILL_ORDER], ledger=running)

    def facts(self):
        return dict(hp=[self.hp, level_of(self.tenths['Hitpoints'] // 10)], eats=self.eats,
                    weapon=dict(id=self.worn[0]['id'], name='weapon') if self.worn else None,
                    death=dict(tick=self.tick, killers=['Goblin']) if self.dead else None,
                    **(dict(food=len(self.food()), food_drops=self.food_drops) if self.report_food else {}),
                    **(dict(hurt=[[b, n] for b, n in sorted(self.hurt.items())]) if self.report_hurt else {}))


class FakeWorld:
    """One world of shared-lockstep-v1 on a local port: `make(slot)` makes a slot's Body at its first reset. `delay`
    seconds before each batch reply; `log` holds (tick, slot, op, result) of every item; `random_events`, when given, is
    in the hello (today's engine does not report it). food: the facts hold food and food_drops and the hello names them
    (True), or neither (False). hurt: the facts hold the life's hurt count and the hello names it (True), or neither
    (False)."""

    def __init__(self, world, make, delay=0.0, capacity=2046, random_events=None, failures=(), bad_leave=False, broken=(),
                 food=False, hurt=False):
        self.world, self.make, self.delay, self.capacity, self.random_events = world, make, delay, capacity, random_events
        self.food, self.hurt = food, hurt
        self.failures, self.bad_leave, self.resets, self.pings = set(failures), bad_leave, {}, 0
        self.broken, self.steps = set(broken), {}  # (slot, n): the slot's n-th step returns an observation without player
        self.server = socket.create_server(('127.0.0.1', 0))
        self.port, self.bodies, self.busy, self.due, self.lives = self.server.getsockname()[1], {}, {}, set(), {}
        self.tick, self.log = 100, []
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        conn, _ = self.server.accept()
        send = lambda msg: conn.sendall(('PROTO|' + json.dumps(msg) + '\n').encode())
        send(dict(type='ready', protocol='shared-lockstep-v1', capacity=self.capacity, worldId=self.world))
        with conn, conn.makefile('rb') as lines:
            for line in lines:
                reply = self.handle(json.loads(line))
                if reply is None:
                    return
                if reply.get('type') == 'batch':
                    time.sleep(self.delay)
                send(reply)

    def handle(self, request):
        cmd = request['cmd']
        if cmd == 'hello':
            hello = dict(type='hello', protocol='shared-lockstep-v1', capacity=self.capacity, worldId=self.world,
                         tick=self.tick, xpRate=1, kTicks=2, skills=list(SKILL_ORDER), answerLimitMs=request['answerLimitMs'],
                         **(dict(factFields=['hp', 'weapon', 'eats', 'death'] + ['food', 'food_drops'] * self.food
                                 + ['hurt'] * self.hurt) if self.food or self.hurt else {}))
            return hello if self.random_events is None else dict(hello, randomEvents=self.random_events)
        if cmd == 'places':
            (cx, cz), r = request['centres'][0], request['radius']
            ring = [(cx + dx, cz + dz) for dx in range(-r, r + 1) for dz in range(-r, r + 1)]
            return dict(type='places', tiles=[dict(x=x, z=z) for x, z in (ring * request['count'])[:request['count']]])
        if cmd == 'close':
            return None
        if cmd == 'ping':
            self.pings += 1
            return dict(type='pong', tick=self.tick)
        slots = [item['slot'] for item in request['items']]
        if set(slots) & set(self.busy) or self.due - set(slots) or any(
                item['op'] != 'reset' and item['slot'] not in self.bodies for item in request['items']):
            return dict(type='error', code='due-without-item', message=f'items {slots} for due {sorted(self.due)}')
        left = []
        for item in request['items']:
            slot, t0 = item['slot'], self.tick
            self.due.discard(slot)
            if item['op'] == 'leave':  # out at once, with the ledger of the life it ends
                ledger = self.close(slot, t0)
                if ledger and self.bad_leave:
                    ledger['total'][SKILL_ORDER.index('Attack')] += 1
                left.append(dict(slot=slot, kind='left', tick=t0, ticks=0, obs=None, done=True, dead=False,
                                 pay=dict(self.bodies[slot].pay(), ledger=None), facts=self.bodies[slot].facts(),
                                 ledger=ledger))
                self.lives.pop(slot, None)
                continue
            n = self.resets[slot] = self.resets.get(slot, -1) + (item['op'] == 'reset')
            if item['op'] == 'reset' and (slot, n) in self.failures:  # its setup fails: no life follows
                ledger = self.close(slot, t0)
                self.lives.pop(slot, None)
                body = self.bodies[slot] = self.bodies[slot] if slot in self.bodies else self.make(slot)
                body.report_food, body.report_hurt = self.food, self.hurt
                body.tick = t0
                body.reset(item['seed'], item['cfg'])  # the new player, whose setup then fails
                body.tick = t0 + 6
                result = dict(kind='error', code='setup', message='died during setup', obs=None, done=True, dead=False,
                              ledger=ledger, slot=slot, ticks=6, end=body.tick)
                self.busy[slot] = result
                self.log.append((t0, slot, 'reset', result))
                continue
            if item['op'] == 'reset':
                ledger = self.close(slot, t0)
                body = self.bodies[slot] = self.bodies[slot] if slot in self.bodies else self.make(slot)
                body.report_food, body.report_hurt = self.food, self.hurt
                body.tick = t0 + (4 if slot not in self.lives else 0)  # a first join's login
                body.reset(item['seed'], item['cfg'])
                obs = body.step(A_STYLE0, setup=True)
                wield = item['setup']['wield']
                if wield:
                    k = next(k for k, h in enumerate(obs['heldSel']) if h['id'] == NAMES[wield][0] and h['option'] == 'Wield')
                    obs = body.step(page_row(A_HELD0, k), setup=True)
                self.lives[slot] = body.tick
                result = dict(kind='reset', obs=obs, done=False, dead=False, ledger=ledger)
            else:
                body = self.bodies[slot]
                body.tick = t0
                obs = body.step(item['action'])
                self.steps[slot] = self.steps.get(slot, 0) + 1
                if (slot, self.steps[slot]) in self.broken:
                    obs = {k: v for k, v in obs.items() if k != 'player'}
                result = dict(kind='step', obs=obs, done=body.dead, dead=body.dead)
            result.update(slot=slot, ticks=body.tick - t0, end=body.tick)
            self.busy[slot] = result
            self.log.append((t0, slot, item['op'], result))
        if left or not self.busy:
            return dict(type='batch', tick=self.tick, results=left)
        self.tick = min(r['end'] for r in self.busy.values())
        out = []
        for slot in sorted((s for s, r in self.busy.items() if r['end'] == self.tick), reverse=True):  # not in slot order
            result = self.busy.pop(slot)
            del result['end']
            result['tick'] = self.tick
            pay = self.bodies[slot].pay()
            out.append(dict(result, pay=pay if result['kind'] != 'error' else dict(pay, ledger=None),
                            facts=self.bodies[slot].facts()))
            self.due.add(slot)
        return dict(type='batch', tick=self.tick, results=out)

    def close(self, slot, tick):
        """The ledger of the slot's life that this reset closes (None at a first join): the grants since its start."""
        if slot not in self.lives:
            return None
        total = [0] * len(SKILL_ORDER)
        for stat, xp10, _ in self.bodies[slot].grants:
            total[stat] += xp10
        return dict(start=self.lives[slot], end=tick, grants=list(self.bodies[slot].grants), total=total)


def fake(tmp_path, *, worlds=2, agents=3, make=None, life=('fixed', 40), delays=None, key=5, n_envs=None, workers=0,
         world=None, leg=1, death_price=0.0, collect_device=None, all_skills=False, kit_extra=(), starts=V2, inner=None,
         items='item-v2', start_hp=False, start_rule=None):
    """A RealBatch over `worlds` fake worlds of `agents` Bodies (default: agent j's decisions take 2 + j % 3 ticks) with
    `workers` worker processes, on leg `leg`'s draw streams, at a death price of `death_price` (free-play-v2's 50, or
    v1's 0), the collection's calls on a copy on `collect_device` (None: the model's own policy), every stat paid and an
    extra kit with free-play-v3's all_skills and kit_extra, free-play-v15's inner pay with `inner`, free-play-v12's drawn
    start hitpoints with `start_hp`, free-play-v14's start rule with `start_rule`,
    in the world `starts`; world(w): FakeWorld's options for world w."""
    make = make or (lambda j: Body(2 + j % 3))
    fakes = [FakeWorld(w, lambda k, w=w: make(w * agents + k), (delays or [0.0] * worlds)[w], **(world(w) if world else {}))
             for w in range(worlds)]
    batch = R.RealBatch([f.port for f in fakes], agents=agents, starts=starts, life=life, key=key, output=tmp_path,
                        n_envs=n_envs, stall=30.0, radius=2, workers=workers, leg=leg, death_price=death_price,
                        collect_device=collect_device, all_skills=all_skills, kit_extra=kit_extra, inner=inner,
                        items=items, start_hp=start_hp, start_rule=start_rule)
    return batch, fakes


WORKERS = pytest.mark.parametrize('workers', (0, 1, 2))  # over 3 worlds: one worker serves two worlds with 2


def learner(batch, n_steps=8):
    return build(R.RealEnv(batch), seed=3, device='cpu', n_steps=n_steps, batch_size=16, n_epochs=1, free_play=True,
                 half_life_ticks=1000., value_norm=POPART)


def collect(model, rollouts=1):
    _, callback = model._setup_learn(rollouts * model.n_envs * model.n_steps, None)
    for _ in range(rollouts):
        model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
    return model.rollout_buffer


def lives(path):
    return [json.loads(line) for line in (path / 'lives.jsonl').read_text().splitlines()]


def result_with(tenths, tick, ticks):
    obs = plain_raw()
    obs.update(tick=tick)
    return dict(kind='step', tick=tick, ticks=ticks, obs=obs, projected=project_v6(obs), done=False, dead=False,
                pay=dict(xp=[tenths[n.lower()] for n in SKILL_ORDER], base=[1] * len(SKILL_ORDER)),
                facts=dict(hp=[10, 10], weapon=None, eats=0, death=None))


# The first three lives' draws of each of four agents of a village-world-v2 batch of key 5 that names no sites (fake
# engine, radius 2), made before village-world-v3: (agent, life, site, tile, record, reset seed, length).
V2_DRAWS = [[0, 0, 'rimmington_mine_west', [2987, 3233], 0, 408473, 40], [0, 1, 'rimmington_coast', [2978, 3198], 1, 974186, 40],
            [0, 2, 'rimmington_coast', [2979, 3196], 2, 361518, 40], [1, 0, 'draynor_manor_south', [3091, 3326], 0, 210221, 40],
            [1, 1, 'rimmington_mine_west', [2984, 3230], 1, 59107, 40], [1, 2, 'port_sarim_north', [3050, 3272], 2, 481185, 40],
            [2, 0, 'rimmington_coast', [2980, 3199], 0, 860188, 40], [2, 1, 'falador_farm_west', [3004, 3279], 1, 535605, 40],
            [2, 2, 'rimmington_coast', [2980, 3198], 2, 482384, 40], [3, 0, 'rimmington_mine_west', [2983, 3229], 0, 755178, 40],
            [3, 1, 'draynor_manor_south', [3090, 3324], 1, 212005, 40], [3, 2, 'draynor_manor_south', [3090, 3328], 2, 485127, 40]]
SIX = ['port_sarim_north', 'draynor_west', 'falador_farm_west', 'rimmington_coast', 'rimmington_mine_west',
       'draynor_manor_south']
SHORE = ['rimmington_shore_east', 'rimmington_shore_west']


def draws(batch, lives=3):
    """Each agent's first `lives` draws (begin): (agent, life, site, tile, record, reset seed, length)."""
    out = []
    for a in batch.agents:
        for _ in range(lives):
            item = a.begin(batch.key, batch.sites, batch.tiles, batch.life)
            out.append([a.j, a.life, a.site, list(a.tile), a.record['index'], item['seed'], a.length])
    return out


def test_a_village_world_v2_batch_draws_what_it_drew_before_and_refuses_a_shore_site(tmp_path):
    """A village-world-v2 batch that names no sites draws its six, in SITES' old order and at the batch's radius, and
    the same sites, tiles, records, reset seeds and lengths as before the shore sites were added, with no site radius in
    its settings (so its legs' world records are unchanged); naming a shore site is refused before any world is
    joined, and so is a site that no world has."""
    batch, _ = fake(tmp_path, worlds=1, agents=4, key=5)
    settings = batch.settings_in_effect
    assert batch.sites == SIX and list(settings['sites']) == SIX and 'site_radius' not in settings
    assert settings['radius'] == 2 and settings['tiles'] == dict.fromkeys(SIX, 25)  # (2 x 2 + 1)^2 places a site
    assert draws(batch) == V2_DRAWS
    batch.close()
    for world, site in ((V2, 'rimmington_shore_east'), (V2, 'rimmington_shore_west'), (WORLD_V3, 'rimmington_check')):
        with pytest.raises(ValueError, match=f'distinct sites of {world} '):
            R.check_settings(starts=world, life=('fixed', 40), worlds=1, agents=1, sites=['draynor_west', site], radius=16)


def test_a_village_world_v3_batch_draws_eight_sites_and_takes_the_shore_starts_within_their_own_radius(tmp_path):
    """A village-world-v3 batch that names no sites draws from eight, the six and the two shore sites; the shore sites'
    places lie within their own radius of their tiles (8 east, 3 west) and the six's within the batch's (2 here), both
    recorded in its settings (site_radius: the shore sites alone). Its start records are v2's. A v3 batch of the six
    alone records no site radius."""
    from sim import watch
    batch, _ = fake(tmp_path / 'v3', worlds=1, agents=8, key=5, starts=WORLD_V3)
    settings = batch.settings_in_effect
    own = dict(rimmington_shore_east=8, rimmington_shore_west=3)
    assert batch.sites == SIX + SHORE and settings['site_radius'] == own and settings['radius'] == 2
    reach = {site: max(max(abs(x - watch.SITES[site][0]), abs(z - watch.SITES[site][1])) for x, z in tiles)
             for site, tiles in batch.tiles.items()}
    assert reach == {**dict.fromkeys(SIX, 2), **own}
    drawn = draws(batch, lives=20)
    assert {d[2] for d in drawn} == set(SIX + SHORE) and all(
        max(abs(x - watch.SITES[site][0]), abs(z - watch.SITES[site][1])) <= reach[site] for _, _, site, (x, z), *_ in drawn)
    batch.close()
    old, _ = fake(tmp_path / 'v2', worlds=1, agents=8, key=5)
    assert [d[4:6] for d in draws(old, lives=20)] == [d[4:6] for d in drawn]  # the same start records and reset seeds
    old.close()
    alone = R.RealBatch([FakeWorld(0, lambda k: Body(2)).port], agents=1, starts=WORLD_V3, life=('fixed', 40), key=5,
                        output=tmp_path / 'alone', sites=SIX, radius=2)
    assert 'site_radius' not in alone.settings_in_effect and alone.settings_in_effect['starts'] == WORLD_V3
    alone.close()


@settings(max_examples=60, deadline=None)
@given(st.lists(st.dictionaries(st.sampled_from([s.lower() for s in SKILL_ORDER]), st.integers(0, 40000), max_size=4),
                min_size=1, max_size=30), st.sampled_from(range(len(SKILL_ORDER))), st.integers(-1, 1), st.data())
def test_a_lifes_pay_is_its_fractional_level_gain_and_a_tenth_off_the_engines_ledger_fails_its_check(steps, stat, off,
                                                                                                    data):
    """Over any XP grants in tenths (level crossings included), the pay of a life's decisions adds up to its fractional
    level gain over the paid skills from its start and the row's XP is the ledger's; a running ledger one tenth off in any
    stat fails the check at that reply, and a closing ledger one tenth off, or over other ticks, fails it at the life's
    end (a pay read in whole XP would)."""
    skills = [s.lower() for s in SKILL_ORDER]
    agent = R.Agent(0, 0, 0, 1, 'danger', 0)
    agent.begin(1, ['draynor_west'], dict(draynor_west=[(3070, 3277)]), ('fixed', 10 ** 5))
    tenths, tick, levels = dict.fromkeys(skills, 0), 100, dict.fromkeys(skills, 1)
    tenths['hitpoints'] = 11540
    zero = [0] * len(skills)
    agent.arm(dict(result_with(tenths, tick, 10), kind='reset'), dict(tenths), levels, zero, skills)
    start, pays = dict(tenths), []
    wrong_at = data.draw(st.integers(0, len(steps) - 1)) if off else None
    for k, gains in enumerate(steps):
        for s, v in gains.items():
            tenths[s] += v
        tick += 3
        last = k == len(steps) - 1
        result = result_with(tenths, tick, 3)
        if last:
            result.update(done=True, dead=True, facts=dict(result['facts'], death=dict(tick=tick, killers=[])))
        running = [tenths[s] - start[s] for s in skills]
        if k == wrong_at:
            replay = R.Agent(0, 0, 0, 1, 'danger', 0)
            replay.__dict__.update({key: value for key, value in agent.__dict__.items()})
            wrong = [v + off * (n == stat) for n, v in enumerate(running)]
            with pytest.raises(R.RealStop, match='running ledger'):
                replay.stepped(result, dict(tenths), levels, wrong, skills)
        pay, ticks, end = agent.stepped(result, dict(tenths), levels, running, skills)
        assert ticks == 3 and pay >= 0 and end == ('death' if last else None)
        pays.append(pay)
    total = [tenths[s] - start[s] for s in skills]
    ledger = dict(start=agent.arm_tick, end=tick, grants=[], total=total)
    pending = dict(agent.pending)
    row, problem = agent.settle(ledger, skills, True)
    assert problem is None and row['ledger'] == 'match' and agent.checked == 1
    exact = sum(fractional_level(tenths[s]) - fractional_level(start[s]) for s in R.S)
    assert row['paid'] == pytest.approx(exact, rel=1e-12, abs=1e-12) == math.fsum(pays)
    assert row['xp'] == {s: v for s, v in zip(skills, total) if v and s in R.S}
    for wrong in (dict(ledger, total=[v + off * (k == stat) for k, v in enumerate(total)]), dict(ledger, end=tick + 1)):
        if wrong != ledger:
            agent.pending = dict(pending)
            assert 'ledger' in agent.settle(wrong, skills, True)[1]


@WORKERS
def test_a_death_ends_only_that_agents_life_and_it_starts_its_next_from_its_next_record(tmp_path, workers):
    """Agent 1 dies at its fourth decision: that life ends as a death with the engine's killers and no bootstrap, and its
    pay check runs at the reset of its next life, which starts from the next record of its stream (the engine's setup
    inside the reset) while the other agents of its world and the other worlds play on in their first."""
    batch, fakes = fake(tmp_path, worlds=3, agents=3, life=('fixed', 400), workers=workers,
                        make=lambda j: Body(3, death=4 if j == 1 else None))
    buffer = collect(learner(batch, n_steps=8))
    rows = lives(tmp_path)
    assert [(r['agent'], r['life'], r['terminal'], r['killers'], r['record'], r['decisions'], r['ledger'], r['grants'])
            for r in rows] == [(1, 0, 'death', ['Goblin'], 0, 4, 'match', 4 * len(GAIN))]
    assert [slot for _, slot, op, _ in fakes[0].log if op == 'reset'] == [0, 1, 2, 1]
    assert fakes[0].bodies[1].cfgs[1][1]['setStats'] == batch.agents[1].record['set_stats']
    assert batch.agents[1].record['index'] == 1 and all(a.life == 0 for a in batch.agents if a.j != 1)
    died = buffer.infos['terminal'] == TERMINAL_DEATH
    assert int(died.sum()) == 1 and float(buffer.cut_values[died].abs().sum()) == 0
    assert int(buffer.segments['agent'][int(died.nonzero()[0, 1])]) == 1
    assert batch.counts['ledger_checks'] == 1 and rows[0]['paid'] > 0
    batch.close()


def test_a_cut_life_ends_past_its_length_values_its_last_observation_and_waits_for_its_check(tmp_path):
    """Lives of 30 ticks with decisions of 2, 3 and 4 ticks end at their first decision point at or past 30 ticks; each
    cut row's bootstrap is the critic's value of that life's last observation from the recurrent state its last decision
    left, with no episode start; no other row bootstraps; and every life that ended in a rollout passed its ledger check
    before the rollout went to the update (its row is in lives.jsonl)."""
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    batch, fakes = fake(tmp_path, worlds=1, agents=3, life=('fixed', 30))
    model = learner(batch, n_steps=8)
    _, callback = model._setup_learn(3 * model.n_envs * model.n_steps, None)
    space = model.observation_space
    as_model = lambda k, v: v.to(MODEL_DTYPES[np.dtype(space[k].dtype)])
    for _ in range(3):
        model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
        buffer, rows = model.rollout_buffer, lives(tmp_path)
        checked = {(r['agent'], round(r['paid'], 4)) for r in rows}
        ended = (buffer.infos['terminal'] != TERMINAL_RUNNING).nonzero().tolist()
        assert ended and all((int(buffer.segments['agent'][c]), round(float(buffer.infos['episode_return'][t, c]), 4))
                             in checked for t, c in ended)
    assert {(r['agent'], r['decisions'], r['overrun']) for r in rows if r['life'] == 0} == {(0, 15, 0), (1, 10, 0), (2, 8, 2)}
    cut = buffer.infos['terminal'] == TERMINAL_HORIZON
    assert bool(cut.any()) and float(buffer.cut_values[~cut].abs().sum()) == 0
    logged = {(slot, r['tick']): r['obs'] for _, slot, _, r in fakes[0].log if 'tick' in r}  # replied items
    for t, c in cut.nonzero().tolist():  # a life by its agent and its pay (Body's lives differ)
        j, paid = int(buffer.segments['agent'][c]), float(buffer.infos['episode_return'][t, c])
        row = next(r for r in rows if r['agent'] == j and abs(r['paid'] - paid) < 1e-4)
        last = {k: as_model(k, torch.as_tensor(np.asarray(v))[None])
                for k, v in project_v6(logged[(j, row['arm_tick'] + row['ticks'])]).items()}
        obs = {k: as_model(k, v[t, c][None]) for k, v in buffer.observations.items()}
        state = lambda store: store[t, c][None].transpose(0, 1).contiguous()
        with torch.no_grad():
            _, _, _, after = model.policy(obs, RNNStates((state(buffer.hidden_states_pi), state(buffer.cell_states_pi)),
                                                         (state(buffer.hidden_states_vf), state(buffer.cell_states_vf))),
                                          buffer.episode_starts[t, c][None])
            value = model.policy.predict_values(last, after.vf, torch.zeros(1)).item()
        assert float(buffer.cut_values[t, c]) == pytest.approx(value, abs=1e-5)
    batch.close()


def test_the_rollouts_and_lives_do_not_depend_on_the_order_the_worlds_reply_in(tmp_path):
    """Two worlds of three agents, world 0 slow to reply in one run and world 1 in the other: agent j is slot j mod 3 of
    world j // 3, and both runs give the same rollouts bit for bit, the same segments by agent and the same lives."""
    from sim.train import rollout_digest
    runs = []
    for k, delays in enumerate(([0.02, 0.0], [0.0, 0.02])):
        batch, _ = fake(tmp_path / f'run{k}', life=('fixed', 30), delays=delays, n_envs=4)
        assert [(a.j, a.world, a.slot) for a in batch.agents] == [(j, j // 3, j % 3) for j in range(6)]
        model = learner(batch, n_steps=8)
        digests, agents = [], []
        for _ in range(3):
            buffer = collect(model)
            digests.append(rollout_digest(buffer))
            agents.append(buffer.segments['agent'].tolist())
        runs.append((digests, agents, lives(tmp_path / f'run{k}'), batch.updates[-1]['segments_per_agent']))
        batch.close()
    assert runs[0] == runs[1] and len(runs[0][2]) > 0


def test_worker_processes_give_the_same_rollouts_and_lives_as_one_process(tmp_path):
    """Two worlds of three agents collected three rollouts in this process and again with two worker processes, which
    read, decode and project the replies: the same rollouts bit for bit, the same segments and the same lives."""
    from sim.train import rollout_digest
    runs = []
    for workers in (0, 2):
        fakes = [FakeWorld(w, lambda k, w=w: Body(2 + (3 * w + k) % 3)) for w in range(2)]
        batch = R.RealBatch([f.port for f in fakes], agents=3, starts=V2, life=('fixed', 30), key=5, n_envs=4, stall=30.0,
                            radius=2, output=tmp_path / f'workers{workers}', workers=workers)
        model, out = learner(batch), []
        for _ in range(3):
            buffer = collect(model)
            out.append((rollout_digest(buffer), buffer.segments['agent'].tolist()))
        batch.close()
        runs.append((out, lives(tmp_path / f'workers{workers}')))
    assert runs[0] == runs[1] and len(runs[0][1]) > 0


@WORKERS
def test_a_ledger_difference_stops_the_run_before_the_next_update_and_keeps_its_evidence(tmp_path, workers):
    """Results that read ten tenths more woodcutting than the engine granted (from the first life's fifth decision on)
    stop sim.train at that reply's pay check against the engine's running ledger, inside the collection, before an update
    uses the row: the run fails with the reason, real-stop.json keeps the agent and the engine's result, and no latest.zip
    is saved."""
    from sim.train import train
    batch, _ = fake(tmp_path / 'lives', worlds=3, life=('fixed', 30), workers=workers,
                    make=lambda j: Body(2 + j % 3, corrupt=5 if j == 4 else None))
    with pytest.raises(R.RealStop, match='running ledger') as stopped:
        train(tmp_path / 'run', decisions=4 * 9 * 8, seed=1, worlds=9, n_steps=8, batch_size=16, n_epochs=1, env='real',
              environment=R.RealEnv(batch), task=FREE_PLAY_TASK, validate=True, verbose=False)
    evidence = json.loads(stopped.value.path.read_text())
    assert evidence['agent'] == 4 and evidence['result']['pay']['ledger'] and evidence['life'] == 0
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['status'] == 'failed' and 'RealStop' in summary['error'] and summary['engine'] == 'real'
    assert len(summary['real']['updates']) < 4 and not (tmp_path / 'run' / 'latest.zip').exists()


D = FREE_PLAY_V2['death_price']  # free-play-v2's price of a death, in levels
WRONG = {'missing': lambda result, price: 0.0,  # wrong death prices (sim/real.py death_charge) the checks must stop
         'another size': lambda result, price: price - 1 if result['dead'] else 0.0,
         'no death': lambda result, price: price}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.dictionaries(st.sampled_from([s.lower() for s in SKILL_ORDER]), st.integers(0, 40000), max_size=4),
                min_size=2, max_size=30), st.booleans(), st.sampled_from(sorted(WRONG)), st.data())
def test_free_play_v2_pays_a_deaths_level_gain_less_its_price_and_a_wrong_price_fails_the_check(steps, died, wrong, data):
    """free-play-v2 at one agent (D = 50): over any XP grants, the transition that the engine ends by a death pays its
    level gain less D and a life cut by its length pays its level gain alone; the row holds the level gain (paid), the
    price and the return with it apart, and the life's rewards equal its ledger level gain less D exactly when the
    engine's death record ends it. A price missing at the death, a price of another size there, or a price at a reply that
    holds no engine death fails the price check at that reply; a life whose price or rewards changed fails at its end."""
    skills = [s.lower() for s in SKILL_ORDER]
    agent = R.Agent(0, 0, 0, 1, 'danger', 0, D)
    agent.begin(1, ['draynor_west'], dict(draynor_west=[(3070, 3277)]), ('fixed', 10 ** 5 if died else 3 * len(steps)))
    tenths, tick, levels = dict.fromkeys(skills, 0), 100, dict.fromkeys(skills, 1)
    tenths['hitpoints'] = 11540
    agent.arm(dict(result_with(tenths, tick, 10), kind='reset'), dict(tenths), levels, [0] * len(skills), skills)
    start, rewards = dict(tenths), []
    at = data.draw(st.integers(0, len(steps) - 2)) if wrong == 'no death' else len(steps) - 1 if died else None
    for k, gains in enumerate(steps):
        for s, v in gains.items():
            tenths[s] += v
        tick += 3
        dies = died and k == len(steps) - 1
        result = result_with(tenths, tick, 3)
        if dies:
            result.update(done=True, dead=True, facts=dict(result['facts'], death=dict(tick=tick, killers=['Goblin'])))
        running = [tenths[s] - start[s] for s in skills]
        if k == at:
            replay = R.Agent(0, 0, 0, 1, 'danger', 0, D)
            replay.__dict__.update(agent.__dict__)
            with pytest.MonkeyPatch.context() as patch, pytest.raises(R.RealStop, match='price check'):
                patch.setattr(R, 'death_charge', WRONG[wrong])
                replay.stepped(result, dict(tenths), levels, running, skills)
        pay, ticks, end = agent.stepped(result, dict(tenths), levels, running, skills)
        assert end == ('death' if dies else 'horizon' if k == len(steps) - 1 else None) and agent.charge == D * dies
        rewards.append(pay - agent.charge)
    ledger = dict(start=agent.arm_tick, end=tick, grants=[], total=[tenths[s] - start[s] for s in skills])
    pending = dict(agent.pending)
    row, problem = agent.settle(ledger, skills, True)
    exact, owed = sum(fractional_level(tenths[s]) - fractional_level(start[s]) for s in R.S), D * died
    assert problem is None and row['price'] == owed and row['paid'] == pytest.approx(exact, rel=1e-12, abs=1e-12)
    assert row['return_with_price'] == pytest.approx(exact - owed, abs=1e-9) == math.fsum(rewards)
    for change in (dict(price=D - owed), dict(rewarded=pending['rewarded'] + 0.5)):
        agent.pending = dict(pending, **change)
        assert 'rewards' in agent.settle(ledger, skills, True)[1]


def test_free_play_v2_is_free_play_v1_with_the_price_at_the_engines_death_alone(tmp_path):
    """The same fake worlds and learner under free-play-v1 and v2, one rollout each: every row is the same but the
    engine's death, whose reward is its level gain less D (each cut life keeps its pay and its bootstrap); v1's life rows,
    counts, settings and checks hold no price; v2's life rows hold the level gain, the price and the return with it apart,
    its counts, update record and --validate counts the levels beside the prices (paid_sum the level gain in both), and
    the run's stored rows hold one price for its one engine death. --validate, which reads the stored rows, fails a v2
    rollout whose death row lacks its price and one with a price at a row that no death ended."""
    from sim.train import CheckFailed, validate_free_play
    runs = {}
    for price in (0.0, D):
        batch, _ = fake(tmp_path / str(price), life=('fixed', 20), death_price=price,
                        make=lambda j: Body(3, death=4 if j == 1 else None))
        model = learner(batch)
        if price:  # as sim.train sets a free-play-v2 model (VERSION_SAVED)
            model.free_play_task, model.death_price = FREE_PLAY_V2['version'], price
        b, carried = collect(model), dict(paid=torch.zeros(6, dtype=torch.float64), done=torch.ones(6))
        runs[price] = dict(rewards=b.rewards.clone(), terminal=b.infos['terminal'].clone(), actions=b.actions.clone(),
                           cut=b.cut_values.clone(), update=dict(batch.updates[-1]), settings=batch.settings_in_effect,
                           checks=validate_free_play(model, carried))
        if price:
            death = (b.infos['terminal'] == TERMINAL_DEATH).nonzero()[0].tolist()
            for row, change, reason in ((death, D, 'sum to its fractional level gain'),
                                        ([0, (death[1] + 1) % 6], -D, 'negative')):
                b.rewards[tuple(row)] += change
                with pytest.raises(CheckFailed, match=reason):
                    validate_free_play(model, carried)
                b.rewards[tuple(row)] -= change
        runs[price].update(final=batch.finish(), rows=lives(tmp_path / str(price)))
    v1, v2 = runs[0.0], runs[D]
    died = v1['terminal'] == TERMINAL_DEATH
    assert int(died.sum()) == 1 and torch.equal(v1['terminal'], v2['terminal']) and torch.equal(v1['actions'], v2['actions'])
    assert torch.equal(v1['rewards'][~died], v2['rewards'][~died]) and torch.equal(v1['cut'], v2['cut'])
    assert float(v2['rewards'][died]) == pytest.approx(float(v1['rewards'][died]) - D, abs=1e-5)
    priced = ('price', 'return_with_price')
    assert not any(k in r for r in v1['rows'] for k in priced) and len(v1['rows']) == len(v2['rows'])
    for one, two in zip(v1['rows'], v2['rows']):
        assert {k: v for k, v in two.items() if k not in priced} == one
        assert two['price'] == D * (one['terminal'] == 'death')
        assert two['return_with_price'] == pytest.approx(one['paid'] - two['price'], abs=1e-9)
    assert not {'levels_paid', 'prices', 'row_prices'} & set(v1['final']) and 'death_price' not in v1['settings']
    assert not {'prices', 'rewards_sum'} & set(v1['checks']) and 'rows_prices' not in v1['update']
    assert (v2['settings']['task'], v2['settings']['death_price']) == (FREE_PLAY_V2['version'], D)
    assert (v2['final']['deaths'], v2['final']['priced_deaths'], v2['final']['row_deaths']) == (1, 1, 1)
    assert v2['final']['row_prices'] == pytest.approx(D, abs=1e-4) == v2['update']['rows_prices'] == v2['checks']['prices']
    assert v2['checks']['paid_sum'] == pytest.approx(v1['checks']['paid_sum'], abs=1e-4) == v2['update']['rows_levels_paid']
    assert v2['checks']['rewards_sum'] == pytest.approx(v1['checks']['paid_sum'] - D, abs=1e-4)


def test_a_v2_reply_whose_dead_and_death_record_disagree_fails_the_price_check():
    """free-play-v2 prices a death by the reply's `dead` and checks it against the engine's death record (facts.death):
    a reply that is dead with no death record, or one with a death record that is not dead, fails the price check at
    that reply. A free-play-v1 life reads no death record: the same replies pass, as before."""
    skills = [s.lower() for s in SKILL_ORDER]
    for dead, record in ((True, None), (False, dict(tick=103, killers=['Goblin']))):
        for price in (D, 0.0):
            agent = R.Agent(0, 0, 0, 1, 'danger', 0, price)
            agent.begin(1, ['draynor_west'], dict(draynor_west=[(3070, 3277)]), ('fixed', 10 ** 5))
            tenths, levels = dict.fromkeys(skills, 0), dict.fromkeys(skills, 1)
            agent.arm(dict(result_with(tenths, 100, 10), kind='reset'), dict(tenths), levels, [0] * len(skills), skills)
            tenths['woodcutting'] += 10
            result = result_with(tenths, 103, 3)
            result.update(dead=dead, done=dead, facts=dict(result['facts'], death=record))
            running = [10 * (s == 'woodcutting') for s in skills]
            if price:
                with pytest.raises(R.RealStop, match='price check'):
                    agent.stepped(result, dict(tenths), levels, running, skills)
            else:
                assert agent.stepped(result, dict(tenths), levels, running, skills)[2] == ('death' if dead else None)


@pytest.mark.parametrize('where', ('the row before', 'no row'))
def test_a_price_missing_from_its_stored_row_stops_the_run_at_the_hand_over_without_validate(tmp_path, monkeypatch, where):
    """free-play-v2's check of the stored rows runs in every run (no --validate here): a death's price written to the
    row before the death, or missing from the stored row, while the agent's own pay and price are right, stops the run
    when that segment is handed to the learner, before its update."""
    original = R.RealBatch._completed

    def tampered(self, done, cuts):
        completed = original(self, done, cuts)

        def write(agent, pay, ticks, end):
            completed(agent, pay, ticks, end)
            if end == 'death':  # the stored row alone: the agent's books keep the right pay and price
                t, j = self.pos[agent.j] - 1, agent.j
                self.stage['rewards'][t, j] += D
                if where == 'the row before':
                    self.stage['rewards'][t - 1, j] -= D
        return write
    monkeypatch.setattr(R.RealBatch, '_completed', tampered)
    batch, _ = fake(tmp_path, life=('fixed', 30), death_price=D, make=lambda j: Body(3, death=4 if j == 1 else None))
    with pytest.raises(R.RealStop, match='stored rows hold a price'):
        collect(learner(batch))
    batch.close()
    assert 'stored rows' in json.loads((tmp_path / 'real-stop.json').read_text())['reason']


@pytest.mark.parametrize('wrong', sorted(WRONG))
def test_a_wrong_death_price_stops_the_run_at_that_reply_and_keeps_its_evidence(tmp_path, monkeypatch, wrong):
    """free-play-v2 in sim.train over the fake engine: a price missing at the engine's death, a price of another size
    there, or a price at a reply with no engine death stops the run at that reply's price check, inside the collection
    and before an update uses the row; real-stop.json keeps the agent and the engine's reply, and no latest.zip is
    saved."""
    from sim.train import train
    monkeypatch.setattr(R, 'death_charge', WRONG[wrong])
    batch, _ = fake(tmp_path / 'lives', life=('fixed', 30), death_price=D, make=lambda j: Body(3, death=4 if j == 1 else None))
    with pytest.raises(R.RealStop, match='price check') as stopped:
        train(tmp_path / 'run', decisions=2 * 6 * 8, seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real',
              environment=R.RealEnv(batch), task=FREE_PLAY_V2['version'], validate=True, verbose=False)
    evidence = json.loads(stopped.value.path.read_text())
    assert evidence['result']['dead'] == (wrong != 'no death') and (wrong == 'no death' or evidence['agent'] == 1)
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['status'] == 'failed' and not summary['real']['updates'] and not (tmp_path / 'run' / 'latest.zip').exists()


def test_free_play_v2_trains_only_in_the_real_engine_and_starts_from_v1_weights_by_weights_alone(tmp_path, monkeypatch):
    """Entry points of free-play-v2. sim.train --env real trains it on the fake engine with --validate from a free-play-v1
    checkpoint by --weights, recorded with both versions and the note that the value statistics were learned without the
    price; its summary, life rows and checkpoint name the version and the price, its prices are 50 times its deaths and
    its lives' level gain stays apart from them; a v2 leg continues a v2 checkpoint. A v2 checkpoint never starts a v1
    run, --init and a leg never cross the versions, an environment at another price is refused, the simulator engines and the fake environment refuse
    free-play-v2 by name, and the command takes --death-price 50 with free-play-v2 and with nothing else."""
    import sys
    import zipfile
    from sim import train as T
    from sim.vec import CpuBatch
    from sim.workers import CpuWorkers
    common = dict(seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real', validate=True, verbose=False)
    v2 = FREE_PLAY_V2['version']
    dying = lambda j: Body(3, death=4 if j == 1 else None)
    env = lambda name, price, **kw: R.RealEnv(fake(tmp_path / f'{name}-lives', life=('fixed', 30), death_price=price,
                                                   make=dying, **kw)[0])
    T.train(tmp_path / 'v1', decisions=2 * 48, environment=env('v1', 0.0), task=FREE_PLAY_TASK, **common)
    v1_zip = tmp_path / 'v1' / 'latest.zip'
    run = T.train(tmp_path / 'v2', decisions=3 * 48, environment=env('v2', D), task=v2, weights=v1_zip, checkpoint_every=48,
                  **common)
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=FREE_PLAY_TASK, run=v2)
    assert run['weights']['note'] == T.WEIGHTS_NOTES[v2]
    assert run['weights']['death_prices'] == dict(checkpoint=0.0, run=D)
    assert (run['free_play']['task'], run['free_play']['death_price'], run['learner']['free_play_task'],
            run['learner']['death_price']) == (v2, D, v2, D)
    lives_, finish, rows, checks = run['lives'], run['real_finish'], lives(tmp_path / 'v2-lives'), run['checks']['rollouts']
    # the summary's lives (the rollouts' episodes), --validate's read of the stored rows and the life rows agree
    assert lives_['deaths'] == sum(c['deaths'] for c in checks) >= 1
    assert lives_['prices'] == pytest.approx(sum(c['prices'] for c in checks), abs=1e-3)
    assert sum(r['price'] for r in rows) == D * sum(r['terminal'] == 'death' for r in rows) == finish['prices'] > 0
    assert finish['row_prices'] == pytest.approx(D * finish['row_deaths'], abs=1e-3) and finish['row_deaths'] >= lives_['deaths']
    with zipfile.ZipFile(tmp_path / 'v2' / 'latest.zip') as archive:
        saved = json.loads(archive.read('data'))
    assert (saved['free_play_task'], saved['death_price']) == (v2, D)
    v2_zip = tmp_path / 'v2' / 'latest.zip'
    leg = T.train(tmp_path / 'leg', decisions=3 * 48, environment=env('leg', D, leg=2), task=v2, checkpoint_every=48,
                  leg_from=tmp_path / 'v2' / 'checkpoints' / 'ppo_96_steps.zip', leg=2, **common)
    assert leg['status'] == 'completed' and (leg['leg']['updates_at_start'], leg['leg']['collected_not_trained']) == (2, 0)
    assert leg['leg']['policy_sha256'] == run['checkpoint_digests']['ppo_96_steps.zip']
    assert leg['real_finish']['priced_deaths'] >= 1
    assert (leg['learner']['free_play_task'], leg['learner']['death_price']) == (v2, D)
    refusals = ((dict(task=FREE_PLAY_TASK, weights=v2_zip), 0.0, {}, r"its weights never start a run of \('free-play-v1'"),
                (dict(task=v2, init=v1_zip), D, {}, r"optimizer state never starts a run of \('free-play-v2', 50.0\)"),
                (dict(task=v2, leg_from=v1_zip, leg=2, decisions=2 * 48), D, dict(leg=2),  # v1's own total
                 "other learner settings .*'free_play_task'"),
                (dict(task=v2), 0.0, {}, "the environment's task settings .* are not free-play-v2's"))
    for k, (change, price, batch, reason) in enumerate(refusals):
        environment = env(f'refused-{k}', price, **batch)
        with pytest.raises(ValueError, match=reason):
            T.train(tmp_path / f'refused-{k}', environment=environment, **{**common, 'decisions': 3 * 48, **change})
        environment.close()
    engine = 'built for the real engine only'
    with pytest.raises(ValueError, match=f'{engine}.*the fake environment does not pay it'):
        T.train(tmp_path / 'fake', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=v2)
    with pytest.raises(ValueError, match='unknown task free-play-v2'):  # the Python reference's own task_spec
        CpuBatch(None, None, 4, task=v2)
    with pytest.raises(ValueError, match=f'{engine}.*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=v2)
    with pytest.raises(ValueError, match=f'{engine}.*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=v2, engine='native', pack=tmp_path / 'any.pack')
    base = ['sim.train', 'train', '--output', str(tmp_path / 'cli'), '--decisions', '48', '--n-steps', '8', '--worlds', '6']
    for extra in (['--task', v2, '--env', 'real'], ['--task', v2, '--env', 'real', '--death-price', '49'],
                  ['--task', FREE_PLAY_TASK, '--env', 'real', '--death-price', '50']):
        monkeypatch.setattr(sys, 'argv', base + extra)
        with pytest.raises(SystemExit, match='--death-price 50 goes with --task free-play-v2'):
            T.main()
    monkeypatch.setattr(sys, 'argv', base + ['--task', v2, '--env', 'fake', '--death-price', '50'])
    with pytest.raises(ValueError, match=engine):
        T.main()
    assert not (tmp_path / 'cli').exists()


def test_sim_train_trains_in_the_real_engine_from_a_checkpoint_of_its_contract_and_refuses_another(tmp_path):
    """sim.train train --env real on the fake engine: every rollout passes --validate checked segment by segment (rows
    collected before the last update with their own weights); the summary records the engine, the batch's settings, each
    update's rows by age with PPO's approximate KL and clip fraction for each age (before the update, age 0 in segments of
    age 0 only is the behaviour policy itself), and its segments; at the run's end every live agent leaves and its life's
    ledger is checked; a second run starts from the first's checkpoint (--weights,
    checked against it on its first rollout); a free-play checkpoint of another observation contract is refused, and the
    batch refuses an engine whose hello reports random events on."""
    from sim.train import train
    common = dict(seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real', task=FREE_PLAY_TASK, validate=True,
                  verbose=False)
    first, _ = fake(tmp_path / 'first-lives', life=('fixed', 30))
    run = train(tmp_path / 'first', decisions=3 * 6 * 8, environment=R.RealEnv(first), **common)
    assert run['status'] == 'completed' and run['engine'] == 'real' and run['lives']['lives'] > 0
    effect = run['free_play']['batch_in_effect']
    assert (effect['engine'], effect['protocol'], effect['observation'], effect['worlds'], effect['agents_per_world'],
            effect['agents'], effect['n_envs'], effect['ledger'], effect['radius']) == \
        ('real', 'shared-lockstep-v1', 'obs-v6', 2, 3, 6, 6, True, 2)
    checks = run['checks']['rollouts']
    assert all(c['problems'] == [] for c in checks) and sum(c['lagged_rows'] for c in checks) > 0
    updates = run['real']['updates']
    assert len(updates) == 3 and updates[0]['lagged_share'] == 0 and any(u['lagged_share'] > 0 for u in updates[1:])
    kinds = ('0', '0_mixed', '1', '2', '3', '4+')
    assert all(u['before']['0']['approx_kl'] == pytest.approx(0, abs=1e-6) for u in updates if u['before']['0']['rows'])
    assert all(sum(u['rows_by_age'].values()) == 48 == sum(u['before'][k]['rows'] for k in kinds) for u in updates)
    assert all(u['rows_by_age']['0'] == u['before']['0']['rows'] + u['before']['0_mixed']['rows'] for u in updates)
    assert all(u['after'][k]['approx_kl'] > 0 for u in updates[:-1] for k in kinds if u['after'][k]['rows'])
    assert 'after' not in updates[-1] and any(u['before']['1']['approx_kl'] > 0 for u in updates[1:] if u['before']['1']['rows'])
    finish = run['real_finish']  # every live agent left, and each life's ledger was checked
    assert finish['left'] == 6 and finish['ledger_checks'] == finish['lives'] + finish['left'] > 6
    assert all(u['segments_per_agent'][1] >= 1 for u in updates) and run['real']['counts']['ledger_checks'] > 0
    second, _ = fake(tmp_path / 'second-lives', life=('fixed', 30))
    again = train(tmp_path / 'second', decisions=6 * 8, environment=R.RealEnv(second),
                  weights=tmp_path / 'first' / 'latest.zip', **common)
    assert again['weights']['policy_equal'] and again['source_check']['max_difference'] <= 1e-5
    assert again['policy_sha256_at_start'] == run['policy_sha256_at_end']
    other = train(tmp_path / 'fake', decisions=4 * 16, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1,
                  task=FREE_PLAY_TASK, verbose=False)
    assert other['free_play']['observation'] != 'obs-v6'
    third, _ = fake(tmp_path / 'third-lives', life=('fixed', 30))
    with pytest.raises(ValueError, match='obs-v6|observation'):
        train(tmp_path / 'third', decisions=6 * 8, environment=R.RealEnv(third), weights=tmp_path / 'fake' / 'latest.zip',
              **common)
    with pytest.raises(ValueError, match='--batch-setting workers=N'):  # the CPU workers' option is not the engine's
        train(tmp_path / 'fourth', decisions=6 * 8, environment=R.RealEnv(third), workers=2, **common)
    third.close()
    on = FakeWorld(0, lambda k: Body(2), random_events=True)
    with pytest.raises(ValueError, match='random events True'):
        R.RealBatch([on.port], agents=1, starts=V2, life=('fixed', 30), key=1, output=tmp_path / 'refused')


class Preempted(Exception):
    """A machine taken away (tests): the run stops as a rollout starts, after that boundary's checkpoint."""


def preempted_after(updates):
    """A callback that stops a run as it starts rollout `updates` + 1 of this process. sim.train's own callbacks run
    before the run's, so the boundary checkpoint of update `updates` is saved first, as a real run saves it before a
    preemption takes the machine."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Preempt(BaseCallback):
        started = 0

        def _on_step(self):
            return True

        def _on_rollout_start(self):
            self.started += 1
            if self.started > updates:
                raise Preempted(f'the machine was taken at rollout {self.started}')
    return Preempt()


def stopped(run, *args, **kwargs):
    """A run stopped by preempted_after: the summary its failure wrote."""
    with pytest.raises(Preempted):
        run(*args, **kwargs)
    return json.loads((args[0] / 'summary.json').read_text())


def test_two_legs_equal_one_run_through_the_first_legs_end_and_the_second_draws_no_start_again(tmp_path):
    """4 updates in one run against a first leg of the same run stopped after 2 (a preemption) and a second leg from
    its checkpoint at 2 updates (sim.train leg_from, leg). A real-engine checkpoint is saved after its update, so the one
    run's checkpoint at 2 updates and leg 1's are the same weights, PopArt statistics and Adam state, and leg 2 starts
    from them with the counts (decisions and updates) and runs to the run's total, its own checkpoint named by the
    whole run's count. Leg 1 draws the streams of a run before legs (each life's draws as the one run's); leg 2's agents
    each begin a new life from leg 2's streams, so no start of leg 1 is drawn again, and each of leg 2's life rows names
    its leg. After update 2 the two runs differ, as expected: the one run's rollouts 3 and 4 continue the lives in flight
    and the segments left in its pool at update 2, under generators that went on; leg 2's start new lives from other
    draws, under generators seeded by (seed, 2)."""
    import io
    import zipfile
    from baseline.telemetry import sha256
    from sim import watch
    from sim.train import LEG_SCOPES, train
    rollout, life = 6 * 8, ('uniform', 20, 40)
    common = dict(seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real', task=FREE_PLAY_TASK, validate=True,
                  verbose=False, checkpoint_every=2 * rollout, decisions=4 * rollout)
    start = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{2 * rollout}_steps.zip'
    runs = {}
    for name, leg in (('one', 1), ('leg1', 1), ('leg2', 2)):
        batch, _ = fake(tmp_path / f'{name}-lives', life=life, leg=leg)
        if name == 'leg1':
            runs[name] = stopped(train, tmp_path / name, environment=R.RealEnv(batch), callbacks=(preempted_after(2),),
                                 **common)
        else:
            given = dict(leg_from=start, leg=2) if leg == 2 else {}
            runs[name] = train(tmp_path / name, environment=R.RealEnv(batch), **given, **common)
    one, leg1, leg2 = runs['one'], runs['leg1'], runs['leg2']
    at_two = tmp_path / 'one' / 'checkpoints' / f'ppo_{2 * rollout}_steps.zip'  # the weights (PopArt's buffers among them)
    assert leg1['status'] == 'failed' and 'Preempted' in leg1['error'] and sorted(leg1['checkpoint_digests']) == [start.name]
    assert one['checkpoint_digests'][at_two.name] == leg1['checkpoint_digests'][start.name] == leg2['policy_sha256_at_start']

    def saved(path, entry):
        with zipfile.ZipFile(path) as archive:
            return archive.read('data') if entry == 'data' else torch.load(io.BytesIO(archive.read(entry)), weights_only=False)
    a, b = saved(at_two, 'policy.optimizer.pth'), saved(start, 'policy.optimizer.pth')
    assert a['state'].keys() == b['state'].keys() and all(
        torch.equal(a['state'][k][n], b['state'][k][n]) for k in a['state'] for n in a['state'][k])
    for path in (at_two, start):  # saved after update 2: its count and its updates agree
        assert [json.loads(saved(path, 'data'))[k] for k in ('num_timesteps', '_n_updates')] == [2 * rollout, 2]
    assert {k: leg2['leg'][k] for k in ('number', 'parent', 'sha256', 'policy_sha256', 'decisions_at_start',
                                        'updates_at_start', 'collected_not_trained', 'stream_offset', 'policy_equal',
                                        'first', 'scopes', 'decisions_this_leg', 'updates_whole_run')} == \
        dict(number=2, parent=str(tmp_path / 'leg1'), sha256=sha256(start), policy_sha256=leg1['checkpoint_digests'][start.name],
             decisions_at_start=2 * rollout, updates_at_start=2, collected_not_trained=0, stream_offset=R.LEG_STREAMS,
             policy_equal=True, first=str(tmp_path / 'leg1'), scopes=LEG_SCOPES, decisions_this_leg=2 * rollout,
             updates_whole_run=4)
    assert leg2['total_timesteps'] == 4 * rollout and leg2['ppo_updates'] == 2 and sorted(leg2['checkpoint_digests']) == \
        [f'ppo_{4 * rollout}_steps.zip']  # the leg's own checkpoint, named by the whole run's count
    lineage = json.loads((tmp_path / 'leg2' / 'lineage.json').read_text())
    assert (lineage['leg'], lineage['parent'], lineage['parent_sha256'], lineage['first'], lineage['parent_record']['number'],
            lineage['unverified']) == (2, str(start), sha256(start), str(tmp_path / 'leg1'), 1, None)
    assert leg2['leg']['optimizer_steps'] > 0 and leg2['source_check']['max_difference'] <= 1e-5
    assert len(leg2['real']['updates']) == 2 and json.loads(saved(tmp_path / 'leg2' / 'latest.zip', 'data'))['_n_updates'] == \
        json.loads(saved(tmp_path / 'one' / 'latest.zip', 'data'))['_n_updates'] == 4
    assert all(c['problems'] == [] for c in leg2['checks']['rollouts'])
    rows = {name: lives(tmp_path / f'{name}-lives') for name in runs}
    assert all(r['leg'] == 2 for r in rows['leg2']) and not any('leg' in r for r in rows['leg1'] + rows['one'])
    xp = [json.loads(line) for line in (tmp_path / 'leg2-lives' / 'lives-xp.jsonl').read_text().splitlines()]
    assert xp and all(r['leg'] == 2 for r in xp)
    draw = lambda r: (r['reset_seed'], r['site'], tuple(r['tile']), r['length'])
    whole = {(r['agent'], r['life']): draw(r) for r in rows['one']}
    assert all(whole[r['agent'], r['life']] == draw(r) for r in rows['leg1'] if (r['agent'], r['life']) in whole)
    assert {draw(r) for r in rows['leg2']}.isdisjoint(draw(r) for r in rows['leg1'] + rows['one'])
    group = watch.world_rules(V2)['start_group']
    first = {r['agent']: r for r in rows['leg2'] if r['life'] == 0}
    assert sorted(first) == list(range(6)) and all(  # each agent's first life of leg 2, from stream j + LEG_STREAMS
        first[j]['reset_seed'] == watch.Records(5, j + R.LEG_STREAMS, group).next()['reset_seed'] for j in first)
    assert one['policy_sha256_at_end'] != leg2['policy_sha256_at_end']  # expected to differ after update 2 (docstring)
    imitated = start.with_name('imitated.zip')  # as a run with self-imitation (self-imitation-v1, now removed) saved it
    with zipfile.ZipFile(start) as source, zipfile.ZipFile(imitated, 'w') as copy:
        for entry in source.infolist():
            raw = source.read(entry)
            copy.writestr(entry, json.dumps(dict(json.loads(raw), self_imitation_setting=dict(version='self-imitation-v1')))
                          if entry.filename == 'data' else raw)
    refusals = ((dict(n_epochs=2), 2, "other learner settings .*'n_epochs': \\[1, 2\\]"),
                (dict(ent_coef=0.05), 2, "other learner settings .*'ent_coef': \\[0.01, 0.05\\]"),
                ({}, 1, "the environment draws leg 1's streams, the run declares leg 2"),
                (dict(leg_from=imitated), 2, "other learner settings .*'self_imitation_setting'"))
    for k, (change, leg, reason) in enumerate(refusals):
        batch, _ = fake(tmp_path / f'refused-{k}', life=life, leg=leg)
        with pytest.raises(ValueError, match=reason):
            train(tmp_path / f'refused-{k}-run', environment=R.RealEnv(batch), **{**common, 'leg_from': start, 'leg': 2,
                                                                                    **change})
        batch.close()


def test_a_leg_continues_the_leg_its_checkpoint_names_to_the_runs_total_in_its_world_from_its_seed(tmp_path):
    """Legs 1, 2 and 3 of one run with an environment seed (the batch's key), legs 1 and 2 stopped after an update each:
    leg 2 takes the seed from leg 1's checkpoint and leg 3 from leg 2's, each leg's checkpoint saves its leg record
    (number, first leg, seed) and each leg leaves lineage.json at its start, the stopped leg 2 too. Refused before
    the leg starts anything: another leg number than the checkpoint's plus one, another total (the remainder, or more),
    another seed, --validate or --checkpoint-every, a checkpoint saved before its update (as early ones were), and one
    without its leg record unless --unrecorded-parent, which then continues only as leg 2, declares what it cannot check
    and keeps the key check; once the environment is built, another world. A checkpoint with its record refuses
    --unrecorded-parent."""
    import zipfile
    from sb3_contrib import RecurrentPPO
    from sim.train import LEG_UNVERIFIED, train
    rollout, life = 6 * 8, ('uniform', 20, 40)
    common = dict(seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real', task=FREE_PLAY_TASK, validate=True,
                  verbose=False, checkpoint_every=rollout, decisions=3 * rollout)
    ckpt = lambda name, n: tmp_path / name / 'checkpoints' / f'ppo_{n * rollout}_steps.zip'
    env = lambda name, leg=1, **kw: R.RealEnv(fake(tmp_path / f'{name}-lives', life=kw.pop('life', life), leg=leg, **kw)[0])
    runs = dict(leg1=stopped(train, tmp_path / 'leg1', environment=env('leg1'), env_seed=5,
                             callbacks=(preempted_after(1),), **common))
    runs['leg2'] = stopped(train, tmp_path / 'leg2', environment=env('leg2', 2), leg_from=ckpt('leg1', 1), leg=2,
                           callbacks=(preempted_after(1),), **common)
    runs['leg3'] = train(tmp_path / 'leg3', environment=env('leg3', 3), leg_from=ckpt('leg2', 2), leg=3, **common)
    assert runs['leg3']['status'] == 'completed' and runs['leg3']['leg']['updates_at_start'] == 2
    assert [runs[n]['env_seed'] for n in runs] == [5, 5, 5] and runs['leg3']['leg']['first'] == str(tmp_path / 'leg1')
    for name, number in (('leg1', 1), ('leg2', 2), ('leg3', 3)):
        (saved_ckpt,) = (tmp_path / name / 'checkpoints').glob('*.zip')  # each leg saved one, after its update
        with zipfile.ZipFile(saved_ckpt) as archive:
            record = json.loads(archive.read('data'))['real_leg']
        assert (record['number'], record['first'], record['env_seed'], record['decisions']) == \
            (number, str(tmp_path / 'leg1'), 5, 3 * rollout)
    for name, parent in (('leg2', ckpt('leg1', 1)), ('leg3', ckpt('leg2', 2))):
        lineage = json.loads((tmp_path / name / 'lineage.json').read_text())
        assert (lineage['leg'], lineage['parent'], lineage['env_seed']) == (int(name[-1]), str(parent), 5)
    rows = [r for name in ('leg2', 'leg3') for r in lives(tmp_path / f'{name}-lives')]
    assert {r['leg'] for r in rows} == {2, 3}
    drawn = {name: {r['reset_seed'] for r in lives(tmp_path / f'{name}-lives')} for name in runs}
    assert all(drawn.values()) and not (drawn['leg1'] & drawn['leg2'] or drawn['leg1'] & drawn['leg3'] or
                                        drawn['leg2'] & drawn['leg3'])  # no leg draws another's starts
    crafted = tmp_path / 'crafted'
    crafted.mkdir()
    model = RecurrentPPO.load(str(ckpt('leg2', 2)), device='cpu')
    model._n_updates -= 1  # saved before its last update, as every early real-engine checkpoint
    model.save(str(crafted / 'early.zip'))
    model = RecurrentPPO.load(str(ckpt('leg1', 1)), device='cpu')
    del model.real_leg  # a first leg's checkpoint before checkpoints held their leg record
    model.save(str(crafted / 'unrecorded.zip'))
    early = ((dict(leg_from=ckpt('leg2', 2), leg=2), 'is a checkpoint of leg 2: the leg that continues it is --leg 3, '
              'not --leg 2'),
             (dict(leg_from=ckpt('leg1', 1), leg=3), '--leg 2, not --leg 3'),
             (dict(leg_from=ckpt('leg2', 2), leg=3, decisions=rollout), "other run settings .*'decisions': \\[144, 48\\]"),
             (dict(leg_from=ckpt('leg2', 2), leg=3, decisions=4 * rollout), "'decisions': \\[144, 192\\]"),
             (dict(leg_from=crafted / 'unrecorded.zip', leg=2, unrecorded_parent=True, decisions=2 * rollout),
              f'a run of {3 * rollout} decisions, which its legs continue to: --decisions {3 * rollout}, not'),
             (dict(leg_from=ckpt('leg2', 2), leg=3, env_seed=6), 'draws from environment seed 5'),
             (dict(leg_from=ckpt('leg2', 2), leg=3, validate=False), "other run settings .*'validate': \\[True, False\\]"),
             (dict(leg_from=ckpt('leg2', 2), leg=3, checkpoint_every=2 * rollout), "'checkpoint_every': \\[48, 96\\]"),
             (dict(leg_from=crafted / 'early.zip', leg=3), 'saved before its last update'),
             (dict(leg_from=crafted / 'unrecorded.zip', leg=2), 'holds no leg record'),
             (dict(leg_from=crafted / 'unrecorded.zip', leg=3, unrecorded_parent=True), '--leg 2, not --leg 3'),
             (dict(leg_from=ckpt('leg2', 2), leg=3, unrecorded_parent=True), 'holds its leg record'))
    late = ((dict(leg_from=ckpt('leg2', 2), leg=3, environment=env('world', 3, life=('fixed', 30))),
             "in another world .*'life'"),
            (dict(leg_from=ckpt('leg2', 2), leg=3, environment=env('v3', 3, starts=WORLD_V3)),
             "in another world .*'starts': \\['village-world-v2', 'village-world-v3'\\]"),
            (dict(leg_from=crafted / 'unrecorded.zip', leg=2, unrecorded_parent=True, environment=env('key', 2)),
             'draws from key 5, the leg\'s run from environment seed 1'))
    for k, (change, reason) in enumerate(early + late):
        environment = change.pop('environment', None) or env(f'refused-{k}', change['leg'])
        with pytest.raises(ValueError, match=reason):
            train(tmp_path / f'refused-{k}', environment=environment, **dict(common, **change))
        assert (tmp_path / f'refused-{k}').exists() == (k >= len(early))
        environment.close()
    declared = train(tmp_path / 'declared', environment=env('declared', 2), leg_from=crafted / 'unrecorded.zip', leg=2,
                     unrecorded_parent=True, env_seed=5, **common)
    assert declared['status'] == 'completed' and declared['leg']['unverified'] == list(LEG_UNVERIFIED)
    assert json.loads((tmp_path / 'declared' / 'lineage.json').read_text())['unverified'] == list(LEG_UNVERIFIED)
    assert declared['leg']['first'] == str(crafted)
    crossed = train(tmp_path / 'crossed', environment=env('crossed', starts=WORLD_V3), weights=ckpt('leg2', 2),
                    **dict(common, decisions=rollout))  # --weights starts either world from the other's policy
    assert crossed['status'] == 'completed' and crossed['free_play']['batch_in_effect']['starts'] == WORLD_V3

V3 = FREE_PLAY_V3['version']
OUTSIDE = dict(GAIN, Firemaking=137)  # tenths a decision grants, one stat outside the seven paid by free-play-v1


@settings(max_examples=60, deadline=None)
@given(st.lists(st.dictionaries(st.sampled_from([s.lower() for s in SKILL_ORDER]), st.integers(0, 40000), max_size=4),
                min_size=1, max_size=30), st.sampled_from([s.lower() for s in SKILL_ORDER if s.lower() not in R.S]),
       st.integers(-1, 1), st.data())
def test_free_play_v3_pays_every_stats_level_gain_and_a_tenth_off_in_a_new_stat_fails_its_check(steps, new, off, data):
    """free-play-v3 at one agent (every stat of the ledger paid): over any XP grants, a decision pays the fractional
    level gain over every stat where free-play-v1 pays the seven's alone, so XP outside the seven pays under v3 only;
    the v3 row holds the seven's gain (paid_seven, v1's paid) apart from every stat's (paid), and every stat's XP and
    whole levels. A running ledger a tenth off in a stat outside the seven fails the check at that reply under either,
    and a closing ledger a tenth off there fails it at the life's end: those two ledger checks covered every stat before
    free-play-v3, and this test shows they still do for a v3 agent."""
    skills = [s.lower() for s in SKILL_ORDER]
    agents = dict(v1=R.Agent(0, 0, 0, 1, 'danger', 0),
                  v3=R.Agent(0, 0, 0, 1, 'danger', 0, 0.0, tuple(skills), all_skills=True))
    tenths, tick = dict.fromkeys(skills, 0), 100
    tenths['hitpoints'] = 11540
    base = lambda: {s: level_of(tenths[s] // 10) for s in skills}  # the whole levels the XP gives
    levels = base()
    for agent in agents.values():
        agent.begin(1, ['draynor_west'], dict(draynor_west=[(3070, 3277)]), ('fixed', 10 ** 5))
        agent.arm(dict(result_with(tenths, tick, 10), kind='reset'), dict(tenths), levels, [0] * len(skills), skills)
    start, pays = dict(tenths), dict(v1=[], v3=[])
    wrong_at = data.draw(st.integers(0, len(steps) - 1)) if off else None
    for k, gains in enumerate(steps):
        for s, v in gains.items():
            tenths[s] += v
        tick += 3
        result, running = result_with(tenths, tick, 3), [tenths[s] - start[s] for s in skills]
        for name, agent in agents.items():
            if k == wrong_at:
                replay = R.Agent(0, 0, 0, 1, 'danger', 0)
                replay.__dict__.update(agent.__dict__)
                with pytest.raises(R.RealStop, match='running ledger'):
                    replay.stepped(result, dict(tenths), levels, [v + off * (s == new) for s, v in zip(skills, running)],
                                   skills)
            pays[name].append(agent.stepped(result, dict(tenths), levels, running, skills)[0])
    rows, total = {}, [tenths[s] - start[s] for s in skills]
    ledger = dict(start=agents['v1'].arm_tick, end=tick, grants=[], total=total)
    for name, agent in agents.items():
        agent.close_life('left', base())
        pending = dict(agent.pending)
        rows[name], problem = agent.settle(ledger, skills, True)
        assert problem is None and rows[name]['ledger'] == 'match'
        if off:
            agent.pending = dict(pending)
            assert 'ledger' in agent.settle(dict(ledger, total=[v + off * (s == new) for s, v in zip(skills, total)]),
                                            skills, True)[1]
    gain = lambda names: sum(fractional_level(tenths[s]) - fractional_level(start[s]) for s in names)
    assert rows['v1']['paid'] == pytest.approx(gain(R.S), abs=1e-9) == math.fsum(pays['v1'])
    assert rows['v3']['paid'] == pytest.approx(gain(skills), abs=1e-9) == math.fsum(pays['v3'])
    assert rows['v3']['paid_seven'] == pytest.approx(rows['v1']['paid'], abs=1e-9) and 'paid_seven' not in rows['v1']
    assert rows['v3']['xp_all'] == {s: v for s, v in zip(skills, total) if v} and rows['v3']['xp'] == rows['v1']['xp']
    assert rows['v3']['gain_all_s'] == sum(level_of(tenths[s] // 10) - level_of(start[s] // 10) for s in skills)


def test_a_free_play_v3_run_pays_every_stat_with_a_tinderbox_in_each_kit_and_keeps_the_seven_apart(tmp_path):
    """sim.train --task free-play-v3 on the fake engine, whose bodies also gain Firemaking XP, from a free-play-v1
    checkpoint by --weights (recorded with both versions and its note): each reset's kit holds the record's items and one
    tinderbox (a v1 run's, the record's alone); every check is clean with --validate; each life row's paid is every
    stat's fractional gain and paid_seven the seven's, apart by exactly the Firemaking gain its XP gives; each update and
    the progress rows hold the seven's and every stat's gain apart; the summary, the batch's settings and the checkpoint
    name the task, its 21 paid stats and its kit. The same v3 run with the collection's calls on a processor copy
    (collect_device) gives the same lives and weights. lives-xp.jsonl holds every stat's XP for each life of every
    version: the v1 run's Firemaking XP is there, while its lives.jsonl rows keep the seven's alone."""
    import csv
    import zipfile
    from sim import train as T
    worlds = {}

    def env(name, task, copy=None):
        batch, fakes = fake(tmp_path / f'{name}-lives', life=('fixed', 30), make=lambda j: Body(3, gain=OUTSIDE),
                            collect_device=copy, **T.task_batch(task))
        worlds[name] = fakes
        return R.RealEnv(batch)
    T.train(tmp_path / 'v1', decisions=48, environment=env('v1', FREE_PLAY_TASK), **DEVICES)
    run = T.train(tmp_path / 'v3', decisions=2 * 48, environment=env('v3', V3), weights=tmp_path / 'v1' / 'latest.zip',
                  **dict(DEVICES, task=V3))
    copy = T.train(tmp_path / 'v3-copy', decisions=2 * 48, environment=env('v3-copy', V3, 'cpu'),
                   weights=tmp_path / 'v1' / 'latest.zip', **dict(DEVICES, task=V3))
    assert copy['status'] == 'completed' and copy['policy_sha256_at_end'] == run['policy_sha256_at_end']
    assert lives(tmp_path / 'v3-copy-lives') == lives(tmp_path / 'v3-lives')
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=FREE_PLAY_TASK, run=V3)
    assert run['weights']['note'] == T.WEIGHTS_NOTES[V3]
    for name, tail in (('v1', ''), ('v3', ',tinderbox:1')):
        cfgs = [(body, cfg) for f in worlds[name] for body in f.bodies.values() for _, cfg in body.cfgs]
        assert cfgs and all(cfg['seedItems'].endswith(tail) and 'tinderbox' not in cfg['seedItems'][:-len(tail) or None]
                            for _, cfg in cfgs)
    rows = [r for r in lives(tmp_path / 'v3-lives') if r['decisions']]
    assert rows and all('firemaking' in r['xp_all'] and 'firemaking' not in r['xp'] for r in rows)
    for r in rows:  # every stat's pay less the seven's is the Firemaking gain its XP gives (it starts at level 1)
        assert r['paid'] - r['paid_seven'] == pytest.approx(fractional_level(r['xp_all']['firemaking']) - 1, abs=1e-9)
    assert not any('paid_seven' in r for r in lives(tmp_path / 'v1-lives'))
    updates, settled = run['real']['updates'], run['lives']['settled']
    assert all(u['levels_all'] > u['levels_seven'] for u in updates if u['levels_all'])
    assert settled['levels_all'] > settled['levels_seven'] and len(run['lives']['paid_skills']) == len(SKILL_ORDER)
    assert run['lives']['paid_skills'] == run['learner']['paid_skills'] and run['real']['settings']['task'] == V3
    every = [json.loads(line) for line in (tmp_path / 'v1-lives' / 'lives-xp.jsonl').read_text().splitlines()]
    assert len(every) == len(lives(tmp_path / 'v1-lives')) and all('firemaking' in r['xp'] for r in every if r['xp'])
    progress = list(csv.DictReader(open(tmp_path / 'v3' / 'metrics' / 'progress.csv')))
    assert all(row['real/levels_seven'] != '' and row['real/levels_all'] != '' for row in progress)
    assert (run['free_play']['task'], run['free_play']['kit_extra'], run['real']['settings']['kit_extra']) == \
        (V3, [['tinderbox', 1]], [['tinderbox', 1]])
    with zipfile.ZipFile(tmp_path / 'v3' / 'latest.zip') as archive:
        saved = json.loads(archive.read('data'))
    assert (saved['free_play_task'], saved['kit_extra'], len(saved['paid_skills'])) == (V3, [['tinderbox', 1]], len(SKILL_ORDER))
    assert run['learner']['paid_skills'] == saved['paid_skills'] == run['real']['settings']['paid_skills']


def test_free_play_v3_trains_in_the_real_engine_only_and_crosses_versions_by_weights_alone(tmp_path, monkeypatch):
    """free-play-v3's entry points: the fake environment, the Python reference and the C engine refuse it by name;
    --death-price stays free-play-v2's alone; a v3 checkpoint never starts a v1 or v2 run, a v2 checkpoint never starts a
    v3 run, --init and a leg never cross the versions, and an environment of another task is refused, before any update."""
    import sys
    from sim import train as T
    from sim.vec import CpuBatch
    from sim.workers import CpuWorkers
    built = lambda name, task, **kw: R.RealEnv(fake(tmp_path / f'{name}-lives', life=('fixed', 30), **T.task_batch(task),
                                                    **kw)[0])
    for name, task in (('v1', FREE_PLAY_TASK), ('v2', FREE_PLAY_V2['version']), ('v3', V3)):
        T.train(tmp_path / name, decisions=48, environment=built(name, task), **dict(DEVICES, task=task))
    zips = {name: tmp_path / name / 'latest.zip' for name in ('v1', 'v2', 'v3')}
    refusals = ((dict(task=FREE_PLAY_TASK, weights=zips['v3']), {}, r"its weights never start a run of \('free-play-v1'"),
                (dict(task=FREE_PLAY_V2['version'], weights=zips['v3']), {}, r"never start a run of \('free-play-v2'"),
                (dict(task=V3, weights=zips['v2']), {}, r"never start a run of \('free-play-v3'"),
                (dict(task=V3, init=zips['v1']), {}, r"optimizer state never starts a run of \('free-play-v3'"),
                (dict(task=V3, leg_from=zips['v1'], leg=2, decisions=48), dict(leg=2),  # v1's own total
                 "other learner settings .*'free_play_task'"))
    for k, (change, batch, reason) in enumerate(refusals):
        environment = built(f'refused-{k}', change['task'], **batch)
        with pytest.raises(ValueError, match=reason):
            T.train(tmp_path / f'refused-{k}', environment=environment, **{**DEVICES, 'decisions': 96, **change})
        environment.close()
    with pytest.raises(ValueError, match='a batch pays one version'):  # a price with every stat paid
        R.RealBatch([0], agents=1, starts=V2, life=('fixed', 30), key=1, output=tmp_path / 'pair', death_price=D,
                    all_skills=True)
    environment = built('other-task', FREE_PLAY_TASK)
    with pytest.raises(ValueError, match="the environment's task settings .* are not free-play-v3's"):
        T.train(tmp_path / 'other-task', decisions=48, environment=environment, **dict(DEVICES, task=V3))
    environment.close()
    engine = "free-play-v3 \\(every stat of the engine's ledger paid and a tinderbox in the start kit\\) is built for the real engine only"
    with pytest.raises(ValueError, match=f'{engine}.*the fake environment does not pay it'):
        T.train(tmp_path / 'fake', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V3)
    with pytest.raises(ValueError, match=f'{engine}.*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V3)
    with pytest.raises(ValueError, match=f'{engine}.*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V3, engine='native', pack=tmp_path / 'any.pack')
    with pytest.raises(ValueError, match='unknown task free-play-v3'):
        CpuBatch(None, None, 4, task=V3)
    monkeypatch.setattr(sys, 'argv', ['sim.train', 'train', '--output', str(tmp_path / 'cli'), '--decisions', '48',
                                      '--n-steps', '8', '--worlds', '6', '--task', V3, '--env', 'real', '--death-price', '50'])
    with pytest.raises(SystemExit, match='--death-price 50 goes with --task free-play-v2'):
        T.main()
    assert not (tmp_path / 'cli').exists()


# ---- the life rows' four reads -----------------------------------------------------------------------------------------
# the fake engine's scenario (key 25): agent 0 is hit for 3 hitpoints and heals them, agent 1 loses 2, regains 1, then 1,
# agent 2 dies at its fourth decision from full hitpoints, agent 3 is never hit
HITS = {0: {(0, 2): 3, (0, 3): -3}, 1: {(0, 2): 2, (0, 4): -1, (0, 5): -1}}
DEATHS = {2: 4}


def one_life(replies, life_facts=True):
    """Agent 0 of a batch played by hand through one life: `replies` the engine's facts of its arm and of each decision's
    reply (hp [current, base], eats, food, food_drops; a reply with dead True, or of hp 0, holds the engine's death
    record and ends the life), else the life left at the end; no XP is gained. Returns its life row."""
    site = 'draynor_west'
    agent = R.Agent(0, 0, 0, 5, R.watch.world_rules(V2)['start_group'], 0, life_facts=life_facts)
    agent.begin(5, [site], {site: [tuple(R.watch.SITES[site])]}, R.check_life(('fixed', 100000)))
    xp, base, skills, tick, end = {s: 0 for s in R.S}, {s: 1 for s in R.S}, list(R.S), 100, None
    agent.arm(dict(tick=tick, ticks=6, facts=dict(replies[0], weapon=None, death=None), projected={}, obs=None), xp, base,
              [0] * len(skills), skills)
    for facts in replies[1:]:
        facts = dict(facts)
        tick, dead = tick + 2, facts.pop('dead', facts['hp'][0] == 0)
        facts = dict(facts, weapon=None, death=dict(tick=tick, killers=['Goblin']) if dead else None)
        end = agent.stepped(dict(tick=tick, ticks=2, done=dead, dead=dead, facts=facts, projected={}, obs=None), xp, base,
                            [0] * len(skills), skills)[2]
    if end is None:
        agent.close_life('left', base)
    return agent.settle(None, skills, False)[0]


def test_a_life_row_reads_the_engines_tool_drops_beside_the_four_reads_when_the_engine_reports_them():
    """facts.tool_drops (env-ts FACT_FIELDS: the life's Drops that took a tool, which mask-v6 allows) reaches the life row
    as tool_drops beside the four reads, as the engine last reported it; a row of an engine that reports none has none,
    and its four reads are as before."""
    fact = lambda hp, drops=0, **more: dict(hp=hp, eats=0, food=1, food_drops=drops, **more)  # noqa: E731
    row = one_life([fact([10, 10], tool_drops=0), fact([10, 10], tool_drops=1), fact([10, 10], 1, tool_drops=2)])
    assert (row['tool_drops'], row['food_drops']) == (2, 1) and set(R.LIFE_FACTS) <= set(row)
    plain = one_life([fact([10, 10]), fact([10, 10], 1)])
    assert 'tool_drops' not in plain and plain['food_drops'] == 1 and set(R.LIFE_FACTS) <= set(plain)

def test_the_four_reads_count_the_half_hitpoint_decisions_their_food_their_eats_and_the_food_drops():
    """Lives played by hand: the reads count the decisions at half hitpoints or less (2 * hp <= base, before the
    decision), those of them with food in the pack, the Eats at them and the life's drops of food; `paid` stays the
    levels alone (0 here), and a batch whose engine names no food facts keeps none of the four."""
    fact = lambda hp, eats=0, food=0, drops=0: dict(hp=hp, eats=eats, food=food, food_drops=drops)  # noqa: E731
    died = one_life([fact([10, 10], food=2), fact([4, 10], food=2), fact([7, 10], 1, 1), fact([7, 10], 2, 0),
                     fact([5, 10], 2), fact([5, 10], 2, 1), fact([5, 10], 2, 0, 1), fact([0, 10], 2, 0, 1)])
    assert died['terminal'] == 'death' and died['paid'] == 0.0
    assert {k: died[k] for k in R.LIFE_FACTS} == dict(half_hp_decisions=4, half_hp_food_decisions=2, half_hp_eats=1,
                                                      food_drops=1) and died['eats'] == 2
    healed = one_life([fact([10, 10], food=1), fact([8, 10]), fact([10, 10], 1), fact([10, 10], 2), fact([9, 10], 2),
                       fact([10, 10], 2)])
    assert healed['terminal'] == 'left' and healed['half_hp_decisions'] == 0
    plain = one_life([fact([10, 10]), fact([4, 10]), fact([4, 10])], life_facts=False)
    assert not set(R.LIFE_FACTS) & set(plain)


def test_a_batch_keeps_the_four_reads_when_its_engine_reports_food_and_none_when_it_does_not(tmp_path):
    """The fake engine's scenario (HITS, DEATHS) in free-play-v3: the life rows hold the four reads when the engine reports
    food, none of them when it does not, and are otherwise the same."""
    from sim import train as T

    def play(name, task, food=True):
        batch, _ = fake(tmp_path / name, worlds=2, agents=2, key=25, life=('fixed', 30), world=lambda w: dict(food=food),
                        make=lambda j: Body(3, hits=HITS.get(j), death=DEATHS.get(j)), **T.task_batch(task))
        batch.play(1, actor='wait', rounds=40)
        return lives(tmp_path / name)
    v3, plain = play('v3', V3), play('v3-plain', V3, food=False)
    life = {(r['agent'], r['life']): r for r in v3}
    assert life[2, 0]['terminal'] == 'death' and life[0, 0]['half_hp_decisions'] == 0
    assert all(set(R.LIFE_FACTS) <= set(r) for r in v3) and not any(set(R.LIFE_FACTS) & set(r) for r in plain)
    assert [{k: v for k, v in r.items() if k not in R.LIFE_FACTS} for r in v3] == plain


class Pages(Body):
    """A Body whose observations also offer the pack's page turns (rows 67 and 68), which the fake world plays as nothing."""

    def obs(self):
        o = super().obs()
        for row in R.PAGE_TURNS:
            o['mask'][row] = 1
        return o


def test_a_life_row_counts_the_decisions_that_turned_the_pack_page_in_every_version(tmp_path):
    """page_turns, a read that nothing pays: each life row counts the decisions of its life that played row 67 or 68
    (R.PAGE_TURNS: sim/agent.py A_INV_PREV and A_INV_NEXT). Random play over bodies that offer both rows, in free-play-v1
    and in free-play-v15 (the rare-XP pay): every life row's page_turns is the count of page turns answered in that life,
    and a life that turned no page (or answered nothing) counts 0."""
    from sim import train as T
    from sim.agent import A_INV_NEXT, A_INV_PREV
    assert R.PAGE_TURNS == (A_INV_PREV, A_INV_NEXT) == (67, 68)
    for name, task in (('v1', FREE_PLAY_TASK), ('v15', FREE_PLAY_V15['version'])):
        batch, _ = fake(tmp_path / name, worlds=2, agents=2, key=25, life=('fixed', 30), make=lambda j: Pages(3),
                        world=lambda w: dict(hurt=True), **T.task_batch(task))
        answered, answer = {}, batch.answer

        def logged(agents, actions, answer=answer, answered=answered):
            for agent, action in zip(agents, actions):
                answered[agent.j, agent.life] = answered.get((agent.j, agent.life), 0) + (int(action) in R.PAGE_TURNS)
            return answer(agents, actions)
        batch.answer = logged
        batch.play(3, actor='random', rounds=60)
        rows = lives(tmp_path / name)
        assert rows and {(r['agent'], r['life']): r['page_turns'] for r in rows} == \
            {key: answered.get(key, 0) for key in {(r['agent'], r['life']) for r in rows}}
        assert sum(r['page_turns'] for r in rows) > 0 and set(answered) <= {(r['agent'], r['life']) for r in rows}


V10 = FREE_PLAY_V10['version']
RAW_FOOD_KIT = (('tinderbox', 1), ('raw_chicken', 3))  # the removed free-play-v7's kit: a batch refuses it


class Unstamped(Body):
    """A Body of an engine that ignores the reset's item contract: its observations stamp item-v2 whatever the life's cfg
    names."""

    def reset(self, seed, cfg):
        obs = super().reset(seed, cfg)
        self.contract = 'item-v2'
        return dict(obs, contractVersion='item-v2')


def test_free_play_v10_plays_the_page_of_kinds_starts_from_v3_weights_alone_and_nothing_else_loads_across(tmp_path):
    """free-play-v10 (sim/workers.py FREE_PLAY_V10): free-play-v3 on item-v3. Every life's reset names item-v3 and its
    observations stamp it; the batch, its summary and its checkpoint name v10 and item-v3, and its policy records the
    item contract beside its act, mask and feedback versions. A v3 checkpoint starts a v10 run by --weights, recorded with
    both versions, both item contracts and the note, every check clean with --validate. Nothing else crosses: v10 weights
    start no v3 run, v1 and v7 weights no v10 run, and a v3 checkpoint's optimizer state (--init) none, by the recorded
    contract. Leg 2 of a v10 run stays v10 under item-v3; a leg under v3 is refused. An observation that stamps another
    item contract than the run's stops the run. The fake environment, the Python reference and the C engine refuse v10
    by name, and a batch with item-v3 and another pay or kit is refused."""
    import zipfile
    from sb3_contrib import RecurrentPPO
    from baseline.policy import CONTRACT, run_contract
    from sim import train as T
    from sim.workers import CpuWorkers
    rollout = 6 * 8
    worlds = {}

    def env(name, task, leg=1, make=lambda j: Body(3)):
        batch, worlds[name] = fake(tmp_path / f'{name}-lives', key=25, life=('fixed', 30), leg=leg, make=make,
                                   **T.task_batch(task))
        return R.RealEnv(batch)
    cfgs = lambda name: [cfg for f in worlds[name] for body in f.bodies.values() for _, cfg in body.cfgs]
    for name, task in (('v1', FREE_PLAY_TASK), ('v3', V3)):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), **dict(DEVICES, task=task))
    run = T.train(tmp_path / 'v10', decisions=2 * rollout, environment=env('v10', V10), weights=tmp_path / 'v3' / 'latest.zip',
                  **dict(DEVICES, task=V10))
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=V3, run=V10) and run['weights']['note'] == T.WEIGHTS_NOTES[V10]
    assert run['weights']['item_contracts'] == dict(checkpoint='item-v2', run='item-v3')
    assert (run['real']['settings']['task'], run['real']['settings']['item_contract']) == (V10, 'item-v3')
    assert run['free_play']['items'] == 'item-v3' and 'item_contract' not in json.loads(
        (tmp_path / 'v3' / 'summary.json').read_text())['real']['settings']
    assert cfgs('v10') and all(c['contract'] == 'item-v3' for c in cfgs('v10'))
    assert cfgs('v3') and all(c['contract'] == 'item-v2' for c in cfgs('v3'))
    saved = RecurrentPPO.load(str(tmp_path / 'v10' / 'latest.zip'), device='cpu')
    assert saved.policy.contract == run_contract('item-v3') == dict(CONTRACT, items='item-v3')
    assert RecurrentPPO.load(str(tmp_path / 'v3' / 'latest.zip'), device='cpu').policy.contract == CONTRACT
    with zipfile.ZipFile(tmp_path / 'v10' / 'latest.zip') as archive:
        assert json.loads(archive.read('data'))['free_play_task'] == V10
    refusals = ((V3, dict(weights='v10'), r"its weights never start a run of \('free-play-v3'|trained under"),
                (V10, dict(weights='v1'), r"its weights never start a run of \('free-play-v10'"),
                (V10, dict(init='v3'), 'trained under'))
    for k, (task, source, reason) in enumerate(refusals):
        environment = env(f'refused-{k}', task)
        with pytest.raises(ValueError, match=reason):
            T.train(tmp_path / f'refused-{k}', environment=environment, **{**DEVICES, 'decisions': rollout, 'task': task},
                    **{key: tmp_path / value / 'latest.zip' for key, value in source.items()})
        environment.close()
    common = dict(DEVICES, task=V10, checkpoint_every=rollout, decisions=3 * rollout)
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', V10), env_seed=5, callbacks=(preempted_after(1),), **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', V10, 2), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and all(c['problems'] == [] for c in leg2['checks']['rollouts'])
    assert (leg2['real']['settings']['task'], leg2['real']['settings']['item_contract']) == (V10, 'item-v3')
    assert cfgs('leg2') and all(c['contract'] == 'item-v3' for c in cfgs('leg2'))
    environment = env('leg-v3', V3, 2)
    with pytest.raises(ValueError, match='under other learner settings|would continue|trained under'):
        T.train(tmp_path / 'leg-v3', environment=environment, leg_from=parent, leg=2, **{**common, 'task': V3})
    environment.close()
    with pytest.raises(R.RealStop, match="an observation of the item contract 'item-v2'; the run plays 'item-v3'"):
        T.train(tmp_path / 'unstamped', environment=env('unstamped', V10, make=lambda j: Unstamped(3)), **common)
    engine = f'{V10} \\(free-play-v3 on the page of kinds \\(item-v3\\).*\\) is built for the real engine only'
    with pytest.raises(ValueError, match=f'{engine}.*the fake environment does not pay it'):
        T.train(tmp_path / 'fake', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V10)
    with pytest.raises(ValueError, match=f'{engine}.*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V10)
    with pytest.raises(ValueError, match=f'{engine}.*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V10, engine='native', pack=tmp_path / 'any.pack')
    for k, (pay, reason) in enumerate(((dict(all_skills=False), 'an item contract of'),
                                       (dict(all_skills=True, kit_extra=RAW_FOOD_KIT), 'a start kit of none or'),
                                       (dict(all_skills=True, kit_extra=FREE_PLAY_V3['kit_extra'], inner='first-use-v1'),
                                        'an inner pay of free-play-v15'),
                                       (dict(all_skills=True, kit_extra=FREE_PLAY_V3['kit_extra'], items='item-v5'),
                                        'an item contract of'))):
        with pytest.raises(ValueError, match=reason):
            R.RealBatch([0], agents=1, starts=V2, life=('fixed', 30), key=1, output=tmp_path / f'mixed-{k}',
                        **{'items': 'item-v3', **pay})


V11 = FREE_PLAY_V11['version']


def test_free_play_v11_plays_the_wide_contract_and_v3_or_v10_weights_cross_by_the_tested_conversion_alone(tmp_path):
    """free-play-v11 (sim/workers.py FREE_PLAY_V11): free-play-v10 on the wide contract. Every life's reset names
    item-v4, its observations stamp it and mask-v6 and project to obs-v7; the batch, its summary and its checkpoint name
    v11 and item-v4, the learner's action space has act-v4's 128 rows, and its policy records obs-v7 with act-v4, mask-v6
    and item-v4. A v10 checkpoint, and a v3 one, start a v11 run by --weights through widen_state alone, recorded with
    both versions, both item contracts, the conversion and the note, every check clean with --validate. Nothing else
    crosses: v11 weights start no v10 run, v1 and v7 weights no v11 run, and a v10 checkpoint's optimizer state (--init)
    none. Leg 2 of a v11 run stays v11. An observation that stamps another item contract stops the run. The fake
    environment, the Python reference and the C engine refuse v11 by name, and a batch with item-v4 and another pay or
    kit is refused."""
    import zipfile
    from sb3_contrib import RecurrentPPO
    from baseline.policy import CONTRACT, run_contract
    from sim import train as T
    from sim.workers import CpuWorkers
    rollout = 6 * 8
    worlds = {}

    def env(name, task, leg=1, make=lambda j: Body(3)):
        batch, worlds[name] = fake(tmp_path / f'{name}-lives', key=25, life=('fixed', 30), leg=leg, make=make,
                                   **T.task_batch(task))
        return R.RealEnv(batch)
    cfgs = lambda name: [cfg for f in worlds[name] for body in f.bodies.values() for _, cfg in body.cfgs]
    for name, task in (('v1', FREE_PLAY_TASK), ('v3', V3), ('v10', V10)):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), **dict(DEVICES, task=task))
    for source, version, items in (('v10', V10, 'item-v3'), ('v3', V3, 'item-v2')):
        run = T.train(tmp_path / f'v11-{source}', decisions=2 * rollout, environment=env(f'v11-{source}', V11),
                      weights=tmp_path / source / 'latest.zip', **dict(DEVICES, task=V11))
        assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
        assert run['weights']['task_versions'] == dict(checkpoint=version, run=V11)
        assert run['weights']['note'] == T.WEIGHTS_NOTES[V11]
        assert run['weights']['item_contracts'] == dict(checkpoint=items, run='item-v4')
        assert run['weights']['conversion']['run'] == dict(observation='obs-v7', contract=run_contract('item-v4'))
        assert run['weights']['conversion']['checkpoint'] == dict(observation='obs-v6', contract=run_contract(items))
        assert (run['real']['settings']['task'], run['real']['settings']['item_contract']) == (V11, 'item-v4')
        assert run['free_play']['items'] == 'item-v4'
        assert cfgs(f'v11-{source}') and all(c['contract'] == 'item-v4' for c in cfgs(f'v11-{source}'))
    saved = RecurrentPPO.load(str(tmp_path / 'v11-v10' / 'latest.zip'), device='cpu')
    assert saved.policy.contract == run_contract('item-v4') == dict(CONTRACT, actions='act-v4', mask='mask-v6',
                                                                    items='item-v4')
    assert saved.policy.observation == 'obs-v7' and saved.action_space.n == 128
    assert saved.observation_space['sel_ids'].shape == (16,) and saved.observation_space['mask'].shape == (128,)
    with zipfile.ZipFile(tmp_path / 'v11-v10' / 'latest.zip') as archive:
        assert json.loads(archive.read('data'))['free_play_task'] == V11
    refusals = ((V10, dict(weights='v11-v10'), r"its weights never start a run of \('free-play-v10'|trained under"),
                (V11, dict(weights='v1'), r"its weights never start a run of \('free-play-v11'"),
                (V11, dict(init='v10'), 'inputs|trained under'))
    for k, (task, source, reason) in enumerate(refusals):
        environment = env(f'refused-{k}', task)
        with pytest.raises(ValueError, match=reason):
            T.train(tmp_path / f'refused-{k}', environment=environment, **{**DEVICES, 'decisions': rollout, 'task': task},
                    **{key: tmp_path / value / 'latest.zip' for key, value in source.items()})
        environment.close()
    common = dict(DEVICES, task=V11, checkpoint_every=rollout, decisions=3 * rollout)
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', V11), env_seed=5, callbacks=(preempted_after(1),), **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', V11, 2), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and all(c['problems'] == [] for c in leg2['checks']['rollouts'])
    assert (leg2['real']['settings']['task'], leg2['real']['settings']['item_contract']) == (V11, 'item-v4')
    with pytest.raises(R.RealStop, match="an observation of the item contract 'item-v2'; the run plays 'item-v4'"):
        T.train(tmp_path / 'unstamped', environment=env('unstamped', V11, make=lambda j: Unstamped(3)), **common)
    engine = f'{V11} \\(free-play-v10 on the page of 16 kinds \\(item-v4\\).*\\) is built for the real engine only'
    with pytest.raises(ValueError, match=f'{engine}.*the fake environment does not pay it'):
        T.train(tmp_path / 'fake', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V11)
    with pytest.raises(ValueError, match=f'{engine}.*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V11)
    with pytest.raises(ValueError, match=f'{engine}.*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V11, engine='native', pack=tmp_path / 'any.pack')
    for k, (pay, reason) in enumerate(((dict(all_skills=False), 'an item contract of'),
                                       (dict(all_skills=True, kit_extra=RAW_FOOD_KIT), 'a start kit of none or'),
                                       (dict(all_skills=True, kit_extra=FREE_PLAY_V3['kit_extra'], inner='first-use-v1'),
                                        'an inner pay of free-play-v15'))):
        with pytest.raises(ValueError, match=reason):
            R.RealBatch([0], agents=1, starts=V2, life=('fixed', 30), key=1, output=tmp_path / f'wide-mixed-{k}',
                        **{'items': 'item-v4', **pay})


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device for the update on the GPU; a skip '
                          'is not a pass')
DEVICES = dict(seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real', task=FREE_PLAY_TASK, validate=True,
               verbose=False)


@pytest.fixture
def cudnn_tf32():
    """The process's torch.backends.cudnn.allow_tf32, put back after a test whose run sets it (sim.train --cudnn-tf32
    sets it for the run's process and leaves it)."""
    before = torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cudnn.allow_tf32 = before


def test_a_collection_copy_takes_the_weights_after_each_update_and_collects_as_the_models_own_policy(tmp_path,
                                                                                                      monkeypatch):
    """The collection's policy calls on a copy (collect_device; here a processor copy beside a processor model, the
    hand-over a CUDA run with --collect-device cpu makes): the copy takes the model's weights at the start of each
    collection (HANDOVER), so 3 updates with --validate give the rollouts, checks, lives and weights of the model's own
    policy, bit for bit. Each update's record holds the digest the copy took (the start's, then the weights after each
    update: its checkpoint's) and the hand-over's seconds, which stay out of the update's seconds (here a hand-over made
    a second slow). The summary, the batch's settings and every checkpoint name the devices, and both runs keep the
    processor's device note; a run with no copy records none of the rest, as before. A copy that misses its hand-over
    plays old weights, which --validate (each version's rows checked with the model's own weights of that version)
    fails."""
    import zipfile
    from sim.train import CheckFailed, train
    rollout, runs, hand_over = 6 * 8, {}, R.RealBatch._hand_over

    def slow(self, model):
        time.sleep(1.0)
        return hand_over(self, model)
    monkeypatch.setattr(R.RealBatch, '_hand_over', slow)
    for name, copy in (('own', None), ('copy', 'cpu')):
        batch, _ = fake(tmp_path / f'{name}-lives', life=('fixed', 30), collect_device=copy)
        runs[name] = train(tmp_path / name, decisions=3 * rollout, environment=R.RealEnv(batch), checkpoint_every=rollout,
                           **DEVICES)
    own, copy = runs['own'], runs['copy']
    read = lambda run: [(c['problems'], c['paid_sum'], c['recurrent_max_gap'], c['reset_max_gap'], c['lagged_rows'])
                        for c in run['checks']['rollouts']]
    assert copy['status'] == 'completed' and read(own) == read(copy) and own['first_rollout'] == copy['first_rollout']
    assert own['checkpoint_digests'] == copy['checkpoint_digests'] and own['policy_sha256_at_end'] == copy['policy_sha256_at_end']
    assert lives(tmp_path / 'own-lives') == lives(tmp_path / 'copy-lives')
    digests = copy['checkpoint_digests']
    assert [u['handover'] for u in copy['real']['updates']] == \
        [copy['policy_sha256_at_start'], digests[f'ppo_{rollout}_steps.zip'], digests[f'ppo_{2 * rollout}_steps.zip']]
    assert copy['devices'] == dict(update='cpu', collection='cpu', handover=R.HANDOVER, gpu=None, cudnn_tf32=None,
                                   matmul_tf32=None)
    assert copy['real']['settings']['collect_device'] == 'cpu' and 'collect_device' not in own['real']['settings']
    assert 'devices' not in own and not any('handover' in u or 'handover_seconds' in u for u in own['real']['updates'])
    assert all(u['handover_seconds'] >= 1.0 for u in copy['real']['updates'])
    assert all(u['ppo_update_seconds'] < 1.0 for u in copy['updates'])  # the slow hand-over is not in the update's time
    assert own['real']['device_note'] == copy['real']['device_note'] == R.DEVICE_NOTE
    for name, devices in (('own', None), ('copy', copy['devices'])):
        with zipfile.ZipFile(tmp_path / name / 'latest.zip') as archive:
            assert json.loads(archive.read('data')).get('real_devices') == devices
    monkeypatch.setattr(R.RealBatch, '_hand_over', lambda self, model: 'skipped')  # the copy keeps its first weights
    batch, _ = fake(tmp_path / 'stale-lives', life=('fixed', 30), collect_device='cpu')
    with pytest.raises(CheckFailed, match='stored LSTM states reproduce the collected outputs only within') as stale:
        train(tmp_path / 'stale', decisions=3 * rollout, environment=R.RealEnv(batch), **DEVICES)
    found = re.search(r"beyond the tolerance: the (log probability|value) at step \d+, column \d+ \(\S+ collected\) differs "
                      r"by \S+, (\S+) times its tolerance .*; the row's weights are version 1, 0 updates old", str(stale.value))
    assert found and float(found[2]) >= 100  # by orders of magnitude, the row's version and age named


def test_a_cuda_run_names_its_devices_and_weights_cross_devices_recorded(tmp_path):
    """sim.train refuses a real-engine run on CUDA that does not name where its collection's calls run and cuDNN's TF32,
    the two flags anywhere else, and --validate with the calls on the GPU and TF32 on, before any output. --weights
    crosses devices both ways, recorded: a checkpoint that names a GPU run's devices starts a processor run, and a
    processor checkpoint starts a run with a collection copy; a processor checkpoint starting a processor run records no
    devices, as before. --init from the GPU run's checkpoint records both sides too; a leg from it on the processor is
    refused (a leg keeps its run's devices)."""
    from sb3_contrib import RecurrentPPO
    from sim.train import train
    refusals = ((dict(env='real', device='auto'), 'updates on cpu, cuda or cuda:N, not --device auto'),
                (dict(env='real', device='mps'), 'not --device mps'),
                (dict(env='real', device='cuda:1'), 'names where its collection'),  # cuda:N passes the gate
                (dict(env='real', device='cuda'), 'names where its collection'),
                (dict(env='real', device='cuda', collect_device='cpu'), 'names where its collection'),
                (dict(env='real', device='cpu', collect_device='cpu', cudnn_tf32='off'), 'for a real-engine run on CUDA'),
                (dict(env='fake', device='cuda', cudnn_tf32='off'), 'for a real-engine run on CUDA'),
                (dict(env='real', device='cuda', collect_device='cuda', cudnn_tf32='on'), 'give --cudnn-tf32 off'))
    for k, (change, reason) in enumerate(refusals):
        with pytest.raises(ValueError, match=reason):
            train(tmp_path / f'refused-{k}', **dict(DEVICES, decisions=48, **change))
        assert not (tmp_path / f'refused-{k}').exists()
    batch, _ = fake(tmp_path / 'cpu-lives', life=('fixed', 30))
    train(tmp_path / 'cpu', decisions=48, environment=R.RealEnv(batch), **DEVICES)
    gpu = dict(update='cuda:0', collection='cpu', handover=R.HANDOVER, gpu='a GPU', cudnn_tf32=False, matmul_tf32=False)
    model = RecurrentPPO.load(str(tmp_path / 'cpu' / 'latest.zip'), device='cpu')
    model.real_devices = gpu  # as a CUDA run of 96 decisions saves it after its first update
    model._total_timesteps, model.real_leg = 96, dict(model.real_leg, decisions=96)
    model.save(str(tmp_path / 'gpu.zip'))
    records = {}
    for name, weights, copy in (('from-gpu', tmp_path / 'gpu.zip', None), ('to-copy', tmp_path / 'cpu' / 'latest.zip', 'cpu'),
                                ('cpu-cpu', tmp_path / 'cpu' / 'latest.zip', None)):
        batch, _ = fake(tmp_path / f'{name}-lives', life=('fixed', 30), collect_device=copy)
        run = train(tmp_path / name, decisions=48, environment=R.RealEnv(batch), weights=weights, **DEVICES)
        assert run['status'] == 'completed' and run['weights']['policy_equal']
        records[name] = run['weights'].get('devices')
    assert records['from-gpu'] == dict(checkpoint=gpu, run='processor')
    assert records['to-copy'] == dict(checkpoint='processor (none recorded)', run=dict(
        update='cpu', collection='cpu', handover=R.HANDOVER, gpu=None, cudnn_tf32=None, matmul_tf32=None))
    assert records['cpu-cpu'] is None
    batch, _ = fake(tmp_path / 'init-lives', life=('fixed', 30))
    run = train(tmp_path / 'init', decisions=96, environment=R.RealEnv(batch), init=tmp_path / 'gpu.zip', **DEVICES)
    assert run['status'] == 'completed' and run['init']['devices'] == dict(checkpoint=gpu, run='processor')
    batch, _ = fake(tmp_path / 'leg-lives', life=('fixed', 30), leg=2)
    with pytest.raises(ValueError, match="on other devices .*'update': 'cuda:0'"):  # a leg keeps its run's devices
        train(tmp_path / 'leg', decisions=96, environment=R.RealEnv(batch), leg_from=tmp_path / 'gpu.zip', leg=2,
              **DEVICES)
    batch.close()


@CUDA
@pytest.mark.parametrize('collect', ('cpu', 'cuda'))
def test_a_cuda_run_updates_on_the_gpu_and_validates_with_its_collection_on_either_device(tmp_path, collect, cudnn_tf32):
    """On a CUDA device: from a processor checkpoint (--weights, recorded), 3 updates on the GPU with the collection's
    calls on the processor copy or on the GPU, --cudnn-tf32 off and --validate, every check clean; the summary names the
    GPU and the devices, and the run's checkpoint starts a processor run by --weights, recorded."""
    from sim.train import train
    batch, _ = fake(tmp_path / 'cpu-lives', life=('fixed', 30))
    train(tmp_path / 'cpu', decisions=48, environment=R.RealEnv(batch), **DEVICES)
    batch, _ = fake(tmp_path / 'gpu-lives', life=('fixed', 30), collect_device='cpu' if collect == 'cpu' else None)
    run = train(tmp_path / 'gpu', decisions=3 * 48, environment=R.RealEnv(batch), weights=tmp_path / 'cpu' / 'latest.zip',
                device='cuda', collect_device=collect, cudnn_tf32='off', **DEVICES)
    devices = run['devices']
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert devices['update'].startswith('cuda') and devices['collection'].startswith(collect) and devices['gpu']
    assert devices['cudnn_tf32'] is False and run['weights']['devices']['checkpoint'] == 'processor (none recorded)'
    assert run['real']['device_note'] == R.DEVICE_NOTE_GPU
    batch, _ = fake(tmp_path / 'back-lives', life=('fixed', 30))
    back = train(tmp_path / 'back', decisions=48, environment=R.RealEnv(batch), weights=tmp_path / 'gpu' / 'latest.zip',
                 **DEVICES)
    assert back['status'] == 'completed' and back['weights']['devices'] == dict(checkpoint=devices, run='processor')


def test_rows_three_updates_old_are_checked_with_their_own_weights_and_a_segment_too_old_is_dropped(tmp_path):
    """Agents whose decisions take 9 and 40 ticks among agents of 2-tick decisions: the slow one's segments reach their
    batch three or more updates after their first rows, the histogram counts them, --validate checks each row with the
    weights of its own version and passes; the slowest one's segments, more than eight updates old, are dropped."""
    from sim.train import train
    ticks = {4: 40, 5: 9}
    batch, _ = fake(tmp_path / 'lives', life=('fixed', 5000), make=lambda j: Body(ticks.get(j, 2)))
    run = train(tmp_path / 'run', decisions=24 * 6 * 8, seed=1, worlds=6, n_steps=8, batch_size=16, n_epochs=1, env='real',
                environment=R.RealEnv(batch), task=FREE_PLAY_TASK, validate=True, verbose=False)
    updates = run['real']['updates']
    assert all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert any(u['rows_by_age']['3'] + u['rows_by_age']['4+'] > 0 for u in updates)
    assert sum(u['dropped_segments'] for u in updates) > 0
    assert all(u['segments_per_agent'][0] == 0 for u in updates)  # the slowest agent gives none


def test_an_item_that_does_not_come_back_within_500_world_ticks_stops_the_run(tmp_path):
    """A world goes on ticking for its other agents while one agent's item takes 600 ticks: the run stops with the
    stuck agent named and its evidence kept, without waiting for the item."""
    batch, _ = fake(tmp_path, worlds=1, agents=3, life=('fixed', 5000), make=lambda j: Body(600 if j == 2 else 2))
    with pytest.raises(R.RealStop, match=r'agents \[2\] have not come back within 500') as stopped:
        batch.play(1, rounds=2000)
    assert json.loads(stopped.value.path.read_text())['world_ticks'][0] > 500
    batch.close()


# A session recorded from the real engine, game-derived data kept out of git in the ignored logs/ (the canonical checkout's
# logs/real-check/ holds it): NODE_RANDOM_EVENTS=false RL_SHARED_AGENTS=3 scripts/stack.py ... --lockstep --xp-rate 1
# --worlds 1 -- python -m sim.real play --starts village-world-v2 --life '["fixed",600]' --agents 3 --sites draynor_west
# --radius 8 --key 12 --seed 12 --rounds 4000 --until-death --transcript PATH
TRANSCRIPT = dict(path='logs/real-check/real-transcript.jsonl.gz', agents=3, starts=V2, life=('fixed', 600), key=12,
                  sites=['draynor_west'], radius=8, seed=12, rounds=4000)


class Replay:
    """The recorded engine of TRANSCRIPT on a local port: each request must equal the next recorded one (else the reply
    is an error, as the engine refuses a request), and is answered with the recorded reply."""

    def __init__(self, path):
        import gzip
        from pathlib import Path
        rows = [json.loads(line) for line in gzip.open(Path(__file__).resolve().parents[2] / path, 'rt')]
        self.pairs, k = [], 0
        while k < len(rows):
            request, reply = rows[k]['request'], rows[k + 1].get('reply') if k + 1 < len(rows) else None
            self.pairs.append((request, reply))
            k += 2 if reply is not None else 1
        self.server = socket.create_server(('127.0.0.1', 0))
        self.port, self.served, self.differed = self.server.getsockname()[1], 0, None
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        conn, _ = self.server.accept()
        send = lambda msg: conn.sendall(('PROTO|' + json.dumps(msg) + '\n').encode())
        send(dict(type='ready', protocol='shared-lockstep-v1', capacity=3, worldId='d1'))
        with conn, conn.makefile('rb') as lines:
            for line in lines:
                request, (recorded, reply) = json.loads(line), self.pairs[self.served]
                if request != recorded:
                    self.differed = (self.served, request, recorded)
                    send(dict(type='error', code='bad-request', message=f'request {self.served} differs from the record'))
                    return
                self.served += 1
                if reply is None:
                    return
                send(reply)


def test_the_client_plays_a_session_recorded_from_the_real_engine_back_request_for_request():
    """A session recorded from the real engine (sim.real play --transcript: one world of three agents at draynor_west,
    random legal play until a death and a cut had passed their ledger check, then every live agent left): the batch
    sends the recorded requests in their order (hello, places, joins by reset with the setup, steps, the leaves, close)
    and reads the engine's own replies as the engine wrote them: pay in tenths and the running ledger at every result,
    a death with its killer, three cut lives and three left at the end, each closing ledger a match."""
    import tempfile
    from pathlib import Path
    if not (Path(__file__).resolve().parents[2] / TRANSCRIPT['path']).exists():
        pytest.skip(f"{TRANSCRIPT['path']} is absent (private, game-derived: copy it from the canonical checkout's logs/); "
                    'a skip is not a pass')
    t, replay = TRANSCRIPT, Replay(TRANSCRIPT['path'])
    with tempfile.TemporaryDirectory() as out:
        batch = R.RealBatch([replay.port], agents=t['agents'], starts=t['starts'], life=t['life'], key=t['key'],
                            output=out, sites=t['sites'], radius=t['radius'])
        played = batch.play(t['seed'], rounds=t['rounds'], until_death=True)
        rows = lives(Path(out))
    assert replay.differed is None and replay.served == len(replay.pairs)
    final = played['final']
    assert (played['decisions'], played['lives'], played['deaths'], played['cut']) == (155, 4, 1, 3)
    assert (final['left'], final['ledger_checks'], final['running_checks']) == (3, 7, 157)
    assert [r['ledger'] for r in rows] == ['match'] * 7 and sum(r['grants'] for r in rows) == 15
    assert [(r['agent'], r['killers']) for r in rows if r['terminal'] == 'death'] == [(0, ['Dark wizard'])]


@WORKERS
def test_a_failed_reset_gives_no_row_and_its_agent_starts_again_from_its_next_record(tmp_path, workers):
    """Agent 1's first join and agent 2's second reset fail in their setup: neither gives a row or a decision; each is
    written to failed-resets.jsonl and counted; agent 2's failed reset still closes its first life, whose ledger is
    checked; each agent starts again from its next start record, and the run goes on."""
    batch, _ = fake(tmp_path, worlds=3, agents=3, life=('fixed', 30), workers=workers, make=lambda j: Body(3),
                    world=lambda w: dict(failures={(1, 0), (2, 1)} if w == 0 else ()))
    collect(learner(batch, n_steps=8), rollouts=2)
    failed = [json.loads(line) for line in (tmp_path / 'failed-resets.jsonl').read_text().splitlines()]
    assert [(f['agent'], f['life'], f['record'], f['code']) for f in failed] == [(1, 0, 0, 'setup'), (2, 1, 1, 'setup')]
    rows = lives(tmp_path)
    assert all((r['agent'], r['life']) not in {(1, 0), (2, 1)} for r in rows)
    assert (2, 0) in {(r['agent'], r['life']) for r in rows}  # the life the failed reset closed, checked
    assert next(r for r in rows if r['agent'] == 1)['record'] == 1 and batch.counts['failed_resets'] == 2
    assert batch.counts['resets'] > batch.counts['failed_resets'] and batch.updates[0]['failed_resets'] >= 1
    batch.close()


def test_five_failed_resets_of_one_agent_in_a_row_stop_the_run(tmp_path):
    """An agent whose resets fail five times in a row stops the run, with its evidence, before any update."""
    fakes = [FakeWorld(0, lambda k: Body(3), failures={(0, n) for n in range(5)})]
    batch = R.RealBatch([fakes[0].port], agents=2, starts=V2, life=('fixed', 30), key=5, output=tmp_path, stall=30.0, radius=2)
    with pytest.raises(R.RealStop, match='5 resets in a row failed') as stopped:
        collect(learner(batch, n_steps=8))
    assert json.loads(stopped.value.path.read_text())['agent'] == 0 and not batch.updates
    batch.close()


@WORKERS
def test_a_failed_check_at_the_runs_end_leaves_the_saved_weights_and_an_incomplete_run(tmp_path, workers):
    """Leaves whose closing ledger differs from the pay: the run has made its two updates, recorded them and saved
    latest.zip before the leaves, and ends failed with the reason, not complete."""
    from sim.train import train
    batch, _ = fake(tmp_path / 'lives', worlds=3, life=('fixed', 400), workers=workers, make=lambda j: Body(2),
                    world=lambda w: dict(bad_leave=True))
    with pytest.raises(R.RealStop, match='pay check failed'):
        train(tmp_path / 'run', decisions=2 * 9 * 8, seed=1, worlds=9, n_steps=8, batch_size=16, n_epochs=1, env='real',
              environment=R.RealEnv(batch), task=FREE_PLAY_TASK, verbose=False)
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['status'] == 'failed' and 'pay check failed' in summary['error'] and 'real_finish' not in summary
    assert (tmp_path / 'run' / 'latest.zip').exists() and len(summary['updates']) == 2 and summary['policy_sha256_at_end']


@pytest.mark.parametrize('workers', (0, 2))
def test_the_worlds_are_pinged_while_the_learner_works_between_collections(tmp_path, monkeypatch, workers):
    """Between two collections (the learner's update, its checks, a checkpoint) every world gets a ping now and then,
    from this process or from the workers, and the next collection reads its replies as before."""
    monkeypatch.setitem(R.REAL, 'ping_seconds', 0.05)
    fakes = [FakeWorld(w, lambda k, w=w: Body(2 + (3 * w + k) % 3)) for w in range(2)]
    batch = R.RealBatch([f.port for f in fakes], agents=3, starts=V2, life=('fixed', 30), key=5, output=tmp_path,
                        stall=30.0, radius=2, workers=workers)
    model = learner(batch, n_steps=8)
    collect(model)
    time.sleep(0.5)
    assert all(f.pings > 0 for f in fakes) and batch.pings > 0
    collect(model)
    assert len(batch.updates) == 2 and batch.ping_error is None
    batch.close()


def test_the_collections_policy_threads_change_no_row_and_are_restored_after_it(tmp_path):
    """Two worlds of 32 agents, torch at 4 threads, collected twice from the same start: once as it is and once with
    policy_threads=1. Every policy call of the second ran on one thread, the four were back when the collection
    returned, and the rows are the same bit for bit (actions, log-probabilities, values, states, every info field:
    the policy's operations on these shapes compute each output element in one thread whatever the count)."""
    from sim.train import rollout_digest
    before, runs = torch.get_num_threads(), []
    torch.set_num_threads(4)
    try:
        for threads in (None, 1):
            fakes = [FakeWorld(w, lambda k, w=w: Body(2 + (32 * w + k) % 3)) for w in range(2)]
            batch = R.RealBatch([f.port for f in fakes], agents=32, starts=V2, life=('fixed', 30), key=5, stall=30.0,
                                radius=2, output=tmp_path / f'threads{threads}', policy_threads=threads)
            model, seen = learner(batch), []
            forward = model.policy.forward
            model.policy.forward = lambda *args, **kwargs: (seen.append(torch.get_num_threads()), forward(*args, **kwargs))[1]
            digests = [rollout_digest(collect(model)) for _ in range(2)]
            runs.append((digests, lives(tmp_path / f'threads{threads}')))
            assert set(seen) == {threads or 4} and torch.get_num_threads() == 4
            assert all(u['threads_before']['torch'] == 4 for u in batch.updates) and batch.updates[0]['threads_after']['torch'] == 4
            assert batch.settings_in_effect['policy_threads'] == threads
            batch.close()
    finally:
        torch.set_num_threads(before)
    assert runs[0] == runs[1]


def test_a_worker_that_ends_stops_the_run_with_its_evidence_and_close_leaves_no_process(tmp_path):
    """Two workers over three worlds; one is killed between two collections: the next collection stops the run with
    its reason in real-stop.json, and close ends every worker process."""
    batch, _ = fake(tmp_path, worlds=3, life=('fixed', 30), workers=2)
    assert batch.settings_in_effect['workers'] == dict(asked=2, running=2)
    model = learner(batch)
    collect(model)
    processes = list(batch.transport.processes)
    processes[0].kill()
    processes[0].join(timeout=10)
    with pytest.raises(R.RealStop, match='worker') as stopped:
        collect(model)
    assert 'worker' in json.loads(stopped.value.path.read_text())['reason']
    batch.close()
    assert not any(p.is_alive() for p in processes)


@pytest.mark.parametrize('workers', (0, 1))
def test_an_observation_that_does_not_project_stops_the_run_naming_its_agent(tmp_path, workers):
    """Slot 2 of world 1 returns an observation without its player at its third step: the projection fails, in this
    process or in a worker, and the run stops naming that agent, with its result in real-stop.json."""
    batch, _ = fake(tmp_path, worlds=2, life=('fixed', 400), workers=workers, world=lambda w: dict(broken={(2, 3)} if w else ()))
    with pytest.raises(R.RealStop, match='does not project') as stopped:
        collect(learner(batch))
    evidence = json.loads(stopped.value.path.read_text())
    assert (evidence['agent'], evidence['world'], evidence['slot']) == (5, 1, 2) and 'player' not in evidence['result']['obs']
    batch.close()


class Telemetry:
    """A demo world's local telemetry port (env-ts/demo-services.ts) beside a FakeWorld: GET /status with the declared
    spectator present and `players` as playerStates; POST /demo-view keeps each view with the number of items the world
    had seen then (0: before any agent joined) and of status reads before it (0: before the demo waited for its
    spectator), and refuses a body without a tile or an agent to follow."""

    def __init__(self, world, players):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.views, self.looks, self.present = [], 0, True
        self.audit = dict(refusedLogins=0, droppedPackets={}, moves=1, collisionRestores=0, spectatorOnlyNpcTicks=4, views=0)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.looks += 1
                view = outer.views[-1][0] if outer.views else None
                self.reply(200, dict(schema=1, tick=world.tick, playerStates=players, demo=dict(spectator=dict(
                    declared='viewer', present=[dict(name='viewer')] if outer.present else [], view=view, **outer.audit))))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not set(body) & {'tile', 'follow'}:
                    return self.reply(400, dict(error='a view names a tile, an agent to follow or both'))
                outer.views.append((body, len(world.log), outer.looks))
                outer.audit['views'] += 1
                if 'tile' in body:  # the spectator stands there from the next tick
                    me = next(p for p in players if p['name'] == 'viewer')
                    me['x'], me['z'] = body['tile']
                self.reply(200, dict(view=body, tick=world.tick, audit=dict(outer.audit)))

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/status'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def demo_world(tmp_path, monkeypatch, agents=6, **options):
    """A saved free-play checkpoint of the real engine (learner over the fake batch) and its launch.json, a FakeWorld of
    `agents` agents with its Telemetry, and a demo world's environment of scripts/stack.py; sim/real.py's ROOT is
    tmp_path, so its demos write in tmp_path/logs/demo."""
    from sim import watch
    from sim.tests.test_watch import TREE
    from baseline import config
    batch, _ = fake(tmp_path / 'trained-lives')
    model = learner(batch)
    checkpoint = tmp_path / 'latest.zip'
    model.save(str(checkpoint))
    model.env.close()
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps(dict(name='real-run', xp_rate=1, command=[
        'python', '-m', 'sim.train', 'train', '--env', 'real', '--task', FREE_PLAY_TASK, '--seed', '303', '--batch-setting',
        f'starts="{V2}"', 'life=["fixed",40]', f'agents={agents}', '--output', '/tmp/run/free'])))
    world = FakeWorld(0, lambda k: Body(2 + k % 3), **options)
    falador, port_sarim = watch.SITES['falador_farm_west'], watch.SITES['port_sarim_north']
    players = [dict(name='viewer', x=falador[0], z=falador[1], level=0)] + \
        [dict(name=f'wd1s{k}', x=falador[0] + 3 * k, z=falador[1], level=0) for k in range(4)] + \
        [dict(name=f'wd1s{k}', x=port_sarim[0], z=port_sarim[1] + k, level=0) for k in (4, 5)]
    telemetry = Telemetry(world, players)
    monkeypatch.setattr(R, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'XP_RATE', 1)
    monkeypatch.setattr(watch, 'checkout_tree', lambda: TREE)
    environ = dict(RL_DEMO_SPECTATOR='viewer', RL_DEMO_PACE_MS='5', RL_SHARED_AGENTS=str(agents), NODE_RANDOM_EVENTS='false',
                   RL_WORLDS=json.dumps([dict(env_port=world.port, status_url=telemetry.url)]))
    return dict(checkpoint=checkpoint, launch=launch, world=world, telemetry=telemetry, environ=environ)


def test_a_demo_plays_one_checkpoint_with_many_agents_in_one_demo_world_watched_and_labelled(tmp_path, monkeypatch):
    """sim.real demo over the fake engine: the spectator's first view is set before any agent joins (the first site's
    tile, and the agent to follow), the checkpoint plays every agent of the one world, sampled from its own recurrent
    state, from both sites; every agent leaves at the end through the ledger checks; demo.json, lives.jsonl and
    views.jsonl are labelled a demo in every row, and the looks count the agents the spectator can see (15 tiles) and
    those near each site."""
    d = demo_world(tmp_path, monkeypatch)
    monkeypatch.setitem(R.DEMO, 'view_every', 0.1)
    out = tmp_path / 'logs' / 'demo' / 'rehearsal' / 'play'
    record = R.demo(out, checkpoint=d['checkpoint'], launch=d['launch'], starts=V2, sites=['falador_farm_west',
                    'port_sarim_north'], key=5, seed=9, seconds=1.0, agents=6, radius=2, follow=3, environ=d['environ'],
                    say=lambda *a, **k: None)
    from sim import watch
    assert d['telemetry'].views == [(dict(tile=list(watch.SITES['falador_farm_west']), follow=3), 0, 0)]
    assert record['status'] == 'complete' and record['label'] == R.DEMO['label'] == 'demo: not evaluation evidence'
    saved = json.loads((out / 'demo.json').read_text())
    assert saved['status'] == 'complete' and saved['label'] == R.DEMO['label'] and saved['settings_in_effect']['demo'] == R.DEMO['label']
    assert (saved['settings']['decoding'], saved['settings']['life'], saved['policy']['run']['engine']) == ('sampled', ['fixed', 40], 'real')
    audit = dict(refusedLogins=0, droppedPackets={}, moves=1, collisionRestores=0, spectatorOnlyNpcTicks=4, views=1)
    assert saved['first_view_reply']['audit'] == audit and saved['spectator_audit'] == audit
    joined = sorted({slot for _, slot, op, _ in d['world'].log if op == 'reset'})
    assert joined == list(range(6)) and record['decisions'] > 6 and record['policy_seconds'] > 0
    final = record['final']  # every agent left the world; each life it ended passed its ledger check
    assert d['world'].lives == {} and 0 < final['left'] <= 6 and final['ledger_checks'] == final['lives'] + final['left']
    rows = lives(out)
    assert rows and all(r['demo'] == R.DEMO['label'] and r['ledger'] == 'match' for r in rows)
    assert {r['site'] for r in rows} == {'falador_farm_west', 'port_sarim_north'}
    looks = [json.loads(line) for line in (out / 'views.jsonl').read_text().splitlines()]
    assert len(looks) >= 2 and all(v['demo'] == R.DEMO['label'] and v['kind'] == 'view' for v in looks)
    assert all((v['agents'], v['in_view'], v['near_site'], v['audit'], v['wakes_island_npcs']) == (
        6, 4, dict(falador_farm_west=4, port_sarim_north=2), audit, False) for v in looks)
    rises = [json.loads(line) for line in (out / 'xp.jsonl').read_text().splitlines()]  # the fake grants GAIN a decision
    assert rises and all(r['demo'] == R.DEMO['label'] for r in rises)
    gain = {k.lower(): v for k, v in GAIN.items()}  # the batch names stats as the engine's hello does
    assert {r['stat'] for r in rises} == set(gain) and all(r['tenths'] == gain[r['stat']] * (1 + r['life']) for r in rises)
    assert record['world_ticks'] > 0 and record['wall_seconds'] >= record['seconds']
    assert record['ticks_per_second'] == pytest.approx(record['world_ticks'] / record['wall_seconds'], rel=0.02)


def test_a_demo_refuses_another_folder_a_world_that_is_no_demo_and_its_world_refuses_training(tmp_path, monkeypatch):
    """Refused before any view is set or any agent joins: an output outside logs/demo/ or one that exists, a stack that
    is no demo world or has two worlds, a site outside SITES, an agent to follow outside the world and start levels the
    watch's parser refuses. A demo world
    refuses `play` and training (check_stack), and the demo refuses a stack that is no demo world."""
    from sim import watch
    d = demo_world(tmp_path, monkeypatch)
    common = dict(checkpoint=d['checkpoint'], launch=d['launch'], starts=V2, sites=['falador_farm_west'], key=5, seed=9,
                  seconds=1.0, agents=6, say=lambda *a, **k: None)
    (tmp_path / 'logs' / 'demo' / 'taken').mkdir(parents=True)
    two = dict(d['environ'], RL_WORLDS=json.dumps([dict(env_port=1, status_url='x')] * 2))
    no_demo = {k: v for k, v in d['environ'].items() if not k.startswith('RL_DEMO')}
    refusals = ((tmp_path / 'logs' / 'real-modal' / 'x', {}, d['environ'], 'writes inside'),
                (tmp_path / 'logs' / 'demo', {}, d['environ'], 'writes inside'),
                (tmp_path / 'logs' / 'demo' / 'taken', {}, d['environ'], 'exists'),
                (tmp_path / 'logs' / 'demo' / 'a', {}, no_demo, '--demo-spectator NAME --demo-pace-ms MS'),
                (tmp_path / 'logs' / 'demo' / 'b', {}, two, 'one world'),
                (tmp_path / 'logs' / 'demo' / 'c', dict(sites=['rimmington_check']), d['environ'], 'SITES'),
                (tmp_path / 'logs' / 'demo' / 'e', dict(start_levels='woodcutting=50,woodcutting=40'), d['environ'],
                 'named twice'),
                (tmp_path / 'logs' / 'demo' / 'd', dict(follow=6), d['environ'], 'slot in 0..5'))
    for output, change, environ, reason in refusals:
        with pytest.raises(ValueError, match=reason):
            R.demo(output, **dict(common, **change), environ=environ)
    assert d['telemetry'].views == [] and d['world'].log == []
    assert sorted(p.name for p in (tmp_path / 'logs' / 'demo').iterdir()) == ['taken']
    for demo, environ, reason in ((False, d['environ'], 'serves only sim.real demo'),
                                  (True, no_demo, 'runs inside scripts/stack.py --demo-spectator')):
        with pytest.raises(ValueError, match=reason):
            R.check_stack(environ, 6, demo=demo)
    from sim.tests.test_watch import TREE
    stack = R.check_stack(no_demo, 6)
    assert (stack['shared_agents'], stack['engine_build']) == (6, dict(name='pinned', tree=TREE))
    assert [name for name, (x, z) in watch.SITES.items() if R.wakes_island_npcs(x, z)] == [
        'rimmington_coast', 'rimmington_mine_west', 'rimmington_shore_east', 'rimmington_shore_west']  # views recorded
    monkeypatch.setattr('sys.argv', ['sim.real', 'demo', '--output', str(tmp_path / 'elsewhere'), '--real-checkpoint',
                                     str(d['checkpoint']), '--launch', str(d['launch']), '--starts', V2, '--sites',
                                     'falador_farm_west', '--key', '5', '--seed', '9', '--seconds', '1'])
    with pytest.raises(SystemExit, match=r'\[demo\] a demo writes inside'):
        R.main()


def test_a_demo_plays_every_agent_at_the_watchs_profile_at_a_rimmington_site_and_the_watch_reads_its_traced_lives(
        tmp_path, monkeypatch):
    """sim.real demo --start-levels --trace at rimmington_mine_west: the first view stands at the site before any agent
    joins; every agent (not only the odd ones of old-life-v1) starts every life at the profile, each record as the watch's
    profiled gives it, armed under old-life-v1's level check; the trace holds each result's observation, and demo_reads
    gives the watch's life_reads of each closed life: its ticks (a horizon at the life's length, a cut at its last
    result), its decisions, the levels it armed at and its time by activity, which sums to its ticks. A profile outside a
    demo, or beside a start rule, is refused."""
    from baseline.world import read_jsonl
    from sim import watch
    from sim.workers import OLD_LIFE
    d = demo_world(tmp_path, monkeypatch, agents=2)
    text = 'woodcutting=50,firemaking=52,attack=31,strength=37,defence=30,hitpoints=34'
    profile = watch.parse_levels(text)
    out = tmp_path / 'logs' / 'demo' / 'profile' / 'play'
    record = R.demo(out, checkpoint=d['checkpoint'], launch=d['launch'], starts=V2, sites=['rimmington_mine_west'],
                    key=5, seed=9, seconds=1.0, agents=2, radius=2, follow=0, start_levels=text, trace=True,
                    environ=d['environ'], say=lambda *a, **k: None)
    assert record['status'] == 'complete'
    assert d['telemetry'].views[0] == (dict(tile=list(watch.SITES['rimmington_mine_west']), follow=0), 0, 0)
    saved = json.loads((out / 'demo.json').read_text())
    assert saved['settings']['start_levels'] == profile and saved['settings']['trace'] == 'observations.jsonl.gz'
    assert (saved['settings_in_effect']['profile'], saved['settings_in_effect']['profile_agents']) == (
        profile, 'every agent (a demo)')
    for slot in (0, 1):  # every reset of both agents names the profiled record's levels, the rest as drawn
        records = watch.Records(5, slot, 'danger')
        cfgs = [cfg for _, cfg in d['world'].bodies[slot].cfgs]
        assert len(cfgs) >= 2 and all(cfg['setStats'] == watch.profiled(records.next(), profile)['set_stats']
                                      for cfg in cfgs)
    rows = lives(out)
    assert rows and all(r['profile'] and r['start_levels'] == {**r['start_levels'], **profile} for r in rows)
    traced = list(read_jsonl(out / 'observations.jsonl.gz'))
    assert all(r['demo'] == R.DEMO['label'] and r['obs'] is not None for r in traced)
    arms = [r for r in traced if r['result'] == 'reset']
    final = record['final']  # the results that came back while the agents left are traced too
    assert len(arms) == final['resets'] and all(r['action'] is None for r in arms)
    assert len(traced) - len(arms) == final['decisions'] > record['decisions'] - 2
    reads = R.demo_reads(out)
    assert json.loads((out / 'reads.json').read_text())['demo'] == R.DEMO['label']
    closed = {(r['agent'], r['life']): r for r in rows}
    assert set(reads) == {f'{a}:{life}' for a, life in closed} and {r['terminal'] for r in rows} >= {'horizon', 'left'}
    for name, read in reads.items():
        row = closed[tuple(map(int, name.split(':')))]
        assert read['ticks'] == (row['length'] if row['terminal'] == 'horizon' else row['ticks'])
        assert read['decisions'] == read['decisions_traced'] == row['decisions']
        assert sum(read['time'].values()) == read['ticks'] and set(read['time']) <= {'fight', 'woodcutting', 'none'}
        assert {s: read['levels0'][s] for s in profile if s in read['levels0']} == {  # the watch's seven old skills
            s: profile[s] for s in ('attack', 'defence', 'strength', 'hitpoints', 'woodcutting')}
        assert read['made'] is None and read['kills'] is None and read['bank']['deposits'] is None
    for change in (dict(), dict(start_rule=OLD_LIFE, label='demo')):
        with pytest.raises(ValueError, match='a demo\'s alone'):
            R.RealBatch([d['world'].port], agents=2, starts=V2, life=['fixed', 40], key=5, output=tmp_path / 'x',
                        profile=profile, **change)


def test_the_busiest_view_holds_the_most_agents_and_the_view_command_moves_the_spectator_there(tmp_path, monkeypatch):
    """busiest: the level-0 tile whose 15-tile view holds the most agents (the spectator and other levels left out).
    sim.real view --busiest moves the spectator there only when that view holds more agents than its own, --follow
    and --tile set those views, and --every without --busiest is refused."""
    group = [dict(name=f'wd1s{k}', x=2920 + k, z=3300 + k % 3, level=0) for k in range(10)]
    pair = [dict(name=f'wd1s{10 + k}', x=3045 + k, z=3260, level=0) for k in range(5)]
    upstairs = [dict(name=f'wd1s{15 + k}', x=3045, z=3260, level=1) for k in range(8)]
    players = [dict(name='viewer', x=3002, z=3279, level=0)] + group + pair + upstairs
    status = dict(playerStates=players)
    at, count = R.busiest(status, 'viewer')
    assert count == 10 and max(abs(at[0] - 2924), abs(at[1] - 3301)) <= 1  # in the middle of the group, not at an edge
    assert R.busiest(dict(playerStates=players[:1] + upstairs), 'viewer') == (None, 0)
    island = [dict(name=f'wd1s{30 + k}', x=3100 + k, z=3150, level=0) for k in range(12)]  # the world refuses views there
    other, seen = R.busiest(dict(playerStates=island + pair), 'viewer')
    assert seen == 5 and not R.in_island_chunks(*other) and R.in_island_chunks(3100, 3150)
    world = FakeWorld(0, lambda k: Body(2))
    telemetry = Telemetry(world, players)
    said = []
    assert R.view(telemetry.url, crowd=True, say=lambda *a, **k: said.append(a[0])) == (at, 10)
    assert telemetry.views[-1][0] == dict(tile=list(at)) and said == [f'[view] 10 agents in view at {at[0]},{at[1]} (tick 100)']
    assert R.view(telemetry.url, crowd=True, say=said.append) == (at, 10) and len(telemetry.views) == 1  # already there
    R.view(telemetry.url, follow=12)
    R.view(telemetry.url, tile=(3050, 3270))
    assert [v for v, _, _ in telemetry.views[1:]] == [dict(follow=12), dict(tile=[3050, 3270])]
    for argv, reason in ((['--busiest', '--every', '0'], 'one of a tile'), (['--follow', '3', '--every', '5'], 'one of a tile')):
        monkeypatch.setattr('sys.argv', ['sim.real', 'view', '--status', telemetry.url, *argv])
        with pytest.raises(SystemExit, match=reason):
            R.main()
    assert len(telemetry.views) == 3


def test_a_demo_that_fails_before_it_plays_says_so_in_its_record_with_the_cause(tmp_path, monkeypatch):
    """demo.json, once written, ends with the failure and its cause: a spectator who never logs in (the wait's
    timeout) or a world whose hello the batch refuses (random events on); the spectator's counts are kept too."""
    d = demo_world(tmp_path, monkeypatch)
    common = dict(checkpoint=d['checkpoint'], launch=d['launch'], starts=V2, sites=['falador_farm_west'], key=5, seed=9,
                  seconds=1.0, agents=6, say=lambda *a, **k: None)
    d['telemetry'].present = False
    with pytest.raises(TimeoutError):
        R.demo(tmp_path / 'logs' / 'demo' / 'alone', **common, wait=0.3, environ=d['environ'])
    alone = json.loads((tmp_path / 'logs' / 'demo' / 'alone' / 'demo.json').read_text())
    assert (alone['status'], alone['cause'].split(':')[0]) == ('failed', 'TimeoutError') and 'world_ticks' not in alone
    assert alone['spectator_audit']['spectatorOnlyNpcTicks'] == 4 and alone['label'] == R.DEMO['label']
    d['telemetry'].present = True
    on = FakeWorld(0, lambda k: Body(2), random_events=True)
    events = dict(d['environ'], RL_WORLDS=json.dumps([dict(env_port=on.port, status_url=d['telemetry'].url)]))
    with pytest.raises(ValueError, match='random events True'):
        R.demo(tmp_path / 'logs' / 'demo' / 'events', **common, environ=events)
    refused = json.loads((tmp_path / 'logs' / 'demo' / 'events' / 'demo.json').read_text())
    assert refused['status'] == 'failed' and refused['cause'].startswith('ValueError: world 0 plays at XP rate 1 with random events True')


def test_a_demo_plays_a_free_play_v3_checkpoint_with_its_saved_kit(tmp_path, monkeypatch):
    """A free-play-v3 checkpoint (it saves its task, its paid stats and its kit) plays in a demo with its kit: every
    reset's kit holds the record's items and one tinderbox, and the demo's records name the task and the kit; the
    free-play-v1 checkpoint's demo keeps the record's kit alone."""
    from sb3_contrib import RecurrentPPO
    d = demo_world(tmp_path, monkeypatch)
    model = RecurrentPPO.load(str(d['checkpoint']), device='cpu')
    model.__dict__.update(free_play_task=V3, paid_skills=[n.lower() for n in SKILL_ORDER], kit_extra=[['tinderbox', 1]])
    model.save(str(tmp_path / 'v3.zip'))
    launch = json.loads(d['launch'].read_text())
    launch['command'][launch['command'].index('--task') + 1] = V3
    (tmp_path / 'v3-launch.json').write_text(json.dumps(launch))
    common = dict(starts=V2, sites=['falador_farm_west'], key=5, seed=9, seconds=0.5, agents=6, radius=2,
                  environ=d['environ'], say=lambda *a, **k: None)
    record = R.demo(tmp_path / 'logs' / 'demo' / 'v3', checkpoint=tmp_path / 'v3.zip', launch=tmp_path / 'v3-launch.json',
                    **common)
    kits = [cfg['seedItems'] for body in d['world'].bodies.values() for _, cfg in body.cfgs]
    assert record['status'] == 'complete' and kits and all(k.endswith(',tinderbox:1') and k.count('tinderbox') == 1 for k in kits)
    assert (record['settings']['task'], record['settings']['kit_extra']) == (V3, [['tinderbox', 1]])
    assert (record['settings_in_effect']['task'], record['settings_in_effect']['kit_extra']) == (V3, [['tinderbox', 1]])
    plain = demo_world(tmp_path / 'v1', monkeypatch)
    again = R.demo(tmp_path / 'v1' / 'logs' / 'demo' / 'v1', checkpoint=plain['checkpoint'], launch=plain['launch'],
                   **dict(common, environ=plain['environ']))
    kits = [cfg['seedItems'] for body in plain['world'].bodies.values() for _, cfg in body.cfgs]
    assert again['status'] == 'complete' and kits and not any('tinderbox' in k for k in kits)
    assert (again['settings']['task'], again['settings']['kit_extra']) == (FREE_PLAY_TASK, [])



# ---- free-play-v12: the drawn start hitpoints -------------------------------------------------------------------------
V12 = FREE_PLAY_V12['version']
# reset_cfgs' digests of free-play-v1, v3 and v10, recorded before free-play-v12: their resets as they were
OLD_RESETS = {'free-play-v1': 'e75176a3395bdca2572bcfacd528f8863f34235ebdbae64921d46e907851e25b',
              'free-play-v3': '511b69334e2a5d3474ec1d80c8a12e2c8311ee04929ecf661066689aa1daba1c',
              'free-play-v10': 'cae92d32b7022de23072e1d47a61706800f2b64d768419f6357187ebda081993'}


def reset_cfgs(tmp_path, task, hurt=False, make=lambda j: Body(3)):
    """A fake batch of `task` (key 25, 2 worlds of 2 agents, 30-tick lives) played with WAIT for 40 rounds: (its life rows,
    its reset cfgs slot by slot, their sha256 as sorted JSON)."""
    import hashlib
    from sim import train as T
    batch, fakes = fake(tmp_path / task, worlds=2, agents=2, key=25, life=('fixed', 30), make=make,
                        world=lambda w: dict(food=True, hurt=hurt), **T.task_batch(task))
    batch.play(1, actor='wait', rounds=40)
    cfgs = [cfg for f in fakes for slot in sorted(f.bodies) for _, cfg in f.bodies[slot].cfgs]
    return lives(tmp_path / task), cfgs, hashlib.sha256(json.dumps(cfgs, sort_keys=True).encode()).hexdigest()


def test_free_play_v12_starts_each_life_at_its_drawn_hitpoints_and_the_other_versions_as_before(tmp_path):
    """Every reset of a free-play-v12 batch names its start record's drawn hitpoints (startHp; the danger group
    draws 0 to 3 under the base), and each life starts at them (start_hp, the engine's hitpoints at the arm, read in
    every life row when the engine reports hurt). The resets of free-play-v1, v3 and v10 are byte for byte those
    recorded before v12 (OLD_RESETS): no start hitpoints."""
    for task, digest in OLD_RESETS.items():
        _, cfgs, found = reset_cfgs(tmp_path / 'old', task)
        assert found == digest and cfgs and not any('startHp' in c for c in cfgs)
    group = R.watch.world_rules(V2)['start_group']
    for task in (V12,):
        rows, cfgs, _ = reset_cfgs(tmp_path, task, hurt=True)
        by_agent = {}  # reset_cfgs lists the cfgs world by world, slot by slot: agent j is world j // 2, slot j % 2
        for cfg in cfgs:
            by_agent.setdefault(len(by_agent), cfg)
        lives_ = {(r['agent'], r['life']): r for r in rows}
        for j in range(4):
            stream = R.watch.Records(25, j, group)
            for n in range(sum(r['agent'] == j for r in rows)):
                drawn = stream.next()['hp_drawn']
                assert lives_[j, n]['start_hp'][0] == drawn <= lives_[j, n]['start_hp'][1]
        assert cfgs and all(type(c.get('startHp')) is int and c['startHp'] >= 1 for c in cfgs)
        assert any(r['start_hp'][0] < r['start_hp'][1] for r in rows) and all('hurt_ticks' in r for r in rows)


def test_free_play_v12_crosses_from_v10_alone_refuses_other_engines_and_a_v12_leg_goes_on(tmp_path):
    """--weights: a v10 checkpoint starts a v12 run, recorded with both versions and the note, every check clean with
    --validate. Nothing else crosses: v12 weights start no v10 run and v3 weights no v12 run. Leg 2 of a v12 run stays
    v12. The fake environment, the Python reference and the C engine refuse v12 by name, and the batch refuses the drawn
    hitpoints off the page of kinds and with a pay that no kept version has. A checkpoint of each removed study arm
    (sim/workers.py REMOVED) is refused by name for --weights, and one for --init and for a leg, before any update."""
    import zipfile
    from sb3_contrib import RecurrentPPO
    from sim import train as T
    from sim.workers import CpuWorkers, REMOVED
    rollout = 6 * 8

    def env(name, task, leg=1):
        return R.RealEnv(fake(tmp_path / f'{name}-lives', key=25, life=('fixed', 30), leg=leg,
                              world=lambda w: dict(food=True, hurt=True),
                              make=lambda j: Body(3, hits=HITS.get(j % 4), death=DEATHS.get(j % 4)),
                              **T.task_batch(task))[0])
    for name, task in (('v10', V10), ('v3', V3)):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), **dict(DEVICES, task=task))
    run = T.train(tmp_path / 'v12', decisions=2 * rollout, environment=env('v12', V12),
                  weights=tmp_path / 'v10' / 'latest.zip', **dict(DEVICES, task=V12))
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=V10, run=V12)
    assert run['weights']['note'] == T.WEIGHTS_NOTES[V12] and run['real']['settings']['start_hp'] == 'drawn'
    assert run['free_play']['start_hp'] == 'drawn' and run['real']['settings']['task'] == V12
    with zipfile.ZipFile(tmp_path / 'v12' / 'latest.zip') as archive:
        assert json.loads(archive.read('data'))['free_play_task'] == V12
    for k, (task, source) in enumerate(((V10, 'v12'), (V12, 'v3'))):
        environment = env(f'refused-{k}', task)
        with pytest.raises(ValueError, match=rf"never start a run of \('{task}'"):
            T.train(tmp_path / f'refused-{k}', environment=environment, weights=tmp_path / source / 'latest.zip',
                    **{**DEVICES, 'decisions': rollout, 'task': task})
        environment.close()
    model = RecurrentPPO.load(str(tmp_path / 'v12' / 'latest.zip'), device='cpu')
    for removed in sorted(REMOVED):  # a v12 checkpoint relabelled as each removed arm
        model.free_play_task = removed
        model.save(str(tmp_path / f'{removed}.zip'))
        how = [dict(weights=tmp_path / f'{removed}.zip')] + ([dict(init=tmp_path / f'{removed}.zip'),
                                                               dict(leg_from=tmp_path / f'{removed}.zip', leg=2)]
                                                              if removed == 'free-play-v13' else [])
        for k, source in enumerate(how):
            environment = env(f'removed-{removed}-{k}', V12, source.get('leg', 1))
            with pytest.raises(ValueError, match=rf'the checkpoint .*{removed}\.zip is of {removed} \(.*\), a study arm '
                                                 'removed from the code'):
                T.train(tmp_path / f'removed-{removed}-{k}', environment=environment,
                        **{**DEVICES, 'decisions': rollout, 'task': V12, **source})
            environment.close()
    common = dict(DEVICES, task=V12, checkpoint_every=rollout, decisions=3 * rollout)
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', V12), env_seed=5, callbacks=(preempted_after(1),), **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', V12, 2), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and leg2['real']['settings']['task'] == V12
    with pytest.raises(ValueError, match=f'{V12} .*is built for the real engine only.*the fake environment'):
        T.train(tmp_path / 'fake-v12', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V12)
    with pytest.raises(ValueError, match=f'{V12} .*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V12)
    with pytest.raises(ValueError, match=f'{V12} .*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V12, engine='native', pack=tmp_path / 'any.pack')
    v12 = T.task_batch(V12)
    for k, change in enumerate((dict(inner='first-use-v1'), dict(items='item-v2'), dict(items='item-v4'))):
        with pytest.raises(ValueError):
            R.RealBatch([0], agents=1, starts=V2, life=('fixed', 30), key=1, output=tmp_path / f'pays-{k}',
                        **{**v12, **change})


def test_every_free_play_versions_batch_settings_reach_the_real_engines_batch_through_make_env(monkeypatch, tmp_path):
    """sim/train.py hands make_env a version's own settings (task_batch): for every free-play version they bind to its
    parameters and reach RealBatch as they are (free-play-v12's, v14's and v15's start_hp and start_rule included)."""
    from sim import train as T
    seen = {}

    class Batch:
        def __init__(self, ports, **kwargs):
            seen.update(kwargs)
    monkeypatch.setattr(R, 'RealBatch', Batch)
    monkeypatch.setattr(R, 'RealEnv', lambda batch, device: batch)
    monkeypatch.setattr(R, 'check_stack', lambda environ, agents: {})
    for task in FREE_PLAY_TASKS:
        seen.clear()
        R.make_env(4, 1, tmp_path, 'cpu', dict(starts=V2, life=('fixed', 30), worlds=1, agents=4), ports=[1],
                   **T.task_batch(task))
        assert {k: seen[k] for k in T.task_batch(task)} == T.task_batch(task), task



class Unapplied(Body):
    """A Body of an older engine in all but its hurt count: its reset ignores the start record's drawn
    hitpoints (startHp), so every life starts at full."""

    def reset(self, seed, cfg):
        return super().reset(seed, {k: v for k, v in cfg.items() if k != 'startHp'})


class HitInReset(Body):
    """A Body whose reset applies startHp and then loses 1 hitpoint before the arm, as a new player does that an npc
    which attacks first hits during its reset item (a Highwayman at falador_farm_west in the real engine)."""

    def __init__(self, ticks, lost=1):
        super().__init__(ticks)
        self.lost = lost

    def reset(self, seed, cfg):
        obs = super().reset(seed, cfg)
        self.hp = max(0, self.hp - self.lost)
        return obs


def test_the_drawn_start_hitpoints_need_an_engine_that_counts_hurt_and_a_life_that_starts_above_them_stops_the_run(tmp_path):
    """free-play-v12, as v14 and v15, is refused before it plays by a world whose hello names no hurt count (an older
    engine, which would start every life at full hitpoints), and a life whose hitpoints at its arm are above its
    record's drawn hitpoints and the heals its reset item had time for (start_hp_most), or are none, stops the run: v12
    never plays as v10. A life hit during its reset item arms below its drawn hitpoints and plays on."""
    from sim import train as T
    for task in (V12, FREE_PLAY_V14['version'], FREE_PLAY_V15['version']):
        with pytest.raises(ValueError, match='hello names no hurt field'):
            fake(tmp_path / f'old-{task}', worlds=1, agents=2, key=25, world=lambda w: dict(food=True), **T.task_batch(task))
    batch, _ = fake(tmp_path / 'unapplied', worlds=1, agents=2, key=25, life=('fixed', 30), make=lambda j: Unapplied(3),
                    world=lambda w: dict(food=True, hurt=True), **T.task_batch(V12))
    with pytest.raises(R.RealStop, match="started the life at hitpoints .*, none or above its start record's drawn"):
        batch.play(1, actor='wait', rounds=10)
    batch, _ = fake(tmp_path / 'hit', worlds=1, agents=2, key=25, life=('fixed', 30), make=lambda j: HitInReset(3),
                    world=lambda w: dict(food=True, hurt=True), **T.task_batch(V12))
    batch.play(1, actor='wait', rounds=40)
    rows = [json.loads(line) for line in (tmp_path / 'hit' / 'lives.jsonl').read_text().splitlines()]
    group = R.watch.world_rules(V2)['start_group']
    streams = {j: R.watch.Records(25, j, group) for j in range(2)}
    drawn = [streams[r['agent']].next()['hp_drawn'] for r in sorted(rows, key=lambda r: (r['agent'], r['life']))]
    assert len(rows) >= 4 and [r['start_hp'][0] for r in sorted(rows, key=lambda r: (r['agent'], r['life']))] == \
        [d - 1 for d in drawn] and min(drawn) >= 2
    batch, _ = fake(tmp_path / 'none', worlds=1, agents=2, key=25, life=('fixed', 30), make=lambda j: HitInReset(3, 99),
                    world=lambda w: dict(food=True, hurt=True), **T.task_batch(V12))
    with pytest.raises(R.RealStop, match=r"started the life at hitpoints \[0, \d+\], none or above"):
        batch.play(1, actor='wait', rounds=10)
    # the game's healing restarts at each reset's login, so a heal before the arm needs a reset item of 100 ticks
    assert [R.start_hp_most(7, 10, ticks) for ticks in (6, 14, 99, 100, 250, 400)] == [7, 7, 7, 8, 9, 10]
    assert R.start_hp_most(9, 10, 400) == 10


def test_a_leg_goes_on_across_the_engine_that_added_the_hurt_count(tmp_path):
    """Leg 1 of a free-play-v10 run on worlds whose hello names no hurt count (an older engine) is continued by leg 2 on
    worlds that name it: hurt_facts, as life_facts, is the launch's own (LEG_LAUNCH), not the world's."""
    from sim import train as T
    rollout = 6 * 8
    common = dict(DEVICES, task=V10, checkpoint_every=rollout, decisions=3 * rollout)

    def env(name, leg, hurt):
        return R.RealEnv(fake(tmp_path / f'{name}-lives', key=25, life=('fixed', 30), leg=leg,
                              world=lambda w: dict(food=True, hurt=hurt), **T.task_batch(V10))[0])
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', 1, False), env_seed=5, callbacks=(preempted_after(1),),
            **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', 2, True), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and leg2['real']['settings']['hurt_facts'] == list(R.HURT_FACTS)


def test_a_demo_plays_a_free_play_v12_checkpoint_from_the_drawn_start_hitpoints(tmp_path, monkeypatch):
    """A free-play-v12 checkpoint plays in a demo under its own start rule: every reset names the record's drawn
    hitpoints, and the demo's records say so."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import run_contract
    d = demo_world(tmp_path, monkeypatch, food=True, hurt=True)
    model = RecurrentPPO.load(str(d['checkpoint']), device='cpu')
    model.__dict__.update(free_play_task=V12, paid_skills=[n.lower() for n in SKILL_ORDER], kit_extra=[['tinderbox', 1]])
    model.policy_kwargs = dict(model.policy_kwargs, contract=run_contract('item-v3'))
    model.save(str(tmp_path / 'v12.zip'))
    launch = json.loads(d['launch'].read_text())
    launch['command'][launch['command'].index('--task') + 1] = V12
    (tmp_path / 'v12-launch.json').write_text(json.dumps(launch))
    record = R.demo(tmp_path / 'logs' / 'demo' / 'v12', checkpoint=tmp_path / 'v12.zip', launch=tmp_path / 'v12-launch.json',
                    starts=V2, sites=['falador_farm_west'], key=5, seed=9, seconds=0.5, agents=6, radius=2,
                    environ=d['environ'], say=lambda *a, **k: None)
    cfgs = [cfg for body in d['world'].bodies.values() for _, cfg in body.cfgs]
    assert record['status'] == 'complete' and cfgs and all(type(c.get('startHp')) is int for c in cfgs)
    assert record['settings']['start_hp'] == record['settings_in_effect']['start_hp'] == 'drawn'
    assert record['settings_in_effect']['task'] == V12


def test_a_demo_plays_a_public_copy_with_its_public_launch(tmp_path, monkeypatch):
    """sim.real demo plays the public copy of a free-play-v12 checkpoint (scripts/export_policy.py) with its public
    launch.json, through every check of its own, to the end: its policy's name tables are filled from the table in use,
    and its records name no run, seed or folder of the training run."""
    from pathlib import Path
    from sb3_contrib import RecurrentPPO
    from baseline.policy import run_contract
    from sim.tests.test_watch import export_script
    d = demo_world(tmp_path, monkeypatch, food=True, hurt=True)
    model = RecurrentPPO.load(str(d['checkpoint']), device='cpu')
    model.__dict__.update(free_play_task=V12, paid_skills=[n.lower() for n in SKILL_ORDER], kit_extra=[['tinderbox', 1]])
    model.policy_kwargs = dict(model.policy_kwargs, contract=run_contract('item-v3'))
    model.save(str(tmp_path / 'v12.zip'))
    launch = json.loads(d['launch'].read_text())
    launch['command'][launch['command'].index('--task') + 1] = V12
    (tmp_path / 'v12-launch.json').write_text(json.dumps(launch))
    done = export_script().export(tmp_path / 'v12.zip', tmp_path / 'public' / 'v12', tmp_path / 'v12-launch.json')
    public, public_launch = Path(done['files'][0]), Path(done['files'][2])
    assert 'real-run' not in public_launch.read_text() and '--seed' not in public_launch.read_text()
    record = R.demo(tmp_path / 'logs' / 'demo' / 'public', checkpoint=public, launch=public_launch, starts=V2,
                    sites=['falador_farm_west'], key=5, seed=9, seconds=0.5, agents=6, radius=2, environ=d['environ'],
                    say=lambda *a, **k: None)
    assert record['status'] == 'complete' and record['settings_in_effect']['task'] == V12
    assert record['policy']['policy']['names']['vectors'] == 'the table in use (a public copy holds none)'
    assert record['policy']['run']['launch']['name'] == '' and record['settings']['life'] == ['fixed', 40]


# ---- free-play-v14: the start rule old-life-v1 (longer lives, stage 2) -----------------------------------------------
V14 = FREE_PLAY_V14['version']
OLD_LIFE_TEXT = 'woodcutting=50,firemaking=52,attack=31,strength=37,defence=30,hitpoints=34'  # the old-life profile
# reset_cfgs(hurt=True) of free-play-v12 made before free-play-v14: (the resets' sha256, the life rows'
# sha256 as sorted JSON)
V12_BEFORE = {V12: ('887e3537b11dee2b5fbc30f548c5da7ebbd0865418940a43aacfc5ca0596cde8',
                    'a031a1ef6d53c57969741e6df11c482252d97501e50ee90bdf7e3ba68d2863c6')}


def played(tmp_path, task, agents=4, make=lambda j: Body(3)):
    """A fake batch of `task` (key 25, 2 worlds of `agents` agents, 30-tick lives, the engine's hurt count) played with
    WAIT for 40 rounds: (the batch, its life rows by (agent, life), its lives-xp rows by (agent, life), each agent's
    resets (seed, cfg) in order)."""
    from sim import train as T
    batch, fakes = fake(tmp_path / task, worlds=2, agents=agents, key=25, life=('fixed', 30), make=make,
                        world=lambda w: dict(food=True, hurt=True), **T.task_batch(task))
    batch.play(1, actor='wait', rounds=40)
    rows = lambda name: {(r['agent'], r['life']): r for r in
                         (json.loads(line) for line in (tmp_path / task / name).read_text().splitlines())}
    resets = {w * agents + slot: f.bodies[slot].cfgs for w, f in enumerate(fakes) for slot in f.bodies}
    return batch, rows('lives.jsonl'), rows('lives-xp.jsonl'), resets


def test_free_play_v14_starts_its_odd_agents_on_the_old_life_profile_and_its_even_ones_as_free_play_v12(tmp_path):
    """free-play-v14 (old-life-v1): the agents of odd index, two of each world's four, play every life from their record
    as the watch's profiled() gives it with the old-life profile, and its reset names the profiled levels and hitpoints;
    the agents of even index play exactly as the same agents of a free-play-v12 batch (the same resets and the same life
    rows), as the record streams are not touched. Each life row and lives-xp row of v14 says which kind the life is and
    its start levels, and a v12 row says neither. The resets and life rows of free-play-v12 are those made
    before v14 (V12_BEFORE)."""
    import hashlib
    for task, digests in V12_BEFORE.items():
        rows, _, found = reset_cfgs(tmp_path / 'before', task, hurt=True)
        assert (found, hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()) == digests
    profile = R.watch.parse_levels(OLD_LIFE_TEXT)
    _, rows12, xp12, resets12 = played(tmp_path, V12)
    v14, rows14, xp14, resets14 = played(tmp_path, V14)
    assert v14.task_version() == V14 and v14.settings_in_effect['start_rule'] == OLD_LIFE
    assert v14.settings_in_effect['profile'] == profile and list(v14.profile) == list(profile)
    assert [a.j for a in v14.agents if a.profile is not None] == [1, 3, 5, 7]  # fixed: half of each world
    assert set(rows14) == set(rows12) and len(rows14) >= 16 and set(xp14) == set(rows14)
    group = R.watch.world_rules(V2)['start_group']
    for j in range(8):
        stream = R.watch.Records(25, j, group)
        assert len(resets14[j]) == len(resets12[j]) >= 2
        for n, ((seed, cfg), (seed12, cfg12)) in enumerate(zip(resets14[j], resets12[j])):
            drawn = stream.next()
            record = R.watch.profiled(drawn, profile if j % 2 else None)
            assert (seed, cfg) == (seed12, dict(cfg12, setStats=record['set_stats'], startHp=record['hp_drawn']))
            assert (cfg['setStats'] == cfg12['setStats']) is (j % 2 == 0)
            if (j, n) not in rows14:  # its life had not ended when the play stopped
                continue
            row, xp = rows14[j, n], xp14[j, n]
            assert row['profile'] is xp['profile'] is bool(j % 2)
            assert row['start_levels'] == xp['start_levels'] == record['levels']
            assert row['start_hp'] == [record['hp_drawn'], record['levels']['hitpoints']]
            if j % 2 == 0:
                assert {k: v for k, v in row.items() if k not in R.START_FIELDS} == rows12[j, n]
                assert {k: v for k, v in xp.items() if k not in R.START_FIELDS} == xp12[j, n]
            else:
                assert row['start_hp'][1] == 34 and row['start_levels']['firemaking'] == 52
    assert not any(k in r for r in [*rows12.values(), *xp12.values()] for k in R.START_FIELDS)


class SkipsWoodcutting(Body):
    """A Body whose reset skips the start's woodcutting level (as a native reset skips a stat name it does not know), so
    the stat stays at its default level of 1."""

    def reset(self, seed, cfg):
        stats = ','.join(p for p in cfg['setStats'].split(',') if not p.startswith('woodcutting:'))
        return super().reset(seed, dict(cfg, setStats=stats))


def test_a_v14_life_arms_within_its_profiled_hitpoints_at_the_profiles_levels_or_the_run_stops(tmp_path):
    """free-play-v12's start check holds for a profiled life with its own drawn hitpoints and base (start_hp_most): a life
    hit during its reset item arms below its profiled drawn hitpoints and plays on; a profiled life that arms at the
    profile's full hitpoints above its drawn ones (an engine that ignores startHp) stops the run, and so does one whose
    engine left a stat of the profile at another level. The batch refuses the rule with an odd number of agents a world,
    with the cost of time hurt, without the drawn hitpoints, and any other profile."""
    from sim import train as T
    profile, group = R.watch.parse_levels(OLD_LIFE_TEXT), R.watch.world_rules(V2)['start_group']
    _, rows, _, _ = played(tmp_path / 'hit', V14, agents=2, make=lambda j: HitInReset(3))
    for (j, n), row in rows.items():
        stream = R.watch.Records(25, j, group)
        record = R.watch.profiled([stream.next() for _ in range(n + 1)][-1], profile if j % 2 else None)
        assert row['start_hp'] == [record['hp_drawn'] - 1, record['levels']['hitpoints']] and record['hp_drawn'] >= 2
    assert {j % 2 for j, _ in rows} == {0, 1}
    batch, _ = fake(tmp_path / 'unapplied', worlds=1, agents=2, key=25, life=('fixed', 30),
                    make=lambda j: Unapplied(3) if j % 2 else Body(3), world=lambda w: dict(food=True, hurt=True),
                    **T.task_batch(V14))
    with pytest.raises(R.RealStop, match=r"agent 1 life \d+: the engine started the life at hitpoints \[34, 34\], none or "
                                         r"above its start record's drawn 3[1-3]"):
        batch.play(1, actor='wait', rounds=40)
    batch, _ = fake(tmp_path / 'skipped', worlds=1, agents=2, key=25, life=('fixed', 30),
                    make=lambda j: SkipsWoodcutting(3), world=lambda w: dict(food=True, hurt=True), **T.task_batch(V14))
    with pytest.raises(R.RealStop, match=r"agent 1 life 0: the engine armed the life at levels .*'woodcutting': 1.*, not "
                                         r"the profile's"):
        batch.play(1, actor='wait', rounds=10)
    v14 = T.task_batch(V14)
    for k, (agents, change) in enumerate(((3, {}), (2, dict(start_hp=False)),
                                          (2, dict(start_rule=dict(OLD_LIFE, levels=dict(OLD_LIFE['levels'], mining=20)))),
                                          (2, dict(start_rule=dict(OLD_LIFE, agents='even'))))):
        with pytest.raises(ValueError, match='old-life-v1 goes with free-play-v12 alone'):
            R.RealBatch([0], agents=agents, starts=V2, life=('fixed', 30), key=1, output=tmp_path / f'refused-{k}',
                        **{**v14, **change})


def test_free_play_v14_starts_from_v12_weights_alone_and_a_leg_keeps_its_rule(tmp_path):
    """--weights: a free-play-v12 checkpoint starts a free-play-v14 run, recorded with both versions and the note, every
    check clean with --validate, its run, batch and checkpoint naming v14 and old-life-v1, its lives of both kinds.
    Nothing else crosses into v14 (v10 weights) and v14 weights start no v12 run. Leg 2 of a v14 run goes on
    under its rule. The fake environment, the Python reference and the C engine refuse v14 by name."""
    import zipfile
    from sim import train as T
    from sim.workers import CpuWorkers
    rollout = 6 * 8

    def env(name, task, leg=1):  # 3 worlds of 2 agents: an even number of agents a world
        return R.RealEnv(fake(tmp_path / f'{name}-lives', worlds=3, agents=2, key=25, life=('fixed', 30), leg=leg,
                              world=lambda w: dict(food=True, hurt=True),
                              make=lambda j: Body(3, hits=HITS.get(j % 4), death=DEATHS.get(j % 4)),
                              **T.task_batch(task))[0])
    for name, task in (('v10', V10), ('v12', V12)):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), **dict(DEVICES, task=task))
    run = T.train(tmp_path / 'v14', decisions=2 * rollout, environment=env('v14', V14),
                  weights=tmp_path / 'v12' / 'latest.zip', **dict(DEVICES, task=V14))
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=V12, run=V14) and run['weights']['note'] == T.WEIGHTS_NOTES[V14]
    assert run['real']['settings']['task'] == V14 and run['real']['settings']['start_hp'] == 'drawn'
    assert run['real']['settings']['start_rule'] == run['free_play']['start_rule'] == OLD_LIFE
    assert {r['profile'] for r in lives(tmp_path / 'v14-lives')} == {True, False}
    with zipfile.ZipFile(tmp_path / 'v14' / 'latest.zip') as archive:
        assert json.loads(archive.read('data'))['free_play_task'] == V14
    for k, (task, source) in enumerate(((V14, 'v10'), (V12, 'v14'))):
        environment = env(f'refused-{k}', task)
        with pytest.raises(ValueError, match=rf"never start a run of \('{task}'"):
            T.train(tmp_path / f'refused-{k}', environment=environment, weights=tmp_path / source / 'latest.zip',
                    **{**DEVICES, 'decisions': rollout, 'task': task})
        environment.close()
    common = dict(DEVICES, task=V14, checkpoint_every=rollout, decisions=3 * rollout)
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', V14), env_seed=5, callbacks=(preempted_after(1),), **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', V14, 2), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and leg2['real']['settings']['start_rule'] == OLD_LIFE
    assert {r['profile'] for r in lives(tmp_path / 'leg2-lives')} == {True, False}
    with pytest.raises(ValueError, match=f'{V14} .*is built for the real engine only.*the fake environment'):
        T.train(tmp_path / 'fake-v14', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V14)
    with pytest.raises(ValueError, match=f'{V14} .*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V14)
    with pytest.raises(ValueError, match=f'{V14} .*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V14, engine='native', pack=tmp_path / 'any.pack')


# ---- free-play-v15: the pay rare-xp-v2 (directed exploration: a pay for XP of a rare kind) ---------------------------
V15 = FREE_PLAY_V15['version']
# played() of free-play-v12 and v14 made before free-play-v15: sha256 of the life rows, of the lives-xp rows
# (each sorted by agent and life, as sorted JSON) and of the resets
V12_V14_BEFORE = {V12: ('d1f4ca2525e4c558fd43139c3062dba01068e890d0427d2b615bb1f68c7c729a',
                        'a18ff2230c74fa76c79ecf47eb93f4e32c91cd58541a6593b4f8ab3cb4b04083',
                        'd4fe99d68ea0ad88bd2c32cc000ee7dbbe90f1b64f0007b791b36be22f8fe4de'),
                  V14: ('b17a857af44d8b61c3e9e65101689a652670dc3d1dd248968f75a2340f61c9c7',
                        '626ea38b5ba49141c771f3027d41f6dd72cd929d79b124007a266ee7f519567d',
                        '49ef0dd9d61421d44888396fa4e29ac64affd42034e53fd68d85f76fdedc34fd')}
STATS = ('attack', 'mining', 'thieving')


def rare_pay(window=R.RareXp.WINDOW):
    """A RareXp whose rows go to a list (`.rows`), over a window of `window` decisions (the pay's own, or a test's small
    one), and step(rose, agent=0, tick=None): one decision whose XP rose in the stats `rose`; its rows."""
    class Pay(R.RareXp):
        WINDOW = window
    rows = []
    pay = Pay(lambda kind, **row: rows.append(row), 'rare-xp.jsonl')  # its rows stay in the list: the path is never read
    pay.rows = rows

    def step(rose, agent=0, life=0, decision=1):
        return pay.fresh(types.SimpleNamespace(j=agent, world=0, slot=agent, life=life, decisions=decision,
                                               rose=tuple(rose), rare_ticks=[]), dict(tick=100 + pay.decisions))
    pay.step = step
    return pay


@settings(max_examples=80, deadline=None)
@given(st.lists(st.sets(st.sampled_from(STATS)), max_size=120), st.integers(1, 30), st.integers(0, 120))
def test_rare_xp_v2_pays_each_event_by_its_stats_count_in_the_window_before_it_and_replays_and_restores(events, window,
                                                                                                         cut):
    """rare-xp-v2 over any events, with a small window: each row's r is the number of decisions with XP in its stat among
    the `window` decisions before it (counted here by brute force) and its pay 10 / (10 + r); a stat with no XP pays
    nothing; the replay of the rows from the start gives every decision's pay and the same window; and a window saved
    after `cut` decisions and restored pays the rest exactly as the run that went on. Over any span of at most `window`
    decisions, a stat's pay is at most 10 (H(n + 9) - H(9)), n its events in the span (the k-th pays at most
    10 / (9 + k)): about 88 levels for 65,536 events in an update, while a steady rate of r an update pays
    r * 10 / (10 + 2 r), under 5 (test_rare_xp_v2_pays_a_steady_rate_under_five_levels_an_update)."""
    pay = rare_pay(window)
    start = pay.replay_state()
    for n, rose in enumerate(events):
        rows = pay.step(sorted(rose, key=STATS.index), decision=n + 1)
        assert [r['stat'] for r in rows] == sorted(rose, key=STATS.index)
        for r in rows:
            seen = sum(r['stat'] in events[m] for m in range(max(0, n - window), n))
            assert (r['n'], r['r'], r['pay']) == (n, seen, 10 / (10 + seen))
    assert {r['stat'] for r in pay.rows} == set().union(*events) if events else not pay.rows
    paid, _, problem = type(pay).replay(pay.rows, start)
    assert problem is None and pay.same(start)
    want = {}
    for r in pay.rows:
        want[0, 0, r['n'] + 1] = want.get((0, 0, r['n'] + 1), 0.0) + r['pay']
    assert paid == want
    for stat in STATS:  # the bound over every span of at most `window` decisions
        for a in range(len(events)):
            got = sum(r['pay'] for r in pay.rows if r['stat'] == stat and a <= r['n'] < a + window)
            k = sum(stat in e for e in events[a:a + window])
            assert got <= 10 * sum(1 / (9 + i) for i in range(1, k + 1)) + 1e-12
    cut = min(cut, len(events))
    first = rare_pay(window)
    for n, rose in enumerate(events[:cut]):
        first.step(sorted(rose, key=STATS.index), decision=n + 1)
    saved = json.loads(json.dumps(first.saved()))  # as a checkpoint holds it
    again = rare_pay(window)
    again.restore(saved)
    for n, rose in enumerate(events[cut:], start=cut):
        again.step(sorted(rose, key=STATS.index), decision=n + 1)
    assert again.rows == [r for r in pay.rows if r['n'] >= cut] and again.saved() == pay.saved()


def test_rare_xp_v2_pays_a_steady_rate_under_five_levels_an_update_and_a_farmed_stat_less_for_each_event():
    """With the pay's own window (131,072 decisions, two updates of 65,536), a stat with XP at every m-th decision pays,
    in the third update, r * 10 / (10 + 2 r) levels or less (r = 65,536 / m its events an update): under 5 levels
    whatever m, and each event pays less the more often the stat comes (a farmed stat's pay falls). A rare stat keeps
    its pay: 9 events an update pay about 0.36 each. The replay of all the rows agrees."""
    update = 65_536
    per_event, per_update = {}, {}
    for m in (7, 100, 7_000):
        pay = rare_pay()
        start = pay.replay_state()
        for n in range(3 * update):
            pay.step(('mining',) if n % m == 0 else (), decision=n + 1)
        third = [r['pay'] for r in pay.rows if r['n'] >= 2 * update]
        r = -(-update // m)
        per_update[m], per_event[m] = sum(third), third[-1]
        assert per_update[m] < 5 and per_update[m] <= r * 10 / (10 + 2 * (r - 1)) + 1e-9
        assert R.RareXp.replay(pay.rows, start)[2] is None and pay.same(start)
    assert per_event[7] < per_event[100] < per_event[7_000] and 0.3 < per_event[7_000] < 0.4
    assert R.RareXp.WINDOW == 2 * update and R.RareXp.pay(0) == 1.0 and R.RareXp.pay(10) == 0.5


def test_the_rare_xp_replay_finds_a_changed_pay_a_moved_row_and_a_missing_row():
    """The recount's replay (RareXp.replay, apart from the pay's own window) refuses a row whose pay or r was changed,
    a row moved before the one ahead of it, a stat twice in one decision, and, by the comparison of the windows (same),
    a row left out."""
    pay = rare_pay(50)
    for n in range(80):
        pay.step(['attack', 'mining'][:1 + n % 2] if n % 3 else ['thieving'], decision=n + 1)
    rows, small, fresh = pay.rows, type(pay), lambda: rare_pay(50).replay_state()
    state = fresh()
    assert small.replay(rows, state)[2] is None and pay.same(state)
    changed = [dict(r, pay=r['pay'] + 2 ** -40) if k == 60 else r for k, r in enumerate(rows)]
    assert 'recorded with r' in small.replay(changed, fresh())[2]
    k = next(k for k in range(len(rows) - 1) if rows[k]['n'] < rows[k + 1]['n'])
    moved = rows[:k] + [rows[k + 1], rows[k]] + rows[k + 2:]
    assert 'out of order' in small.replay(moved, fresh())[2]
    twice = rows[:5] + [rows[4]] + rows[5:]
    assert 'out of order' in small.replay(twice, fresh())[2]
    state = fresh()
    assert small.replay(rows[:-1], state)[2] is None and not pay.same(state)


def test_the_rare_xp_check_from_the_grants_gives_a_grant_on_a_results_own_tick_to_the_next_decision():
    """RareXp.check_life: decision k owns the grants the engine stamped in [t(k-1), t(k)), t(0) the arm's tick: a grant
    at the arm's tick is the first decision's, a grant at a result's own tick the next decision's (the engine stamps a
    grant with its cycle's tick and increments the tick after the cycle, sim/real.py). Rows that pay the decisions the
    grants give pass; a row one decision off, a grant not paid, and a grant outside the life's decisions fail."""
    skills = [s.lower() for s in SKILL_ORDER]
    a, m = skills.index('attack'), skills.index('mining')
    p = dict(rare_ticks=[100, 103, 107, 110])  # the arm at 100; decisions 1, 2, 3 end at 103, 107, 110
    grants = [[a, 40, 100], [m, 175, 103], [a, 12, 106], [m, 175, 109]]  # 100 -> 1, 103 -> 2, 106 -> 2, 109 -> 3
    rows = lambda *pairs: [dict(decision=k, stat=s) for k, s in pairs]
    good = rows((1, 'attack'), (2, 'mining'), (2, 'attack'), (3, 'mining'))
    assert R.RareXp.check_life(dict(p, kinds=good), dict(grants=grants), skills) is None
    late = rows((1, 'attack'), (1, 'mining'), (2, 'attack'), (3, 'mining'))  # the grant at 103 read as decision 1's
    assert 'that the pay rows lack' in R.RareXp.check_life(dict(p, kinds=late), dict(grants=grants), skills)
    assert R.RareXp.check_life(dict(p, kinds=good[:3]), dict(grants=grants), skills) is not None
    for tick in (99, 110):
        assert 'outside the life' in R.RareXp.check_life(dict(p, kinds=good), dict(grants=grants + [[a, 1, tick]]),
                                                         skills)


class LateGrants(Body):
    """A Body whose ledger stamps each grant at its result's own tick: the engine's convention broken."""

    def step(self, action, setup=False):
        obs = super().step(action, setup)
        if not setup and self.grants:
            for g in self.grants:
                if g[2] == self.tick - self.step_ticks:
                    g[2] = self.tick
        return obs


def test_free_play_v15_pays_rare_xp_beside_the_levels_and_its_life_rows_read_it_by_stat(tmp_path):
    """A free-play-v15 batch plays v14 (its resets, its kinds of life) with the pay rare-xp-v2: its rows give each life
    row's rare_xp, by stat in row order, and rare_xp_all; its lives-xp rows hold rare_xp too; every life passes the
    check from the engine's grants; and the life rows less the pay's fields are v14's. v12's and v14's rows and resets
    are those made before v15 (V12_V14_BEFORE). An engine whose ledger stamps a grant at the result's own tick stops the
    run at a life's close."""
    import hashlib
    digest = lambda x: hashlib.sha256(json.dumps(sorted(x.values(), key=lambda r: (r['agent'], r['life'])),
                                                 sort_keys=True).encode()).hexdigest()
    for task, (lives_digest, xp_digest, resets_digest) in V12_V14_BEFORE.items():
        _, rows, xp, resets = played(tmp_path / 'before', task)
        assert (digest(rows), digest(xp)) == (lives_digest, xp_digest)
        assert hashlib.sha256(json.dumps(resets, sort_keys=True, default=list).encode()).hexdigest() == resets_digest
    _, rows14, xp14, resets14 = played(tmp_path, V14)
    v15, rows15, xp15, resets15 = played(tmp_path, V15)
    assert v15.task_version() == V15 and resets15 == resets14 and set(rows15) == set(rows14)
    assert (v15.settings_in_effect['inner_pay'], v15.settings_in_effect['inner_window']) == ('rare-xp-v2', 131_072)
    paid = [json.loads(line) for line in (tmp_path / V15 / 'rare-xp.jsonl').read_text().splitlines()]
    for key, row in rows15.items():
        mine = [r for r in paid if (r['agent'], r['life']) == key]
        by_stat = {}
        for r in mine:
            by_stat[r['stat']] = by_stat.get(r['stat'], 0.0) + r['pay']
        assert row['rare_xp'] == xp15[key]['rare_xp'] == by_stat and row['ledger'] == 'match'
        assert row['rare_xp_all'] == pytest.approx(sum(by_stat.values()), rel=1e-12)
        fields = ('rare_xp', 'rare_xp_all', 'inner', 'kinds')
        assert {k: v for k, v in row.items() if k not in fields} == rows14[key]
        assert {k: v for k, v in xp15[key].items() if k != 'rare_xp'} == xp14[key]
    assert [r['n'] for r in paid] == sorted(r['n'] for r in paid) and paid[0]['pay'] == 1.0
    from sim import train as T
    batch, _ = fake(tmp_path / 'late', worlds=1, agents=2, key=25, life=('fixed', 30), make=lambda j: LateGrants(3),
                    world=lambda w: dict(food=True, hurt=True), **T.task_batch(V15))
    with pytest.raises(R.RealStop, match="rare-xp-v2 against the engine's grants: (the engine's grants give XP at|a "
                                         "grant of .* outside the life's decisions)"):
        batch.play(1, actor='wait', rounds=40)
    v15 = T.task_batch(V15)
    for k, change in enumerate((dict(start_rule=None), dict(start_hp=False),
                                dict(items='item-v2'))):
        with pytest.raises(ValueError):
            R.RealBatch([0], agents=2, starts=V2, life=('fixed', 30), key=1, output=tmp_path / f'refused-{k}',
                        **{**v15, **change})


def test_free_play_v15_starts_from_v12_weights_its_window_goes_on_in_a_leg_and_v15_weights_start_v14(tmp_path):
    """--weights: a free-play-v12 checkpoint starts a free-play-v15 run (both versions and the note recorded), every check
    clean with --validate, the rare-XP pay in its updates and its checkpoint saving the window (rare_xp_window); a v15
    checkpoint starts a v14 run, recorded with the pair's own note. v10 and v14 weights start no v15 run, and v15
    weights no v12 run. A row written with another pay stops the run at the recount, before its first update. Leg 2 of
    a v15 run goes on from its checkpoint's window: its decisions' numbers go on, and its first rows count the window's
    events. The fake environment, the Python reference and the C engine refuse v15 by name."""
    import zipfile
    from sim import train as T
    from sim.workers import CpuWorkers
    rollout = 6 * 8

    def env(name, task, leg=1):  # 3 worlds of 2 agents, as v14's
        return R.RealEnv(fake(tmp_path / f'{name}-lives', worlds=3, agents=2, key=25, life=('fixed', 30), leg=leg,
                              world=lambda w: dict(food=True, hurt=True),
                              make=lambda j: Body(3, hits=HITS.get(j % 4), death=DEATHS.get(j % 4)),
                              **T.task_batch(task))[0])
    for name, task in (('v10', V10), ('v12', V12)):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), **dict(DEVICES, task=task))
    for name, task, source in (('v14', V14, 'v12'),):
        T.train(tmp_path / name, decisions=rollout, environment=env(name, task), weights=tmp_path / source / 'latest.zip',
                **dict(DEVICES, task=task))
    run = T.train(tmp_path / 'v15', decisions=2 * rollout, environment=env('v15', V15),
                  weights=tmp_path / 'v12' / 'latest.zip', **dict(DEVICES, task=V15))
    assert run['status'] == 'completed' and all(c['problems'] == [] for c in run['checks']['rollouts'])
    assert run['weights']['task_versions'] == dict(checkpoint=V12, run=V15) and run['weights']['note'] == T.WEIGHTS_NOTES[V15]
    assert run['real']['settings']['task'] == V15 and run['free_play']['inner_pay'] == 'rare-xp-v2'
    assert all(u['rows_inner_paid'] > 0 for u in run['real']['updates'])
    with zipfile.ZipFile(tmp_path / 'v15' / 'latest.zip') as archive:
        data = json.loads(archive.read('data'))
    assert data['free_play_task'] == V15 and data['rare_xp_window']['version'] == 'rare-xp-v2'
    assert data['rare_xp_window']['decisions'] > 0 and data['rare_xp_window']['counts']
    back = T.train(tmp_path / 'v14-from-v15', decisions=rollout, environment=env('v14-from-v15', V14),
                   weights=tmp_path / 'v15' / 'latest.zip', **dict(DEVICES, task=V14))
    assert back['weights']['task_versions'] == dict(checkpoint=V15, run=V14)
    assert back['weights']['note'] == T.PAIR_NOTES[V15, V14] and 'rare_xp_window' in back['weights']['saved_settings_not_imported']
    assert not (tmp_path / 'v14-from-v15-lives' / 'rare-xp.jsonl').exists()
    for k, (task, source) in enumerate(((V15, 'v10'), (V15, 'v14'), (V12, 'v15'))):
        environment = env(f'refused-{k}', task)
        with pytest.raises(ValueError, match=rf"never start a run of \('{task}'"):
            T.train(tmp_path / f'refused-{k}', environment=environment, weights=tmp_path / source / 'latest.zip',
                    **{**DEVICES, 'decisions': rollout, 'task': task})
        environment.close()
    common = dict(DEVICES, task=V15, checkpoint_every=rollout, decisions=3 * rollout)
    tampered = env('tampered', V15)
    write = tampered.batch.first.write

    def tamper(kind, **row):  # the first row written with half a level more than the batch paid
        write(kind, **(dict(row, pay=row['pay'] + 0.5) if not tamper.done else row))
        tamper.done = True
    tamper.done, tamper.close = False, write.close
    tampered.batch.first.write = tamper
    with pytest.raises(R.RealStop, match='the inner pay recount failed: .* recorded with r 0 and pay 1.5'):
        T.train(tmp_path / 'tampered', environment=tampered, **common)
    assert [q.name for q in (tmp_path / 'tampered').rglob('*.zip')] == ['initial.zip']
    stopped(T.train, tmp_path / 'leg1', environment=env('leg1', V15), env_seed=5, callbacks=(preempted_after(1),), **common)
    parent = tmp_path / 'leg1' / 'checkpoints' / f'ppo_{rollout}_steps.zip'
    with zipfile.ZipFile(parent) as archive:
        window = json.loads(archive.read('data'))['rare_xp_window']
    leg2 = T.train(tmp_path / 'leg2', environment=env('leg2', V15, 2), leg_from=parent, leg=2, **common)
    assert leg2['status'] == 'completed' and leg2['real']['settings']['task'] == V15
    rows = [json.loads(line) for line in (tmp_path / 'leg2-lives' / 'rare-xp.jsonl').read_text().splitlines()]
    first = next(r for r in rows if r['stat'] == 'woodcutting')
    assert rows[0]['n'] >= window['decisions'] and first['r'] == window['counts']['woodcutting'] > 0
    assert first['pay'] == 10 / (10 + window['counts']['woodcutting'])
    with pytest.raises(ValueError, match=f'{V15} .*is built for the real engine only.*the fake environment'):
        T.train(tmp_path / 'fake-v15', decisions=64, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1, task=V15)
    with pytest.raises(ValueError, match=f'{V15} .*the Python reference does not pay it'):
        CpuWorkers(4, 1, None, task=V15)
    with pytest.raises(ValueError, match=f'{V15} .*the C engine does not pay it'):
        CpuWorkers(4, 1, None, task=V15, engine='native', pack=tmp_path / 'any.pack')
