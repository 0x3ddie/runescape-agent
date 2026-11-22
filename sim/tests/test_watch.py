"""sim/watch.py, native-watch-v1. A fake native world speaks the engine's
protocol and follows its rules, not the runner's: fresh incarnations, the armed capture with its clock and its tick
boundaries, a combat capture that closes on the lethal hit before it records it and stops at its first failure, levels
that follow XP, decision-point waits that run past the end, status rows with their lockstep flag. The runner plays it as
it would play scripts/stack.py's worlds: a death starts the next life from the next record with the ticks left, the
guard cuts a run, a failed, hung or actorless world is reported as such, a pickup of an excluded drop ends a life apart,
the sites rotate over the worlds, and the monitors and propose account the saved evidence as played. Each run applies
the rules of the world it names (a cabbage ends a village-world-v1 life, not a village-world-v2 one), plays a policy
only of a run that trained in that world, and records the world and its archive; a run record that names none is
village-world-v1's. The env-ts bounds and the Python ones agree."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import pytest

from baseline import config
from baseline.resource_task import MAX_HORIZON
from baseline.tests.test_goal_task import DECLARED
from baseline.tests.test_native_flow import CONTAINERS
from baseline.tests.test_resource_task import raw as plain_raw
from act_contract import A_ACT0
from obs import LEVEL_XP, N_INV, SKILL_ORDER
from sim import watch as W
from sim.agent import A_DROP0, A_HELD0, A_STYLE0, A_WAIT
from wide_contract import ITEM_V4, MASK_V6, N_ACTIONS_V4, N_PAGE, page_entry, page_row
from sim.spec import START_LEVELS, START_WEAPONS

ROOT = Path(__file__).resolve().parents[2]
NAMES = dict(bronze_axe=(1351, 'Bronze axe'), net=(303, 'Small fishing net'), bronze_pickaxe=(1265, 'Bronze pickaxe'),
             shrimp=(315, 'Shrimps'), bread=(2309, 'Bread'), bronze_dagger=(1205, 'Bronze dagger'),
             bronze_sword=(1277, 'Bronze sword'), tinderbox=(590, 'Tinderbox'), raw_chicken=(2138, 'Raw chicken'))
LOGS, CLUE, CHISEL, GOBLIN, CHICKEN, CABBAGE, RAW, BURNT = 1511, 2677, 1755, 7, 8, 1965, 317, 323
ORES = dict(copper=(436, 'Copper ore'), tin=(438, 'Tin ore'), iron=(440, 'Iron ore'))
ORE_IDS = {ident for ident, _ in ORES.values()}
EMPTY, REGROW = 450, 6  # a mined-out rock's loc type (any other than the live one) and the ticks until it grows back
V1, V2, V3 = 'village-world-v1', 'village-world-v2', 'village-world-v3'
REAL_SITES = dict(W.SITES)  # the module's own (the tests' fixture puts SITES in its place)
SITES = dict(north=(3200, 3200), south=(3200, 3100), east=(3300, 3200), west=(3100, 3200))
TREE = config.ENGINE_TREES['pinned']
ENVIRON = dict(NODE_RANDOM_EVENTS='false')


def whole_xp(level):
    return LEVEL_XP[level - 2] // 10 if level >= 2 else 0


def level_of(xp):
    """The base level of whole XP `xp` by the engine's table (rl/obs.py LEVEL_XP)."""
    return max(level for level in range(1, 100) if whole_xp(level) <= xp)


class Fake:
    """One persistent native world: its transport (reset, step, close) and its resource control (control). A reset makes
    a fresh incarnation from the declared start (cfg setStats, seedItems, spawnTele). A step advances `step_ticks` world
    ticks, or `waits[(life, tick from arm at the decision)]` ticks (a decision-point wait, which a capture's end does not
    cut). Each world tick plays `script[(life, tick from arm)]` whether the capture is open or not, and the capture
    records only while open:
    ('log',) a log produced (woodcutting XP 25); ('xp', skill, n); ('hit', nid, damage) the agent hits an npc of 3
    hitpoints; ('hurt', nid, damage) an npc hits the agent; ('lethal', nid) an npc's hit takes the agent's last
    hitpoints: the capture closes on the death before the combat capture records the loss; ('roll', nid, damage) an
    npc's attack on the agent rolled, and ('land', damage) its damage landing later (as the engine queues it), recorded
    only while the capture is open, so not in the horizon's own cycle; ('combat_fail', cause) the
    combat capture fails and records nothing after; ('clue',) a clue on the ground; ('offer',) a clue only among the
    Take affordances; ('take',) the clue picked up; ('loot', id, name) and ('lose', id) another item in and out of the
    inventory, ('lose1', id) one of them; ('fail',) the transport fails; ('hang',) the step blocks until `release` is set. Levels follow XP by
    the engine's table. Items keep their slots, as the engine's inventory does. `arm_ground` lays a clue at the first
    reset. `rocks`: the Rimmington mine's rocks (sim/watch.py MINE_ROCKS) stand in this world, offered as 'Rocks' Mine
    rows within 15 tiles (nearest first, their loc type the live one or EMPTY while mined out); the player stays where it
    is unless an ACT on a rock walks it beside the rock, where a live rock whose level the Mining level reaches gives its
    ore (a production event), 17 XP (35 for iron) and turns empty for REGROW ticks; an item-v2 DROP of an occupied entry
    of page 0 is legal and drops it; the pack holds N_INV items and a full one takes no ore."""

    def __init__(self, script=(), step_ticks=3, waits=(), sleep=0.0, arm_ground=False, rocks=False):
        self.script, self.step_ticks, self.waits, self.sleep = dict(script), step_ticks, dict(waits), sleep
        self.rocks = {tile: 0 for tile in W.MINE_ROCKS} if rocks else None  # ticks until each rock is live again
        self.contract = 'item-v2'
        self.arm_ground, self.release = arm_ground, threading.Event()
        self.tick, self.epoch, self.life, self.cfgs, self.calls, self.capture = 100, 0, -1, [], [], None

    # -- transport ---------------------------------------------------------------------------------------------------
    def reset(self, seed, cfg):
        self.epoch, self.life, self.tick = self.epoch + 1, self.life + 1, self.tick + 4
        self.cfgs.append((seed, deepcopy(cfg)))
        self.contract = cfg.get('contract', 'item-v2')  # the item contract the life's reset names
        levels = {name.capitalize(): int(v) for name, v in (p.split(':') for p in cfg['setStats'].split(','))}
        self.xp = {name: whole_xp(levels.get(name, 10 if name == 'Hitpoints' else 1)) for name in SKILL_ORDER}
        self.items = [dict(id=NAMES[n][0], name=NAMES[n][1], count=1)  # n items in n slots, as the engine gives them
                      for n, c in (p.split(':') for p in cfg['seedItems'].split(',')) for _ in range(int(c))]
        _, mx, mz, lx, lz = map(int, cfg['spawnTele'].split(','))
        self.x, self.z = mx * 64 + lx, mz * 64 + lz
        self.hp = level_of(self.xp['Hitpoints'])
        if 'startHp' in cfg:  # free-play-v12, v14, v15: the record's drawn hitpoints, under the base the levels set
            assert 1 <= cfg['startHp'] <= self.hp, cfg['startHp']
            self.hp = cfg['startHp']
        self.worn, self.dead, self.capture, self.offer = [], False, None, False
        self.ground = [self.clue()] if self.arm_ground and self.life == 0 else []
        return self.obs()

    def step(self, action):
        if self.sleep:
            time.sleep(self.sleep)
        held, entry = self.held(), page_entry(action)  # a pack option's entry: rows 29..36, or act-v4's 96..103
        if self.rocks is not None and (entry is None or entry[0] != A_HELD0):  # a Wield at setup as in any world
            return self.mine_step(action, entry)
        if self.contract == ITEM_V4 and entry is not None and entry[0] == A_DROP0:  # mask-v6's drop: the kind's first slot
            self.items[self.kinds()[entry[1]]['slot']] = None
        elif entry is not None and entry[0] == A_HELD0 and entry[1] < len(held):
            h = held[entry[1]]
            item, self.items[h['slot']] = self.items[h['slot']], None
            if h['option'] == 'Eat':  # the combat capture records each change of the agent's hitpoints
                before, self.hp = self.hp, min(level_of(self.xp['Hitpoints']), self.hp + 3)
                if self.capture:
                    self.combat(self.capture, kind='agent-hp', hpBefore=before, hpAfter=self.hp)
            else:
                self.worn.append(item)
        if self.rocks is None:
            self.x += 1  # one tile a decision
        c = self.capture
        wait = self.waits.get((self.life, self.tick - c['start']), self.step_ticks) if c else self.step_ticks
        for _ in range(wait):  # a step's ticks run out whatever happens in them (env-ts/engine/agent.ts step)
            self.advance()
        return self.obs()

    def mine_step(self, action, entry):
        """A step of the mining world (class docstring)."""
        rows, mined = self.rock_rows(), None
        if entry is not None and entry[0] == A_DROP0 and self.items[entry[1]] is not None:
            self.items[entry[1]] = None
        elif A_ACT0 <= action < A_ACT0 + len(rows):
            rock = rows[action - A_ACT0]
            self.x, self.z = rock['x'] + 1, rock['z']
            ore, _, level = W.MINE_ROCKS[(rock['x'], rock['z'])]
            if self.rocks[(rock['x'], rock['z'])] == 0 and level_of(self.xp['Mining']) >= level:
                mined = (rock['x'], rock['z']), ore
        for _ in range(self.step_ticks):
            self.advance()
        if mined and sum(1 for i in self.items if i) < N_INV:
            before, (ident, name) = self.holdings(), ORES[mined[1]]
            self.add(dict(id=ident, name=name, count=1))
            self.xp['Mining'] += 35 if mined[1] == 'iron' else 17
            self.rocks[mined[0]] = REGROW
            c = self.capture
            if c and not c['closed']:
                c['events'].append(dict(kind='native-op', tick=self.tick, before=before, after=self.holdings(), opcode=4302,
                                        productionSource=True, scriptId=52, scriptName='[label,mining]', pc=5,
                                        stackTail=[93, ident, 1], ground=[], resourceMutation=True, ownedSlotSwap=None))
        return self.obs()

    def rock_rows(self):
        """The rocks within 15 tiles as the engine's act rows give them, nearest first."""
        near = sorted((max(abs(x - self.x), abs(z - self.z)), x, z) for x, z in self.rocks
                      if max(abs(x - self.x), abs(z - self.z)) <= 15)
        return [dict(kind='loc', id=W.MINE_ROCKS[(x, z)][1] if self.rocks[(x, z)] == 0 else EMPTY, name='Rocks',
                     option='Mine', x=x, z=z, level=0, dx=x - self.x, dz=z - self.z, distance=d, steps=d, count=1,
                     combatLevel=0) for d, x, z in near]

    def close(self):
        self.calls.append('close')

    def advance(self):
        self.tick += 1
        if self.rocks is not None:
            self.rocks = {tile: max(0, left - 1) for tile, left in self.rocks.items()}
        c = self.capture
        if c and not c['closed']:
            c['events'].append(dict(kind='tick-boundary', tick=self.tick, holdings=self.holdings()))  # the cycle's end
            if self.tick == c['end']:
                self.close_capture('horizon')
        for event in self.script.get((self.life, self.tick - c['start']), ()) if c else ():
            self.apply(c, event)

    def combat(self, c, **event):
        if not c['closed'] and c['combat_failure'] is None:
            c['combat'].append(dict(event, tick=self.tick))

    def apply(self, c, event):
        kind, open_ = event[0], not c['closed']
        if kind == 'log':
            before = self.holdings()
            self.add(dict(id=LOGS, name='Logs', count=1))
            if open_:
                c['events'].append(dict(kind='native-op', tick=self.tick, before=before, after=self.holdings(), opcode=4302,
                                        productionSource=True, scriptId=51, scriptName='[label,get_logs]', pc=5,
                                        stackTail=[93, LOGS, 1], ground=[], resourceMutation=True, ownedSlotSwap=None))
            self.xp['Woodcutting'] += 25
        elif kind == 'xp':
            self.xp[event[1]] += event[2]
        elif kind == 'hit':
            nid, damage = event[1], event[2]
            life = c['lives'].setdefault(nid, dict(key=f'{nid}:0', hp=3, seen=False))
            if not life['seen']:
                self.combat(c, kind='life', life=life['key'], nid=nid, type=41 if nid == CHICKEN else 100, hp=3)
                life['seen'] = True
            self.combat(c, kind='hit', life=life['key'], hitter='agent', damage=damage)
            after = max(0, life['hp'] - damage)
            self.combat(c, kind='damage', life=life['key'], hitter='agent', hpBefore=life['hp'], hpAfter=after)
            life['hp'] = after
            self.xp['Attack'] += 4 * damage
            self.xp['Hitpoints'] += damage
        elif kind == 'hurt':
            self.combat(c, kind='agent-hit', damage=event[2], nid=event[1], type=100, hp=self.hp)
            self.combat(c, kind='agent-hp', hpBefore=self.hp, hpAfter=self.hp - event[2])
            self.hp -= event[2]
        elif kind == 'roll':
            self.combat(c, kind='agent-hit', damage=event[2], nid=event[1], type=100, hp=self.hp)
        elif kind == 'land':
            self.combat(c, kind='agent-hp', hpBefore=self.hp, hpAfter=self.hp - event[1])
            self.hp -= event[1]
        elif kind == 'lethal':  # the lethal hit: the death closes the capture in the cycle, before any agent-hp row
            self.combat(c, kind='agent-hit', damage=self.hp, nid=event[1], type=100, hp=self.hp)
            self.hp, self.dead = 0, True
            if open_:
                self.close_capture('death')
        elif kind == 'combat_fail':
            c['combat_failure'] = c['combat_failure'] or event[1]
        elif kind == 'clue':
            self.ground.append(self.clue())
        elif kind == 'offer':
            self.offer = True
        elif kind == 'take':
            self.ground, self.offer = [], False
            self.add(dict(id=CLUE, name='Clue scroll', count=1))
        elif kind == 'loot':
            self.add(dict(id=event[1], name=event[2], count=1))
        elif kind == 'lose':
            self.items = [None if i and i['id'] == event[1] else i for i in self.items]
        elif kind == 'lose1':
            self.items[next(k for k, i in enumerate(self.items) if i and i['id'] == event[1])] = None
        elif kind == 'fail':
            raise RuntimeError('engine-env closed the connection')
        elif kind == 'hang':
            self.release.wait(30)

    def clue(self):
        return dict(id=CLUE, name='Clue scroll', count=1, dx=1, dz=0, distance=1, x=self.x + 1, z=self.z)

    def add(self, item):
        free = next((k for k, i in enumerate(self.items) if i is None), len(self.items))
        self.items[free:free + 1] = [item]

    def holdings(self):
        return dict(inventory=[dict(slot=k, id=i['id'], count=i['count']) for k, i in enumerate(self.items)
                               if i and (i['id'] == LOGS or i['id'] in ORE_IDS)], bank=[])

    def close_capture(self, reason):
        c = self.capture
        c['terminal'] = dict(reason=reason, tick=self.tick, holdings=self.holdings(), toolAvailable=True,
                             lifecycle=dict(phase='dying' if reason == 'death' else 'alive', failure=None,
                                            deathSequence=int(reason == 'death'), respawnSequence=0,
                                            deathTick=self.tick if reason == 'death' else None))
        c['events'].append(dict(kind='terminal', **c['terminal']))
        c['closed'] = True

    def per(self):
        """The entries of the life's inventory page: item-v4's 16, else 8."""
        return N_PAGE if self.contract == ITEM_V4 else 8

    def kinds(self):
        """item-v3's and item-v4's page of kinds as the engine shows it (env-ts/core/items.ts kindPage): the pack's first
        8 (16 under item-v4) kinds in the order of their first slots, each its first slot's item with the kind's total."""
        kinds = {}
        for k, item in enumerate(self.items):
            if item:
                kinds.setdefault(item['id'], dict(slot=k, id=item['id'], count=item['count'], name=item['name'],
                                                  total=0))['total'] += item['count']
        return list(kinds.values())[:self.per()]

    @staticmethod
    def option(name):
        """An item's one option the fake offers: Eat for food, Wield for a weapon, else none."""
        return 'Eat' if name in ('Shrimps', 'Bread') else 'Wield' if name in ('Bronze dagger', 'Bronze sword') else None

    def held(self):
        out = []
        page = self.kinds() if self.contract in ('item-v3', ITEM_V4) else \
            [dict(slot=k, **item) for k, item in enumerate(self.items[:8]) if item is not None]
        for item in page:
            option = self.option(item['name'])
            if option:
                out.append(dict(slot=item['slot'], id=item['id'], count=item['count'], option=option, name=item['name']))
        return out

    def obs(self):
        o = plain_raw()
        held = self.held()
        mask = [0] * (N_ACTIONS_V4 if self.contract == ITEM_V4 else 96)  # act-v4's 128 rows under item-v4 (mask-v6)
        mask[A_WAIT] = mask[A_STYLE0] = 1
        for k in range(len(held)):
            mask[page_row(A_HELD0, k)] = 1
        if self.contract == ITEM_V4:  # mask-v6: a drop legal for every entry that holds an item (the pack is never full)
            for k in range(len(self.kinds())):
                mask[page_row(A_DROP0, k)] = 1
        empty = dict(slot=-1, id=-1, count=0, name='')
        page = (self.items + [None] * 8)[:8]
        offer = [dict(kind='obj', id=CLUE, name='Clue scroll', option='Take', x=self.x + 9, z=self.z, level=0, dx=9, dz=0,
                      distance=9, steps=9, count=1, combatLevel=0)] if self.offer else []
        levels = {n: level_of(xp) for n, xp in self.xp.items()}
        inv_sel = [dict(slot=k, id=i['id'], count=i['count'], name=i['name'], options=[]) if i else dict(empty)
                   for k, i in enumerate(page)] if self.contract not in ('item-v3', ITEM_V4) else \
            [dict(i, options=[]) for i in self.kinds()] + [dict(empty, total=0)] * (self.per() - len(self.kinds()))
        if self.contract == ITEM_V4:  # its page entries carry their items' own options, as the engine's do
            inv_sel = [dict(e, options=[self.option(e['name'])] if self.option(e['name']) else []) for e in inv_sel]
        if self.contract == ITEM_V4:
            o['maskVersion'] = MASK_V6
        if self.rocks is not None:  # the mine's rows, and item-v2's drop of an occupied entry of page 0
            offer = self.rock_rows()
            for k in range(len(offer)):
                mask[A_ACT0 + k] = 1
            for k, item in enumerate(page):
                mask[A_DROP0 + k] = int(item is not None)
        o.update(tick=self.tick, mask=mask, heldSel=held, actSel=offer, ground=list(self.ground), bankOpen=False,
                 invSel=inv_sel, contractVersion=self.contract,  # the life's contract, stamped; item-v3's page of kinds
                 inventory=[dict(slot=k, **i, options=[]) for k, i in enumerate(self.items) if i],
                 equipment=[dict(slot=3, id=i['id'], name=i['name']) for i in self.worn],
                 player=dict(x=self.x, z=self.z, level=0, hp=self.hp, maxHp=levels['Hitpoints'], isDead=self.dead),
                 skills=[dict(name=n, level=levels[n], baseLevel=levels[n], xp=xp) for n, xp in self.xp.items()],
                 npcs=[dict(id=GOBLIN, name='Goblin', dx=1, dz=0, distance=1, combatLevel=2, options=['Attack']),
                       dict(id=CHICKEN, name='Chicken', dx=2, dz=0, distance=2, combatLevel=1, options=['Attack'])],
                 done=self.dead)
        return o

    # -- resource control --------------------------------------------------------------------------------------------
    def identity(self):
        return dict(worldBootId='boot', accountId='agent', accountEpoch=self.epoch, incarnationId=f'p{self.epoch}')

    def control(self, operation, **kwargs):
        self.calls.append(operation)
        identity = self.identity()
        if operation == 'identity':
            return dict(type='resource_identity', identity=identity)
        assert kwargs['identity'] == identity
        if operation == 'arm':
            assert kwargs['targets'] == 'gathering-v1'
            self.capture = dict(start=self.tick, end=self.tick + kwargs['horizon'], events=[], combat=[], lives={},
                                closed=False, terminal=None, retired=False, combat_failure=None)
            return dict(type='resource_capture', identity=identity, audit=self.audit())
        c = self.capture
        if operation == 'budget':
            tick = c['terminal']['tick'] if c['terminal'] else min(self.tick, c['end'])
            return dict(type='resource_budget', identity=identity, budget=dict(
                schema='resource-budget-v1', startTick=c['start'], endTick=c['end'], observedTick=tick,
                horizon=c['end'] - c['start'], remainingTicks=c['end'] - tick,
                status=c['terminal']['reason'] if c['terminal'] else 'active'))
        if operation == 'retire':
            c['retired'], c['closed'] = True, True
        return dict(type='resource_capture', identity=identity, audit=self.audit())

    def audit(self):
        c = self.capture
        return dict(schema='native-resource-capture-v2', flowEvidenceVersion=1, slotSwapEvidenceVersion=1,
                    ownedContainers=deepcopy(CONTAINERS), targetScope='gathering-v1', targets=deepcopy(DECLARED),
                    identity=self.identity(), initial=dict(inventory=[], bank=[]), last=self.holdings(),
                    startTick=c['start'], endTick=c['end'],
                    observedTick=c['terminal']['tick'] if c['terminal'] else min(self.tick, c['end']),
                    events=[dict(e, sequence=k) for k, e in enumerate(c['events'], 1)], terminal=deepcopy(c['terminal']),
                    failure=W.RETIRED if c['retired'] else None, closed=c['closed'],
                    completed=bool(c['terminal']) and c['terminal']['reason'] == 'horizon',
                    evidenceComplete=bool(c['terminal']), lifecycleObserved=True,
                    combat=dict(schema='native-combat-capture-v1', agentHp=c.get('armed_hp'), failure=c['combat_failure'],
                                events=[dict(e, sequence=k) for k, e in enumerate(c['combat'], 1)]))


class Service:
    """baseline/world.py ResourceService's face: each call reads a status row (lockstep, as the stack's worlds report)
    into `samples` and its `record`, then answers from the fake world's control."""

    def __init__(self, fake):
        self.fake, self.samples, self.record = fake, [], lambda kind, **data: None

    def __call__(self, operation, **kwargs):
        row = dict(schema=1, bootId='boot', tick=self.fake.tick, lockstep=True)
        self.samples.append(row)
        self.record('server_clock', server=row)
        reply = self.fake.control(operation, **kwargs)
        if operation == 'arm':  # the combat capture binds at arm with the agent's hitpoints
            self.fake.capture['armed_hp'] = self.fake.hp
            reply['audit'] = self.fake.audit()
        return reply


class Tables:
    """The objects of excluded_items' rules that these worlds drop or hold: a clue scroll (a trail category), logs and a
    cabbage (loot outside village-world-v1's slice, inside village-world-v2's); the pick rows that village-world-v2's
    archive holds, unless `picks` is False."""
    obj_types = {CLUE: dict(debugname='trail_clue_easy_simple001', name='Clue scroll', category=7),
                 LOGS: dict(debugname='logs', name='Logs', category=1), CABBAGE: dict(debugname='cabbage', name='Cabbage')}

    def __init__(self, picks=True):
        self.meta = dict(content=dict(categories=dict(trail_clue_easy=7, logs=1), **(dict(picks={}) if picks else {})))


def launch(tmp_path, fakes, **kwargs):
    """watch_run over `fakes`, one a world, each behind a Service, with the test sites and WAIT unless told otherwise."""
    worlds = [dict(env_port=k, status_url='http://127.0.0.1/status', boot_id=f'b{k}') for k in range(len(fakes))]
    settings = dict(world=V1, actor='wait', sites=list(SITES), rotation=0, key=5, seed=9, ticks=1000, guard=60.0, rehearsal=True,
                    worlds=worlds, join=lambda world, record, k: (fakes[k], Service(fakes[k])), tables=Tables(),
                    environ=ENVIRON, tree=TREE)
    settings.update(kwargs)
    return W.watch_run(tmp_path / 'run', **settings)


@pytest.fixture(autouse=True)
def sites(monkeypatch):
    from baseline import evaluate
    monkeypatch.setattr(W, 'SITES', dict(SITES))
    monkeypatch.setattr(config, 'XP_RATE', 1)
    monkeypatch.setattr(evaluate, 'engine_tree', lambda engine=None: TREE)  # check's read of the vendor checkout


def test_a_death_starts_the_next_life_from_the_next_record_with_the_ticks_left(tmp_path):
    fakes = [Fake({(0, 300): [('lethal', GOBLIN)]}), Fake(), Fake(), Fake()]
    result = launch(tmp_path, fakes)
    assert result['status'] == 'complete' and [r['status'] for r in result['results']] == ['complete'] * 4
    first, second = result['results'][0]['lives']
    assert (first['reason'], first['ticks'], first['horizon']) == ('death', 300, 1000)
    assert (second['reason'], second['ticks'], second['horizon'], second['record']) == ('horizon', 700, 700, 1)
    assert all(r['armed_ticks'] == 1000 and r['lockstep'] is True for r in result['results'])
    # the next life is a fresh incarnation of the same world, from world 0's second start record
    records = W.Records(5, 0, 'danger')
    drawn = [records.next(), records.next()]
    assert [cfg['setStats'] for _, cfg in fakes[0].cfgs] == [d['set_stats'] for d in drawn]
    assert [cfg['seedItems'] for _, cfg in fakes[0].cfgs] == [d['seed_items'] for d in drawn]
    assert [seed for seed, _ in fakes[0].cfgs] == [d['reset_seed'] for d in drawn]
    assert fakes[0].epoch == 2 and fakes[0].calls.count('release') == 2 and fakes[0].calls[-1] == 'close'
    assert [len(r['lives']) for r in result['results']] == [2, 1, 1, 1]
    assert [cfg['maxEpisodeTicks'] >= 2 * MAX_HORIZON for _, cfg in fakes[1].cfgs] == [True]
    # the status rows the service read go to the world's own file; world_lives keeps none of them
    from baseline.world import read_jsonl
    assert Counter(r['kind'] for r in read_jsonl(tmp_path / 'run/world-1.jsonl.gz'))['server_clock'] > 0
    assert not any(row['kind'] == 'server_clock' for _, rows in W.world_lives(tmp_path / 'run/world-1.jsonl.gz') for row in rows)


def test_a_watch_keeps_one_control_connection_a_world_over_some_hundred_decisions_and_plays_as_before(tmp_path):
    """A world's control calls (a status row a second, two budget reads a decision) share one kept connection to
    its engine's control port, counted by the port, and the run is the one that the in-process service plays."""
    from contextlib import ExitStack
    from baseline.tests.test_world import ControlServer
    from baseline.world import ResourceService

    def port(k, fake):  # the engine's control port in front of a fake world, as demo-services answers
        def answer(method, path, body):
            if method == 'GET':
                return 200, dict(schema=1, bootId=f'b{k}', tick=fake.tick, tickRateMs=config.SPEED_MS, xpRate=config.XP_RATE,
                                 monotonicMs=time.monotonic() * 1000, players=1, actors=1, lockstep=True)
            reply = fake.control(body['operation'], **{key: body[key] for key in ('identity', 'horizon', 'targets') if key in body})
            if body['operation'] == 'arm':  # as Service: the combat capture binds at arm with the agent's hitpoints
                fake.capture['armed_hp'] = fake.hp
                reply['audit'] = fake.audit()
            return 200, reply
        return ControlServer(answer)
    fakes, plain = [Fake({(0, 300): [('lethal', GOBLIN)]}), Fake(), Fake(), Fake()], [Fake({(0, 300): [('lethal', GOBLIN)]}), Fake(), Fake(), Fake()]
    with ExitStack() as stack:
        ports = [stack.enter_context(port(k, fake)) for k, fake in enumerate(fakes)]
        join = lambda world, record, k: (fakes[k], ResourceService(ports[k].url, f'b{k}', 'agent', record,
                                                                   speed_ms=config.SPEED_MS, xp_rate=config.XP_RATE))
        result = launch(tmp_path / 'http', fakes, join=join)
    expected = launch(tmp_path / 'plain', plain)
    played = lambda run: [(r['status'], r['armed_ticks'], [(life['status'], life['reason'], life['ticks'], life['decisions'])
                                                            for life in r['lives']]) for r in run['results']]
    assert result['status'] == expected['status'] == 'complete' and played(result) == played(expected)
    assert [fake.calls for fake in fakes] == [fake.calls for fake in plain]  # the same control calls in the same order
    assert all(sum(life['decisions'] for life in r['lives']) > 300 for r in result['results'])
    assert [p.connections for p in ports] == [1] * 4 and all(p.requests > 600 for p in ports)


def test_each_world_plays_its_rotated_site_from_its_own_record_stream(tmp_path):
    fakes = [Fake() for _ in range(4)]
    result = launch(tmp_path, fakes, rotation=1, ticks=60)
    order = [list(SITES)[(k + 1) % 4] for k in range(4)]
    assert result['order'] == order and [r['site'] for r in result['results']] == order
    for k, fake in enumerate(fakes):
        (seed, cfg), = fake.cfgs
        assert cfg['spawnTele'] == W.tele(*SITES[order[k]]) and result['results'][k]['tile'] == list(SITES[order[k]])
        record = W.Records(5, k, 'danger').next()  # the stream stays with the world whatever site it plays
        assert (cfg['setStats'], seed) == (record['set_stats'], record['reset_seed'])


def test_the_guard_cuts_every_world_and_the_run_is_incomplete_not_a_result(tmp_path):
    fakes = [Fake(sleep=0.01) for _ in range(4)]
    result = launch(tmp_path, fakes, ticks=100_000, guard=0.3)
    assert result['status'] == 'incomplete' and all(r['status'] == 'cut' for r in result['results'])
    for fake, r in zip(fakes, result['results']):
        life = r['lives'][-1]
        assert (life['status'], life['reason']) == ('cut', 'guard') and 0 < life['ticks'] < 100_000
        assert fake.calls[-4:] == ['retire', 'read', 'release', 'close']  # the capture retired, its trace archived
    read = W.monitors(tmp_path / 'run')
    assert read['status'] == 'incomplete' and read['note'].startswith('incomplete')
    assert all(life['status'] == 'cut' for life in read['lives']) and read['summary'] == dict(lives=0)
    assert read['incomplete']['lives'] == 4 and not read['capture_failures']  # a retire is no capture failure


def test_an_excluded_pickup_ends_its_life_apart_and_leaves_the_run_complete(tmp_path):
    fakes = [Fake({(0, 30): [('clue',)], (0, 60): [('take',)]}), Fake({(0, 30): [('clue',)]}), Fake(arm_ground=True),
             Fake({(0, 30): [('offer',)]})]
    result = launch(tmp_path, fakes)
    first, second = result['results'][0]['lives'][:2]
    assert (first['status'], first['reason'], first['items'], first['excluded_seen'], first['ticks']) == \
        ('incomplete', 'excluded pickup', [CLUE], 30, 60)
    assert second['horizon'] == 1000 - 60 and result['results'][0]['status'] == 'complete'
    assert result['status'] == 'complete' and result['reasons'] == []  # the pickup life is apart, not a reason
    assert [(a['world'], a['life'], a['reason'], a['excluded_seen']) for a in result['apart']] == \
        [(0, 0, 'excluded pickup', 30), (1, 0, 'horizon', 30), (2, 0, 'horizon', 0), (3, 0, 'horizon', 30)]
    read = W.monitors(tmp_path / 'run')
    assert read['status'] == 'complete' and read['note'] is None
    assert read['apart']['pickup']['lives'] == 1 and read['apart']['flagged']['lives'] == 3
    assert read['summary']['lives'] == 1 and read['summary']['armed_ticks'] == 940  # the clean life only: world 0's next
    flagged = {r['world']: (r['excluded_seen'], r['excluded_items']) for r in read['lives'] if r['excluded_seen'] is not None}
    assert flagged == {0: (30, [CLUE]), 1: (30, [CLUE]), 2: (0, [CLUE]), 3: (30, [CLUE])}  # at arm; among the Take offers


def test_a_cabbage_ends_a_village_world_v1_life_and_not_a_village_world_v2_life(tmp_path):
    """The rehearsal's pickups of 13 November: a cabbage taken from a field, never in view as a drop. It is loot outside
    village-world-v1's slice and in village-world-v2's declared world (its field's Pick). A run.json that names no world
    is village-world-v1's, whatever it played."""
    script = {(0, 60): [('loot', CABBAGE, 'Cabbage')]}
    runs = {world: launch(tmp_path / world, [Fake(script), Fake(), Fake(), Fake()], world=world) for world in (V1, V2)}
    v1, v2 = (runs[world]['results'][0]['lives'] for world in (V1, V2))
    assert (v1[0]['status'], v1[0]['reason'], v1[0]['items'], v1[0]['excluded_seen'], v1[0]['ticks']) == \
        ('incomplete', 'excluded pickup', [CABBAGE], None, 60) and v1[1]['horizon'] == 1000 - 60
    assert [(life['status'], life['reason'], life['ticks']) for life in v2] == [('complete', 'horizon', 1000)]
    assert [a['reason'] for a in runs[V1]['apart']] == ['excluded pickup'] and runs[V2]['apart'] == []
    for world, result in runs.items():
        path = tmp_path / world / 'run' / 'run.json'
        saved = json.loads(path.read_text())
        assert result['status'] == 'complete' and saved['world'] == world and saved['rules'] == W.world_rules(world)
        assert (CABBAGE in saved['excluded']['ids']) == ('cabbage' in saved['rules']['loot_outside']) == (world == V1)
        assert W.monitors(path.parent)['world'] == world
    del saved['world']  # village-world-v2's run.json without its world
    path.write_text(json.dumps(saved))
    assert W.monitors(path.parent)['world'] == V1 == W.run_world(saved)


def test_village_world_v3_plays_v2s_rules_and_a_watch_plays_the_shore_sites_in_either_world(tmp_path, monkeypatch):
    """village-world-v3 (REAL_WORLDS, the real engine's only) applies village-world-v2's rules and plays over its
    archive: its watch is v2's (a cabbage ends no life), recorded as v3. The sites stay tied to SITES, not to the world:
    a v2 watch plays the two shore sites at their tiles as a v3 watch does. An unknown world is refused by name."""
    from sim.free import world_tables
    monkeypatch.setattr(W, 'SITES', REAL_SITES)
    four = ['rimmington_shore_east', 'rimmington_shore_west', 'port_sarim_north', 'draynor_west']
    script = {(0, 60): [('loot', CABBAGE, 'Cabbage')]}
    fakes = {world: [Fake(script), Fake(), Fake(), Fake()] for world in (V2, V3)}
    runs = {world: launch(tmp_path / world, fakes[world], world=world, sites=four) for world in (V2, V3)}
    saved = {world: json.loads((tmp_path / world / 'run' / 'run.json').read_text()) for world in (V2, V3)}
    for world in (V2, V3):
        assert runs[world]['status'] == 'complete' and saved[world]['world'] == world
        assert [cfg['spawnTele'] for fake in fakes[world] for _, cfg in fake.cfgs] == [W.tele(*REAL_SITES[s]) for s in four]
        assert [(life['reason'], life['ticks']) for life in runs[world]['results'][0]['lives']] == [('horizon', 1000)]
    assert saved[V3]['rules'] == saved[V2]['rules'] and saved[V3]['excluded'] == saved[V2]['excluded']
    assert world_tables(V3) == world_tables(V2) and W.village(V3)['content'] == 'picks'
    assert set(W.world_sites(V3)) == set(REAL_SITES) and W.world_sites(V2) == W.VILLAGE_SITES  # every site has its tile
    with pytest.raises(ValueError, match=r"world of \['village-world-v1', 'village-world-v2', 'village-world-v3'\]"):
        W.world_rules('village-world-v4')


def test_a_run_plays_over_its_worlds_archive_and_records_it(tables, tmp_path, monkeypatch, capsys):
    """Without a stand-in, a run (and `check`) loads the archive its world plays over (sim/free.py world_tables) and
    records its path and sha256; a missing archive is refused before any world is joined."""
    from baseline.telemetry import sha256
    from sim import free
    path = free.world_tables(V2)
    if not path.exists():
        pytest.skip(f'private simulator tables ({path.name}) absent; sim/build.py says how each archive was made')
    for world in (V1, V2):
        archive = free.world_tables(world)
        result = launch(tmp_path / world, [Fake() for _ in range(4)], world=world, tables=None)
        saved = json.loads((tmp_path / world / 'run' / 'run.json').read_text())
        assert result['status'] == 'complete' and saved['world'] == world
        assert saved['tables'] == dict(path=str(archive), sha256=sha256(archive))
        ids = {v['debugname']: k for k, v in tables.obj_types.items()}
        assert (ids['cabbage'] in saved['excluded']['ids']) == (world == V1)
        monkeypatch.setattr(sys, 'argv', ['sim.watch', 'check', '--output', str(tmp_path / f'check-{world}'), '--world',
                                          world, '--actor', 'wait', '--sites', *SITES, '--rotation', '0', '--key', '5',
                                          '--seed', '9'])
        W.main()
        assert f'{archive} (sha256 {sha256(archive)})' in capsys.readouterr().out
    assert sha256(free.world_tables(V1)) != sha256(path)
    joined = []
    monkeypatch.setattr(free, 'world_tables', lambda world: tmp_path / 'absent.npz')
    with pytest.raises(ValueError, match='absent.npz is missing: village-world-v2 plays over it'):
        launch(tmp_path / 'missing', [None] * 4, world=V2, tables=None, join=lambda w, r, k: joined.append(k))
    assert joined == [] and not (tmp_path / 'missing').exists()
    for argv in (['run', '--output', str(tmp_path / 'cli')], ['check', '--output', str(tmp_path / 'cli')]):  # no default
        monkeypatch.setattr(sys, 'argv', ['sim.watch', *argv, '--actor', 'wait', '--sites', *SITES, '--rotation', '0',
                                          '--key', '5', '--seed', '9'])
        with pytest.raises(SystemExit):
            W.main()
        assert '--world' in capsys.readouterr().err
    assert not (tmp_path / 'cli').exists()


@pytest.fixture(scope='module')
def tiny(tables, tmp_path_factory, name_table_path):
    """A tiny free-play-v1 run of village-world-v1 through the training command (sim/tests/test_free_play_eval.py)."""
    from sim.tests.test_free_play_eval import train_tiny
    return train_tiny(tmp_path_factory.mktemp('tiny'), tables, name_table_path)


def test_a_policy_of_another_world_is_refused_before_any_native_world_starts(tiny, tmp_path, monkeypatch):
    """A policy actor plays only a run that trained in the watch's world over that world's archive (sim/hour.py
    policy_run, trained_world): a village-world-v1 run is refused by a village-world-v2 watch, the same run relabelled
    village-world-v2 over village-world-v2's archive by a village-world-v1 watch, and the run relabelled
    village-world-v2 over village-world-v1's archive by a village-world-v2 watch; by `run` before any world is joined
    and by `check`, which needs no stack."""
    from baseline.telemetry import sha256
    from sim import free
    from sim.tests.test_free_play_eval import TINY, run_copy
    archive = free.world_tables(V2)
    if not archive.exists():
        pytest.skip(f'private simulator tables ({archive.name}) absent; sim/build.py says how each archive was made')
    other = run_copy(tiny, tmp_path / 'other', batch=dict(world=V2), tables=dict(path=str(archive), sha256=sha256(archive)))
    wrong = run_copy(tiny, tmp_path / 'wrong', batch=dict(world=V2))  # the first world's archive record kept
    joined = []
    refusals = ((V2, tiny, 'trained in another world than this evaluation: world differ'),
                (V1, other, 'trained in another world than this evaluation: world differ'),
                (V2, wrong, rf"trained over another archive \({sha256(free.world_tables(V1))}\) than {V2}'s"))
    for world, run, reason in refusals:
        for actor in ('policy', 'initial'):
            with pytest.raises(ValueError, match=reason):
                launch(tmp_path / f'{world}-{actor}', [None] * 4, world=world, actor=actor, run=run, policy_seed=TINY,
                       join=lambda w, r, k: joined.append(k))
        monkeypatch.setattr(sys, 'argv', ['sim.watch', 'check', '--output', str(tmp_path / 'check'), '--world', world,
                                          '--actor', 'policy', '--run', str(run), '--policy-seed', str(TINY), '--sites',
                                          *SITES, '--rotation', '0', '--key', '5', '--seed', '9'])
        with pytest.raises(SystemExit, match=reason):
            W.main()
    assert joined == [] and not any(p.name.startswith('village-world') for p in tmp_path.iterdir())
    for world, run in ((V1, tiny), (V2, other)):
        plays = W.check_settings(output=tmp_path / 'checked', world=world, actor='policy', sites=list(SITES), rotation=0,
                                 key=5, seed=9, ticks=1000, guard=60.0, run=run, policy_seed=TINY, tables=Tables())
        assert plays['facts']['run']['batch_in_effect']['world'] == world and set(plays['models']) == {'policy'}


def test_a_real_engine_checkpoint_plays_under_its_own_checks_and_records_its_training_run(tiny, tmp_path, monkeypatch):
    """--real-checkpoint FILE: one checkpoint of a real-engine run (sim.train --env real; here the tiny run's
    latest.zip, an obs-v6 free-play checkpoint, stands for one), which has no summary.json while incomplete, plays as
    `policy` under its own saved checks, and run.json records its path, sha256, tensor digest, decisions, the real engine,
    the env-ts commit this watch plays with and, with --launch, the training run's launch.json, start weights, batch
    settings and the env-ts commit of its source. Refused before any world, by `check` too: a launch of another engine
    or world, a checkpoint of another contract or name table, another actor or a --run beside it, a --launch alone."""
    import subprocess
    import names
    import numpy as np
    from baseline.telemetry import sha256
    from sim import free
    from sim.train import train
    from sim.workers import FREE_PLAY_TASK
    git = lambda *args: subprocess.run(['git', *args], cwd=Path(W.__file__).resolve().parents[1], capture_output=True,
                                       text=True, check=True).stdout.strip()
    head = git('rev-parse', 'HEAD')
    last = git('log', '-1', '--format=%H', head, '--', 'env-ts')  # the env-ts this tree plays, and the launch's source
    checkpoint, tiny_run = tiny / 'latest.zip', json.loads((tiny / 'summary.json').read_text())
    command = ['python', '-m', 'sim.train', 'train', '--env', 'real', '--task', FREE_PLAY_TASK, '--seed', '303', '--weights',
               '/data/starts/s303/latest.zip', '--batch-setting', f'starts="{V2}"', 'life=["uniform",6000,36000]',
               'agents=32', '--output', '/tmp/run/free']
    source = dict(commit=head, changed=['env-ts/core/observe.ts', 'sim/train.py'], untracked=[])
    record = dict(name='real-run', source=source, command=command, xp_rate=1)
    files = {}
    for name, changed in (('launch', {}), ('cpu', {5: 'cpu'}), ('v1', {13: f'starts="{V1}"'})):
        files[name] = tmp_path / f'{name}.json'
        files[name].write_text(json.dumps(dict(record, command=[changed.get(i, c) for i, c in enumerate(command)])))
    common = dict(world=V2, actor='policy', sites=list(SITES), rotation=0, key=5, seed=9, ticks=1000, guard=60.0)
    plays = W.check_settings(output=tmp_path / 'checked', real_checkpoint=checkpoint, launch=files['launch'], tables=Tables(),
                             **common)
    assert set(plays['models']) == {'policy'} and set(plays['facts']) == {'policy', 'run'}
    facts = plays['facts']
    assert (facts['policy']['path'], facts['policy']['sha256'], facts['policy']['policy_sha256'], facts['policy']['observation'],
            facts['policy']['free_play']) == (str(checkpoint), sha256(checkpoint), tiny_run['policy_sha256_at_end'], 'obs-v6',
                                              True)
    played = facts['run']['env_ts']['played']
    assert played['commit'] == last and isinstance(played['changed'], list)
    assert facts['run'] == dict(
        engine='real', checkpoint=str(checkpoint), decisions=tiny_run['decisions'], start='/data/starts/s303/latest.zip',
        launch=dict(path=str(files['launch']), sha256=sha256(files['launch']), name='real-run', command=command,
                    source=source),
        settings=dict(starts=V2, life=['uniform', 6000, 36000], agents=32),
        env_ts=dict(trained=last, trained_changed=['env-ts/core/observe.ts'], played=played),
        source="the checkpoint and the training run's launch.json")
    alone = W.check_settings(output=tmp_path / 'checked', real_checkpoint=checkpoint, tables=Tables(), **common)
    assert (alone['facts']['run']['launch'], alone['facts']['run']['start'], alone['facts']['run']['env_ts']['trained']) == \
        (None, None, None)

    def no_world(world, record, k):
        raise OSError('no world')
    stopped = launch(tmp_path, [None] * 4, real_checkpoint=checkpoint, launch=files['launch'], join=no_world, **common)
    assert stopped['status'] == 'stopped' and json.loads((tmp_path / 'run' / 'run.json').read_text())['policy'] == \
        json.loads(json.dumps(facts, default=str))
    other = train(tmp_path / 'fake', decisions=4 * 16, seed=1, worlds=4, n_steps=16, batch_size=16, n_epochs=1,
                  task=FREE_PLAY_TASK, verbose=False)
    assert other['free_play']['observation'] != 'obs-v6'
    table = tmp_path / 'other-names.npz'
    with np.load(os.environ['RL_NAME_TABLE']) as saved:
        np.savez(table, kind=saved['kind'], name=saved['name'], vector=saved['vector'] + np.float16(1))
    archive = tmp_path / 'archive.npz'  # the archive must exist; each refusal comes before it is read
    archive.write_bytes(b'')
    monkeypatch.setattr(free, 'world_tables', lambda world: archive)
    refusals = ((dict(launch=files['cpu']), 'is no free-play-v1 run .*of the real engine .*--env cpu'),
                (dict(launch=files['v1']), f'in {V2}: .*starts {V1}'),
                (dict(real_checkpoint=tmp_path / 'fake' / 'latest.zip'), 'not obs-v6'),
                (dict(actor='initial'), 'plays as the policy actor, alone'),
                (dict(run=tmp_path / 'fake', policy_seed=1), 'plays as the policy actor, alone'),
                (dict(real_checkpoint=None, run=tmp_path / 'fake', policy_seed=1), '--launch goes with --real-checkpoint'),
                (dict(names=table), 'another name table than RL_NAME_TABLE'))  # last: the tests' table is not restored
    for change, reason in refusals:
        given = dict(actor='policy', real_checkpoint=checkpoint, launch=files['launch'], run=None, policy_seed=None, names=None)
        given.update(change)
        if given['names'] is not None:
            monkeypatch.setenv('RL_NAME_TABLE', str(given['names']))
            names.name_table.cache_clear()
        argv = ['sim.watch', 'check', '--output', str(tmp_path / 'check'), '--world', V2, '--actor', given['actor'], '--sites',
                *SITES, '--rotation', '0', '--key', '5', '--seed', '9']
        for flag in ('real_checkpoint', 'launch', 'run', 'policy_seed'):
            if given[flag] is not None:
                argv += [f"--{flag.replace('_', '-')}", str(given[flag])]
        monkeypatch.setattr(sys, 'argv', argv)
        with pytest.raises(SystemExit, match=reason):
            W.main()
    assert not (tmp_path / 'check').exists()


def test_a_free_play_v2_checkpoint_plays_under_the_same_checks_with_its_version_and_price_recorded(tiny, tmp_path):
    """A free-play-v2 checkpoint of a real-engine run (it saves its free_play_task and death_price; here the tiny run's
    weights saved so) plays as `policy` under the checks of a free-play-v1 one, and run.json records the version and the
    price it saved (the watch pays none; a v1 checkpoint's record holds neither, as before). Refused before any world: a
    launch of the other version beside either, a checkpoint that names the version without a price or a price without
    the version, and a v2 checkpoint as a simulator run's policy."""
    from sb3_contrib import RecurrentPPO
    from sim.tests.test_free_play_eval import TINY
    v1, v2 = 'free-play-v1', W.FREE_PLAY_V2_NAME
    checkpoints, launches = {v1: tiny / 'latest.zip'}, {}
    for name, saved in ((v2, dict(free_play_task=v2, death_price=50.0)), ('no price', dict(free_play_task=v2)),
                        ('no name', dict(death_price=50.0))):
        variant = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
        variant.__dict__.update(saved)
        checkpoints[name] = tmp_path / f'{name}.zip'
        variant.save(str(checkpoints[name]))
    for version in (v1, v2):
        launches[version] = tmp_path / f'{version}.json'
        launches[version].write_text(json.dumps(dict(name=version, source=dict(commit='c'), xp_rate=1, command=[
            'python', '-m', 'sim.train', 'train', '--env', 'real', '--task', version,
            *(['--death-price', '50'] if version == v2 else []), '--batch-setting', f'starts="{V2}"'])))
    common = dict(output=tmp_path / 'checked', world=V2, actor='policy', sites=list(SITES), rotation=0, key=5, seed=9,
                  ticks=1000, guard=60.0, tables=Tables())
    for launch in (launches[v2], None):
        plays = W.check_settings(real_checkpoint=checkpoints[v2], launch=launch, **common)
        run, policy = plays['facts']['run'], plays['facts']['policy']
        assert (run['task'], run['death_price'], policy['observation'], policy['free_play']) == (v2, 50.0, 'obs-v6', True)
    v1_run = W.check_settings(real_checkpoint=checkpoints[v1], launch=launches[v1], **common)['facts']['run']
    assert not {'task', 'death_price'} & set(v1_run)
    for checkpoint, launch, reason in ((v2, v1, f'is no {v2} run'), (v1, v2, f'is no {v1} run'),
                                       ('no price', v2, 'names the task free-play-v2 and saves'),
                                       ('no name', v1, 'names the task None and saves')):
        with pytest.raises(ValueError, match=reason):
            W.check_settings(real_checkpoint=checkpoints[checkpoint], launch=launches[launch], **common)
    with pytest.raises(ValueError, match="a simulator run's policy is free-play-v1's"):
        W.load_actor('policy', tiny, TINY, checkpoints[v2], V1)


def test_a_free_play_v3_checkpoint_plays_with_its_tinderbox_and_its_monitors_add_every_stats_gain(tiny, tmp_path):
    """A free-play-v3 checkpoint of a real-engine run (it saves its free_play_task, its paid stats and its extra kit;
    here the tiny run's weights saved so) plays as `policy` under the checks of a free-play-v1 one: run.json records its
    version, paid stats and kit, every reset's kit holds the record's items and the tinderbox, and the monitors add every
    stat's level gain (Firemaking XP in a world's first life). A free-play-v1 checkpoint's watch keeps the record's kit and
    its records as before. A checkpoint that names v3 without its kit, a launch of another version, and a checkpoint of
    each removed study arm (sim/workers.py REMOVED) are refused, the last by name."""
    from sb3_contrib import RecurrentPPO
    from sim.workers import REMOVED
    v3, kit, stats = W.FREE_PLAY_V3_NAME, [['tinderbox', 1]], [s.lower() for s in SKILL_ORDER]
    for name, task, saved in (('v3', v3, dict(kit_extra=kit)), ('v3-no-kit', v3, {}),
                              *((removed, removed, dict(kit_extra=kit)) for removed in REMOVED)):
        model = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
        model.__dict__.update(free_play_task=task, paid_skills=stats, **saved)
        model.save(str(tmp_path / f'{name}.zip'))
    script = {(0, 30): [('xp', 'Firemaking', 2_000)]}
    plays = {}
    for name, checkpoint in (('v3', tmp_path / 'v3.zip'), ('v1', tiny / 'latest.zip')):
        fakes = [Fake(script), Fake(), Fake(), Fake()]
        result = launch(tmp_path / name, fakes, world=V2, actor='policy', real_checkpoint=checkpoint, ticks=300)
        cfgs = [cfg for f in fakes for _, cfg in f.cfgs]
        plays[name] = result, cfgs, W.monitors(tmp_path / name / 'run')
    result, cfgs, read = plays['v3']
    assert (result['policy']['run']['task'], result['policy']['run']['kit_extra']) == (v3, kit)
    assert result['policy']['run']['paid_skills'] == stats and cfgs
    assert all(cfg['seedItems'].endswith(',tinderbox:1') and cfg['seedItems'].count('tinderbox') == 1 for cfg in cfgs)
    every = [block['every_skill'] for block in (read['summary'], read['incomplete']) if block.get('lives')]
    assert every and any(e['lives_with_xp'].get('firemaking') for e in every)
    result, cfgs, read = plays['v1']
    assert not {'task', 'paid_skills', 'kit_extra'} & set(result['policy']['run'])
    assert cfgs and not any('tinderbox' in cfg['seedItems'] for cfg in cfgs)
    assert all('every_skill' not in block for block in (read['summary'], read['incomplete']))
    common = dict(output=tmp_path / 'checked', world=V2, actor='policy', sites=list(SITES), rotation=0, key=5, seed=9,
                  ticks=1000, guard=60.0, tables=Tables())
    launch_v1 = tmp_path / 'v1-launch.json'
    launch_v1.write_text(json.dumps(dict(name='v1', source=dict(commit='c'), xp_rate=1, command=[
        'python', '-m', 'sim.train', 'train', '--env', 'real', '--task', 'free-play-v1', '--batch-setting', f'starts="{V2}"'])))
    for checkpoint, launch_file, reason in ((tmp_path / 'v3-no-kit.zip', None, 'names the task free-play-v3 and saves'),
                                            (tmp_path / 'v3.zip', launch_v1, 'is no free-play-v3 run'),
                                            *((tmp_path / f'{removed}.zip', None,
                                               rf'is of {removed} \(.*\), a study arm removed from the code')
                                              for removed in REMOVED)):
        with pytest.raises(ValueError, match=reason):
            W.check_settings(real_checkpoint=checkpoint, launch=launch_file, **common)


def test_an_existing_output_folder_is_refused_before_any_world_starts_or_is_joined(tmp_path, monkeypatch):
    """A run writes a new folder: `check` refuses one that exists before scripts/stack.py starts any world, and `run`
    refuses it with its reason, not a traceback, before it joins a world. The folder keeps what it held."""
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'run.json').write_text('{"kept": true}')
    joined = []
    with pytest.raises(ValueError, match='run exists: a run writes a new folder'):
        launch(tmp_path, [None] * 4, join=lambda w, r, k: joined.append(k))
    for command in ('check', 'run'):
        monkeypatch.setattr(sys, 'argv', ['sim.watch', command, '--output', str(out), '--world', V1, '--actor', 'wait',
                                          '--sites', *SITES, '--rotation', '0', '--key', '5', '--seed', '9'])
        with pytest.raises(SystemExit, match=r'^\[watch\] .*/run exists: a run writes a new folder$'):
            W.main()
    assert joined == [] and [p.name for p in out.iterdir()] == ['run.json']
    assert json.loads((out / 'run.json').read_text()) == {'kept': True}


def test_check_refuses_an_engine_checkout_other_than_the_trials_build(tmp_path, monkeypatch):
    """`check` reads the engine checkout's tree (baseline/evaluate.py engine_tree, a local git read), as `run` does in
    check_stack, and refuses another build before scripts/stack.py starts any world."""
    from baseline import evaluate
    reads = []
    monkeypatch.setattr(evaluate, 'engine_tree', lambda engine=None: reads.append(engine) or 'other')
    monkeypatch.setattr(sys, 'argv', ['sim.watch', 'check', '--output', str(tmp_path / 'run'), '--world', V1, '--actor',
                                      'wait', '--sites', *SITES, '--rotation', '0', '--key', '5', '--seed', '9'])
    with pytest.raises(SystemExit, match=r'the engine checkout \(other\) is not the pinned build'):
        W.main()
    assert reads == [None] and not (tmp_path / 'run').exists()


def test_a_missing_engine_checkout_is_the_watchs_refusal_not_a_traceback(tmp_path, monkeypatch):
    """engine_tree runs git with check=True: on a folder that is no git checkout git fails (here a real git call on an
    empty folder outside every repository). `check` refuses it in its own words before scripts/stack.py starts, and
    `run` (watch_run, inside the stack) before it joins a world."""
    import subprocess
    from baseline import evaluate
    empty = tmp_path / 'rs-sdk'
    empty.mkdir()
    git = lambda engine=None: subprocess.run(['git', '-C', str(empty), 'status', '--porcelain'], capture_output=True,  # noqa: E731
                                             text=True, check=True, env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path))).stdout
    monkeypatch.setattr(evaluate, 'engine_tree', git)
    with pytest.raises(subprocess.CalledProcessError):  # the failure the refusal stands for
        git()
    joined = []
    with pytest.raises(ValueError, match=r'no engine checkout at vendor/rs-sdk \(CalledProcessError\)'):
        launch(tmp_path, [None] * 4, tree=None, join=lambda w, r, k: joined.append(k))
    monkeypatch.setattr(sys, 'argv', ['sim.watch', 'check', '--output', str(tmp_path / 'run'), '--world', V1, '--actor',
                                      'wait', '--sites', *SITES, '--rotation', '0', '--key', '5', '--seed', '9'])
    with pytest.raises(SystemExit, match=r'^\[watch\] no engine checkout at vendor/rs-sdk'):
        W.main()
    assert joined == [] and not (tmp_path / 'run').exists()


def test_the_monitors_account_the_saved_jsonl_as_the_world_played_it(tmp_path, monkeypatch):
    # life 0 of world 0: two logs, three goblin hits that kill it, a chicken hit, a goblin hurts the agent twice, an eat
    # (the actor's first decision at tick 100 or later), woodcutting XP that raises levels, a chisel taken, lost and taken
    # again, then a chicken hit and a goblin's lethal hit at tick 400 (in the step of ticks 400 to 402): the hit's XP
    # counts, and the combat capture never records the lethal hit's lost hitpoints
    script = {(0, 10): [('log',)], (0, 20): [('log',)], (0, 50): [('hit', GOBLIN, 1)], (0, 53): [('hit', GOBLIN, 2)],
              (0, 80): [('hit', CHICKEN, 1)], (0, 90): [('hurt', GOBLIN, 2)], (0, 95): [('hurt', GOBLIN, 3)],
              (0, 140): [('loot', CHISEL, 'Chisel')], (0, 150): [('lose', CHISEL)], (0, 170): [('loot', CHISEL, 'Chisel')],
              (0, 200): [('xp', 'Woodcutting', 10_000)], (0, 400): [('hit', CHICKEN, 1), ('lethal', GOBLIN)]}
    fakes = [Fake(script), Fake(), Fake(), Fake()]

    class EatOnce:
        def reset(self, seed):
            self.done = False

        def act(self, observation, task):
            tick = task.clock.last_tick - task.clock.start
            k = next((k for k, h in enumerate(task.raw['heldSel']) if h['option'] == 'Eat'), None)
            if not self.done and tick >= 100 and k is not None:
                self.done = True
                return A_HELD0 + k
            return A_WAIT
    monkeypatch.setattr(W, 'make_actor', lambda *a: EatOnce())
    launch(tmp_path, fakes, ticks=1000)
    read = W.monitors(tmp_path / 'run')
    life = read['lives'][0]
    record = W.Records(5, 0, 'danger').next()
    base_hp, wood = record['levels']['hitpoints'], record['levels']['woodcutting']
    assert life['made'] == {'logs': 2} and life['audit'] == 'match'
    assert life['kills'] == {'goblin': 1} and life['damage_dealt'] == {'goblin': 3, 'chicken': 2}
    # 2 + 3 hurt, then the lethal hit takes the base_hp - 5 + 3 (the eat) left: in damage taken, killer goblin
    assert life['damage_taken'] == {'goblin': 5 + base_hp - 2} == life['damage_planned']
    assert life['death'] == dict(tick=400, killer='goblin') and life['ticks'] == 400 and life['overrun'] == 2
    assert life['eats'] == 1 and life['eat_log'][0][1:] == [base_hp - 5, 3] and life['eats_full'] == 0
    # XP rose at ticks 12 (log 10, seen at the decision of tick 12), 21, 51, 54, 81, 201 and in the death step: wood,
    # wood, fight, fight, fight, wood, fight; ticks from the previous XP observation count for its activity, and a dead
    # agent gains nothing after its terminal, so the death step's stretch to the death goes to its fight
    assert [a[:2] for a in life['activity']] == [['woodcutting', 12], ['fight:goblin', 51], ['fight:chicken', 81],
                                                ['woodcutting', 201], ['fight:chicken', 400]]
    assert life['time'] == {'woodcutting': 21 + 120, 'fight:goblin': 54 - 21, 'fight:chicken': 81 - 54 + 400 - 201,
                            'none': 0}
    assert life['switches'] == 4 and sum(life['time'].values()) == life['ticks']
    # one tile east a decision, but not the death step's observation, which lies past the death
    assert life['distance'] == life['farthest'] == life['decisions'] - 1
    gained = level_of(whole_xp(wood) + 10_050) - wood
    assert gained > 1 and life['gain']['woodcutting'] == gained and life['levels1']['woodcutting'] == wood + gained
    assert life['xp'] == {'woodcutting': 10_050, 'attack': 20, 'hitpoints': 5} and life['xp_last_step'] == {}
    assert (life['terminal'], life['reason'], life['end_cycle_hp']) == ('death', 'death', None)
    assert life['new_items'] == {'Chisel': dict(tick=141, most=1)}  # taken again: still one new item, first at 141
    # 400 tick boundaries, 2 productions and the terminal; 2 npc lives with 3 hits and 3 damages, 3 attacks on the agent
    # and 3 changes of its hitpoints (2 hurts and the eat; the lethal hit's loss is not recorded), and the death step's hit
    assert life['capture'] == dict(events=403, combat=16, bound=20000, failure=None, combat_failure=None)
    assert life['setup_ticks'] == 3 * (1 + record['worn'])  # the style step and a Wield of a worn start
    world = read['worlds'][0]
    assert (world['lives'], world['deaths'], world['armed_ticks'], world['armed_ticks_run']) == (2, 1, 1000, 1000)
    assert read['summary']['made'] == {'logs': 2} and read['summary']['killers'] == {'goblin': 1}


def test_the_monitors_count_fish_cooked_and_burnt_and_the_eats_below_full_hitpoints(tmp_path, monkeypatch):
    # life 0 of world 0: three raw shrimps; one cooked (a cooked one comes and the cooking XP rises), one burnt (a burnt
    # fish comes, no XP), one dropped (neither); an eat at full hitpoints, a goblin's hit, then an eat below full
    script = {(0, 10): [('loot', RAW, 'Raw shrimps')] * 3,
              (0, 30): [('lose1', RAW), ('loot', NAMES['shrimp'][0], 'Shrimps'), ('xp', 'Cooking', 30)],
              (0, 50): [('lose1', RAW), ('loot', BURNT, 'Burnt fish')], (0, 70): [('lose1', RAW)],
              (0, 120): [('hurt', GOBLIN, 4)]}

    class EatTwice:
        def reset(self, seed):
            self.eaten = 0

        def act(self, observation, task):
            tick = task.clock.last_tick - task.clock.start
            k = next((k for k, h in enumerate(task.raw['heldSel']) if h['option'] == 'Eat'), None)
            if k is not None and self.eaten < 2 and tick >= (80, 150)[self.eaten]:
                self.eaten += 1
                return A_HELD0 + k
            return A_WAIT
    monkeypatch.setattr(W, 'make_actor', lambda *a: EatTwice())
    launch(tmp_path, [Fake(script), Fake(), Fake(), Fake()], ticks=1000)
    read = W.monitors(tmp_path / 'run')
    life = read['lives'][0]
    assert (life['fish_cooked'], life['fish_burnt']) == (1, 1) and life['xp']['cooking'] == 30
    assert (life['eats'], life['eats_full'], life['eats_below_full']) == (2, 1, 1)
    assert [e[1] < life['hp0'] for e in life['eat_log']] == [False, True]
    assert all((r['fish_cooked'], r['fish_burnt'], r['eats_below_full']) == (0, 0, 0) for r in read['lives'][1:])
    assert (read['summary']['fish_cooked'], read['summary']['fish_burnt'], read['summary']['eats_below_full']) == (1, 1, 1)


def test_the_step_past_the_horizon_is_read_apart_and_the_horizon_cycle_loss_where_it_can_be_read(tmp_path):
    # world 0's decision at tick 297 waits 30 ticks (a decision point), past the horizon of 300: mining XP and a chisel
    # come at 310. In worlds 0 and 1 a goblin's attack rolled at 298 lands in the horizon's own cycle, which the combat
    # capture does not record; world 1's last observation lies at the horizon, world 0's past it.
    landing = {(0, 298): [('roll', GOBLIN, 2)], (0, 300): [('land', 2)]}
    fakes = [Fake({(0, 150): [('xp', 'Fishing', 40)], (0, 310): [('xp', 'Mining', 5000), ('loot', CHISEL, 'Chisel')],
                   **landing}, waits={(0, 297): 30}), Fake(landing), Fake(), Fake()]
    result = launch(tmp_path, fakes, ticks=300)
    assert result['status'] == 'complete' and result['results'][0]['lives'][0]['ticks'] == 300
    read = W.monitors(tmp_path / 'run')
    life, at_horizon = read['lives'][0], read['lives'][1]
    mining = W.Records(5, 0, 'danger').next()['levels']['mining']
    assert (life['ticks'], life['overrun']) == (300, 27) and life['xp_last_step'] == {'mining': 5000}
    assert life['xp'] == {'fishing': 40} and life['levels1']['mining'] == mining and 'mining' not in life['gain']
    assert life['activity'] == [['fishing', 150, 150, 1]] and life['time'] == {'fishing': 150, 'none': 150}
    assert life['new_items'] == {} and life['distance'] == life['decisions'] - 1  # nothing past the horizon counts
    assert (life['end_cycle_hp'], life['damage_taken'], life['damage_planned']) == (None, {}, {'goblin': 2})
    assert (at_horizon['overrun'], at_horizon['end_cycle_hp'], at_horizon['damage_taken']) == (0, 2, {'goblin': 2})
    assert read['summary']['unusable'] == dict(combat=0, production=0, end_cycle_hp=1)


def test_a_combat_capture_failure_makes_its_life_incomplete_and_its_combat_reads_unusable(tmp_path):
    fakes = [Fake(), Fake({(0, 30): [('combat_fail', 'combat capture overflow')], (0, 40): [('hit', GOBLIN, 3)]}),
             Fake(), Fake()]
    result = launch(tmp_path, fakes, ticks=300)
    life = result['results'][1]['lives'][0]
    assert (life['status'], life['reason'], life['capture_failure']) == \
        ('incomplete', 'horizon', 'combat capture: combat capture overflow')  # the terminal's reason stays
    assert result['status'] == 'incomplete' and any('combat capture overflow' in r for r in result['reasons'])
    read = W.monitors(tmp_path / 'run')
    bad = next(r for r in read['lives'] if r['world'] == 1)
    assert (bad['kills'], bad['damage_taken'], bad['damage_dealt'], bad['damage_planned']) == (None, None, None, None)
    assert [a[0] for a in bad['activity']] == ['fight']  # the hit's XP stands, with no npc to name
    assert read['capture_failures'] == [dict(world=1, life=0, cause='combat capture: combat capture overflow')]
    assert read['summary']['lives'] == 3 and read['incomplete']['lives'] == 1
    assert read['incomplete']['unusable'] == dict(combat=1, production=0, end_cycle_hp=0)


def test_a_death_in_a_life_whose_combat_capture_failed_is_still_a_death(tmp_path):
    import gzip
    from baseline.world import read_jsonl
    fakes = [Fake(), Fake({(0, 30): [('combat_fail', 'enemy type changed within a life')], (0, 200): [('lethal', GOBLIN)]}),
             Fake(), Fake()]
    result = launch(tmp_path, fakes, ticks=300)
    life = result['results'][1]['lives'][0]
    assert (life['status'], life['reason'], life['ticks']) == ('incomplete', 'death', 200) and life['capture_failure']
    assert result['results'][1]['lives'][1]['horizon'] == 100  # the death started the next life
    path = tmp_path / 'run/world-1.jsonl.gz'

    def death_read():
        read = W.monitors(tmp_path / 'run')
        world = next(w for w in read['worlds'] if w['world'] == 1)
        dead = next(r for r in read['lives'] if r['world'] == 1 and r['life'] == 0)
        return world['deaths'], read['incomplete']['deaths'], dead['death'], dead['terminal']
    assert death_read() == (1, 1, dict(tick=200, killer=None), 'death')  # no killer: the combat reads are unusable
    rows = list(read_jsonl(path))  # the row as earlier code wrote it: its reason overwritten by the capture failure
    for row in rows:
        if row['kind'] == 'watch_life_end' and row['life'] == 0:
            row['reason'] = 'capture failure'
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        stream.writelines(json.dumps(row) + '\n' for row in rows)
    assert death_read() == (1, 1, dict(tick=200, killer=None), 'death')  # the trace's terminal still says death


def test_a_failed_hung_or_actorless_world_is_reported_with_its_own_cause_and_ticks(tmp_path, monkeypatch):
    monkeypatch.setitem(W.WATCH, 'grace', 0.5)
    hung = Fake({(0, 30): [('hang',)]})
    fakes = [Fake({(0, 90): [('fail',)]}), hung, Fake(), Fake()]
    original = W.make_actor

    def make(*args):  # world 2's actor cannot be made
        if threading.current_thread().name == 'world-2':
            raise RuntimeError('no actor')
        return original(*args)
    monkeypatch.setattr(W, 'make_actor', make)
    try:
        result = launch(tmp_path, fakes, ticks=300, guard=3.0)
        read = W.monitors(tmp_path / 'run')  # while world 1's thread is still held, so nothing writes its file meanwhile
    finally:
        hung.release.set()
    by = {r['world']: r for r in result['results']}
    failed = by[0]['lives'][0]
    assert by[0]['status'] == 'stopped' and 'engine-env closed the connection' in by[0]['cause']
    assert (failed['status'], failed['reason'], failed['ticks']) == ('incomplete', 'failure', 87)  # its last clock tick
    assert by[0]['armed_ticks'] == 87 and fakes[0].calls[-1] == 'close'
    assert by[1]['status'] == 'hung' and 'no end within the guard' in by[1]['cause']
    assert by[2]['status'] == 'stopped' and 'no actor' in by[2]['cause'] and by[2]['lives'] == []
    assert fakes[2].calls == ['close'] and by[3]['status'] == 'complete'
    assert result['status'] == 'incomplete'
    held = next(w for w in read['worlds'] if w['world'] == 1)
    assert (held['status'], held['lives']) == ('hung', 1)  # its open life, read up to its last row
    world = next(w for w in read['worlds'] if w['world'] == 0)
    assert world['armed_ticks'] == world['armed_ticks_run'] == 87  # run.json and the monitors agree
    assert next(r for r in read['lives'] if r['world'] == 0)['reason'] == 'failure'


def test_the_monitors_keep_the_status_and_cause_of_a_run_that_stopped_before_play(tmp_path):
    def join(world, record, k):
        if k == 2:
            raise ConnectionError('engine-env did not answer')
        return fakes[k], Service(fakes[k])
    fakes = [Fake() for _ in range(4)]
    result = launch(tmp_path, fakes, join=join)
    assert result['status'] == 'stopped' and 'results' not in result and fakes[0].calls == ['close']
    read = W.monitors(tmp_path / 'run')
    assert read['status'] == 'stopped' and read['note'].startswith('stopped') and read['worlds'] == []
    assert read['reasons'] == [result['cause']] and 'engine-env did not answer' in read['cause']


def test_the_monitors_read_an_unfinalized_run_from_its_world_files(tmp_path):
    launch(tmp_path, [Fake({(0, 100): [('lethal', CHICKEN)]}), Fake(), Fake(), Fake()], ticks=300)
    path = tmp_path / 'run/run.json'
    run = json.loads(path.read_text())
    for key in ('results', 'reasons', 'apart', 'seconds'):
        run.pop(key)
    path.write_text(json.dumps(dict(run, status='running')))  # as a run the stack killed before its last save
    read = W.monitors(tmp_path / 'run')
    assert read['status'] == 'unfinalized' and read['note'].startswith('unfinalized')
    assert [(w['world'], w['site'], w['status'], w['lives'], w['armed_ticks']) for w in read['worlds']] == \
        [(0, 'north', 'unfinalized', 2, 300), (1, 'south', 'unfinalized', 1, 300), (2, 'east', 'unfinalized', 1, 300),
         (3, 'west', 'unfinalized', 1, 300)]


def test_the_rehearsal_proposes_from_every_world_and_refuses_one_it_did_not_measure(tmp_path):
    def run(name, rows, rehearsal=True, status='complete', drop=False, world=None):  # world None: a record naming none
        folder = tmp_path / name
        folder.mkdir()
        results = [dict(world=k, site='s', status=s, armed_ticks=t, seconds=t / r if r else None, lives=[])
                   for k, (r, s, t) in enumerate(rows)]
        (folder / 'run.json').write_text(json.dumps(dict(rehearsal=rehearsal, actor=name, status=status,
                                                         **({} if drop else dict(results=results)),
                                                         **({} if world is None else dict(world=world)))))
        return folder
    ok = lambda rate, status='complete', ticks=36000: (rate, status, ticks)
    fast, slow = run('wait', [ok(400), ok(300), ok(350), ok(390)]), run('random', [ok(80), ok(90), ok(85), ok(95)])
    plan = W.propose([fast, slow])
    assert plan['slowest']['ticks_per_second'] == 80 and plan['fits_hours'] == 8 and plan['proposed_ticks'] == 36000
    assert [(w['actor'], w['world']) for w in plan['worlds']] == [('wait', k) for k in range(4)] + [('random', k) for k in range(4)]
    assert plan['world'] == V1 == W.propose([fast, run('named', [ok(90)] * 4, world=V1)])['world']  # no world named: v1's
    v2 = run('v2', [ok(90)] * 4, world=V2)
    assert W.propose([v2])['world'] == V2
    with pytest.raises(ValueError, match=f'more than one world: {V1}: {fast}; {V2}: {v2}'):
        W.propose([fast, v2])
    cut = run('initial', [ok(55, 'cut', 33000), ok(60), ok(70), ok(65)])  # a cut world counts up to its cut
    assert W.propose([cut])['proposed_ticks'] == 30000  # 600 s x 55/s = 33,000: five hours
    bad = {'hung': run('h', [ok(90), ok(None, 'hung', None), ok(90), ok(90)]),
           'stopped': run('s', [ok(90), ok(90), ok(90, 'stopped', 0)]),
           'not finalized': run('u', [], status='running', drop=True),
           'not a rehearsal': run('p', [ok(100)] * 4, rehearsal=False),
           '3 worlds': run('t', [ok(90)] * 3)}
    path = run('l', [ok(90)] * 4)
    lockstep = json.loads((path / 'run.json').read_text())
    lockstep['results'][3]['lockstep'] = False
    (path / 'run.json').write_text(json.dumps(lockstep))
    bad['not lockstep'] = path
    for word, folder in bad.items():
        with pytest.raises(ValueError, match=word):
            W.propose([fast, folder])
    with pytest.raises(ValueError, match=r'world 1 \(s\): hung.*world 2 \(s\): stopped'):
        W.propose([bad['hung'], run('s2', [ok(90), ok(90), ok(90, 'stopped', 0), ok(90)])])


def test_the_sites_rotate_over_the_worlds_and_each_record_stream_stays_with_its_world():
    sites = list(SITES)
    orders = [W.site_order(sites, r) for r in range(4)]
    assert all(sorted(o) == sorted(sites) for o in orders)  # every rotation plays all four sites
    assert all(sorted(o[k] for o in orders) == sorted(sites) for k in range(4))  # every world (record stream) meets each
    assert W.site_order(sites, 4) == orders[0]
    with pytest.raises(ValueError):
        W.site_order(sites[:3], 0)
    with pytest.raises(ValueError):
        W.site_order([sites[0]] * 4, 0)


def test_the_start_records_follow_the_free_play_start_draw_and_each_world_has_its_own_stream():
    streams = [W.Records(11, k, 'danger') for k in range(4)]
    drawn = [[s.next() for _ in range(200)] for s in streams]
    assert [r['index'] for r in drawn[0]] == list(range(200))
    again = W.Records(11, 0, 'danger')
    assert [again.next() for _ in range(200)] == drawn[0] and drawn[0] != drawn[1]
    for r in (r for rows in drawn for r in rows):
        assert all(r['levels'][s] in START_LEVELS[s] for s in START_LEVELS) and r['weapon'] in START_WEAPONS
        assert 1 <= len(r['food']) <= 3 and set(r['food']) <= {'shrimp', 'bread'}
        assert r['levels']['hitpoints'] - 3 <= r['hp_drawn'] <= r['levels']['hitpoints'] and 1 <= r['reset_seed'] < 10 ** 6
        assert r['seed_items'] == ','.join(f'{n}:1' for n in ('bronze_axe', 'net', 'bronze_pickaxe', *r['food'], r['weapon']))
        assert r['set_stats'] == ','.join(f'{s}:{r["levels"][s]}' for s in START_LEVELS)
    rows = [r for rows in drawn for r in rows]
    worn = sum(r['worn'] for r in rows) / len(rows)
    assert 0.45 < worn < 0.55 and Counter(len(r['food']) for r in rows).keys() == {1, 2, 3}


def test_a_launch_outside_the_declared_trial_is_refused_before_any_world_is_joined(tmp_path, monkeypatch):
    joined = []
    base = dict(worlds=[dict(env_port=k) for k in range(4)], join=lambda w, r, k: joined.append(k), tables=Tables())
    cases = [dict(environ={}), dict(environ=dict(NODE_RANDOM_EVENTS='true')), dict(tree='other'),
             dict(worlds=[dict(env_port=0)] * 3), dict(sites=['north', 'south', 'east', 'nowhere']), dict(ticks=0),
             dict(ticks=MAX_HORIZON + 1), dict(guard=0), dict(actor='rotation'), dict(key=None),
             dict(actor='wait', run=tmp_path), dict(actor='policy'), dict(actor='initial', policy_seed=77),
             dict(world=None), dict(world='village-world-v4'), dict(world=V2, tables=Tables(picks=False))]
    for k, case in enumerate(cases):
        with pytest.raises(ValueError):
            launch(tmp_path / str(k), [None] * 4, **{**base, **case})
    with pytest.raises(ValueError, match='village-world-v2 plays over an archive with the picks rows'):
        launch(tmp_path / 'picks', [None] * 4, **{**base, 'world': V2, 'tables': Tables(picks=False)})
    monkeypatch.setattr(config, 'XP_RATE', 25)
    with pytest.raises(ValueError, match='XP rate'):
        launch(tmp_path / 'xp', [None] * 4, **base)
    with pytest.raises(TypeError, match='world'):  # a run names its world: no default
        W.watch_run(tmp_path / 'none', actor='wait', sites=list(SITES), rotation=0, key=5, seed=9, **base)
    assert joined == [] and not any((tmp_path / str(k)).exists() for k in [*range(len(cases)), 'picks', 'xp', 'none'])


def test_each_policy_actor_keeps_its_own_draws_whatever_the_other_threads_do():
    import torch

    class Draw:  # a sampled actor: each act is one draw of torch's generator
        def reset(self, seed):
            torch.manual_seed(seed)

        def act(self, observation, task=None):
            return int(torch.randint(0, 10 ** 9, (1,)))
    alone = W.Locked(Draw(), threading.Lock())
    alone.reset(123)
    expected = [alone.act(None) for _ in range(50)]
    lock, out, together = threading.Lock(), {}, threading.Barrier(4)

    def play(k):  # the four threads' draws interleave: each waits for the others, then yields between draws
        actor = W.Locked(Draw(), lock)
        actor.reset(123 if k == 0 else k)
        together.wait()
        out[k] = []
        for _ in range(50):
            out[k].append(actor.act(None))
            time.sleep(0.001)
    threads = [threading.Thread(target=play, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] == expected and out[1] != expected


def test_excluded_items_follow_the_simulators_exclusions(tables):
    from sim.engine import loot_outside, trail_categories
    excluded = W.excluded_items(tables, V1)
    trail = trail_categories(tables)
    assert excluded and set(excluded.values()) == {'clue_drops', 'loot_drops'}
    for obj, why in excluded.items():
        assert (tables.obj_types[obj]['category'] in trail) == (why == 'clue_drops')
        assert why == 'clue_drops' or loot_outside(tables, obj)
    ids = {v['debugname']: k for k, v in tables.obj_types.items()}
    assert excluded[ids['beer']] == 'loot_drops' and ids['bronze_dagger'] not in excluded and ids['logs'] not in excluded
    v2 = W.excluded_items(tables, V2)  # village-world-v2's loot list: LOOT_OUTSIDE without the cabbage
    assert excluded[ids['cabbage']] == 'loot_drops' and {k: v for k, v in excluded.items() if k != ids['cabbage']} == v2


def test_the_env_ts_bounds_are_the_python_ones():
    control = (ROOT / 'env-ts/core/resource-control.ts').read_text()
    capture = (ROOT / 'env-ts/engine/resource-capture.ts').read_text()
    combat = (ROOT / 'env-ts/engine/combat-capture.ts').read_text()
    number = lambda pattern, text: int(re.search(pattern, text).group(1).replace('_', ''))
    assert number(r'MAX_RESOURCE_HORIZON=([\d_]+)', control) == MAX_HORIZON
    assert number(r'HOUR_TICKS = ([\d_]+)', capture) == W.WATCH['hour']
    assert number(r'installResourceCapture\(host: ResourceCaptureHost, maxEvents=([\d_]+)\)', capture) == W.WATCH['event_bound']
    assert number(r'installCombatCapture\(host: CombatCaptureHost, maxEvents = ([\d_]+)\)', combat) == W.WATCH['event_bound']
    assert [W.event_bound(t) for t in (1200, 6000, 6001, 36000)] == [20000, 20000, 40000, 120000]
    assert "'capture retired before complete horizon'" in capture and W.RETIRED == 'capture retired before complete horizon'



class Unstamped(Fake):
    """A world whose engine ignores the reset's item contract: its observations stamp item-v2 whatever the cfg names."""

    def reset(self, seed, cfg):
        super().reset(seed, cfg)
        self.contract = 'item-v2'
        return self.obs()


def test_a_free_play_v10_checkpoint_plays_its_kit_on_the_page_of_kinds_and_no_other_contract_plays(tiny, tmp_path):
    """A free-play-v10 checkpoint (its policy records item-v3) plays as `policy` under item-v3 with its kit: every reset's
    cfg names item-v3, each observation stamps it, the record names v10, and the setup's Wield finds the record's weapon
    among the page's kinds (a setup row for each life whose record starts it worn). A v10 checkpoint that records item-v2,
    or a v3 one that records item-v3, is refused before any world plays; a world whose observations stamp another
    contract than the life's ends each life incomplete and stops."""
    import gzip
    from sb3_contrib import RecurrentPPO
    from baseline.policy import run_contract
    kit, stats = [['tinderbox', 1]], [s.lower() for s in SKILL_ORDER]

    def tag(name, task, items):
        model = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
        model.__dict__.update(free_play_task=task, paid_skills=stats, kit_extra=kit)
        model.policy_kwargs = dict(model.policy_kwargs, contract=run_contract(items))
        model.save(str(tmp_path / f'{name}.zip'))
        return tmp_path / f'{name}.zip'
    v10 = tag('v10', W.FREE_PLAY_V10_NAME, 'item-v3')
    fakes = [Fake(), Fake(), Fake(), Fake()]
    result = launch(tmp_path / 'v10', fakes, world=V2, actor='policy', real_checkpoint=v10, ticks=300)
    cfgs = [cfg for f in fakes for _, cfg in f.cfgs]
    assert result['status'] == 'complete' and (result['policy']['run']['task'], result['policy']['run']['kit_extra']) == \
        (W.FREE_PLAY_V10_NAME, kit)
    assert cfgs and all(c['contract'] == 'item-v3' and c['seedItems'].endswith(',tinderbox:1') for c in cfgs)
    rows = [json.loads(line) for k in range(4) for line in gzip.open(tmp_path / 'v10' / 'run' / f'world-{k}.jsonl.gz', 'rt')]
    stamps = {r['observation']['contractVersion'] for r in rows if isinstance(r.get('observation'), dict)}
    setups = [r for r in rows if r.get('kind') == 'setup']
    assert stamps == {'item-v3'} and setups and all(r['observation']['heldSel'] is not None for r in setups)
    for name, task, items in (('v10-as-v2', W.FREE_PLAY_V10_NAME, 'item-v2'), ('v3-as-v3', W.FREE_PLAY_V3_NAME, 'item-v3')):
        with pytest.raises(ValueError, match='trained under'):
            launch(tmp_path / name, [Fake() for _ in range(4)], world=V2, actor='policy',
                   real_checkpoint=tag(name, task, items), ticks=300)
        assert not (tmp_path / name / 'run').exists()
    plain = launch(tmp_path / 'unstamped', [Unstamped() for _ in range(4)], world=V2, actor='policy', real_checkpoint=v10,
                   ticks=300)
    lives = [life for w in plain['results'] for life in w['lives']]
    assert plain['status'] != 'complete' and lives and all(life['status'] == 'incomplete' for life in lives)
    assert all("an observation of the item contract 'item-v2', the life's is 'item-v3'" in life['cause'] for life in lives)

def test_the_learner_and_the_training_command_import_no_part_of_the_watch():
    """In a fresh interpreter sim.train, sim.learn and sim.workers leave
    sim.watch unimported, and none of their sources names it."""
    import subprocess
    import sys
    for name in ('learn', 'train', 'workers'):
        assert 'watch' not in (ROOT / 'sim' / f'{name}.py').read_text(), name
    code = 'import sys; sys.path[:0] = [".", "rl"]; import sim.train, sim.learn, sim.workers; print("sim.watch" in sys.modules)'
    done = subprocess.run([sys.executable, '-I', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.split() == ['False'], done.stderr[-2000:]


# ---- free-play-v11 in the watch (item-v4, act-v4, mask-v6, obs-v7) -----------------------------------------------------
NINE_KINDS = [['bread', 1], ['shrimp', 1], ['bronze_dagger', 1], ['bronze_sword', 1], ['raw_chicken', 1], ['tinderbox', 1]]


def tagged(tiny, path, task, items, kit, **saved):
    """The tiny checkpoint saved as one of free-play version `task` with every stat paid and `kit`, its policy recording
    item contract `items`; under item-v4 its policy is first widened to obs-v7 and act-v4's 128 rows (widen_state), as
    --weights does."""
    from gymnasium import spaces
    from sb3_contrib import RecurrentPPO
    from baseline.adapter import observation_space
    from baseline.policy import run_contract, widen_state
    from sim.spec import dtype
    from sim.workers import observation_table
    model = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
    model.__dict__.update(free_play_task=task, paid_skills=[s.lower() for s in SKILL_ORDER], kit_extra=kit, **saved)
    model.policy_kwargs = dict(model.policy_kwargs, contract=run_contract(items))
    if items == ITEM_V4:
        import numpy as np
        observation, shapes = observation_table(ITEM_V4)
        space = observation_space({k: np.zeros(v, dtype(k)) for k, v in shapes.items()})
        model.policy_kwargs['observation'] = observation
        wide = type(model.policy)(space, spaces.Discrete(N_ACTIONS_V4), model.lr_schedule, use_sde=model.use_sde,
                                  **model.policy_kwargs)
        wide.load_state_dict(widen_state(model.policy.state_dict(), wide), strict=True)
        model.policy, model.observation_space, model.action_space = wide, space, spaces.Discrete(N_ACTIONS_V4)
    model.save(str(path))
    return path


def world_rows(run):
    import gzip
    return [json.loads(line) for k in range(4) for line in gzip.open(run / f'world-{k}.jsonl.gz', 'rt')]


class Stamped(Fake):
    """A world whose engine stamps `stamp` (item-v2 or item-v3) whatever item contract the reset names."""

    def __init__(self, stamp):
        super().__init__()
        self.stamp = stamp

    def reset(self, seed, cfg):
        super().reset(seed, cfg)
        self.contract = self.stamp
        return self.obs()


def test_a_free_play_v11_checkpoint_plays_item_v4_lives_under_the_lifes_own_contract_and_no_other_stamp(tiny, tmp_path):
    """A free-play-v11 checkpoint (obs-v7, act-v4, mask-v6, item-v4) plays as `policy` with its kit: every life's reset
    row holds the life's own cfg, contract item-v4, the engine's reset gets it, each observation stamps item-v4 and its
    mask has 128 rows, a row of act-v4's second groups (the drop of entry 8) is offered where the pack holds 9 kinds,
    and the setup's Wield finds the record's weapon among the page's kinds. A world whose observations stamp item-v3 or item-v2 ends each life incomplete, by the
    stamp, and stops. The record names v11; a v11 checkpoint that records item-v3 is refused before any world plays."""
    v11 = tagged(tiny, tmp_path / 'v11.zip', W.FREE_PLAY_V11_NAME, ITEM_V4, NINE_KINDS)
    fakes = [Fake(), Fake(), Fake(), Fake()]
    result = launch(tmp_path / 'v11', fakes, world=V2, actor='policy', real_checkpoint=v11, ticks=300)
    assert result['status'] == 'complete' and result['policy']['run']['task'] == W.FREE_PLAY_V11_NAME
    assert result['policy']['policy']['observation'] == 'obs-v7'
    cfgs = [cfg for f in fakes for _, cfg in f.cfgs]
    assert cfgs and all(c['contract'] == ITEM_V4 for c in cfgs)
    rows = world_rows(tmp_path / 'v11' / 'run')
    resets = [r for r in rows if r['kind'] == 'reset']
    assert resets and all(r['cfg']['contract'] == ITEM_V4 for r in resets)
    observations = [r['observation'] for r in rows if isinstance(r.get('observation'), dict)]
    assert {o['contractVersion'] for o in observations} == {ITEM_V4} and {len(o['mask']) for o in observations} == {128}
    assert max(len([e for e in o['invSel'] if e['slot'] >= 0]) for o in observations) == 9
    setups = [r for r in rows if r['kind'] == 'setup']
    assert setups and all(r['observation']['equipment'] for r in setups)
    assert any(o['mask'][page_row(A_DROP0, 8)] for o in observations), 'no row of the second groups was offered'
    for stamp in ('item-v3', 'item-v2'):
        plain = launch(tmp_path / stamp, [Stamped(stamp) for _ in range(4)], world=V2, actor='policy', real_checkpoint=v11,
                       ticks=300)
        lives = [life for w in plain['results'] for life in w['lives']]
        assert plain['status'] != 'complete' and lives and all(life['status'] == 'incomplete' for life in lives)
        assert all(f"an observation of the item contract '{stamp}', the life's is 'item-v4'" in life['cause'] for life in lives)
    with pytest.raises(ValueError, match='trained under|inputs'):
        launch(tmp_path / 'as-v3', [Fake() for _ in range(4)], world=V2, actor='policy',
               real_checkpoint=tagged(tiny, tmp_path / 'as-v3.zip', W.FREE_PLAY_V11_NAME, 'item-v3', NINE_KINDS), ticks=300)
    assert not (tmp_path / 'as-v3' / 'run').exists()


def test_the_monitors_read_an_item_v4_lifes_eats_drops_and_deposits_on_both_row_groups(tiny, tmp_path, monkeypatch):
    """The life reads of an item-v4 life: a scripted actor (in the policy's place) drops page entry 8 where the pack
    holds 9 kinds (row 104, act-v4's second drop group; a life whose record starts the weapon worn holds 8), drops the
    bread (a first-group drop row) and eats the shrimps (a pack option row); the life's watch_tool_drops, watch_food_drops and eats
    are what it did, counted by the pack's fall, and the summary sums them."""
    from sim.agent import TOOL_PROTECT_RE
    from sim.hour import FreePolicyActor
    done = {}

    def reset(self, seed):
        self.life = done.setdefault(seed, dict(rows=[], tools=0, food=0, eats=0))

    def act(self, observation, task=None, trace=False):
        raw, life = task.raw, self.life
        entries = [e for e in raw['invSel'] if e['slot'] >= 0]
        held = [k for k, h in enumerate(raw['heldSel']) if h['option'] == 'Eat' and h['name'] == 'Shrimps']
        if len(entries) == 9 and not life['rows']:
            row, life['tools'] = page_row(A_DROP0, 8), int(bool(TOOL_PROTECT_RE.search(entries[8]['name'])))
            life['food'] += 'Eat' in (entries[8].get('options') or ())
        elif any(e['name'] == 'Bread' for e in entries):
            k = next(k for k, e in enumerate(entries) if e['name'] == 'Bread')
            row, life['food'] = page_row(A_DROP0, k), life['food'] + ('Eat' in entries[k]['options'])
        elif held:
            row, life['eats'] = page_row(A_HELD0, held[0]), life['eats'] + 1
        else:
            row = A_WAIT
        assert observation['mask'][row] == 1
        life['rows'].append(row)
        return row
    monkeypatch.setattr(FreePolicyActor, 'reset', reset)
    monkeypatch.setattr(FreePolicyActor, 'act', act)
    v11 = tagged(tiny, tmp_path / 'v11.zip', W.FREE_PLAY_V11_NAME, ITEM_V4, NINE_KINDS)
    result = launch(tmp_path, [Fake() for _ in range(4)], world=V2, actor='policy', real_checkpoint=v11, ticks=300)
    assert result['status'] == 'complete'
    read = W.monitors(tmp_path / 'run')
    lives = read['lives']
    assert lives and any(done[r['seed']]['rows'][0] == 104 for r in lives)
    assert [(r['watch_tool_drops'], r['watch_food_drops'], r['eats']) for r in lives] == \
        [(done[r['seed']]['tools'], done[r['seed']]['food'], done[r['seed']]['eats']) for r in lives], \
        [(r['seed'], r['status'], done[r['seed']]['rows'][:8]) for r in lives]
    assert sum(r['watch_tool_drops'] for r in lives) > 0 and sum(r['eats'] for r in lives) > 0
    assert (read['summary']['watch_tool_drops'], read['summary']['watch_food_drops']) == \
        (sum(r['watch_tool_drops'] for r in lives), sum(r['watch_food_drops'] for r in lives))


def test_an_item_v4_lifes_start_writes_the_rows_of_an_item_v3_life_of_the_same_record(tiny, tmp_path):
    """WatchEpisode.prepare_start's item-v4 steps are the ORACLE method's: on the same records and seeds, an item-v4
    life's rows from its reset to its arm are of the same kinds in the same order as an item-v3 life's."""
    def start_rows(name, task, items):
        checkpoint = tagged(tiny, tmp_path / f'{name}.zip', task, items, NINE_KINDS)
        launch(tmp_path / name, [Fake() for _ in range(4)], world=V2, actor='policy', real_checkpoint=checkpoint, ticks=300)
        out = []
        for row in world_rows(tmp_path / name / 'run'):
            if row['kind'] == 'watch_life':
                out.append([])
            elif out and (not out[-1] or out[-1][-1] != 'arm'):
                out[-1].append('arm' if row['kind'] == 'resource_control' and row['operation'] == 'arm' else row['kind'])
        return out
    v10, v11 = start_rows('v10', W.FREE_PLAY_V10_NAME, 'item-v3'), start_rows('v11', W.FREE_PLAY_V11_NAME, ITEM_V4)
    assert v10 == v11 and len(v10) >= 4 and all(rows[0] == 'reset' and rows[-1] == 'arm' for rows in v10)
    assert any('setup' in rows for rows in v10)


# The digest of baseline/resource_task.py ResourceEpisode.prepare_start's source (an ORACLE file): sim/watch.py
# WatchEpisode.prepare_start repeats its steps for an item-v4 life, so a change of it is a change to review there.
PREPARE_START_SOURCE = '773f49d40c2a9dd587193a3779fd6a522ad856e2adf32972ac824ee1305945ce'


def test_the_oracle_start_that_the_item_v4_override_repeats_is_unchanged():
    import hashlib
    import inspect
    from baseline.resource_task import ResourceEpisode
    found = hashlib.sha256(inspect.getsource(ResourceEpisode.prepare_start).encode()).hexdigest()
    assert found == PREPARE_START_SOURCE, ('baseline/resource_task.py ResourceEpisode.prepare_start changed: review '
                                           'sim/watch.py WatchEpisode.prepare_start, which repeats its steps for item-v4 '
                                           f'lives, then record its digest {found}')


def contract_rows(tiny, tmp_path, task, items):
    """The sha256 of every row a watch of four fake worlds writes when a checkpoint of `task` under `items` plays it with
    the tinderbox kit (village-world-v2, 300 ticks a world, sampled), each row as JSON with sorted keys, the wall-clock
    fields left out: a life end's seconds and wall, and its controls' total and max."""
    import hashlib
    checkpoint = tagged(tiny, tmp_path / f'{task}.zip', task, items, [['tinderbox', 1]])
    launch(tmp_path / task, [Fake() for _ in range(4)], world=V2, actor='policy', real_checkpoint=checkpoint, ticks=300)
    digest = hashlib.sha256()
    for row in world_rows(tmp_path / task / 'run'):
        if row['kind'] == 'watch_life_end':
            row = dict(row, seconds=None, wall=None, controls={k: dict(v, total=None, max=None)
                                                               for k, v in (row.get('controls') or {}).items()})
        digest.update(json.dumps(row, sort_keys=True).encode())
    return digest.hexdigest()


# contract_rows of a free-play-v3 checkpoint (item-v2) and a free-play-v10 one (item-v3), recorded earlier (the
# watch before free-play-v11) with this file's Fake and the tiny run.
MAIN_ROWS = {'item-v2': '32bd47ab58650c1952ad4d431b17f5e267010f337516ab0adb94feb6e7b2d761',
             'item-v3': '3683cd4a77bed5291660591498ae2c22cd79938ccf70f26f197b3400a7e73f26'}


def test_item_v2_and_item_v3_watch_lives_write_the_rows_they_wrote_on_main(tiny, tmp_path):
    """Every item-v2 and item-v3 life calls the ORACLE start as it is: the rows of their watch are those main wrote, byte
    for byte but for wall-clock time (contract_rows, MAIN_ROWS)."""
    assert {'item-v2': contract_rows(tiny, tmp_path, W.FREE_PLAY_V3_NAME, 'item-v2'),
            'item-v3': contract_rows(tiny, tmp_path, W.FREE_PLAY_V10_NAME, 'item-v3')} == MAIN_ROWS



def test_a_free_play_v12_to_v14_checkpoint_plays_each_life_from_its_drawn_hitpoints_and_a_v10_one_from_full(tiny, tmp_path):
    """The watch plays a free-play-v12, v14 or v15 checkpoint with the hitpoints rule it trained under: each life's reset
    names its record's drawn hitpoints (startHp), the life starts at them, and run.json says so (hp_applied); a v10
    checkpoint's resets name none and its lives start at full hitpoints (hp_applied false), as before. A v14 one plays
    the fixed form's records as drawn, every life, as a v12 one does (no start rule of agents in a watch), and with
    --start-levels every life with the profile, as profiled() gives each record; a v15 one as a v14 one, its rare-XP pay
    recorded and not paid."""
    for name, task, saved in (('v12', W.FREE_PLAY_V12_NAME, {}),
                              ('v14', W.FREE_PLAY_V14_NAME, {}), ('v15', W.FREE_PLAY_V15_NAME, dict(inner_pay='rare-xp-v2')),
                              ('v10', W.FREE_PLAY_V10_NAME, {})):
        checkpoint = tagged(tiny, tmp_path / f'{name}.zip', task, 'item-v3', [['tinderbox', 1]], **saved)
        played = [Fake() for _ in range(4)]
        result = launch(tmp_path / name, played, world=V2, actor='policy', real_checkpoint=checkpoint, ticks=300)
        assert result['status'] == 'complete' and result['hp_applied'] is (name != 'v10')
        if name == 'v15':  # its rare-XP pay recorded, unpaid
            assert result['policy']['run']['inner_pay'] == 'rare-xp-v2' and 'inner_pay_in_watch' in result['policy']['run']
        rows, starts = world_rows(tmp_path / name / 'run'), []
        for row in rows:
            if row['kind'] == 'watch_life':
                starts.append([row['record']['hp_drawn'], None, None])
            elif row['kind'] == 'reset':
                starts[-1][1] = row['cfg'].get('startHp')
            elif row['kind'] == 'episode_start':
                starts[-1][2] = row['observation']['player']['hp']
        if name == 'v10':
            assert starts and all(cfg is None for _, cfg, _ in starts)
        else:
            assert starts and all(drawn == cfg == hp for drawn, cfg, hp in starts)
            assert any(hp < row['observation']['player']['maxHp'] for row in rows if row['kind'] == 'episode_start'
                       for hp in [row['observation']['player']['hp']])
        if name in ('v12', 'v14', 'v15'):  # as drawn: each world's first reset is its record's own
            assert [f.cfgs[0][1]['setStats'] for f in played] == \
                [W.Records(5, k, 'danger').next()['set_stats'] for k in range(4)]
    v14 = tmp_path / 'v14.zip'
    old = [Fake() for _ in range(4)]
    result = launch(tmp_path / 'v14-old', old, world=V2, actor='policy', real_checkpoint=v14, ticks=300,
                    start_levels=PROFILE)
    assert result['status'] == 'complete' and result['start_levels'] == W.parse_levels(PROFILE)
    for k, fake in enumerate(old):
        stream = W.Records(5, k, 'danger')
        records = [W.profiled(stream.next(), W.parse_levels(PROFILE)) for _ in fake.cfgs]
        assert [(cfg['setStats'], cfg['startHp']) for _, cfg in fake.cfgs] == \
            [(r['set_stats'], r['hp_drawn']) for r in records]



def test_a_control_actor_beside_a_v12_or_later_policy_starts_from_the_drawn_hitpoints_when_told_and_a_policy_never_changes_its_rule(
        tiny, tmp_path):
    """--start-hp drawn: a control actor (WAIT here) starts each life at its record's drawn hitpoints, as a v12 or later
    policy beside it does, and run.json says so (hp_applied); without it a control starts at full, as before. A policy
    plays under its own version's rule: --start-hp drawn beside a v10 checkpoint is refused before any world plays."""
    plain = launch(tmp_path / 'plain', [Fake() for _ in range(4)], world=V2, ticks=300)
    told = [Fake() for _ in range(4)]
    drawn = launch(tmp_path / 'drawn', told, world=V2, ticks=300, start_hp='drawn')
    assert plain['hp_applied'] is False and drawn['hp_applied'] is True
    rows = world_rows(tmp_path / 'drawn' / 'run')
    drawn_hp = [r['record']['hp_drawn'] for r in rows if r['kind'] == 'watch_life']
    assert drawn_hp and [r['cfg'].get('startHp') for r in rows if r['kind'] == 'reset'] == drawn_hp
    assert not any('startHp' in r['cfg'] for r in world_rows(tmp_path / 'plain' / 'run') if r['kind'] == 'reset')
    v10 = tagged(tiny, tmp_path / 'v10.zip', W.FREE_PLAY_V10_NAME, 'item-v3', [['tinderbox', 1]])
    with pytest.raises(ValueError, match='a policy plays under its own rule'):
        launch(tmp_path / 'v10', [Fake() for _ in range(4)], world=V2, actor='policy', real_checkpoint=v10, ticks=300,
               start_hp='drawn')
    assert not (tmp_path / 'v10' / 'run').exists()


# ---- longer lives, stage 1: start levels of a profile and the mining script -----------------------------------------
FIXED_RECORDS = 'fb077884cfce79f31b52c99424a4c9cd71bb45d09bc8ab39f0e498631b9b770f'  # the first 64 records of the fixed
# watch's four streams (key 20261113, group danger), sha256 of their JSON: the same as before the profile existed
PROFILE = 'woodcutting=50,firemaking=52,attack=31,strength=37,defence=30,hitpoints=34'  # an old life's profile
MINE = 'rimmington_mine_west'


def fixed_records():
    """The first 64 records of each of the fixed watch's four record streams."""
    return [[stream.next() for _ in range(64)] for stream in (W.Records(20261113, k, 'danger') for k in range(4))]


def test_without_start_levels_the_records_and_the_run_are_as_before(tmp_path):
    """The default path is the same bit for bit: the fixed watch's record streams hash as before the profile existed,
    profiled(record, None) is the record itself, and a run without the option names none and resets each life at its
    record's own levels and drawn hitpoints."""
    import hashlib
    rows = [r for stream in fixed_records() for r in stream]
    assert hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() == FIXED_RECORDS
    assert all(W.profiled(r, None) is r for r in rows)
    fakes = [Fake() for _ in range(4)]
    assert launch(tmp_path, fakes, start_hp='drawn')['status'] == 'complete'
    assert 'start_levels' not in json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert 'start_levels' not in W.monitors(tmp_path / 'run')
    for k, fake in enumerate(fakes):
        record = W.Records(5, k, 'danger').next()
        assert (fake.cfgs[0][1]['setStats'], fake.cfgs[0][1]['startHp']) == (record['set_stats'], record['hp_drawn'])


def test_a_profile_sets_the_named_stats_keeps_every_other_draw_and_the_records_hitpoints_deficit(tmp_path):
    from baseline.world import read_jsonl
    profile = W.parse_levels(PROFILE)
    assert list(profile) == ['attack', 'defence', 'strength', 'hitpoints', 'woodcutting', 'firemaking']  # SKILL_ORDER's
    added = ('levels', 'hp_drawn', 'set_stats', 'profile', 'drawn_levels', 'drawn_hp')
    for drawn in (r for stream in fixed_records() for r in stream):
        played = W.profiled(drawn, profile)
        assert {k: v for k, v in played.items() if k not in added} == {k: v for k, v in drawn.items() if k not in added}
        assert played['levels'] == {**drawn['levels'], **profile} and played['profile'] == profile
        assert (played['levels']['fishing'], played['levels']['mining']) == (drawn['levels']['fishing'], drawn['levels']['mining'])
        assert (played['drawn_levels'], played['drawn_hp']) == (drawn['levels'], drawn['hp_drawn'])
        assert played['hp_drawn'] == 34 - (drawn['levels']['hitpoints'] - drawn['hp_drawn'])  # the record's own deficit
        assert played['set_stats'] == ','.join(f'{s}:{level}' for s, level in played['levels'].items())
    low = dict(drawn, levels=dict(drawn['levels'], hitpoints=12), hp_drawn=1)  # a deficit of 11 under a level of 10
    assert W.profiled(low, dict(hitpoints=10))['hp_drawn'] == 1
    assert W.profiled(drawn, dict(mining=20))['hp_drawn'] == drawn['hp_drawn']  # hitpoints not named: as drawn
    fakes = [Fake() for _ in range(4)]
    assert launch(tmp_path, fakes, start_hp='drawn', start_levels=PROFILE)['status'] == 'complete'
    run = tmp_path / 'run'
    assert json.loads((run / 'run.json').read_text())['start_levels'] == profile == W.monitors(run)['start_levels']
    for k, fake in enumerate(fakes):
        record = W.profiled(W.Records(5, k, 'danger').next(), profile)
        assert (fake.cfgs[0][1]['setStats'], fake.cfgs[0][1]['startHp']) == (record['set_stats'], record['hp_drawn'])
    begun = next(r for r in read_jsonl(run / 'world-0.jsonl.gz') if r['kind'] == 'watch_life')
    assert begun['record']['profile'] == profile and begun['record']['levels']['firemaking'] == 52
    read = W.monitors(run)['lives'][0]
    assert read['levels0']['woodcutting'] == 50 and read['hp0'] == read['hp_drawn'] == begun['record']['hp_drawn']
    for name in ('real-run-final', 'final'):  # a profile watch is not named as a fixed-form one
        with pytest.raises(ValueError, match='not as a fixed-form watch'):
            W.check_settings(output=tmp_path / name, world=V1, actor='wait', sites=list(SITES), rotation=0, key=5,
                             seed=9, ticks=1000, guard=60.0, tables=Tables(), start_levels=PROFILE)


def test_an_unknown_stat_or_a_bad_level_in_a_profile_is_refused_before_any_world_is_joined(tmp_path):
    for text in ('woodcuting=50', 'stat18=5', 'mining=20,mining=30', 'mining=0', 'mining=100', 'hitpoints=9',
                 'mining:20', 'mining=x', ''):
        with pytest.raises(ValueError):
            W.parse_levels(text)
    joined = []
    with pytest.raises(ValueError, match="unknown stat 'woodcuting'"):
        launch(tmp_path, [None] * 4, start_levels='woodcuting=50', join=lambda w, r, k: joined.append(k))
    assert joined == [] and not (tmp_path / 'run').exists()


def act_mine(raw, mask):
    """The mining script's action on a native observation `raw` under `mask`."""
    from types import SimpleNamespace
    from sim.mine_script import MineScript
    return MineScript().act(dict(mask=mask), SimpleNamespace(raw=raw))


def rock_row(tile, live=True):
    return dict(kind='loc', id=W.MINE_ROCKS[tile][1] if live else EMPTY, name='Rocks', option='Mine', x=tile[0], z=tile[1])


def test_the_mining_script_mines_the_nearest_live_rock_its_level_allows_and_drops_ore_at_a_full_pack():
    from sim.agent import A_CLOSE_IF, A_DIALOG_CONTINUE, A_INV_NEXT, A_INV_PREV
    iron, tin, tin2 = (2982, 3234), (2986, 3235), (2984, 3237)
    outside = dict(kind='loc', id=2090, name='Rocks', option='Mine', x=3000, z=3000)  # a copper rock not of the mine
    rows = [rock_row(iron), rock_row(tin, live=False), rock_row(tin2), outside]
    mask = [0] * 96
    mask[A_WAIT] = 1
    for k in range(len(rows)):
        mask[A_ACT0 + k] = 1
    skills = lambda level: [dict(name='Mining', level=level, baseLevel=level, xp=0)]  # noqa: E731
    base = dict(actSel=rows, skills=skills(10), inventory=[], invSel=[], invPage=0)
    assert act_mine(base, mask) == A_ACT0 + 2  # iron needs 15 and the near tin is mined out: the next tin
    assert act_mine(dict(base, skills=skills(15)), mask) == A_ACT0  # at 15 the nearest, iron
    assert act_mine(dict(base, actSel=[rows[1], outside]), mask) == A_WAIT  # an empty rock and one outside the table
    assert act_mine(base, [m if k != A_ACT0 + 2 else 0 for k, m in enumerate(mask)]) == A_WAIT  # its row not legal
    assert act_mine(dict(base, optionBusy=True), mask) == A_WAIT
    dialog = list(mask)
    dialog[A_DIALOG_CONTINUE] = dialog[A_CLOSE_IF] = 1
    assert act_mine(dict(base, dialogOpen=True), dialog) == A_DIALOG_CONTINUE
    dialog[A_DIALOG_CONTINUE] = 0
    assert act_mine(dict(base, dialogOpen=True), dialog) == A_CLOSE_IF
    tools = [dict(slot=k, id=ident, name=name) for k, (ident, name) in enumerate(NAMES[n] for n in ('bronze_axe', 'net', 'bronze_pickaxe'))]
    ores = [dict(slot=k, id=436, name='Copper ore') for k in range(3, N_INV)]
    page = lambda p, items: [next((dict(i, count=1) for i in items if i['slot'] == s), dict(slot=-1, id=-1, count=0))  # noqa: E731
                             for s in range(8 * p, 8 * p + 8)]
    full = dict(base, inventory=tools + ores, invSel=page(0, tools + ores))
    drops = list(mask)
    for k in range(8):
        drops[A_DROP0 + k] = 1
    assert act_mine(full, drops) == A_DROP0 + 3  # the first ore of page 0, never a tool
    axes = lambda slots: [dict(slot=k, id=1351, name='Bronze axe') for k in slots]  # noqa: E731
    copper = lambda slots: [dict(slot=k, id=436, name='Copper ore') for k in slots]  # noqa: E731
    turns = list(drops)
    turns[A_INV_NEXT] = turns[A_INV_PREV] = 1
    late = tools + axes(range(3, 16)) + copper(range(16, N_INV))  # ores on pages 2 and 3 only
    assert act_mine(dict(full, inventory=late, invSel=page(0, late)), turns) == A_INV_NEXT
    early = tools + axes(range(3, 16)) + copper(range(16, 24)) + axes(range(24, N_INV))  # ores on page 2 only
    assert act_mine(dict(full, inventory=early, invSel=page(3, early), invPage=3), turns) == A_INV_PREV


def test_the_mining_script_plays_one_site_in_every_world_and_its_reads_show_the_rock_reached_and_the_ores(tmp_path, monkeypatch):
    """At the one site of a single-site watch every world plays its own record stream. The script reaches a rock in its
    first step, mines live rocks its level allows, drops ore at a full pack, and its life reads count the ores for
    each 10,000 ticks. With a profile's mining level it takes iron. A script stuck where no rock is in reach reads as
    stuck (no rock reached, no ore), not as a rate. It plays MINE_SITES alone."""
    from baseline.world import read_jsonl
    monkeypatch.setitem(W.SITES, MINE, REAL_SITES[MINE])
    fakes = [Fake(rocks=True) for _ in range(4)]
    result = launch(tmp_path, fakes, actor='script-mine', sites=[MINE], ticks=600)
    run = tmp_path / 'run'
    saved = json.loads((run / 'run.json').read_text())
    assert result['status'] == 'complete' and saved['order'] == [MINE] * 4 and saved['script']['version'] == 'mine-script-v1'
    reads = W.monitors(run)
    for k, fake in enumerate(fakes):
        life = next(r for r in reads['lives'] if r['world'] == k)
        ores = sum(i['id'] in ORE_IDS for i in fake.items if i)
        assert life['reached_rock'] == 3 and life['made']['ores'] >= ores  # beside a rock after its first step
        assert life['ores_per_10k'] == round(life['made']['ores'] * 10000 / 600, 2)
        requested = [r['action'] for r in read_jsonl(run / f'world-{k}.jsonl.gz') if r['kind'] == 'requested']
        assert sum(1 for i in fake.items if i) >= N_INV - 1 and any(A_DROP0 <= a < A_DROP0 + 8 for a in requested)
        level = W.Records(5, k, 'danger').next()['levels']['mining']
        assert (440 in {i['id'] for i in fake.items if i}) == (level >= 15)  # iron only from Mining 15
    summary = reads['summary']
    assert summary['reached_rock'] == 4 and summary['ores_per_10k'] == round(
        sum(r['made']['ores'] for r in reads['lives']) * 10000 / summary['armed_ticks'], 3)
    assert sum(g['lives'] for g in summary['by_start_mining'].values()) == 4
    fakes = [Fake(rocks=True) for _ in range(4)]
    launch(tmp_path / 'deep', fakes, actor='script-mine', sites=[MINE], ticks=300, start_levels='mining=20')
    assert all(440 in {i['id'] for i in fake.items if i} for fake in fakes)  # a mining level given: iron in every world
    monkeypatch.setitem(W.SITES, MINE, (3200, 3200))  # no rock within 15 tiles
    launch(tmp_path / 'stuck', [Fake(rocks=True) for _ in range(4)], actor='script-mine', sites=[MINE], ticks=300)
    stuck = W.monitors(tmp_path / 'stuck' / 'run')
    assert all(r['reached_rock'] is None and r['made'].get('ores', 0) == 0 for r in stuck['lives'])
    assert (stuck['summary']['reached_rock'], stuck['summary']['ores_per_10k']) == (0, 0.0)
    with pytest.raises(ValueError, match='script-mine plays rimmington_mine_west alone'):
        launch(tmp_path / 'other', [None] * 4, actor='script-mine', sites=list(SITES))
    assert launch(tmp_path / 'wait', [Fake() for _ in range(4)], sites=[MINE], ticks=100)['order'] == [MINE] * 4


def test_the_mine_table_is_the_listed_rocks_less_clay_and_the_mines_iron_as_the_vendor_map_holds_them():
    listed = set(config.START_SETS['rimmington']['rocks'])
    ours = {tile for tile, (ore, _, _) in W.MINE_ROCKS.items() if ore in ('copper', 'tin')}
    assert ours <= listed and listed - ours == {(2986, 3239), (2987, 3240)}
    site = REAL_SITES[MINE]
    assert min(max(abs(x - site[0]), abs(z - site[1])) for x, z in W.MINE_ROCKS) == 3
    assert W.MINE_SITES == (MINE,) and all(level == (15 if ore == 'iron' else 1) for ore, _, level in W.MINE_ROCKS.values())
    content = ROOT / 'vendor/rs-sdk/server/content'
    if not (content / 'maps/m46_50.jm2').exists():
        pytest.skip('vendor map absent (vendor/rs-sdk is not checked out here)')
    names = dict(line.split('=', 1) for line in (content / 'pack/loc.pack').read_text().split('\n') if '=' in line)
    lines = (content / 'maps/m46_50.jm2').read_text().split('\n')
    boot = {}
    for line in lines[lines.index('==== LOC ====') + 1:]:
        if line.startswith('===='):
            break
        found = re.match(r'(\d+) (\d+) (\d+): (\d+)', line)
        if found and found.group(1) == '0':
            boot[(46 * 64 + int(found.group(2)), 50 * 64 + int(found.group(3)))] = int(found.group(4))
    rocks = {tile: ident for tile, ident in boot.items() if re.fullmatch(r'(copper|tin|iron)rock\d', names[str(ident)])}
    assert {tile: ident for tile, (_, ident, _) in W.MINE_ROCKS.items()} == rocks
    assert all(names[str(ident)].startswith(ore) for ore, ident, _ in W.MINE_ROCKS.values())
    assert {names[str(boot[t])] for t in listed - ours} <= {'clayrock1', 'clayrock2'}


def test_no_trainer_learner_or_distillation_module_imports_the_mining_script():
    import subprocess
    modules = ('train', 'learn', 'workers', 'real')
    for name in modules:
        assert 'mine_script' not in (ROOT / 'sim' / f'{name}.py').read_text(), name
    code = ('import sys; sys.path[:0] = [".", "rl"]; import sim.watch, ' + ', '.join(f'sim.{m}' for m in modules)
            + '; print("sim.mine_script" in sys.modules)')  # sim.real reads the watch; the script loads only when it plays
    done = subprocess.run([sys.executable, '-I', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.split() == ['False'], done.stderr[-2000:]


# ---- a public copy of a policy (scripts/export_policy.py) ------------------------------------------------------------
def export_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('export_policy',
                                                  Path(__file__).resolve().parents[2] / 'scripts' / 'export_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_public_copy_holds_no_name_vector_or_path_and_plays_as_the_policy_it_copies(tiny, tmp_path, monkeypatch):
    """scripts/export_policy.py on a real-engine checkpoint (the tiny run's weights tagged free-play-v12, with a leg record
    that names its container folder and an engine error that names another, and its run's launch.json with a run name,
    seeds, start weights and an output folder): the copy's name tables hold no vectors, no string of its metadata holds
    an absolute path, its episode buffers are empty, its fingerprint is the table in use's, its public launch keeps only
    the XP rate, --env, --task, --half-life and the batch settings starts and life, and its sha256 file names the three
    files. The watch's check takes the copy with its public launch. The watch's and the demo's loader (sim/hour.py
    load_free_policy) takes the copy by its digest and fills its tables from the table in use: tensor for tensor the
    policy it copies, so it plays the same actions. A table rebuilt one float16 step off in every value plays the copy by its fingerprint and is refused for the
    original, which has none beside it. A run refuses the copy for --weights and --init. A stem whose files exist is
    refused."""
    import hashlib
    import io
    import zipfile
    import numpy as np
    import torch
    import names
    from sb3_contrib import RecurrentPPO
    from baseline.policy import run_contract
    from sim import train as T
    from sim.hour import load_free_policy
    source = tagged(tiny, tmp_path / 'v12.zip', W.FREE_PLAY_V12_NAME, 'item-v3', [['tinderbox', 1]])
    model = RecurrentPPO.load(str(source), device='cpu')
    model.real_leg = dict(number=1, first='/tmp/run/real-s999-xx/free', env_seed=5,
                          world=dict(env_ts=dict(commit=None, error="git -C '/app' log failed"), task='free-play-v12'))
    model.save(str(source))
    run_launch = tmp_path / 'launch.json'
    run_launch.write_text(json.dumps(dict(name='real-s999-xx', launch='abc', source=dict(commit='0' * 40), xp_rate=1,
                                          spend_limit=30.0, command=[
        'python', '-m', 'sim.train', 'train', '--env', 'real', '--task', W.FREE_PLAY_V12_NAME, '--half-life', '6000',
        '--seed', '999', '--env-seed', '5', '--weights', '/data/bank/real-s998/free/latest.zip', '--batch-setting',
        f'starts="{V2}"', 'life=["uniform",6000,36000]', 'worlds=16', '--output', '/tmp/run/real-s999-xx/free'])))
    export = export_script().export
    done = export(source, tmp_path / 'public' / 'policy', run_launch)
    stem = tmp_path / 'public' / 'policy'
    copy, fingerprint, launch, sums = (stem.with_name(f'policy{s}') for s in ('.zip', '.names.json', '.launch.json',
                                                                               '.sha256'))
    assert done['files'] == [str(copy), str(fingerprint), str(launch), str(sums)] and done['tables_zeroed'] == 4
    assert json.loads(launch.read_text()) == dict(
        note="a public copy of the training run's launch.json: only what sim.watch and sim.real demo read "
             '(scripts/export_policy.py)', name='', xp_rate=1,
        command=['python', '-m', 'sim.train', 'train', '--env', 'real', '--task', W.FREE_PLAY_V12_NAME, '--half-life',
                 '6000', '--batch-setting', f'starts="{V2}"', 'life=["uniform",6000,36000]'])
    checked = W.check_settings(output=tmp_path / 'checked', world=V2, actor='policy', sites=list(SITES), rotation=0,
                               key=5, seed=9, ticks=1000, guard=60.0, tables=Tables(), real_checkpoint=copy, launch=launch)
    assert checked['facts']['run']['launch']['name'] == '' and checked['facts']['run']['start'] is None
    assert sorted(d['value'] for d in done['blanked']) == ['/tmp/run/real-s999-xx/free', "git -C '/app' log failed"]
    with zipfile.ZipFile(copy) as archive:
        data = archive.read('data').decode()
        saved = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    assert '/tmp/' not in data and "'/app'" not in data and 'real-s999' not in data
    tables = [k for k in saved if k.endswith('embedding.table')]
    assert len(tables) == 4 and not any(saved[k].any() for k in tables)
    public = RecurrentPPO.load(str(copy), device='cpu')
    assert public.real_leg['first'] == '' and public.real_leg['env_seed'] == 5 and not public.ep_info_buffer
    assert json.loads(fingerprint.read_text())['digest'] == names.name_table().digest
    assert sums.read_text() == ''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n'
                                       for f in (copy, fingerprint, launch))
    contract = run_contract('item-v3')
    original, facts = load_free_policy(source, contract=contract)
    played, played_facts = load_free_policy(copy, contract=contract)
    assert facts['names'] == dict(path='digest', table=names.name_table().digest, vectors="the checkpoint's own")
    assert played_facts['names']['vectors'] == 'the table in use (a public copy holds none)'
    assert played_facts['policy_sha256'] == facts['policy_sha256']
    mine = original.policy.state_dict()
    assert all(torch.equal(v, mine[k]) for k, v in played.policy.state_dict().items())
    models, run = W.load_actor('policy', None, None, None, V2, real_checkpoint=copy, launch=None)  # the demo's call
    assert run['run']['task'] == W.FREE_PLAY_V12_NAME
    table_path = os.environ['RL_NAME_TABLE']
    with np.load(table_path) as data:
        kind, name, vector = data['kind'], data['name'], data['vector']
    stepped = tmp_path / 'stepped.npz'
    np.savez(stepped, kind=kind, name=name, vector=np.nextafter(vector, np.where(vector < 0, -np.inf, np.inf)
                                                                 .astype(np.float16)))
    monkeypatch.setenv('RL_NAME_TABLE', str(stepped))
    names.name_table.cache_clear()
    _, rebuilt = load_free_policy(copy, contract=contract)
    assert rebuilt['names']['path'] == 'fingerprint' and rebuilt['names']['trained'] == facts['names']['table']
    with pytest.raises(ValueError, match='another name table than RL_NAME_TABLE$'):
        load_free_policy(source, contract=contract)
    monkeypatch.setenv('RL_NAME_TABLE', table_path)
    names.name_table.cache_clear()
    plain = export(tiny / 'latest.zip', tmp_path / 'public' / 'v1')  # a free-play-v1 copy, for the trainer's loaders
    run_model = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
    for load in (T.load_weights, T.load_init):
        with pytest.raises(ValueError, match='holds no name vectors'):
            load(run_model, Path(plain['files'][0]), 'cpu')
    with pytest.raises(ValueError, match='a public copy is written fresh'):
        export(source, stem)
