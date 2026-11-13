"""Simulated worlds behind the learner's environment interface.

CpuBatch is sim/spec.py's batched interface over B CPU reference worlds: the four-goal task on a
layout pool (sim/world.py), the training environment of sim/workers.py. `trace` dumps one
of its runs under keyed random legal actions, world by world and tick by tick.

A step or a setup the simulator cannot model never becomes a transition: CpuBatch flags the
world in info['unsupported'], keeps the evidence and refuses to step that world again until reset,
so the collector must stop before any update uses the rollout. (The goal-v1 Draynor CPU workers,
SimSchedule and SimWorkers, were removed with the SB3 CPU-worker training path; sim/learn.py is the
training path.)

    .venv-sim/bin/python -m sim.vec trace --pool logs/sim/world/dev.npz --worlds 4 --steps 100 --output logs/sim/trace.jsonl.gz
"""
import argparse
from collections import OrderedDict
import gzip
import json
from pathlib import Path

import numpy as np

from baseline import config
from baseline.resource_task import GOALS
from sim.data import Tables
from sim.engine import Rng, Unsupported, World
from sim.spec import (HORIZON_TICKS, INFO_FIELDS, OBSERVATION_SHAPES, TERMINAL_DEATH, TERMINAL_HORIZON, check_batch,
                      dtype, goal_set, slot_goal)
from sim.audit import kills
from sim.combat import COMBAT, GROUPS, check_bindings, combat_npcs, draw_combat, fights, kit, outside, task_spec
from sim.task import FAMILIES, WORLD_START_TICK, GoalEpisode, draw_start, object_families, produced
from sim.world import ORIGIN, LayoutTables, load_pool, world_of


# info['facts'] of the step that ends an episode (zero elsewhere): its combat group (1 + GROUPS index; 0 for gathering),
# success (return 1) and, for a combat episode, sim.combat.fights' food and healing: items eaten, hitpoints gained by
# eating, natural healing and Hitpoints level-ups, fights an enemy began, items eaten without a gain; for a search episode
# (sim/search.py) whether an eligible tree was observed and the tick from arm it first was (-1 if never); for every episode
# the distinct 8x8 cells the player stood in
COMBAT_FACTS = ('eaten', 'eat_gain', 'regen_gain', 'levelup_gain', 'aggressed', 'eats_without_gain')
SEARCH_FACTS = ('search', 'discovered', 'discovery_ticks')
# a supply episode (sim/search.py supply-wood-v1): its variant (1 ample, 2 scarce; 0 none), the tick from arm its grove first
# had no standing tree and the ticks from then to its first log elsewhere (-1 when never)
SUPPLY_FACTS = ('supply', 'exhausted_tick', 'elsewhere_gap')
EPISODE_FACTS = ('group', 'success', *COMBAT_FACTS, *SEARCH_FACTS, 'cells', *SUPPLY_FACTS)
SUPPLY_TABLES = 32  # scarce variant layouts a CpuBatch keeps built (the least recently drawn is dropped)


# ---- the batched interface (sim/spec.py) over CPU reference worlds ------------------------------------
class CpuBatch:
    """B worlds of the four-goal task (sim/task.py) on a layout pool; world slot b asks for GOALS[b % 4], or under a
    single-goal set (sim/spec.py goal_set) for that goal. Each episode is a fresh World on the layout task.draw_start
    picks for (key, b, episode), seeded by that draw too, so a batched port reproduces the episodes. The state is held
    here: `reset(key)` and `step(state, action)` return a small record of it for the interface's shape."""

    def __init__(self, tables, layouts, n, horizon=HORIZON_TICKS, cfg=None, derived=None, first=0, task='goal-activity-v2',
                 search=None, goals=GOALS):
        """task: goal-activity-v2, or v3 (sim/combat.py: melee slots play sustained combat, with its own horizon;
        `horizon` is then the gathering goals' and must be v2's), or v4 (v3 with the wood slots on sim/search.py's search
        starts, drawn from `search`: the search pool's layouts and their derived tables or None), or v5 (v3 with the wood
        slots on sim/search.py's supply starts, `search` then the supply pool; each slot's variant layout is built when
        drawn and the last SUPPLY_TABLES kept), or v11 (v5 on its bound pool and goal set: sim/combat.py BOUND, refused
        otherwise). goals: the declared goal set (sim/spec.py goal_set)."""
        self.spec = task_spec(task)
        if self.spec['combat'] and horizon != HORIZON_TICKS:
            raise ValueError(f'{task} keeps the gathering goals\' {HORIZON_TICKS}-tick horizon')
        check_bindings(self.spec, ((layout.seed, world_of(layout)['version']) for layout in layouts), goal_set(goals))
        self.tables, self.layouts, self.n, self.horizon = tables, layouts, n, horizon
        self.cfg = dict(cfg or config.TASK_CFG)
        self.layout_tables = [LayoutTables(tables, layout, None if derived is None else derived[k])
                              for k, layout in enumerate(layouts)]
        from sim.search import SEARCH
        for lt in self.layout_tables:
            if lt.derived is None:  # windows computed as queried keep the search layouts' bound (a memo: no output changes)
                lt.CACHE = SEARCH['cache']
        # first: the global slot of world 0, so that a slice of a larger batch (sim/workers.py) draws its goals and
        # starts as the whole batch does
        self.first, goals = first, goal_set(goals)
        self.goals = [slot_goal(goals, first + b) for b in range(n)]
        self.family = object_families(tables)
        self.regions = {}  # layout index -> sim.combat.outside, for mugger placement
        if self.spec['search'] != (search is not None):
            raise ValueError(f'{task} {"needs" if self.spec["search"] else "takes no"} search pool')
        if search is not None:
            from sim.search import search_tiles
            self.search_layouts, derived = search
            from sim.search import SEARCH
            self.search_tables = [LayoutTables(tables, layout, None if derived is None else derived[k])
                                  for k, layout in enumerate(self.search_layouts)]
            for lt in self.search_tables:  # the 64-tile layouts compute their path windows as queried (no derived tables)
                lt.CACHE = SEARCH['cache']
            if self.spec['supply']:
                from sim.search import groves, supply_sites
                self.supply_sites = [supply_sites(layout, tables) for layout in self.search_layouts]
                self.supply_groves = [groves(layout, tables) for layout in self.search_layouts]
                self.variant_tables = OrderedDict()
            else:
                self.search_tiles = [search_tiles(layout, tables) for layout in self.search_layouts]
        self.key, self.evidence = None, []

    def _supply_tables(self, start):
        """LayoutTables of a supply start's variant layout: the pool's own for ample, a scarce one built once and kept
        among the last SUPPLY_TABLES."""
        from sim.search import SUPPLY, scarce_rows, variant_layout
        layout = self.search_layouts[start['layout']]
        if start['variant'] == 'ample':
            return self.search_tables[start['layout']]
        tile = (start['tile'][0] - ORIGIN[0], start['tile'][1] - ORIGIN[1])
        _, keep = scarce_rows(layout, self.tables, self.supply_groves[start['layout']][start['grove']], tile)
        key = (start['layout'], start['grove'], keep)
        if key not in self.variant_tables:
            lt = LayoutTables(self.tables, variant_layout(layout, self.tables, start))
            lt.CACHE = SUPPLY['cache']
            self.variant_tables[key] = lt
            while len(self.variant_tables) > SUPPLY_TABLES:
                self.variant_tables.popitem(last=False)
        self.variant_tables.move_to_end(key)
        return self.variant_tables[key]

    def _episode(self, b):
        slot, episode = self.first + b, self.episodes[b]
        if self.spec['supply'] and self.goals[b] == 'wood':
            from sim.search import SupplyTracker, draw_supply, supply_episode
            start = self.starts[b] = draw_supply(self.key, slot, episode, self.search_layouts, self.supply_sites)
            task = self.tasks[b] = supply_episode(start, self._supply_tables(start), self.tables, self.cfg)
            layout = self.search_layouts[start['layout']]
            grove = self.supply_groves[start['layout']][start['grove']]
            self.trackers[b] = SupplyTracker(task, [(int(layout.locs[k, 1]), int(layout.locs[k, 2])) for k in grove])
            return task.projected
        if self.spec['search'] and self.goals[b] == 'wood':
            from sim.search import SEARCH, Tracker, draw_search
            start = self.starts[b] = draw_search(self.key, slot, episode, self.search_layouts, self.search_tiles)
            world = World(self.search_tables[start['layout']], seed=start['world_seed'], start_tick=WORLD_START_TICK)
            task = self.tasks[b] = GoalEpisode(world, self.cfg, 'wood', horizon=SEARCH['horizon'], scale=1.0)
            obs = task.start(start['tile'], start['weapon'], start['levels'])
            self.trackers[b] = Tracker(task)
            return obs
        self.trackers[b] = None
        start = draw_start(self.key, slot, episode, self.layouts)
        lt, combat = self.layout_tables[start['layout']], self.spec['combat'] and self.goals[b] == 'melee'
        if combat:
            layout = self.layouts[start['layout']]
            if start['layout'] not in self.regions:
                self.regions[start['layout']] = outside(layout)
            start['combat'] = c = draw_combat(self.key, slot, episode, layout, start, self.tables,
                                              region=self.regions[start['layout']])
            world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK,
                          npcs=combat_npcs(layout, self.tables, c['muggers']), exclusions=COMBAT['exclusions'])
            task = GoalEpisode(world, self.cfg, 'melee', horizon=COMBAT['horizon'], scale=1.0, combat=True)
        else:
            world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK)
            task = GoalEpisode(world, self.cfg, self.goals[b], horizon=self.horizon, scale=1.0)
        self.tasks[b], self.starts[b] = task, start
        if combat:
            return task.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], c['food']), hp=c['hp'],
                              wield=c['worn'])
        return task.start(start['tile'], start['weapon'], start['levels'])

    def _facts(self, b, task, ended):
        success, tracker, cells = int(ended['episode']['r'] == 1.0), self.trackers[b], len(task.agent.visited)
        searched = [0, 0, -1] if tracker is None or self.starts[b].get('supply') else \
            [1, int(tracker.discovered is not None), -1 if tracker.discovered is None else tracker.discovered]
        supplied = [0, -1, -1]
        if self.starts[b].get('supply'):  # success: the quota of certified logs and survival (sim/search.py SUPPLY)
            from sim.search import SUPPLY
            success = int(bool(ended['report']['survived']) and ended['outcomes'] >= SUPPLY['quota'])
            gap = tracker.gap()
            supplied = [1 + SUPPLY['variants'].index(self.starts[b]['variant']),
                        -1 if tracker.exhausted is None else tracker.exhausted, -1 if gap is None else gap]
        if 'combat' not in self.starts[b]:
            return [0, success] + [0] * len(COMBAT_FACTS) + searched + [cells] + supplied
        s = fights(task)['summary']
        return [1 + GROUPS.index(self.starts[b]['combat']['group']), success, s['eats'] + s['eats_without_gain'],
                s['eat_gain'], s['regen_gain'], s['levelup_gain'], s['aggressed'], s['eats_without_gain']] + searched + \
            [cells] + supplied

    def _state(self):
        return dict(key=self.key, episode=np.asarray(self.episodes, np.int32))

    def reset(self, key):
        self.key, self.episodes = int(key), [0] * self.n
        self.tasks, self.starts, self.failed, self.evidence = [None] * self.n, [None] * self.n, [False] * self.n, []
        self.trackers = [None] * self.n
        obs = [self._episode(b) for b in range(self.n)]
        return self._state(), self._stack(obs)

    def _stack(self, obs):
        out = {name: np.stack([np.asarray(o[name], dtype(name)) for o in obs]) for name in OBSERVATION_SHAPES}
        check_batch(out, self.n)
        return out

    def step(self, state, action):
        if state['key'] != self.key:
            raise ValueError('a state of another reset')
        info = {name: np.zeros(self.n, kind) for name, kind in INFO_FIELDS.items()}
        # the episode's game effects so far (sim.train rollout_evidence)
        info.update(made=np.zeros((self.n, len(FAMILIES)), np.int32), kills=np.zeros(self.n, np.int32),
                    facts=np.zeros((self.n, len(EPISODE_FACTS)), np.int32))
        reward, done, obs = np.zeros(self.n, np.float32), np.zeros(self.n, bool), []
        for b, task in enumerate(self.tasks):
            if self.failed[b]:
                raise RuntimeError(f'world {b} reached unsupported content; reset before stepping it again')
            info['goal'][b] = GOALS.index(task.goal)
            before = task.world.tick
            try:
                o, r, d, ended = task.step(int(action[b]))
            except Unsupported as error:
                self.failed[b] = info['unsupported'][b] = True
                pool = self.search_layouts if self.starts[b].get('search') or self.starts[b].get('supply') else self.layouts
                self.evidence.append(dict(reason=str(error), world=self.first + b, key=self.key, episode=self.episodes[b],
                                          start=self.starts[b], seed=pool[self.starts[b]['layout']].seed,
                                          decision=task.decisions, tick=task.world.tick, action=int(action[b]),
                                          observation=task.raw))
                obs.append(task.projected)
                continue
            info['decision_ticks'][b] = task.world.tick - before
            if self.trackers[b] is not None:
                self.trackers[b].observe()
            events = task.world.events[task.event_start:]
            info['made'][b], info['kills'][b] = produced(events, self.family), kills(events, task.agent.name, task.enemies)
            reward[b], done[b] = r, d
            if d:
                info['terminal'][b] = TERMINAL_DEATH if ended['report']['reason'] == 'death' else TERMINAL_HORIZON
                info['episode_return'][b] = ended['episode']['r']
                info['episode_decisions'][b] = ended['episode']['l']
                info['episode_ticks'][b] = ended['episode']['native_ticks']
                info['outcomes'][b] = ended['outcomes']
                info['facts'][b] = self._facts(b, task, ended)
                self.episodes[b] += 1
                o = self._episode(b)
            obs.append(o)
        return self._state(), self._stack(obs), reward, done, info


# ---- trace dump: CpuBatch under keyed random legal actions, tick by tick ------------------------------
R_ACTION = 9


def world_state(world):
    """What a batched port must agree on after a tick: the player, npcs, changed and added locs (a gate's
    open leaves), ground items."""
    p = world.player
    player = None if p is None else dict(
        x=p.x, z=p.z, level=p.level, path=[list(w) for w in p.path], delayed=p.delayed, levels=list(p.levels),
        xp=list(p.stats), inv=p.inv.listed(), worn=p.worn.listed(), target=None if p.target is None else
        [type(p.target).__name__, getattr(p.target, 'nid', getattr(p.target, 'index', -1)), p.target_op],
        modal=sorted(p.modal_state), varps={k: p.varps[k] for k in ('action_delay', 'eat_delay', 'com_mode', 'damagestyle',
                                                                    'damagetype', 'com_maxhit', 'lastcombat')})
    npcs = [dict(nid=n.nid, type=n.type, x=n.x, z=n.z, active=n.active, hp=n.levels[3], mode=n.mode, life=n.life,
                 delayed=n.delayed) for n in world.npcs]
    added = sorted((loc for at in world.loc_at.values() for loc in at if loc.index >= world.n_stored_locs), key=lambda loc: loc.index)
    locs = [dict(index=loc.index, type=loc.type, x=loc.x, z=loc.z, angle=loc.angle, active=loc.active, timer=loc.lifecycle_tick)
            for loc in world.locs + added if loc.is_changed() or not loc.active or loc.index >= world.n_stored_locs]
    objs = [dict(type=o.type, count=o.count, x=o.x, z=o.z, ticks=o.ticks) for o in world.objs if o.is_valid()]
    return dict(tick=world.tick, player=player, npcs=npcs, locs=locs, objs=objs, events=len(world.events))


def trace(env, key, steps, write):
    """Steps env with legal actions drawn by Rng(key) on (R_ACTION, step, world); write() receives each
    episode's start (its draw and the world at arm), a row per world tick (the state after it), and
    one per decision with the actions and outcomes."""
    state, obs = env.reset(key)
    started = lambda b: write(dict(kind='start', world=b, episode=env.episodes[b], start=env.starts[b],
                                   state=world_state(env.tasks[b].world)))
    for b in range(env.n):
        started(b)
    rng = Rng(key)
    for step in range(steps):
        hooks = []
        for b, task in enumerate(env.tasks):
            world, cycle = task.world, task.world.cycle

            def ticked(world=world, cycle=cycle, b=b):
                cycle()
                write(dict(kind='tick', world=b, step=step, state=world_state(world)))
            world.cycle = ticked
            hooks.append((world, cycle, ticked))
        legal = [np.flatnonzero(obs['mask'][b]) for b in range(env.n)]
        action = np.asarray([legal[b][int(rng(R_ACTION, step, b) * len(legal[b]))] for b in range(env.n)], np.int32)
        state, obs, reward, done, info = env.step(state, action)
        for world, cycle, ticked in hooks:
            if world.cycle is ticked:
                world.cycle = cycle
        write(dict(kind='decision', step=step, action=action.tolist(), reward=reward.tolist(), done=done.tolist(),
                   info={k: v.tolist() for k, v in info.items()}))
        for b in np.flatnonzero(done):
            started(int(b))
        if info['unsupported'].any():
            write(dict(kind='unsupported', evidence=env.evidence))
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    dump = sub.add_parser('trace', help='keyed random legal play on CPU reference worlds, tick by tick')
    dump.add_argument('--pool', type=Path, required=True, help='a sim.world pool file')
    dump.add_argument('--worlds', type=int, default=4)
    dump.add_argument('--steps', type=int, default=100)
    dump.add_argument('--key', type=int, default=7)
    dump.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    layouts, derived = load_pool(args.pool)
    env = CpuBatch(Tables.load(), layouts, args.worlds, derived=None if derived[0] is None else derived)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        trace(env, args.key, args.steps, lambda row: stream.write(json.dumps(row, default=str) + '\n'))
    print(f'[vec] wrote {args.output}: {args.worlds} worlds, {args.steps} decisions, key {args.key}')


if __name__ == '__main__':
    main()
