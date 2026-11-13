"""search-wood-v1, the wood goal of goal-activity-v4.

A search episode starts where no tree is in anything the policy receives: every tree and oak of the layout stands
farther (Chebyshev, from its origin) than SEARCH['hidden'] tiles from the start tile, beyond every scan that lists a loc
(sim/agent.py: locs and affordances within 15 tiles, places within 32). The player must move to find one, then chop.
The certified outcome, return and audit are goal-activity-v2's wood rules (two newly produced logs still owned and
survival to the horizon; sim/audit.py); only the start and the horizon differ. Any route or strategy that produces the
logs counts. Fish, mine and melee keep goal-activity-v3's tasks (sim/combat.py).

Layouts come from the unchanged compact generator (sim/world.py); a layout serves the search when at least one tile of
its start component is a search tile (`search_tiles`): every tree and oak hidden, and a normal tree reachable by
walking. Hidden positions vary with the layouts; nothing about them is supplied. `draw_search` keeps task.draw_start's
keyed layout, weapon, levels and world seed and draws the start tile among the search tiles (purpose R_SEARCH).

The policy's navigation assistance is declared, not removed (act-v3): clicks at 4, 9 and 16 tiles in eight directions,
routed by the client's tryMove with its 3 x 3 fallback, the ACT listing within 15 tiles with the client's route to each
target, and npcs within 15 tiles. No waypoint, city label, target direction or frontier rule is supplied.

Instruments (development only, never training labels): `audit_start` lists any tree information in a start's raw
observation; `twin` moves a layout's hidden trees elsewhere beyond the scans, and `leakage` tests whether two worlds with
different hidden positions give the same first observation (19 of 20 development pairs did); the oracle reference (sim/run.py reference_action) reads the
world's trees and shows each start can be solved within the horizon; the explorer reference uses only the observation
(an untrained heuristic that clicks toward the least-visited sector, its memory dead reckoning from the feedback's dx and
dz) and shows the task needs no hidden information.

    .venv-sim/bin/python -m sim.search capacity --split development --layouts 200
    .venv-sim/bin/python -m sim.search pool --split train --count 128 --output logs/search/pool-train-128.npz
    .venv-sim/bin/python -m sim.search reference --split development --layouts 100 --output logs/search/dev/reference
    .venv-sim/bin/python -m sim.search leakage --split development --pairs 20 --initial CKPT --output logs/search/dev/leakage
    .venv-sim/bin/python -m sim.search evaluate --initial CKPT --policies initial wait random --split development \
        --layouts 50 --key 13 --output logs/search/dev/start-s101

supply-wood-v1, the wood goal of goal-activity-v5 (SUPPLY below), is sustained
production: a start beside one grove of the supply world, in paired ample and scarce variants, under the native
depletion and respawn rules, paid per certified log up to a cap. Its instruments and evaluation:

    .venv-sim/bin/python -m sim.search supply-pool --split train --count 128 --output logs/supply/pools/train-128.npz
    .venv-sim/bin/python -m sim.search supply-reference --split development --layouts 100 --output logs/supply/dev/reference
    .venv-sim/bin/python -m sim.search evaluate-supply --checkpoint CKPT --initial CKPT --policies final initial oracle stay \
        --split development --layouts 20 --key 13 --output logs/supply/dev/eval
    .venv-sim/bin/python -m sim.search replay-supply --split development --record 3 --variant scarce --policy CKPT \
        --output logs/supply/dev/replay
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

from act_contract import CLICK_ROWS, N_CLICK
from sim.agent import A_ACT0, A_CLOSE_IF, A_DIALOG_CONTINUE, A_HELD0, A_WAIT, ACT_SCAN_TILES, VISIT_CELL, family
from sim.combat import outside
from sim.engine import WOODCUTTING, Rng
from sim.world import (OAK, ORIGIN, SEARCH_WORLD, SPLITS, SUPPLY_WORLD, TREE, generate, relocate, split_seeds,
                       without_locs)

SEARCH = dict(
    version='search-wood-v1', task='goal-activity-v4',
    world=SEARCH_WORLD['version'],  # sim/world.py SEARCH_WORLD: the generator's draws on the whole 64-tile chunk
    # frozen as literals (act-v2's place scan, and it plus one compass move), so act-v3 draws the same records
    scan=32,                   # the farthest an act-v2 observation listed a loc (Chebyshev; places)
    # every tree and oak origin farther than this from the start tile: beyond the scan even after one compass move
    hidden=42,
    trees=(TREE, OAK),         # debug names of the hidden locs; an oak serves the goal from Woodcutting 15
    target=2,                  # goal-activity-v2's wood target and return, min(logs, 2) / 2
    horizon=240,               # native ticks from arm, goal-activity-v2's; frozen 22 October from development references
    pool=128,                  # training search layouts (train split, first found from index 0)
    cache=64,                  # path windows a search layout keeps per worker (sim.world LayoutTables.CACHE): no derived tables
    # a stall (evaluation telemetry, not a rule of the task): at least `decisions` decisions spanning at least `ticks`
    # ticks within `displacement` tiles (Chebyshev) of the window's first position, with no log produced in it
    stall=dict(decisions=20, ticks=50, displacement=3),
)
R_SEARCH = 30  # keyed draw purpose of the search start tile (sim/task.py R_* are 1-5, sim/combat.py's 16-21)
TREE_NAMES = ('Tree', 'Oak')  # the game names of SEARCH['trees'], as the observation lists them
INTERFACE = ('page', 'style', 'close', 'dialog', 'held', 'use', 'drop')


def tree_locs(layout, tables):
    """The rows of `layout.locs` that are trees or oaks."""
    return [k for k, row in enumerate(layout.locs) if tables.loc_types[int(row[0])]['debugname'] in SEARCH['trees']]


def search_tiles(layout, tables):
    """bool [W, H]: the start component's tiles with every tree and oak origin beyond SEARCH['hidden'] (Chebyshev) and a
    normal tree reachable by walking (the layout's wood distance)."""
    W, H = layout.flags.shape
    xs, zs = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
    nearest = np.full((W, H), 1 << 20)
    for k in tree_locs(layout, tables):
        x, z = int(layout.locs[k, 1]), int(layout.locs[k, 2])
        nearest = np.minimum(nearest, np.maximum(abs(xs - x), abs(zs - z)))
    return outside(layout) & (nearest > SEARCH['hidden']) & (layout.goal_dist[0] >= 0)


def iter_search_layouts(split, tables, first=0):
    """The split's layouts from seed index `first` that have a search tile, in seed order."""
    lo, hi = SPLITS[split]
    for seed in range(lo + first, hi):
        layout = generate(seed, tables, SEARCH_WORLD)
        if search_tiles(layout, tables).any():
            yield layout


def search_layouts(split, count, tables, first=0):
    """The first `count` layouts of iter_search_layouts: (layouts, seeds scanned)."""
    out = []
    for layout in iter_search_layouts(split, tables, first):
        out.append(layout)
        if len(out) == count:
            return out, layout.seed - SPLITS[split][0] - first + 1
    raise ValueError(f'the {split} split has only {len(out)} search layouts from index {first}')


def draw_search(key, slot, episode, layouts, tiles, layout=None):
    """task.draw_start's keyed start (layout, weapon, levels, world seed) with the tile drawn among the layout's search
    tiles (`tiles[layout]`, x-major order) under purpose R_SEARCH."""
    from sim.task import draw_start
    start = draw_start(key, slot, episode, layouts, layout=layout)
    found = np.argwhere(tiles[start['layout']])
    x, z = found[int(Rng.draw(Rng(key).prefix(slot, episode), R_SEARCH) * len(found))]
    return dict(start, tile=(ORIGIN[0] + int(x), ORIGIN[1] + int(z)), search=True)


def eligible(name, woodcutting):
    return name == 'Tree' or (name == 'Oak' and woodcutting >= 15)


def sees_tree(raw, woodcutting):
    """Whether an eligible tree is in the observation: a scanned loc (affordances are a subset)."""
    return any(eligible(t['name'], woodcutting) for t in raw.get('locs', []))


def audit_start(raw):
    """Tree information in a search start's raw observation (any tree or oak, whatever the level), as a list of problems;
    empty when nothing there names, points at or offers one."""
    problems = [f"{part} lists {t['name']}" for part in ('locs', 'actSel', 'npcs')
                for t in raw.get(part, []) if t.get('name') in TREE_NAMES]
    problems += [f"actSel offers {a['option']} on {a['name']}" for a in raw.get('actSel', []) if a['option'] == 'Chop down']
    problems += [f"ground lists {g['name']}" for g in raw.get('ground', [])]
    feedback = raw.get('feedback') or {}
    problems += [f"chat: {c}" for c in feedback.get('chat', [])]
    if (feedback.get('dialog') or {}).get('text'):
        problems.append('a dialog is open')
    return problems


# ---- episodes -------------------------------------------------------------------------------------------------
def episode(record, layout, lt, tables, cfg=None, exclusions=()):
    """A started search episode (GoalEpisode, wood) of a start record (draw_search's dict) in a fresh world. exclusions:
    for a measurement longer than the task, which reaches the random-event roll at tick 500 (sim/engine.py EXCLUSIONS)."""
    from baseline import config
    from sim.engine import World
    from sim.task import WORLD_START_TICK, GoalEpisode
    world = World(lt, seed=record['world_seed'], start_tick=WORLD_START_TICK, exclusions=exclusions)
    task = GoalEpisode(world, cfg or config.TASK_CFG, 'wood', horizon=SEARCH['horizon'], scale=1.0)
    task.start(record['tile'], record['weapon'], record['levels'])
    return task


class Tracker:
    """What a search episode has seen and where it has been, from its raw observations: the tick (from arm) at which an
    eligible tree was first observed, the decisions' positions and the logs produced."""

    def __init__(self, task):
        self.task, self.discovered, self.cells = task, None, set()
        self.observe()

    def observe(self):
        task = self.task
        p = task.agent.player
        self.cells.add((p.x // VISIT_CELL, p.z // VISIT_CELL))
        if self.discovered is None and sees_tree(task.raw, p.levels[WOODCUTTING]):
            self.discovered = task.world.tick - task.start_tick


# ---- references (feasibility instruments, never labels) ---------------------------------------------------------
class OracleActor:
    """sim/run.py's reference controller: reads the world's trees, acts only through legal published actions."""

    def reset(self, seed):
        pass

    def act(self, observation, task=None, trace=False):
        from sim.run import reference_action
        action = reference_action(task)
        return (action, None) if trace else action


class ExplorerActor:
    """Observation only: chop an offered eligible tree (fewest steps), else take the legal click whose target tile is
    open and lies in the 8-tile cell its memory holds least (ties: the farthest click, then the lowest slot). Its memory
    is dead reckoning from the feedback's dx and dz, which the policy also gets: the cells it stood in at its decisions
    and the cells it clicked toward, each counted, from the start tile as (0, 0)."""

    def reset(self, seed):
        self.x = self.z = 0
        self.cells = Counter()

    def act(self, observation, task=None, trace=False):
        obs, wc = task.agent.last_observation, task.agent.player.levels[WOODCUTTING]
        fb, mask = obs['feedback'], obs['mask']
        self.x, self.z = self.x + fb['dx'], self.z + fb['dz']
        self.cells[(self.x // VISIT_CELL, self.z // VISIT_CELL)] += 1
        action = A_WAIT
        if mask[A_DIALOG_CONTINUE]:
            action = A_DIALOG_CONTINUE
        elif obs['dialogOpen'] or obs['modalOpen']:
            action = A_CLOSE_IF
        elif not obs.get('optionBusy'):
            chop = [(a['steps'], k) for k, a in enumerate(obs['actSel'])
                    if a['option'] == 'Chop down' and eligible(a['name'], wc) and mask[A_ACT0 + k]]
            cell = lambda c: ((self.x + c['dx']) // VISIT_CELL, (self.z + c['dz']) // VISIT_CELL)
            clicks = [(self.cells[cell(c)], -max(abs(c['dx']), abs(c['dz'])), k) for k, c in enumerate(obs['clicks'])
                      if c['open'] and mask[CLICK_ROWS[k]]]
            if chop:
                action = A_ACT0 + min(chop)[1]
            elif clicks:
                k = min(clicks)[2]
                self.cells[cell(obs['clicks'][k])] += 1
                action = CLICK_ROWS[k]
        return (action, None) if trace else action


def actor_for(policy, paths, sampled=False):
    from sim.combat import ControlActor, PolicyActor
    if policy in paths:
        return PolicyActor(paths[policy], deterministic=not sampled)
    if policy == 'oracle':
        return OracleActor()
    if policy == 'explorer':
        return ExplorerActor()
    return ControlActor(policy)


# ---- supply-wood-v1: sustained production -----------------------------------
# The wood goal of goal-activity-v5. The start stands beside one grove (its "local" supply) with every other grove beyond
# the 15-tile loc scan. Two variants of each start share the layout, tile, levels, weapon and world seed: `ample` keeps
# the grove as drawn; `scarce` keeps only its normal tree nearest the start (a normal tree gives one log, then stands as a
# stump for 120-199 ticks; sim/engine.py get_logs, the native rule). Nothing else changes: other groves, mechanics and
# observations are the same, and nothing marks a grove, a route or the variant. The return pays each certified log owned
# at the horizon, min(logs, cap) / cap, and death settles it to zero; success is at least `quota` logs and survival.
SUPPLY = dict(
    version='supply-wood-v1', task='goal-activity-v5', world=SUPPLY_WORLD['version'],
    near=6,                    # a start tile stands within this many tiles (Chebyshev) of a tree of its grove ...
    apart=ACT_SCAN_TILES,      # ... and farther than the loc scan (15) from every tree of the other groves
    linkage=13,                # trees closer than this share a grove (sim/world.py keeps groves at least 14 apart)
    ample=3,                   # the start's grove, and at least one other, hold at least this many normal trees
    variants=('ample', 'scarce'),
    exclusions=('random_events',),  # sim/engine.py EXCLUSIONS: the readiness roll at tick 500, as sustained combat
    # frozen 23 October from the development references (100 paired records, logs/supply/dev/reference): at 300
    # ticks inventory (21 free slots) rarely binds (the oracle's 90th percentile 17 logs), and 8 logs separates staying in
    # a scarce grove (0 of 100) from moving (oracle 97, observation-only explorer 96) and from staying in an ample one (96)
    horizon=300, cap=20, quota=8,
    pool=128, cache=64,
)
R_SUPPLY_GROVE, R_SUPPLY_TILE, R_SUPPLY_VARIANT = 31, 32, 33  # keyed draw purposes (R_SEARCH is 30)


def chebyshev_map(layout, points):
    """int [W, H]: each tile's Chebyshev distance to the nearest of `points` (local (x, z)); large where none."""
    W, H = layout.flags.shape
    xs, zs = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
    out = np.full((W, H), 1 << 20)
    for x, z in points:
        out = np.minimum(out, np.maximum(abs(xs - x), abs(zs - z)))
    return out


def groves(layout, tables):
    """The layout's tree and oak rows grouped by single linkage at SUPPLY['linkage'] tiles (Chebyshev), each group in row
    order, the groups ordered by their first row."""
    rows = tree_locs(layout, tables)
    pos = {k: (int(layout.locs[k, 1]), int(layout.locs[k, 2])) for k in rows}
    root = {k: k for k in rows}

    def find(k):
        while root[k] != k:
            k = root[k]
        return k
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if max(abs(pos[a][0] - pos[b][0]), abs(pos[a][1] - pos[b][1])) < SUPPLY['linkage']:
                root[find(b)] = find(a)
    found = {}
    for k in rows:
        found.setdefault(find(k), []).append(k)
    return sorted(found.values(), key=min)


def is_normal(layout, tables, k):
    return tables.loc_types[int(layout.locs[k, 0])]['debugname'] == TREE


def supply_sites(layout, tables):
    """[(grove index, bool [W, H] start tiles)] of the groves that can be a start's local supply: at least SUPPLY['ample']
    normal trees, another grove with as many, and start-component tiles within SUPPLY['near'] of one of its trees whose
    nearest tree of any other grove is beyond SUPPLY['apart'] (a normal tree reachable by walking)."""
    found, out = groves(layout, tables), []
    normal = [sum(is_normal(layout, tables, k) for k in g) for g in found]
    base = outside(layout) & (layout.goal_dist[0] >= 0)
    at = lambda rows: [(int(layout.locs[k, 1]), int(layout.locs[k, 2])) for k in rows]
    for i, grove in enumerate(found):
        if normal[i] < SUPPLY['ample'] or not any(n >= SUPPLY['ample'] for j, n in enumerate(normal) if j != i):
            continue
        others = [k for j, g in enumerate(found) if j != i for k in g]
        tiles = base & (chebyshev_map(layout, at(grove)) <= SUPPLY['near']) & (chebyshev_map(layout, at(others)) > SUPPLY['apart'])
        if tiles.any():
            out.append((i, tiles))
    return out


def iter_supply_layouts(split, tables, first=0):
    """The split's supply-world layouts from seed index `first` with a supply site, in seed order."""
    lo, hi = SPLITS[split]
    for seed in range(lo + first, hi):
        layout = generate(seed, tables, SUPPLY_WORLD)
        if supply_sites(layout, tables):
            yield layout


def supply_layouts(split, count, tables, first=0):
    """The first `count` layouts of iter_supply_layouts: (layouts, seeds scanned)."""
    out = []
    for layout in iter_supply_layouts(split, tables, first):
        out.append(layout)
        if len(out) == count:
            return out, layout.seed - SPLITS[split][0] - first + 1
    raise ValueError(f'the {split} split has only {len(out)} supply layouts from index {first}')


def draw_supply(key, slot, episode, layouts, sites, layout=None, variant=None):
    """task.draw_start's keyed start (layout, weapon, levels, world seed) with the local grove (purpose R_SUPPLY_GROVE)
    among the layout's sites, the tile (R_SUPPLY_TILE) among that grove's start tiles and the variant (R_SUPPLY_VARIANT,
    equally likely) unless given: an evaluation plays both variants of one record."""
    from sim.task import draw_start
    start = draw_start(key, slot, episode, layouts, layout=layout)
    h = Rng(key).prefix(slot, episode)
    options = sites[start['layout']]
    grove, tiles = options[int(Rng.draw(h, R_SUPPLY_GROVE) * len(options))]
    found = np.argwhere(tiles)
    x, z = found[int(Rng.draw(h, R_SUPPLY_TILE) * len(found))]
    if variant is None:
        variant = SUPPLY['variants'][int(Rng.draw(h, R_SUPPLY_VARIANT) * len(SUPPLY['variants']))]
    return dict(start, tile=(ORIGIN[0] + int(x), ORIGIN[1] + int(z)), grove=int(grove), variant=variant, supply=True)


def scarce_rows(layout, tables, grove_rows, tile):
    """(rows the scarce variant removes, the row it keeps): the grove's trees and oaks except its normal tree nearest the
    start tile (local (x, z); ties: the lower row)."""
    keep = min((k for k in grove_rows if is_normal(layout, tables, k)),
               key=lambda k: (max(abs(int(layout.locs[k, 1]) - tile[0]), abs(int(layout.locs[k, 2]) - tile[1])), k))
    return [k for k in grove_rows if k != keep], keep


def variant_layout(layout, tables, record):
    """The layout a supply record plays: as drawn (ample), or without its grove's other trees (scarce)."""
    if record['variant'] == 'ample':
        return layout
    tile = (record['tile'][0] - ORIGIN[0], record['tile'][1] - ORIGIN[1])
    removed, _ = scarce_rows(layout, tables, groves(layout, tables)[record['grove']], tile)
    out = without_locs(layout, tables, removed)
    if out is None:
        raise RuntimeError(f'layout {layout.seed}: the scarce variant of grove {record["grove"]} fails a check')
    return out


# The WASD-only variant (decided 2 November): a feasibility instrument with four
# one-tile clicks (north, east, south and west) on the radius-4 slots of those directions, dispatched as the click rows
# are; the other slots are empty. Reported, never gated, and never a policy's click table.
WASD = tuple({0: (0, 1), 2: (1, 0), 4: (0, -1), 6: (-1, 0)}.get(k) for k in range(N_CLICK))


def supply_episode(record, lt, tables, cfg=None, clicks=None):
    """A started supply episode (GoalEpisode, wood, SUPPLY's horizon and cap, its exclusions) of a start record in a fresh
    world on `lt` (LayoutTables over the record's variant layout). clicks: an instrument's click table (WASD) in place of
    act-v3's; its observations are not act-v3's, so the episode projects only their mask, which no policy reads."""
    from baseline import config
    from sim.engine import World
    from sim.task import WORLD_START_TICK, GoalEpisode

    class InstrumentEpisode(GoalEpisode):
        def project(self):
            self.projected = dict(mask=np.asarray(self.raw['mask'], np.int8))
            return self.projected
    world = World(lt, seed=record['world_seed'], start_tick=WORLD_START_TICK, exclusions=SUPPLY['exclusions'])
    task = (GoalEpisode if clicks is None else InstrumentEpisode)(world, cfg or config.TASK_CFG, 'wood',
                                                                  horizon=SUPPLY['horizon'], scale=1.0,
                                                                  target=SUPPLY['cap'])
    if clicks is not None:
        task.agent.click_offsets = clicks
    task.start(record['tile'], record['weapon'], record['levels'])
    return task


class SupplyTracker(Tracker):
    """A search Tracker that also follows the start's grove (telemetry read from the world, never an input): how many of
    its trees stand ready to cut at each decision, the first tick none does (local supply exhausted), and each log produced
    as (tick from arm, local), local when the player stood within 3 tiles of one of the grove's tree positions."""

    def __init__(self, task, grove_positions):
        from sim.run import family_ids
        self.positions = [(ORIGIN[0] + x, ORIGIN[1] + z) for x, z in grove_positions]
        self.exhausted, self.available, self.logs = None, [], []
        self.family, self.seen = family_ids(task.world.t, 'wood'), task.event_start
        super().__init__(task)

    def standing(self):
        w = self.task.world
        names = {TREE, OAK}
        return sum(any(loc.active and w.t.loc_types[loc.type]['debugname'] in names for loc in w.loc_at.get((0, x, z), ()))
                   for x, z in self.positions)

    def observe(self):
        super().observe()
        task, p = self.task, self.task.agent.player
        events = task.world.events
        for tick, kind, data in events[self.seen:]:
            if kind == 'produce' and data['obj'] in self.family:
                self.logs += [(tick - task.start_tick, self.local(p.x, p.z))] * data['count']
        self.seen = len(events)
        n = self.standing()
        self.available.append(n)
        if n == 0 and self.exhausted is None:
            self.exhausted = task.world.tick - task.start_tick

    def gap(self):
        """Ticks from the grove's first exhaustion to the first log produced elsewhere after it, or None."""
        after = [t for t, local in self.logs if not local and self.exhausted is not None and t >= self.exhausted]
        return None if not after else after[0] - self.exhausted

    def local(self, x, z):
        return any(max(abs(x - a), abs(z - b)) <= 3 for a, b in self.positions)


class StayActor:
    """The oracle restricted to the start's grove (privileged): chop its nearest standing tree, else wait. The staying
    strategy's best case, a feasibility instrument, never a label."""

    def __init__(self, positions):
        self.positions = {(ORIGIN[0] + x, ORIGIN[1] + z) for x, z in positions}

    def reset(self, seed):
        pass

    def act(self, observation, task=None, trace=False):
        from sim.run import reference_targets
        obs, mask = task.agent.last_observation, task.agent.last_observation['mask']
        action = A_WAIT
        if mask[A_DIALOG_CONTINUE]:
            action = A_DIALOG_CONTINUE
        elif obs['dialogOpen'] or obs['modalOpen']:
            action = A_CLOSE_IF
        elif not obs.get('optionBusy'):
            targets = {t for t in reference_targets(task) if (t[1], t[2]) in self.positions}
            offered = [k for k, sel in enumerate(obs['actSel']) if (sel['kind'], sel['x'], sel['z'], sel['option']) in targets]
            if offered:
                action = A_ACT0 + offered[0]
        return (action, None) if trace else action


def supply_trial(record, layout, tables, actor, seed, horizon=None, trace=False, cfg=None, hook=None, clicks=None):
    """One supply trial of a record (its variant included) with `actor`: (decision rows, diagnostic rows or None, final
    info, SupplyTracker, task). horizon: the frozen one, or a development measurement's longer window. hook(task): called
    after arm and after every decision (the replay's snapshots). clicks: an instrument's click table (supply_episode)."""
    from sim.world import LayoutTables
    played = variant_layout(layout, tables, record)
    lt = LayoutTables(tables, played)
    lt.CACHE = SUPPLY['cache']
    saved = SUPPLY['horizon']
    SUPPLY['horizon'] = horizon or saved
    try:
        task = supply_episode(record, lt, tables, cfg, clicks)
    finally:
        SUPPLY['horizon'] = saved
    grove = groves(layout, tables)[record['grove']]
    positions = [(int(layout.locs[k, 1]), int(layout.locs[k, 2])) for k in grove]
    actor.reset(seed)
    tracker = SupplyTracker(task, positions)
    obs, rows, diag, p = task.projected, [], [] if trace else None, task.agent.player
    info = None
    if hook:
        hook(task)
    for _ in range(4000):
        tick, raw = task.world.tick, task.raw
        if trace:
            action, extra = actor.act(obs, task, trace=True)
            diag.append(dict(tick=tick - task.start_tick, legal=np.flatnonzero(obs['mask']).tolist(), **(extra or {})))
        else:
            action = actor.act(obs, task)
        kind = family(action)
        option = raw['actSel'][action - A_ACT0]['option'] if kind == 'act' and action - A_ACT0 < len(raw['actSel']) else \
            raw['heldSel'][action - A_HELD0]['option'] if kind == 'held' and action - A_HELD0 < len(raw['heldSel']) else None
        obs, reward, done, info = task.step(action)
        ended = task.raw['feedback'].get('ended') if not done else None
        tracker.observe()
        rows.append([tick - task.start_tick, action, option, round(float(reward), 6), p.x - ORIGIN[0], p.z - ORIGIN[1],
                     tracker.discovered is not None, ended and ended['reason'], bool(ended and ended['refused']),
                     task.raw.get('actionNote'), p.hp, tracker.available[-1]])
        if hook:
            hook(task)
        if done:
            return rows, diag, info, tracker, task
    return rows, diag, None, tracker, task


def replay_supply(output, record, layout, tables, actor, seed, label=''):
    """A watched replay of one supply trial (reproducible: the record, policy file, sampling mode and seed replay it
    exactly): trace.json with the layout's map and, after arm and every decision, the tick, position, every tree's state,
    logs so far, hitpoints, food and the decision; and replay.html, a self-contained page that draws it with a time
    slider. Returns the trial's outcome."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trees = [(int(layout.locs[k, 1]), int(layout.locs[k, 2]), tables.loc_types[int(layout.locs[k, 0])]['debugname'])
             for k in tree_locs(layout, tables)]
    local = set(groves(layout, tables)[record['grove']])
    frames = []
    from sim.run import family_ids
    wood = family_ids(tables, 'wood')

    def snapshot(task):
        w, p = task.world, task.agent.player
        state = []
        for x, z, _ in trees:
            here = w.loc_at.get((0, ORIGIN[0] + x, ORIGIN[1] + z), ())
            state.append('standing' if any(loc.active and w.t.loc_types[loc.type]['debugname'] in (TREE, OAK) for loc in here)
                         else 'stump' if any(loc.active for loc in here) else 'none')
        from sim.combat import food_count
        frames.append(dict(tick=w.tick - task.start_tick, x=p.x - ORIGIN[0], z=p.z - ORIGIN[1], trees=state, hp=p.hp,
                           food=food_count(task.raw), logs=sum(d['count'] for _, kind, d in w.events[task.event_start:]
                                                               if kind == 'produce' and d['obj'] in wood)))
    rows, _, info, tracker, task = supply_trial(record, layout, tables, actor, seed, hook=snapshot)
    for frame, row in zip(frames[1:], rows):
        frame.update(action=row[1], family=family(row[1]), option=row[2], note=row[9])
    walls = [[int(x), int(z)] for x, z in zip(*np.nonzero(layout.flags & 0x100))]
    water = [[int(x), int(z)] for x, z in zip(*np.nonzero(layout.flags & 0x200000))]
    outcome = dict(logs=int(info['outcomes']) if info else None, survived=bool(info and info['report']['survived']),
                   success=bool(info and info['report']['survived'] and info['outcomes'] >= SUPPLY['quota']),
                   logs_local=sum(flag for _, flag in tracker.logs), exhausted_tick=tracker.exhausted, gap=tracker.gap())
    trace = dict(label=label, record=record, size=list(layout.flags.shape), quota=SUPPLY['quota'], horizon=SUPPLY['horizon'],
                 trees=[dict(x=x, z=z, kind=kind, local=k in local) for (x, z, kind), k in zip(trees, tree_locs(layout, tables))],
                 walls=walls, water=water, frames=frames, outcome=outcome)
    (output / 'trace.json').write_text(json.dumps(trace, default=str) + '\n')
    (output / 'replay.html').write_text(REPLAY_PAGE.replace('__TRACE__', json.dumps(trace, default=str)))
    return outcome


REPLAY_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Supply replay</title><style>
body{font:13px system-ui,sans-serif;margin:16px;color:#1b1b1b;background:#fcfcfb} svg{background:#f3f1ea;border:1px solid #d9d8d4}
#info{margin:8px 0;white-space:pre} input[type=range]{width:520px}</style></head><body>
<div id="title"></div><svg id="map" width="520" height="520"></svg><div><button id="play">Play</button>
<input id="t" type="range" min="0" value="0"></div><div id="info"></div><script>
const T=__TRACE__, S=8, W=T.size[0], H=T.size[1], svg=document.getElementById('map'), ns='http://www.w3.org/2000/svg';
document.getElementById('title').textContent=`${T.label} record ${T.record.record??''} ${T.record.variant}: ${T.outcome.logs} logs`+
 ` (quota ${T.quota}), ${T.outcome.survived?'survived':'died'}; local supply exhausted at tick ${T.outcome.exhausted_tick??'never'};`+
 ` first log elsewhere after it ${T.outcome.gap==null?'none':T.outcome.gap+' ticks later'}`;
const el=(n,a)=>{const e=document.createElementNS(ns,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
const at=(x,z)=>[x*S,(H-1-z)*S];
for(const [x,z] of T.water){const [a,b]=at(x,z);el('rect',{x:a,y:b,width:S,height:S,fill:'#b9d3ea'})}
for(const [x,z] of T.walls){const [a,b]=at(x,z);el('rect',{x:a,y:b,width:S,height:S,fill:'#9b9a96'})}
const trees=T.trees.map(t=>{const [a,b]=at(t.x,t.z);return el('circle',{cx:a+S/2,cy:b+S/2,r:S*0.45,stroke:t.local?'#eb6834':'none','stroke-width':2})});
const trail=el('polyline',{fill:'none',stroke:'#2a78d6','stroke-width':1.5,'stroke-opacity':0.6}), me=el('circle',{r:S*0.5,fill:'#0b0b0b'});
const slider=document.getElementById('t'); slider.max=T.frames.length-1;
function draw(i){const f=T.frames[i];
 f.trees.forEach((s,k)=>trees[k].setAttribute('fill',s==='standing'?(T.trees[k].kind==='oaktree'?'#2f6b2f':'#5fa35f'):s==='stump'?'#8a6a45':'#ddd'));
 trail.setAttribute('points',T.frames.slice(0,i+1).map(g=>{const [a,b]=at(g.x,g.z);return `${a+S/2},${b+S/2}`}).join(' '));
 const [a,b]=at(f.x,f.z); me.setAttribute('cx',a+S/2); me.setAttribute('cy',b+S/2);
 document.getElementById('info').textContent=`tick ${f.tick}  logs ${f.logs}  hitpoints ${f.hp}  food ${f.food}\n`+
  `decision: ${f.family??'(arm)'} ${f.option??''} ${f.note??''}\norange ring: the start's grove; green: tree; dark green: oak; brown: stump`}
slider.oninput=()=>draw(+slider.value); let timer=null;
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}
 timer=setInterval(()=>{if(+slider.value>=+slider.max){clearInterval(timer);timer=null;return}slider.value=+slider.value+1;draw(+slider.value)},120)};
draw(0);</script></body></html>
"""


def supply_measures(rows, produced, tracker):
    """A supply trial's production measures: the logs produced at the start's grove and elsewhere, the tick the grove
    first had no standing tree and the gap from it to the first log elsewhere (SupplyTracker), plus measures()."""
    local = sum(flag for _, flag in tracker.logs)
    gap = tracker.gap()
    return dict(measures(rows, produced, tracker.discovered, len(tracker.cells)), logs_local=local,
                logs_elsewhere=len(tracker.logs) - local, exhausted_tick=tracker.exhausted,
                first_elsewhere_after=None if gap is None else tracker.exhausted + gap, exhausted_to_elsewhere=gap)


def supply_reference(output, *, split, layouts, tables, key=13, first=0, horizon=1200,
                     policies=('oracle', 'stay', 'explorer', 'wait'), part=(0, 1), clicks='act-v3'):
    """Development references on supply records (both variants, never the final split): each policy's logs by tick over
    `horizon`, where they were made and when the start's grove ran out (the frozen
    horizon, cap and quota came from the act-v2 ones). clicks: 'act-v3', or 'wasd' for the WASD-only variant."""
    from sim.engine import Unsupported
    if split == 'final':
        raise ValueError('development instruments never read the final split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pool, _ = supply_layouts(split, layouts, tables, first)
    sites = [supply_sites(layout, tables) for layout in pool]
    rows_out = []
    for k in range(part[0], layouts, part[1]):
        for variant in SUPPLY['variants']:
            record = draw_supply(key, 0, k, pool, sites, layout=k, variant=variant)
            grove = groves(pool[k], tables)[record['grove']]
            positions = [(int(pool[k].locs[r, 1]), int(pool[k].locs[r, 2])) for r in grove]
            for policy in policies:
                actor = dict(oracle=OracleActor, explorer=ExplorerActor)[policy]() if policy in ('oracle', 'explorer') else \
                    StayActor(positions) if policy == 'stay' else __import__('sim.combat', fromlist=['ControlActor']).ControlActor(policy)
                base = dict(record=k, variant=variant, policy=policy, layout=pool[k].seed, grove=record['grove'],
                            levels=record['levels'], local_normal=sum(is_normal(pool[k], tables, r) for r in grove))
                try:
                    rows, _, info, tracker, task = supply_trial(record, pool[k], tables, actor, 505001 + k, horizon=horizon,
                                                                clicks=None if clicks == 'act-v3' else WASD)
                except Unsupported as error:
                    rows_out.append(dict(base, unsupported=str(error)))
                    continue
                produced = produced_ticks(task)
                rows_out.append(dict(base, survived=bool(info and info['report']['survived']), produced=produced,
                                     decisions=len(rows), **supply_measures(rows, produced, tracker)))
    (output / f'reference-{part[0]}.json').write_text(json.dumps(dict(split=split, layouts=layouts, key=key, first=first,
                                                                     horizon=horizon, clicks=clicks, policies=list(policies),
                                                                     trials=rows_out),
                                                                indent=1, default=str) + '\n')
    return rows_out


def supply_records(split, layouts, key, tables, first=0, only=None):
    """The frozen supply records: record k on the k-th supply layout of the split from seed index `first`, with
    draw_supply(key, 0, k, ..., layout=k); each is played in both variants. Returns (layouts, records without a variant,
    seeds scanned, sites)."""
    pool, scanned = supply_layouts(split, layouts, tables, first)
    sites = {k: supply_sites(pool[k], tables) for k in range(layouts)}
    wanted = range(layouts) if only is None else only
    recs = {k: {key2: v for key2, v in draw_supply(key, 0, k, pool, sites, layout=k, variant='ample').items() if key2 != 'variant'}
            for k in wanted}
    return {k: pool[k] for k in wanted}, recs, scanned, sites


def evaluate_supply(output, *, checkpoint=None, initial=None, previous=None, policies=('final', 'initial'),
                    split='development', layouts=200, key=13, first=0, part=(0, 1), diagnostics=4, seed=505001, tables=None,
                    sampled=False):
    """Every policy on both variants of the same frozen supply records, each trial in a fresh world (trial seed seed +
    record, the same for both variants). part=(i, n): this process plays records i, i + n, ... (merge_supply joins the
    parts). Every trial keeps its decision rows (SUPPLY_FIELDS); checkpoint policies keep diagnostic rows (legal actions,
    five likeliest actions, value) for their first `diagnostics` records and every failure. Controls: oracle, stay (the
    oracle within the start's grove), explorer, wait, random. Writes results-{i}.json and trials-{i}.jsonl.gz."""
    from sim.combat import ControlActor
    from sim.data import Tables
    from sim.engine import Unsupported
    from baseline.telemetry import sha256
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = tables or Tables.load()
    t0 = time.monotonic()
    mine = range(part[0], layouts, part[1])
    pool, recs, scanned, _ = supply_records(split, layouts, key, tables, first, only=mine)
    paths = {name: p for name, p in dict(final=checkpoint, initial=initial, previous=previous).items() if p}
    rows, timing = [], {}
    stream = gzip.open(output / f'trials-{part[0]}.jsonl.gz', 'wt')
    for policy in policies:
        t1 = time.monotonic()
        shared = actor_for(policy, paths, sampled) if policy in paths or policy in ('oracle', 'explorer') else None
        for k in mine:
            grove = groves(pool[k], tables)[recs[k]['grove']]
            positions = [(int(pool[k].locs[r, 1]), int(pool[k].locs[r, 2])) for r in grove]
            tile = (recs[k]['tile'][0] - ORIGIN[0], recs[k]['tile'][1] - ORIGIN[1])
            others = [r for j, g in enumerate(groves(pool[k], tables)) if j != recs[k]['grove'] for r in g]
            far = min(max(abs(int(pool[k].locs[r, 1]) - tile[0]), abs(int(pool[k].locs[r, 2]) - tile[1])) for r in others)
            for variant in SUPPLY['variants']:
                rec = dict(recs[k], variant=variant)
                actor = shared or (StayActor(positions) if policy == 'stay' else ControlActor(policy))
                base = dict(policy=policy, record=k, variant=variant, layout=pool[k].seed, grove=rec['grove'],
                            alternative_tiles=far)
                try:
                    decisions, diag, info, tracker, task = supply_trial(rec, pool[k], tables, actor, seed + k, trace=policy in paths)
                except Unsupported as error:  # counted apart, never scored
                    rows.append(dict(base, unsupported=str(error)))
                    stream.write(json.dumps(dict(base, start=rec, unsupported=str(error)), default=str) + '\n')
                    continue
                if info is None:
                    rows.append(dict(base, decision_limit=True))
                    continue
                paid = sum(r[3] for r in decisions)
                if abs(paid - info['episode']['r']) > 1e-6:
                    raise RuntimeError('paid rewards disagree with the audited return')
                logs = produced_ticks(task)
                survived = bool(info['report']['survived'])
                row = dict(base, success=survived and info['outcomes'] >= SUPPLY['quota'], logs=int(info['outcomes']),
                           episode_return=info['episode']['r'], survived=survived, decisions=info['episode']['l'],
                           ticks=info['episode']['native_ticks'], min_hp=min(r[10] for r in decisions),
                           logs_by_100=[sum(t < h for t in logs) for h in range(100, SUPPLY['horizon'] + 1, 100)],
                           **supply_measures(decisions, logs, tracker))
                rows.append(row)
                keep = policy in paths and (k < diagnostics or not row['success'])
                stream.write(json.dumps(dict(base, start=rec, outcome=row, report=info['report'], decisions=decisions,
                                             produced=logs, available=tracker.available,
                                             diagnostic=diag if keep else None), default=str) + '\n')
        timing[policy] = time.monotonic() - t1
        print(f'[supply] part {part[0]}/{part[1]}: {policy} {timing[policy]:.0f}s', flush=True)
    stream.close()
    from sim.agent import HELD_CONTRACT, MASK_CONTRACT
    from sim.engine import RETALIATE_CONTRACT
    result = dict(version=SUPPLY['version'], horizon=SUPPLY['horizon'], cap=SUPPLY['cap'], quota=SUPPLY['quota'], split=split,
                  layouts=layouts, key=key, first=first, scanned=scanned, part=list(part), sampled=sampled, mask=MASK_CONTRACT,
                  retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT,
                  decision_fields=list(SUPPLY_FIELDS), diagnostics=diagnostics, seed=seed, policies=list(policies),
                  variants=list(SUPPLY['variants']),
                  checkpoints={n: dict(path=str(p), sha256=sha256(p)) for n, p in paths.items() if n in policies},
                  layout_seeds=[pool[k].seed for k in mine], timing_seconds=timing, wall_seconds=time.monotonic() - t0,
                  trials=rows)
    (output / f'results-{part[0]}.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def summarize_supply(trials):
    """Counts and medians of one policy's scored supply trials in one variant."""
    from baseline.compare import wilson
    scored = [t for t in trials if 'success' in t]
    ok = sum(t['success'] for t in scored)
    median = lambda key: (lambda v: v[len(v) // 2] if v else None)(sorted(t[key] for t in scored if t.get(key) is not None))
    return dict(trials=len(trials), scored=len(scored), unsupported=sum('unsupported' in t for t in trials),
                decision_limit=sum('decision_limit' in t for t in trials), successes=ok, interval=wilson(ok, len(scored)),
                deaths=sum(not t['survived'] for t in scored), logs_mean=round(float(np.mean([t['logs'] for t in scored])), 3)
                if scored else None, logs_elsewhere=sum(t['logs_elsewhere'] for t in scored),
                exhausted=sum(t['exhausted_tick'] is not None for t in scored),
                went_elsewhere=sum(t['exhausted_to_elsewhere'] is not None for t in scored),
                median_exhausted_to_elsewhere=median('exhausted_to_elsewhere'), stalls=sum(t['stalls'] for t in scored),
                stalled_ticks=sum(t['stalled_ticks'] for t in scored), refused=sum(t['refused'] for t in scored),
                interface_alternations=sum(t['interface_alternations'] for t in scored), valid=len(scored) == len(trials))


def merge_supply(output):
    """Join a supply evaluation's parts into results.json with per-policy, per-variant summaries; refuses missing or
    repeated (policy, record, variant) trials."""
    output = Path(output)
    parts = [json.loads(p.read_text()) for p in sorted(output.glob('results-*.json'))]
    if not parts:
        raise ValueError(f'no parts in {output}')
    first, n = parts[0], parts[0]['part'][1]
    if sorted(p['part'][0] for p in parts) != list(range(n)) or any(p['part'][1] != n for p in parts):
        raise ValueError('parts missing or of different splits')
    for key in ('version', 'horizon', 'cap', 'quota', 'split', 'layouts', 'key', 'first', 'policies', 'seed', 'checkpoints',
                'sampled', 'mask', 'retaliate', 'held'):  # a part with no `held` (held-v1) differs from a held-v2 part
        if any(p.get(key) != first.get(key) for p in parts):
            raise ValueError(f'parts differ in {key}')
    trials = sorted((t for p in parts for t in p['trials']),
                    key=lambda t: (first['policies'].index(t['policy']), t['record'], SUPPLY['variants'].index(t['variant'])))
    for policy in first['policies']:
        for variant in SUPPLY['variants']:
            if sorted(t['record'] for t in trials if t['policy'] == policy and t['variant'] == variant) != list(range(first['layouts'])):
                raise ValueError(f'{policy} {variant}: records missing or repeated')
    result = {k: v for k, v in first.items() if k not in ('part', 'trials', 'layout_seeds', 'timing_seconds', 'wall_seconds')}
    result.update(parts=n, timing_seconds={p['part'][0]: p['timing_seconds'] for p in parts},
                  cells={policy: {v: summarize_supply([t for t in trials if t['policy'] == policy and t['variant'] == v])
                                  for v in SUPPLY['variants']} for policy in first['policies']},
                  trials=trials)
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


# ---- one trial and its measures -------------------------------------------------------------------------------
# A decision row of play(), in this order.
DECISION_FIELDS = ('tick', 'action', 'option', 'reward', 'x', 'z', 'sees_tree', 'ended', 'refused', 'note')
# A supply trial's decision row adds the hitpoints after the decision and the standing trees of the start's grove.
SUPPLY_FIELDS = (*DECISION_FIELDS, 'hp', 'grove_standing')


def stalls(rows, produced):
    """SEARCH['stall'] windows in a trial's decision rows: (count, ticks stalled). `produced`: ticks of logs."""
    rule = SEARCH['stall']
    count = ticks = 0
    i = 0
    while i < len(rows):
        x0, z0, t0 = rows[i][4], rows[i][5], rows[i][0]
        j = i
        while j + 1 < len(rows) and max(abs(rows[j + 1][4] - x0), abs(rows[j + 1][5] - z0)) <= rule['displacement'] \
                and not any(t0 <= t <= rows[j + 1][0] for t in produced):
            j += 1
        span = (rows[j][0] - t0) if j > i else 0
        if j - i + 1 >= rule['decisions'] and span >= rule['ticks']:
            count, ticks = count + 1, ticks + span
            i = j + 1
        else:
            i += 1
    return count, ticks


def interface_loops(actions):
    """A-B-A alternations of interface actions (INTERFACE families) in a sequence of actions."""
    kinds = [family(a) for a in actions]
    return sum(1 for k in range(2, len(actions)) if actions[k] == actions[k - 2] != actions[k - 1]
               and kinds[k] in INTERFACE and kinds[k - 1] in INTERFACE)


def measures(rows, produced, discovered, cells):
    """A trial's search measures from its decision rows (DECISION_FIELDS), the ticks of logs produced, the discovery
    tick and the distinct 8x8 cells visited: revisits (moves into an earlier cell other than the previous one),
    stalls, refused requests, interface actions and their A-B-A alternations, path length in tiles."""
    seq = [(r[4] // VISIT_CELL, r[5] // VISIT_CELL) for r in rows]
    seen, revisits = set(), 0
    for k, cell in enumerate(seq):
        if k and cell != seq[k - 1] and cell in seen:
            revisits += 1
        seen.add(cell)
    actions = [r[1] for r in rows]
    kinds = [family(a) for a in actions]
    alternations = interface_loops(actions)
    count, stalled = stalls(rows, produced)
    path = sum(max(abs(a[4] - b[4]), abs(a[5] - b[5])) for a, b in zip(rows, rows[1:]))
    first = produced[0] if produced else None
    return dict(discovery_tick=discovered, first_log_tick=first,
                discovery_to_log=None if first is None or discovered is None else first - discovered,
                cells=cells, revisits=revisits, stalls=count, stalled_ticks=stalled, path_tiles=path,
                refused=sum(bool(r[8]) for r in rows), ended=dict(Counter(r[7] for r in rows if r[7])),
                interface_actions=sum(k in INTERFACE for k in kinds), interface_alternations=alternations,
                families=dict(Counter(kinds)))


def play(task, actor, seed, trace=False, limit=2400):
    """One search trial: (decision rows, diagnostic rows or None, final info, tracker). Diagnostic rows keep each decision's
    legal actions, the five likeliest actions and the value (checkpoint actors)."""
    actor.reset(seed)
    tracker = Tracker(task)
    obs, rows, diag, p = task.projected, [], [] if trace else None, task.agent.player
    for _ in range(limit):
        tick, raw = task.world.tick, task.raw
        if trace:
            action, extra = actor.act(obs, task, trace=True)
            diag.append(dict(tick=tick - task.start_tick, legal=np.flatnonzero(obs['mask']).tolist(), **(extra or {})))
        else:
            action = actor.act(obs, task)
        kind = family(action)
        option = raw['actSel'][action - A_ACT0]['option'] if kind == 'act' and action - A_ACT0 < len(raw['actSel']) else \
            raw['heldSel'][action - A_HELD0]['option'] if kind == 'held' and action - A_HELD0 < len(raw['heldSel']) else None
        obs, reward, done, info = task.step(action)
        ended = task.raw['feedback'].get('ended') if not done else None
        tracker.observe()
        rows.append([tick - task.start_tick, action, option, round(float(reward), 6), p.x - ORIGIN[0], p.z - ORIGIN[1],
                     tracker.discovered is not None, ended and ended['reason'], bool(ended and ended['refused']),
                     task.raw.get('actionNote')])
        if done:
            return rows, diag, info, tracker
    return rows, diag, None, tracker


def produced_ticks(task):
    """Ticks (from arm) of the logs the episode produced, inside its window."""
    from sim.run import family_ids
    family = family_ids(task.world.t, 'wood')
    return sorted(tick - task.start_tick for tick, kind, data in task.world.events[task.event_start:]
                  if kind == 'produce' and data['obj'] in family and tick < task.terminal['tick'] for _ in range(data['count']))


# ---- records and evaluation -------------------------------------------------------------------------------------
def records(split, layouts, key, tables, first=0, only=None):
    """The frozen search records: record k on the k-th search layout of the split from seed index `first`, with
    draw_search(key, 0, k, ..., layout=k). Returns (layouts, records) as dicts by record index (`only` limits both)."""
    pool, scanned = search_layouts(split, layouts, tables, first)
    tiles = {k: search_tiles(pool[k], tables) for k in range(layouts)}
    wanted = range(layouts) if only is None else only
    return {k: pool[k] for k in wanted}, {k: draw_search(key, 0, k, pool, tiles, layout=k) for k in wanted}, scanned


def evaluate(output, *, checkpoint=None, initial=None, previous=None, policies=('final', 'initial', 'random', 'wait'),
             split='development', layouts=200, key=13, first=0, part=(0, 1), diagnostics=4, seed=404001, tables=None,
             sampled=False):
    """Every policy on the same frozen records (records()), each trial in a fresh world. part=(i, n): this process plays
    records i, i + n, ... (merge() joins the parts). Checkpoint policies keep full diagnostic rows for their first
    `diagnostics` records and every failed trial. Writes results-{i}.json and trials-{i}.jsonl.gz."""
    from sim.data import Tables
    from sim.engine import Unsupported
    from sim.world import LayoutTables
    from baseline.telemetry import sha256
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = tables or Tables.load()
    t0 = time.monotonic()
    mine = range(part[0], layouts, part[1])
    pool, recs, scanned = records(split, layouts, key, tables, first, only=mine)
    paths = {name: p for name, p in dict(final=checkpoint, initial=initial, previous=previous).items() if p}
    rows, timing = [], {}
    stream = gzip.open(output / f'trials-{part[0]}.jsonl.gz', 'wt')
    for policy in policies:
        actor, t1 = actor_for(policy, paths, sampled), time.monotonic()
        for k in mine:
            rec, base = recs[k], dict(policy=policy, record=k, layout=pool[k].seed)
            task = episode(rec, pool[k], LayoutTables(tables, pool[k]), tables)
            problems = audit_start(task.raw)
            if problems:
                raise RuntimeError(f'record {k} (layout {pool[k].seed}) shows a tree at arm: {problems}')
            try:
                decisions, diag, info, tracker = play(task, actor, seed + k, trace=policy in paths)
            except Unsupported as error:  # counted apart, never scored
                rows.append(dict(base, unsupported=str(error)))
                stream.write(json.dumps(dict(base, start=rec, unsupported=str(error)), default=str) + '\n')
                continue
            if info is None:
                rows.append(dict(base, decision_limit=True))
                continue
            paid = sum(r[3] for r in decisions)
            if abs(paid - info['episode']['r']) > 1e-6:
                raise RuntimeError('paid rewards disagree with the audited return')
            logs = produced_ticks(task)
            row = dict(base, success=info['episode']['r'] == 1.0, episode_return=info['episode']['r'],
                       survived=info['report']['survived'], logs=info['outcomes'], decisions=info['episode']['l'],
                       ticks=info['episode']['native_ticks'],
                       success_tick=logs[SEARCH['target'] - 1] if info['episode']['r'] == 1.0 else None,
                       **measures(decisions, logs, tracker.discovered, len(tracker.cells)))
            rows.append(row)
            keep = policy in paths and (k < diagnostics or not row['success'])
            stream.write(json.dumps(dict(base, start=rec, outcome=row, report=info['report'], decisions=decisions,
                                         diagnostic=diag if keep else None), default=str) + '\n')
        timing[policy] = time.monotonic() - t1
        print(f'[search] part {part[0]}/{part[1]}: {policy} {timing[policy]:.0f}s', flush=True)
    stream.close()
    from sim.agent import HELD_CONTRACT, MASK_CONTRACT
    from sim.engine import RETALIATE_CONTRACT
    result = dict(version=SEARCH['version'], horizon=SEARCH['horizon'], hidden=SEARCH['hidden'], split=split, layouts=layouts,
                  key=key, first=first, scanned=scanned, part=list(part), sampled=sampled, mask=MASK_CONTRACT,
                  retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT,
                  decision_fields=list(DECISION_FIELDS), diagnostics=diagnostics, seed=seed, policies=list(policies),
                  checkpoints={n: dict(path=str(p), sha256=sha256(p)) for n, p in paths.items() if n in policies},
                  layout_seeds=[pool[k].seed for k in mine], timing_seconds=timing, wall_seconds=time.monotonic() - t0,
                  trials=rows)
    (output / f'results-{part[0]}.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def summarize(trials):
    """Counts and medians of one policy's scored trials."""
    from baseline.compare import wilson
    scored = [t for t in trials if 'success' in t]
    ok = sum(t['success'] for t in scored)
    median = lambda key: (lambda v: v[len(v) // 2] if v else None)(sorted(t[key] for t in scored if t.get(key) is not None))
    return dict(trials=len(trials), scored=len(scored), unsupported=sum('unsupported' in t for t in trials),
                decision_limit=sum('decision_limit' in t for t in trials), successes=ok, interval=wilson(ok, len(scored)),
                deaths=sum(not t['survived'] for t in scored), discovered=sum(t['discovery_tick'] is not None for t in scored),
                discovered_no_log=sum(t['discovery_tick'] is not None and not t['logs'] for t in scored),
                median_discovery_tick=median('discovery_tick'), median_discovery_to_log=median('discovery_to_log'),
                median_success_tick=median('success_tick'), mean_cells=round(float(np.mean([t['cells'] for t in scored])), 2) if scored else None,
                revisits=sum(t['revisits'] for t in scored), stalls=sum(t['stalls'] for t in scored),
                refused=sum(t['refused'] for t in scored), interface_actions=sum(t['interface_actions'] for t in scored),
                interface_alternations=sum(t['interface_alternations'] for t in scored), valid=len(scored) == len(trials))


def merge(output):
    """Join an evaluation's parts into results.json with per-policy summaries; refuses missing or repeated records."""
    output = Path(output)
    parts = [json.loads(p.read_text()) for p in sorted(output.glob('results-*.json'))]
    if not parts:
        raise ValueError(f'no parts in {output}')
    first, n = parts[0], parts[0]['part'][1]
    if sorted(p['part'][0] for p in parts) != list(range(n)) or any(p['part'][1] != n for p in parts):
        raise ValueError('parts missing or of different splits')
    for key in ('version', 'horizon', 'split', 'layouts', 'key', 'first', 'policies', 'seed', 'checkpoints', 'sampled', 'mask',
                'retaliate', 'held'):  # a part with no `held` (held-v1) differs from a held-v2 part
        if any(p.get(key) != first.get(key) for p in parts):
            raise ValueError(f'parts differ in {key}')
    trials = sorted((t for p in parts for t in p['trials']), key=lambda t: (first['policies'].index(t['policy']), t['record']))
    for policy in first['policies']:
        if sorted(t['record'] for t in trials if t['policy'] == policy) != list(range(first['layouts'])):
            raise ValueError(f'{policy}: records missing or repeated')
    result = {k: v for k, v in first.items() if k not in ('part', 'trials', 'layout_seeds', 'timing_seconds', 'wall_seconds')}
    result.update(parts=n, timing_seconds={p['part'][0]: p['timing_seconds'] for p in parts},
                  cells={policy: summarize([t for t in trials if t['policy'] == policy]) for policy in first['policies']},
                  trials=trials)
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


# ---- development instruments ------------------------------------------------------------------------------------
def capacity(split, layouts, tables, first=0):
    """How many of the split's first `layouts` layouts have a search tile, and how many tiles each."""
    counts = [int(search_tiles(generate(s, tables, SEARCH_WORLD), tables).sum()) for s in split_seeds(split, layouts, first)]
    have = sorted(c for c in counts if c)
    return dict(split=split, layouts=layouts, first=first, with_search_tiles=len(have),
                tiles_quantiles=[have[int(q * (len(have) - 1))] for q in (0, .25, .5, .75, 1)] if have else None)


def twin(layout, tables, tile):
    """A copy of `layout` whose trees and oaks are all moved by one translation to tiles still beyond SEARCH['hidden'] of
    the local start `tile`, their centre at least 12 tiles from where it was, every generator check passing and the tile
    still a search tile; the first such translation in a fixed order (largest shift first). None if there is none."""
    rows, (W, H) = tree_locs(layout, tables), layout.flags.shape
    occupied = np.zeros((W, H), bool)
    for k, (ty, x, z, *_) in enumerate(layout.locs):
        if k not in rows:
            lt = tables.loc_types[int(ty)]
            occupied[max(x - 1, 0):x + lt['width'] + 1, max(z - 1, 0):z + lt['length'] + 1] = True
    old = np.asarray([layout.locs[k, 1:3] for k in rows])
    shifts = sorted(((dx, dz) for dx in range(-40, 41) for dz in range(-40, 41)), key=lambda s: (-max(map(abs, s)), s))
    for dx, dz in shifts:
        new = old + (dx, dz)
        if max(abs(dx), abs(dz)) < 12 or (new < 1).any() or (new[:, 0] > W - 2).any() or (new[:, 1] > H - 2).any():
            continue
        if (np.maximum(abs(new[:, 0] - tile[0]), abs(new[:, 1] - tile[1])) <= SEARCH['hidden']).any():
            continue
        free = True
        for j, (x, z) in zip(rows, new):
            lt = tables.loc_types[int(layout.locs[j, 0])]
            for a in range(x, x + lt['width']):
                for b in range(z, z + lt['length']):
                    free = free and a < W - 1 and b < H - 1 and not occupied[a, b] and \
                        all(f[a, b] == 0 for f in (layout.flags, layout.flags_open, layout.flags_snapshot))
        if not free:
            continue
        moved = relocate(layout, tables, {k: tuple(int(v) for v in xz) for k, xz in zip(rows, new)})
        if moved is not None and search_tiles(moved, tables)[tile]:
            return moved
    return None


def leakage(output, *, split, pairs, tables, initial=None, key=13, first=0, seed=404001):
    """For the split's first search layouts, each with its record's start and a twin (twin()): whether the first projected
    observations are identical in every array, and how the oracle and (with `initial`) the greedy policy fare in each
    world. A policy that sees nothing different acts the same until the observations part."""
    from obs import project
    from sim.world import LayoutTables
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, pool = [], {}
    for k, layout in enumerate(iter_search_layouts(split, tables, first)):
        if len(rows) == pairs:
            break
        pool[k] = layout
        rec = draw_search(key, 0, k, pool, {k: search_tiles(layout, tables)}, layout=k)
        tile = (rec['tile'][0] - ORIGIN[0], rec['tile'][1] - ORIGIN[1])
        other = twin(layout, tables, tile)
        if other is None:
            continue
        tasks = [episode(rec, lay, LayoutTables(tables, lay), tables) for lay in (layout, other)]
        a, b = (project(t.raw) for t in tasks)
        differing = sorted(name for name in a if not np.array_equal(a[name], b[name]))
        row = dict(record=k, layout=layout.seed, tile=tile, audit=[audit_start(t.raw) for t in tasks],
                   trees=[[list(map(int, lay.locs[j, 1:3])) for j in tree_locs(lay, tables)] for lay in (layout, other)],
                   first_observation_differs=differing)
        for name, actor in (('oracle', OracleActor()), *((('initial', actor_for('initial', dict(initial=initial))),) if initial else ())):
            runs = []
            for lay in (layout, other):
                task = episode(rec, lay, LayoutTables(tables, lay), tables)
                decisions, _, info, tracker = play(task, actor, seed + k)
                runs.append(dict(success=bool(info and info['episode']['r'] == 1.0), discovery_tick=tracker.discovered,
                                 actions=[r[1] for r in decisions], ticks=[r[0] for r in decisions]))
            same = 0
            while same < min(len(runs[0]['actions']), len(runs[1]['actions'])) and runs[0]['actions'][same] == runs[1]['actions'][same]:
                same += 1
            row[name] = dict(success=[r['success'] for r in runs], discovery_tick=[r['discovery_tick'] for r in runs],
                             same_actions_before_divergence=same,
                             divergence_tick=runs[0]['ticks'][same] if same < len(runs[0]['ticks']) else None)
        rows.append(row)
        print(f"[search] pair {len(rows)}: layout {layout.seed} differs in {differing or 'nothing'}", flush=True)
    result = dict(split=split, pairs=pairs, key=key, first=first, identical=sum(not r['first_observation_differs'] for r in rows),
                  rows=rows)
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def reference(output, *, split, layouts, tables, key=13, first=0, horizon=None, policies=('oracle', 'explorer', 'wait', 'random')):
    """The references and controls on the split's first search records (a measurement of the horizon; never labels), and the
    oracle on the same layouts' goal-activity-v2 wood starts (the visible-resource control the instrument must pass)."""
    from baseline import config
    from sim.engine import World
    from sim.spec import HORIZON_TICKS
    from sim.task import WORLD_START_TICK, GoalEpisode, draw_start
    from sim.world import LayoutTables
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exclusions = ()
    if horizon is not None:  # a measurement at another horizon: past tick 500 it declares sustained combat's exclusion
        SEARCH['horizon'], exclusions = horizon, ('random_events',) if horizon > 480 else ()
    pool, recs, scanned = records(split, layouts, key, tables, first)
    lts = {k: LayoutTables(tables, pool[k]) for k in pool}
    visible = []
    for k in pool:
        start = draw_start(key, 0, k, pool, layout=k)
        task = GoalEpisode(World(lts[k], seed=start['world_seed'], start_tick=WORLD_START_TICK), config.TASK_CFG, 'wood',
                           horizon=HORIZON_TICKS, scale=1.0)
        task.start(start['tile'], start['weapon'], start['levels'])
        _, _, info, _ = play(task, OracleActor(), 0)
        visible.append(bool(info and info['episode']['r'] == 1.0))
    rows = []
    for policy in policies:
        actor = actor_for(policy, {})
        for k in pool:
            task = episode(recs[k], pool[k], lts[k], tables, exclusions=exclusions)
            decisions, _, info, tracker = play(task, actor, 7 + k)
            logs = produced_ticks(task)
            rows.append(dict(policy=policy, record=k, layout=pool[k].seed, success=info['episode']['r'] == 1.0,
                             nearest_tree_steps=int(pool[k].goal_dist[0][recs[k]['tile'][0] - ORIGIN[0], recs[k]['tile'][1] - ORIGIN[1]]),
                             success_tick=logs[SEARCH['target'] - 1] if len(logs) >= SEARCH['target'] else None,
                             **measures(decisions, logs, tracker.discovered, len(tracker.cells))))
    summary = {}
    for policy in policies:
        mine = [r for r in rows if r['policy'] == policy]
        ticks = sorted(r['success_tick'] for r in mine if r['success_tick'] is not None)
        summary[policy] = dict(successes=sum(r['success'] for r in mine), trials=len(mine),
                               two_logs_by=dict(zip(('median', 'p90', 'max'), (ticks[len(ticks) // 2], ticks[int(.9 * (len(ticks) - 1))],
                                                                             ticks[-1]))) if ticks else None,
                               discovered=sum(r['discovery_tick'] is not None for r in mine))
    result = dict(split=split, layouts=layouts, scanned=scanned, key=key, first=first, horizon=SEARCH['horizon'],
                  visible_control=dict(oracle_successes=sum(visible), trials=len(visible)), summary=summary, trials=rows)
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def main():
    from sim.data import Tables
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capacity', help='search tiles in the existing generator\'s layouts')
    pool = sub.add_parser('pool', help='save the split\'s first search layouts (no derived tables)')
    spool = sub.add_parser('supply-pool', help='save the split\'s first supply layouts (no derived tables)')
    sref = sub.add_parser('supply-reference', help='development references on paired supply records (scarce and ample)')
    sev = sub.add_parser('evaluate-supply', help='policies on both variants of frozen supply records (one part)')
    smg = sub.add_parser('merge-supply', help='join a supply evaluation\'s parts')
    srp = sub.add_parser('replay-supply', help='a watched replay of one supply trial (trace.json and replay.html)')
    srp.add_argument('--policy', type=Path, help='a checkpoint (else --control)')
    srp.add_argument('--control', choices=('oracle', 'stay', 'explorer', 'wait', 'random'))
    srp.add_argument('--split', choices=SPLITS, default='development')
    srp.add_argument('--first', type=int, default=0)
    srp.add_argument('--layouts', type=int, default=200)
    srp.add_argument('--key', type=int, default=13)
    srp.add_argument('--record', type=int, required=True)
    srp.add_argument('--variant', choices=('ample', 'scarce'), required=True)
    srp.add_argument('--sampled', action='store_true')
    srp.add_argument('--seed', type=int, default=505001, help='the pass\'s trial seed base (the trial uses base + record)')
    srp.add_argument('--label', default='')
    srp.add_argument('--tables', type=Path)
    srp.add_argument('--output', type=Path, required=True)
    ref = sub.add_parser('reference', help='references and controls on development search records')
    leak = sub.add_parser('leakage', help='twin worlds with different hidden trees: whether their first observations agree')
    ev = sub.add_parser('evaluate', help='policies on frozen search records (one part)')
    mg = sub.add_parser('merge', help='join an evaluation\'s parts')
    for p in (cap, pool, ref, leak, ev, spool, sref):
        p.add_argument('--split', choices=SPLITS, default='development')
        p.add_argument('--first', type=int, default=0, help='the split\'s seed index to start from')
    spool.add_argument('--count', type=int, required=True)
    sref.add_argument('--layouts', type=int, default=100)
    sref.add_argument('--horizon', type=int, default=600, help='the measurement window (longer than the task\'s)')
    sref.add_argument('--part', type=int, nargs=2, default=(0, 1))
    sref.add_argument('--clicks', choices=('act-v3', 'wasd'), default='act-v3',
                      help="act-v3's 24 clicks, or the WASD-only feasibility variant's four one-tile clicks")
    sref.add_argument('--policies', nargs='+', choices=('oracle', 'stay', 'explorer', 'wait'),
                      default=['oracle', 'stay', 'explorer', 'wait'])
    for p in (cap, ref, ev):
        p.add_argument('--layouts', type=int, default=200)
    for p in (ref, leak, ev):
        p.add_argument('--key', type=int, default=13)
    pool.add_argument('--count', type=int, required=True)
    for p in (pool, ref, leak, ev, mg, spool, sref, sev, smg):
        p.add_argument('--output', type=Path, required=True)
    sev.add_argument('--split', choices=SPLITS, default='development')
    sev.add_argument('--first', type=int, default=0)
    sev.add_argument('--layouts', type=int, default=200)
    sev.add_argument('--key', type=int, default=13)
    sev.add_argument('--checkpoint', type=Path)
    sev.add_argument('--initial', type=Path)
    sev.add_argument('--previous', type=Path)
    sev.add_argument('--policies', nargs='+', default=['final', 'initial'],
                     choices=('final', 'initial', 'previous', 'random', 'wait', 'oracle', 'stay', 'explorer'))
    sev.add_argument('--part', type=int, nargs=2, default=(0, 1))
    sev.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
    sev.add_argument('--sampled', action='store_true')
    sev.add_argument('--seed', type=int, default=505001, help='trial seed base (a sampled pass declares its own)')
    sev.add_argument('--tables', type=Path, help='the simulator archive as given, away from the checkout (its pins unchecked)')
    ref.add_argument('--horizon', type=int, help='a measurement at another horizon, not the frozen one')
    leak.add_argument('--pairs', type=int, default=20)
    leak.add_argument('--initial', type=Path)
    ev.add_argument('--checkpoint', type=Path)
    ev.add_argument('--initial', type=Path)
    ev.add_argument('--previous', type=Path)
    ev.add_argument('--policies', nargs='+', default=['final', 'initial', 'random', 'wait'],
                    choices=('final', 'initial', 'previous', 'random', 'wait', 'oracle', 'explorer'))
    ev.add_argument('--part', type=int, nargs=2, default=(0, 1))
    ev.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
    ev.add_argument('--sampled', action='store_true', help='sample the checkpoints\' actions')
    ev.add_argument('--seed', type=int, default=404001, help='the sampling-seed base: record k samples under seed + k')
    for p in (cap, pool, ref, leak, ev, spool, sref):
        p.add_argument('--tables', type=Path, help='the simulator archive as given, away from the checkout (its pins unchecked)')
    args = parser.parse_args()
    if args.command == 'merge':
        result = merge(args.output)
        print(json.dumps(result['cells'], indent=1))
        return
    if args.command == 'replay-supply':
        tables = Tables.load(args.tables, check_pins=False) if args.tables else Tables.load()
        pool, recs, _, _ = supply_records(args.split, args.layouts, args.key, tables, args.first, only=[args.record])
        record = dict(recs[args.record], record=args.record, variant=args.variant)
        if args.policy:
            from sim.combat import PolicyActor
            actor = PolicyActor(str(args.policy), deterministic=not args.sampled)
        else:
            grove = groves(pool[args.record], tables)[record['grove']]
            positions = [(int(pool[args.record].locs[r, 1]), int(pool[args.record].locs[r, 2])) for r in grove]
            actor = StayActor(positions) if args.control == 'stay' else actor_for(args.control, {})
        print(json.dumps(replay_supply(args.output, record, pool[args.record], tables, actor, args.seed + args.record,
                                       args.label)))
        return
    if args.command == 'merge-supply':
        result = merge_supply(args.output)
        print(json.dumps(result['cells'], indent=1))
        return
    if args.command == 'evaluate-supply':
        if (args.split == 'final') != args.final:
            parser.error('the final split needs --final, and --final needs the final split')
        tables = Tables.load(args.tables, check_pins=False) if args.tables else Tables.load()
        result = evaluate_supply(args.output, checkpoint=args.checkpoint, initial=args.initial, previous=args.previous,
                                 policies=tuple(args.policies), split=args.split, layouts=args.layouts, key=args.key,
                                 first=args.first, part=tuple(args.part), tables=tables, sampled=args.sampled, seed=args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=1, default=str))
        return
    if args.command in ('evaluate',) and (args.split == 'final') != args.final:
        parser.error('the final split needs --final, and --final needs the final split')
    if args.command in ('capacity', 'pool', 'reference', 'leakage', 'supply-pool', 'supply-reference') and args.split == 'final':
        parser.error('development instruments never read the final split')
    tables = Tables.load(args.tables, check_pins=False) if args.tables else Tables.load()
    if args.command == 'capacity':
        print(json.dumps(capacity(args.split, args.layouts, tables, args.first), indent=1))
    elif args.command == 'pool':
        from sim.world import save_pool
        layouts, scanned = search_layouts(args.split, args.count, tables, args.first)
        print(f'[search] wrote {save_pool(args.output, layouts)}: {len(layouts)} {args.split} search layouts of {scanned} scanned')
    elif args.command == 'supply-pool':
        from sim.world import save_pool
        layouts, scanned = supply_layouts(args.split, args.count, tables, args.first)
        print(f'[supply] wrote {save_pool(args.output, layouts, version=SUPPLY["world"])}: {len(layouts)} {args.split} supply '
              f'layouts of {scanned} scanned')
    elif args.command == 'supply-reference':
        rows = supply_reference(args.output, split=args.split, layouts=args.layouts, tables=tables, first=args.first,
                                horizon=args.horizon, part=tuple(args.part), clicks=args.clicks,
                                policies=tuple(args.policies))
        print(json.dumps(dict(trials=len(rows), unsupported=sum('unsupported' in r for r in rows))))
    elif args.command == 'reference':
        result = reference(args.output, split=args.split, layouts=args.layouts, tables=tables, key=args.key, first=args.first,
                           horizon=args.horizon)
        print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=1))
    elif args.command == 'leakage':
        result = leakage(args.output, split=args.split, pairs=args.pairs, tables=tables, initial=args.initial, key=args.key,
                         first=args.first)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1))
    else:
        result = evaluate(args.output, checkpoint=args.checkpoint, initial=args.initial, previous=args.previous,
                          policies=tuple(args.policies), split=args.split, layouts=args.layouts, key=args.key, first=args.first,
                          part=tuple(args.part), tables=tables, sampled=args.sampled, seed=args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=1, default=str))


if __name__ == '__main__':
    main()
