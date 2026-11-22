"""The changed worlds of the evaluation scorecard (read 2, the changed-world pairs, and read 3, breadth; adopted as
descriptions on 15 November 2026): a village-world-v2 record played twice, once
as generated and once with one fact changed. A description only: no claim, no gate, nothing in the reward, and no
training world changes. No ORACLE file changes either: a change edits a generated record, and the life is built from
the edited record here.

A record (Record) is what a free-play life of sim/free.py Life is built from: the generated layout (sim/world.py), the
draw_village start (sim/combat.py) and the world's npc spawns (sim/combat.py village_npcs, whose nids are positions in
that list). A change (CHANGES) is a pure function of the base record. Its start, kit and world seed are the base
record's, and the npcs that remain keep their base nids, so their keyed draws (sim/engine.py World.rng on (purpose,
tick, nid)) are the base record's: a pair shares its random numbers until the lives part. The layout is edited
consistently and checked again by the generator's own checks (sim/world.py _checked, with no count of start tiles):
a removed loc's collision leaves the closed, open and snapshot maps, the walkable tiles, reach and goal distances are
computed again, the gate's rows are renumbered, and the start tiles stay the base record's (outside() floods from
them). A layout alone cannot hold a changed spawn list (village-world-v2 spawns a goblin at every chicken spawn, and
nids are positions), so `life` builds the world from the record's spawns by sim/combat.py village_life's steps.

- no-highwayman: the highwayman is not there (its spawn and its layout row).
- no-chickens: the chickens are not there; the pen and the goblins at the chicken spawns stay.
- open-pen: the pen's gate or door stands open in the map itself (the open stages that its own script puts there, in
  place of the closed leaves). It stays open: the open stages close with their own scripts and come back after 500
  ticks, as an opened barrier does in the base world.
- only-wood, only-mine, only-fish, only-fight: the sources of the other three kinds and the highwayman are not there
  (trees: tree, oak, willow, yew; rocks: copper, tin, iron; the fishing spots; the chickens and the goblins at their
  spawns). The field, the obstacles, the fences and the pen stay.
- no-food: the start's kit holds no food. The field's plants stay (the cabbage can be eaten).

`diff` lists what differs between two records; `check` refuses a changed record that differs from its base in more
than its named fact. One record's diff, printed:

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.changed diff --change no-chickens --record 3 --first 7000 --key 239
"""
import argparse
import copy
from collections import Counter
from dataclasses import dataclass, replace
import json

import numpy as np

from sim import paths, world as W
from sim.free import Life, SPLIT

WORLD = 'village-world-v2'
WOOD = (W.TREE, W.OAK, W.WILLOW, W.YEW)
MINE = tuple(name for names in W.ROCKS.values() for name in names)
FISH = (W.SPOT,)
FIGHT = (*W.CHICKENS, 'goblin')  # the goblins spawn at the chicken spawns (sim/combat.py combat_npcs)
KINDS = dict(wood=dict(locs=WOOD, npcs=()), mine=dict(locs=MINE, npcs=()), fish=dict(locs=(), npcs=FISH),
             fight=dict(locs=(), npcs=FIGHT))
SCRIPTS = ('wood', 'fish', 'mine', 'melee', 'chickens', 'rotation')  # the scripts a changed world can play


@dataclass(frozen=True)
class Record:
    layout: W.Layout
    start: dict
    spawns: tuple  # World's npcs: dicts of nid, type, x, z, level, size


@dataclass(frozen=True)
class Change:
    name: str
    locs: tuple = ()        # loc debugnames whose rows go
    npcs: tuple = ()        # npc debugnames whose spawns (and layout rows) go
    pen: bool = False       # the pen's barrier stands open
    food: bool = False      # the kit holds no food
    keeps: str = None       # only-X: the kind whose sources stay
    scripts: tuple = SCRIPTS  # the candidates for the changed world's best script (sim/free.py controls)


def _only(kind):
    others = [k for k in KINDS if k != kind]
    return Change(f'only-{kind}', locs=tuple(n for k in others for n in KINDS[k]['locs']),
                  npcs=tuple(n for k in others for n in KINDS[k]['npcs']) + (W.HUNTER,), keeps=kind,
                  scripts=('melee', 'chickens') if kind == 'fight' else (kind,))


CHANGES = {c.name: c for c in (
    Change('no-highwayman', npcs=(W.HUNTER,)),
    Change('no-chickens', npcs=W.CHICKENS, scripts=tuple(s for s in SCRIPTS if s != 'chickens')),
    Change('open-pen', pen=True),
    *(_only(kind) for kind in KINDS),
    Change('no-food', food=True),
)}


def _ids(types, names):
    return {k for k, v in types.items() if v['debugname'] in names}


def _rows(a):
    return [tuple(int(v) for v in row) for row in a]


def base_record(record, first, key, tables, split=None, muggers=False, world=WORLD):
    """Record `record` of a record set as sim/free.py Life builds it: the split's layout first + record of the world's
    declared layouts, the draw_village start of (key, world slot 0, episode record) and the world's spawns."""
    from sim.combat import VILLAGES, draw_village, village_npcs
    if world != WORLD:
        raise ValueError(f'the changed worlds are {WORLD}\'s, not {world}\'s')
    layout = W.generate(W.split_seeds(split or SPLIT, 1, first + record)[0], tables, W.WORLDS[VILLAGES[world]['layouts']])
    start = draw_village(key, 0, record, [layout], tables, muggers, layout=0)
    return Record(layout, start, tuple(village_npcs(layout, tables, start['village']['muggers'])))


def applies(change, rec, tables):
    """Whether the base record holds the change's fact (an only-X world: whether it holds X's sources)."""
    c = CHANGES[change]
    if c.pen:
        return bool((rec.layout.gate >= 0).any())
    if c.food:
        return bool(rec.start['village']['food'])
    locs, npcs = (KINDS[c.keeps]['locs'], KINDS[c.keeps]['npcs']) if c.keeps else (c.locs, c.npcs)
    loc_ids, npc_ids = _ids(tables.loc_types, locs), _ids(tables.npc_types, npcs)
    return any(int(r[0]) in loc_ids for r in rec.layout.locs) or any(s['type'] in npc_ids for s in rec.spawns)


def _relaid(layout, tables, flags, locs, gate, open_locs, npcs):
    """The edited layout through the generator's checks, the base start tiles kept."""
    out = W._checked(layout.seed, layout.attempt, *flags, locs, gate, open_locs, npcs, _rows(layout.fish), W._Types(tables),
                     tables, dict(W.world_of(layout), min_starts=0))
    if out is None:
        raise ValueError(f'layout {layout.seed}: the changed layout fails a generator check')
    return replace(out, start=layout.start.copy())


def edit(change, rec, tables):
    """The changed record of `rec` (module docstring); `rec` is left as it was."""
    c = CHANGES[change]
    layout = rec.layout
    if c.food:
        start = copy.deepcopy(rec.start)
        start['village']['food'] = []
        return Record(layout, start, tuple(dict(s) for s in rec.spawns))
    locs, gate = _rows(layout.locs), [int(k) for k in layout.gate if k >= 0]
    open_locs, npcs = _rows(layout.open_locs), _rows(layout.npcs)
    flags = [layout.flags.copy(), layout.flags_open.copy(), layout.flags_snapshot.copy()]
    if c.pen:
        if len(gate) != len(open_locs):
            raise ValueError(f'layout {layout.seed}: {len(gate)} leaves, {len(open_locs)} open stages')
        for k, row in zip(gate, open_locs):  # each leaf's row holds its open stage: no other row moves
            locs[k] = row
        flags[0] = layout.flags_open.copy()
        gate, open_locs = [], []
    loc_ids, npc_ids = _ids(tables.loc_types, c.locs), _ids(tables.npc_types, c.npcs)
    removed = {k for k, row in enumerate(locs) if row[0] in loc_ids}
    if removed & set(gate):
        raise ValueError('a pen barrier is never removed')
    for k in sorted(removed):
        type_id, x, z, shape, angle = locs[k]
        for state in flags:
            for a, b, mask in paths.collision_masks(x, z, shape, angle, tables.loc_types[type_id]):
                state[a, b] &= ~mask
    gate = [k - sum(r < k for r in removed) for k in gate]
    locs = [row for k, row in enumerate(locs) if k not in removed]
    npcs = [row for row in npcs if row[1] not in npc_ids]
    spawns = tuple(dict(s) for s in rec.spawns if s['type'] not in npc_ids)
    return Record(_relaid(layout, tables, flags, locs, gate, open_locs, npcs), copy.deepcopy(rec.start), spawns)


def records(change, first, count, key, tables, split=None, world=WORLD):
    """The records among 0 .. count - 1 of a record set whose base holds the change's fact."""
    return [r for r in range(count) if applies(change, base_record(r, first, key, tables, split, world=world), tables)]


# ---- what differs ------------------------------------------------------------------------------------------------
ROWS = ('locs', 'open_locs', 'npcs', 'fish')
MAPS = ('flags', 'flags_open', 'flags_snapshot', 'walkable', 'start')


def _start_paths(a, b, at=''):
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return [p for k in sorted(a) for p in _start_paths(a[k], b[k], f'{at}.{k}' if at else k)]
    return [] if a == b else [at]


def diff(base, changed, tables):
    """What differs between two records: per row array (ROWS) the rows removed and added, and whether the rows both
    hold keep their order; per map (MAPS) the tiles that differ, and for the collision maps the bits added; per goal the
    tiles whose distance differs; the gate's leaves by their rows; the seed and attempt; the spawns removed and added
    (nid, debugname, x, z, the rest of the dict) and whether the spawns both hold are equal in order; the start
    record's fields that differ (start_record)."""
    a, b, out = base.layout, changed.layout, {}
    for name in ROWS:
        ra, rb = _rows(getattr(a, name)), _rows(getattr(b, name))
        if ra != rb:
            gone, new = Counter(ra) - Counter(rb), Counter(rb) - Counter(ra)
            kept = lambda rows, other: [r for r in rows if r not in other]
            out[name] = dict(removed=sorted(gone.elements()), added=sorted(new.elements()),
                             order_kept=kept(ra, gone) == kept(rb, new))
    for name in MAPS:
        ma, mb = getattr(a, name), getattr(b, name)
        if not np.array_equal(ma, mb):
            out[name] = dict(tiles=_rows(np.argwhere(ma != mb)))
            if name.startswith('flags'):
                out[name]['bits_added'] = int(np.count_nonzero(mb & ~ma))
    goals = [k for k in range(len(a.goal_dist)) if not np.array_equal(a.goal_dist[k], b.goal_dist[k])]
    if goals:
        out['goal_dist'] = {k: int(np.count_nonzero(a.goal_dist[k] != b.goal_dist[k])) for k in goals}
    leaves = lambda layout: [_rows(layout.locs)[int(k)] for k in layout.gate if k >= 0]
    if leaves(a) != leaves(b):
        out['gate'] = dict(base=leaves(a), changed=leaves(b))
    if (a.seed, a.attempt, type(a)) != (b.seed, b.attempt, type(b)):
        out['identity'] = dict(base=[a.seed, a.attempt, type(a).__name__], changed=[b.seed, b.attempt, type(b).__name__])
    spawn = lambda s: (s['nid'], tables.npc_types[s['type']]['debugname'], s['x'], s['z'],
                       tuple(sorted((k, v) for k, v in s.items() if k not in ('nid', 'type', 'x', 'z'))))
    sa, sb = [spawn(s) for s in base.spawns], [spawn(s) for s in changed.spawns]
    if sa != sb:
        out['spawns'] = dict(removed=[s for s in sa if s not in sb], added=[s for s in sb if s not in sa],
                             kept_equal=[s for s in sa if s in sb] == [s for s in sb if s in sa])
    fields = _start_paths(base.start, changed.start)
    if fields:
        out['start_record'] = {f: [_at(base.start, f), _at(changed.start, f)] for f in fields}
    return out


def _at(d, path):
    for k in path.split('.') if path else ():
        d = d[k]
    return d


def check(change, base, changed, tables):
    """The changed record's diff against its change: the rows and spawns it removes are those of the named types (all
    of them), open-pen's leaves hold the base's open stages and its closed map is the base's open map, no-food empties
    the food and nothing else, a removed loc's collision leaves only its own tiles and no map gains a bit, the walkable
    tiles only grow where locs go, and the kept spawns keep their nids. Returns the diff; raises ValueError naming
    what lies outside the fact."""
    c, d = CHANGES[change], diff(base, changed, tables)
    a, b = base.layout, changed.layout
    wrong = []

    def expect(key, ok, why):
        if not ok:
            wrong.append(f'{key}: {why}')
    loc_ids, npc_ids = _ids(tables.loc_types, c.locs), _ids(tables.npc_types, c.npcs)
    want_locs = sorted(r for r in _rows(a.locs) if r[0] in loc_ids)
    leaves = [_rows(a.locs)[int(k)] for k in a.gate if k >= 0]
    if c.pen:
        want = dict(removed=sorted(leaves), added=sorted(_rows(a.open_locs)), order_kept=True)
        expect('locs', d.get('locs') == want, f"{d.get('locs')} is not the leaves replaced by their open stages")
        expect('open_locs', d.get('open_locs', {}).get('removed') == sorted(_rows(a.open_locs)) and not _rows(b.open_locs),
               'the open stages are not the map\'s own')
        expect('flags', np.array_equal(b.flags, a.flags_open), 'the closed map is not the base\'s open map')
        expect('gate', not leaves or d.get('gate') == dict(base=leaves, changed=[]), 'the pen still has a barrier')
    else:
        expect('locs', d.get('locs', dict(removed=[], added=[], order_kept=True)) ==
               dict(removed=want_locs, added=[], order_kept=True), 'not the rows of the named loc types alone')
        expect('open_locs', 'open_locs' not in d, 'changed')
        expect('gate', 'gate' not in d, 'the pen\'s leaves changed')
    removed_tiles = {(x, z) for type_id, lx, lz, shape, angle in want_locs
                     for x, z, _ in paths.collision_masks(lx, lz, shape, angle, tables.loc_types[type_id])}
    for name in ('flags', 'flags_open', 'flags_snapshot'):
        if name in d and not (c.pen and name == 'flags'):
            expect(name, d[name]['bits_added'] == 0 and set(map(tuple, d[name]['tiles'])) <= removed_tiles,
                   'a bit added, or a tile no removed loc covers')
    expect('walkable', 'walkable' not in d or (bool(want_locs) and not (a.walkable & ~b.walkable).any()),
           'walkable tiles changed where no loc went, or some were lost')
    expect('start', 'start' not in d, 'the start tiles changed')
    expect('fish', 'fish' not in d, 'the fishing spots\' tiles changed')
    expect('identity', 'identity' not in d, 'another seed, attempt or class')
    expect('goal_dist', 'goal_dist' not in d or bool(want_locs or c.pen or set(c.npcs) & set(W.CHICKENS)),
           'the goal distances changed with nothing they read')
    want_rows = sorted(r for r in _rows(a.npcs) if r[1] in npc_ids)
    expect('npcs', d.get('npcs', dict(removed=[], added=[], order_kept=True)) ==
           dict(removed=want_rows, added=[], order_kept=True), 'not the layout rows of the named npc types alone')
    gone = [s for s in base.spawns if s['type'] in npc_ids]
    expect('spawns', [s['nid'] for s in changed.spawns] == [s['nid'] for s in base.spawns if s not in gone]
           and all(s in base.spawns for s in changed.spawns) and len(d.get('spawns', {}).get('removed', [])) == len(gone),
           'not the spawns of the named npc types alone, or a kept spawn changed')
    want_start = {'village.food': [base.start['village']['food'], []]} if c.food else None
    expect('start_record', d.get('start_record') == want_start, f"{d.get('start_record')}")
    if not (want_locs or want_rows or gone or c.pen or c.food):
        wrong.append('nothing changed: the record does not hold the fact')
    if wrong:
        raise ValueError(f'{change} on layout {a.seed}: ' + '; '.join(wrong))
    return d


# ---- the life ------------------------------------------------------------------------------------------------------
def task(rec, tables, xp_rate, ticks):
    """A started free-play life of a record: sim/combat.py village_life's steps, with the record's spawns as the
    world's npcs (village_life spawns village_npcs of the layout)."""
    from sim.combat import check_village, kit, village_of
    from sim.engine import World
    from sim.search import SEARCH
    from sim.task import WORLD_START_TICK, FreePlayEpisode
    layout, start = rec.layout, rec.start
    lt = W.LayoutTables(tables, layout)
    lt.CACHE = SEARCH['cache']  # as sim/free.py Life sets it (a memo: no output changes)
    v, village = start['village'], village_of(layout)
    check_village(village, tables, v['placed'])
    world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK, npcs=[dict(s) for s in rec.spawns],
                  exclusions=village['exclusions'], xp_rate=xp_rate, loot_outside=village.get('loot_outside'))
    life = FreePlayEpisode(world, ticks, xp_rate)
    life.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], v['food']), hp=v['hp'],
               wield=v['worn'])
    return life


class ChangedLife(Life):
    """sim/free.py Life of record `record` with change `change` (None: the base record, built the same way): its
    `row` and `hunter` as Life's, with `change`, `base_record` and `spawns` beside."""

    def __init__(self, change, first, key, record, muggers, xp_rate, ticks, tables, split=None, world=WORLD):
        if muggers is not False:
            raise ValueError(f'{WORLD} holds no muggers: pass False')
        self.key, self.record, self.muggers, self.xp_rate, self.ticks, self.world = key, record, muggers, xp_rate, ticks, world
        self.change, self.base_record = change, base_record(record, first, key, tables, split, muggers, world)
        rec = self.base_record if change is None else edit(change, self.base_record, tables)
        self.layout, self.start, self.spawns = rec.layout, rec.start, rec.spawns
        self.task = task(rec, tables, xp_rate, ticks)
        p = self.task.agent.player
        self.base, self.xp = list(p.base_levels), list(p.stats)


def life(change, first, key, record, muggers, xp_rate, ticks, tables, split=None, world=WORLD):
    """The life sim/hour.py plays for a changed world: Life's arguments with the change's name first."""
    return ChangedLife(change, first, key, record, muggers, xp_rate, ticks, tables, split, world)


def main():
    from sim.free import _tables
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('diff', help="one record's changed record against its base, checked")
    one.add_argument('--change', required=True, choices=sorted(CHANGES))
    one.add_argument('--record', type=int, required=True)
    one.add_argument('--first', type=int, required=True)
    one.add_argument('--key', type=int, required=True)
    one.add_argument('--split', default=SPLIT)
    args = parser.parse_args()
    tables = _tables(WORLD)
    base = base_record(args.record, args.first, args.key, tables, args.split)
    if not applies(args.change, base, tables):
        raise SystemExit(f'record {args.record} does not hold the fact of {args.change}')
    d = check(args.change, base, edit(args.change, base, tables), tables)
    for name in MAPS:
        if name in d:
            d[name]['tiles'] = len(d[name]['tiles'])
    print(json.dumps(dict(change=args.change, record=args.record, layout=base.layout.seed, diff=d), default=str, indent=1))


if __name__ == '__main__':
    main()
