"""The native task draws and ledgers (sim/native/task.c) against the oracle. For many (key, slot, episode)
and every task version and goal set, the C start draws equal CpuBatch._episode's draw dicts field by field
(task.draw_start, combat.draw_combat with its kit and npcs, search.draw_supply with the variant record its episode
plays), and their errors are the oracle's. The ledgers equal the oracle's functions on the same events: info made and
kills step by step (task.produced, audit.kills), and at the terminal audit.audit, combat.fights' summary and
CpuBatch._facts, on Hypothesis event streams and on the events of short CpuBatch runs (read from world.events).

The pack here is a test fixture holding what the draws and ledgers read, built from the pools and the archive's type
metadata with the oracle's own functions as rs.h defines each field (the exporter writes the full pack); it
is extracted private data, written under the ignored logs/native/ and removed afterwards."""
from collections import Counter
from dataclasses import fields as dataclass_fields
import json
from pathlib import Path
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from baseline.native_value import GATHERING_SCRIPTS
from baseline.resource_task import GOAL_FAMILIES, GOALS
from sim import native
from sim.audit import AuditError, audit, kills
from sim.build import DATA
from sim.combat import COMBAT, GROUPS, camp, combat_npcs, draw_combat, fights, kit, outside, task_spec
from sim.engine import Rng
from sim.search import SUPPLY, draw_supply, groves, is_normal, scarce_rows, supply_sites
from sim.spec import START_KIT, goal_set, slot_goal
from sim.task import draw_start, object_families, produced
from sim.world import ORIGIN, PADDED, Layout

ROOT = Path(__file__).resolve().parents[2]
COMPACT_POOL = ROOT / 'logs/sim/world/train-8-nc1-derived.npz'  # never the 64-layout pool: its derived tables are huge
SUPPLY_POOL = ROOT / 'logs/supply/pools/train-128.npz'
PACKS = ROOT / 'logs/native/tests'
TASKS = {'goal-activity-v2': 2, 'goal-activity-v3': 3, 'goal-activity-v5': 5}
POOL_COMPACT, POOL_SUPPLY = 0, 1  # rs.h RS_LAYOUT_*
START_KINDS = dict(short=0, combat=1, supply=2)
AGENT = 'wd1a0'
# The produce sites' scripts (sim/engine.py produce calls; rs.h RsConsts s_produce_*), all GATHERING_SCRIPTS members.
PRODUCE_SCRIPTS = ('[label,get_logs]', '[label,get_ore_normal]', '[label,get_ore_fast]', '[proc,fish_roll]')
NOT_GATHERING = '[label,not_gathering]'
INT64 = st.integers(-2 ** 63, 2 ** 63 - 1)
INDEX = st.integers(0, 2 ** 31 - 1)


# ---- inputs: the archive's types (meta only: no collision arrays) and the pools' layouts (no derived tables) --------
@pytest.fixture(scope='module')
def types():
    if not DATA.exists():
        pytest.skip('simulator tables logs/sim/data/draynor-nc1.npz absent (docs/RUNNING.md says how to build them)')
    from sim.data import Tables
    with np.load(DATA) as data:
        meta = json.loads(data['meta'].tobytes())
    return Tables(meta, {})  # what LayoutTables reads of its base (sim/world.py SHARED)


def lean_pool(path):
    """sim.world.load_pool's layouts without the derived arrays."""
    if not path.exists():
        pytest.skip(f'pool {path.name} absent (docs/RUNNING.md)')
    names = [f.name for f in dataclass_fields(Layout)]
    with np.load(path) as data:
        arrays = {k: data[k] for k in names + [f'n_{k}' for k in PADDED] if k in data.files}
    out = []
    for k in range(len(arrays['seed'])):
        row = {}
        for name in names:
            value = arrays[name][k]
            if name in PADDED:
                n = arrays[f'n_{name}'][k]
                value = value[:n] if PADDED[name] == 0 else value[:, :n]
            row[name] = int(value) if name in ('seed', 'attempt') else value
        out.append(Layout(**row))
    return out


@pytest.fixture(scope='module')
def pools():
    return lean_pool(COMPACT_POOL), lean_pool(SUPPLY_POOL)


# ---- the fixture pack ---------------------------------------------------------------------------------------------
def tiles_of(mask):
    """A bool [W, H] grid's tiles as layout tile indices, in np.argwhere's (x-major) order."""
    H = mask.shape[1]
    return [int(x) * H + int(z) for x, z in np.argwhere(mask)]


def supply_keys(layout, types, index):
    """Every scarce variant key (layout, grove, kept row) a draw on this supply layout can reach."""
    found = groves(layout, types)
    return {(index, g, scarce_rows(layout, types, found[g], (int(x), int(z)))[1])
            for g, tiles in supply_sites(layout, types) for x, z in np.argwhere(tiles)}


OFFSETS = ('flags_boot', 'steps_boot', 'steps_snap', 'sides', 'entity_reach', 'tile_src', 'src_tile', 'start_tile', 'locs',
           'loc_reach', 'open_locs', 'npcs', 'fish', 'order0', 'dist0', 'parent0', 'snap_dist', 'camp',
           'outside', 'grove_off', 'grove_row', 'grove_normal', 'site_grove', 'site_off', 'site_tile')  # heap offsets


class Heaps:
    def __init__(self):
        self.parts = {dt: [] for dt in ('i32', 'i16', 'u16', 'u8')}
        self.size = dict.fromkeys(self.parts, 0)

    def put(self, dt, values):
        values = np.asarray(values, dict(i32=np.int32, i16=np.int16, u16=np.uint16, u8=np.uint8)[dt]).ravel()
        if values.size == 0:
            return -1
        at = self.size[dt]
        self.parts[dt].append(values)
        self.size[dt] += values.size
        return at


def write_pack(path, types, compact, supply, failing=(), keys=None, regions=True):
    """A pack of the fields the task draws and ledgers read (rs.h): strings, consts, npc and obj types, the compact
    and supply pools with their start tiles, npcs, camps, outside regions (empty without `regions`), groves and sites,
    and a variant record per scarce key (`keys`, default every reachable one; `failing` keys get rec -1, a variant
    that fails its check). Returns {key: rec}."""
    sec = native.load().constants()['sections']
    writer = native.PackWriter(path, native.manifest(inputs=dict(fixture='sim/tests/test_native_draws.py')))
    strings = ['', *PRODUCE_SCRIPTS, NOT_GATHERING, *sorted(set(GATHERING_SCRIPTS) - set(PRODUCE_SCRIPTS))]
    blob, offsets = b'', []
    for text in strings:
        offsets.append(len(blob))
        blob += text.encode() + b'\0'
    writer.add(sec['STRINGS'], blob)
    writer.add(sec['STR_OFF'], np.asarray(offsets, np.uint32))

    consts = np.zeros(1, native.record_dtype('RsConsts'))
    npc = {v['debugname']: k for k, v in types.npc_types.items()}  # combat_npcs' and enemy_types' (last wins)
    for name in ('goblin', 'mugger', 'chicken', 'chicken_brown'):
        consts[f'npc_{name}'] = npc[name]
    consts['kit'] = [types.obj_id(name) for name, _ in START_KIT]
    consts['kit_count'] = [count for _, count in START_KIT]
    consts['weapons'] = [types.obj_id(name) for name in ('bronze_dagger', 'bronze_sword')]
    consts['combat_tools'] = [types.obj_id(name) for name, _ in COMBAT['tools']]
    consts['combat_tool_count'] = [count for _, count in COMBAT['tools']]
    consts['combat_food'] = [types.obj_id(name) for name in COMBAT['food']]
    for field, script in zip(('get_logs', 'ore_normal', 'ore_fast', 'fish_roll'), PRODUCE_SCRIPTS):
        consts[f's_produce_{field}'] = strings.index(script)
    writer.add(sec['CONSTS'], consts, record='RsConsts')

    npcs = np.zeros(max(types.npc_types) + 1, native.record_dtype('RsNpcType'))
    for k, v in types.npc_types.items():
        npcs[k]['present'], npcs[k]['size'] = 1, v['size']
    writer.add(sec['NPC_TYPES'], npcs, record='RsNpcType')
    objs = np.zeros(max(types.obj_types) + 1, native.record_dtype('RsObjType'))
    objs['present'] = [k in types.obj_types for k in range(len(objs))]
    objs['made_family'] = object_families(types)
    for g, goal in enumerate(GOALS):  # GoalEpisode.family and audit's family: the goal items and their notes
        for base in GOAL_FAMILIES.get(goal, ()):
            for obj in (base, types.obj_types.get(base, {}).get('certlink', -1)):
                if obj is not None and 0 <= obj < len(objs):
                    objs[obj]['family_mask'] |= 1 << g
    writer.add(sec['OBJ_TYPES'], objs, record='RsObjType')

    heaps, recs = Heaps(), []
    T = 64 * 64
    per_tile = dict(flags_boot=heaps.put('i32', np.zeros(T)), steps_boot=heaps.put('u8', np.zeros(T)),
                    sides=heaps.put('u8', np.zeros(T)), entity_reach=heaps.put('u16', np.zeros(T)),
                    tile_src=heaps.put('i16', np.full(T, -1)))
    per_tile['steps_snap'] = per_tile['steps_boot']  # the draws read none of these; a layout needs them all

    def record(layout, kind, index):
        W, H = layout.flags.shape
        r = np.zeros(1, native.record_dtype('RsLayoutRec'))
        for name in OFFSETS:
            r[name] = -1
        r['gate'] = -1
        r['kind'], r['pool_index'], r['W'], r['H'], r['T'], r['x0'], r['z0'] = kind, index, W, H, W * H, *ORIGIN
        r['seed'], r['attempt'] = layout.seed, layout.attempt
        for name, at in per_tile.items():
            r[name] = at
        start = tiles_of(layout.start)
        r['n_start'], r['start_tile'] = len(start), heaps.put('i16', start)
        r['n_locs'], r['locs'] = len(layout.locs), heaps.put('i32', layout.locs)
        r['n_npcs'], r['npcs'] = len(layout.npcs), heaps.put('i32', layout.npcs)
        return r

    for k, layout in enumerate(compact):
        r = record(layout, 0, k)
        spawns = [x * layout.flags.shape[1] + z for x, z in camp(layout, types)]
        region = tiles_of(outside(layout)) if regions else []
        r['n_camp'], r['camp'] = len(spawns), heaps.put('i16', spawns)
        r['n_outside'], r['outside'] = len(region), heaps.put('i16', region)
        recs.append(r)
    reachable = set()
    for k, layout in enumerate(supply):
        r = record(layout, 1, k)
        found = groves(layout, types)
        rows = [row for g in found for row in g]
        r['n_groves'], r['grove_off'] = len(found), heaps.put('i32', np.cumsum([0] + [len(g) for g in found]))
        r['grove_row'] = heaps.put('i32', rows)
        r['grove_normal'] = heaps.put('u8', [is_normal(layout, types, row) for row in rows])
        sites = supply_sites(layout, types)
        r['n_sites'], r['site_grove'] = len(sites), heaps.put('i32', [g for g, _ in sites])
        r['site_off'] = heaps.put('i32', np.cumsum([0] + [int(tiles.sum()) for _, tiles in sites]))
        r['site_tile'] = heaps.put('i16', [t for _, tiles in sites for t in tiles_of(tiles)])
        recs.append(r)
        reachable |= supply_keys(layout, types, k)
    variants, out = [], {}
    for key in sorted(reachable if keys is None else keys):
        if key in failing:
            variants.append((*key, -1))
            out[key] = -1
            continue
        r = record(supply[key[0]], 2, key[0])  # stands in for the variant layout (the draws read only its index)
        r['grove'], r['keep'] = key[1], key[2]
        out[key] = len(recs)
        variants.append((*key, len(recs)))
        recs.append(r)
    layouts = np.zeros(len(recs), native.record_dtype('RsLayoutRec'))  # (np.concatenate drops the struct padding)
    for k, r in enumerate(recs):
        layouts[k] = r[0]
    writer.add(sec['LAYOUTS'], layouts, record='RsLayoutRec')
    for dt in heaps.parts:
        writer.add(sec[f'HEAP_{dt.upper()}'], np.concatenate(heaps.parts[dt]))
    rows = np.zeros(len(variants), native.record_dtype('RsVariantKey'))
    for k, (layout, grove, keep, rec) in enumerate(variants):
        rows[k] = (layout, grove, keep, rec)
    writer.add(sec['VARIANTS'], rows, record='RsVariantKey')
    writer.close()
    return out


class Fixture(SimpleNamespace):
    def __repr__(self):  # failure reports never print the pools' contents (private data)
        return 'Fixture(pack, pools, sites, groves)'


@pytest.fixture(scope='module')
def fixture(types, pools):
    compact, supply = pools
    PACKS.mkdir(parents=True, exist_ok=True)
    path = PACKS / 'draws.pack'
    recs = write_pack(path, types, compact, supply)
    pack = native.open_pack(path)
    regions = [outside(layout) for layout in compact]
    sites = [supply_sites(layout, types) for layout in supply]
    supply_groves = [groves(layout, types) for layout in supply]
    yield Fixture(pack=pack, types=types, compact=compact, supply=supply, regions=regions, sites=sites,
                  groves=supply_groves, recs=recs)
    del pack
    path.unlink(missing_ok=True)


# ---- the oracle's draws ---------------------------------------------------------------------------------------------
def oracle_draw(f, task, goal, key, slot, episode):
    """CpuBatch._episode's draws (vec.py:114-151) for a slot of `goal`: (start dict, the layout record it plays, kit,
    combat npcs or None)."""
    spec = task_spec(task)
    if spec['supply'] and goal == 'wood':
        start = draw_supply(key, slot, episode, f.supply, f.sites)
        n = len(f.compact)
        if start['variant'] == 'ample':
            rec = n + start['layout']
        else:  # CpuBatch._supply_tables' key
            layout = f.supply[start['layout']]
            tile = (start['tile'][0] - ORIGIN[0], start['tile'][1] - ORIGIN[1])
            _, keep = scarce_rows(layout, f.types, f.groves[start['layout']][start['grove']], tile)
            rec = f.recs[(start['layout'], start['grove'], keep)]
        return start, rec, START_KIT + ((start['weapon'], 1),), None
    start = draw_start(key, slot, episode, f.compact)
    if not (spec['combat'] and goal == 'melee'):
        return start, start['layout'], START_KIT + ((start['weapon'], 1),), None
    layout = f.compact[start['layout']]
    start['combat'] = c = draw_combat(key, slot, episode, layout, start, f.types, region=f.regions[start['layout']])
    npcs = [(n['nid'], n['type'], n['x'], n['z'], n['level'], n['size'])
            for n in combat_npcs(layout, f.types, c['muggers'])]
    return start, start['layout'], tuple(kit(start['weapon'], c['food'])), npcs


def start_kind(start):
    return 'supply' if start.get('supply') else 'combat' if 'combat' in start else 'short'


def check_draw(f, task, goal, key, slot, episode):
    start, rec, kit_names, npcs = oracle_draw(f, task, goal, key, slot, episode)
    raw, kit_ids, spawns = f.pack.episode_draw(TASKS[task], GOALS.index(goal), key, slot, episode)
    assert native.decode_start(raw) == start, (task, goal, key, slot, episode)
    assert raw['kind'] == START_KINDS[start_kind(start)] and raw['rec'] == rec
    assert kit_ids == [(f.types.obj_id(name), count) for name, count in kit_names]
    assert spawns == npcs
    return start


@settings(deadline=None, max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(INT64, INDEX, INDEX)
def test_episode_draws_equal_the_oracle(fixture, key, slot, episode):
    """Every task version and goal set: the start CpuBatch draws for (key, slot, episode) and the slot's goal."""
    for task in TASKS:
        for goals in (GOALS, *((g,) for g in GOALS)):
            check_draw(fixture, task, slot_goal(goal_set(goals), slot), key, slot, episode)


@settings(deadline=None, max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(INT64, INDEX, INDEX)
def test_draw_start_equals_the_oracle_on_either_pool(fixture, key, slot, episode):
    """draw_start alone on each pool, the supply pool's tile included (draw_supply replaces it)."""
    for pool, layouts in ((POOL_COMPACT, fixture.compact), (POOL_SUPPLY, fixture.supply)):
        assert native.decode_start(fixture.pack.draw_start(pool, key, slot, episode)) == \
            draw_start(key, slot, episode, layouts)


def test_a_keyed_sweep_reaches_every_draw_branch(fixture):
    """A fixed sweep of CpuBatch-like (key, slot, episode) triples equals the oracle and reaches every group, food
    count, worn state, mugger, variant and supply site, so the comparison above is not a few branches deep."""
    seen = Counter()
    for key in (0, 3, 7, -1, 2 ** 40 + 5):
        for slot in range(16):
            for episode in range(12):
                start = check_draw(fixture, 'goal-activity-v5', 'melee', key, slot, episode)
                melee = start['combat']
                wood = check_draw(fixture, 'goal-activity-v5', 'wood', key, slot, episode)
                check_draw(fixture, 'goal-activity-v2', 'melee', key, slot, episode)
                seen.update([('group', melee['group']), ('food', len(melee['food'])), ('worn', melee['worn']),
                             ('muggers', len(melee['muggers'])), ('variant', wood['variant']),
                             ('site', wood['grove'] > 0), ('hp below base', melee['hp'] < start['levels']['hitpoints'])])
    assert {k for (kind, k) in seen if kind == 'group'} == set(GROUPS)
    assert {k for (kind, k) in seen if kind == 'food'} == {0, 1, 2, 3, 4}
    for fact in (('worn', True), ('worn', False), ('muggers', 1), ('variant', 'ample'), ('variant', 'scarce'),
                 ('site', True), ('hp below base', True)):
        assert seen[fact] > 0, fact


def native_error(pack, call):
    with pytest.raises(native.load().Error) as caught:
        call()
    return native.as_python(caught.value, pack)


def test_draw_errors_are_the_oracles(fixture):
    """A layout with too few mugger tiles (combat.py, ValueError) and a scarce variant that fails its check
    (search.py:361-362, RuntimeError, before the episode is built) raise the oracle's class and text; a scarce key the
    pack lacks is a pack error. The rest of those draws equal the oracle's."""
    compact, supply, types = fixture.compact, fixture.supply, fixture.types
    layout, keys = compact[0], sorted(supply_keys(supply[0], types, 0))
    failing = {k for k in keys if k[1] == keys[0][1]}
    path = PACKS / 'draw-errors.pack'
    write_pack(path, types, [layout], [supply[0]], failing=failing, keys=failing, regions=False)
    try:
        pack = native.open_pack(path)
        sites = [supply_sites(supply[0], types)]
        hit = Counter()
        for slot in range(64):
            start = draw_start(5, slot, 0, [layout])
            call = lambda: pack.episode_draw(5, GOALS.index('melee'), 5, slot, 0)
            try:
                start['combat'] = draw_combat(5, slot, 0, layout, start, types, region=np.zeros_like(layout.start))
            except ValueError as expected:
                got = native_error(pack, call)
                assert type(got) is ValueError and str(got) == str(expected)
                hit['muggers'] += 1
            else:
                assert native.decode_start(call()[0]) == start
            drawn = draw_supply(5, slot, 0, [supply[0]], sites)
            call = lambda: pack.episode_draw(5, GOALS.index('wood'), 5, slot, 0)
            if drawn['variant'] == 'ample':
                assert native.decode_start(call()[0]) == drawn
                continue
            got = native_error(pack, call)
            assert type(got) is RuntimeError
            if drawn['grove'] == keys[0][1]:  # variant_layout's text (search.py:362)
                assert str(got) == f'layout {supply[0].seed}: the scarce variant of grove {drawn["grove"]} fails a check'
                hit['failing'] += 1
            else:
                assert str(got).startswith('native pack: ')
                hit['absent'] += 1
        assert hit['muggers'] and hit['failing'] and hit['absent'], hit
    finally:
        path.unlink(missing_ok=True)


# ---- ledgers: events as RsEvent rows -----------------------------------------------------------------------------
EVENT_KINDS = dict(agent_hp=0, pickup=1, produce=2, transfer=3, remove=4, npc_damage=5, npc_defeated=6, excluded=7)
CAUSES = {'damage': 0, 'eat': 1, 'regen': 2, 'levelup': 3, None: 4}
WHYS = dict(eat=0, bury=1, drop=2)
LOCATIONS = dict(inv=0, worn=1, bank=2)
WHATS = dict(clue_drop=0, loot_drop=1)
STRINGS = ['', *PRODUCE_SCRIPTS, NOT_GATHERING]


def event_row(event):
    """A (tick, kind, data) of world.events as the RsEvent the engine appends (rs.h field comments)."""
    tick, kind, d = event
    k = EVENT_KINDS[kind]
    if kind == 'agent_hp':
        source = d['source']
        return (tick, k, CAUSES[d['cause']], 0, source is not None, 0, d['planned'], d['before'], d['after'],
                0 if source is None else source, 0)
    if kind == 'pickup':
        return (tick, k, 0, 0, 0, d['obj'], d['count'], 0, 0, 0, 0)
    if kind == 'produce':
        return (tick, k, 0, 0, 0, d['obj'], d['count'], STRINGS.index(d['script']), 0, 0, 0)
    if kind == 'transfer':
        return (tick, k, LOCATIONS[d['src']], LOCATIONS[d['dst']], 0, d['obj'], d['count'], d['to'], 0, 0, 0)
    if kind == 'remove':
        return (tick, k, WHYS[d['why']], 0, 0, d['obj'], d['count'], 0, 0, 0, 0)
    if kind == 'excluded':
        return (tick, k, WHATS[d['what']], 0, 0, d['obj'], 0, d['npc'], 0, 0, 0)
    mine = d['source'] == AGENT
    if kind == 'npc_damage':
        return (tick, k, mine, 0, 0, d['npc'], d['type'], d['life'], d['damage'], d['hp'], d['full'])
    return (tick, k, mine, 0, 0, d['npc'], d['type'], d['life'], 0, 0, 0)


def holdings_rows(h):
    return tuple([tuple(map(int, row)) for row in h[loc]] for loc in ('inventory', 'bank', 'worn'))


def as_error(pack, args):
    return native.as_python(native.load().Error(*args), pack)


def same_error(got, expected):
    """The C error is the oracle's class and text (a holdings mismatch drops Python's counter reprs, rs.h)."""
    assert type(got) is type(expected), (got, expected)
    text = str(expected)
    if 'holdings not explained by the events' in text:
        text = text.split(':')[0]
    assert str(got) == text, (got, expected)


def check_ledgers(pack, types, events, cuts, goal, enemies, target, terminal, initial, final, start, visited=0,
                  tracker=None, expected_steps=None, expected_facts=None, task=None):
    """The C ledgers against the oracle's functions over `events` (since arm): each step cut's made and kills (and
    CpuBatch's own info, `expected_steps`), then the terminal accounts and info facts. Returns the outcome: 'running',
    'closed', or the oracle's error text."""
    kind, group, variant = start
    got = pack.ledgers(goal=GOALS.index(goal), enemies=sorted(enemies), target=target,
                       events=[event_row(e) for e in events], cuts=cuts,
                       terminal=None if terminal is None else (dict(horizon=1, death=2)[terminal['reason']],
                                                               terminal['tick']),
                       initial=holdings_rows(initial), final=holdings_rows(final), kind=kind, group=group,
                       variant=variant, visited=visited, exhausted=-1 if tracker is None or tracker.exhausted is None
                       else tracker.exhausted, logs=[] if tracker is None else [(t, int(b)) for t, b in tracker.logs])
    family = object_families(types)
    if expected_steps is not None:
        assert got['steps'] == expected_steps
    for k, cut in enumerate(cuts):
        try:
            want = (*map(int, produced(events[:cut], family)), kills(events[:cut], AGENT, enemies))
        except AuditError as error:
            same_error(as_error(pack, got['error']), error)
            assert len(got['steps']) == k
            return str(error)
        assert got['steps'][k] == want, (k, cut)
    assert got['error'] is None
    if terminal is None:
        return 'running'
    terminal = dict(terminal, holdings=final)
    try:
        report = audit(events, initial, terminal, goal, types, AGENT, enemies)
    except AuditError as error:
        same_error(as_error(pack, got['audit']), error)
        return str(error)
    assert got['audit'] == dict(survived=report['survived'], reason=dict(horizon=1, death=2)[report['reason']],
                                tick=report['tick'], produced=report['produced'], retained=report['retained'],
                                retained_upper=report['retained_upper'], picked=report['picked'],
                                kills=report['kills']['lower'])
    task = task or SimpleNamespace(
        world=SimpleNamespace(events=events, t=types, npcs=[SimpleNamespace(nid=k, base=41) for k in range(8)]),
        event_start=0, terminal=terminal, agent=SimpleNamespace(name=AGENT), enemies=enemies, arm_hp=10, start_tick=0)
    s = fights(task)['summary']
    assert got['fights'] == [s['eats'] + s['eats_without_gain'], s['eat_gain'], s['regen_gain'], s['levelup_gain'],
                             s['aggressed'], s['eats_without_gain']]
    if expected_facts is not None:
        assert got['facts'] == list(map(int, expected_facts))
    return 'closed'


# ---- Hypothesis event streams -------------------------------------------------------------------------------------
@st.composite
def streams(draw, objs):
    """An episode's events since arm, holdings at arm and close, and step cuts: mostly consistent with the holdings
    (so the audit's later checks are reached), sometimes not; npc hits and defeats over a few npcs and lives by the
    agent, nobody or another player; heals of every cause; a terminal inside or past the last event."""
    held = Counter()
    initial = dict(inventory=[], bank=[], worn=[])
    for loc, n in (('inventory', 4), ('bank', 1), ('worn', 2)):
        for slot in range(draw(st.integers(0, n))):
            obj, count = draw(st.sampled_from(objs)), draw(st.integers(0, 3))
            initial[loc].append((slot, obj, count))
            held[loc, obj] += count
    events, tick, hit = [], 0, []
    npc = st.integers(0, 3)
    for _ in range(draw(st.integers(0, 40))):
        tick += draw(st.sampled_from((0, 0, 1, 3)))
        kind = draw(st.sampled_from(('npc_damage', 'npc_damage', 'npc_defeated', 'agent_hp', 'agent_hp', 'produce',
                                     'pickup', 'transfer', 'remove', 'excluded')))
        obj, count = draw(st.sampled_from(objs)), draw(st.integers(0, 2))
        if kind in ('npc_damage', 'npc_defeated'):
            data = dict(npc=draw(npc), type=draw(st.sampled_from((41, 1017, 100, 175, 999))),
                        life=draw(st.integers(0, 2)), source=draw(st.sampled_from((AGENT, AGENT, None, 'other'))))
            if kind == 'npc_damage':
                full = draw(st.integers(1, 5))
                data.update(damage=draw(st.integers(0, 3)), hp=draw(st.integers(0, full)), full=full)
                hit.append((data['npc'], data['type'], data['life']))
            elif hit and draw(st.integers(0, 19)):  # mostly a life that was hit (else the defeat error)
                data['npc'], data['type'], data['life'] = draw(st.sampled_from(hit))
        elif kind == 'agent_hp':
            cause = draw(st.sampled_from(tuple(CAUSES)))
            before = draw(st.integers(0, 12))
            source = draw(st.sampled_from((None, 0, 1, 2, 3))) if cause == 'damage' else None
            data = dict(cause=cause, planned=draw(st.integers(0, 4)), before=before,
                        after=draw(st.integers(0, 12)), source=source)
        elif kind == 'produce':
            data = dict(obj=obj, count=count, script=draw(st.sampled_from(PRODUCE_SCRIPTS + PRODUCE_SCRIPTS[:1] * 6
                                                                          + (NOT_GATHERING,))))
            held['inventory', obj] += count
        elif kind == 'pickup':
            data = dict(obj=obj, count=count)
            held['inventory', obj] += count
        elif kind == 'transfer':
            src, dst = draw(st.sampled_from(tuple(LOCATIONS))), draw(st.sampled_from(tuple(LOCATIONS)))
            name = dict(inv='inventory', worn='worn', bank='bank')
            have = [o for (loc, o), c in held.items() if loc == name[src] and c > 0]
            if have and draw(st.integers(0, 9)):
                obj = draw(st.sampled_from(have))
                count = draw(st.integers(1, held[name[src], obj]))
            to = draw(st.sampled_from((obj, obj, draw(st.sampled_from(objs)))))
            data = dict(src=src, dst=dst, obj=obj, to=to, count=count)
            held[name[src], obj] -= count
            held[name[dst], to] += count
        elif kind == 'remove':
            have = [o for (loc, o), c in held.items() if loc == 'inventory' and c > 0]
            if have and draw(st.integers(0, 9)):
                obj = draw(st.sampled_from(have))
                count = draw(st.integers(1, held['inventory', obj]))
            data = dict(obj=obj, count=count, why=draw(st.sampled_from(tuple(WHYS))))
            held['inventory', obj] -= count
        else:
            data = dict(what=draw(st.sampled_from(tuple(WHATS))), obj=obj, npc=draw(npc))
        events.append((tick, kind, data))
    final = dict(inventory=[], bank=[], worn=[])
    for (loc, obj), count in sorted(held.items()):
        if count > 0 or draw(st.integers(0, 3)) == 0:
            final[loc].append((len(final[loc]), obj, max(count, 0)))
    if final['inventory'] and draw(st.integers(0, 7)) == 0:  # a holding the events do not explain
        slot, obj, count = final['inventory'][0]
        final['inventory'][0] = (slot, obj, count + 1)
    last = events[-1][0] if events else 0
    terminal = dict(reason=draw(st.sampled_from(('horizon', 'death'))),
                    tick=draw(st.sampled_from((last + 1, last + 1, max(last, 1), last + 5))))
    cuts = sorted(draw(st.lists(st.integers(0, len(events)), max_size=4))) + [len(events)]
    return events, cuts, initial, final, terminal


@pytest.fixture(scope='module')
def ledger_objs(types):
    """A few obj ids of every role: goal items and their notes, food, kit, coins."""
    names = ('logs', 'raw_shrimp', 'copper_ore', 'shrimp', 'bread', 'bronze_axe', 'coins', 'bones')
    found = [types.obj_id(name) for name in names if name in types.obj_ids]
    notes = [types.obj_types[o]['certlink'] for o in found[:3] if types.obj_types[o].get('certlink', -1) >= 0]
    return sorted(set(found + notes))


@settings(deadline=None, max_examples=400, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                  HealthCheck.too_slow])
@given(st.data())
def test_ledgers_equal_the_oracle_on_event_streams(fixture, ledger_objs, data):
    """made and kills step by step, then audit, fights and their errors, for every goal and both enemy sets."""
    events, cuts, initial, final, terminal = data.draw(streams(ledger_objs))
    goal = data.draw(st.sampled_from(GOALS))
    enemies = data.draw(st.sampled_from((frozenset((41, 1017)), frozenset((100, 175)))))
    check_ledgers(fixture.pack, fixture.types, events, cuts, goal, enemies, 5, terminal, initial, final, (0, 0, -1))


def test_the_streams_reach_every_ledger_outcome(fixture, ledger_objs):
    """The strategy above reaches clean closes, each audit error and the defeat error (a fixed Hypothesis sample)."""
    outcomes = Counter()

    @settings(deadline=None, max_examples=300, database=None, derandomize=True,
              suppress_health_check=[HealthCheck.too_slow])
    @given(st.data())
    def sample(data):
        events, cuts, initial, final, terminal = data.draw(streams(ledger_objs))
        goal = data.draw(st.sampled_from(GOALS))
        outcome = check_ledgers(fixture.pack, fixture.types, events, cuts, goal, frozenset((41, 1017)), 5, terminal,
                                initial, final, (0, 0, -1))
        outcomes['production' if outcome.startswith('production by') else outcome.split(':')[0]] += 1
    sample()
    for outcome in ('closed', 'production', 'transfer of items not held', 'removal of items not held',
                    'inventory holdings not explained by the events', 'an npc defeated without a hit'):
        assert outcomes[outcome] > 0, (outcome, outcomes)


# ---- recorded episodes: CpuBatch runs, their world.events and info ----------------------------------------------------
def recorded_run(types, compact, supply, task, n, steps, key):
    """Keyed random legal play (vec.R_ACTION's rule, as sim/tests/test_workers.py run) of a CpuBatch, recording per
    world the events of each episode since arm, the step cuts and info, and at each close the ended task."""
    from sim.vec import CpuBatch
    env = CpuBatch(types, compact, n, task=task, search=(supply, None) if task == 'goal-activity-v5' else None)
    state, obs = env.reset(key)
    rng, episodes = Rng(key), []
    current = [dict(task=env.tasks[b], tracker=env.trackers[b], start=env.starts[b], cuts=[], steps=[])
               for b in range(n)]
    for step in range(steps):
        legal = [np.flatnonzero(obs['mask'][b]) for b in range(n)]
        action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(n)], np.int32)
        state, obs, reward, done, info = env.step(state, action)
        if info['unsupported'].any():  # the run stops there (CpuBatch refuses that world from now on)
            break
        for b in range(n):
            ep = current[b]
            t = ep['task']
            ep['cuts'].append(len(t.world.events) - t.event_start)
            ep['steps'].append((*map(int, info['made'][b]), int(info['kills'][b])))
            if done[b]:
                episodes.append(dict(ep, facts=info['facts'][b].copy()))
                current[b] = dict(task=env.tasks[b], tracker=env.trackers[b], start=env.starts[b], cuts=[], steps=[])
    return episodes + [dict(ep, facts=None) for ep in current if ep['cuts']]


def start_ints(start):
    """(RsStart kind, group, variant) of a start dict."""
    kind = start_kind(start)
    return (START_KINDS[kind], GROUPS.index(start['combat']['group']) if kind == 'combat' else 0,
            SUPPLY['variants'].index(start['variant']) if kind == 'supply' else -1)


@pytest.fixture(scope='module')
def recorded(types, pools):
    """Keyed samples that hold every fact the test needs (a death among them). mask-v5 changed keyed random play
    (9 November 2026): v5's key 11 then ended no episode in a death at 400, 500 or 600 steps, and key 12 did."""
    compact, supply = pools
    return {task: recorded_run(types, compact, supply[:4], task, n, steps, key) for task, n, steps, key in (
        ('goal-activity-v5', 8, 400, 12), ('goal-activity-v3', 4, 150, 13), ('goal-activity-v2', 4, 150, 12))}


def test_ledgers_equal_cpubatch_on_recorded_episodes(fixture, recorded):
    """Per step, CpuBatch's info made and kills; at each close, the audit, the fights summary and info facts, from the
    episode's own events, holdings, start, visited cells and tracker."""
    outcomes = Counter()
    for task, episodes in recorded.items():
        for ep in episodes:
            t, closed = ep['task'], ep['facts'] is not None
            events = t.world.events[t.event_start:]
            outcome = check_ledgers(fixture.pack, fixture.types, events, ep['cuts'], t.goal, t.enemies, t.target,
                                    t.terminal if closed else None, t.initial,
                                    t.terminal['holdings'] if closed else t.initial, start_ints(ep['start']),
                                    visited=len(t.agent.visited), tracker=ep['tracker'], expected_steps=ep['steps'],
                                    expected_facts=ep['facts'], task=t)
            outcomes[outcome, start_kind(ep['start'])] += 1
            if closed:
                outcomes[t.terminal['reason']] += 1
            outcomes['kills'] += ep['steps'][-1][3] if ep['steps'] else 0
            outcomes['made'] += sum(ep['steps'][-1][:3]) if ep['steps'] else 0
            outcomes['hits'] += sum(kind == 'npc_damage' for _, kind, _ in events)
            outcomes['eats'] += sum(kind == 'remove' and d['why'] == 'eat' for _, kind, d in events)
    assert outcomes['closed', 'short'] >= 10 and outcomes['closed', 'supply'] and outcomes['closed', 'combat'], outcomes
    for fact in ('death', 'kills', 'made', 'hits', 'eats'):
        assert outcomes[fact] > 0, (fact, outcomes)
