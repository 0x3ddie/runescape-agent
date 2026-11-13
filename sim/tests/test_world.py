"""The compact world generator (sim/world.py): declared limits, reachability (through a pen's gate), and
frozen tables that are the reference's own answers on each collision state of a layout."""
import numpy as np
import pytest

from sim import paths, world
from sim.engine import SCRIPTS, Player, World


@pytest.fixture(scope='module')
def layouts(tables):
    return [world.generate(seed, tables) for seed in world.split_seeds('development', 40)]


@pytest.fixture(scope='module')
def pen(layouts):
    return next(layout for layout in layouts if layout.gate[0] >= 0)


@pytest.fixture(scope='module')
def derived(tables, pen):
    return world.derive(pen, tables)


def flags_fn(flags):
    return lambda x, z: int(flags[x, z]) if 0 <= x < world.W and 0 <= z < world.H else world.FLAG_NULL


def test_splits_are_disjoint_and_a_seed_always_draws_the_same_layout(tables, layouts):
    ranges = sorted(world.SPLITS.values())
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    again = world.generate(layouts[3].seed, tables)
    assert all(np.array_equal(a, b) for a, b in zip(layouts[3].arrays().values(), again.arrays().values()))
    assert not np.array_equal(layouts[0].flags, layouts[1].flags)
    for layout in layouts:  # a pen is the seed's own draw, so the fraction is pen_fraction
        coin = np.random.default_rng([layout.seed, world.PEN_STREAM]).random() < world.WORLD['pen_fraction']
        assert coin == (layout.gate[0] >= 0) and (len(layout.open_locs) == 2) == coin
        assert coin or (np.array_equal(layout.flags, layout.flags_open) and np.array_equal(layout.flags, layout.flags_snapshot))


def test_every_resource_is_operable_and_every_start_reaches_every_goal_in_the_declared_steps(tables, layouts):
    for layout in layouts:
        flags = world._flags_fn(layout.flags)
        operable = lambda tiles: any(layout.walkable[x, z] for x, z in tiles if 0 <= x < world.W and 0 <= z < world.H)
        names = [tables.loc_types[int(ty)]['debugname'] for ty in layout.locs[:, 0]]
        for (type_id, x, z, shape, angle), name in zip(layout.locs, names):
            if name in (world.TREE, world.OAK) or any(name in kinds for kinds in world.ROCKS.values()):
                lt = tables.loc_types[int(type_id)]
                assert operable(paths.loc_reaching(flags, x, z, lt['width'], lt['length'], angle, shape, 0)), name
        assert {world.TREE, 'tinrock1'} <= set(names) and set(names) & set(world.ROCKS['copper'])
        lo, hi = world.WORLD['fish_coords']
        assert lo <= len(layout.fish) <= hi
        for x, z in layout.fish:  # a spot stands on water and is fished from the shore
            assert layout.flags[x, z] & world.FLOOR and operable(paths.entity_reaching_live(flags, x, z))
        hens = [t for _, t, _, _ in layout.npcs if tables.npc_types[int(t)]['debugname'] in world.CHICKENS]
        assert world.WORLD['chickens'][0] <= len(hens) <= world.WORLD['chickens'][1]
        assert layout.start.sum() >= world.WORLD['min_starts'] and not (layout.start & ~layout.walkable).any()
        d = layout.goal_dist[:, layout.start]
        assert (d >= 0).all() and d.max() <= world.WORLD['goal_steps']


def test_a_pen_holds_its_chickens_until_its_gate_opens(tables, layouts):
    pens = [layout for layout in layouts if layout.gate[0] >= 0]
    assert pens
    for layout in pens:
        start = np.zeros((world.W, world.H), bool)
        start[tuple(np.argwhere(layout.start)[0])] = True
        closed = world.flood(paths.step_masks(layout.flags), start) >= 0
        opened = world.flood(paths.step_masks(layout.flags_open), start) >= 0
        for _, t, x, z in layout.npcs:
            if tables.npc_types[int(t)]['debugname'] in world.CHICKENS:
                beside = paths.entity_reaching_live(flags_fn(layout.flags_open), int(x), int(z))
                assert not closed[int(x), int(z)] and any(opened[a, b] for a, b in beside)
        names = [tables.loc_types[int(layout.locs[k, 0])]['debugname'] for k in layout.gate]
        assert names == list(world.PEN_GATE) and (layout.locs[layout.gate, 3] == world.WALL).all()


def test_the_gate_scripts_give_the_frozen_open_state_close_it_and_revert_it(tables, layouts):
    for layout in (layout for layout in layouts if layout.gate[0] >= 0):
        w, p = World(world.LayoutTables(tables, layout)), None
        p = Player(w, 'tester')
        state = lambda: np.asarray(w._flags).reshape(world.W, world.H)
        added = lambda: sorted((loc for at in w.loc_at.values() for loc in at if loc.index >= w.n_stored_locs and loc.active),
                               key=lambda loc: loc.index)
        main = w.locs[int(layout.gate[0])]
        op = lambda loc: list(SCRIPTS[tables.loc_types[loc.type]['triggers']['OPLOC1']](w, p, loc))
        op(main)
        assert np.array_equal(state(), layout.flags_open)
        assert [(loc.type, loc.x - world.ORIGIN[0], loc.z - world.ORIGIN[1], loc.shape, loc.angle) for loc in added()] == \
            [tuple(int(v) for v in row) for row in layout.open_locs]
        op(added()[1])  # the outer open leaf closes both
        assert np.array_equal(state(), layout.flags) and not w.changed and not added()
        op(w.locs[int(layout.gate[1])])  # the outer closed leaf opens both
        assert np.array_equal(state(), layout.flags_open)
        for _ in range(501):  # loc_del and loc_add last 500 ticks
            w.process_zones()
        assert np.array_equal(state(), layout.flags) and not w.changed


def test_the_declared_boundary_blocks_everything_beyond_the_layout(tables, layouts):
    w = World(world.LayoutTables(tables, layouts[0]))
    x0, z0 = world.ORIGIN
    for x, z in ((x0 - 1, z0), (x0 + world.W, z0 + 5), (x0 + 3, z0 + world.H)):
        assert w.flags(0, x, z) == world.FLAG_NULL
    steps = paths.step_masks(layouts[0].flags)
    for k, (dx, dz) in enumerate(paths.STEPS):
        xs, zs = np.nonzero((steps >> k) & 1)
        assert len(xs) and ((xs + dx >= 0) & (xs + dx < world.W) & (zs + dz >= 0) & (zs + dz < world.H)).all()


@pytest.mark.parametrize('state', [0, 1])
def test_frozen_tables_are_the_path_finder_sight_and_reach_ports_in_each_state(tables, pen, derived, state):
    collision = (pen.flags, pen.flags_open)[state]
    flags = flags_fn(collision)
    rng = np.random.default_rng(7 + state)
    sources = np.argwhere(pen.walkable)
    rows = lambda prefix, i: [(derived[prefix + k] if prefix else derived[k][state])[i].reshape(world.W, world.H)
                              for k in ('order', 'dist', 'parent')]
    for x, z in sources[rng.integers(len(sources), size=60)]:
        x, z = int(x), int(z)
        i = x * world.H + z
        live = paths.search(paths.step_masks(collision), 0, 0, x, z, bounded=True)
        snapshot = paths.search(paths.step_masks(pen.flags_snapshot), 0, 0, x, z, bounded=True)
        frozen, frozen_snapshot = world._window(*rows('', i), x, z), world._window(*rows('snapshot_', i), x, z)
        for _ in range(5):
            tx, tz = (int(v) for v in rng.integers(0, world.W, 2))
            assert paths.path_to_tile(frozen, x, z, tx, tz) == paths.path_to_tile(live, x, z, tx, tz)
            assert paths.path_to_tile(frozen_snapshot, x, z, tx, tz) == paths.path_to_tile(snapshot, x, z, tx, tz)
            dx, dz = (int(v) for v in rng.integers(-world.LOS_R, world.LOS_R + 1, 2))
            assert bool(derived['los'][state, x, z, dx + world.LOS_R, dz + world.LOS_R]) == \
                paths.line_of_sight(flags, x, z, x + dx, z + dz)
        mask = int(derived['entity_reach'][state, x, z])
        assert [t for b, t in enumerate(paths.NEIGHBOURS) if mask >> b & 1] == \
            [(a - x, b - z) for a, b in paths.entity_reaching_live(flags, x, z)]
    gate = set(int(k) for k in pen.gate)  # the closed leaves stand in state 0, the open stages in state 1
    present = (derived['loc_reach'][state] >= 0).any(axis=1).tolist()
    assert present == [not (state and k in gate) for k in range(len(pen.locs))] + [bool(state)] * len(pen.open_locs)
    if state == 0:  # goal distances (floods from the targets) are the path finder's own depths from each start
        starts = np.argwhere(pen.start)
        reach = {int(t) for k, row in enumerate(derived['loc_reach'][0][:len(pen.locs)])
                 if tables.loc_types[int(pen.locs[k, 0])]['debugname'] == world.TREE for t in row if t >= 0}
        reach = [t for t in reach if pen.walkable[t // world.H, t % world.H]]
        for x, z in starts[rng.integers(len(starts), size=20)]:
            assert pen.goal_dist[0][x, z] == derived['dist'][0][x * world.H + z][reach].min()


def test_layout_tables_compute_what_derive_freezes(tables, pen, derived):
    lazy, frozen = world.LayoutTables(tables, pen), world.LayoutTables(tables, pen, derived)
    rng = np.random.default_rng(3)
    for src in rng.integers(0, len(lazy.meta['sources']), 30):
        src = int(src)
        assert np.array_equal(lazy.parents[src], frozen.parents[src]) and np.array_equal(lazy.orders[src], frozen.orders[src])
        for i, j in rng.integers(0, 2 * 48 + 1, (10, 2)):
            assert lazy.walks[src, int(i), int(j)] == frozen.walks[src, int(i), int(j)]
    assert np.array_equal(lazy.sides, frozen.sides) and np.array_equal(lazy.los, frozen.los)
    assert lazy.meta['loc_reach'] == frozen.meta['loc_reach']


def test_a_world_runs_on_a_layout_and_its_spots_move_among_the_shore_coordinates(tables, layouts):
    layout = layouts[1]
    w = World(world.LayoutTables(tables, layout), seed=3)
    p = Player(w, 'tester')  # npcs turn only near a player (patch 0009)
    p.x, p.z = (int(v) + o for v, o in zip(np.argwhere(layout.start)[0], world.ORIGIN))
    w.pending_login = (p, lambda world, p: None)
    coords = {(world.ORIGIN[0] + int(x), world.ORIGIN[1] + int(z)) for x, z in layout.fish}
    spots = [n for n in w.npcs if tables.npc_types[n.type]['debugname'] == world.SPOT]
    seen = set()
    for _ in range(600):
        w.cycle()
        seen |= {(n.x, n.z) for n in spots}
    assert seen <= coords and len(seen) > len(spots)
    hens = [n for n in w.npcs if n not in spots]
    assert all(max(abs(n.x - n.start_x), abs(n.z - n.start_z)) <= tables.npc_types[n.type]['wanderrange'] for n in hens)
    assert any((n.x, n.z) != (n.start_x, n.start_z) for n in hens)


def test_a_pool_round_trips(tables, layouts, pen, derived, tmp_path):
    other = next(layout for layout in layouts if layout.gate[0] < 0)
    path = world.save_pool(tmp_path / 'pool.npz', [pen, other], [derived, world.derive(other, tables)])
    back, frozen = world.load_pool(path)
    for a, b in zip([pen, other], back):
        assert all(np.array_equal(x, y) for x, y in zip(a.arrays().values(), b.arrays().values()))
    assert all(np.array_equal(derived[k], frozen[0][k]) for k in derived)


# ---- the C port (sim/_paths.c) on frozen layout tables: call by call, then decision by decision ----
needs_c = pytest.mark.skipif(paths._C is None, reason='the C port is off (SIM_PATHS=python) or not built')


@needs_c
def test_the_c_walk_map_answers_and_refuses_as_the_reference(tables, pen, derived):
    frozen = world.LayoutTables(tables, pen, derived)
    assert frozen.walkmap is not None and world.LayoutTables(tables, pen).walkmap is None
    ref = paths.IMPLS['python']
    rng = np.random.default_rng(4)
    r = frozen.meta['walk_r']

    def outcome(fn, *args):
        try:
            return fn(*args)
        except paths.OutsideTables as error:
            return str(error)
    for src in range(len(frozen.meta['sources'])):
        sx, sz, _ = frozen.meta['sources'][src]
        for _ in range(12):
            x, z = sx + int(rng.integers(-r - 3, r + 4)), sz + int(rng.integers(-r - 3, r + 4))
            for name in ('walk_steps', 'walk_approach'):
                assert outcome(getattr(paths, name), frozen, src, sx, sz, x, z) == outcome(ref[name], frozen, src, sx, sz, x, z)


@needs_c
def test_a_rollout_with_the_c_path_finder_is_the_reference_rollout(tables, pen, derived, monkeypatch):
    """CpuBatch on the frozen pen (its gate changes the collision as it is played) under keyed random legal actions:
    with paths.use('c') every decision's observation, reward, end and info equals the Python reference's. act-v3's
    agent asks the path finder for its affordances' walk steps (walk_approach), which the C port answers."""
    from sim.engine import Rng
    from sim.vec import CpuBatch
    searches = [0]

    def rollout(impl):
        paths.use(impl)
        search = paths.walk_approach

        def counted(*args, **kwargs):
            searches[0] += impl == 'c'
            return search(*args, **kwargs)
        monkeypatch.setattr(paths, 'walk_approach', counted)
        try:
            env = CpuBatch(tables, [pen], 8, horizon=240, derived=[derived])
            state, obs = env.reset(11)
            rng, rows = Rng(11), [{k: np.array(v) for k, v in obs.items()}]
            for step in range(150):
                legal = [np.flatnonzero(obs['mask'][b]) for b in range(8)]
                action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(8)], np.int32)
                state, obs, reward, done, info = env.step(state, action)
                rows.append({**{f'obs/{k}': np.array(v) for k, v in obs.items()}, 'reward': reward.copy(), 'done': done.copy(),
                             **{f'info/{k}': np.array(v) for k, v in info.items()}})
            return rows
        finally:
            monkeypatch.undo()
            paths.use('c')
    reference, ours = rollout('python'), rollout('c')
    for step, (a, b) in enumerate(zip(reference, ours)):
        assert a.keys() == b.keys(), step
        for k in a:
            assert np.array_equal(a[k], b[k]), (step, k)
    assert sum(int(r['done'].sum()) for r in reference[1:]) >= 8 and searches[0] > 0, searches
