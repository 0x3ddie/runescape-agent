"""The simulator's path finder returns the engine's own answers (samples from sim/extract.ts):
on the boot map from the stored trees and grids of each level, with doors and gates moved from
its own collision changes and search. act-v3 removed the engine's routes to entities and locs from
World, but their samples still check the reach rules a server op is judged by after the client's
walk (World.entity_reaching and loc_reaching, stored and live, which World.in_operable reads), so
the test answers them as World did before act-v3. The SDK long route and the compass moves
are gone (their samples are not answered); the client's route is sim/tests/test_clicks.py's."""
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from sim import paths
from sim.engine import Loc, World


def engine_route(world, level, sx, sz, reaching, dest, rect):
    """GameMap.findPathToEntity or findPathToLoc as the simulator answered them before act-v3 (World.path_to_entity and
    path_to_loc through paths.path_to_rect): the path finder's tree from the source, the end tile among the
    reach strategy's tiles, and the walk back."""
    tree = world.tree(level, sx, sz)
    return paths.backtrack(tree, sx, sz, paths.end_tile(tree, sx, sz, reaching, dest, rect))


def answer(world, level, q):
    """The simulator's answer to an engine sample, in the sample's form."""
    if q['kind'] == 'tile':
        return paths.packed(world.path_to_tile(level, q['sx'], q['sz'], q['x'], q['z']), level)
    if q['kind'] == 'entity':
        reaching = world.entity_reaching(level, q['x'], q['z'])
        return paths.packed(engine_route(world, level, q['sx'], q['sz'], reaching, (q['x'], q['z']), (1, 1)), level)
    if q['kind'] == 'loc':
        if isinstance(q['loc'], int):  # a map loc, by its index
            loc = world.locs[q['loc']]
        else:  # a loc of a door scenario, its reach searched live
            loc = loc_from(q['loc'], len(world.locs))
            assert world.t.loc_types[loc.type]['forceapproach'] == q['loc']['forceapproach']
        assert loc.level == level
        rect = (loc.length, loc.width) if loc.angle & 1 else (loc.width, loc.length)
        return paths.packed(engine_route(world, level, q['sx'], q['sz'], world.loc_reaching(loc), (loc.x, loc.z), rect),
                            level)
    if q['kind'] == 'reach':
        reaching = set(world.entity_reaching(level, q['x'], q['z']))
        return sum(1 << b for b, (dx, dz) in enumerate(paths.NEIGHBOURS) if (q['x'] + dx, q['z'] + dz) in reaching)
    return world.line_of_sight(level, q['x'], q['z'], q['tx'], q['tz'])


EXPECTED = dict(tile='waypoints', entity='waypoints', loc='waypoints', reach='mask', los='los')


def test_engine_path_samples(tables):
    """Ground-floor paths, and on the first floor paths, reach and sight from the stored tables."""
    world = World(tables)
    samples = tables.meta['path_samples']
    upper = [s for s in samples if s['level'] == 1]
    assert len(samples) - len(upper) >= 3000 and {s['kind'] for s in samples if s['level'] == 0} == {'tile', 'entity', 'loc'}
    assert len(upper) >= 500 and {s['kind'] for s in upper} == set(EXPECTED)
    for s in samples:
        assert answer(world, s['level'], s) == s[EXPECTED[s['kind']]], s


def loc_from(spec, index):
    return Loc(index, spec['type'], spec['type'], spec['x'], spec['z'], spec['shape'], spec['angle'], spec['width'],
               spec['length'], level=spec['level'])


def test_engine_answers_with_doors_moved(tables):
    scenarios = tables.meta['door_samples']
    assert len(scenarios) >= 10 and sum(len(s['queries']) for s in scenarios) >= 1000
    for scenario in scenarios:
        world = World(tables)
        for i in scenario['removed']:
            world.loc_collision(world.locs[i], False)
        for spec in scenario['added']:
            world.loc_collision(loc_from(spec, len(world.locs)), True)
        assert world.changed
        assert {q['kind'] for q in scenario['queries']} <= set(EXPECTED)
        for q in scenario['queries']:
            assert answer(world, scenario['level'], q) == q[EXPECTED[q['kind']]], q


needs_c = pytest.mark.skipif(paths._C is None, reason='the C port is off (SIM_PATHS=python) or not built')
REF = paths.IMPLS['python']
FLAGS = (0, paths.LOC, paths.WALL_NORTH, paths.WALL_EAST, paths.WALL_SOUTH, paths.WALL_WEST, paths.WALL_NORTH | paths.WALL_EAST)


def random_tree(seed, bounded, blocked):
    """A collision grid of random walls and blocked tiles, its step masks, a source and both trees from it."""
    rng = np.random.default_rng(seed)
    size = (int(rng.integers(3, 48)), int(rng.integers(3, 48))) if bounded else (int(rng.integers(130, 160)),) * 2
    flags = np.where(rng.random(size) < blocked, rng.choice(FLAGS, size), 0)
    steps = paths.step_masks(flags)
    # the window of an unbounded grid must lie inside it: HALF <= s <= size - HALF
    sx, sz = (int(rng.integers(size[k])) if bounded else int(rng.integers(paths.HALF, size[k] - paths.HALF + 1)) for k in (0, 1))
    x0, z0 = (0, 0) if bounded else (int(rng.integers(-500, 500)), int(rng.integers(-500, 500)))
    return rng, steps, x0, z0, sx + x0, sz + z0


@needs_c
@settings(deadline=None, max_examples=150)
@given(st.integers(0, 2 ** 32 - 1), st.booleans(), st.floats(0, 0.7))
def test_the_c_search_fills_the_reference_tree(seed, bounded, blocked):
    _, steps, x0, z0, sx, sz = random_tree(seed, bounded, blocked)
    ref, ours = REF['search'](steps, x0, z0, sx, sz, bounded), paths._c_search(steps, x0, z0, sx, sz, bounded)
    for name in ('parents', 'orders', 'dists'):
        assert np.array_equal(getattr(ref, name), getattr(ours, name)), name
        assert getattr(ref, name).dtype == getattr(ours, name).dtype
    assert (ref.order_r, ref.dist_r, ref.none) == (ours.order_r, ours.dist_r, ours.none)


@needs_c
@settings(deadline=None, max_examples=150)
@given(st.integers(0, 2 ** 32 - 1), st.booleans(), st.floats(0, 0.7))
def test_the_c_end_tile_and_backtrack_answer_as_the_reference(seed, bounded, blocked):
    rng, steps, x0, z0, sx, sz = random_tree(seed, bounded, blocked)
    tree = REF['search'](steps, x0, z0, sx, sz, bounded)
    for _ in range(20):
        dest = (sx + int(rng.integers(-40, 41)), sz + int(rng.integers(-40, 41)))
        rect = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        reaching = [(dest[0] + int(rng.integers(-2, 3)), dest[1] + int(rng.integers(-2, 3))) for _ in range(int(rng.integers(0, 6)))]
        end = REF['end_tile'](tree, sx, sz, reaching, dest, rect)
        assert paths._c_end_tile(tree, sx, sz, reaching, dest, rect) == end
        assert paths._c_backtrack(tree, sx, sz, end) == REF['backtrack'](tree, sx, sz, end)


def outcome(fn, *args):
    try:
        return fn(*args)
    except paths.OutsideTables as error:
        return ('OutsideTables', str(error))


@needs_c
def test_the_c_end_tile_on_stored_trees_answers_and_refuses_as_the_reference(tables):
    """The boot map's stored trees (their own order and distance radii): the same end tiles, and the same refusals
    for tiles beyond the stored tables."""
    rng = np.random.default_rng(2)
    for src in rng.choice(len(tables.meta['sources']), 200, replace=False):
        sx, sz, _ = tables.meta['sources'][src]
        tree = paths.Tree.stored(tables, src)
        for _ in range(10):
            dest = (sx + int(rng.integers(-70, 71)), sz + int(rng.integers(-70, 71)))
            reaching = [(dest[0] + int(rng.integers(-1, 2)), dest[1] + int(rng.integers(-1, 2))) for _ in range(3)]
            end = outcome(REF['end_tile'], tree, sx, sz, reaching, dest, (1, 1))
            assert outcome(paths._c_end_tile, tree, sx, sz, reaching, dest, (1, 1)) == end
            if not isinstance(end, tuple) or end[0] != 'OutsideTables':
                assert paths._c_backtrack(tree, sx, sz, end) == REF['backtrack'](tree, sx, sz, end)
