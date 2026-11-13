"""search-wood-v1 (sim/search.py) and its declared world (sim/world.py SEARCH_WORLD): hidden trees at every search start,
keyed starts, a twin world whose hidden trees differ and whose first observation is checked, the v4 batch's wood
slots, and supply-wood-v1 (goal-activity-v5)."""
import numpy as np
import pytest

from obs import project
from sim import search
from sim.world import ORIGIN, SEARCH_WORLD, LayoutTables, generate, load_pool, relocate, save_pool, world_of


@pytest.fixture(scope='module')
def found(tables):
    """The first three development search layouts."""
    return search.search_layouts('development', 3, tables)[0]


def local(tile):
    return tile[0] - ORIGIN[0], tile[1] - ORIGIN[1]


def test_the_search_world_is_the_whole_chunk_and_a_pool_of_it_round_trips(tables, found, tmp_path):
    again = generate(found[0].seed, tables, SEARCH_WORLD)
    assert all(np.array_equal(a, b) for a, b in zip(found[0].arrays().values(), again.arrays().values()))
    assert found[0].flags.shape == SEARCH_WORLD['size'] and world_of(found[0]) is SEARCH_WORLD
    layouts, derived = load_pool(save_pool(tmp_path / 'pool.npz', found))
    assert derived == [None] * 3 and all(np.array_equal(a.flags, b.flags) for a, b in zip(layouts, found))
    t = LayoutTables(tables, found[0])
    assert t.meta['data']['x1'] - t.meta['data']['x0'] + 1 == SEARCH_WORLD['size'][0]


def test_every_search_tile_hides_every_tree_beyond_one_compass_move(tables, found):
    for layout in found:
        tiles = search.search_tiles(layout, tables)
        trees = layout.locs[search.tree_locs(layout, tables), 1:3]
        for x, z in np.argwhere(tiles):
            assert np.maximum(abs(trees[:, 0] - x), abs(trees[:, 1] - z)).min() > search.SEARCH['hidden'] > 32
            assert layout.goal_dist[0][x, z] >= 0


def test_a_search_start_is_keyed_keeps_the_other_draws_and_shows_no_tree(tables, found):
    from sim.task import draw_start
    tiles = [search.search_tiles(layout, tables) for layout in found]
    rec = search.draw_search(13, 0, 4, found, tiles)
    assert rec == search.draw_search(13, 0, 4, found, tiles)
    plain = draw_start(13, 0, 4, found)
    assert {k: rec[k] for k in plain if k != 'tile'} == {k: v for k, v in plain.items() if k != 'tile'}
    assert tiles[rec['layout']][local(rec['tile'])]
    layout = found[rec['layout']]
    task = search.episode(rec, layout, LayoutTables(tables, layout), tables)
    assert search.audit_start(task.raw) == [] and task.horizon == search.SEARCH['horizon']


def test_relocating_a_tree_and_back_gives_the_layout_again(tables, found):
    layout = found[0]
    k = search.tree_locs(layout, tables)[0]
    there = relocate(layout, tables, {k: (int(layout.locs[k, 1]) + 1, int(layout.locs[k, 2]))})
    if there is None:
        pytest.skip('the neighbouring tile does not hold a tree')
    back = relocate(there, tables, {k: tuple(int(v) for v in layout.locs[k, 1:3])})
    assert all(np.array_equal(a, b) for a, b in zip(layout.arrays().values(), back.arrays().values()))


def test_a_twin_world_hides_its_trees_elsewhere_behind_the_same_first_observation(tables, found):
    for k, layout in enumerate(found):
        rec = search.draw_search(13, 0, k, {k: layout}, {k: search.search_tiles(layout, tables)}, layout=k)
        other = search.twin(layout, tables, local(rec['tile']))
        if other is None:
            continue
        a, b = (project(search.episode(rec, lay, LayoutTables(tables, lay), tables).raw) for lay in (layout, other))
        assert all(np.array_equal(a[n], b[n]) for n in a)
        rows = search.tree_locs(layout, tables)
        assert abs(other.locs[rows, 1:3] - layout.locs[rows, 1:3]).max() >= 12
        return
    pytest.skip('no twin among the first layouts')


def test_the_oracle_finds_and_chops_a_hidden_tree_and_wait_does_not(tables, found):
    layout = found[0]
    rec = search.draw_search(13, 0, 0, {0: layout}, {0: search.search_tiles(layout, tables)}, layout=0)
    results = {}
    for policy in ('oracle', 'wait'):
        task = search.episode(rec, layout, LayoutTables(tables, layout), tables)
        rows, _, info, tracker = search.play(task, search.actor_for(policy, {}), 0)
        results[policy] = (info['episode']['r'], tracker.discovered)
    assert results['oracle'][0] == 1.0 and results['oracle'][1] is not None
    assert results['wait'] == (0.0, None)


def test_the_v4_batch_plays_search_starts_for_wood_and_v3_for_the_rest(tables, found):
    from sim.task import draw_start
    from sim.vec import EPISODE_FACTS, CpuBatch
    from sim.world import split_seeds
    pool = [generate(s, tables) for s in split_seeds('development', 2)]
    env = CpuBatch(tables, pool, 4, task='goal-activity-v4', search=(found, None))
    state, obs = env.reset(5)
    assert env.starts[0]['search'] and env.tasks[0].horizon == search.SEARCH['horizon']
    assert search.search_tiles(found[env.starts[0]['layout']], tables)[local(env.starts[0]['tile'])]
    for b in (1, 2):  # fish and mine: v3's starts on the same key
        assert env.starts[b] == draw_start(5, b, 0, pool)
    with pytest.raises(ValueError, match='search pool'):
        CpuBatch(tables, pool, 4, task='goal-activity-v3', search=(found, None))
    rng = np.random.default_rng(0)
    for _ in range(400):
        action = np.asarray([rng.choice(np.flatnonzero(obs['mask'][b])) for b in range(4)], np.int32)
        tracker = env.trackers[0]
        state, obs, reward, done, info = env.step(state, action)
        if done[0]:
            facts = dict(zip(EPISODE_FACTS, info['facts'][0].tolist()))
            assert facts['search'] == 1 and info['episode_ticks'][0] == search.SEARCH['horizon']
            assert facts['discovered'] == int(tracker.discovered is not None)
            assert facts['discovery_ticks'] == (-1 if tracker.discovered is None else tracker.discovered)
            return
    pytest.fail('the search episode did not end')


def test_evaluate_s_command_line_passes_the_sampling_seed_through(monkeypatch, tmp_path):
    """a sampled pass declares its seed base on the command line; the default is the frozen one."""
    from sim.data import Tables
    seen = {}
    monkeypatch.setattr(Tables, 'load', classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr(search, 'evaluate', lambda output, **kw: seen.update(kw) or {})
    monkeypatch.setattr('sys.argv', ['sim.search', 'evaluate', '--checkpoint', 'final.zip', '--policies', 'final', '--sampled',
                                     '--seed', '909001', '--output', str(tmp_path)])
    search.main()
    assert seen['sampled'] is True and seen['seed'] == 909001
    monkeypatch.setattr('sys.argv', ['sim.search', 'evaluate', '--checkpoint', 'final.zip', '--policies', 'final', '--output', str(tmp_path)])
    search.main()
    assert seen['sampled'] is False and seen['seed'] == 404001


# ---- supply-wood-v1 (goal-activity-v5) ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def supply(tables):
    layouts, _ = search.supply_layouts('development', 3, tables)
    return layouts, [search.supply_sites(layout, tables) for layout in layouts]


def test_a_scarce_start_keeps_one_normal_tree_of_its_grove_and_nothing_else(tables, supply):
    layouts, sites = supply
    for k, layout in enumerate(layouts):
        ample = search.draw_supply(5, 0, k, layouts, sites, layout=k, variant='ample')
        scarce = dict(ample, variant='scarce')
        assert {key: v for key, v in ample.items() if key != 'variant'} == {key: v for key, v in scarce.items() if key != 'variant'}
        thinned = search.variant_layout(layout, tables, scarce)
        grove = search.groves(layout, tables)[ample['grove']]
        tile = (ample['tile'][0] - ORIGIN[0], ample['tile'][1] - ORIGIN[1])
        removed, keep = search.scarce_rows(layout, tables, grove, tile)
        assert search.is_normal(layout, tables, keep) and set(removed) | {keep} == set(grove)
        expected = sorted(tuple(int(v) for v in row) for j, row in enumerate(layout.locs) if j not in removed)
        assert sorted(tuple(int(v) for v in row) for row in thinned.locs) == expected
        assert np.array_equal(thinned.npcs, layout.npcs) and np.array_equal(thinned.fish, layout.fish)
        assert search.outside(thinned)[tile]  # the start tile is still in the start component
        far = [r for j, g in enumerate(search.groves(layout, tables)) if j != ample['grove'] for r in g]
        assert min(max(abs(int(layout.locs[r, 1]) - tile[0]), abs(int(layout.locs[r, 2]) - tile[1])) for r in far) > \
            search.SUPPLY['apart']


def test_the_v5_batch_plays_supply_starts_with_the_frozen_horizon_cap_and_quota(tables):
    from pathlib import Path
    from sim.spec import TERMINAL_DEATH
    from sim.vec import EPISODE_FACTS, CpuBatch
    root = Path(__file__).resolve().parents[2]
    layouts, derived = load_pool(root / 'logs/sim/world/train-8-nc1-derived.npz')
    env = CpuBatch(tables, layouts, 8, derived=derived, task='goal-activity-v5', search=(search.supply_layouts('train', 4, tables)[0], None))
    state, obs = env.reset(3)
    rng, seen = np.random.default_rng(1), []
    for _ in range(3000):
        action = np.asarray([rng.choice(np.flatnonzero(obs['mask'][b])) for b in range(8)], np.int32)
        state, obs, reward, done, info = env.step(state, action)
        for b in np.flatnonzero(done):
            if env.goals[b] == 'wood':
                facts = dict(zip(EPISODE_FACTS, info['facts'][b].tolist()))
                seen.append(facts['supply'])
                assert info['episode_ticks'][b] == search.SUPPLY['horizon'] and facts['search'] == 0
                logs, survived = int(info['outcomes'][b]), info['terminal'][b] != TERMINAL_DEATH
                assert info['episode_return'][b] == pytest.approx(min(logs, search.SUPPLY['cap']) / search.SUPPLY['cap'] * survived)
                assert facts['success'] == int(survived and logs >= search.SUPPLY['quota'])
        if len(seen) >= 8 and {1, 2} <= set(seen):
            return
    pytest.fail(f'supply episodes seen: {seen}')
