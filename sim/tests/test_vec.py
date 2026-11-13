"""The learners' environment adapters: an unsupported step never becomes a transition, and CpuBatch
holds sim/spec.py's batched interface for the four-goal task."""
import numpy as np
import pytest

from baseline.resource_task import GOALS
from sim import world as W
from sim.engine import Rng, Unsupported
from sim.task import draw_start
from sim.vec import CpuBatch


def uncovered(self, action):
    raise Unsupported('engine script x')


@pytest.fixture(scope='module')
def pool(tables):
    return [W.generate(seed, tables) for seed in W.split_seeds('development', 6)]


def play(env, key, steps):
    state, obs = env.reset(key)
    rng, rows = Rng(key), []
    for step in range(steps):
        legal = [np.flatnonzero(obs['mask'][b]) for b in range(env.n)]
        action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(env.n)])
        state, obs, reward, done, info = env.step(state, action)
        rows.append((obs, reward, done, info))
    return rows


def test_the_cpu_batch_holds_the_spec_and_settles_each_episode_to_its_audit(tables, pool):
    env = CpuBatch(tables, pool, 8, horizon=40)
    paid, ended = np.zeros(8), 0
    for obs, reward, done, info in play(env, 3, 60):
        assert info['goal'].tolist() == [GOALS.index(GOALS[b % 4]) for b in range(8)] and not info['unsupported'].any()
        paid += reward
        for b in np.flatnonzero(done):
            assert abs(paid[b] - info['episode_return'][b]) < 1e-6 and info['terminal'][b] == 1
            assert info['episode_ticks'][b] == 40 and obs['budget_vec'][b, 0] == 1.0  # the next episode's first
            assert obs['goal_vec'][b].argmax() == b % 4
            paid[b], ended = 0, ended + 1
    assert ended >= 8


def test_starts_are_keyed_draws_that_do_not_see_the_goal(tables, pool):
    a, b = draw_start(5, 3, 0, pool), draw_start(5, 3, 0, pool)
    assert a == b and a != draw_start(5, 3, 1, pool) and a != draw_start(6, 3, 0, pool)
    layout = pool[a['layout']]
    assert layout.start[a['tile'][0] - W.ORIGIN[0], a['tile'][1] - W.ORIGIN[1]]
    first, second = (play(CpuBatch(tables, pool, 4, horizon=30), 11, 12) for _ in range(2))
    for (o1, r1, d1, _), (o2, r2, d2, _) in zip(first, second):
        assert all(np.array_equal(o1[k], o2[k]) for k in o1) and np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_the_path_window_bound_of_a_pool_without_derived_tables_changes_no_output(tables, pool):
    """Native gap Stage 3 (R13): a pool without derived tables keeps 64 path windows a layout; the cache is a memo."""
    # the training workers pass sim.world load_pool's list, one None a layout (sim/workers.py)
    bounded, wide = (CpuBatch(tables, pool, 4, horizon=30, derived=[None] * len(pool)) for _ in range(2))
    assert {lt.CACHE for lt in bounded.layout_tables} == {64}
    assert {lt.CACHE for lt in CpuBatch(tables, pool, 4, horizon=30).layout_tables} == {64}
    for lt in wide.layout_tables:
        lt.CACHE = 1  # every window computed again when asked
    for (o1, r1, d1, _), (o2, r2, d2, _) in zip(play(bounded, 11, 40), play(wide, 11, 40)):
        assert all(np.array_equal(o1[k], o2[k]) for k in o1) and np.array_equal(r1, r2) and np.array_equal(d1, d2)
    lt = wide.layout_tables[0]  # both caches (boot map and snapshot), asked twice, against fresh windows
    for _ in range(2):
        for src in range(0, len(lt.meta['sources']), 7):
            for snapshot in (False, True):
                got, fresh = lt.window(src, snapshot), W._window(*lt.rows(src, snapshot), *lt.local(src))
                assert all(np.array_equal(getattr(got, k), getattr(fresh, k)) for k in ('parents', 'orders', 'dists'))


def test_an_unsupported_world_is_flagged_and_refuses_to_step_again(tables, pool, monkeypatch):
    env = CpuBatch(tables, pool, 4, horizon=40)
    state, obs = env.reset(1)
    monkeypatch.setattr(env.tasks[2].agent, 'step', lambda action: uncovered(None, action))
    state, obs, reward, done, info = env.step(state, np.zeros(4, np.int32))
    assert info['unsupported'].tolist() == [False, False, True, False] and not done[2] and reward[2] == 0
    assert env.evidence[0]['world'] == 2 and env.evidence[0]['reason'] == 'engine script x'
    with pytest.raises(RuntimeError):
        env.step(state, np.zeros(4, np.int32))


def test_a_v11_batch_refuses_another_pool_or_goal_set(tables, pool):
    """CpuBatch goal-activity-v11 refuses every goal set but mine, and the 8-layout pool's training seeds 0-7 or the
    development layouts, before it builds a layout's tables; v5 takes the same layouts and goals."""
    from sim import search
    supply = (search.supply_layouts('train', 1, tables)[0], None)
    train8 = [W.generate(seed, tables) for seed in W.split_seeds('train', 8)]
    for layouts in (train8, pool):
        with pytest.raises(ValueError, match='plays compact-world-v1 train seeds 1000-1127 in order'):
            CpuBatch(tables, layouts, 4, task='goal-activity-v11', search=supply, goals=('mine',))
        for goals in (GOALS, ('fish',)):
            with pytest.raises(ValueError, match=r"plays goals \['mine'\]"):
                CpuBatch(tables, layouts, 4, task='goal-activity-v11', search=supply, goals=goals)
        CpuBatch(tables, layouts, 4, task='goal-activity-v5', search=supply, goals=('mine',))


def test_the_trace_has_every_tick_of_every_world_and_each_episode_start(tables, pool):
    from sim.vec import trace
    rows = []
    trace(CpuBatch(tables, pool, 2, horizon=20), 4, 12, rows.append)
    decisions = [r for r in rows if r['kind'] == 'decision']
    for b in range(2):
        ticks = [r for r in rows if r['kind'] == 'tick' and r['world'] == b]
        assert len(ticks) == sum(d['info']['decision_ticks'][b] for d in decisions)
        starts = [r for r in rows if r['kind'] == 'start' and r['world'] == b]
        assert len(starts) == 1 + sum(d['done'][b] for d in decisions) and starts[0]['state']['tick'] in (12, 13)
