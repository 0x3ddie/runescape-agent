"""sim/workers.py: CPU reference workers behind shared buffers draw and step as one CpuBatch of the whole batch, so a
world's goal, starts, episode sequence and memory do not depend on the worker that holds it; a worker's error reaches
the caller; the training entry point trains on them with every rollout check; and a single-goal set holds its
goal in every slot while the four-goal path stays as it was."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from baseline.resource_task import GOAL_VEC, GOALS
from sim import search, world as W
from sim.engine import Rng
from sim.vec import CpuBatch
from sim.workers import CpuWorkers

ROOT = Path(__file__).resolve().parents[2]
# Keyed random legal play of the four-goal path under the suite's synthetic name table, recorded before goal
# sets existed (digest below): the v2 workers run of test_workers_draw_and_step_as_one_cpu_batch and a v5 batch on the
# private training pool. A deliberate mechanics change records them again: act-v3 (native-click, 3 November 2026;
# act-v2's were e9aed8f3... and 6e1d1f3b...); held-v2 changed the v5 batch only (9 November 2026; act-v3's was
# fdd20b21...); mask-v5 and obs-v5's six-entry goal_vec changed both (9 November 2026; held-v2's were fbe743d2... and
# bb7d6161...); held-v2's two chat lines after a use with no script (11 such uses in the v5 run) changed both again
# (9 November 2026; obs-v5's were f62c3f61... and f7eb72e5...).
FOUR_GOAL_DIGESTS = dict(v2_workers='8873728a2135a138263b2997f30f3d653f4147743c917ccb11d74d1bb05e135d',
                         v5_batch='4773f3700c3019d3f34593a1a8f3a7d6f977a09b92b0e7142c4fe6ff839fb582')


@pytest.fixture(scope='module')
def pool_file(tables, tmp_path_factory):
    layouts = [W.generate(seed, tables) for seed in W.split_seeds('development', 6)]
    path = tmp_path_factory.mktemp('pool') / 'pool.npz'
    W.save_pool(path, layouts)
    return path, layouts


def run(env, key, steps):
    """Keyed random legal play, every returned array copied (the workers' arrays are reused by the next step)."""
    copy = lambda d: {k: np.array(v) for k, v in d.items()}
    state, obs = env.reset(key)
    rows, rng = [copy(obs)], Rng(key)
    for step in range(steps):
        legal = [np.flatnonzero(obs['mask'][b]) for b in range(len(obs['mask']))]
        action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(len(legal))], np.int32)
        state, obs, reward, done, info = env.step(state, action)
        rows.append(dict(obs=copy(obs), reward=np.array(reward), done=np.array(done), info=copy(info)))
    return rows


def flat(row):
    return {f'{k}/{n}': v for k, part in row.items() for n, v in (part.items() if isinstance(part, dict) else [('', part)])}


def digest(rows):
    """sha256 of a run(): every array's name, dtype, shape and bytes, step by step."""
    h = hashlib.sha256()

    def put(arrays):
        for k, v in sorted(arrays.items()):
            v = np.ascontiguousarray(v)
            h.update(k.encode() + str(v.dtype).encode() + str(v.shape).encode() + v.tobytes())
    put(rows[0])
    for row in rows[1:]:
        put(row['obs'])
        put(row['info'])
        put(dict(reward=row['reward'], done=row['done']))
    return h.hexdigest()


@pytest.fixture(scope='module')
def v5_inputs(tables):
    path = ROOT / 'logs/sim/world/train-8-nc1-derived.npz'
    if not path.exists():
        pytest.skip('training pool absent (docs/RUNNING.md)')
    layouts, derived = W.load_pool(path)
    return layouts, derived, (search.supply_layouts('train', 4, tables)[0], None)


def test_workers_draw_and_step_as_one_cpu_batch(tables, pool_file):
    path, layouts = pool_file
    one = run(CpuBatch(tables, layouts, 8, horizon=40), 5, 50)
    workers = CpuWorkers(8, 2, path, horizon=40)
    try:
        two = run(workers, 5, 50)
        timing = workers.timing()
    finally:
        workers.close()
    assert len(one) == len(two)
    for step, (a, b) in enumerate(zip(one, two)):
        a, b = flat(a) if step else a, flat(b) if step else b
        assert a.keys() == b.keys(), step
        for k in a:
            assert np.array_equal(a[k], b[k]), (step, k)
    ended = sum(int(r['done'].sum()) for r in one[1:])
    assert ended >= 8 and not any(r['info']['unsupported'].any() for r in one[1:])
    assert one[1]['info']['goal'].tolist() == [GOALS.index(GOALS[b % 4]) for b in range(8)]
    assert sum(int(r['info']['made'].sum()) for r in one[1:]) > 0  # the game effects are reported, not zeros
    assert timing['steps'] == 50 and timing['workers'] == 2 and 0 <= timing['waiting_share'] < 1


def test_a_worker_error_reaches_the_caller_and_the_workers_still_close(pool_file):
    path, _ = pool_file
    workers = CpuWorkers(8, 2, path, horizon=40)
    try:
        state, obs = workers.reset(1)
        with pytest.raises(RuntimeError, match='worker for slots'):
            workers.step(state, np.full(8, 10 ** 6, np.int32))  # no such action
    finally:
        workers.close()
    assert all(not p.is_alive() for p in workers._processes)


def test_the_training_entry_point_trains_on_cpu_workers_with_every_rollout_check(pool_file, tmp_path):
    from sim.train import train
    path, _ = pool_file
    summary = train(tmp_path / 'run', decisions=2 * 8 * 8, seed=1, worlds=8, n_steps=8, batch_size=16, n_epochs=1,
                    env='cpu', workers=2, pool=path, validate=True, verbose=False)
    assert summary['status'] == 'completed' and summary['gradient_steps'] == 2 * 4
    rows = summary['updates']
    assert all(r['checks']['illegal_actions'] == 0 for r in rows)
    assert all(r['worker_timing']['steps'] == 8 and r['worker_timing']['workers'] == 2 for r in rows)
    assert {'obs/mask', 'actions', 'info/made', 'info/kills'} <= set(summary['first_rollout'])


def test_the_four_goal_path_is_unchanged_for_a_fixed_seed(pool_file):
    path, _ = pool_file
    workers = CpuWorkers(8, 2, path, horizon=40)
    try:
        assert digest(run(workers, 5, 50)) == FOUR_GOAL_DIGESTS['v2_workers']
    finally:
        workers.close()


def test_the_four_goal_v5_batch_is_unchanged_for_a_fixed_seed(tables, v5_inputs):
    layouts, derived, supply = v5_inputs
    env = CpuBatch(tables, layouts, 8, derived=derived, task='goal-activity-v5', search=supply)
    assert digest(run(env, 3, 60)) == FOUR_GOAL_DIGESTS['v5_batch']


def test_a_specialist_batch_asks_for_its_goal_in_every_slot_with_that_goals_starts(tables, v5_inputs):
    layouts, derived, supply = v5_inputs
    for goal in GOALS:
        env = CpuBatch(tables, layouts, 4, derived=derived, task='goal-activity-v5', search=supply, goals=(goal,))
        rows = run(env, 3, 40)
        assert all(r['obs']['goal_vec'].argmax(1).tolist() == [GOALS.index(goal)] * 4 for r in rows[1:]), goal
        assert all(r['info']['goal'].tolist() == [GOALS.index(goal)] * 4 for r in rows[1:]), goal
        assert (rows[0]['goal_vec'] == np.eye(len(GOAL_VEC), dtype=np.float32)[GOALS.index(goal)]).all()
        # v5's wood slots play supply starts and its melee slots sustained combat, whatever slot they hold
        assert [bool(s.get('supply')) for s in env.starts] == [goal == 'wood'] * 4, goal
        assert ['combat' in s for s in env.starts] == [goal == 'melee'] * 4, goal
    with pytest.raises(ValueError, match='goal set'):
        CpuBatch(tables, layouts, 4, derived=derived, goals=('wood', 'fish'))


def test_the_training_entry_point_trains_a_specialist_and_validation_holds_each_slot_to_the_declared_set(pool_file,
                                                                                                       tmp_path):
    import torch
    from sim.learn import build
    from sim.train import CheckFailed, make_env, task_limits, train, validate_rollout
    path, _ = pool_file
    summary = train(tmp_path / 'run', decisions=2 * 8 * 8, seed=1, worlds=8, n_steps=8, batch_size=16, n_epochs=1,
                    env='cpu', workers=2, pool=path, validate=True, verbose=False, goals=('mine',),
                    segments=dict(worlds=4, decisions=8))
    assert summary['status'] == 'completed' and summary['goals'] == ['mine']
    assert {g: v['transitions'] for g, v in summary['per_goal'].items()} == dict(wood=0, fish=0, mine=2 * 8 * 8, melee=0)
    assert all(r['checks']['transitions_per_goal'] == [0, 0, 8 * 8, 0] for r in summary['updates'])
    assert summary['segments']['slots'] == [0, 2, 4, 6] and summary['segments']['updates'] == 2
    for row in summary['updates']:  # the simulator's own observations, replayed under the collecting policy
        with np.load(tmp_path / 'run' / row['segments']['path']) as f:
            assert (f['info__goal'] == GOALS.index('mine')).all() and (f['obs__goal_vec'].argmax(-1) == 2).all()
    env = make_env('cpu', 8, 1, 'cpu', path, goals=('fish',), workers=2)
    try:
        model = build(env, seed=1, device='cpu', n_steps=8, batch_size=16, n_epochs=1)
        _, callback = model._setup_learn(64, None)
        model.collect_rollouts(env, callback, model.rollout_buffer, 8)
        carried = lambda: dict(paid=torch.zeros(8), done=torch.ones(8))
        assert validate_rollout(model, carried(), *task_limits(env.env), goals=('fish',))['transitions_per_goal'] == [0, 64, 0, 0]
        for declared in (GOALS, ('wood',)):
            with pytest.raises(CheckFailed, match='outside the declared set'):
                validate_rollout(model, carried(), *task_limits(env.env), goals=declared)
    finally:
        env.close()
