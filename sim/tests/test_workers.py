"""sim/workers.py: CPU reference workers behind shared buffers draw and step as one CpuBatch of the whole batch, so a
world's goal, starts, episode sequence and memory do not depend on the worker that holds it; a worker's error reaches
the caller; the training entry point trains on them with every rollout check, for the goal tasks and through the
command for free play (cut lives that bootstrap, deaths that do not), where the worker count changes no result, the C
engine collects what the Python reference collects and a start-only run records a real run's start digest; and a
single-goal set holds its goal in
every slot while the four-goal path stays as it was."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from baseline.resource_task import GOAL_VEC, GOALS
from sim import search, world as W
from sim.engine import Rng
from sim.vec import CpuBatch
from sim.workers import FREE_PLAY_TASK, CpuWorkers

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


def free_play_command(tmp_path, monkeypatch, path, *settings, n_steps=64, updates=2, options=(), world='village-world-v1'):
    """sim.train train --task free-play-v1 on the real batch on two CPU workers with --validate; returns its summary.
    settings: the batch settings beside the world it names (`world`; None: none). options: further flags, after the
    others (a later --worlds, --workers or --seed overrides)."""
    import json
    import sys
    from sim.train import main
    monkeypatch.setattr(sys, 'argv', ['sim.train', 'train', '--task', 'free-play-v1', '--env', 'cpu', '--workers', '2',
                                      '--pool', str(path), '--worlds', '8', '--n-steps', str(n_steps), '--batch-size', '128',
                                      '--epochs', '1', '--decisions', str(updates * 8 * n_steps), '--seed', '1', '--validate',
                                      '--batch-setting', *settings, *([f'world="{world}"'] if world else []),
                                      '--output', str(tmp_path / 'run'), *options])
    main()
    return json.loads((tmp_path / 'run' / 'summary.json').read_text())


def test_the_training_entry_point_trains_free_play_on_cpu_workers_with_every_rollout_check(pool_file, tmp_path, monkeypatch):
    """The free-play twin through the command: lives of 200 ticks without muggers end inside the run; the batch's
    settings in effect are recorded; every rollout passes validate_free_play (each life's pay equals its episode_return,
    bootstrap values only at cut lives); the cut lives bootstrap; the lives' facts reach the training evidence; and the
    checkpoint loads as an obs-v6 policy whose digest is the summary's policy_sha256_at_end."""
    import json
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    from sim.task import free_play_cfg
    from sim.train import policy_digest
    from sim.vec import COMBAT_FACTS
    summary = free_play_command(tmp_path, monkeypatch, pool_file[0], 'muggers=false', 'life=["fixed",200]')
    assert summary['status'] == 'completed' and summary['goals'] == [] and summary['per_goal'] is None
    free = summary['free_play']
    assert free['observation'] == 'obs-v6' and free['batch'] == dict(muggers=False, life=['fixed', 200], world='village-world-v1')
    assert free['batch_in_effect'] == dict(world='village-world-v1', observation='obs-v6', xp_rate=1, muggers=False,
                                           life=['fixed', 200], cfg=json.loads(json.dumps(free_play_cfg())))
    with pytest.raises(ValueError, match=r"names muggers, life, world \(--batch-setting\): \['world'\] missing"):
        free_play_command(tmp_path / 'unnamed', monkeypatch, pool_file[0], 'muggers=false', 'life=["fixed",200]', world=None)
    with pytest.raises(RuntimeError, match="the pool holds village-world-v1 layouts, not 'village-world-v2'"):  # a worker's
        free_play_command(tmp_path / 'other', monkeypatch, pool_file[0], 'muggers=false', 'life=["fixed",200]',
                          world='village-world-v2')
    checks = summary['checks']['rollouts']
    assert len(checks) == 2 and all(c['problems'] == [] and c['illegal_actions'] == 0 for c in checks)
    assert all(c['decisions'] == 8 * 64 and c['native_ticks'] >= 2 * c['decisions'] for c in checks)
    lives = summary['lives']
    assert lives['cut'] == sum(c['cut_lives'] for c in checks) >= 8 and lives['deaths'] == sum(c['deaths'] for c in checks)
    assert lives['lives'] == lives['cut'] + lives['deaths'] and lives['mean_ticks'] >= 200
    assert any(c['cut_lives'] and c['cut_value_sum'] != 0 for c in checks) and lives['mean_cut_value'] != 0  # bootstrapped
    facts = [r['evidence']['life_facts'] for r in summary['updates']]
    assert all(set(f) == {*COMBAT_FACTS, 'cells'} for f in facts)
    assert sum(f['cells']['free-play-v1'] for f in facts) >= lives['lives']  # each life stood in a cell
    assert all(r['worker_timing']['workers'] == 2 for r in summary['updates'])
    loaded = RecurrentPPO.load(tmp_path / 'run' / 'latest.zip', device='cpu').policy
    check_names(loaded, observation='obs-v6')
    assert summary['policy_sha256_at_end'] == policy_digest(loaded) and summary['checkpoint_digests'] == {}  # none in 2


def test_a_death_on_the_real_free_play_batch_ends_its_life_without_a_bootstrap(pool_file, tmp_path, monkeypatch):
    """Muggers at the XP rate of 25 in long lives: lives end by death inside the run, and every rollout passes
    validate_free_play, which refuses a bootstrap value at a death."""
    summary = free_play_command(tmp_path, monkeypatch, pool_file[0], 'muggers=true', 'xp_rate=25', 'life=["fixed",6000]',
                                n_steps=128, updates=3)
    assert summary['status'] == 'completed' and summary['free_play']['batch_in_effect']['muggers'] is True
    checks = summary['checks']['rollouts']
    assert all(c['problems'] == [] for c in checks) and summary['lives']['deaths'] == sum(c['deaths'] for c in checks) >= 1
    assert summary['lives']['cut'] == 0


def test_the_worker_count_changes_no_transition_and_no_update_of_a_free_play_run(pool_file, tmp_path, monkeypatch):
    """The same seed on 1, 2 and 4 CPU workers, with muggers that attack and lives cut by their length: the first
    rollout's digests (observations, actions, rewards, ends, values, log-probabilities and info), each update's evidence,
    the lives and the policy after two updates are the same bit for bit, so the worker count is not a setting of a run's
    results (each worker steps its own slots of one CpuBatch, keyed by slot)."""
    counts = (1, 2, 4)
    runs = [free_play_command(tmp_path / str(w), monkeypatch, pool_file[0], 'muggers=true', 'xp_rate=25',
                              'life=["fixed",200]', options=('--workers', str(w))) for w in counts]
    assert all(r['status'] == 'completed' and r['updates'][0]['worker_timing']['workers'] == w for r, w in zip(runs, counts))
    evidence = lambda run: [u['evidence'] for u in run['updates']]
    assert runs[0]['lives']['cut'] >= 8 and all(e['life_facts']['aggressed'][FREE_PLAY_TASK] > 0 for e in evidence(runs[0]))
    for r in runs[1:]:
        assert r['first_rollout'] == runs[0]['first_rollout'] and r['lives'] == runs[0]['lives']
        assert evidence(r) == evidence(runs[0])
        assert r['policy_sha256_at_end'] == runs[0]['policy_sha256_at_end'] != runs[0]['policy_sha256_at_start']


@pytest.fixture(scope='module')
def free_pack(pool_file, name_table_path, tmp_path_factory):
    """The C engine's pack of pool_file's layouts under the suite's name table, exported as
    sim/tests/test_native_free_play.py exports its packs: no route trees (C searches routes live, as the Python reference
    computes them over a pool without derived tables) and one supply layout, which free play does not play."""
    supply = ROOT / 'logs/supply/pools/train-128.npz'
    if not supply.exists():
        pytest.skip('private supply pool absent')
    from names import NameTable
    from sim.build import DATA
    from sim.native_export import export
    work = tmp_path_factory.mktemp('free-pack')
    one = W.save_pool(work / 'supply.npz', W.load_pool(supply)[0][:1], version=search.SUPPLY['world'])
    return export(work / 'free.pack', pool_file[0], one, DATA, names=NameTable(name_table_path), trees=False, variants=1,
                  log=lambda _: None)


def test_free_play_collects_alike_on_the_c_engine_and_on_the_python_reference(pool_file, free_pack, tmp_path, monkeypatch):
    """The training command with the same seed and batch settings (muggers, lives drawn from 100 to 300 ticks, XP rate 1)
    on the Python reference and on the C engine (--engine native) with 1, 2 and 4 workers: the first rollout's digests
    (the observations with their masks, the actions the same policy took, the pay, the life ends, values,
    log-probabilities and info), the lives, each update's checks and evidence and the policy after two updates are the
    same bit for bit, and the settings in effect too. A C run records its engine, its pack with the inputs that the
    pack's manifest names, and the archive that each worker held equal to the pack's: with no --tables, the one the Python
    reference loads (its pins unchecked: C reads only the pack). The C path reads the pack (a pool other than the pack's
    is refused, so equal runs are not the Python reference twice; so is a pack that names another archive, with no
    --tables given), and it refuses a batch setting that the Python reference does not take (NativeBatch's allow_stale
    would pass a stale pack)."""
    from baseline.telemetry import sha256
    from sim.build import DATA
    from sim.native import read_pack
    meta = read_pack(free_pack)[1]
    other = tmp_path / 'other.npz'
    W.save_pool(other, pool_file[1][3:])
    settings = ('muggers=true', 'xp_rate=1', 'life=["uniform",100,300]')
    native = ('--engine', 'native', '--pack', str(free_pack))
    python = free_play_command(tmp_path / 'cpu', monkeypatch, pool_file[0], *settings)
    first = python['checks']['rollouts'][0]
    assert python['engine'] == 'cpu' and first['cut_lives'] + first['deaths'] > 0  # lives end inside the first rollout
    counts = (1, 2, 4)
    for w in counts:
        run = free_play_command(tmp_path / f'c{w}', monkeypatch, pool_file[0], *settings, options=(*native, '--workers', str(w)))
        assert run['status'] == 'completed' and run['updates'][0]['worker_timing']['workers'] == w
        assert run['engine'] == 'native' and run['pack'] == dict(path=str(free_pack), sha256=sha256(free_pack),
                                                                 inputs=meta['inputs'], oracle_hash=meta['oracle_hash'])
        assert run['tables'] == dict(python['tables'], pins_checked=False) == dict(path=str(DATA), sha256=sha256(DATA),
                                                                                   pins_checked=False)
        assert meta['inputs']['tables'] == run['tables']['sha256'] and meta['inputs']['pool'] == run['pool']['sha256']
        assert run['free_play'] == python['free_play']  # the settings in effect among them, the world too
        assert run['free_play']['batch_in_effect']['world'] == 'village-world-v1'
        assert run['first_rollout'] == python['first_rollout'] and run['lives'] == python['lives']
        assert run['checks']['rollouts'] == python['checks']['rollouts']
        assert [u['evidence'] for u in run['updates']] == [u['evidence'] for u in python['updates']]
        assert run['policy_sha256_at_end'] == python['policy_sha256_at_end'] != python['policy_sha256_at_start']
    with pytest.raises(RuntimeError, match='was exported from another pool than'):
        free_play_command(tmp_path / 'other', monkeypatch, other, *settings, options=native)
    forged = tmp_path / 'forged.pack'  # the same pack, but its manifest names another archive (one digit of its sha256)
    raw = bytearray(Path(free_pack).read_bytes())
    at = raw.index(meta['inputs']['tables'].encode())
    raw[at] = ord('0' if raw[at] != ord('0') else '1')
    forged.write_bytes(raw)
    assert read_pack(forged)[1]['inputs']['tables'] != sha256(DATA)
    with pytest.raises(RuntimeError, match=re.escape(f'was exported from another tables than {DATA}')):
        free_play_command(tmp_path / 'forged', monkeypatch, pool_file[0], *settings, options=('--engine', 'native', '--pack',
                                                                                              str(forged)))
    with pytest.raises(RuntimeError, match=r"batch settings that the Python reference does not take: \['allow_stale'\]"):
        free_play_command(tmp_path / 'stale', monkeypatch, pool_file[0], *settings, 'allow_stale=true', options=native)


def test_a_start_only_run_records_the_start_digest_of_a_real_run_of_its_seed(pool_file, tmp_path, monkeypatch):
    """--initial-only builds the batch and the policy, saves initial.zip, records policy_sha256_at_start and stops before
    any rollout. Its digest is the one a real run of the seed records under the same learner settings, though its worlds,
    workers, pool, batch settings and table path (--tables, so its pins go unchecked) differ; another seed's differs. A run
    on the Python reference records the archive it trained on."""
    from sb3_contrib import RecurrentPPO
    from baseline.telemetry import sha256
    from sim.build import DATA
    from sim.train import policy_digest
    path, layouts = pool_file
    real = free_play_command(tmp_path / 'real', monkeypatch, path, 'muggers=false', 'life=["fixed",200]', updates=1)
    assert real['status'] == 'completed' and real['tables'] == dict(path=str(DATA), sha256=sha256(DATA), pins_checked=True)
    other, link = tmp_path / 'other.npz', tmp_path / 'tables.npz'
    W.save_pool(other, layouts[3:])
    link.symlink_to(DATA)
    start = lambda seed: free_play_command(tmp_path / f'start-{seed}', monkeypatch, other, 'muggers=true', 'xp_rate=25',
                                           'life=["fixed",6000]', options=('--worlds', '4', '--workers', '1', '--seed', str(seed),
                                                                           '--tables', str(link), '--initial-only'))
    first, run = start(1), tmp_path / 'start-1' / 'run'
    assert first['status'] == 'initialized' and first['policy_sha256_at_start'] == real['policy_sha256_at_start']
    assert first['learner'] == real['learner'] and first['worlds'] == 4 and 'lives' not in first and 'updates' not in first
    assert policy_digest(RecurrentPPO.load(run / 'initial.zip', device='cpu').policy) == first['policy_sha256_at_start']
    assert not (run / 'latest.zip').exists() and first['tables'] == dict(real['tables'], path=str(link), pins_checked=False)
    assert start(2)['policy_sha256_at_start'] != first['policy_sha256_at_start']


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
