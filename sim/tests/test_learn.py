"""The device learner (sim/learn.py) against SB3-Contrib's own RecurrentPPO path, on CPU.

Declared tolerance, fixed before any comparison was run: every compared quantity must be bitwise equal,
dtype and shape included (torch.equal / np.array_equal, no allclose). Both paths run the same float32 CPU
kernels on the same values in the same order, so exactness is the expected result. Stated exceptions:
- Storage dtype: the device buffer stores ids as int32 where SB3 stores int64, so stored observations are
  compared by value; the minibatches train() receives carry SB3's dtypes and are compared with them.
- Logged time/ values (wall clock) and the device path's extra rollout/ statistics are not compared.
"""
import json
from types import SimpleNamespace
import zipfile

import jax
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger

from baseline import config
from baseline.policy import POPART, MaskedRecurrentPolicy, check_names, load_name_vectors
from names import name_table
from obs import OBSERVATION
from sim.learn import (DeviceEnv, DeviceRecurrentPPO, DeviceRolloutBuffer, FakeEnv, HostReads, HostVecEnv, Unsupported,
                       attach, build)
from sim.spec import TERMINAL_DEATH, TERMINAL_HORIZON

WORLDS, N_STEPS, BATCH, EPOCHS = 4, 8, 12, 2   # 32 transitions: minibatches of 12, 12, 8 per epoch
SETTINGS = dict(n_steps=N_STEPS, batch_size=BATCH, n_epochs=EPOCHS)
CLIP = torch.nn.utils.clip_grad_norm_


def fake(**kwargs):
    return FakeEnv(WORLDS, id_rows=name_table().spec[0], **{'horizon_ticks': 7, 'death_rate': .05, **kwargs})


def reference_model(env, seed):
    """config.LEARNER's RecurrentPPO exactly as baseline/train.build_model makes it, on a host VecEnv."""
    learner = {**config.LEARNER, **SETTINGS, 'policy_kwargs': {**config.LEARNER['policy_kwargs'],
                                                               'names': name_table().spec, 'observation': OBSERVATION}}
    model = RecurrentPPO(MaskedRecurrentPolicy, env, seed=seed, device='cpu', **learner)
    load_name_vectors(model.policy, name_table().vectors)
    return model


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return type(value)(*map(clone, value)) if hasattr(value, '_fields') else tuple(map(clone, value))
    return value


def assert_same(a, b, where='', values_only=False):
    if isinstance(a, dict):
        assert set(a) == set(b), where
        for k in a:
            assert_same(a[k], b[k], f'{where}.{k}', values_only)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{where}[{i}]', values_only)
    elif torch.is_tensor(a):
        assert a.shape == b.shape, f'{where}: {a.shape} != {b.shape}'
        if values_only:
            a, b = a.double(), b.double()
        assert a.dtype == b.dtype, f'{where}: {a.dtype} != {b.dtype}'
        assert torch.equal(a, b), f'{where}: largest difference {(a.double() - b.double()).abs().max()}'
    else:
        assert a == b or (a != a and b != b), f'{where}: {a} != {b}'


class Dumps(KVWriter):
    def __init__(self):
        self.rows = []

    def write(self, key_values, key_excluded, step=0):
        self.rows.append((step, dict(key_values)))

    def close(self):
        pass


def buffer_fields(buffer):
    """The rollout as SB3's buffer holds it before get(): [n_steps, n_envs, ...], hidden states
    [n_steps, layers, n_envs, hidden]; observations in the model's dtypes."""
    if isinstance(buffer, DeviceRolloutBuffer):
        hidden = {k: getattr(buffer, k).transpose(1, 2) for k in
                  ('hidden_states_pi', 'cell_states_pi', 'hidden_states_vf', 'cell_states_vf')}
        obs = {k: v.to(buffer._dtypes[k]) for k, v in buffer.observations.items()}
        fields = dict(actions=buffer.actions, rewards=buffer.rewards, episode_starts=buffer.episode_starts,
                      values=buffer.values, log_probs=buffer.log_probs, advantages=buffer.advantages,
                      returns=buffer.returns)
    else:
        hidden = {k: getattr(buffer, k) for k in ('hidden_states_pi', 'cell_states_pi', 'hidden_states_vf', 'cell_states_vf')}
        obs = buffer.observations
        fields = {k: getattr(buffer, k) for k in ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                                                  'advantages', 'returns')}
    return clone(dict(observations=obs, **fields, **hidden))


def run(model, transitions, monkeypatch, void=False):
    """learn() with every quantity the comparison needs recorded, from fixed torch and numpy generators.
    void: an Unsupported rollout is expected and recorded (log.unsupported), not raised."""
    log = SimpleNamespace(forward=[], bootstrap=[], samples=[], evaluated=[], grad_norms=[], buffers=[], params=[])
    policy, buffer = model.policy, model.rollout_buffer
    forward, predict_values, evaluate_actions, get = (policy.forward, policy.predict_values, policy.evaluate_actions,
                                                      buffer.get)

    def record_forward(obs, lstm_states, episode_starts, deterministic=False):
        out = forward(obs, lstm_states, episode_starts, deterministic)
        log.forward.append(clone(dict(obs=obs, states=lstm_states, starts=episode_starts, actions=out[0], values=out[1],
                                      log_prob=out[2], logits=policy._distribution.distribution.logits, next_states=out[3])))
        return out

    def record_values(obs, lstm_states, episode_starts):
        out = predict_values(obs, lstm_states, episode_starts)
        log.bootstrap.append(clone(dict(obs=obs, states=lstm_states, starts=episode_starts, values=out)))
        return out

    def record_evaluate(obs, actions, lstm_states, episode_starts):
        out = evaluate_actions(obs, actions, lstm_states, episode_starts)
        log.evaluated.append(clone(out))
        return out

    def record_get(batch_size=None):
        for sample in get(batch_size):
            log.samples.append(clone(sample))
            yield sample

    def record_clip(parameters, max_norm, *args, **kwargs):
        norm = CLIP(parameters, max_norm, *args, **kwargs)
        log.grad_norms.append(norm.clone())
        return norm
    policy.forward, policy.predict_values, policy.evaluate_actions, buffer.get = (
        record_forward, record_values, record_evaluate, record_get)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', record_clip)

    class Snapshots(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_start(self):
            log.params.append(clone(self.model.policy.state_dict()))

        def _on_rollout_end(self):
            log.buffers.append(buffer_fields(self.model.rollout_buffer))
    dumps = Dumps()
    model.set_logger(Logger(None, [dumps]))
    torch.manual_seed(7)
    np.random.seed(7)
    log.unsupported = None
    try:
        model.learn(transitions, callback=Snapshots())
    except Unsupported as error:
        if not void:
            raise
        log.unsupported = error.evidence['rows']
    log.params.append(clone(policy.state_dict()))
    log.optimizer = clone(policy.optimizer.state_dict()['state'])
    # SB3 dumps an update's losses after the next rollout; the last update's are still pending.
    rows = [*dumps.rows, (model.num_timesteps, dict(model.logger.name_to_value))]
    log.train = [(step, {k: v for k, v in row.items() if k.startswith('train/')}) for step, row in rows]
    return log


def test_device_learner_matches_sb3_collect_and_train_exactly(monkeypatch):
    """Two rollouts and two updates, with resets inside each rollout and at the boundary: the device
    collector and buffer reproduce SB3's collect_rollouts, RecurrentDictRolloutBuffer and train() (stock: the
    fast update makes no evaluate_actions call to compare; test_the_fast_update_equals_the_stock_update)."""
    monkeypatch.setattr(DeviceRecurrentPPO, 'FAST_UPDATE', False)
    seed = 3
    reference = reference_model(HostVecEnv(fake(), seed), seed)
    device = build(DeviceEnv(fake(), seed), seed=seed, device='cpu', **SETTINGS)
    assert_same(dict(device.policy.state_dict()), dict(reference.policy.state_dict()), 'initial parameters')
    expected = run(reference, 2 * WORLDS * N_STEPS, monkeypatch)
    actual = run(device, 2 * WORLDS * N_STEPS, monkeypatch)

    # The data cover what the comparison must exercise.
    starts = torch.stack([row['starts'] for row in expected.forward])          # [decisions, worlds]
    assert starts[1:N_STEPS].any() and starts[N_STEPS + 1:].any(), 'no reset inside a rollout'
    assert starts[N_STEPS].any() and not starts[N_STEPS].all(), 'need a reset at the rollout boundary and a bootstrap'
    terminals = {t for rollout in device.rollouts for t in rollout['episodes']['terminal']}
    assert {TERMINAL_HORIZON, TERMINAL_DEATH} <= terminals
    masks = torch.stack([sample.mask.mean() for sample in expected.samples])
    assert (masks < 1).any() and len(expected.samples) == 2 * EPOCHS * 3, 'need padded minibatches'
    assert int(device.env.state.illegal) == 0

    # Collection: observations, LSTM states in and out (reset where episodes start), actions from the
    # same sampling noise, logits, values, log-probabilities; the bootstrap values at each boundary.
    assert len(actual.forward) == len(expected.forward) == 2 * N_STEPS
    for i, (a, e) in enumerate(zip(actual.forward, expected.forward)):
        assert_same(a, e, f'forward {i}')
    assert_same(actual.bootstrap, expected.bootstrap, 'bootstrap')
    # The stored rollout, its returns and advantages (SB3's GAE, done terminal, boundary bootstrapped).
    for i, (a, e) in enumerate(zip(actual.buffers, expected.buffers)):
        assert_same(a['observations'], e['observations'], f'rollout {i} observations')
        assert_same({k: v for k, v in a.items() if k != 'observations'},
                    {k: v for k, v in e.items() if k != 'observations'}, f'rollout {i}')
    # train(): every minibatch (sequences, padding, masks, LSTM states, dtypes), its policy outputs and
    # gradient norm, the logged loss terms of each update, and every parameter and Adam moment after it.
    assert_same(actual.samples, expected.samples, 'minibatches')
    assert_same(actual.evaluated, expected.evaluated, 'evaluate_actions')
    assert_same(actual.grad_norms, expected.grad_norms, 'gradient norms')
    assert len(actual.train) == len(expected.train) == 3 and all(row for _, row in expected.train[1:])
    for (step_a, a), (step_e, e) in zip(actual.train, expected.train):
        assert step_a == step_e
        assert_same(a, e, f'logged losses at {step_e}')
    assert_same(actual.params, expected.params, 'parameters')
    assert_same(actual.optimizer, expected.optimizer, 'optimizer state')
    assert not torch.equal(actual.params[0]['action_net.weight'], actual.params[-1]['action_net.weight'])


def test_the_fast_update_equals_the_stock_update(monkeypatch):
    """DeviceRecurrentPPO.fast_train against RecurrentPPO.train() on the same rollouts (resets inside them and at the
    boundary, padded minibatches): every gradient norm, the logged loss terms, the parameters after each update and
    Adam's state are bitwise equal."""
    seed, logs = 3, []
    for fast in (False, True):
        monkeypatch.setattr(DeviceRecurrentPPO, 'FAST_UPDATE', fast)
        logs.append(run(build(DeviceEnv(fake(), seed), seed=seed, device='cpu', **SETTINGS), 2 * WORLDS * N_STEPS, monkeypatch))
    stock, fast = logs
    assert len(stock.samples) == 2 * EPOCHS * 3 and not fast.samples and not fast.evaluated  # fast_train took over
    assert (torch.stack([sample.mask.mean() for sample in stock.samples]) < 1).any(), 'need padded minibatches'
    assert_same(fast.grad_norms, stock.grad_norms, 'gradient norms')
    assert_same(fast.train, stock.train, 'logged losses')
    assert_same(fast.params, stock.params, 'parameters')
    assert_same(fast.optimizer, stock.optimizer, 'optimizer state')


def small_space():
    return spaces.Dict(dict(ids=spaces.Box(0, 4000, (3,), np.int64), mask=spaces.Box(0, 1, (5,), np.int8),
                            vec=spaces.Box(-np.inf, np.inf, (2, 2), np.float32)))


def filled(n_steps, n_envs, starts, gamma, gae_lambda, seed, last_done):
    """The same random rollout in SB3's buffer and the device buffer, returns computed by each."""
    space, hidden = small_space(), (n_steps, 1, n_envs, 4)
    rng = np.random.default_rng(seed)
    sb3 = RecurrentDictRolloutBuffer(n_steps, space, spaces.Discrete(96), hidden, 'cpu', gae_lambda, gamma, n_envs)
    mine = DeviceRolloutBuffer(n_steps, space, spaces.Discrete(96), hidden, 'cpu', gae_lambda, gamma, n_envs)
    for step in range(n_steps):
        obs = dict(ids=rng.integers(0, 4000, (n_envs, 3)), mask=rng.integers(0, 2, (n_envs, 5)).astype(np.int8),
                   vec=rng.normal(size=(n_envs, 2, 2)).astype(np.float32))
        action = rng.integers(0, 96, n_envs)
        reward = rng.normal(size=n_envs).astype(np.float32)
        value, log_prob = (torch.tensor(rng.normal(size=(n_envs, 1)), dtype=torch.float32),
                           torch.tensor(rng.normal(size=n_envs), dtype=torch.float32))
        states = RNNStates(*(tuple(torch.tensor(rng.normal(size=(1, n_envs, 4)), dtype=torch.float32) for _ in range(2))
                             for _ in range(2)))
        sb3.add(obs, action.reshape(-1, 1), reward, starts[step], value, log_prob, lstm_states=states)
        mine.add({k: torch.as_tensor(v) for k, v in obs.items()}, torch.as_tensor(action), torch.as_tensor(reward),
                 torch.as_tensor(starts[step], dtype=torch.float32), value, log_prob, lstm_states=states)
    last = torch.tensor(rng.normal(size=(n_envs, 1)), dtype=torch.float32)
    sb3.compute_returns_and_advantage(last, last_done)
    mine.compute_returns_and_advantage(last, torch.as_tensor(last_done))
    return sb3, mine


def test_discounted_gae_matches_sb3_bitwise():
    """With gamma and lambda below 1 the GAE loop does inexact float32 arithmetic; it is still SB3's."""
    rng = np.random.default_rng(0)
    starts = rng.random((64, 6)) < .1
    sb3, mine = filled(64, 6, starts, .99, .95, 1, rng.random(6) < .5)
    assert_same(clone(mine.advantages), clone(sb3.advantages), 'advantages')
    assert_same(clone(mine.returns), clone(sb3.returns), 'returns')
    assert np.abs(sb3.advantages).max() > 0


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 9), st.integers(1, 5), st.data())
def test_minibatches_match_sb3_for_any_layout(n_steps, n_envs, data):
    """get(batch_size) yields SB3's sequences, padding, masks, LSTM states and dtypes for any rollout
    length, world count, batch size, episode-start pattern and split index."""
    total = n_steps * n_envs
    starts = np.array(data.draw(st.lists(st.lists(st.booleans(), min_size=n_envs, max_size=n_envs),
                                         min_size=n_steps, max_size=n_steps)))
    batch_size = data.draw(st.integers(1, total + 2))
    sb3, mine = filled(n_steps, n_envs, starts, 1., 1., data.draw(st.integers(0, 99)), np.zeros(n_envs, bool))
    for epoch_seed in (data.draw(st.integers(0, 2 ** 31)), data.draw(st.integers(0, 2 ** 31))):
        np.random.seed(epoch_seed)
        expected = [clone(s) for s in sb3.get(batch_size)]
        np.random.seed(epoch_seed)
        actual = [clone(s) for s in mine.get(batch_size)]
        assert_same(actual, expected, 'samples')
    assert_same(torch.as_tensor(mine.values), torch.as_tensor(sb3.values), 'flattened values')
    assert_same(torch.as_tensor(mine.returns), torch.as_tensor(sb3.returns), 'flattened returns')


def test_an_unsupported_world_stops_the_rollout_before_bootstrap_or_update(tmp_path):
    """sim/spec.py: an unsupported step voids its rollout. The first rollout is clean and trains; the
    second has world 1 unsupported at its step 3 and raises with the evidence written down."""
    model = build(DeviceEnv(fake(unsupported=((N_STEPS + 3, 1),)), seed=5), seed=5, device='cpu', **SETTINGS)
    model.evidence_dir = tmp_path
    bootstraps, predict_values = [], model.policy.predict_values
    model.policy.predict_values = lambda *args: bootstraps.append(1) or predict_values(*args)
    with pytest.raises(Unsupported) as raised:
        model.learn(4 * WORLDS * N_STEPS)
    assert model._n_updates == EPOCHS and len(bootstraps) == 1, 'trained or bootstrapped from the void rollout'
    [row] = raised.value.evidence['rows']
    assert (row['step'], row['world'], row['decision'], row['unsupported']) == (3, 1, N_STEPS + 3, True)
    assert set(row) >= {'action', 'goal', 'terminal', 'episode_return', 'decision_ticks'}
    assert raised.value.path == tmp_path / f'unsupported-{WORLDS * N_STEPS}.json' and raised.value.path.exists()


def test_a_meaningless_observation_that_breaks_the_policy_still_reports_unsupported(tmp_path):
    """With an empty mask after the unsupported step, the policy's own check fails mid-rollout; the
    collector reports the unsupported world, not only the policy error."""
    model = build(DeviceEnv(fake(unsupported=((2, 0),), garbage=True), seed=5), seed=5, device='cpu', **SETTINGS)
    with pytest.raises(Unsupported) as raised:
        model.learn(WORLDS * N_STEPS)
    assert isinstance(raised.value.__cause__, ValueError) and model._n_updates == 0
    assert [(r['step'], r['world']) for r in raised.value.evidence['rows']] == [(2, 0)]


def test_a_saved_model_is_a_standard_checkpoint_the_evaluator_loads(tmp_path):
    """baseline/evaluate.py's load (RecurrentPPO.load, check_names with obs-v2) gives the same action
    probabilities and values for the same inputs and recurrent states, and the same decisions."""
    env = DeviceEnv(fake(), seed=9)
    model = build(env, seed=9, device='cpu', **SETTINGS)
    model.learn(WORLDS * N_STEPS)
    model.save(tmp_path / 'latest.zip')
    with zipfile.ZipFile(tmp_path / 'latest.zip') as saved:
        data = saved.read('data').decode()
    assert '_last_obs' not in data and 'rollouts' not in data and 'evidence_dir' not in data
    loaded = RecurrentPPO.load(tmp_path / 'latest.zip', device='cpu')
    check_names(loaded.policy)
    assert loaded.policy.observation == OBSERVATION and type(loaded.policy) is MaskedRecurrentPolicy
    obs = env.reset()
    generator = torch.Generator().manual_seed(0)
    states = [torch.randn((1, WORLDS, 128), generator=generator) for _ in range(4)]
    starts = torch.tensor([1., 0., 0., 1.])
    model.policy.set_training_mode(False)
    with torch.no_grad():
        a, _ = model.policy.get_distribution(obs, tuple(states[:2]), starts)
        b, _ = loaded.policy.get_distribution(obs, tuple(states[:2]), starts)
        assert_same(a.distribution.probs, b.distribution.probs, 'probabilities')
        assert_same(model.policy.predict_values(obs, tuple(states[2:]), starts),
                    loaded.policy.predict_values(obs, tuple(states[2:]), starts), 'values')
    one = {k: v[0].numpy() for k, v in obs.items()}
    state = tuple(s[:, :1].numpy() for s in states[:2])
    for first in (True, False):
        assert_same(clone(model.predict(one, state, np.array([first]), deterministic=True)),
                    clone(loaded.predict(one, state, np.array([first]), deterministic=True)), 'predict')


SECOND_DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else None


@pytest.mark.skipif(SECOND_DEVICE is None, reason='needs a CUDA or MPS device')
def test_a_checkpoint_loaded_onto_a_device_keeps_adams_step_counters_on_the_host(tmp_path):
    """SB3 loads a checkpoint's tensors onto the model's device and torch keeps a loaded step counter
    there, so the A100 run read 140 counters from the GPU every optimizer step. The moments go to the device."""
    model = build(DeviceEnv(fake(), seed=9), seed=9, device='cpu', **SETTINGS)
    model.learn(WORLDS * N_STEPS)
    model.save(tmp_path / 'latest.zip')
    saved = {i: s['step'].clone() for i, s in model.policy.optimizer.state_dict()['state'].items()}
    model.policy.to(SECOND_DEVICE)
    model.set_parameters(str(tmp_path / 'latest.zip'), device=SECOND_DEVICE)
    state = model.policy.optimizer.state_dict()['state']
    assert {s['step'].device.type for s in state.values()} == {'cpu'}
    assert {s['exp_avg'].device.type for s in state.values()} == {SECOND_DEVICE}
    assert_same({i: s['step'] for i, s in state.items()}, saved, 'step counters')


def outside_policy_reads(n_steps, reads):
    model = build(DeviceEnv(fake(), seed=11), seed=11, device='cpu', n_steps=n_steps, batch_size=BATCH, n_epochs=1)
    model.train = lambda: None  # collection only
    forward = model.policy.forward

    def marked(*args, **kwargs):
        reads.inside = True
        try:
            return forward(*args, **kwargs)
        finally:
            reads.inside = False
    model.policy.forward = marked
    reads.reset()
    with reads, jax.transfer_guard('disallow'):
        model.learn(WORLDS * n_steps)
    return reads.counts


def test_collection_reads_nothing_back_per_decision(monkeypatch):
    """Outside the policy's forward pass, a rollout's host reads do not grow with its length: zero per
    decision. JAX's transfer guard forbids every
    host-to-device copy during collection. The policy's own per-call reads are upstream and reported."""
    reads = HostReads()
    for name in ('numpy', 'tolist', 'item', 'cpu'):
        monkeypatch.setattr(torch.Tensor, name, reads.counted(name, getattr(torch.Tensor, name)))
    short, long = outside_policy_reads(4, reads), outside_policy_reads(12, reads)
    # Per rollout: the unsupported check, the episode statistics and the bootstrap's name lookups.
    assert short[False] == long[False], (short, long)
    per_call = {k: v / 4 for k, v in short[True].items()}
    assert per_call == {k: v / 12 for k, v in long[True].items()}, (short, long)


def test_the_training_entry_point_writes_a_complete_run(tmp_path):
    """sim.train train: checkpoints the evaluator can load, metrics, and a summary that reports the
    batching departures from config.LEARNER and equal transitions per goal."""
    from sim.train import train
    summary = train(tmp_path / 'run', decisions=2 * WORLDS * N_STEPS, seed=1, worlds=WORLDS, n_steps=N_STEPS,
                    batch_size=BATCH, n_epochs=1, checkpoint_every=WORLDS * N_STEPS, verbose=False)
    run_dir = tmp_path / 'run'
    assert summary['status'] == 'completed' and summary['ppo_updates'] == 2 and summary['gradient_steps'] == 6
    # Every update's losses, the last included (SB3 dumps them only at the next iteration)
    assert len(summary['losses']) == 2 and all(np.isfinite(v) for r in summary['losses'] for v in r.values())
    assert summary['changes_from_learner'] == dict(n_steps=dict(learner=1024, here=N_STEPS),
                                                   batch_size=dict(learner=256, here=BATCH), n_epochs=dict(learner=4, here=1))
    assert {g['transitions'] for g in summary['per_goal'].values()} == {2 * N_STEPS}
    assert [u['speed_paths'] for u in summary['updates']] == [['fused', 'valid_rows']] * 2  # on by default
    assert summary['learner']['gamma'] == config.LEARNER['gamma'] and summary['learner']['learning_rate'] == 3e-4
    for name in ('initial.zip', 'latest.zip', 'summary.json', 'metrics/progress.csv'):
        assert (run_dir / name).exists(), name
    assert len(list((run_dir / 'checkpoints').glob('ppo_*_steps.zip'))) == 2
    check_names(RecurrentPPO.load(run_dir / 'latest.zip', device='cpu').policy)


def test_the_refused_count_reads_the_feedback_refused_flag_not_a_walk_east():
    """rollout_evidence's refused_feedback (sim/train.py refused_flags) counts the stored observations whose feedback
    reports a refused end: a click refused with no route counts, a 9-tile walk east and a quiet start do not. The
    literal index 21 of feedback-v1 reads dx / 16 under feedback-v2's 12 families."""
    from pathlib import Path
    from feedback_contract import quiet
    from obs import project
    from sim.train import refused_flags
    capture = Path(__file__).resolve().parents[2] / 'rl' / 'tests' / 'fixtures' / 'reset.json'
    if not capture.exists():
        pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository')
    reset = json.loads(capture.read_text())
    click = dict(action=17, family='click', kind='', name='', option='0,4', item='')
    refused = dict(quiet(), action=click, ended=dict(action=17, reason='failed', refused=True), ticks=2)
    walked = dict(quiet(running=True), action=click, ticks=18, dx=9)
    vecs = torch.as_tensor(np.stack([project(dict(reset, feedback=fb))['fb_vec'] for fb in (refused, walked, quiet())]))
    assert refused_flags(vecs).tolist() == [1, 0, 0]
    assert refused_flags(vecs[None]).tolist() == [[1, 0, 0]]   # [steps, worlds] as the rollout buffer stores them


def test_collection_never_recompiles_the_environment():
    """the warm-up compiles reset and step with the arguments collection passes (keys and actions
    committed to the device, as DLPack makes them); two rollouts must not compile either again."""
    env = fake()
    build(DeviceEnv(env, seed=1), seed=1, device='cpu', **SETTINGS).learn(2 * WORLDS * N_STEPS)
    assert (env.reset._cache_size(), env.step._cache_size()) == (1, 1)


def test_telemetry_only_reads():
    """The read-only telemetry (Tracker, parameter_change) leaves every update exactly as it was."""
    def learned(telemetry):
        model = build(DeviceEnv(fake(), 7), seed=7, device='cpu', **SETTINGS)
        if telemetry:
            attach(model)
        model.learn(3 * WORLDS * N_STEPS)
        return model
    plain, tracked = learned(False), learned(True)
    a, b = plain.policy.state_dict(), tracked.policy.state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert len(tracked.update_sizes) == 3 and set(tracked.update_sizes[-1]) == {'actor', 'critic'}
    assert tracked.tracker.acc and set(tracked.tracker.report()) >= {'actor.encoder0', 'critic.encoder0'}


def test_telemetry_after_a_step_waits_for_nothing():
    """A graphed update runs the telemetry after every replay, so accumulating a step must not size anything by its
    data (a device wait). Padded rows (goal -1) still count for no goal: the sums are those of the real rows."""
    from baseline.resource_task import GOALS
    tracker = attach(build(DeviceEnv(fake(), 7), seed=7, device='cpu', **SETTINGS)).tracker
    g = torch.Generator().manual_seed(0)
    goal = torch.randint(-1, len(GOALS), (64,), generator=g)
    tracker.goals = dict(actor=goal, critic=goal.roll(5))
    tracker.last = {name: torch.relu(torch.randn(64, 8, generator=g)) for name, _ in tracker.layers}
    reads = HostReads()
    with reads:
        tracker.accumulate()
    assert not reads.counts[False] and not reads.counts[True]
    for name, _ in tracker.layers:
        goals, h = tracker.goals[name.split('.')[0]], tracker.last[name]
        real = goals >= 0
        a = h[real].abs()
        expected = (torch.zeros(len(GOALS), 8).index_add_(0, goals[real], a),
                    torch.zeros(len(GOALS), 8).index_add_(0, goals[real], (a > 0).float()),
                    torch.bincount(goals[real], minlength=len(GOALS)).float())
        assert all(torch.equal(x, y) for x, y in zip(tracker.acc[name], expected)), name


def test_a_recovery_state_with_a_removed_treatment_is_refused():
    from sim.train import CheckFailed, put_treatment_state
    assert put_treatment_state(None) is None
    with pytest.raises(CheckFailed, match='removed treatments'):
        put_treatment_state(dict(distill=dict(updates=1, sampler=torch.zeros(4, dtype=torch.uint8))))
    with pytest.raises(CheckFailed, match='removed treatments'):
        put_treatment_state(dict(distill=None, rnd=dict(step=torch.zeros(1))))


def test_the_ppo_profile_counts_device_time_once():
    """an aten operator's self device time repeats its kernels'; the total counts the device's own events
    (and a legacy profiler's CPU events), not the operators or user annotations, as torch's table does."""
    from torch.autograd import DeviceType
    from sim.train import device_seconds
    event = lambda device, us, legacy=False, annotation=False: SimpleNamespace(
        device_type=device, self_device_time_total=us, is_legacy=legacy, is_user_annotation=annotation)
    events = [event(DeviceType.CPU, 100.), event(DeviceType.CUDA, 100.), event(DeviceType.CUDA, 50., annotation=True),
              event(DeviceType.CPU, 30., legacy=True)]
    assert device_seconds(events) == pytest.approx(130e-6)


def test_a_resumed_run_continues_from_its_last_complete_update_with_its_optimizer_and_counters(tmp_path):
    """sim.train train --recovery saves every complete update; --resume restores the policy, Adam, the counters and
    the generators, then trains only the decisions left, in new episodes; checkpoints fall on update boundaries."""
    from sim.train import load_recovery, train
    rollout = WORLDS * N_STEPS
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, checkpoint_every=rollout,
                  verbose=False, recovery=tmp_path / 'rec')
    train(tmp_path / 'a', decisions=2 * rollout, **common)
    rows = [json.loads(line) for line in (tmp_path / 'rec' / 'updates.jsonl').read_text().splitlines()]
    assert [r['num_timesteps'] for r in rows] == [rollout, 2 * rollout]
    saved = torch.load(tmp_path / 'rec' / 'state.pt', weights_only=False)
    final = RecurrentPPO.load(tmp_path / 'a' / 'latest.zip', device='cpu')
    assert all(torch.equal(saved['policy'][k], v) for k, v in final.policy.state_dict().items())
    model = build(DeviceEnv(fake(), 9), seed=9, device='cpu', **dict(SETTINGS, n_epochs=1))
    record = load_recovery(model, tmp_path / 'rec')
    assert (model.num_timesteps, record['num_timesteps']) == (2 * rollout, 2 * rollout)
    restored = model.policy.optimizer.state_dict()['state']
    assert all(torch.equal(restored[k]['exp_avg'], v['exp_avg']) for k, v in saved['optimizer']['state'].items())
    resumed = train(tmp_path / 'b', decisions=3 * rollout, resume=tmp_path / 'rec', env_seed=2, **common)
    assert resumed['status'] == 'completed' and resumed['total_timesteps'] == 3 * rollout and resumed['ppo_updates'] == 1
    assert resumed['resume']['num_timesteps'] == 2 * rollout and resumed['env_seed'] == 2
    rows = [json.loads(line) for line in (tmp_path / 'rec' / 'updates.jsonl').read_text().splitlines()]
    assert [r['num_timesteps'] for r in rows] == [rollout, 2 * rollout, 3 * rollout]
    assert sorted(p.name for p in (tmp_path / 'b' / 'checkpoints').glob('*.zip')) == [f'ppo_{3 * rollout}_steps.zip']


def test_a_resume_refuses_another_goal_set_than_its_recovery_state_trained(tmp_path):
    from sim.train import CheckFailed, train
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False,
                  recovery=tmp_path / 'rec')
    first = train(tmp_path / 'a', decisions=WORLDS * N_STEPS, goals=('wood',), **common)
    assert first['per_goal']['wood']['transitions'] == WORLDS * N_STEPS and first['per_goal']['fish']['transitions'] == 0
    with pytest.raises(CheckFailed, match='trained goals'):
        train(tmp_path / 'b', decisions=2 * WORLDS * N_STEPS, resume=tmp_path / 'rec', **common)
    again = train(tmp_path / 'c', decisions=2 * WORLDS * N_STEPS, resume=tmp_path / 'rec', goals=('wood',), **common)
    assert again['status'] == 'completed' and again['resume']['goals'] == ['wood']


def test_a_resume_refuses_another_task_than_its_recovery_state_trained(tmp_path):
    """The recovery record names its task (states of goal-activity-v5 and v11 must not cross); a record
    written before the field resumes as before."""
    from sim.train import CheckFailed, train
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False,
                  recovery=tmp_path / 'rec')
    train(tmp_path / 'a', decisions=WORLDS * N_STEPS, **common)
    record = json.loads((tmp_path / 'rec' / 'state.json').read_text())
    assert record['task'] == 'goal-activity-v2'
    (tmp_path / 'rec' / 'state.json').write_text(json.dumps(dict(record, task='goal-activity-v11')))
    with pytest.raises(CheckFailed, match='trained task goal-activity-v11, this run declares goal-activity-v2'):
        train(tmp_path / 'b', decisions=2 * WORLDS * N_STEPS, resume=tmp_path / 'rec', **common)
    del record['task']
    (tmp_path / 'rec' / 'state.json').write_text(json.dumps(record))
    again = train(tmp_path / 'c', decisions=2 * WORLDS * N_STEPS, resume=tmp_path / 'rec', **common)
    assert again['status'] == 'completed'


def test_each_update_saves_a_fixed_sample_that_replays_under_the_policy_that_collected_it(tmp_path, monkeypatch):
    """Every update saves the same world slots' first decisions with what the update reads; a sample replays under
    the checkpoint that collected it (recovery naming: segments-N under ppo_N_steps), refuses another policy and catches a
    changed log-probability or a changed segment-start state; saving it changes no update. The one-step replay
    repeats the collection itself at its 8 rows, the 4 segments in their slots and copies of them beside: with the name
    table projected whole, as on CUDA, that is exact; with the CPU's own lookups it is within ATOL, as is the learner's
    sequence form."""
    from sim.train import ATOL, CheckFailed, GOALS, SEGMENTS, replay_segments, segment_slots, train
    worlds, steps = 8, 8
    rollout, segments = worlds * steps, dict(worlds=4, decisions=6)
    common = dict(decisions=3 * rollout, seed=1, worlds=worlds, n_steps=steps, batch_size=BATCH, n_epochs=1,
                  checkpoint_every=rollout, verbose=False)
    run = train(tmp_path / 'a', recovery=tmp_path / 'rec', segments=segments, **common)
    train(tmp_path / 'b', **common)
    saved, unsaved = (RecurrentPPO.load(tmp_path / k / 'latest.zip', device='cpu').policy.state_dict() for k in 'ab')
    assert all(torch.equal(saved[k], unsaved[k]) for k in saved)  # read-only: no generator or update moves
    assert run['segments']['updates'] == 3 and run['segments']['slots'] == [0, 1, 2, 3]
    from sim.train import policy_digest
    initial = RecurrentPPO.load(tmp_path / 'a' / 'initial.zip', device='cpu').policy
    assert run['policy_sha256_at_start'] == policy_digest(initial)  # what the arms of one seed compare
    assert max(run['segments'][f'worst_{k}gap'] for k in ('log_prob_', 'value_', 'sequence_log_prob_', 'sequence_value_')) <= ATOL
    for k, row in enumerate(run['updates']):
        path = tmp_path / 'a' / row['segments']['path']
        assert path.name == f'segments-{k * rollout:09d}.npz' and row['segments']['bytes'] == path.stat().st_size
        with np.load(path) as f:
            assert f['obs__mask'].shape == (6, 4, 96) and f['hidden_states_pi'].shape == (4, 1, 128)
            assert (f['info__goal'] == np.arange(4)).all()  # slot b asks for GOALS[b % 4] at every step
        policy = RecurrentPPO.load(tmp_path / 'a' / ('initial.zip' if k == 0 else f'checkpoints/ppo_{k * rollout}_steps.zip'),
                                   device='cpu').policy
        replayed = replay_segments(path, policy)
        assert replayed['returns_exact'] and replayed['rows'] == worlds
        assert max(replayed[k] for k in ('log_prob_gap', 'value_gap', 'sequence_log_prob_gap', 'sequence_value_gap')) <= ATOL
    with pytest.raises(CheckFailed, match='another policy'):
        replay_segments(tmp_path / 'a' / run['updates'][1]['segments']['path'],
                        RecurrentPPO.load(tmp_path / 'a' / 'initial.zip', device='cpu').policy)
    with np.load(tmp_path / 'a' / run['updates'][0]['segments']['path']) as f:
        arrays = {k: f[k] for k in f.files}
    arrays['log_probs'][2, 1] += 1e-3
    np.savez_compressed(tmp_path / 'changed.npz', **arrays)
    changed = replay_segments(tmp_path / 'changed.npz', RecurrentPPO.load(tmp_path / 'a' / 'initial.zip', device='cpu').policy)
    assert changed['log_prob_gap'] > 1e-4
    with np.load(tmp_path / 'a' / run['updates'][1]['segments']['path']) as f:  # mid-episode: the start states are read
        arrays = {k: f[k] for k in f.files}
    assert not arrays['episode_starts'][0, 1]
    arrays['hidden_states_pi'][1] += 1e-2  # segment 1's stored start state: every later step's output follows it
    np.savez_compressed(tmp_path / 'changed.npz', **arrays)
    changed = replay_segments(tmp_path / 'changed.npz',
                              RecurrentPPO.load(tmp_path / 'a' / f'checkpoints/ppo_{rollout}_steps.zip', device='cpu').policy)
    assert changed['log_prob_gap'] > ATOL
    from baseline.policy import NameEmbedding
    monkeypatch.setattr(NameEmbedding, 'FULL_TABLE', True)  # CUDA's lookups: no row depends on another's names
    whole = train(tmp_path / 'whole', segments=segments, **common)
    assert whole['segments']['worst_log_prob_gap'] == whole['segments']['worst_value_gap'] == 0.0
    # the declared sample: 16 slots per goal of 64, spread over a shared batch; a specialist's spread over every slot
    shared = segment_slots(1024, GOALS, SEGMENTS['worlds'])
    assert len(shared) == 64 and all(sum(s % 4 == g for s in shared) == 16 for g in range(4)) and max(shared) > 960
    assert segment_slots(1024, ('fish',), SEGMENTS['worlds']) == list(range(0, 1024, 16))
    with pytest.raises(ValueError):
        segment_slots(8, GOALS, 6)


def test_the_in_run_segment_check_hashes_the_collecting_policy_once_per_update(tmp_path, monkeypatch):
    """The in-run check's replay takes the digest save_segments computed for the same policy (A6, after phase 1), so a
    run hashes its policy once at the start and once per update, not twice, and once more at the end for latest.zip's
    policy_sha256_at_end (no periodic checkpoint falls in two updates); the replay still reads the digest back from the
    saved file and refuses a sample another policy collected. The replay-segments command passes no digest and hashes
    the checkpoint it is given."""
    import sim.train
    from sim.train import CheckFailed, policy_digest, replay_segments, train
    worlds, steps = 8, 8
    hashed = []
    monkeypatch.setattr(sim.train, 'policy_digest', lambda policy: hashed.append(policy) or policy_digest(policy))
    run = train(tmp_path / 'a', decisions=2 * worlds * steps, seed=1, worlds=worlds, n_steps=steps, batch_size=BATCH,
                n_epochs=1, verbose=False, segments=dict(worlds=4, decisions=6))
    assert run['segments']['updates'] == 2 and run['checkpoint_digests'] == {} and len(hashed) == 1 + 2 + 1  # start, updates, end
    initial = RecurrentPPO.load(tmp_path / 'a' / 'initial.zip', device='cpu').policy
    first, second = (tmp_path / 'a' / row['segments']['path'] for row in run['updates'])
    assert replay_segments(first, initial, digest=policy_digest(initial))['returns_exact']
    with pytest.raises(CheckFailed, match='another policy'):  # update 1's sample, collected after an update
        replay_segments(second, initial, digest=policy_digest(initial))
    hashed.clear()
    with pytest.raises(CheckFailed, match='another policy'):
        replay_segments(second, initial)
    assert hashed == [initial]


def test_a_sample_that_does_not_replay_in_the_learners_sequence_shape_stops_the_run_before_its_update(tmp_path, monkeypatch):
    """Runs after phase 1 (decision of 28 October 2026): the sequence-shape replay gates at SEQUENCE_ATOL beside the
    one-step check. A sequence reconstruction whose observations run one step ahead of their actions and stored outputs
    (an off-by-one in DeviceRolloutBuffer._get_samples, which the learner also uses) passes the one-step replay, which
    never reads it, and stops the run with CheckFailed before PPO's first update; the same run without the shift completes."""
    import sim.train
    from sim.learn import DeviceRecurrentPPO
    from sim.train import ATOL, SEQUENCE_ATOL, CheckFailed, train
    worlds, steps = 8, 8
    common = dict(decisions=worlds * steps, seed=1, worlds=worlds, n_steps=steps, batch_size=BATCH, n_epochs=1,
                  verbose=False, segments=dict(worlds=4, decisions=6))
    normal = train(tmp_path / 'normal', **common)
    assert normal['status'] == 'completed' and normal['segments']['updates'] == normal['ppo_updates'] == 1
    assert max(normal['segments'][f'worst_sequence_{k}_gap'] for k in ('log_prob', 'value')) <= SEQUENCE_ATOL
    original, replay, replayed, updates = (DeviceRolloutBuffer._get_samples, sim.train.replay_segments, [], [])

    def shifted(self, rows, first, max_length):  # each row's observation read from the next stored row
        return original(self, rows, first, max_length)._replace(
            observations=original(self, rows.roll(-1), first, max_length).observations)
    monkeypatch.setattr(DeviceRolloutBuffer, '_get_samples', shifted)
    monkeypatch.setattr(sim.train, 'replay_segments', lambda *a, **k: replayed.append(replay(*a, **k)) or replayed[-1])
    monkeypatch.setattr(DeviceRecurrentPPO, 'train', lambda self: updates.append(self.num_timesteps))
    with pytest.raises(CheckFailed, match='update 0 do not replay in the learner\'s sequence shape: a sequence gap above'):
        train(tmp_path / 'shifted', **common)
    assert updates == [] and len(replayed) == 1  # stopped before the update it would have fed
    assert replayed[0]['log_prob_gap'] <= ATOL and replayed[0]['value_gap'] <= ATOL and replayed[0]['returns_exact']
    assert max(replayed[0]['sequence_log_prob_gap'], replayed[0]['sequence_value_gap']) > SEQUENCE_ATOL


def test_a_weights_run_takes_only_the_policy_tensors_and_starts_from_their_outputs(tmp_path):
    """sim.train train --weights: the checkpoint's policy and critic tensors with this run's own settings and a fresh
    optimizer; its first rollout is checked against the checkpoint loaded as itself before any update."""
    from sim.train import train
    rollout = WORLDS * N_STEPS
    train(tmp_path / 'src', decisions=2 * WORLDS * 2 * N_STEPS, seed=1, worlds=WORLDS, n_steps=2 * N_STEPS, batch_size=BATCH,
          n_epochs=2, verbose=False)
    run = train(tmp_path / 'run', decisions=rollout, seed=3, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1,
                verbose=False, weights=tmp_path / 'src' / 'latest.zip')
    assert run['weights']['optimizer'] == 'fresh' and run['weights']['saved_settings_not_imported']['n_steps'] == 2 * N_STEPS
    assert run['learner']['n_steps'] == N_STEPS and run['learner']['n_epochs'] == 1
    assert run['source_check']['max_difference'] == 0.0 and run['source_check']['steps'] == N_STEPS
    assert run['gradient_steps'] == -(-rollout // BATCH)  # Adam counts this run's steps only
    with pytest.raises(ValueError):
        train(tmp_path / 'both', decisions=rollout, seed=3, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1,
              verbose=False, weights=tmp_path / 'src' / 'latest.zip', init=tmp_path / 'src' / 'latest.zip')


def test_a_weights_or_init_run_refuses_a_same_shape_checkpoint_of_another_or_no_contract(tmp_path):
    """sim.train train --weights and --init check the versions the checkpoint records (check_names) before loading it:
    the same tensors saved with another mask version, or with no contract recorded (as before obs-v3), are refused and
    never taken under this run's versions. Before the check both loaded with every tensor equal."""
    from baseline.policy import CONTRACT
    from sim.train import train
    rollout = WORLDS * N_STEPS
    settings_ = dict(decisions=rollout, seed=3, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False)
    train(tmp_path / 'src', **dict(settings_, seed=1))
    source = RecurrentPPO.load(tmp_path / 'src' / 'latest.zip', device='cpu')
    check_names(source.policy)
    tensors = {k: v.clone() for k, v in source.policy.state_dict().items()}
    for name, contract in (('other', dict(CONTRACT, mask='mask-v2')), ('none', None)):
        kwargs = {k: v for k, v in source.policy_kwargs.items() if k != 'contract'}
        source.policy_kwargs = kwargs if contract is None else dict(kwargs, contract=contract)
        source.save(tmp_path / f'{name}.zip')
        saved = RecurrentPPO.load(tmp_path / f'{name}.zip', device='cpu').policy
        assert getattr(saved, 'contract', None) == contract
        assert all(torch.equal(v, tensors[k]) for k, v in saved.state_dict().items())   # only the record differs
        for start in ('weights', 'init'):
            with pytest.raises(ValueError, match='trained under'):
                train(tmp_path / f'{start}-{name}', **settings_, **{start: tmp_path / f'{name}.zip'})


@settings(max_examples=60, deadline=None)
@given(per=st.integers(1, 8), spread=st.integers(1, 6), goal=st.sampled_from(('wood', 'fish', 'mine', 'melee', None)))
def test_the_segment_sample_splits_equally_between_the_declared_goals_and_spans_the_batch(per, spread, goal):
    from sim.spec import goal_set, slot_goal
    from sim.train import GOALS, segment_slots
    goals = GOALS if goal is None else (goal,)
    worlds = len(goals) * per * spread
    slots = segment_slots(worlds, goals, len(goals) * per)
    assert len(set(slots)) == len(slots) == len(goals) * per and all(0 <= w < worlds for w in slots)
    counts = {g: sum(slot_goal(goal_set(goals), w) == g for w in slots) for g in goals}
    assert set(counts.values()) == {per}
    assert max(slots) >= worlds - len(goals) * spread  # the last slots are sampled too, not only the first
    with pytest.raises(ValueError):
        segment_slots(worlds, goals, len(goals) * per * spread + len(goals))


def test_the_graphed_collector_forward_is_the_policy_call_operation_for_operation():
    """GraphedForward._forward (the arithmetic its CUDA graph captures), run eagerly on the CPU against the policy's own
    call from the same generator state: the same actions, and values, log-probabilities and LSTM states bitwise without
    the whole-table name projection (equal within float32 rounding with it)."""
    from baseline.policy import NameEmbedding
    from sim.learn import GraphedForward
    model = build(DeviceEnv(fake(), 3), seed=3, device='cpu', **SETTINGS)
    obs = model.env.reset()
    rng = torch.Generator().manual_seed(5)
    states = RNNStates(*(tuple(torch.randn((1, WORLDS, 128), generator=rng) for _ in range(2)) for _ in range(2)))
    starts = torch.tensor([1., 0., 0., 1.])
    graphed = GraphedForward(model.policy)
    graphed.obs, graphed.states, graphed.starts = obs, [*states.pi, *states.vf], starts
    for full_table in (False, True):
        torch.manual_seed(11)
        with torch.no_grad():
            expected = model.policy(obs, states, starts)
        torch.manual_seed(11)
        NameEmbedding.FULL_TABLE = full_table
        try:
            with torch.no_grad():
                actions, values, log_prob, out = graphed._forward()
        finally:
            NameEmbedding.FULL_TABLE = None
        got = (actions, values, log_prob, *out)
        want = (expected[0], expected[1], expected[2], *expected[3].pi, *expected[3].vf)
        assert torch.equal(got[0], want[0])
        for a, b in zip(got[1:], want[1:]):
            assert torch.equal(a, b) if not full_table else torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_the_graphed_update_step_is_the_fast_update_on_padded_shapes(monkeypatch):
    """GraphedSteps._body (the step its CUDA graphs capture: static row indices, the LSTM grid padded to a bucket, the
    whole-table name projection), run eagerly on the CPU in place of each replay, against fast_train on the same rollouts:
    the same logged losses and parameters within float32 rounding."""
    from baseline.policy import NameEmbedding
    from sim.learn import GraphedSteps
    settings = dict(SETTINGS, batch_size=8)  # GraphedSteps needs equal minibatches: 32 transitions, 4 of 8
    logs = []
    for graphed in (False, True):
        model = build(DeviceEnv(fake(), 3), seed=3, device='cpu', **settings)
        if graphed:
            steps = GraphedSteps(model)
            monkeypatch.setattr(steps, 'prepare', lambda clip: setattr(steps, 'signature', (clip,)))

            class Eager:  # replay() runs the captured body eagerly
                def __init__(self, shape):
                    self.shape = shape

                def replay(self):
                    steps._body(*self.shape)
            monkeypatch.setattr(steps, '_capture', lambda n, t: Eager((n, t)))
            model._graphs = steps
            monkeypatch.setattr(DeviceRecurrentPPO, 'GRAPHED_UPDATE', True)
            monkeypatch.setattr(NameEmbedding, 'FULL_TABLE', True)
        logs.append(run(model, 2 * WORLDS * N_STEPS, monkeypatch))
        monkeypatch.setattr(DeviceRecurrentPPO, 'GRAPHED_UPDATE', False)
        monkeypatch.setattr(NameEmbedding, 'FULL_TABLE', None)
    fast, graphed = logs
    assert steps.graphs and all(n >= 1 and t >= 1 for n, t in steps.graphs)
    for (_, a), (_, b) in zip(fast.train, graphed.train):
        for k in a:
            if k.startswith('train/') and isinstance(a[k], float):
                assert b[k] == pytest.approx(a[k], rel=1e-4, abs=1e-6), k
    for a, b in zip(fast.params[-1].values(), graphed.params[-1].values()):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


class Calls(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


def test_a_fresh_build_is_the_same_tensors_from_the_same_seed_and_loads_no_weights(monkeypatch):
    """Arm N's initialization: build gives the same tensors from a seed and others from
    another, records act-v3's contract and reads no saved weights (nothing of phase 1's files)."""
    from baseline.policy import CONTRACT, check_names
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('a fresh build loaded saved weights'))
    tensors = lambda seed: build(DeviceEnv(fake(), 3), seed=seed, device='cpu', **SETTINGS)
    a, b, other = tensors(101), tensors(101), tensors(202)
    same = lambda x, y: all(torch.equal(p, q) for p, q in zip(x.policy.parameters(), y.policy.parameters()))
    assert same(a, b) and not same(a, other)
    assert a.policy.contract == CONTRACT
    check_names(a.policy)


def test_merged_lookups_give_the_per_input_rows_bit_for_bit_and_their_gradients_within_rounding(monkeypatch):
    """NameEmbedding.MERGED with the whole table projected: each embedding looks up all its inputs at once (6 lookups and
    6 embedding backwards a training call, not 28 under obs-v3), so values, log-probabilities and entropies are the per-input lookups'
    bit for bit; only the six embedding parameters' gradients change, by the rounding of a reordered sum."""
    from baseline.policy import NameEmbedding
    model = build(DeviceEnv(fake(), 3), seed=3, device='cpu', **SETTINGS)
    model.learn(WORLDS * N_STEPS)
    policy, buffer = model.policy, model.rollout_buffer
    data = buffer.compact(*next(buffer.layout(BATCH)))
    monkeypatch.setattr(NameEmbedding, 'FULL_TABLE', True)
    out, grads, calls = {}, {}, {}
    for merged in (False, True):
        monkeypatch.setattr(NameEmbedding, 'MERGED', merged)
        policy.optimizer.zero_grad(set_to_none=True)
        with Calls() as counted:
            values, log_prob, entropy = policy.evaluate_compact(data.observations, data.actions, data.lstm_states, data.keep,
                                                                data.pos, data.n_seq, data.max_length)
            (values.square().mean() + log_prob.mean() + entropy.mean()).backward()
        out[merged], calls[merged] = (values, log_prob, entropy), counted.counts
        grads[merged] = {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}
    assert all(torch.equal(a, b) for a, b in zip(out[False], out[True]))
    assert [(calls[m]['embedding'], calls[m]['embedding_dense_backward']) for m in (False, True)] == [(28, 28), (6, 6)]
    embeddings = {f'{m}.{p}' for m in ('features_extractor', 'vf_features_extractor') for p in ('embedding.project.weight',
                                                                                                   'options.weight')}
    embeddings |= {'slot_embedding.project.weight', 'slot_options.weight'}
    assert set(grads[False]) == set(grads[True]) >= embeddings
    for name, a in grads[False].items():
        b = grads[True][name]
        assert torch.equal(a, b) if name not in embeddings else torch.allclose(a, b, rtol=1e-5, atol=1e-6 * a.abs().max()), name


def test_the_recurrent_check_catches_a_changed_state_and_a_start_that_reads_the_stored_state(monkeypatch):
    """sim/train.py recurrent_gaps, several steps a call and the episode starts in more than one pair of calls: the
    rollout as collected passes (no start reads its stored state); a changed stored state and a policy that keeps the
    stored state at an episode start fail."""
    import sim.train as train
    from sim.train import ATOL, recurrent_gaps
    model = build(DeviceEnv(fake(), 5), seed=5, device='cpu', **SETTINGS)
    _, callback = model._setup_learn(WORLDS * N_STEPS, None)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)  # the policy that collected it, no update
    b = model.rollout_buffer
    monkeypatch.setattr(train, 'VALIDATE_ROWS', 2 * WORLDS)  # 2 steps a call; starts in pairs of calls of 8 rows
    starts = b.episode_starts.nonzero()
    assert len(starts) > 2 * WORLDS and (b.episode_starts == 0).any()
    gap, reset_gap = recurrent_gaps(model)
    assert gap <= ATOL and reset_gap == 0
    step, world = (b.episode_starts == 0).nonzero()[-1].tolist()
    b.hidden_states_pi[step, world] += 1
    assert recurrent_gaps(model)[0] > ATOL
    b.hidden_states_pi[step, world] -= 1
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    leak = lambda self, features, states, starts, lstm: RecurrentActorCriticPolicy._process_sequence(  # SB3's loop, no reset
        features, states, torch.zeros_like(starts), lstm)
    monkeypatch.setattr(type(model.policy), '_process_sequence', leak)
    assert recurrent_gaps(model)[1] > ATOL


@pytest.mark.parametrize('value_norm', [None, POPART], ids=['linear', 'popart'])
def test_the_scaled_recurrent_check_takes_rounding_at_a_large_value_and_fails_a_wrong_row(value_norm):
    """sim/train.py recurrent_gaps scaled, validate_free_play's, with a plain value head and a PopArt one: with
    the value moved to about 80 before the collection (the head's bias, or PopArt's mean), the rollout passes; one stored
    value three float32 rounding steps away (3 * 2**-17 at 80: beyond ATOL alone) passes within ATOL + RTOL x its scale.
    One stored log-probability 1e-3 away fails as the log-probability's, its scale from the row's legal logits (the mask's
    fill of -1e9 is not among them) and the actor's terms before them. At the same values a row moved (another column's stored states) and every row
    after one update of the weights fail by two orders of magnitude or more, the worst row named. The unscaled check is
    unchanged."""
    from sim.train import ATOL, RTOL, recurrent_gaps
    model = build(DeviceEnv(fake(), 5), seed=5, device='cpu', value_norm=value_norm, **SETTINGS)
    with torch.no_grad():
        head = model.policy.value_net
        (head.mean if value_norm else head.bias).add_(80)
    _, callback = model._setup_learn(WORLDS * N_STEPS, None)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)  # 4 rows a call; the check's one call: 32
    b = model.rollout_buffer
    assert bool(((b._values >= 64) & (b._values < 128)).all())  # rounding steps of 2**-17
    step, world = (b.episode_starts == 0).nonzero()[-1].tolist()  # a row that reads its stored states
    b._values[step, world] += 3 * 2 ** -17
    gap, reset_gap, worst, scales = recurrent_gaps(model, scaled=True)
    assert gap > ATOL and recurrent_gaps(model) == (gap, reset_gap)  # the unscaled check would fail this row
    assert worst['ratio'] <= 1 and (worst['output'], worst['step'], worst['column']) == ('value', step, world)
    assert worst['scale'] >= 80 and worst['tolerance'] == ATOL + RTOL * worst['scale']
    assert scales['value'] >= 80 and scales['log probability'] < 100
    b.log_probs[step, world] += 1e-3
    worst = recurrent_gaps(model, scaled=True)[2]
    assert worst['ratio'] >= 10 and (worst['output'], worst['step'], worst['column']) == ('log probability', step, world)
    b.log_probs[step, world] -= 1e-3
    other = (world + 1) % WORLDS
    for stored in (b.hidden_states_pi, b.cell_states_pi, b.hidden_states_vf, b.cell_states_vf):
        stored[step, world] = stored[step, other]
    worst = recurrent_gaps(model, scaled=True)[2]
    assert worst['ratio'] >= 100 and (worst['step'], worst['column']) == (step, world)
    model.train()  # one update: every stored row is now of other weights
    assert recurrent_gaps(model, scaled=True)[2]['ratio'] >= 100


def test_a_log_probability_is_held_to_the_terms_the_actor_sums_before_its_logits():
    """sim/train.py recurrent_gaps scaled (21 November 2026): a log-probability's scale is its row's largest
    legal |logit| and the actor's largest first-encoder-layer and LSTM gate terms. A new policy's legal logits are below
    1 while its first encoder layer sums terms of about 15: one stored log-probability 1.5e-5 away (beyond ATOL + RTOL x
    its logits, about 1.03e-5) passes within its scale of 10 or more; 1e-4 away it fails; and every stored row after one
    update of the weights, and a row moved to another column's stored states, fail by two orders of magnitude or
    more."""
    from sim.train import ATOL, RTOL, computed_from, recurrent_gaps
    model = build(DeviceEnv(fake(), 5), seed=5, device='cpu', **SETTINGS)
    _, callback = model._setup_learn(WORLDS * N_STEPS, None)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)
    b = model.rollout_buffer
    step, world = (b.episode_starts == 0).nonzero()[-1].tolist()  # a row that reads its stored states
    o = {k: v[step, world][None] for k, v in b.observations.items()}
    states = [s[step, world][:, None].contiguous() for s in (b.hidden_states_pi, b.cell_states_pi)]
    with torch.no_grad(), computed_from(model.policy) as seen:
        model.policy.get_distribution(o, states, b.episode_starts[step, world][None])
    logits = float(torch.where(o['mask'] != 0, seen['logits'].abs(), 0).amax())
    assert logits < 1 and ATOL + RTOL * logits < 1.5e-5
    b.log_probs[step, world] += 1.5e-5
    gap, _, worst, _ = recurrent_gaps(model, scaled=True)
    assert gap > ATOL and worst['ratio'] <= 1 and worst['scale'] >= 10
    assert (worst['output'], worst['step'], worst['column']) == ('log probability', step, world)
    b.log_probs[step, world] += 1e-4 - 1.5e-5
    worst = recurrent_gaps(model, scaled=True)[2]
    assert worst['ratio'] > 1 and (worst['step'], worst['column']) == (step, world)
    b.log_probs[step, world] -= 1e-4
    other = (world + 1) % WORLDS
    for stored in (b.hidden_states_pi, b.cell_states_pi, b.hidden_states_vf, b.cell_states_vf):
        stored[step, world] = stored[step, other]
    worst = recurrent_gaps(model, scaled=True)[2]
    assert worst['ratio'] >= 100 and (worst['step'], worst['column']) == (step, world)
    model.train()  # one update: every stored row is now of other weights
    assert recurrent_gaps(model, scaled=True)[2]['ratio'] >= 100
