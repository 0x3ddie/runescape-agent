import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils._python_dispatch import TorchDispatchMode

from baseline.adapter import GameVecEnv
from baseline.policy import MaskedRecurrentPolicy
from obs import OBSERVATION
from baseline.tests.test_resource_task import raw as engine_raw
from names import name_table


def sample(value=1.0, legal=(0, 7)):
    mask = np.zeros(96, dtype=np.int8)
    mask[list(legal)] = 1
    return {'player_vec': np.array([value, -value], np.float32),
            'npc_ids': np.array([1, 0], np.int64), 'mask': mask}


def batch(items):
    return {key: np.stack([item[key] for item in items]) for key in items[0]}


class Backend:
    n_envs = 2

    def __init__(self):
        self.calls = self.resets = self.closed = 0
        self.result = None

    def reset(self, seed=None):
        self.resets += 1
        self.seed = seed
        return batch([sample(), sample(2)])

    def step(self, actions):
        self.calls += 1
        if self.result is not None:
            return copy.deepcopy(self.result)
        return batch([sample(self.calls + 1), sample(self.calls + 2)]), np.ones(2), np.zeros(2, bool), np.zeros(2, bool), [{}, {}]

    def close(self):
        self.closed += 1


def make_model(env):
    torch.set_num_threads(1)
    return RecurrentPPO(MaskedRecurrentPolicy, env, n_steps=4, batch_size=8,
                        n_epochs=1, seed=11, device='cpu',
                        policy_kwargs=dict(lstm_hidden_size=16, names=name_table().spec, observation=OBSERVATION,
                                           features_extractor_kwargs=dict(features_dim=16),
                                           net_arch=dict(pi=[16], vf=[16])))


def state(policy, n=2):
    shape = (1, n, policy.lstm_actor.hidden_size)
    return RNNStates((torch.zeros(shape), torch.zeros(shape)),
                     (torch.zeros(shape), torch.zeros(shape)))


@pytest.mark.parametrize('terminated,truncated,expected', [(False, True, True),
                                                          (True, False, False),
                                                          (True, True, False)])
def test_autoreset_final_observation_and_timeout_precedence(terminated, truncated, expected):
    backend = Backend()
    env = GameVecEnv(backend, sample())
    env.seed(101)
    env.reset()
    backend.result = (batch([sample(99), sample(2)]), np.array([2., 3.]),
                      np.array([terminated, False]), np.array([truncated, False]),
                      [{'final_observation': sample(5), 'final_info': {'episode': {'r': 2, 'l': 3}}}, {}])
    obs, rewards, done, infos = env.step(np.array([0, 7]))
    assert backend.resets == 1 and backend.seed == 101
    assert done.tolist() == [True, False]
    assert obs['player_vec'][0, 0] == 99
    assert infos[0]['terminal_observation']['player_vec'][0] == 5
    assert infos[0]['TimeLimit.truncated'] is expected
    assert infos[0]['episode'] == {'r': 2, 'l': 3}


def test_missing_terminal_observation_poisoned_batch_cannot_resume():
    backend = Backend()
    env = GameVecEnv(backend, sample())
    env.reset()
    backend.result = (batch([sample(), sample()]), np.zeros(2),
                      np.array([True, False]), np.zeros(2, bool), [{}, {}])
    with pytest.raises(ValueError, match='final observation'):
        env.step(np.array([0, 0]))
    assert backend.closed == 1
    with pytest.raises(RuntimeError):
        env.reset()


@pytest.mark.parametrize('actions', [np.array([1, 0]), np.array([96, 0]),
                                      np.array([0., 0.]), np.array([0])])
def test_invalid_actions_do_not_reach_engine(actions):
    backend = Backend()
    env = GameVecEnv(backend, sample())
    env.reset()
    with pytest.raises(ValueError):
        env.step(actions)
    assert backend.calls == 0


def test_real_empty_mask_rejected_at_adapter_and_actor():
    env = GameVecEnv(Backend(), sample())
    model = make_model(env)
    empty = sample(legal=())
    with pytest.raises(ValueError):
        env._validate(batch([empty, empty]))
    obs = obs_as_tensor(batch([empty, empty]), 'cpu')
    with pytest.raises(ValueError, match='All-masked'):
        model.policy.forward(obs, state(model.policy), torch.ones(2))
    assert model.policy._active_mask is None


def test_collection_evaluation_and_prediction_use_same_masks_and_logprobs():
    env = GameVecEnv(Backend(), sample())
    model = make_model(env)
    policy = model.policy
    obs = obs_as_tensor(batch([sample(legal=(7,)), sample(legal=(0, 7))]), 'cpu')
    initial = state(policy)
    starts = torch.ones(2)
    with torch.no_grad():
        actions, values, logprob, new_state = policy.forward(obs, initial, starts)
        evaluated_values, evaluated_logprob, entropy = policy.evaluate_actions(obs, actions, initial, starts)
        distribution, predicted_state = policy.get_distribution(obs, initial.pi, starts)
    assert actions[0].item() == 7
    assert distribution.distribution.probs[0, 7].item() == 1
    assert torch.all(distribution.distribution.probs[:, 1:7] == 0)
    torch.testing.assert_close(logprob, evaluated_logprob)
    torch.testing.assert_close(values, evaluated_values)
    torch.testing.assert_close(new_state.pi[0], predicted_state[0])
    assert policy._active_mask is None and torch.isfinite(entropy).all()


def test_memory_changes_outputs_and_resets_only_requested_actor():
    model = make_model(GameVecEnv(Backend(), sample()))
    p = model.policy
    obs = obs_as_tensor(batch([sample(), sample(2)]), 'cpu')
    initial = state(p)
    with torch.no_grad():
        _, _, _, carried = p.forward(obs, initial, torch.ones(2))
        _, _, _, mixed = p.forward(obs, carried, torch.tensor([1., 0.]))
        _, _, _, reset = p.forward(obs, initial, torch.ones(2))
    torch.testing.assert_close(mixed.pi[0][:, 0], reset.pi[0][:, 0])
    assert not torch.allclose(mixed.pi[0][:, 1], reset.pi[0][:, 1])


def test_padded_rows_finite_but_nonzero_all_masked_rows_fail():
    p = make_model(GameVecEnv(Backend(), sample())).policy
    obs = obs_as_tensor(batch([sample(), sample()]), 'cpu')
    for value in obs.values():
        value[1] = 0
    outputs = p.evaluate_actions(obs, torch.zeros(2, dtype=torch.long), state(p), torch.ones(2))
    assert all(torch.isfinite(value).all() for value in outputs)
    obs['player_vec'][1, 0] = 1
    with pytest.raises(ValueError, match='All-masked'):
        p.evaluate_actions(obs, torch.zeros(2, dtype=torch.long), state(p), torch.ones(2))
    obs['mask'][0, 3] = 2  # checked with the rows in one read, and reported first
    with pytest.raises(ValueError, match='Invalid action mask'):
        p.evaluate_actions(obs, torch.zeros(2, dtype=torch.long), state(p), torch.ones(2))


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.starts, self.ends = [], []

    def _on_rollout_start(self):
        self.starts.append(self.model._last_lstm_states.pi[0].clone())

    def _on_rollout_end(self):
        self.ends.append(self.model._last_lstm_states.pi[0].clone())
        # Before optimization, sampled recurrent sequences reproduce stored logp.
        for data in self.model.rollout_buffer.get(batch_size=8):
            with torch.no_grad():
                _, logp, _ = self.model.policy.evaluate_actions(
                    data.observations, data.actions.long().flatten(), data.lstm_states, data.episode_starts)
            valid = data.mask.bool()
            torch.testing.assert_close(logp[valid], data.old_log_prob[valid], atol=2e-6, rtol=2e-6)

    def _on_step(self):
        return True


def test_upstream_rollouts_preserve_memory_update_weights_and_roundtrip(tmp_path):
    model = make_model(GameVecEnv(Backend(), sample()))
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    capture = Capture()
    # Synthetic optimizer contract test; no game rollout or capability estimate.
    model.learn(total_timesteps=16, callback=capture)
    assert len(capture.starts) == 2
    torch.testing.assert_close(capture.starts[1], capture.ends[0])
    assert torch.any(capture.starts[1] != 0)
    assert any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
    path = tmp_path / 'policy.zip'
    model.save(path)
    loaded = RecurrentPPO.load(path, device='cpu')
    obs = batch([sample(), sample(2)])
    carried = tuple(v.detach().numpy() for v in model._last_lstm_states.pi)
    a, s = model.predict(obs, state=carried, episode_start=np.zeros(2, bool), deterministic=True)
    b, t = loaded.predict(obs, state=carried, episode_start=np.zeros(2, bool), deterministic=True)
    np.testing.assert_array_equal(a, b)
    for x, y in zip(s, t):
        np.testing.assert_array_equal(x, y)


def test_actual_item_v2_projection_fits_schema_and_encoder():
    from obs import project
    projected = project(engine_raw())
    model = make_model(GameVecEnv(Backend(), projected))
    actions, _ = model.predict(batch([projected, projected]), deterministic=True)
    assert actions.tolist() == [0, 0]


def test_act_slots_reach_upstream_encoder_and_checkpoint_schema(tmp_path):
    from obs import project
    raw = dict(engine_raw(), actSel=[dict(kind='loc', id=1276, name='Tree', option='Chop down', x=101, z=100, level=0,
                                          dx=1, dz=0, distance=1, steps=1, count=1, combatLevel=0)], mask=[0]*96)
    raw['mask'][1]=1
    projected=project(raw)
    env=GameVecEnv(Backend(),projected);model=make_model(env)
    observations=batch([projected,projected])
    actions,_=model.predict(observations,deterministic=True)
    assert actions.tolist()==[1,1]
    tensor=obs_as_tensor(observations,'cpu')
    with torch.no_grad():
        before=model.policy.features_extractor(tensor)
        # The name table is not loaded, so this untrained policy sees slot geometry, not names.
        tensor['act_feats']=tensor['act_feats'].clone();tensor['act_feats'][:,0,0]+=.5
        after=model.policy.features_extractor(tensor)
    assert not torch.equal(before,after)
    path=tmp_path/'act-policy';model.save(path)
    loaded=RecurrentPPO.load(path,device='cpu')
    assert loaded.observation_space['act_feats'].shape==(16,11)  # ten affordance features and the bank distance
    np.testing.assert_array_equal(loaded.predict(observations,deterministic=True)[0],actions)
    wrong={k:v for k,v in observations.items() if not k.startswith('act_')}
    with pytest.raises(ValueError,match='keys differ'):env._validate(wrong)


def test_actual_upstream_timeout_bootstrap_uses_final_not_reset_observation():
    backend = Backend()
    env = GameVecEnv(backend, sample())
    model = make_model(env)
    backend.result = (batch([sample(99), sample(88)]), np.array([2., 3.]),
                      np.array([False, True]), np.array([True, True]),
                      [{'final_observation': sample(5)}, {'final_observation': sample(6)}])
    seen = []

    def constant_value(obs, states, starts):
        seen.append((obs['player_vec'].clone(), starts.clone()))
        return torch.full((obs['player_vec'].shape[0], 1), 7.)

    model.policy.predict_values = constant_value
    _, callback = model._setup_learn(total_timesteps=8)
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
    np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], 2 + model.gamma * 7)
    np.testing.assert_allclose(model.rollout_buffer.rewards[:, 1], 3)
    assert all(values[0, 0] == 5 and starts[0] == 0 for values, starts in seen[:-1])
    assert seen[-1][0][0, 0] == 99


def test_names_v1_policy_reads_a_frozen_table_kept_in_the_checkpoint(tmp_path, monkeypatch):
    import names
    from baseline.policy import CONTRACT, NameEmbedding, check_names, load_name_vectors
    def table_at(path, vectors):
        np.savez(path, kind=np.array(['npc', 'npc', 'obj']), name=np.array(['guard', 'jailer', 'logs']), vector=vectors)
        monkeypatch.setenv('RL_NAME_TABLE', str(path)); names.name_table.cache_clear()
        return names.name_table()
    vectors = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], np.float16)  # a guard and a jailer read alike
    table = table_at(tmp_path / 'names.npz', vectors)
    model = RecurrentPPO(MaskedRecurrentPolicy, GameVecEnv(Backend(), sample()), n_steps=4, batch_size=8, seed=11, device='cpu',
                         policy_kwargs=dict(lstm_hidden_size=16, names=table.spec, observation=OBSERVATION, contract=CONTRACT,
                                            net_arch=dict(pi=[16], vf=[16]), features_extractor_kwargs=dict(features_dim=16)))
    load_name_vectors(model.policy, table.vectors)
    embedding = model.policy.features_extractor.embedding
    assert isinstance(embedding, NameEmbedding)
    with torch.no_grad():
        unknown, guard, jailer, logs = embedding(torch.tensor([0, 1, 2, 3]))
    assert not unknown.any() and torch.equal(guard, jailer) and not torch.equal(guard, logs)
    model.save(tmp_path / 'policy')
    loaded = RecurrentPPO.load(tmp_path / 'policy', device='cpu')
    torch.testing.assert_close(loaded.policy.features_extractor.embedding.table, embedding.table)
    check_names(loaded.policy)
    # act-v2, mask-v2 and feedback-v1 weights record no contract; another contract is refused too, never converted.
    for other in (None, dict(CONTRACT, actions='act-v2'), dict(CONTRACT, mask='mask-v2'), dict(CONTRACT, feedback='feedback-v1')):
        loaded.policy.contract = other
        with pytest.raises(ValueError, match='trained under'):
            check_names(loaded.policy)
    loaded.policy.contract = CONTRACT
    # A checkpoint saved before obs-v2 records no observation version: refused, never converted.
    loaded.policy.observation = None
    with pytest.raises(ValueError, match='obs-v1 inputs'):
        check_names(loaded.policy)
    check_names(loaded.policy, observation=None)
    table_at(tmp_path / 'other.npz', vectors[::-1].copy())
    with pytest.raises(ValueError, match='another name table'):
        check_names(loaded.policy)


def test_a_rebuilt_name_table_plays_a_checkpoint_only_by_its_fingerprint_and_a_public_copy_takes_its_vectors(tmp_path,
                                                                                                                monkeypatch):
    """check_names: the exact table passes by its digest. A table whose every value moved one float16 step away from zero
    (the worst such rebuild) has another digest: refused without the trained table's fingerprint, taken with it (path
    'fingerprint'). With the fingerprint, a table with one name changed, its vectors scaled by 1.01, or its first and last
    rows' vectors swapped is refused. The fingerprint holds no vector's values. check_names changes nothing: a policy whose
    tables hold no vectors (a public copy's, which its loader fills) keeps them empty."""
    import json
    import names
    from baseline.policy import CONTRACT, check_names, load_name_vectors, name_fingerprint
    rng = np.random.default_rng(3)
    kinds, labels = np.array(['npc'] * 20 + ['obj'] * 20), np.array([f'name {k}' for k in range(40)])
    vectors = (rng.normal(size=(40, 16)) * 0.1).astype(np.float16)

    def table_at(name, kinds=kinds, labels=labels, vectors=vectors):
        path = tmp_path / f'{name}.npz'
        np.savez(path, kind=kinds, name=labels, vector=vectors)
        monkeypatch.setenv('RL_NAME_TABLE', str(path))
        names.name_table.cache_clear()
        return names.name_table()
    table = table_at('trained')
    model = RecurrentPPO(MaskedRecurrentPolicy, GameVecEnv(Backend(), sample()), n_steps=4, batch_size=8, seed=11,
                         device='cpu', policy_kwargs=dict(lstm_hidden_size=16, names=table.spec, observation=OBSERVATION,
                                                          contract=CONTRACT, net_arch=dict(pi=[16], vf=[16]),
                                                          features_extractor_kwargs=dict(features_dim=16)))
    load_name_vectors(model.policy, table.vectors)
    fingerprint = json.loads(json.dumps(name_fingerprint(table)))  # as the file beside a public copy holds it
    assert check_names(model.policy) == dict(path='digest', table=table.digest)
    assert all(len(fingerprint[k]) == 16 for k in ('sums', 'squares', 'weighted'))
    assert not {v for row in vectors.astype(float).tolist() for v in row} & set(fingerprint['sums'] + fingerprint['weighted'])
    away = np.where(vectors < 0, -np.inf, np.inf).astype(np.float16)
    stepped = np.nextafter(vectors, away)
    assert (stepped != vectors).all() and (np.abs(stepped) > np.abs(vectors)).all()
    other = table_at('stepped', vectors=stepped)
    assert other.digest != table.digest
    with pytest.raises(ValueError, match='another name table than RL_NAME_TABLE$'):
        check_names(model.policy)
    taken = check_names(model.policy, fingerprint=fingerprint)
    assert (taken['path'], taken['table'], taken['trained']) == ('fingerprint', other.digest, table.digest)
    for name, change, reason in (('renamed', dict(labels=np.array([*labels[:-1], 'name x'])), 'kinds and names'),
                                 ('scaled', dict(vectors=(vectors.astype(np.float32) * 1.01).astype(np.float16)), '(sums|squares)'),
                                 ('swapped', dict(vectors=vectors[[39, *range(1, 39), 0]]), 'weighted')):
        table_at(name, **change)
        with pytest.raises(ValueError, match=f'by its fingerprint: .*{reason}'):
            check_names(model.policy, fingerprint=fingerprint)
    table = table_at('trained')
    load_name_vectors(model.policy, np.zeros(table.vectors.shape, np.float32))  # a public copy's tables
    assert check_names(model.policy) == dict(path='digest', table=table.digest)
    assert not model.policy.features_extractor.embedding.table.any()


class Syncs(TorchDispatchMode):
    """Counts the operations that wait for a GPU: scalar reads and outputs whose size depends on data."""
    OPS = ('_local_scalar_dense', '_unique2', 'nonzero', 'masked_select')

    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if name in self.OPS:
            self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


@pytest.fixture
def speed_paths():
    """MaskedRecurrentPolicy's learner-speed switches, set by one test and put back after it."""
    defaults = MaskedRecurrentPolicy.FUSED_SEQUENCES, MaskedRecurrentPolicy.VALID_ROWS
    yield MaskedRecurrentPolicy
    MaskedRecurrentPolicy.FUSED_SEQUENCES, MaskedRecurrentPolicy.VALID_ROWS = defaults


def test_the_speed_paths_match_sb3_on_real_rows_and_still_reject_a_real_all_masked_row(speed_paths):
    torch.manual_seed(3)
    env = GameVecEnv(Backend(), sample())
    model = RecurrentPPO(MaskedRecurrentPolicy, env, n_steps=4, batch_size=8, n_epochs=1, seed=11, device='cpu',
                         policy_kwargs=dict(lstm_hidden_size=16, names=name_table().spec, observation=OBSERVATION,
                                            share_features_extractor=False, enable_critic_lstm=True,
                                            features_extractor_kwargs=dict(features_dim=16), net_arch=dict(pi=[16], vf=[16])))
    p = model.policy
    # three sequences of four rows, sequence-major as the buffer flattens them: 4, 2 and 1 real rows, the rest padding
    real = [4, 2, 1]
    rows = [sample(value=0.3 * (k + 1) + 0.1 * t, legal=(0, 7)) if t < n else None
            for k, n in enumerate(real) for t in range(4)]
    obs = obs_as_tensor(batch([r if r is not None else sample() for r in rows]), 'cpu')
    pad = torch.tensor([r is None for r in rows])
    for value in obs.values():
        value[pad] = 0
    starts = torch.zeros(12)
    starts[[0, 8]] = 1  # sequences 0 and 2 start an episode
    hidden = lambda: torch.randn(1, 3, 16)
    states = RNNStates((hidden(), hidden()), (hidden(), hidden()))
    actions = torch.zeros(12, dtype=torch.long)

    def evaluate(fused, valid_rows):
        speed_paths.FUSED_SEQUENCES, speed_paths.VALID_ROWS = fused, valid_rows
        return p.evaluate_actions(obs, actions, states, starts)
    stock = evaluate(False, False)
    for fused, valid_rows in ((True, False), (False, True), (True, True)):
        for got, want in zip(evaluate(fused, valid_rows), stock):
            torch.testing.assert_close(got.flatten()[~pad], want.flatten()[~pad], rtol=1e-5, atol=1e-6)
            if valid_rows:
                assert torch.all(got.flatten()[pad] == 0)
    # each of these operations synchronizes a GPU. A call makes one read for all its checks (stock adds SB3's
    # two loop checks; torch's distribution checks are off here) and one torch.unique per name input for both
    # extractors; sharing that result changes no output
    validate = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        for fused, valid_rows, want in ((False, False, dict(_local_scalar_dense=3, _unique2=1)),
                                        (True, True, dict(_local_scalar_dense=1, _unique2=1, nonzero=1))):
            with Syncs() as syncs:
                shared = evaluate(fused, valid_rows)
            assert syncs.counts == want
            embeddings, p._name_embeddings = p._name_embeddings, []
            try:
                assert all(torch.equal(a, b) for a, b in zip(evaluate(fused, valid_rows), shared))
            finally:
                p._name_embeddings = embeddings
    finally:
        torch.distributions.Distribution.set_default_validate_args(validate)
    # a start after a sequence's first step: the fused path falls back to SB3's loop, exactly
    starts[5] = 1
    speed_paths.FUSED_SEQUENCES, speed_paths.VALID_ROWS = False, False
    loop = p.evaluate_actions(obs, actions, states, starts)
    speed_paths.FUSED_SEQUENCES = True
    assert all(torch.equal(a, b) for a, b in zip(p.evaluate_actions(obs, actions, states, starts), loop))
    # the check the first valid-row prototype skipped: a real row without a legal action is not padding
    speed_paths.VALID_ROWS = True
    obs['mask'][3] = 0
    with pytest.raises(ValueError, match='All-masked'):
        p.evaluate_actions(obs, actions, states, starts)
    assert p._active_mask is None and p._active_real is None


def test_collection_is_bitwise_unchanged_by_the_fused_path(speed_paths):
    p = make_model(GameVecEnv(Backend(), sample())).policy
    obs = obs_as_tensor(batch([sample(), sample(2)]), 'cpu')
    hidden = torch.randn(1, 2, p.lstm_actor.hidden_size)
    states = RNNStates((hidden, hidden.flip(1)), (hidden.flip(1), hidden))
    with torch.no_grad():
        for starts in (torch.zeros(2), torch.tensor([1., 0.])):
            speed_paths.FUSED_SEQUENCES = False
            want = p.forward(obs, states, starts, deterministic=True)
            speed_paths.FUSED_SEQUENCES = True
            got = p.forward(obs, states, starts, deterministic=True)
            assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2])
            assert all(torch.equal(a, b) for a, b in zip(got[3].pi + got[3].vf, want[3].pi + want[3].vf))


def obs_v3_checkpoint(path):
    """A small obs-v3, mask-v3 policy over the projected native reset with goal-v2's four goal entries, saved at path,
    its weights moved off their initialization as a training run would leave them (no column of the first layer is
    zero). Returns one obs-v3 observation."""
    from baseline.policy import CONVERTIBLE
    from baseline.resource_task import GOALS
    from obs import project
    capture = Path(__file__).resolve().parents[2] / 'rl/tests/fixtures/reset.json'
    if not capture.exists():
        pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository')
    base = project(json.loads(capture.read_text()))
    old = dict(base, budget_vec=np.asarray([.5, .024], np.float32), critic_vec=np.zeros(1, np.float32),
               goal_vec=np.zeros(len(GOALS), np.float32))
    model = RecurrentPPO(MaskedRecurrentPolicy, GameVecEnv(Backend(), old), n_steps=4, batch_size=8, n_epochs=1, seed=11,
                         device='cpu', policy_kwargs=dict(lstm_hidden_size=16, names=name_table().spec,
                                                          observation=CONVERTIBLE['observation'],
                                                          contract=CONVERTIBLE['contract'],
                                                          features_extractor_kwargs=dict(features_dim=16),
                                                          net_arch=dict(pi=[16], vf=[16])))
    with torch.no_grad():
        for parameter in model.policy.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(parameter.numel())) * .1)
    model.save(path)
    return old


def test_an_obs_v3_checkpoint_converts_to_obs_v5_with_equal_outputs_for_the_four_goals(tmp_path):
    """convert_checkpoint: an obs-v3, mask-v3 policy over the projected native
    reset with goal-v2's four goal entries is saved, converted and loaded again. The result records obs-v5, mask-v5 and
    the conversion, passes check_names, and for each of the four old goals gives the source's action probabilities and
    values within 1e-6 over a few recurrent steps, with goal-v3's two new entries zero. The source itself is refused by
    check_names, and a converted checkpoint is not converted again."""
    from baseline.policy import CONTRACT, check_names, convert_checkpoint
    from baseline.resource_task import GOAL_VEC, GOALS
    from baseline.telemetry import sha256
    old = obs_v3_checkpoint(tmp_path / 'v3.zip')
    record = convert_checkpoint(tmp_path / 'v3.zip', tmp_path / 'v5.zip')
    source, converted = (RecurrentPPO.load(tmp_path / f'{name}.zip', device='cpu') for name in ('v3', 'v5'))
    assert record['source_sha256'] == sha256(tmp_path / 'v3.zip') and record['source_observation'] == 'obs-v3'
    assert converted.policy_kwargs['conversion'] == record and converted.policy.contract == CONTRACT
    check_names(converted.policy)
    with pytest.raises(ValueError, match='inputs'):
        check_names(source.policy)
    with pytest.raises(ValueError):
        convert_checkpoint(tmp_path / 'v5.zip', tmp_path / 'again.zip')
    assert converted.observation_space['goal_vec'].shape == (len(GOAL_VEC),)
    worst = 0.0
    for g in range(len(GOALS)):
        rows = []
        for value in (0.0, 0.5, 1.0):
            step = {k: v.copy() for k, v in old.items()}
            step['goal_vec'][g] = 1.0
            step['budget_vec'][0] = value
            rows.append(step)
        outputs = []
        for model_, pad in ((source, 0), (converted, len(GOAL_VEC) - len(GOALS))):
            policy = model_.policy
            policy.set_training_mode(False)
            lstm = policy.lstm_actor
            state = RNNStates(*[tuple(torch.zeros(lstm.num_layers, 1, lstm.hidden_size) for _ in range(2))] * 2)
            seen = []
            for k, step in enumerate(rows):
                step = dict(step, goal_vec=np.concatenate([step['goal_vec'], np.zeros(pad, np.float32)]))
                obs, _ = policy.obs_to_tensor({key: v[None] for key, v in step.items()})
                start = torch.as_tensor([float(k == 0)])
                with torch.no_grad():
                    dist, _ = policy.get_distribution(obs, state.pi, start)
                    _, value, _, state = policy.forward(obs, state, start, deterministic=True)
                seen.append((dist.distribution.probs, value))
            outputs.append(seen)
        for (p1, v1), (p2, v2) in zip(*outputs):
            worst = max(worst, float((p1 - p2).abs().max()), float((v1 - v2).abs().max()))
    assert worst <= 1e-6, worst


def test_the_conversion_command_converts_as_the_function_does(tmp_path):
    """`python -m baseline.policy SOURCE OUTPUT` runs the module as __main__, whose classes are not the ones a
    loaded checkpoint holds (baseline.policy's): it must convert as convert_checkpoint does, every saved tensor equal,
    and print the same record. Before the fix it found no goal column and failed to load the padded weights."""
    import os
    import subprocess
    import sys
    from baseline.policy import convert_checkpoint
    obs_v3_checkpoint(tmp_path / 'v3.zip')
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(root / 'rl')]))
    run = subprocess.run([sys.executable, '-m', 'baseline.policy', str(tmp_path / 'v3.zip'), str(tmp_path / 'cli.zip')],
                         cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    record = convert_checkpoint(tmp_path / 'v3.zip', tmp_path / 'function.zip')
    assert json.loads(run.stdout) == json.loads(json.dumps(record))
    cli, function = (RecurrentPPO.load(tmp_path / f'{name}.zip', device='cpu').policy.state_dict()
                     for name in ('cli', 'function'))
    assert cli.keys() == function.keys() and all(torch.equal(cli[k], function[k]) for k in cli)
