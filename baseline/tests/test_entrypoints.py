"""train -> evaluate through the real SB3 learner, against the offline fake world."""
import json

import numpy as np
import pytest
import torch

import baseline.evaluate as evaluate
import baseline.train as train
from baseline.resource_task import GOAL_FAMILIES, ResourceEpisode
from baseline.world import read_jsonl
from baseline.tests.test_resource_task import CFG, World


@pytest.fixture
def worlds(monkeypatch):
    worlds = []
    def connect(record, agents=1, spectator=None):
        worlds[:] = [World('alternate') for _ in range(agents)]
        return [(world, world.control) for world in worlds]
    def make_episode(transport, service, fixture, record):
        transport.record = record
        return ResourceEpisode(transport, service, CFG, 2, 10, record)
    for module in (train, evaluate):
        monkeypatch.setattr(module, 'connect', connect)
        monkeypatch.setattr(module, 'make_episode', make_episode)
        monkeypatch.setattr(module, 'delivery_report', lambda service, min_windows: dict(ok=True, windows=[]))
    return worlds


def test_trained_checkpoint_evaluates_against_controls(worlds, tmp_path):
    summary = train.train(tmp_path / 'run', transitions=1024, seed=1, agents=1, kl_coef=0, device='cpu')
    assert summary['status'] == 'completed' and summary['ppo_transitions'] == 1024
    assert summary['ppo_updates'] > 0 and summary['changed_tensors'] > 0
    assert summary['manifest']['code']['commit'] and summary['manifest']['game']['xp_rate'] == 25
    # No KL target here, so the one update takes every step of its 4 epochs x 4 minibatches.
    assert summary['actor_gradient_steps'] == summary['actor_gradient_steps_full'] == 16
    # A fresh run uses LEARNER as it stands, with no KL of either kind.
    learner = summary['learner']
    assert {k: learner[k] for k in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
                                    'clip_range', 'ent_coef')} == \
        {k: train.config.LEARNER[k] for k in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                                              'gae_lambda', 'clip_range', 'ent_coef')}
    assert learner['target_kl'] is None and not learner['kl_weight'] and summary['critic_warmup'] == 0
    # The one rollout's buffer, as the update used it.
    sample = np.load(tmp_path / 'run/rollouts.npz')
    assert sample['rollout'].tolist() == [0] and sample['rewards'].shape == (1, 1024, 1)
    np.testing.assert_allclose(sample['returns'], sample['advantages'] + sample['values'], rtol=1e-5, atol=1e-6)
    rows = list(read_jsonl(tmp_path / 'run/episodes.jsonl.gz'))
    terminals = [r for r in rows if r['kind'] == 'task_terminal']
    measures = [r for r in rows if r['kind'] == 'episode_measures']
    assert len(measures) == len(terminals) > 0 and all('episode' in r and 'agent' in r for r in rows if r['kind'] != 'bc_epoch')
    assert [m['final_value'] for m in measures] == [t['report']['reward_eligible_lower'] for t in terminals]
    assert sum(r['kind'] == 'effect' for r in rows) >= sum(m['decisions'] for m in measures)
    for policy, checkpoint in (('wait', None), ('checkpoint', tmp_path / 'run/latest.zip')):
        results = evaluate.evaluate(tmp_path / policy, policy=policy, checkpoint=checkpoint,
                                    fixtures=('short', 'long'), trials=1)
        assert results['status'] == 'completed' and results['manifest']['code']['commit']
        assert [t['fixture'] for t in results['trials']] == ['short', 'long']
        assert all(t['report']['survived'] and t['final_value'] == t['report']['reward_eligible_lower']
                   for t in results['trials'])


def test_parallel_agents_train_under_the_kl_penalty(worlds, tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO
    monkeypatch.setitem(train.config.FINETUNE, 'critic_warmup', 2048)
    train.train(tmp_path / 'first', transitions=1024, seed=1, agents=1, kl_coef=0, device='cpu')
    warm = train.train(tmp_path / 'warm', transitions=2048, seed=2, agents=2, kl_coef=.5,
                       init=tmp_path / 'first/latest.zip', device='cpu')
    assert warm['critic_warmup'] == 2048 and len(worlds) == 2 and all(w.epoch > 1 for w in worlds)
    start, after = (RecurrentPPO.load(tmp_path / f'{run}/latest.zip', device='cpu').policy.state_dict()
                    for run in ('first', 'warm'))
    assert all(torch.equal(start[k], after[k]) for k in start if k.startswith(('action_net', 'lstm_actor')))
    assert any(not torch.equal(start[k], after[k]) for k in start if k.startswith('value_net'))
    summary = train.train(tmp_path / 'kl', transitions=4096, seed=3, agents=2, kl_coef=.5,
                          init=tmp_path / 'first/latest.zip', device='cpu')
    assert summary['status'] == 'completed' and summary['ppo_transitions'] == 4096
    logged = [json.loads(line) for line in (tmp_path / 'kl/metrics/progress.json').read_text().splitlines()]
    assert any('train/kl_to_reference' in row for row in logged)


def test_kl_penalty_is_zero_against_itself_and_positive_against_another_policy(worlds, tmp_path):
    from sb3_contrib import RecurrentPPO
    train.train(tmp_path / 'run', transitions=0, seed=1, agents=1, kl_coef=0, device='cpu')
    policy = RecurrentPPO.load(tmp_path / 'run/latest.zip', device='cpu').policy
    obs = {k: torch.as_tensor(np.zeros((3, *s.shape), s.dtype)) for k, s in policy.observation_space.spaces.items()}
    obs['mask'][:, :3] = 1
    state = torch.zeros(policy.lstm_actor.num_layers, 3, policy.lstm_actor.hidden_size)
    states = type('States', (), dict(pi=(state, state), vf=(state, state)))
    starts, actions = torch.ones(3), torch.zeros(3)
    with torch.no_grad():
        obs['ref_probs'] = policy.get_distribution(obs, (state, state), starts)[0].distribution.probs
        policy.kl_weight = 10.
        _, _, same = policy.evaluate_actions(obs, actions, states, starts)
        policy.kl_weight = 0.
        _, _, plain = policy.evaluate_actions(obs, actions, states, starts)
        obs['ref_probs'] = torch.zeros_like(obs['ref_probs']); obs['ref_probs'][:, 1] = 1
        policy.kl_weight = 10.
        _, _, other = policy.evaluate_actions(obs, actions, states, starts)
    torch.testing.assert_close(same, plain)
    assert torch.all(other < plain) and policy.kl_log[-1] > 0


@pytest.mark.parametrize('name,first', [('act', 1), ('click', 17)])
def test_target_scores_follow_the_target_not_the_slot(worlds, tmp_path, name, first):
    from sb3_contrib import RecurrentPPO
    train.train(tmp_path / 'run', transitions=0, seed=1, agents=1, kl_coef=0, device='cpu')
    policy = RecurrentPPO.load(tmp_path / 'run/latest.zip', device='cpu').policy
    assert hasattr(policy, f'{name}_scorer')
    obs = {k: torch.as_tensor(np.zeros((1, *s.shape), s.dtype)) for k, s in policy.observation_space.spaces.items()}
    state = torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
    probs = lambda o: policy.get_distribution(o, (state, state), torch.ones(1))[0].distribution.probs[0]
    if name == 'click':
        # The click table's slots keep their offsets (the trunk reads it in slot order); one scorer serves both row
        # groups, so equal slot features score equally: slot 0 on row 17 and slot 12 on row 84.
        obs['mask'][:, [0, 17, 84]] = 1
        obs['click_feats'][0, 0] = obs['click_feats'][0, 12] = torch.tensor([.25, -.5, 1.])
        with torch.no_grad():
            p = probs(obs)
        torch.testing.assert_close(p[17], p[84])
        return
    obs['mask'][:, [0, first, first + 1]] = 1
    keys = [k for k in (f'{name}_ids', f'{name}_op_ids', f'{name}_feats') if k in obs]
    for key in keys:
        obs[key][0, :2] = torch.rand(obs[key][0, :2].shape) * (8 if key.endswith('_ids') else 1)  # rows of the test table
    swapped = {k: v.clone() for k, v in obs.items()}
    for key in keys:
        swapped[key][0, [0, 1]] = obs[key][0, [1, 0]]
    with torch.no_grad():
        before, after = (probs(o) for o in (obs, swapped))
    torch.testing.assert_close(before[[first, first + 1]], after[[first + 1, first]])


def test_the_per_update_kl_limit_can_be_removed_and_its_early_stops_are_counted(worlds, tmp_path, monkeypatch):
    monkeypatch.setitem(train.config.FINETUNE, 'critic_warmup', 1024)
    train.train(tmp_path / 'first', transitions=1024, seed=1, agents=1, kl_coef=0, device='cpu')
    runs = {limit: train.train(tmp_path / f'run-{limit}', transitions=2048, seed=2, agents=1, kl_coef=.5,
                               init=tmp_path / 'first/latest.zip', device='cpu', target_kl=limit)
            for limit in (None, -1.)}
    free, stopped = runs[None], runs[-1.]
    assert free['target_kl'] is None and stopped['target_kl'] == -1.
    assert free['actor_gradient_steps'] == free['actor_gradient_steps_full'] == 16  # 4 epochs x 4 minibatches
    # The fake world offers one legal action, so the policy's KL stays 0; a negative limit (KL > -1.5)
    # still ends every update at its first minibatch, before any step, and the count shows it.
    assert stopped['actor_gradient_steps'] == 0


def test_resume_continues_a_run_as_it_was(worlds, tmp_path, monkeypatch):
    """--resume keeps the policy, critic, optimizer state, settings and step count: no
    fine-tuning settings, critic warm-up or KL, unlike --init."""
    from sb3_contrib import RecurrentPPO
    train.train(tmp_path / 'first', transitions=1024, seed=1, agents=1, kl_coef=0, device='cpu')
    first = RecurrentPPO.load(tmp_path / 'first/latest.zip', device='cpu')
    monkeypatch.setattr(train, 'RECOVERY_EVERY', 2)
    summary = train.train(tmp_path / 'more', transitions=2048, seed=2, agents=1, kl_coef=0, device='cpu',
                          resume=tmp_path / 'first/latest.zip')
    assert summary['resumed'] == dict(num_timesteps=1024, ppo_updates=4, optimizer_steps=[16])
    assert summary['critic_warmup'] == 0 and summary['ppo_transitions'] == 2048 and summary['total_timesteps'] == 3072
    assert summary['actor_gradient_steps'] == summary['actor_gradient_steps_full'] == 32
    assert {k: summary['learner'][k] for k in ('learning_rate', 'ent_coef', 'gae_lambda', 'target_kl', 'kl_weight')} == \
        dict(learning_rate=train.config.LEARNER['learning_rate'], ent_coef=train.config.LEARNER['ent_coef'],
             gae_lambda=train.config.LEARNER['gae_lambda'], target_kl=None, kl_weight=0.)
    # The saved run is the starting point exactly; the optimizer carries on from its 16 steps.
    start = RecurrentPPO.load(tmp_path / 'more/initial.zip', device='cpu')
    assert all(torch.equal(v, start.policy.state_dict()[k]) for k, v in first.policy.state_dict().items())
    after = RecurrentPPO.load(tmp_path / 'more/latest.zip', device='cpu').policy.optimizer.state_dict()['state']
    assert {int(v['step']) for v in after.values()} == {48}
    # Rollouts keep the run's numbering (1 and 2 here, so no 8th is sampled); a reset count would sample 0.
    assert not (tmp_path / 'more/rollouts.npz').exists()
    # A recovery checkpoint is the model after its last finished update, not one taken mid-rollout.
    assert json.loads((tmp_path / 'more/recovery.json').read_text()) == dict(num_timesteps=2048, ppo_updates=8,
                                                                          optimizer_steps=16)
    recovered = RecurrentPPO.load(tmp_path / 'more/recovery.zip', device='cpu')
    assert recovered.num_timesteps == 2048 and {int(v['step']) for v in recovered.policy.optimizer.state_dict()['state'].values()} == {32}
    with pytest.raises(ValueError):
        train.train(tmp_path / 'bad', transitions=1024, seed=2, agents=1, kl_coef=.5, device='cpu',
                    resume=tmp_path / 'first/latest.zip')


def test_a_goal_run_trains_on_balanced_goals_and_evaluates_only_with_goal_inputs(monkeypatch, tmp_path):
    """The goal task end to end (goal-v2 inputs, the two native goal fixtures): the goal input in the
    model, goals in balanced random pairs, and no evaluation that would zero-fill or drop the goal of
    a mismatched checkpoint."""
    from sb3_contrib import RecurrentPPO
    from baseline.tests.test_goal_task import GoalWorld
    def connect(record, agents=1, spectator=None):
        return [(world, world.control) for world in [GoalWorld('alternate') for _ in range(agents)]]
    def make_episode(transport, service, fixture, record):
        transport.record = record
        return ResourceEpisode(transport, service, CFG, 2, 10, record, goal=fixture.get('goal'))
    for module in (train, evaluate):
        monkeypatch.setattr(module, 'connect', connect)
        monkeypatch.setattr(module, 'make_episode', make_episode)
        monkeypatch.setattr(module, 'delivery_report', lambda service, min_windows: dict(ok=True, windows=[]))
    goals = train.config.GOAL_FIXTURES
    with pytest.raises(ValueError):
        train.train(tmp_path / 'mixed', transitions=0, seed=1, agents=1, kl_coef=0, device='cpu',
                    fixtures=[goals[0], train.config.TRAIN_FIXTURES[0]])
    summary = train.train(tmp_path / 'goal', transitions=1024, seed=1, agents=1, kl_coef=0, device='cpu', fixtures=goals)
    assert summary['task'] == 'goal-activity-v2' and summary['status'] == 'completed'
    rows = list(read_jsonl(tmp_path / 'goal/episodes.jsonl.gz'))
    order = [r['fixture']['goal'] for r in rows if r['kind'] == 'scheduled_episode']
    pairs = [tuple(order[i:i + 2]) for i in range(0, len(order) - 1, 2)]
    assert all(set(p) == {'wood', 'fish'} for p in pairs) and len(set(pairs)) == 2  # balanced, both orders
    assert all(r['goal'] in ('wood', 'fish') for r in rows if r['kind'] == 'episode_measures')
    checkpoint = tmp_path / 'goal/latest.zip'
    assert 'goal_vec' in RecurrentPPO.load(checkpoint, device='cpu').observation_space.spaces
    results = evaluate.evaluate(tmp_path / 'eval', policy='checkpoint', checkpoint=checkpoint,
                                fixtures=('goal_wood', 'goal_fish'), trials=1)
    assert [(t['goal'], t['goal_return'] == t['reward']) for t in results['trials']] == [('wood', True), ('fish', True)]
    assert all(set(t['produced_by_activity']) == set(GOAL_FAMILIES) for t in results['trials'])
    train.train(tmp_path / 'value', transitions=0, seed=1, agents=1, kl_coef=0, device='cpu')
    for model, fixtures in ((checkpoint, ('short',)), (tmp_path / 'value/latest.zip', ('goal_wood',))):
        with pytest.raises(ValueError):
            evaluate.evaluate(tmp_path / f'refused-{len(fixtures[0])}', policy='checkpoint', checkpoint=model,
                              fixtures=fixtures, trials=1)


def test_fresh_worlds_give_every_trial_a_world_of_its_own(worlds, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'worlds', lambda: [{}] * 3)
    results = evaluate.evaluate(tmp_path / 'fresh', policy='wait', fixtures=('short', 'long'), trials=1,
                                fresh_worlds=True, first_trial=5)
    assert [w.epoch for w in worlds] == [1, 1, 0] and all(w.closed for w in worlds)
    assert [t['trial'] for t in results['trials']] == [5, 5] and results['fresh_worlds']
    with pytest.raises(ValueError, match='need as many worlds'):
        evaluate.evaluate(tmp_path / 'too-few', policy='wait', fixtures=('short', 'long'), trials=2, fresh_worlds=True)
