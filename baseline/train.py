"""Train the agent natively: recurrent PPO, optionally held near a saved policy by a KL penalty.

Runs inside worlds booted by scripts/stack.py (see docs/RUNNING.md), one agent
per world (--worlds N for N agents); agents rotate through config.TRAIN_FIXTURES,
or through --fixtures.
From a pretrained policy (--init), PPO first trains only the critic, then fine-tunes gently (config.FINETUNE).
--resume continues a run instead: its policy, critic, optimizer state, settings and step count, with nothing changed.
The behavior-cloning warm start and the start archive were removed. Writes initial.zip,
checkpoints/, latest.zip, metrics/, episodes.jsonl.gz (one
line per decision 'effect' and per episode 'episode_measures', see baseline/telemetry.py),
traces.jsonl.gz (full rows of selected episodes), rollouts.npz (sampled PPO rollout buffers) and
summary.json (with the run manifest and the learner settings read back from the model).

The native transfer study runs with --study SET --weights ZIP --kl-coef 0 inside a stack with a world
service (scripts/stack.py --engine): config.LEARNER's fresh model with only the saved policy and critic tensors
loaded (fresh optimizer, schedules, counters, buffers and recurrent state; its outputs are checked against the
saved model on the first rollout); slot i always asks for goal GOALS[i % 4], so every update holds equal
transitions per goal; every episode draws a training start record of SET and runs in a fresh world. It saves
checkpoints/decisions-<n>.zip at the --checkpoints decision counts (complete-update boundaries), a recovery state
after every update (recovery/, the model with its optimizer and counters, and the random states) and
updates.jsonl (per update: decisions, times, episodes, checks). --validate checks each rollout before its update
(check_rollout) and each update after it.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from baseline import config
from baseline.resource_task import GOAL_FAMILIES, GOAL_SCORING, GOAL_VEC, GOALS, VALUE_SCORING
from baseline.telemetry import EpisodeRecorder, TraceSink, manifest, sha256
from baseline.world import FreshWorlds, JsonlEvidence, connect, delivery_report, join, make_episode
from baseline.task_schedule import FreshWorldSchedule, ParallelSchedules, ScheduledResourceBackend
from obs import OBSERVATION, project

# A full trace is several KB per decision: training keeps each decision's effect and each
# episode's measures and outcome; TraceSink keeps the full rows of selected episodes.
TRAINING_EVENTS = {'task_terminal', 'collection_tail', 'scheduled_episode', 'episode_measures', 'effect'}
# Every 8th rollout buffer (the first included) is saved as the update used it: rewards, critic
# values, returns and advantages, so a run's credit assignment can be read later without
# recomputing it with the final critic. Row t of rollout r is decision r * n_steps + t of each agent.
ROLLOUT_SAMPLE_EVERY = 8
# recovery.zip (and recovery.json, its counters) is written as every 16th rollout starts: after an
# update has finished, unlike checkpoints/, which SB3 saves mid-rollout. --resume takes it.
RECOVERY_EVERY = 16
BUFFER_FIELDS = ('rewards', 'values', 'returns', 'advantages', 'episode_starts', 'actions', 'log_probs')
# Tensors of the critic (value network) in MaskedRecurrentPolicy's state; the rest belong to the actor.
CRITIC = ('vf_features_extractor.', 'lstm_critic.', 'value_net.', 'mlp_extractor.value_net.')


def build_model(environment, *, init, seed, device):
    from sb3_contrib import RecurrentPPO
    from baseline.policy import CONTRACT, MaskedRecurrentPolicy, check_names, load_name_vectors
    from names import name_table
    if init is not None:
        # Policy, critic and optimizer state, and the saved PPO settings and step count.
        model = RecurrentPPO.load(init, env=environment, device=device, seed=seed)
        check_names(model.policy)
        return model
    learner = deepcopy(config.LEARNER)
    table = name_table()
    learner['policy_kwargs']['names'] = table.spec
    learner['policy_kwargs']['observation'] = OBSERVATION
    learner['policy_kwargs']['contract'] = CONTRACT
    model = RecurrentPPO(MaskedRecurrentPolicy, environment, seed=seed, device=device, **learner)
    load_name_vectors(model.policy, table.vectors)
    return model


def load_weights(model, path):
    """Weights-only initialization: every policy and critic tensor of a saved model (the frozen name table
    included) into `model`, built fresh from config.LEARNER, whose optimizer, schedules, step counters, rollout
    buffer and recurrent state stay new. The saved model must have the same network (tensor names, shapes and
    dtypes, and network inputs); its saved PPO settings and optimizer are read for the record only.
    Returns the source model, for the check of its outputs (check_rollout)."""
    import torch
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    from baseline.reference import KEY
    source = RecurrentPPO.load(path, device=model.device)
    check_names(source.policy)
    inputs = lambda m: {k: (v.shape, str(v.dtype)) for k, v in m.observation_space.spaces.items() if k != KEY}
    if inputs(source) != inputs(model):
        raise ValueError(f'{path} has other network inputs than this task')
    mine, theirs = model.policy.state_dict(), source.policy.state_dict()
    if {k: (v.shape, v.dtype) for k, v in mine.items()} != {k: (v.shape, v.dtype) for k, v in theirs.items()}:
        raise ValueError(f'{path} has another network than config.LEARNER builds')
    model.policy.load_state_dict(theirs, strict=True)
    if not all(torch.equal(v, theirs[k]) for k, v in model.policy.state_dict().items()):
        raise RuntimeError(f'the weights of {path} did not load exactly')
    source.policy.set_training_mode(False)
    adam = [int(v['step']) for v in source.policy.optimizer.state_dict()['state'].values() if 'step' in v]
    saved = dict(path=str(path), sha256=sha256(path), tensors=len(theirs), num_timesteps=source.num_timesteps,
                 ppo_updates=source._n_updates, optimizer_steps=max(adam, default=0),
                 settings={k: getattr(source, k) for k in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
                                                           'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl')})
    return source, saved


# The checks of a collected rollout (--validate), before PPO uses it: recomputed values and log-probabilities must
# equal the collected ones within ATOL (float32, one device); the rest is exact.
ATOL = 1e-5


class CheckFailed(RuntimeError):
    """A --validate check failed: the run stops before PPO uses the rollout."""


def replay(policy, buffer, steps=None):
    """Each step's log-probability of the collected action and value, recomputed by `policy` from the stored
    observations and LSTM states; and the largest change at an episode start when the stored state is zeroed
    (which must be none: a new episode starts from zero memory)."""
    import torch
    device, gap = policy.device, 0.
    logp, values = [], []
    tensor = lambda a: torch.as_tensor(a, device=device)
    with torch.no_grad():
        for t in range(buffer.buffer_size if steps is None else steps):
            obs = {k: tensor(v[t]) for k, v in buffer.observations.items() if k in policy.observation_space.spaces}
            pi = (tensor(buffer.hidden_states_pi[t]), tensor(buffer.cell_states_pi[t]))
            vf = (tensor(buffer.hidden_states_vf[t]), tensor(buffer.cell_states_vf[t]))
            starts = tensor(buffer.episode_starts[t]).float()
            dist, _ = policy.get_distribution(obs, pi, starts)
            value = policy.predict_values(obs, vf, starts).flatten()
            logp.append(dist.log_prob(tensor(buffer.actions[t, :, 0]).long()).cpu().numpy())
            values.append(value.cpu().numpy())
            if bool(starts.any()):
                keep = (1 - starts).view(1, -1, 1)
                dist0, _ = policy.get_distribution(obs, tuple(s * keep for s in pi), starts)
                value0 = policy.predict_values(obs, tuple(s * keep for s in vf), starts).flatten()
                gap = max(gap, float((dist0.distribution.logits - dist.distribution.logits).abs().max()),
                          float((value0 - value).abs().max()))
    return np.stack(logp), np.stack(values), gap


def check_rollout(model, carried, goals, source=None):
    """The rollout just collected, before its update: finite observations and legal masks, every sampled action
    legal, the per-slot goals (slot i asks for goals[i]) and so equal transitions per goal, episode starts equal
    to the previous steps' terminal flags (`carried['dones']`, one row per step, the last of the previous
    rollout first), and the stored LSTM states reproducing every collected log-probability and value, from zero
    at an episode start. With `source` (the first rollout of a weights-only run) the saved model's outputs on the
    same inputs must equal the collected ones. Rewards settle per episode in the schedule. Returns the counts;
    raises CheckFailed."""
    b = model.rollout_buffer
    obs, actions, problems = b.observations, b.actions[..., 0].astype(np.int64), []
    for k, v in obs.items():
        if not np.isfinite(v).all():
            problems.append(f'{k}: non-finite values')
    mask = obs['mask']
    if not (np.isin(mask, (0, 1)).all() and mask.any(-1).all()):
        problems.append('mask: not 0/1 or a step with no legal action')
    illegal = int((np.take_along_axis(mask, actions[..., None], -1)[..., 0] != 1).sum())
    if illegal:
        problems.append(f'{illegal} sampled actions masked illegal')
    goal_of_slot = np.asarray([GOALS.index(g) for g in goals])
    if not ((obs['goal_vec'].argmax(-1) == goal_of_slot).all() and (obs['goal_vec'].sum(-1) == 1).all()):
        problems.append('a slot saw a goal other than its own')
    per_goal = np.bincount(np.broadcast_to(goal_of_slot, actions.shape).reshape(-1), minlength=len(GOALS)).tolist()
    if len(set(per_goal)) != 1:
        problems.append(f'unequal transitions per goal: {per_goal}')
    dones = np.asarray(carried['dones'], np.float32)
    if dones.shape != b.episode_starts.shape or not np.array_equal(b.episode_starts, dones):
        problems.append("episode starts differ from the previous steps' terminal flags")
    if not np.isfinite(b.rewards).all():
        problems.append('non-finite rewards')
    logp, values, reset_gap = replay(model.policy, b)
    gap = max(float(np.abs(logp - b.log_probs).max()), float(np.abs(values - b.values).max()))
    if gap > ATOL:
        problems.append(f'stored LSTM states reproduce the collected outputs only within {gap:.3g}')
    if reset_gap > ATOL:
        problems.append(f'an episode start depends on the stored state ({reset_gap:.3g})')
    source_gap = None
    if source is not None:
        logp0, values0, _ = replay(source.policy, b)
        source_gap = max(float(np.abs(logp0 - b.log_probs).max()), float(np.abs(values0 - b.values).max()))
        if source_gap > ATOL:
            problems.append(f'the saved model differs from the loaded one by {source_gap:.3g}')
    counts = dict(decisions=int(actions.size), illegal_actions=illegal, transitions_per_goal=per_goal,
                  episode_starts=int(b.episode_starts.sum()), recurrent_max_gap=gap, reset_max_gap=reset_gap,
                  source_max_gap=source_gap, problems=problems)
    if problems:
        raise CheckFailed('; '.join(problems))
    return counts


def study_slots(study, agents, seed, worlds, record, traces):
    """One FreshWorldSchedule per training slot of start set `study`: slot i asks for GOALS[i % 4]; each
    episode draws one of the set's training start records from the slot's own stream
    (default_rng([STUDY_TRAIN_SEED, seed, i])), whatever its goal, and runs in a fresh world of the set's build."""
    start_set = config.START_SETS[study]
    if tuple(start_set['goals']) != GOALS or agents % len(GOALS):
        raise ValueError(f'study training needs all four goals and a multiple of {len(GOALS)} agents')
    records, tree = config.study_records(study, 'train'), config.ENGINE_TREES[start_set['engine']]
    streams = [np.random.default_rng([config.STUDY_TRAIN_SEED, seed, i]) for i in range(agents)]
    slots = []
    for i in range(agents):
        goal = GOALS[i % len(GOALS)]
        base = dict(id=f'study-train-{study}-{goal}', kind='goal', horizon=240, capacity='declared', goal=goal,
                    start_set=study)
        recorder = EpisodeRecorder(record, traces)
        agent_record = lambda kind, i=i, recorder=recorder, **data: recorder(kind, agent=i, **data)
        choose = lambda k, base=base, rng=streams[i]: dict(base, **records[int(rng.integers(len(records)))])
        def open_episode(world, fixture, episode_record, i=i):
            if world['tree'] != tree:
                raise RuntimeError(f"world {world['id']} runs engine tree {world['tree']}, not {study}'s {tree}")
            transport, service = join(world, record, i)
            return make_episode(transport, service, fixture, episode_record)
        slots.append(FreshWorldSchedule(choose, worlds, start_set['engine'], open_episode, agent_record))
    return slots, streams


def schedule_for(index, transport, service, record, fixtures=config.TRAIN_FIXTURES, traces=None, shuffle_seed=None):
    """Agent `index` starts at a different point in the fixture rotation than its neighbours;
    with shuffle_seed each pass over the rotation goes in its own random order (goal tasks).
    Every row passes through an EpisodeRecorder (episode ids, effects, measures, traces)."""
    fixtures = fixtures[index % len(fixtures):] + fixtures[:index % len(fixtures)]
    recorder = EpisodeRecorder(record, traces)
    def agent_record(kind, **data):
        recorder(kind, agent=index, **data)
    shuffle = None if shuffle_seed is None else np.random.default_rng([shuffle_seed, index])
    return ScheduledResourceBackend(fixtures, lambda fixture: make_episode(transport, service, fixture, agent_record),
                                    agent_record, transport.close, shuffle)


def train(output, *, transitions, seed, agents=config.AGENTS, kl_coef=config.KL_COEF, init=None, device='auto',
          fixtures=None, target_kl=config.FINETUNE['target_kl'], resume=None, simulated=None, weights=None, study=None,
          checkpoints=(), validate=False):
    """simulated: (agents, seed, fixtures) -> (backend, observation sample) replaces the native
    worlds with a simulated backend (sim/vec.py); the learner and everything after is unchanged.
    weights, study, checkpoints and validate: the native transfer study (module docstring)."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from baseline.adapter import GameVecEnv
    from baseline.reference import ReferenceProbs

    batch = config.LEARNER['n_steps'] * agents
    if transitions < 0 or transitions % batch:
        raise ValueError(f'transitions must be a non-negative multiple of n_steps x agents = {batch}')
    if resume is not None and (init is not None or kl_coef):
        raise ValueError('--resume continues a run as it was: no --init or KL penalty')
    if kl_coef and init is None:
        raise ValueError('the KL penalty needs a reference: give --init')
    if weights is not None and (init is not None or resume is not None or kl_coef):
        raise ValueError('--weights starts a fresh learner from saved weights: no --init, --resume or KL penalty')
    if study is not None and (fixtures or simulated is not None or init is not None):
        raise ValueError('--study trains on its own records: no --fixtures, simulator or --init')
    if validate and study is None:
        raise ValueError('--validate checks study rollouts')
    if any(c < 0 or c % batch for c in checkpoints):
        raise ValueError(f'checkpoints must be complete-update boundaries, multiples of {batch}')
    pretrained = init is not None
    rotation = fixtures or (config.STUDY_FIXTURES.values() if study else config.TRAIN_FIXTURES)
    goals = {f.get('goal') is not None for f in rotation}
    if len(goals) != 1:
        raise ValueError('goal fixtures and value fixtures cannot share a run: only one has the goal input')
    goal_task = goals == {True}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evidence = JsonlEvidence(output / 'episodes.jsonl.gz', kinds=TRAINING_EVENTS)
    traces = TraceSink(output / 'traces.jsonl.gz')
    summary = dict(status='started', transitions=transitions, seed=seed, agents=agents, kl_coef=kl_coef,
                   target_kl=target_kl if pretrained else None, study=study, checkpoints=sorted(checkpoints),
                   validate=validate, weights=str(weights) if weights else None, init=str(init) if init else None,
                   resume=str(resume) if resume else None,
                   fixtures=[f['id'] for f in fixtures] if fixtures else None,
                   task=GOAL_SCORING if goal_task else VALUE_SCORING,
                   goal_families={g: sorted(f) for g, f in GOAL_FAMILIES.items()} if goal_task else None,
                   manifest=manifest(output, init_sha256=sha256(init) if init else None,
                                     weights_sha256=sha256(weights) if weights else None,
                                     resume_sha256=sha256(resume) if resume else None))
    # Short episodes finish ~15x more often than long ones, so report them apart (restarts are
    # "drop"). final_value is the reward in base value: 0 after a death, whatever was held.
    outcomes, lock = dict(short=[], long=[]), threading.Lock()
    def record(kind, **data):
        evidence(kind, **data)
        if kind == 'episode_measures':
            with lock:
                outcomes.setdefault(data['fixture_kind'], []).append((data['final_value'], data['survived']))
            print(f"[{data['fixture_kind']} episode] agent {data.get('agent')} final value {data['final_value']}"
                  f" (held at end {data['value_held_at_end']}) survived {data['survived']} deposits {data['deposits']}",
                  flush=True)
    links, environment, rollouts, streams, updates_log = [], None, [], [], None
    try:
        torch.set_num_threads(1)
        if study is not None:
            worlds = FreshWorlds()
            summary['engines'] = worlds.identity()['engines']
            slots, streams = study_slots(study, agents, seed, worlds, record, traces)
            if resume is not None and resume.with_suffix('.json').exists():
                saved = json.loads(resume.with_suffix('.json').read_text(encoding='utf-8'))
                for stream, state in zip(streams, saved['slot_streams']):
                    stream.bit_generator.state = state
            # Observation schema comes from one reset, in a world of its own, before any task clock is armed.
            world = worlds.acquire(config.START_SETS[study]['engine'])
            transport, _ = join(world, record)
            try:
                sample = project(transport.reset(seed=0, cfg=deepcopy(config.TASK_CFG)))
            finally:
                transport.close()
                worlds.release(world['id'])
            sample.update(budget_vec=np.zeros(2, np.float32), critic_vec=np.zeros(1, np.float32),
                          goal_vec=np.zeros(len(GOAL_VEC), np.float32))
            # Each slot prepares its next episode while one runs (FreshWorldSchedule); a few worlds booted ahead
            # cover the preparations that start together.
            worlds.target(config.START_SETS[study]['engine'], max(2, agents // 8))
            backend = ParallelSchedules(slots)
        elif simulated is not None:
            if not goal_task:
                raise ValueError('the simulator runs goal fixtures')
            backend, sample = simulated(agents, seed, rotation)
        else:
            links = connect(record, agents)
            # Observation schema comes from one reset before any task clock is armed.
            raw = links[0][0].reset(seed=0, cfg=deepcopy(config.TASK_CFG))
            sample = project(raw)
            sample.update(budget_vec=np.zeros(2, np.float32), critic_vec=np.zeros(1, np.float32))
            if goal_task:
                sample['goal_vec'] = np.zeros(len(GOAL_VEC), np.float32)
            backend = ParallelSchedules([schedule_for(i, transport, service, record, rotation, traces,
                                                      seed if goal_task else None)
                                         for i, (transport, service) in enumerate(links)])
        environment = ReferenceProbs(GameVecEnv(backend, sample))
        model = build_model(environment, init=init if resume is None else resume, seed=seed, device=device)
        source = None
        if weights is not None:
            source, summary['weights_source'] = load_weights(model, weights)
        if resume is not None and resume.with_suffix('.json').exists():
            saved = json.loads(resume.with_suffix('.json').read_text(encoding='utf-8'))
            if saved['sha256'] != sha256(resume):
                raise ValueError(f'{resume} is not the model its recovery record names')
            torch.set_rng_state(torch.tensor(saved['torch_rng'], dtype=torch.uint8))
            summary['recovery_record'] = {k: v for k, v in saved.items() if k not in ('torch_rng', 'slot_streams')}
        if resume is not None:
            adam = {int(v['step']) for v in model.policy.optimizer.state_dict()['state'].values()}
            summary['resumed'] = dict(num_timesteps=model.num_timesteps, ppo_updates=model._n_updates,
                                      optimizer_steps=sorted(adam))
        model.save(output / 'initial.zip')
        def set_learning_rate(rate):
            model.learning_rate = rate
            model._setup_lr_schedule()
        if pretrained:
            set_learning_rate(config.FINETUNE['learning_rate'])
            # Ends an update once a minibatch's KL passes 1.5 x target_kl; None never ends one
            # early. The KL penalty to the reference (kl_coef) is separate and stays.
            model.target_kl = target_kl
            model.ent_coef = config.FINETUNE['ent_coef']
            model.gae_lambda = model.rollout_buffer.gae_lambda = config.FINETUNE['gae_lambda']
        if kl_coef:
            environment.set_reference(model.policy)
            model.policy.kl_weight = kl_coef / model.ent_coef
        # What PPO will actually use once the actor trains, read back from the model.
        summary['learner'] = dict(
            learning_rate=model.lr_schedule(1.), n_steps=model.n_steps, batch_size=model.batch_size,
            n_epochs=model.n_epochs, gamma=model.gamma, gae_lambda=model.rollout_buffer.gae_lambda,
            clip_range=model.clip_range(1.), ent_coef=model.ent_coef, vf_coef=model.vf_coef,
            max_grad_norm=model.max_grad_norm, target_kl=model.target_kl, kl_weight=model.policy.kl_weight,
            lstm_hidden_size=model.policy.lstm_actor.hidden_size)
        model.set_logger(configure(str(output / 'metrics'), ['stdout', 'json', 'csv']))
        before = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        # SB3 ends an update early when a minibatch's KL passes 1.5 x target_kl and logs nothing
        # about it: count the optimizer steps each update actually takes.
        optimizer_steps, optimizer_step = [0], model.policy.optimizer.step
        def counted_step(*args, **kwargs):
            optimizer_steps[0] += 1
            return optimizer_step(*args, **kwargs)
        model.policy.optimizer.step = counted_step
        full_update = model.n_epochs * -(-batch // model.batch_size)
        goals = [GOALS[i % len(GOALS)] for i in range(agents)]
        updates_log = (output / 'updates.jsonl').open('x', encoding='utf-8', buffering=1) if study else None
        clock = dict(rollout=time.monotonic(), checked=None)
        carried = dict(dones=[np.ones(agents, np.float32)])
        ended = dict(episodes=Counter(), returns=Counter(), deaths=Counter())
        def atomic(path, write):
            temporary = path.with_name(path.name + '.part')
            write(temporary)
            os.replace(temporary, path)
        def save_checkpoint(decisions):
            if decisions in checkpoints:
                (output / 'checkpoints').mkdir(exist_ok=True)
                atomic(output / 'checkpoints' / f'decisions-{decisions}.zip', lambda path: model.save(path))
        def save_recovery():
            """The state after a complete update: the model with its optimizer and counters, and the random
            states; the record is written after the model it names, and replaces the previous pair."""
            folder = output / 'recovery'
            folder.mkdir(exist_ok=True)
            name = f'update-{model._n_updates:05d}'
            atomic(folder / f'{name}.zip', lambda path: model.save(path))
            state = dict(num_timesteps=model.num_timesteps, ppo_updates=model._n_updates,
                         optimizer_steps=optimizer_steps[0], sha256=sha256(folder / f'{name}.zip'),
                         torch_rng=torch.get_rng_state().tolist(), slot_streams=[g.bit_generator.state for g in streams])
            atomic(folder / f'{name}.json', lambda path: path.write_text(json.dumps(state) + '\n', encoding='utf-8'))
            for old in folder.glob('update-*'):
                if not old.name.startswith(name):
                    old.unlink()
        def check_update():
            """After an update: every parameter finite and the losses SB3 logged for it finite; after the first,
            both the actor's and the critic's tensors changed."""
            values = model.logger.name_to_value
            losses = {k: values.get(k) for k in ('train/loss', 'train/value_loss', 'train/policy_gradient_loss',
                                                  'train/entropy_loss')}
            finite = all(v is not None and np.isfinite(v) for v in losses.values()) and \
                all(bool(torch.isfinite(t).all()) for t in model.policy.parameters())
            row = dict(losses=losses, finite=finite)
            if model._n_updates == Progress.first_update + model.n_epochs:
                changed = {k for k, v in model.policy.state_dict().items() if not torch.equal(before[k], v.detach().cpu())}
                row.update(actor_changed=sum(not k.startswith(CRITIC) for k in changed),
                           critic_changed=sum(k.startswith(CRITIC) for k in changed))
                if validate and not (row['actor_changed'] and row['critic_changed']):
                    raise CheckFailed(f'the first update left the actor or the critic unchanged: {row}')
            if validate and not finite:
                raise CheckFailed(f'non-finite update: {row}')
            return row
        class Progress(BaseCallback):
            steps = updates = first_update = 0
            def _on_step(self):
                if study:
                    dones = np.asarray(self.locals['dones'], np.float32)
                    carried['dones'].append(dones)
                    for i in np.flatnonzero(dones):
                        info = self.locals['infos'][i]
                        ended['episodes'][goals[i]] += 1
                        ended['returns'][goals[i]] += float(info['episode']['r'])
                        ended['deaths'][goals[i]] += not info['final_info']['resource_report']['survived']
                    if self.n_calls % 64 == 0:
                        atomic(output / 'progress.json', lambda path: path.write_text(json.dumps(dict(
                            num_timesteps=self.model.num_timesteps, ppo_updates=self.model._n_updates,
                            seconds=time.monotonic() - started)) + '\n', encoding='utf-8'))
                return True
            def _record(self):
                log = self.model.policy.kl_log
                if log:
                    self.logger.record('train/kl_to_reference', float(np.mean(log)))
                    log.clear()
                # Recorded after a rollout, so these describe the update before it, as SB3's train/ rows do.
                if self.model._n_updates > Progress.updates:
                    self.logger.record('train/epochs_started', self.model._n_updates - Progress.updates)
                    self.logger.record('train/gradient_step_share', (optimizer_steps[0] - Progress.steps) / full_update)
                Progress.steps, Progress.updates = optimizer_steps[0], self.model._n_updates
                with lock:
                    recent = {kind: values[-20:] for kind, values in outcomes.items()}
                for kind, values in recent.items():
                    if values:
                        self.logger.record(f'rollout/final_value_{kind}_recent', float(np.mean([v for v, _ in values])))
                        self.logger.record(f'rollout/death_share_{kind}_recent', float(np.mean([not s for _, s in values])))
            def _on_rollout_start(self):
                if study:
                    self._finish_update()
                    clock['rollout'] = time.monotonic()
                    return
                rollouts = self.model.num_timesteps // (self.model.n_steps * self.model.n_envs)
                if self.model.num_timesteps > prior and rollouts % RECOVERY_EVERY == 0:
                    self.model.save(output / 'recovery.zip')
                    (output / 'recovery.json').write_text(json.dumps(dict(
                        num_timesteps=self.model.num_timesteps, ppo_updates=self.model._n_updates,
                        optimizer_steps=optimizer_steps[0])) + '\n', encoding='utf-8')
            def _finish_update(self):
                """The update just made (none before the first rollout): its checks, the recovery state and any
                checkpoint at this decision count, and its row of updates.jsonl."""
                row = clock.pop('row', None)
                if row is not None:
                    row.update(update_seconds=round(time.monotonic() - clock['checked'], 3), **check_update())
                saved = time.monotonic()
                if row is not None:
                    save_recovery()
                save_checkpoint(self.model.num_timesteps)
                if row is not None:
                    row['saved_seconds'] = round(time.monotonic() - saved, 3)
                    updates_log.write(json.dumps(row) + '\n')
            def _on_rollout_end(self):
                if study:
                    collected = time.monotonic()
                    dones = carried['dones']
                    row = dict(num_timesteps=self.model.num_timesteps, ppo_update=self.model._n_updates // self.model.n_epochs + 1,
                               collect_seconds=round(collected - clock['rollout'], 3),
                               episodes={g: ended['episodes'][g] for g in GOALS},
                               mean_return={g: ended['returns'][g] / max(1, ended['episodes'][g]) for g in GOALS},
                               deaths={g: ended['deaths'][g] for g in GOALS})
                    for counter in ended.values():
                        counter.clear()
                    if validate:
                        carried['dones'] = np.stack(dones[:-1])
                        checked = time.monotonic()
                        first = source is not None and self.model._n_updates == Progress.first_update
                        row['checks'] = check_rollout(self.model, carried, goals, source if first else None)
                        row['check_seconds'] = round(time.monotonic() - checked, 3)
                        print(f"[check] update {row['ppo_update']}: {row['checks']['transitions_per_goal']} transitions a goal,"
                              f" recurrent gap {row['checks']['recurrent_max_gap']:.2g}, source gap {row['checks']['source_max_gap']}",
                              flush=True)
                    carried['dones'] = [dones[-1]]
                    clock.update(row=row, checked=time.monotonic())
                # SB3 has just computed this rollout's returns and advantages; the update follows.
                buffer = self.model.rollout_buffer
                index = self.model.num_timesteps // (buffer.buffer_size * buffer.n_envs) - 1
                if index % ROLLOUT_SAMPLE_EVERY == 0:
                    rollouts.append(dict(rollout=index, **{k: getattr(buffer, k).copy() for k in BUFFER_FIELDS}))
                # SB3 dumps the previous update's metrics after this rollout, so the KL lines up with them.
                self._record()
            def _on_training_end(self):
                if study:
                    self._finish_update()
                self._record()
                self.logger.dump(self.num_timesteps)
        # The study saves its declared checkpoints at complete-update boundaries, not every CHECKPOINT_EVERY.
        callbacks = CallbackList([Progress()] if study else [Progress(), CheckpointCallback(
            max(1, config.CHECKPOINT_EVERY // agents), str(output / 'checkpoints'), name_prefix='ppo')])
        warmup = min(config.FINETUNE['critic_warmup'], transitions) // batch * batch if pretrained else 0
        summary.update(critic_warmup=warmup, actor_transitions=transitions - warmup)
        print(f'[train] {warmup} critic-only transitions, then {transitions - warmup} that train the actor', flush=True)
        if pretrained and transitions and transitions <= warmup:
            print('[warning] the actor will not train: a pretrained run trains only the critic for its first '
                  f"{config.FINETUNE['critic_warmup']} transitions", flush=True)
        actor = [model.policy.features_extractor, model.policy.lstm_actor,
                 model.policy.mlp_extractor.policy_net, model.policy.action_net,
                 model.policy.slot_embedding, model.policy.slot_options,
                 *(getattr(model.policy, f'{name}_scorer') for name in dict.fromkeys(p[0] for p in model.policy.pointers))]
        def actor_trainable(flag):
            for module in actor:
                for parameter in module.parameters():
                    parameter.requires_grad_(flag)
        prior = model.num_timesteps
        Progress.updates = Progress.first_update = model._n_updates
        if warmup:
            # The critic learns the imitation policy's own returns first, at full rate:
            # the actor is frozen, so nothing else can move.
            actor_trainable(False)
            set_learning_rate(config.FINETUNE['critic_learning_rate'])
            model.learn(total_timesteps=warmup, callback=callbacks, reset_num_timesteps=init is None)
            actor_trainable(True)
            set_learning_rate(config.FINETUNE['learning_rate'])
        actor_steps = optimizer_steps[0]
        if transitions - warmup:
            model.learn(total_timesteps=transitions - warmup, callback=callbacks,
                        reset_num_timesteps=init is None and resume is None and not warmup)
        model.save(output / 'latest.zip')
        summary.update(ppo_transitions=model.num_timesteps - prior, total_timesteps=model.num_timesteps,
                       ppo_updates=model._n_updates,
                       completed_episodes={kind: len(values) for kind, values in outcomes.items()},
                       changed_tensors=sum(not torch.equal(before[k], v.detach().cpu())
                                           for k, v in model.policy.state_dict().items()),
                       # Optimizer steps after the critic warmup, against a full n_epochs pass per update.
                       actor_gradient_steps=optimizer_steps[0] - actor_steps,
                       actor_gradient_steps_full=(transitions - warmup) // batch * full_update,
                       delivery=[delivery_report(service, min_windows=3) for _, service in links],
                       simulated=backend.counts() if simulated is not None else None,
                       checkpoints_saved=sorted(int(p.stem.split('-')[1]) for p in (output / 'checkpoints').glob('decisions-*.zip')),
                       status='completed')
        if not all(world['ok'] for world in summary['delivery']):
            print('[warning] world tick delivery left 95-105% during training; see summary.json', flush=True)
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if environment is not None: environment.close()
            else:
                for transport, _ in links: transport.close()
        finally:
            evidence.close()
            traces.close()
            if updates_log is not None:
                updates_log.close()
            if rollouts:
                np.savez_compressed(output / 'rollouts.npz', **{k: np.stack([r[k] for r in rollouts]) for k in rollouts[0]})
            summary.update(traces=traces.state(), seconds=time.monotonic() - started)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transitions', type=int, default=config.TRANSITIONS,
                        help='PPO environment steps across all agents; 0 trains behavior cloning only')
    parser.add_argument('--agents', type=int, default=config.AGENTS,
                        help='parallel agents, each in its own world (scripts/stack.py --worlds N)')
    parser.add_argument('--kl-coef', type=float, default=config.KL_COEF,
                        help='weight of KL(imitation || policy); 0 disables it')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--init', type=Path, help='continue from a saved RecurrentPPO .zip (also the KL reference)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--fixtures', nargs='+', help='training fixture ids PPO rotates through (default: all '
                        'value-scored ones); goal-v1 runs give train-goal-wood train-goal-fish')
    parser.add_argument('--target-kl', type=lambda v: None if v == 'none' else float(v), default=config.FINETUNE['target_kl'],
                        help='per-update KL limit for a pretrained run (SB3 target_kl); "none" removes it')
    parser.add_argument('--resume', type=Path,
                        help='continue this run as it was (policy, critic, optimizer, settings, steps); needs --kl-coef 0')

    parser.add_argument('--weights', type=Path,
                        help="a fresh config.LEARNER learner with only this saved model's policy and critic tensors; "
                             'needs --kl-coef 0')
    parser.add_argument('--study', choices=sorted(n for n, s in config.START_SETS.items() if len(s['goals']) == 4),
                        help="the native study's four-goal training on this start set, a fresh world per episode "
                             '(scripts/stack.py --engine)')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[],
                        help='decision counts to save checkpoints/decisions-<n>.zip at (complete-update boundaries)')
    parser.add_argument('--validate', action='store_true', help='check every study rollout before its update')
    args = parser.parse_args()
    known = {f['id']: f for f in (*config.TRAIN_FIXTURES, *config.GOAL_FIXTURES)}
    if any(name not in known for name in args.fixtures or ()):
        parser.error(f'--fixtures takes training fixture ids from {sorted(known)}')
    train(args.output, transitions=args.transitions, seed=args.seed, agents=args.agents, kl_coef=args.kl_coef,
          init=args.init, device=args.device, fixtures=[known[name] for name in args.fixtures] if args.fixtures else None,
          target_kl=args.target_kl, resume=args.resume, weights=args.weights, study=args.study,
          checkpoints=args.checkpoints, validate=args.validate)


if __name__ == '__main__':
    main()
