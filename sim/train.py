"""Simulator training on the device learner (sim/learn.py): config.LEARNER's fresh RecurrentPPO over a
batched sim/spec.py environment, with every per-decision tensor on one device.

    .venv-sim/bin/python -m sim.train train --env cpu --workers 4 --pool logs/sim/world/train-8-nc1-derived.npz \
        --output logs/sim/runs/check-s101 --worlds 32 --n-steps 32 --batch-size 256 --decisions 2048 --validate
    .venv-sim/bin/python -m sim.train bench --worlds 64 --updates 5 --out logs/sim/bench/learner-cpu.json
    .venv-sim/bin/python -m sim.train survey --env cpu --pool POOL --worlds 64 --n-steps 32 --rollouts 4

train writes initial.zip, checkpoints/, latest.zip (standard RecurrentPPO zips that baseline/evaluate.py
loads), metrics/ (SB3's json and csv), unsupported-*.json if a world reached unimplemented content (the
run stops before that update), and summary.json: settings read back from the model, the departures from
config.LEARNER, per-goal transitions and episodes, timing and memory. bench runs train for a fixed number
of updates in a scratch directory, checkpoints included, and prints the timing. survey collects without
updating and counts how often worlds reach content the simulator lacks.

Batching is the one departure from config.LEARNER (n_steps 1024 per world, batch 256, 4 epochs; the native
goal-v1 run used 4 worlds): BATCHING below is the default, chosen for a GPU, and every flag that changes it
is reported. Learning rate, gamma, lambda, clipping, entropy, value and gradient settings are LEARNER's.
--env cpu is the CPU reference (sim/workers.py) on a generated layout pool (`--pool`, a sim/world.py pool saved
with its derived tables; tables from logs/sim/data, names from RL_NAME_TABLE). --env fake (the default) is
sim/learn.FakeEnv, a spec stand-in with no private data. --goals declares the goal set (sim/spec.py goal_set): all
four goals (world slot b asks for GOALS[b % 4]), or one goal in every world slot, phase 1's specialists.
--validate checks every rollout before PPO uses it and the saved checkpoint after the run (validate_rollout,
check_checkpoint; checks.json): legal sampled actions on valid observations, each world slot's goal under the declared
set, terminal flags and recurrent resets, rewards settled to the audited returns, finite losses and complete updates.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import warnings

# JAX would otherwise reserve most of a GPU's memory before torch allocates any.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np  # noqa: E402

from baseline import config  # noqa: E402
from baseline.resource_task import GOAL_TARGETS, GOAL_VEC, GOALS  # noqa: E402
from baseline.telemetry import sha256  # noqa: E402
from sim.agent import HELD_CONTRACT  # noqa: E402
from sim.engine import RETALIATE_CONTRACT  # noqa: E402
from sim.spec import TERMINAL_DEATH, TERMINAL_HORIZON, TERMINAL_RUNNING, goal_set, slot_goal  # noqa: E402

# 1,024 worlds (256 per goal) x 128 decisions = 131,072 transitions per update; each 4,096-transition
# minibatch is 32 whole world-rollouts, 8 per goal; 4 epochs = 128 gradient steps per update.
BATCHING = dict(worlds=1024, n_steps=128, batch_size=4096, n_epochs=4)


def versions(device):
    import gymnasium
    import jax
    import jaxlib
    import sb3_contrib
    import stable_baselines3
    import torch
    result = dict(python=platform.python_version(), os=platform.platform(), machine=platform.machine(),
                  cpu=platform.processor(), cores=os.cpu_count(), numpy=np.__version__, torch=torch.__version__,
                  jax=jax.__version__, jaxlib=jaxlib.__version__, sb3=stable_baselines3.__version__,
                  sb3_contrib=sb3_contrib.__version__, gymnasium=gymnasium.__version__, device=str(device),
                  jax_devices=[str(d) for d in jax.devices()], torch_threads=torch.get_num_threads())
    if str(device).startswith('cuda'):
        result.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
    return result


def synchronize(device):
    """Wait for the device's queued work; returns the seconds waited (0 off CUDA, without importing torch)."""
    if str(device).startswith('cuda'):
        import torch
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    return 0.0


@contextmanager
def staged(events, name, device, **fields):
    """events(name, 'start' | 'end' | 'fail', **fields) around a block, synchronized at both ends so the
    seconds are completed device work; 'end' also gives the wait for the device at the block's end.
    Without events it only runs the block."""
    if events is None:
        yield
        return
    synchronize(device)
    events(name, 'start', **fields)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as error:
        events(name, 'fail', seconds=time.perf_counter() - t0, error=f'{type(error).__name__}: {error}', **fields)
        raise
    wait = synchronize(device)
    events(name, 'end', seconds=time.perf_counter() - t0, device_wait_seconds=wait, **fields)


def memory(device):
    """Peak (and on Linux current) resident memory of this process, and on a GPU the peak and current torch and
    JAX allocations, in MB."""
    import jax
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(peak_rss_mb=round(rss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10), 1))
    try:
        with open('/proc/self/statm') as statm:
            result['rss_mb'] = round(int(statm.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 20, 1)
    except OSError:
        pass
    if str(device).startswith('cuda'):
        stats = jax.devices()[0].memory_stats() or {}
        result.update(torch_peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                      torch_peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
                      torch_allocated_mb=round(torch.cuda.memory_allocated() / 2 ** 20, 1),
                      jax_peak_mb=round(stats.get('peak_bytes_in_use', 0) / 2 ** 20, 1),
                      jax_in_use_mb=round(stats.get('bytes_in_use', 0) / 2 ** 20, 1))
    return result


def make_env(name, worlds, seed, device, pool=None, goals=GOALS, tables=None, workers=None, task='goal-activity-v2',
             search_pool=None, engine='cpu', pack=None):
    """cpu: the CPU reference on the generated layouts of `pool` in `workers` processes (sim/workers.py; tables: the
    archive's path, which each worker loads, default logs/sim/data with its pins checked), or with engine 'native' the C
    port (sim/native.py NativeBatch on `pack`); fake: sim/learn.FakeEnv. goals: the declared goal set (sim/spec.py
    goal_set)."""
    from sim.learn import DeviceEnv, FakeEnv, HostDeviceEnv
    goals = goal_set(goals)
    if name == 'cpu':
        if pool is None:
            raise ValueError('--env cpu trains on generated worlds: give --pool')
        from sim.workers import CpuWorkers
        return HostDeviceEnv(CpuWorkers(worlds, workers or 1, pool, tables=tables, task=task, search_pool=search_pool,
                                        goals=goals, engine=engine, pack=pack), seed, device)
    if task != 'goal-activity-v2':
        raise ValueError(f'{task} runs on the CPU workers only (--env cpu)')
    from names import name_table
    return DeviceEnv(FakeEnv(worlds, id_rows=name_table().spec[0], goals=goals), seed, device)


def rollout_digest(buffer):
    """sha256 of each part of a collected rollout (observations, actions, rewards, episode starts, values,
    log-probabilities and the sim/spec.py info fields with the game effects), to compare two environments' rollouts."""
    from sim.spec import INFO_FIELDS
    parts = {f'obs/{k}': v for k, v in buffer.observations.items()}
    parts.update(actions=buffer.actions, rewards=buffer.rewards, episode_starts=buffer.episode_starts,
                 values=buffer._values, log_probs=buffer.log_probs,
                 **{f'info/{k}': v for k, v in buffer.infos.items() if k in INFO_FIELDS or k in ('made', 'kills')})
    return {k: hashlib.sha256(v[:buffer.pos].contiguous().cpu().numpy().tobytes()).hexdigest() for k, v in sorted(parts.items())}


# --validate: declared before the first checked run. Values and log-probabilities recomputed from the stored
# observations and LSTM states must equal the collected ones within ATOL (float32, same device); everything
# else is exact.
ATOL = 1e-5
# --segments, the learner's sequence-shape replay (replay_segments' sequence_*_gap): its ceiling in the in-run segment
# check, beside the one-step check at ATOL (decision of 28 October 2026: runs after phase 1). Loose on purpose: on a
# GPU the sequence arithmetic differs from the collection's (phase 1's worst over 25 arms: 9.3e-4 log-probability,
# 1.9e-4 value), so it catches a gross error in the learner's sequence reconstruction (sim/learn.py
# DeviceRolloutBuffer._get_samples), not rounding. Phase 1 ran under the frozen rule, the sequence replay recorded
# and never gating. The code cannot tell a phase 1 run from a later one and need not: the ceiling acts only inside a
# training process, phase 1's runs are finished, and the offline replay-segments command only reports the sequence
# gaps, so no frozen phase 1 record is checked again under it.
SEQUENCE_ATOL = 1e-2


class CheckFailed(RuntimeError):
    """A --validate check failed: the run stops before PPO uses the rollout."""


# Stored rows the recurrent check sends through the policy in one call (whole steps of every world).
VALIDATE_ROWS = 8192



def task_limits(env):
    """(horizons, targets) per goal (GOALS order) of a spec environment: CPU workers give their task version's
    (sim/combat.py task_spec); the fake environment plays goal-activity-v2 at one horizon."""
    if hasattr(env, 'horizons'):
        return tuple(env.horizons), tuple(env.targets)
    return (env.horizon,) * len(GOALS), tuple(GOAL_TARGETS[g] for g in GOALS)


def recurrent_gaps(model):
    """validate_rollout's recurrent-state checks on the rollout in the buffer: (gap, reset_gap). gap: the largest
    difference between each step's collected value and log-probability and those recomputed from its stored observation
    and LSTM states. reset_gap: at the episode starts, the largest difference between the outputs from the stored states
    and from zero states (the policy must ignore the stored state there). Every stored row is one single-step sequence
    from its own stored state, so on the CPU VALIDATE_ROWS rows (whole steps) go through the policy at once (a recomputed
    value may then differ from the collected one in its last bits). On a GPU each call keeps the collection's shape, one
    step of every world: there cuDNN's LSTM may run in TF32, whose result depends on the shape. The episode starts of the
    whole rollout are compared together after the steps, VALIDATE_ROWS rows a pair of calls: each zero-state output beside
    the stored-state output of a call of the same shape (a row without a start has the same input in both, so its
    difference is zero by construction). The maxima stay on the device until the end: two reads a rollout."""
    import torch
    b, policy = model.rollout_buffer, model.policy
    obs, actions = b.observations, b.actions[..., 0]
    chunk = 1 if actions.is_cuda else max(1, VALIDATE_ROWS // b.n_envs)
    flat = lambda store: store.reshape(-1, *store.shape[2:])
    state = lambda store, rows: flat(store)[rows].transpose(0, 1).contiguous()
    gap = reset_gap = torch.zeros((), device=actions.device)
    with torch.no_grad():
        for s0 in range(0, b.buffer_size, chunk):
            rows = slice(s0 * b.n_envs, min(b.buffer_size, s0 + chunk) * b.n_envs)
            o = {k: flat(v)[rows].to(b._dtypes[k]) for k, v in obs.items()}
            starts = flat(b.episode_starts)[rows]
            dist, _ = policy.get_distribution(o, (state(b.hidden_states_pi, rows), state(b.cell_states_pi, rows)), starts)
            values = policy.predict_values(o, (state(b.hidden_states_vf, rows), state(b.cell_states_vf, rows)), starts)
            gap = torch.maximum(gap, torch.maximum((dist.log_prob(flat(actions)[rows]) - flat(b.log_probs)[rows]).abs().max(),
                                                   (values.flatten() - flat(b._values)[rows]).abs().max()))
        first = flat(b.episode_starts).nonzero().squeeze(1)
        for r0 in range(0, len(first), VALIDATE_ROWS):
            rows = first[r0:r0 + VALIDATE_ROWS]
            o = {k: flat(v)[rows].to(b._dtypes[k]) for k, v in obs.items()}
            starts = flat(b.episode_starts)[rows]
            pi = (state(b.hidden_states_pi, rows), state(b.cell_states_pi, rows))
            vf = (state(b.hidden_states_vf, rows), state(b.cell_states_vf, rows))
            zero_pi, zero_vf = tuple(torch.zeros_like(t) for t in pi), tuple(torch.zeros_like(t) for t in vf)
            dist, _ = policy.get_distribution(o, pi, starts)
            dist0, _ = policy.get_distribution(o, zero_pi, starts)
            reset_gap = torch.maximum(reset_gap, torch.maximum(
                (dist0.distribution.logits - dist.distribution.logits).abs().max(),
                (policy.predict_values(o, zero_vf, starts) - policy.predict_values(o, vf, starts)).abs().max()))
    return float(gap), float(reset_gap)


def validate_rollout(model, carried, horizons, targets=None, goals=GOALS):
    """The checks on the rollout just collected (before its update). `carried` holds what spans rollouts:
    each world's reward paid so far in its episode and the done flags that end the previous rollout. horizons and
    targets: per goal (task_limits; a single horizon applies to every goal, with goal-activity-v2's targets). goals:
    the declared goal set; every step of world slot b must ask for slot_goal(goals, b), in its info and its goal_vec.
    Returns this rollout's counts; raises CheckFailed on the first violation."""
    import torch
    b = model.rollout_buffer
    obs, actions = b.observations, b.actions[..., 0]
    problems = []
    for k, v in obs.items():
        if v.dtype.is_floating_point and not bool(torch.isfinite(v).all()):
            problems.append(f'{k}: non-finite values')
        elif k.endswith(('_ids', '_op')) and not bool(((v >= 0) & (v <= 4000)).all()):
            problems.append(f'{k}: ids outside 0..4000')
    mask = obs['mask']
    if not bool(((mask == 0) | (mask == 1)).all() & mask.any(-1).all()):
        problems.append('mask: not 0/1 or a world with no legal action')
    illegal = int((mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1) != 1).sum())
    if illegal:
        problems.append(f'{illegal} sampled actions masked illegal')
    infos = b.infos
    terminal, ticks = infos['terminal'], infos['episode_ticks']
    if bool(infos['unsupported'].any()):
        problems.append('an unsupported world-step reached validation')
    expected = torch.cat([carried['done'][None], (terminal[:-1] != TERMINAL_RUNNING).float()])
    if not torch.equal(b.episode_starts, expected):
        problems.append('episode starts differ from the previous steps\' terminal flags')
    horizon_end, death = terminal == TERMINAL_HORIZON, terminal == TERMINAL_DEATH
    goal = infos['goal'].long()
    declared = torch.as_tensor([GOALS.index(slot_goal(goal_set(goals), w)) for w in range(goal.shape[1])], device=goal.device)
    if not bool((goal == declared).all()) or not torch.equal(obs['goal_vec'], torch.nn.functional.one_hot(
            declared, len(GOAL_VEC)).to(obs['goal_vec'].dtype).expand_as(obs['goal_vec'])):
        problems.append(f'a world slot asked for a goal outside the declared set {list(goals)}')
    if isinstance(horizons, int):
        horizons = (horizons,) * len(GOALS)
    targets = targets or tuple(GOAL_TARGETS[g] for g in GOALS)
    horizon = torch.as_tensor(horizons, device=goal.device)[goal]
    if not bool((ticks[horizon_end] == horizon[horizon_end]).all()) or not bool((ticks[death] <= horizon[death]).all()):
        problems.append('an episode ended at the wrong tick for its terminal reason')
    target = torch.as_tensor(targets, device=goal.device)[goal]
    audited = torch.where(horizon_end, torch.minimum(infos['outcomes'], target) / target, torch.zeros_like(target, dtype=torch.float32))
    ended = terminal != TERMINAL_RUNNING
    if not bool(torch.equal(infos['episode_return'][ended], audited[ended].to(torch.float32))):
        problems.append('an episode return differs from min(audited outcomes, target) / target')
    # Each world's paid rewards, summed step by step in float32 as they were paid; read once.
    paid = carried['paid']
    worst = torch.zeros((), device=paid.device)
    for step in range(b.buffer_size):
        paid = paid + b.rewards[step]
        end = ended[step]
        worst = torch.maximum(worst, torch.where(end, (paid - infos['episode_return'][step]).abs(), 0).max())
        paid = torch.where(end, torch.zeros_like(paid), paid)
    worst = float(worst)
    if worst > ATOL:
        problems.append(f'rewards of an episode sum to its return only within {worst:.3g}')
    gap, reset_gap = recurrent_gaps(model)
    if not gap <= ATOL:  # a NaN fails too
        problems.append(f'stored LSTM states reproduce the collected outputs only within {gap:.3g}')
    if not reset_gap <= ATOL:
        problems.append(f'an episode start depends on the stored state ({reset_gap:.3g})')
    carried.update(paid=paid, done=(terminal[-1] != TERMINAL_RUNNING).float())
    counts = dict(decisions=int(actions.numel()), illegal_actions=illegal, episodes_ended=int(ended.sum()),
                  horizon_ends=int(horizon_end.sum()), deaths=int(death.sum()), episode_starts=int(b.episode_starts.sum()),
                  settlement_max_error=worst, recurrent_max_gap=gap, reset_max_gap=reset_gap,
                  transitions_per_goal=torch.bincount(goal.reshape(-1), minlength=len(GOALS)).tolist(), problems=problems)
    if problems:
        raise CheckFailed('; '.join(problems))
    return counts


REQUESTS = ('Chop down', 'Net', 'Mine', 'Prospect', 'Attack', 'Take', 'Wield', 'Eat')  # options counted per goal


def refused_flags(fb_vec):
    """1 where a stored observation's feedback reports a refused option end (rl/feedback_contract.py REFUSED), else 0.
    The index comes from the contract's families and end reasons, never a literal."""
    from feedback_contract import REFUSED
    return (fb_vec[..., REFUSED] > 0.5).long()


def rollout_evidence(model, seen):
    """What the rollout just collected asked for and what happened, per goal, three things counted apart:
    requests (action families, and the ACT/HELD options REQUESTS names), game effects (items
    made per gathering family and eligible kills, from the step infos `made` and `kills`, which count within
    an episode) and episode ends with their certified outcomes (the audited `outcomes`). Also the refusals
    the feedback reports (fb_vec's refused flag of the stored observations, which describes the previous
    action) and the unsupported steps. `seen` carries each world's last counters across
    rollouts. Not a check: validate_rollout's run first."""
    import torch
    from act_contract import A_ACT0, A_HELD0, N_ACT, N_HELD, name_id
    from feedback_contract import FAMILIES as ACTION_FAMILIES
    from sim.agent import family
    b = model.rollout_buffer
    actions, obs, infos = b.actions[..., 0], b.observations, b.infos
    goal, G = infos['goal'].long(), len(GOALS)
    device = actions.device

    def by_goal(values, mask=None):  # [steps, worlds] numbers -> per-goal sums
        values = values if mask is None else torch.where(mask, values, torch.zeros_like(values))
        sums = torch.zeros(G, dtype=values.dtype, device=device).index_add_(0, goal.reshape(-1), values.reshape(-1))
        return dict(zip(GOALS, sums.tolist()))
    names = (*ACTION_FAMILIES, 'other')

    def family_or_other(a):  # the reserved slots (92-95) have no family
        try:
            return family(a)
        except ValueError:
            return 'other'
    table = torch.as_tensor([names.index(family_or_other(a)) for a in range(b.observations['mask'].shape[-1])],
                            device=device)
    kind = table[actions]
    families = torch.zeros(G * len(names), dtype=torch.int64, device=device).index_add_(
        0, (goal * len(names) + kind).reshape(-1), torch.ones_like(kind).reshape(-1)).view(G, len(names)).tolist()
    is_act, is_held = (actions >= A_ACT0) & (actions < A_ACT0 + N_ACT), (actions >= A_HELD0) & (actions < A_HELD0 + N_HELD)
    act_op = obs['act_op_ids'].gather(-1, (actions - A_ACT0).clamp(0, N_ACT - 1)[..., None])[..., 0]
    held_op = obs['held_op_ids'].gather(-1, (actions - A_HELD0).clamp(0, N_HELD - 1)[..., None])[..., 0]
    requested = {name: by_goal(((is_act & (act_op == name_id(name))) | (is_held & (held_op == name_id(name)))).long())
                 for name in REQUESTS}
    ended = infos['terminal'] != TERMINAL_RUNNING
    result = dict(transitions=by_goal(torch.ones_like(goal)), episodes_ended=by_goal(ended.long()),
                  horizon_ends=by_goal((infos['terminal'] == TERMINAL_HORIZON).long()),
                  deaths=by_goal((infos['terminal'] == TERMINAL_DEATH).long()),
                  certified_outcomes=by_goal(infos['outcomes'].long(), ended),
                  episode_return_sum=by_goal(infos['episode_return'], ended), reward_sum=by_goal(b.rewards),
                  requested_families={GOALS[g]: dict(zip(names, row)) for g, row in enumerate(families)},
                  requested_options=requested,
                  refused_feedback=by_goal(refused_flags(obs['fb_vec'])),
                  unsupported_steps=int(infos['unsupported'].sum()))
    if 'made' in infos and 'kills' in infos:
        from sim.task import FAMILIES as MADE
        made, kills, starts = infos['made'].long(), infos['kills'].long(), b.episode_starts > 0
        if not seen:
            seen.update(made=torch.zeros_like(made[0]), kills=torch.zeros_like(kills[0]))
        before_made = torch.where(starts[..., None], 0, torch.cat([seen['made'][None], made[:-1]]))
        before_kills = torch.where(starts, 0, torch.cat([seen['kills'][None], kills[:-1]]))
        gained, killed = made - before_made, kills - before_kills
        seen.update(made=made[-1], kills=kills[-1])
        result.update(items_made={f: by_goal(gained[..., k]) for k, f in enumerate(MADE)}, eligible_kills=by_goal(killed),
                      smallest_counter_change=int(min(gained.min(), killed.min())))
    if 'facts' in infos:  # the CPU workers' ended-episode facts (sim/vec.py EPISODE_FACTS)
        from sim.combat import COMBAT, GROUPS
        from sim.vec import COMBAT_FACTS, EPISODE_FACTS
        facts = infos['facts'].long()
        fact = lambda name: facts[..., EPISODE_FACTS.index(name)]
        result['successes'] = by_goal(fact('success'), ended)
        death = ended & (infos['terminal'] == TERMINAL_DEATH)
        groups = {}
        for i, group in enumerate(GROUPS, 1):
            mine = ended & (fact('group') == i)
            if mine.any():
                count = lambda values: int(values[mine].sum())
                groups[group] = dict(episodes=int(mine.sum()), successes=count(fact('success')), deaths=int((death & mine).sum()),
                                     deaths_after_target=int((death & mine & (infos['outcomes'] >= COMBAT['target'])).sum()),
                                     certified=count(infos['outcomes'].long()),
                                     **{k: count(fact(k)) for k in COMBAT_FACTS})
        if groups:
            result['combat_groups'] = groups
        from sim.search import SUPPLY
        for v, variant in enumerate(SUPPLY['variants'], 1):  # supply episodes (sim/search.py) by variant
            mine = ended & (fact('supply') == v)
            if mine.any():
                left = mine & (fact('elsewhere_gap') >= 0)
                result.setdefault('supply', {})[variant] = dict(
                    episodes=int(mine.sum()), successes=int(fact('success')[mine].sum()), deaths=int((death & mine).sum()),
                    certified=int(infos['outcomes'][mine].long().sum()), exhausted=int((mine & (fact('exhausted_tick') >= 0)).sum()),
                    went_elsewhere=int(left.sum()), elsewhere_gap_sum=int(fact('elsewhere_gap')[left].sum()))
        searched = ended & (fact('search') == 1)
        if searched.any():  # search episodes (sim/search.py): discovery and where the player went
            found = searched & (fact('discovered') == 1)
            result['search'] = dict(episodes=int(searched.sum()), successes=int(fact('success')[searched].sum()),
                                    discovered=int(found.sum()), discovery_ticks_sum=int(fact('discovery_ticks')[found].sum()),
                                    cells_sum=int(fact('cells')[searched].sum()))
    return result


# Per-update evidence (the sample size is a Proposal): every update saves the same world slots' first
# `decisions` decisions with everything the update reads, `worlds` slots spread over the batch and split equally between
# the declared goals (16 per goal of 64 in a shared run), and replays it with the collecting policy before the update. A
# file is named by the decisions before its rollout. Offline, segments-N.npz replays under the checkpoint that holds the
# policy that collected it: initial.zip for 0 and, in a --recovery run only, checkpoints/ppo_N_steps.zip where N is a
# checkpoint boundary (without --recovery SB3 saves ppo_N_steps.zip during the collection, one update earlier); the
# other samples' collecting weights are not kept. ATOL is a same-device tolerance: replay on the run's device.
SEGMENTS = dict(worlds=64, decisions=64)
SEGMENT_FIELDS = (('actions', 'actions'), ('rewards', 'rewards'), ('episode_starts', 'episode_starts'), ('values', '_values'),
                  ('log_probs', 'log_probs'), ('advantages', 'advantages'), ('returns', '_returns'))
SEGMENT_INFOS = ('goal', 'terminal', 'decision_ticks')
SEGMENT_STATES = ('hidden_states_pi', 'cell_states_pi', 'hidden_states_vf', 'cell_states_vf')


def segment_slots(worlds, goals, count):
    """The fixed sample's world slots: for each declared goal, count / len(goals) of the slots that ask for it, evenly
    spaced over the batch (so over the CPU workers' slices too)."""
    goals = goal_set(goals)
    per, left = divmod(count, len(goals))
    if left or per < 1:
        raise ValueError(f'{count} segment worlds do not split equally between the goals {list(goals)}')
    slots = []
    for goal in goals:
        own = [w for w in range(worlds) if slot_goal(goals, w) == goal]
        if per > len(own):
            raise ValueError(f'{per} segment worlds of goal {goal}, but only {len(own)} slots ask for it')
        slots += [own[k * len(own) // per] for k in range(per)]
    return sorted(slots)


def policy_digest(policy):
    """sha256 of the policy's parameters and buffers by name: the weights a saved sample's outputs came from."""
    h = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_segments(model, path, slots, decisions, meta):
    """The rollout just collected (returns and advantages computed, before its update), for each world slot of `slots`
    its first `decisions` decisions: observations, actions, old log-probabilities, values, advantages, returns, rewards,
    episode starts, goal, terminal flag and ticks, and the recurrent states at the segment's first decision. meta (JSON)
    gains the behavior policy's digest. Written atomically as a compressed npz; returns its size in bytes and that digest,
    which the in-run replay of this file takes instead of hashing the same policy again."""
    import torch
    b = model.rollout_buffer
    if decisions > b.buffer_size:
        raise ValueError(f'a {decisions}-decision segment is longer than the {b.buffer_size}-decision rollout')
    keep = torch.as_tensor(slots, device=b.device)
    take = lambda v: v[:decisions].index_select(1, keep).cpu().numpy()
    arrays = {f'obs__{k}': take(v) for k, v in b.observations.items()}
    arrays.update({name: take(getattr(b, store)) for name, store in SEGMENT_FIELDS})
    arrays.update({f'info__{k}': take(b.infos[k]) for k in SEGMENT_INFOS})
    arrays.update({name: getattr(b, name)[0].index_select(0, keep).cpu().numpy() for name in SEGMENT_STATES})
    digest = policy_digest(model.policy)
    arrays['meta'] = np.asarray(json.dumps(dict(meta, slots=list(slots), decisions=decisions, policy_sha256=digest)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size, digest


def replay_segments(path, policy, rows=None, digest=None):
    """Recompute a saved sample's values and log-probabilities with `policy`, which must be the sample's behavior policy
    (its digest, the file's meta policy_sha256), in two shapes, and return each one's largest gaps from the stored outputs
    and whether the stored returns equal advantages plus values bitwise. digest: policy_digest(policy) when the caller
    has it already, as the in-run check has from save_segments for the same unchanged policy; the file's stored digest
    is still read back and compared with it. Without it (the replay-segments command) the policy is hashed here.
    one-step (log_prob_gap, value_gap; --validate's ATOL bounds them): the collector's own computation, the policy's
    forward on one step of every world at a time, its LSTM states carried from the stored segment-start states (reset at
    episode starts), the log-probability of the stored action. The batch has the collection's world count (`rows`,
    default the sample's `worlds`), each segment in its own slot and the other rows copies of segments: GPU kernels are
    chosen by shape. Where the name table is projected whole (baseline.policy NameEmbedding.FULL_TABLE, CUDA's
    default) a row's arithmetic does not depend on the others and the replay repeats the collection exactly; the CPU
    projects only the batch's names, so there the copies can move the last bits.
    sequence (sequence_log_prob_gap, sequence_value_gap; the in-run check bounds them by SEQUENCE_ATOL in runs after
    phase 1, the replay-segments command reports them): the learner's own minibatch (sim/learn.py
    DeviceRolloutBuffer._get_samples: a sequence at each segment's first decision and at each episode start, the stored
    recurrent states, zero padding, the valid-row mask), whose GPU arithmetic differs from the collection's by up to
    about 1e-3 (phase 1)."""
    import torch
    from sim.learn import DeviceRolloutBuffer
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays.pop('meta')))
    if (policy_digest(policy) if digest is None else digest) != meta['policy_sha256']:
        raise CheckFailed(f'{path} was collected by another policy (sha256 {meta["policy_sha256"][:12]})')
    T, S = arrays['rewards'].shape
    lstm, device = policy.lstm_actor, policy.device
    b = DeviceRolloutBuffer(T, policy.observation_space, policy.action_space, (T, lstm.num_layers, S, lstm.hidden_size),
                            device, n_envs=S)
    for k in b.observations:
        b.observations[k].copy_(torch.as_tensor(arrays[f'obs__{k}']))
    for name, store in SEGMENT_FIELDS:
        getattr(b, store).copy_(torch.as_tensor(arrays[name]))
    for name in SEGMENT_STATES:  # only a sequence's first state is read; an episode start's is reset by the policy
        getattr(b, name).zero_()
        getattr(b, name)[0].copy_(torch.as_tensor(arrays[name]))
    b.pos, b.full, b.generator_ready = T, True, True
    flat = torch.arange(T * S, device=device)  # env-major: segment s holds rows s*T .. s*T + T - 1
    order = b._rows(flat)
    first = ((b.episode_starts.reshape(-1)[order] != 0) | (flat % T == 0)).nonzero().squeeze(1)
    longest = int(torch.diff(first, append=first.new_full((1,), len(order))).max())
    samples = b._get_samples(order, first, longest)
    with torch.no_grad():
        values, log_prob, _ = policy.evaluate_actions(samples.observations, samples.actions.long().flatten(),
                                                      samples.lstm_states, samples.episode_starts)
    valid = samples.mask > 1e-8
    gap = lambda a, b: float((a - b)[valid].abs().max())
    sequence = dict(sequence_log_prob_gap=gap(log_prob, samples.old_log_prob),
                    sequence_value_gap=gap(values.flatten(), samples.old_values))
    # one-step: row r of the batch reads segment index[r]; segment s sits at its collection slot when the batch has the
    # collection's size
    R = rows or meta['worlds']
    if R < S:
        raise ValueError(f'{R} rows cannot hold the sample\'s {S} segments')
    place = torch.as_tensor(meta['slots'] if R == meta['worlds'] else range(S), device=device)
    index = torch.arange(R, device=device) % S
    index[place] = torch.arange(S, device=device)
    state = lambda store: store[0].index_select(0, index).transpose(0, 1).contiguous()  # [layers, R, hidden]
    pi = (state(b.hidden_states_pi), state(b.cell_states_pi))
    vf = (state(b.hidden_states_vf), state(b.cell_states_vf))
    log_prob_gap = value_gap = torch.zeros((), device=device)  # torch.maximum keeps a NaN
    with torch.no_grad():
        for t in range(T):
            obs = {k: v[t].to(b._dtypes[k]).index_select(0, index) for k, v in b.observations.items()}
            starts = b.episode_starts[t].index_select(0, index)
            with policy._masked(obs):  # RecurrentActorCriticPolicy.forward, the log-probability of the stored action
                pi_features, vf_features = policy.extract_features(obs)
                latent_pi, pi = policy._process_sequence(pi_features, pi, starts, policy.lstm_actor)
                latent_vf, vf = policy._process_sequence(vf_features, vf, starts, policy.lstm_critic)
                latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
                latent_vf = policy.mlp_extractor.forward_critic(latent_vf)
                step_values = policy.value_net(latent_vf).flatten()
                step_log_prob = policy._get_action_dist_from_latent(latent_pi).log_prob(
                    b.actions[t, :, 0].long().index_select(0, index))
            log_prob_gap = torch.maximum(log_prob_gap, (step_log_prob[place] - b.log_probs[t]).abs().max())
            value_gap = torch.maximum(value_gap, (step_values[place] - b._values[t]).abs().max())
    return dict(path=str(path), segments=S, decisions=T, rows=R, num_timesteps=meta['num_timesteps'],
                log_prob_gap=float(log_prob_gap), value_gap=float(value_gap), **sequence,
                returns_exact=bool(np.array_equal(arrays['returns'], arrays['advantages'] + arrays['values'])))


def host_memory():
    """This host's available and total memory in MB (Linux /proc/meminfo; None elsewhere)."""
    try:
        info = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        return dict(available_mb=round(info['MemAvailable'] / 1024, 1), total_mb=round(info['MemTotal'] / 1024, 1))
    except (OSError, KeyError, ValueError, IndexError):
        return None


SPEED_SWITCHES = dict(fused='FUSED_SEQUENCES', valid_rows='VALID_ROWS')  # baseline.policy MaskedRecurrentPolicy
# replay variants: (fused, valid_rows, how, merged): 'again' repeats stock (on a GPU, its own nondeterminism), 'reordered'
# gives each minibatch its sequences in reverse order (equal in exact arithmetic, another floating-point order); 'fast' is
# sim/learn.py's fast_train (bitwise equal on the CPU), 'graphed' fast_train with GRAPHED_UPDATE (CUDA). merged:
# baseline.policy NameEmbedding.MERGED for the variant (False: one lookup per input and use, the arithmetic before it)
REPLAY_VARIANTS = dict(stock=(False, False, None, False), stock_again=(False, False, 'again', False),
                       reordered=(False, False, 'reorder', False), fused=(True, False, None, False),
                       valid_rows=(False, True, None, False), both=(True, True, None, False),
                       # both_again repeats the current path (on a GPU, its own nondeterminism: the scale to compare with)
                       both_again=(True, True, 'again', False), fast=(True, True, 'fast', False),
                       graphed=(True, True, 'graphed', False),
                       # one lookup per embedding and call (NameEmbedding.MERGED): the same rows, gradients summed in another order
                       fast_merged=(True, True, 'fast', True), graphed_merged=(True, True, 'graphed', True))


def ppo_state(model):
    """What RecurrentPPO.train() reads and changes: the policy, Adam, the rollout buffer, the generators' states, the
    progress and update counters (copies)."""
    import copy
    import torch
    from sim.learn import BUFFER_STORAGE
    b = model.rollout_buffer
    return dict(policy={k: v.detach().clone() for k, v in model.policy.state_dict().items()},
                optimizer=copy.deepcopy(model.policy.optimizer.state_dict()),
                buffer={k: getattr(b, k).clone() for k in BUFFER_STORAGE}, observations={k: v.clone() for k, v in b.observations.items()},
                flags=(b.pos, b.full, b.generator_ready), np_state=np.random.get_state(), torch_state=torch.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                progress=model._current_progress_remaining, n_updates=model._n_updates)


def put_ppo_state(model, state):
    import copy
    import torch
    model.policy.load_state_dict(state['policy'], strict=True)
    model.policy.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))  # a copy: Adam changes its state in place
    b = model.rollout_buffer
    for k, v in state['buffer'].items():
        getattr(b, k).copy_(v)
    for k, v in state['observations'].items():
        b.observations[k].copy_(v)
    b.pos, b.full, b.generator_ready = state['flags']
    np.random.set_state(state['np_state'])
    torch.set_rng_state(state['torch_state'])
    if state['cuda_state'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_state'])
    model._current_progress_remaining, model._n_updates = state['progress'], state['n_updates']


def reversed_sequences(sample):
    """A minibatch with its sequences in reverse order: the same loss in exact arithmetic."""
    import torch
    n_seq = sample.lstm_states.pi[0].shape[1]
    order = torch.arange(n_seq - 1, -1, -1, device=sample.mask.device)
    rows = lambda x: x.reshape(n_seq, -1, *x.shape[1:])[order].reshape(x.shape)
    states = lambda pair: tuple(t[:, order] for t in pair)
    return type(sample)(observations={k: rows(v) for k, v in sample.observations.items()}, actions=rows(sample.actions),
                        old_values=rows(sample.old_values), old_log_prob=rows(sample.old_log_prob), advantages=rows(sample.advantages),
                        returns=rows(sample.returns), lstm_states=type(sample.lstm_states)(states(sample.lstm_states.pi), states(sample.lstm_states.vf)),
                        episode_starts=rows(sample.episode_starts), mask=rows(sample.mask))


def replay_ppo(model, before, variants, device):
    """Replay one PPO phase from `before` under each variant (REPLAY_VARIANTS; the first is the reference): synchronized
    seconds, and the largest parameter difference from the reference after optimizer steps 1, 2, 4, ... and the last, with
    how many parameter elements then differ by more than 1e-5 (Adam turns rounding in a near-zero gradient into a step of
    up to the learning rate, so the largest difference alone overstates a summation-order change)."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import NameEmbedding
    policy_class, buffer = type(model.policy), model.rollout_buffer
    out, reference = {}, None
    switches = model.FAST_UPDATE, model.GRAPHED_UPDATE, NameEmbedding.MERGED
    for name in variants:
        fused, valid_rows, how, merged = REPLAY_VARIANTS[name]
        put_ppo_state(model, before)
        setattr(policy_class, SPEED_SWITCHES['fused'], fused)
        setattr(policy_class, SPEED_SWITCHES['valid_rows'], valid_rows)
        NameEmbedding.MERGED = merged
        if how == 'reorder':
            get = type(buffer).get
            buffer.get = lambda batch_size=None: (reversed_sequences(x) for x in get(buffer, batch_size))
        snaps, count = {}, [0]

        def recorded(optimizer, args, kwargs):
            count[0] += 1
            if count[0] & (count[0] - 1) == 0:  # 1, 2, 4, ...
                snaps[count[0]] = {k: v.detach().clone() for k, v in model.policy.state_dict().items() if v.is_floating_point()}
        handle = model.policy.optimizer.register_step_post_hook(recorded)
        synchronize(device)
        t0 = time.perf_counter()
        try:
            if how in ('fast', 'graphed'):
                type(model).GRAPHED_UPDATE = how == 'graphed'
                model.fast_train()
            else:
                RecurrentPPO.train(model)
            synchronize(device)
        finally:
            handle.remove()
            buffer.__dict__.pop('get', None)
            type(model).FAST_UPDATE, type(model).GRAPHED_UPDATE, NameEmbedding.MERGED = switches
        seconds = time.perf_counter() - t0
        snaps[count[0]] = {k: v.detach().clone() for k, v in model.policy.state_dict().items() if v.is_floating_point()}
        if reference is None:
            reference = snaps
        out[name] = dict(seconds=seconds, optimizer_steps=count[0], param_diff_by_step={
            m: max(float((snaps[m][k].double() - reference[m][k].double()).abs().max()) for k in snaps[m]) for m in sorted(snaps)},
            elements_beyond_1e5_by_step={m: int(sum(int(((snaps[m][k].double() - reference[m][k].double()).abs() > 1e-5).sum())
                                                    for k in snaps[m])) for m in sorted(snaps)},
            elements=int(sum(v.numel() for v in reference[max(reference)].values())))
    return dict(variants=out)


def device_seconds(events):
    """The device's own time in torch.profiler's averaged events, as the operator table's 'Self CUDA time total'
    counts it: kernels, copies and sets on the device, not the host operators that launched them (an aten
    operator's self device time repeats its kernels'; summing both counted it twice)."""
    from torch.autograd import DeviceType
    return sum(e.self_device_time_total for e in events
               if (e.device_type == DeviceType.CPU and e.is_legacy)
               or (e.device_type != DeviceType.CPU and not e.is_user_annotation)) / 1e6


def write_ppo_profile(prof, base, wall):
    """torch.profiler's operator table (by self time, device first when there is one) and its totals: the PPO
    phase's wall seconds, the device's own and the host's self time, kernel launches and synchronizations."""
    events = prof.key_averages()
    cuda = any(getattr(e, 'self_device_time_total', 0) for e in events)
    key = 'self_device_time_total' if cuda else 'self_cpu_time_total'
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix('.txt').write_text(events.table(sort_by=key, row_limit=60))
    count = lambda *names: sum(e.count for e in events if e.key in names)
    totals = dict(wall_seconds=wall, device_kernel_seconds=device_seconds(events),
                  host_self_seconds=sum(e.self_cpu_time_total for e in events) / 1e6,
                  kernel_launches=count('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel'),
                  synchronizations=count('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize'),
                  memcpy_calls=count('cudaMemcpyAsync', 'cudaMemcpy'),
                  top=[dict(op=e.key, count=e.count, self_seconds=getattr(e, key) / 1e6) for e in sorted(events, key=lambda e: -getattr(e, key))[:25]])
    base.with_suffix('.json').write_text(json.dumps(totals, indent=1) + '\n')
    return {k: v for k, v in totals.items() if k != 'top'}


def check_saved_versions(path):
    """baseline/policy.py check_names on a checkpoint as it was saved, before any of its tensors is loaded: the name
    table and the observation, action, mask and feedback versions it records. A checkpoint of the same tensor shapes
    saved under another contract, or with none recorded, is refused, never taken under this run's versions."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    check_names(RecurrentPPO.load(path, device='cpu').policy)


def load_init(model, path, device):
    """Replace a fresh model's policy and optimizer state with a checkpoint's (SB3 set_parameters, exact match);
    return its sha256, the optimizer's step count and whether every policy tensor now equals the checkpoint's."""
    import io
    import zipfile
    import torch
    check_saved_versions(path)
    model.set_parameters(str(path), exact_match=True, device=device)
    with zipfile.ZipFile(path) as archive:
        saved = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    mine = model.policy.state_dict()
    equal = set(saved) == set(mine) and all(torch.equal(saved[k], mine[k].cpu()) for k in saved)
    steps = [int(v['step']) for v in model.policy.optimizer.state_dict()['state'].values() if 'step' in v]
    if not equal:
        raise CheckFailed(f'the initialization {path} did not load exactly')
    return dict(path=str(path), sha256=sha256(path), policy_equal=equal, tensors=len(saved),
                optimizer_steps=max(steps) if steps else 0)


def load_weights(model, path, device):
    """Load only a checkpoint's policy tensors (actor, critic and the frozen name embeddings) into a fresh model built
    with this run's settings: its optimizer, schedules, counters, buffer and recurrent state stay new, so nothing of the
    checkpoint's training settings (n_steps, batching, learning rate) reaches the run. Exact names and shapes, and every
    tensor equal afterwards; returns its sha256 and the saved settings it did not import."""
    import io
    import zipfile
    import torch
    check_saved_versions(path)
    with zipfile.ZipFile(path) as archive:
        saved = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
        data = json.loads(archive.read('data').decode())
    model.policy.load_state_dict(saved, strict=True)
    mine = model.policy.state_dict()
    if set(saved) != set(mine) or not all(torch.equal(saved[k], mine[k].cpu()) for k in saved):
        raise CheckFailed(f'the weights of {path} did not load exactly')
    ignored = {k: data.get(k) for k in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                                        'max_grad_norm', 'num_timesteps', '_n_updates') if k in data}
    return dict(path=str(path), sha256=sha256(path), tensors=len(saved), policy_equal=True, optimizer='fresh',
                saved_settings_not_imported=ignored)


def check_source_outputs(model, path, device):
    """Before the first update of a weights run: the source checkpoint, loaded as itself, gives the same action
    probabilities and values as the run's model on every stored step of the first rollout (its observations, LSTM
    states and episode starts), so the run starts from the actual saved policy."""
    import torch
    from sb3_contrib import RecurrentPPO
    source = RecurrentPPO.load(path, device=device)
    b, worst = model.rollout_buffer, 0.0
    with torch.no_grad():
        for step in range(b.buffer_size):
            o = {k: v[step].to(b._dtypes[k]) for k, v in b.observations.items()}
            pi = (b.hidden_states_pi[step].transpose(0, 1).contiguous(), b.cell_states_pi[step].transpose(0, 1).contiguous())
            vf = (b.hidden_states_vf[step].transpose(0, 1).contiguous(), b.cell_states_vf[step].transpose(0, 1).contiguous())
            starts = b.episode_starts[step]
            for p in (model.policy, source.policy):
                p.set_training_mode(False)
            outputs = [(p.get_distribution(o, pi, starts)[0].distribution.probs, p.predict_values(o, vf, starts))
                       for p in (model.policy, source.policy)]
            worst = max(worst, float((outputs[0][0] - outputs[1][0]).abs().max()), float((outputs[0][1] - outputs[1][1]).abs().max()))
    result = dict(path=str(path), steps=b.buffer_size, max_difference=worst)
    if worst > ATOL:
        raise CheckFailed(f'the run\'s first rollout differs from its source checkpoint: {result}')
    return result


RECOVERY_STATE = 'state.pt'


TREATMENTS = ('distill',)


def treatment_state(model):
    """The treatment state (sim/distill.py), or None for a run without it. A recovery state that holds a removed
    treatment (RND, CBP or PCGrad) is refused."""
    distill = getattr(model, 'distill', None)
    if distill is None:
        return None
    return dict(distill=distill.state_dict())


def tensor_leaves(value, path=''):
    """(path, tensor) of every tensor in a nested state of dicts, lists and tuples."""
    import torch
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensor_leaves(v, f'{path}/{k}')
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            yield from tensor_leaves(v, f'{path}/{k}')


# Treatment state that lives on the CPU whatever the run's device: the distillation sampler's state.
HOST_STATE = ('sampler',)


def put_treatment_state(model, state):
    """treatment_state's content (perhaps loaded on the CPU) into a model attached with the same treatments; refuses a
    mismatch. Then checks every restored tensor: equal to the saved one, on the CPU if HOST_STATE names it and else on
    the run's device. Returns the check's record."""
    import torch
    have = {k: getattr(model, k, None) is not None for k in TREATMENTS}
    saved = {k: bool(state and state.get(k) is not None) for k in TREATMENTS}
    if have != saved:
        raise CheckFailed(f'the recovery state holds treatments {saved}, the run has {have}')
    removed = sorted(k for k, v in (state or {}).items() if k not in TREATMENTS and v is not None)
    if removed:
        raise CheckFailed(f'the recovery state holds removed treatments {removed}')
    if not any(saved.values()):
        return None
    model.distill.load_state_dict(state['distill'])
    device = next(model.policy.parameters()).device
    restored, wanted = dict(tensor_leaves(treatment_state(model))), dict(tensor_leaves(state))
    problems = sorted(set(restored) ^ set(wanted))
    devices = Counter()
    for path, tensor in restored.items():
        if path not in wanted:
            continue
        host = path.rsplit('/', 1)[-1] in HOST_STATE
        if tensor.device.type != ('cpu' if host else device.type):
            problems.append(f'{path} on {tensor.device}')
        before = torch.as_tensor(wanted[path]).cpu()
        if tensor.dtype != before.dtype or not torch.equal(tensor.detach().cpu(), before):
            problems.append(f'{path} differs from the saved value')
        devices[tensor.device.type] += 1
    if problems:
        raise CheckFailed(f'the treatment state did not restore exactly: {problems[:8]}')
    return dict(tensors=len(restored), devices=dict(devices))


def save_recovery(model, directory, record):
    """After a complete update: the policy, Adam, the counters and the random generators' states (no environment or
    recurrent state: a resume starts new episodes), written atomically, with record (a JSON summary) beside it."""
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(policy=model.policy.state_dict(), optimizer=model.policy.optimizer.state_dict(),
                 treatments=treatment_state(model),
                 num_timesteps=model.num_timesteps, n_updates=model._n_updates, np_state=np.random.get_state(),
                 torch_state=torch.get_rng_state(),
                 cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tmp = directory / (RECOVERY_STATE + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, directory / RECOVERY_STATE)
    record = dict(record, num_timesteps=model.num_timesteps, n_updates=model._n_updates, sha256=sha256(directory / RECOVERY_STATE))
    tmp = directory / 'state.json.tmp'
    tmp.write_text(json.dumps(record, indent=1) + '\n')
    os.replace(tmp, directory / 'state.json')
    return record


def load_recovery(model, directory):
    """A save_recovery state into a model built with the same settings: policy, Adam, counters and generators; the
    sha256 must match its record."""
    import torch
    directory = Path(directory)
    record = json.loads((directory / 'state.json').read_text())
    if sha256(directory / RECOVERY_STATE) != record['sha256']:
        raise CheckFailed(f'{directory} state differs from its record')
    # on the host: the generators' states must stay CPU byte tensors; the policy and Adam move theirs to the model's device
    state = torch.load(directory / RECOVERY_STATE, map_location='cpu', weights_only=False)
    model.policy.load_state_dict(state['policy'], strict=True)
    model.policy.optimizer.load_state_dict(state['optimizer'])
    for group in model.policy.optimizer.param_groups:  # Adam's step counters stay on the host, as set_parameters
        for p in group['params']:
            st = model.policy.optimizer.state.get(p, {})
            if 'step' in st:
                st['step'] = st['step'].cpu()
    treatments = put_treatment_state(model, state.get('treatments'))
    model.num_timesteps, model._n_updates = state['num_timesteps'], state['n_updates']
    np.random.set_state(state['np_state'])
    torch.set_rng_state(state['torch_state'])
    if state['cuda_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_state'])
    return dict(record, path=str(directory), treatments_restored=treatments)


def check_checkpoint(model, path, device):
    """The saved zip loads as a standard RecurrentPPO with the obs-v2 tag and gives the same action
    probabilities and values as the live policy on the last rollout's first step (exact)."""
    import torch
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    loaded = RecurrentPPO.load(path, device=device)
    check_names(loaded.policy)
    b = model.rollout_buffer
    o = {k: v[0].to(b._dtypes[k]) for k, v in b.observations.items()}
    pi = (b.hidden_states_pi[0].transpose(0, 1).contiguous(), b.cell_states_pi[0].transpose(0, 1).contiguous())
    vf = (b.hidden_states_vf[0].transpose(0, 1).contiguous(), b.cell_states_vf[0].transpose(0, 1).contiguous())
    starts = b.episode_starts[0]
    with torch.no_grad():
        outputs = [(p.get_distribution(o, pi, starts)[0].distribution.probs, p.predict_values(o, vf, starts))
                   for p in (model.policy, loaded.policy)]
    (p1, v1), (p2, v2) = outputs
    result = dict(path=str(path), class_=type(loaded).__name__, observation=loaded.policy.observation,
                  probs_equal=bool(torch.equal(p1, p2)), values_equal=bool(torch.equal(v1, v2)))
    if not (result['probs_equal'] and result['values_equal']):
        raise CheckFailed(f'the reloaded checkpoint differs: {result}')
    return result


LOSS_KEYS = ('train/loss', 'train/value_loss', 'train/policy_gradient_loss', 'train/entropy_loss', 'train/approx_kl')


def check_losses(logged, updates, gradient_steps, expected_steps):
    """The train/ values RecurrentPPO.train() recorded after each update (read from the logger at once: SB3
    writes them to progress.csv only at the next iteration's dump, so the last update never appears there):
    one record per update, every loss finite, every gradient step taken."""
    result = dict(updates_logged=len(logged), gradient_steps=gradient_steps, expected_gradient_steps=expected_steps,
                  **{k.split('/')[1]: [r.get(k) for r in logged] for k in LOSS_KEYS})
    if len(logged) != updates or gradient_steps != expected_steps or \
            not all(k in r and np.isfinite(r[k]) for r in logged for k in LOSS_KEYS):
        raise CheckFailed(f'incomplete or non-finite updates: {result}')
    return result


def hold(path, poll=0.05):
    """Wait until `path` exists (train's start_when; None: no wait). It reads no generator and touches no tensor."""
    while path is not None and not Path(path).exists():
        time.sleep(poll)


def train(output, *, decisions, seed, worlds, n_steps, batch_size, n_epochs, device='cpu', env='fake',
          checkpoint_every=None, threads=None, verbose=True, pool=None, goals=GOALS, validate=False, tables=None,
          count_syncs=False, environment=None, events=None, callbacks=(), init=None, warmup=1,
          speed=None, ppo_profile=None, replay=None, step_timing=True, workers=None,
          task='goal-activity-v2', weights=None, recovery=None, resume=None, env_seed=None, search_pool=None,
          telemetry=None, distill_memory=None, distill=False, distill_strata=None, segments=None, engine='cpu', pack=None,
          start_when=None):
    """One training run; returns summary.json's content. task: the four-goal task version (sim/combat.py task_spec;
    v3 needs env cpu), recorded in the summary and the recovery state: a resume refuses another. goals: the declared
    goal set (sim/spec.py goal_set), recorded in the summary and the recovery state; a resume refuses another.
    segments: the per-update evidence (SEGMENTS' keys, worlds and decisions): after its rollout's checks every update
    saves its fixed sample to segments/segments-N.npz (save_segments) and replays it with the collecting policy
    (replay_segments); a one-step gap above ATOL or a sequence gap above SEQUENCE_ATOL stops the run before the
    update. weights: a checkpoint
    whose policy tensors only start the run (load_weights: this run's settings, a fresh optimizer); before the first update its first rollout is checked against that checkpoint
    (check_source_outputs). recovery: a directory where every complete update leaves the state a resume needs
    (save_recovery) and a row in updates.jsonl; checkpoints are then saved at complete-update boundaries (every
    checkpoint_every decisions). resume: a recovery directory to continue from: its policy, Adam, counters and
    generators, then the decisions left to `decisions` in new episodes from env_seed (default seed) with new recurrent
    state; the collection after its last complete update is discarded. checkpoint_every: decisions between saved checkpoints
    (default: every 8 updates). validate: the --validate checks (checks.json); a failure stops the run, and
    their time is reported apart (timing.validation_seconds), as is that of each rollout's rollout_evidence
    (summary 'updates', one record per update). count_syncs (CUDA): the synchronizations torch reports during
    the first rollout, which is the warm-up, so the measured rollouts run without the counter.
    environment: a DeviceEnv the caller built (else make_env builds one). events: events(stage, phase, **fields)
    at each stage's start and end, synchronized (staged): policy_build, initial_save, and per update `index`
    collect, checkpoint_save (when one falls in it), validate, evidence and update, then final_save,
    checkpoint_check and loss_check. callbacks: further SB3 callbacks for learn().
    init: a RecurrentPPO zip whose policy and optimizer state replace the fresh model's (SB3 set_parameters,
    exact match) before anything runs; summary 'init' records its sha256 and the check that every parameter
    equals it. warmup: the first updates the steady rates leave out (1 by default). search_pool: goal-activity-v4's
    search layouts or v5's supply layouts (sim/search.py). distill_memory: a sim/distill.py memory whose drift probe runs
    after every update, and with `distill` its self-distillation phase (limited to distill_strata when given); telemetry
    (default: on for v4, v5 and v11) records sim/learn.py Tracker's activation statistics and each update's parameter change
    in every update row, read-only. Per update, summary 'updates' also holds
    collect_parts (the collector's synchronized parts, sim/learn.py step_timing; None with step_timing=False, which
    leaves only the rollout's own boundary synchronizations) and log_seconds (the logger's dump).
    The learner-speed comparison, at fixed minibatch and optimizer settings: speed: (update index, names):
    baseline.policy MaskedRecurrentPolicy's switches (SPEED_SWITCHES) are off before that update and the named ones
    on from its PPO phase onward, so one process times both paths; without it the class defaults (both on) apply.
    Each update records the paths its PPO phase used (summary 'updates' speed_paths). ppo_profile: (update indices,
    directory): those updates' PPO phases run under torch.profiler (CPU and CUDA); its operator table and totals are
    written there. replay: (update index, variants):
    after that update, its PPO phase is replayed from the same inputs under each of REPLAY_VARIANTS (the first, stock,
    is the reference), with the largest parameter difference from it after optimizer steps 1, 2, 4, ... and the
    synchronized seconds, then the real state is put back. Profiled and replayed updates should be warm-up updates:
    their probes' cost falls in their own iteration. start_when: a file whose existence starts the run (hold): the imports,
    the output folder, the inputs' sha256 and the host facts come first, then nothing more until the file exists; the run's
    clock (wall_seconds, the timing) starts after it, so a launcher can start one run's process while the run before
    trains and write the file once that run has ended and passed its checks."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from baseline.policy import NameEmbedding
    from sim.learn import build
    rollout, goals = worlds * n_steps, goal_set(goals)
    if decisions <= 0 or decisions % rollout:
        raise ValueError(f'decisions must be a positive multiple of worlds x n_steps = {rollout}')
    if (weights or resume) and init:
        raise ValueError('init restores a checkpoint\'s optimizer state: use weights or resume')
    if weights and resume:
        raise ValueError('a resume restores its own weights')
    if worlds % len(goals):
        raise ValueError(f'worlds must be a multiple of {len(goals)}: goals are assigned per world slot')
    if validate and env != 'cpu':
        raise ValueError('--validate checks the simulator\'s task clock: it needs --env cpu')
    slots = None if segments is None else segment_slots(worlds, goals, segments['worlds'])
    if segments is not None and not 0 < segments['decisions'] <= n_steps:
        raise ValueError(f'a {segments["decisions"]}-decision segment does not fit a {n_steps}-decision rollout')
    if threads:
        torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    changes = dict(n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs)
    telemetry = task in ('goal-activity-v4', 'goal-activity-v5', 'goal-activity-v11') if telemetry is None else telemetry
    summary = dict(status='started', decisions=decisions, seed=seed, env_seed=env_seed or seed, task=task, worlds=worlds,
                   env=env, engine=engine, pack=None if pack is None else dict(path=str(pack), sha256=sha256(pack)),
                   goals=list(goals), retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT,
                   pool=None if pool is None else dict(path=str(pool), sha256=sha256(pool)),
                   search_pool=None if search_pool is None else dict(path=str(search_pool), sha256=sha256(search_pool)),
                   changes_from_learner={k: dict(learner=config.LEARNER[k], here=v) for k, v in changes.items()
                                         if config.LEARNER[k] != v},
                   rollout_transitions=rollout, host=versions(device),
                   segments=None if segments is None else dict(segments, slots=slots))
    hold(start_when)
    times, marks = dict(), []
    started = time.perf_counter()
    epoch = time.time() - started  # clock time of a perf_counter mark, for the phase epochs of each update

    def tell(stage, phase, **fields):
        if events is not None:
            events(stage, phase, **fields)
    try:
        if environment is None:
            t0 = time.perf_counter()
            environment = make_env(env, worlds, env_seed or seed, device, pool, goals, tables, workers, task, search_pool,
                                   engine, pack)
            times['env_compile_seconds'] = time.perf_counter() - t0
        if tuple(environment.env.goals) != goals:
            raise ValueError(f'the environment asks for goals {list(environment.env.goals)}, the run declares {list(goals)}')
        t0 = time.perf_counter()
        with staged(events, 'policy_build', device):
            model = build(environment, seed=seed, device=device, **changes)
        times['model_build_seconds'] = time.perf_counter() - t0
        switches = {a: getattr(type(model.policy), a) for a in SPEED_SWITCHES.values()}  # the class's, put back at the end
        if speed is not None:
            for a in SPEED_SWITCHES.values():
                setattr(type(model.policy), a, False)
        if init is not None:
            with staged(events, 'policy_init', device):
                summary['init'] = load_init(model, init, device)
        if weights is not None:
            with staged(events, 'policy_weights', device):
                summary['weights'] = load_weights(model, weights, device)
        if telemetry or distill_memory is not None:
            from sim.distill import DISTILL
            from sim.learn import attach
            attach(model, seed=seed, telemetry=telemetry, memory=distill_memory, distill=distill,
                   distill_strata=distill_strata)
            arm = ('T' if distill_strata is not None else 'S') if distill else 'A'
            summary['treatments'] = dict(arm=arm, telemetry=telemetry,
                                         distill=dict({k: v for k, v in DISTILL.items() if k != 'strata'},
                                                      **({'trained_strata': model.distill.strata} if distill_strata is not None
                                                         else {})) if distill else None,
                                         memory=None if distill_memory is None else dict(path=str(distill_memory),
                                                                                         sha256=sha256(distill_memory)))
        if resume is not None:
            with staged(events, 'resume', device):
                summary['resume'] = load_recovery(model, resume)
            saved = summary['resume'].get('goals', list(GOALS))  # states written before goal sets are four-goal runs
            if saved != list(goals):
                raise CheckFailed(f'the recovery state trained goals {saved}, this run declares {list(goals)}')
            saved = summary['resume'].get('task', task)  # states written before goal-activity-v11 record no task
            if saved != task:
                raise CheckFailed(f'the recovery state trained task {saved}, this run declares {task}')
        start_timesteps = model.num_timesteps
        # the weights this process starts from, by their tensors (a zip's sha256 changes with every save): arms of one seed
        # that must start alike compare this
        summary['policy_sha256_at_start'] = policy_digest(model.policy)
        model.evidence_dir = output
        model.set_logger(configure(str(output / 'metrics'), ['stdout', 'json', 'csv'] if verbose else ['json', 'csv']))
        parts, dumps, worker_parts = [], [], []
        model.step_timing = {} if step_timing else None
        dump = model.logger.dump

        def timed_dump(*args, **kwargs):
            t0 = time.perf_counter()
            result = dump(*args, **kwargs)
            dumps.append(time.perf_counter() - t0)
            return result
        model.logger.dump = timed_dump
        if resume is None:
            with staged(events, 'initial_save', device):
                model.save(output / 'initial.zip')
        summary['learner'] = dict(
            learning_rate=model.lr_schedule(1.), n_steps=model.n_steps, batch_size=model.batch_size,
            n_epochs=model.n_epochs, gamma=model.gamma, gae_lambda=model.rollout_buffer.gae_lambda,
            clip_range=model.clip_range(1.), ent_coef=model.ent_coef, vf_coef=model.vf_coef,
            max_grad_norm=model.max_grad_norm, target_kl=model.target_kl, normalize_advantage=model.normalize_advantage,
            lstm_hidden_size=model.policy.lstm_actor.hidden_size,
            parameters=sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
            update_paths=dict(fast=type(model).FAST_UPDATE, graphed_update=type(model).GRAPHED_UPDATE,
                              graphed_collection=type(model).GRAPHED_COLLECTION,
                              merged_lookups=NameEmbedding.merges(torch.empty(0, device=device))))
        steps = [0]

        def counted_step(optimizer, args, kwargs):
            steps[0] += 1
        model.policy.optimizer.register_step_post_hook(counted_step)
        logged, step_counts, waits, memories, collect_seconds, ppo_seconds = [], [], dict(collect=[], update=[]), [], [], []
        treated = []  # per update: telemetry, the distillation record and the parameter change

        def record_losses():
            # RecurrentPPO.train() leaves its train/ values in the logger until the next iteration's dump.
            values = model.logger.name_to_value
            if any(k in values for k in LOSS_KEYS):
                logged.append({k: float(values[k]) for k in LOSS_KEYS + ('train/clip_fraction', 'train/explained_variance')
                               if k in values})
                tracker, distilled = getattr(model, 'tracker', None), getattr(model, 'distill', None)
                sizes = getattr(model, 'update_sizes', None)
                if tracker is not None or distilled is not None:  # the update's telemetry and distillation records
                    at = lambda rows: rows[len(treated)] if rows is not None and len(rows) > len(treated) else None
                    treated.append(dict(telemetry=None if tracker is None else tracker.report(),
                                        distill=at(distilled.records) if distilled is not None else None,
                                        update_size=at(sizes)))

        def update_ended():  # update i (the latest) has finished: synchronized, its losses read
            record_losses()
            waits['update'].append(synchronize(device))
            memories.append(dict(memory(device), host=host_memory()))
            index = len(waits['update']) - 1
            ppo_seconds.append(time.perf_counter() - update_started[0])
            tell('update', 'end', index=index, seconds=ppo_seconds[-1],
                 device_wait_seconds=waits['update'][-1], optimizer_steps=steps[0] - step_counts[-1],
                 losses=logged[-1] if len(logged) > index else None, memory=memories[-1],
                 collect_parts=parts[index] if index < len(parts) else None, log_seconds=dumps[-1] if dumps else None)

        class Clock(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_start(self):  # the previous update has finished and is not yet dumped
                if marks:
                    update_ended()
                    after_update(len(step_counts) - 1)
                else:
                    synchronize(device)
                if count_syncs and not marks:
                    syncs.update(context=warnings.catch_warnings(record=True))
                    syncs['caught'] = syncs['context'].__enter__()
                    warnings.simplefilter('always')
                    torch.cuda.set_sync_debug_mode('warn')
                step_counts.append(steps[0])
                tell('collect', 'start', index=len(step_counts) - 1)
                marks.append(('rollout', time.perf_counter()))

            def _on_rollout_end(self):
                waits['collect'].append(synchronize(device))
                marks.append(('update', time.perf_counter()))
                if model.step_timing is not None:
                    parts.append(dict(model.step_timing))
                    model.step_timing.clear()
                if hasattr(model.env, 'worker_timing'):  # CPU workers: their waiting and communication (sim/workers.py)
                    worker_parts.append(model.env.worker_timing())
                collect_seconds.append(marks[-1][1] - marks[-2][1])
                tell('collect', 'end', index=len(step_counts) - 1, seconds=collect_seconds[-1],
                     device_wait_seconds=waits['collect'][-1])
                if not validate:
                    update_started[0] = time.perf_counter()
                    tell('update', 'start', index=len(step_counts) - 1)
                if 'context' in syncs:
                    torch.cuda.set_sync_debug_mode('default')
                    caught = syncs.pop('caught')
                    syncs.pop('context').__exit__(None, None, None)
                    synced = [w for w in caught if 'synchronizing' in str(w.message)]
                    # by the file whose line asked for it, so a callback's own host reads count apart
                    by_file = Counter(Path(w.filename).name for w in synced)
                    syncs.update(first_rollout=len(synced), by_file=dict(by_file.most_common()), vector_steps=n_steps,
                                 messages=sorted({str(w.message)[:120] for w in caught})[:5])

        probe = dict(torch=None, before=None, replayed=None)
        paths_used = []  # per update, the speed paths its PPO phase ran with

        def before_update(index):  # after the rollout's checks, just before RecurrentPPO.train()
            if speed is not None and index == speed[0]:
                for name in speed[1]:
                    setattr(type(model.policy), SPEED_SWITCHES[name], True)
                tell('speed_paths', 'end', index=index, on=list(speed[1]))
            paths_used.append([n for n, a in SPEED_SWITCHES.items() if getattr(type(model.policy), a)])
            if index == 0:  # what either environment collected first, comparable bit for bit (rollout_digest)
                summary['first_rollout'] = rollout_digest(model.rollout_buffer)
            if replay is not None and index == replay[0]:
                probe['before'] = ppo_state(model)
            if ppo_profile is not None and index in ppo_profile[0]:
                activities = [torch.profiler.ProfilerActivity.CPU] + \
                    ([torch.profiler.ProfilerActivity.CUDA] if str(device).startswith('cuda') else [])
                synchronize(device)
                tell('ppo_profile', 'start', index=index)
                probe['torch'] = (index, torch.profiler.profile(activities=activities), time.perf_counter())
                probe['torch'][1].__enter__()

        def after_update(index):  # the update has ended and its losses are read; before the next rollout starts
            if probe['torch'] is not None:
                synchronize(device)
                k, prof, t0 = probe.pop('torch')
                probe['torch'] = None
                wall = time.perf_counter() - t0
                prof.__exit__(None, None, None)
                tell('ppo_profile', 'end', index=k, **write_ppo_profile(prof, Path(ppo_profile[1]) / f'ppo-update-{k}', wall))
            if probe['before'] is not None:
                before, probe['before'] = probe['before'], None
                kept = dict(steps=steps[0], logged=dict(model.logger.name_to_value), after=ppo_state(model),
                            switches={n: getattr(type(model.policy), a) for n, a in SPEED_SWITCHES.items()})
                try:
                    probe['replayed'] = replay_ppo(model, before, replay[1], device)
                finally:
                    put_ppo_state(model, kept['after'])
                    steps[0] = kept['steps']
                    model.logger.name_to_value.clear()
                    model.logger.name_to_value.update(kept['logged'])
                    for n, a in SPEED_SWITCHES.items():
                        setattr(type(model.policy), a, kept['switches'][n])
                tell('replay', 'end', index=index, **probe['replayed'])
        checks, carried = [], dict(paid=torch.zeros(worlds, device=device), done=torch.ones(worlds, device=device))
        syncs, validating, evidencing, evidence, seen, update_started = {}, [], [], [], {}, [0.]
        sampled = []  # per update: its saved segments and their replay

        class Segments(BaseCallback):  # after the checks: the update's fixed sample, saved and replayed before PPO uses it
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                index, start = len(step_counts) - 1, model.num_timesteps - rollout
                t0 = time.perf_counter()
                with staged(events, 'segments', device, index=index):
                    path = output / 'segments' / f'segments-{start:09d}.npz'
                    size, digest = save_segments(model, path, slots, segments['decisions'],
                                                 dict(num_timesteps=start, worlds=worlds, n_steps=n_steps, goals=list(goals),
                                                      task=task, seed=seed, env_seed=env_seed or seed))
                    replayed = replay_segments(path, model.policy, digest=digest)  # the policy save_segments hashed
                record = dict(path=f'segments/{path.name}', bytes=size, sha256=sha256(path),
                              **{k: replayed[k] for k in ('log_prob_gap', 'value_gap', 'sequence_log_prob_gap',
                                                          'sequence_value_gap', 'returns_exact', 'rows')},
                              seconds=time.perf_counter() - t0)
                sampled.append(record)
                # the one-step replay gates at ATOL, the sequence replay at SEQUENCE_ATOL (a NaN fails either)
                if not (record['log_prob_gap'] <= ATOL and record['value_gap'] <= ATOL) or not record['returns_exact']:
                    raise CheckFailed(f'the saved segments of update {index} do not replay: {record}')
                if not (record['sequence_log_prob_gap'] <= SEQUENCE_ATOL and record['sequence_value_gap'] <= SEQUENCE_ATOL):
                    raise CheckFailed(f'the saved segments of update {index} do not replay in the learner\'s sequence shape: '
                                      f'a sequence gap above {SEQUENCE_ATOL}: {record}')
                update_started[0] = time.perf_counter()

        class Validation(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):  # after the returns are computed, before train() uses the rollout
                index = len(step_counts) - 1
                t0 = time.perf_counter()
                with staged(events, 'validate', device, index=index):
                    checks.append(validate_rollout(model, carried, *task_limits(environment.env), goals=goals))
                    synchronize(device)
                validating.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                with staged(events, 'evidence', device, index=index):
                    evidence.append(rollout_evidence(model, seen))
                evidencing.append(time.perf_counter() - t0)
                update_started[0] = time.perf_counter()
                tell('update', 'start', index=index)

        class Probe(BaseCallback):  # last: after the checks and the evidence, just before the PPO phase
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                if weights is not None and len(step_counts) == 1:
                    with staged(events, 'source_check', device):
                        summary['source_check'] = check_source_outputs(model, weights, device)
                before_update(len(step_counts) - 1)

        class Recovery(BaseCallback):  # after Clock, which has read the update's losses
            def _on_step(self):
                return True

            def _on_rollout_start(self):
                if len(step_counts) > 1:
                    saved_update(len(step_counts) - 2)

        def saved_update(index):
            """Update `index` (of this process) is complete: its record row, the recovery state, a boundary checkpoint."""
            t0 = time.perf_counter()
            row = dict(index=index, num_timesteps=model.num_timesteps, n_updates=model._n_updates,
                       losses=logged[index] if index < len(logged) else None,
                       treatments=treated[index] if index < len(treated) else None,
                       checks={k: v for k, v in checks[index].items() if k != 'problems'} if validate else None,
                       evidence=evidence[index] if validate else None,
                       segments=sampled[index] if index < len(sampled) else None,
                       collect_parts=parts[index] if index < len(parts) else None,
                       worker_timing=worker_parts[index] if index < len(worker_parts) else None, epoch=time.time(),
                       seconds=dict(collect=collect_seconds[index], validation=validating[index] if validate else None,
                                    segments=sampled[index]['seconds'] if index < len(sampled) else None,
                                    evidence=evidencing[index] if validate else None, ppo=ppo_seconds[index]),
                       memory=memories[index])
            if model.num_timesteps % every == 0:
                t1 = time.perf_counter()
                model.save(output / 'checkpoints' / f'ppo_{model.num_timesteps}_steps.zip')
                if treatment_state(model) is not None:
                    torch.save(treatment_state(model), output / 'checkpoints' / f'ppo_{model.num_timesteps}_steps.treatments.pt')
                row.update(checkpoint=f'checkpoints/ppo_{model.num_timesteps}_steps.zip',
                           checkpoint_seconds=time.perf_counter() - t1)
            record = save_recovery(model, recovery, dict(epoch=time.time(), output=str(output), task=task,
                                                         goals=list(goals)))
            row.update(recovery_sha256=record['sha256'], save_seconds=time.perf_counter() - t0)
            with open(Path(recovery) / 'updates.jsonl', 'a') as f:
                f.write(json.dumps(row, default=float) + '\n')
                f.flush()
                os.fsync(f.fileno())
            tell('recovery_save', 'end', index=index, num_timesteps=model.num_timesteps, seconds=row['save_seconds'])

        class Checkpoints(CheckpointCallback):
            seconds = saves = 0
            during = {}  # update index -> seconds of the saves inside its collection

            def _on_step(self):
                t0 = time.perf_counter()
                saved = self.n_calls % self.save_freq == 0
                if saved:
                    synchronize(device)
                    tell('checkpoint_save', 'start', index=len(step_counts) - 1)
                result = super()._on_step()
                if saved:
                    Checkpoints.seconds += time.perf_counter() - t0
                    Checkpoints.saves += 1
                    index = len(step_counts) - 1
                    Checkpoints.during[index] = Checkpoints.during.get(index, 0) + time.perf_counter() - t0
                    tell('checkpoint_save', 'end', index=len(step_counts) - 1, seconds=time.perf_counter() - t0)
                return result
        every = checkpoint_every or 8 * rollout
        if recovery is not None and every % rollout:
            raise ValueError('checkpoints fall on complete updates: checkpoint_every must be a multiple of worlds x n_steps')
        learn_start = time.perf_counter()
        saving = [Recovery()] if recovery is not None else [Checkpoints(max(1, every // worlds), str(output / 'checkpoints'),
                                                                       name_prefix='ppo')]
        callbacks = [Clock(), *saving, *callbacks]
        try:
            model.learn(decisions - start_timesteps, reset_num_timesteps=resume is None,
                        callback=CallbackList(callbacks + ([Validation()] if validate else []) +
                                              ([Segments()] if segments is not None else []) + [Probe()]))
        finally:
            if validate:
                (output / 'checks.json').write_text(json.dumps(dict(rollouts=checks), indent=1) + '\n', encoding='utf-8')
        update_ended()  # the last update, whose losses no dump wrote
        after_update(len(step_counts) - 1)
        if recovery is not None:
            saved_update(len(step_counts) - 1)
        for a, on in switches.items():  # the switches are the class's: leave them as they were for whatever runs next
            setattr(type(model.policy), a, on)
        marks.append(('rollout', time.perf_counter()))
        t0 = time.perf_counter()
        with staged(events, 'final_save', device):
            model.save(output / 'latest.zip')
            if treatment_state(model) is not None:
                torch.save(treatment_state(model), output / 'latest.treatments.pt')
        times['final_save_seconds'] = time.perf_counter() - t0
        # Iteration i spans rollout i, the log dump and update i; the first one includes warm-up.
        starts = [t for kind, t in marks if kind == 'rollout']
        updates = [t for kind, t in marks if kind == 'update']
        collect = [u - s for s, u in zip(starts, updates)]
        update = [s - u for u, s in zip(updates, starts[1:])]
        iteration = [b - a for a, b in zip(starts, starts[1:])]
        if not 0 < warmup < len(iteration) + 1:
            raise ValueError(f'warmup {warmup} leaves no measured update of {len(iteration)}')
        steady = sum(iteration[warmup:])
        episodes = [e for r in model.rollouts for e in zip(*(r['episodes'][k] for k in
                                                              ('goal', 'terminal', 'episode_return', 'episode_decisions')))]
        per_goal = {goal: dict(transitions=sum(r['transitions'][g] for r in model.rollouts),
                               episodes=sum(1 for e in episodes if e[0] == g),
                               deaths=sum(1 for e in episodes if e[0] == g and e[1] == TERMINAL_DEATH),
                               mean_return=float(np.mean(returns)) if (returns := [e[2] for e in episodes if e[0] == g]) else None)
                    for g, goal in enumerate(GOALS)}
        times.update(learn_seconds=starts[-1] - learn_start, first_iteration_seconds=iteration[0],
                     steady_seconds=steady, collect_seconds=collect, update_seconds=update,
                     checkpoint_seconds=Checkpoints.seconds, checkpoint_saves=Checkpoints.saves,
                     validation_seconds=validating, evidence_seconds=evidencing,
                     segment_seconds=[r['seconds'] for r in sampled],
                     device_wait_seconds=waits)
        # the measured iterations without --validate's own work (its checks and the evidence counts)
        unchecked = steady - sum(validating[warmup:]) - sum(evidencing[warmup:])
        # update_seconds spans the end of collection to the next rollout: the checks, the evidence, the log
        # dump and RecurrentPPO.train(); ppo_update_seconds is the last two alone. collect_seconds includes any
        # checkpoint save made during the collection (checkpoint_seconds).
        summary['updates'] = [dict(index=i, iteration_seconds=iteration[i], collect_seconds=collect[i],
                                   checkpoint_seconds=Checkpoints.during.get(i, 0.0),
                                   validation_seconds=validating[i] if validate else None,
                                   evidence_seconds=evidencing[i] if validate else None,
                                   segment_seconds=sampled[i]['seconds'] if i < len(sampled) else None,
                                   ppo_update_seconds=update[i] - (validating[i] + evidencing[i] if validate else 0)
                                   - (sampled[i]['seconds'] if i < len(sampled) else 0),
                                   segments=sampled[i] if i < len(sampled) else None,
                                   device_wait_seconds=dict(collect=waits['collect'][i], update=waits['update'][i]),
                                   optimizer_steps=(step_counts[i + 1] if i + 1 < len(step_counts) else steps[0]) - step_counts[i],
                                   losses=logged[i] if i < len(logged) else None, memory=memories[i],
                                   collect_parts=parts[i] if i < len(parts) else None,
                                   log_seconds=dumps[i] if i < len(dumps) else None,
                                   speed_paths=paths_used[i] if i < len(paths_used) else None,
                                   worker_timing=worker_parts[i] if i < len(worker_parts) else None,
                                   ppo_profiled=ppo_profile is not None and i in ppo_profile[0],
                                   replayed=replay is not None and i == replay[0],
                                   checks={k: v for k, v in checks[i].items() if k != 'problems'} if validate else None,
                                   evidence=evidence[i] if validate else None,
                                   treatments=treated[i] if i < len(treated) else None,
                                   phase_epochs=dict(collect_start=round(epoch + starts[i], 3), collect_end=round(epoch + updates[i], 3),
                                                     update_end=round(epoch + starts[i + 1], 3)))
                              for i in range(len(iteration))]
        summary.update(
            status='completed', total_timesteps=model.num_timesteps, ppo_updates=len(updates),
            sb3_n_updates=model._n_updates, gradient_steps=steps[0],
            native_ticks=sum(r['native_ticks'] for r in model.rollouts), episodes=len(episodes), per_goal=per_goal,
            decisions_per_second=(decisions - start_timesteps) / (starts[-1] - learn_start), warmup_updates=warmup,
            steady_decisions_per_second=(len(iteration) - warmup) * rollout / steady if steady else None,
            steady_decisions_per_second_without_validation=(len(iteration) - warmup) * rollout / unchecked if steady else None,
            cuda_syncs_first_rollout={k: v for k, v in syncs.items() if k != 'context'} or None,
            steady_native_ticks_per_second=sum(r['native_ticks'] for r in model.rollouts[warmup:]) / steady if steady else None,
            timing=times, memory=memory(device), losses=logged, replay=probe['replayed'])
        if segments is not None:
            sizes = [r['bytes'] for r in sampled]
            summary['segments'].update(updates=len(sampled), bytes_total=sum(sizes), bytes_mean=float(np.mean(sizes)),
                                       bytes_max=max(sizes), worst_log_prob_gap=max(r['log_prob_gap'] for r in sampled),
                                       worst_value_gap=max(r['value_gap'] for r in sampled),
                                       worst_sequence_log_prob_gap=max(r['sequence_log_prob_gap'] for r in sampled),
                                       worst_sequence_value_gap=max(r['sequence_value_gap'] for r in sampled),
                                       seconds_mean=float(np.mean([r['seconds'] for r in sampled])))
        if validate:
            per_epoch = -(-rollout // batch_size)
            summary['checks'] = dict(rollouts=checks)
            with staged(events, 'checkpoint_check', device):
                summary['checks'].update(checkpoint=check_checkpoint(model, output / 'latest.zip', device))
            with staged(events, 'loss_check', device):
                own = (decisions - start_timesteps) // rollout  # this process's updates
                summary['checks'].update(losses=check_losses(logged, own, steps[0], own * n_epochs * per_epoch))
            (output / 'checks.json').write_text(json.dumps(summary['checks'], indent=1) + '\n', encoding='utf-8')
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        summary['wall_seconds'] = time.perf_counter() - started
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def bench(*, worlds, n_steps, batch_size, n_epochs, updates, seed, device, env='fake', threads=None, pool=None,
          goals=GOALS, validate=False, tables=None, count_syncs=False):
    """train() for `updates` updates in a scratch directory, one checkpoint every 2 updates. The first update
    includes warm-up; the steady rate covers the rest, logging and checkpoint saves included."""
    with tempfile.TemporaryDirectory() as scratch:
        result = train(Path(scratch) / 'run', decisions=updates * worlds * n_steps, seed=seed, worlds=worlds,
                       n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, device=device, env=env,
                       checkpoint_every=2 * worlds * n_steps, threads=threads, verbose=False, pool=pool, goals=goals,
                       validate=validate, tables=tables, count_syncs=count_syncs)
    result['timing'].update(update_mean_seconds=float(np.mean(result['timing']['update_seconds'])),
                            collect_mean_seconds=float(np.mean(result['timing']['collect_seconds'][1:] or [np.nan])))
    return dict(stage='learner', **result)


def survey(*, worlds, n_steps, rollouts, seed, device, env='cpu', pool=None):
    """Collection only, never an update: how often worlds reach content the simulator lacks under the
    untrained policy. A rollout that Unsupported voids is counted, not trained on, and collection goes
    on from where it stopped (the environment has restarted those worlds). Reports unsupported world-steps
    per 1,000 decisions, their share of episode ends, reasons and goals, voided rollouts and the collection
    rate (environment, policy and buffer; no update)."""
    from sim.learn import Unsupported, build
    t0 = time.perf_counter()
    environment = make_env(env, worlds, seed, device, pool)
    model = build(environment, seed=seed, device=device, n_steps=n_steps, batch_size=worlds * n_steps, n_epochs=1)
    startup = time.perf_counter() - t0
    _, callback = model._setup_learn(worlds * n_steps * rollouts, None)
    rows, voided, ends, ticks = [], 0, Counter(), 0
    t0 = time.perf_counter()
    for _ in range(rollouts):
        try:
            model.collect_rollouts(environment, callback, model.rollout_buffer, n_steps)
        except Unsupported as error:
            voided += 1
            rows += error.evidence['rows']
        ended, counts = model.rollout_buffer.episodes()
        ends.update(zip(ended['goal'], ended['terminal']))
        ticks += counts['native_ticks']
    synchronize(device)
    wall = time.perf_counter() - t0
    decisions, done = model.num_timesteps, sum(ends.values())
    return dict(stage='survey', env=env, worlds=worlds, n_steps=n_steps, rollouts=rollouts, decisions=decisions,
                native_ticks=ticks, unsupported_world_steps=len(rows),
                unsupported_per_1000_decisions=1000 * len(rows) / decisions,
                episode_ends=done, unsupported_share_of_episode_ends=len(rows) / max(1, len(rows) + done),
                voided_rollouts=voided,
                by_goal={goal: dict(unsupported=sum(r['goal'] == g for r in rows), horizon=ends[(g, TERMINAL_HORIZON)],
                                    death=ends[(g, TERMINAL_DEATH)]) for g, goal in enumerate(GOALS)},
                reasons=dict(Counter('unsupported' for r in rows).most_common()),
                first_unsupported_decision=[min((r['decision'] for r in rows if r['world'] == w), default=None)
                                            for w in range(worlds)][:16],
                collect_decisions_per_second=decisions / wall, startup_seconds=startup, wall_seconds=wall,
                memory=memory(device), host=versions(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay-segments', help='replay saved per-update segments with the policy that collected them')
    replay.add_argument('segments', type=Path, nargs='+', help='segments-N.npz files')
    replay.add_argument('--policy', type=Path, required=True, help='the checkpoint that collected them (its digest is checked)')
    replay.add_argument('--device', default='cpu', help='the device the run collected on (ATOL is a same-device tolerance)')
    replay.add_argument('--rows', type=int, help="the one-step replay's batch rows (default: the collection's world count)")
    replay.add_argument('--cudnn-tf32', choices=('on', 'off'), help="diagnosis: torch.backends.cudnn.allow_tf32 for the "
                        "replay (default: torch's own, on)")
    replay.add_argument('--cudnn', choices=('on', 'off'), help='diagnosis: torch.backends.cudnn.enabled for the replay')
    replay.add_argument('--out', type=Path, help='also write the results as JSON here')
    for name in ('train', 'bench', 'survey'):
        command = commands.add_parser(name)
        command.add_argument('--seed', type=int, default=101 if name == 'train' else 909)
        command.add_argument('--device', default='cpu')
        command.add_argument('--env', default='fake', choices=('fake', 'cpu'))
        command.add_argument('--workers', type=int, help='--env cpu: worker processes (sim/workers.py)')
        if name == 'train':
            command.add_argument('--engine', default='cpu', choices=('cpu', 'native'),
                                 help='--env cpu: the Python reference (cpu) or the C port (native, needs --pack)')
            command.add_argument('--pack', type=Path, help='--engine native: the exported pack (python -m sim.native export)')
        command.add_argument('--tables', type=Path, help='--env cpu: the simulator archive (default logs/sim/data, pins checked)')
        command.add_argument('--worlds', type=int, default=BATCHING['worlds'])
        command.add_argument('--pool', type=Path, help='--env cpu: a sim/world.py layout pool saved with --derived')
        command.add_argument('--n-steps', type=int, default=BATCHING['n_steps'], help='decisions per world per update')
        if name in ('train', 'bench'):
            command.add_argument('--batch-size', type=int, default=BATCHING['batch_size'])
            command.add_argument('--epochs', type=int, default=BATCHING['n_epochs'])
            command.add_argument('--threads', type=int, help='torch CPU threads (default: torch\'s own)')
            command.add_argument('--goals', nargs='+', default=list(GOALS), choices=GOALS,
                                 help='the declared goal set: all four (default; slot b asks for GOALS[b %% 4]) or one goal in every slot')
            command.add_argument('--validate', action='store_true', help='check every rollout and the saved checkpoint (checks.json)')
            command.add_argument('--graphed', nargs='*', choices=('update', 'collection'), help='CUDA: the update and/or the '
                                 'collector\'s policy call as captured CUDA graphs (sim/learn.py GRAPHED_UPDATE, '
                                 'GRAPHED_COLLECTION; no value: both)')
            command.add_argument('--stock-update', action='store_true', help='SB3\'s RecurrentPPO.train() instead of '
                                 'sim/learn.py fast_train (the same update; a speed baseline)')
            command.add_argument('--lookups', choices=('merged', 'per-input'), help='baseline.policy NameEmbedding.MERGED: '
                                 'one lookup per embedding and call, or one per input and use (default: merged wherever '
                                 'the whole name table is projected, i.e. on CUDA)')
            command.add_argument('--ppo-profile', type=int, nargs='+', metavar='INDEX', help='these updates\' PPO phases under '
                                 'torch.profiler (operator table and totals in OUTPUT/ppo-profile)')
        if name == 'train':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--decisions', type=int, required=True, help='a multiple of worlds x n_steps')
            command.add_argument('--checkpoint-every', type=int, help='decisions between checkpoints (default 8 updates)')
            command.add_argument('--task', default='goal-activity-v2',
                                 choices=('goal-activity-v2', 'goal-activity-v3', 'goal-activity-v4', 'goal-activity-v5',
                                          'goal-activity-v11'))
            command.add_argument('--search-pool', type=Path, help='goal-activity-v4: the search layouts; v5 and v11: the supply '
                                                                 'layouts (python -m sim.search pool / supply-pool)')
            command.add_argument('--memory', type=Path, help='a sim/distill.py memory: its drift probe after every update')
            command.add_argument('--distill', action='store_true', help='with --memory: the self-distillation phase (arm S)')
            command.add_argument('--distill-strata', nargs='+', help='with --distill: train only these memory strata, each at '
                                 'its arm-S weight (arm T)')
            command.add_argument('--replay', nargs='+', metavar='INDEX VARIANT', help='after update INDEX, replay its PPO phase '
                                 'under each REPLAY_VARIANTS name (the first is the reference): timing and parameter differences')
            command.add_argument('--weights', type=Path, help='start from this checkpoint\'s policy tensors only (fresh optimizer)')
            command.add_argument('--recovery', type=Path, help='leave a resumable state here after every complete update')
            command.add_argument('--resume', type=Path, help='continue from this recovery directory')
            command.add_argument('--env-seed', type=int, help='the environment\'s seed (default --seed; a resume declares a new one)')
            command.add_argument('--segments', action='store_true', help='per-update evidence: save and replay a fixed '
                                 'sample of world segments every update (segments/)')
            command.add_argument('--segment-worlds', type=int, default=SEGMENTS['worlds'],
                                 help='world slots in the sample, split equally between the goals')
            command.add_argument('--segment-decisions', type=int, default=SEGMENTS['decisions'],
                                 help='decisions of each slot, from its rollout\'s first')
            command.add_argument('--start-when', type=Path, help='set up (imports, inputs\' sha256, host facts), then wait '
                                 'until this file exists before the run starts')
        else:
            command.add_argument('--out', type=Path, help='also write the result as JSON here')
        if name == 'bench':
            command.add_argument('--updates', type=int, default=5)
        if name == 'survey':
            command.add_argument('--rollouts', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'replay-segments':
        import torch
        from sb3_contrib import RecurrentPPO
        if args.cudnn_tf32:
            torch.backends.cudnn.allow_tf32 = args.cudnn_tf32 == 'on'
        if args.cudnn:
            torch.backends.cudnn.enabled = args.cudnn == 'on'
        policy = RecurrentPPO.load(args.policy, device=args.device).policy
        results = [dict(replay_segments(path, policy, rows=args.rows), cudnn_tf32=torch.backends.cudnn.allow_tf32,
                        cudnn=torch.backends.cudnn.enabled) for path in args.segments]
        print(json.dumps(results, indent=1))
        if args.out:
            args.out.write_text(json.dumps(results, indent=1) + '\n', encoding='utf-8')
        if any(not (r['log_prob_gap'] <= ATOL and r['value_gap'] <= ATOL) or not r['returns_exact'] for r in results):
            raise SystemExit(f'a sample does not replay within {ATOL}')
        return
    if getattr(args, 'stock_update', False):
        from sim.learn import DeviceRecurrentPPO
        DeviceRecurrentPPO.FAST_UPDATE = False
    if getattr(args, 'lookups', None):
        from baseline.policy import NameEmbedding
        NameEmbedding.MERGED = args.lookups == 'merged'
    if getattr(args, 'graphed', None) is not None:
        from sim.learn import DeviceRecurrentPPO
        if not args.device.startswith('cuda'):
            raise SystemExit('--graphed captures CUDA graphs: give --device cuda')
        paths = set(args.graphed or ('update', 'collection'))
        DeviceRecurrentPPO.GRAPHED_UPDATE, DeviceRecurrentPPO.GRAPHED_COLLECTION = 'update' in paths, 'collection' in paths
    if args.command == 'train':
        result = train(args.output, decisions=args.decisions, seed=args.seed, worlds=args.worlds, n_steps=args.n_steps,
                       batch_size=args.batch_size, n_epochs=args.epochs, device=args.device, env=args.env,
                       checkpoint_every=args.checkpoint_every, threads=args.threads, pool=args.pool, goals=tuple(args.goals),
                       validate=args.validate, workers=args.workers, tables=args.tables, task=args.task, weights=args.weights,
                       recovery=args.recovery, resume=args.resume, env_seed=args.env_seed, search_pool=args.search_pool,
                       distill_memory=args.memory, distill=args.distill, distill_strata=args.distill_strata,
                       segments=dict(worlds=args.segment_worlds, decisions=args.segment_decisions) if args.segments else None,
                       replay=(int(args.replay[0]), tuple(args.replay[1:])) if args.replay else None,
                       ppo_profile=(set(args.ppo_profile), args.output / 'ppo-profile') if args.ppo_profile else None,
                       engine=args.engine, pack=args.pack, start_when=args.start_when)
    elif args.command == 'bench':
        result = bench(worlds=args.worlds, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
                       updates=args.updates, seed=args.seed, device=args.device, env=args.env, threads=args.threads,
                       pool=args.pool, goals=tuple(args.goals), validate=args.validate)
    else:
        result = survey(worlds=args.worlds, n_steps=args.n_steps, rollouts=args.rollouts, seed=args.seed,
                        device=args.device, env=args.env, pool=args.pool)
    print(json.dumps(result, indent=1))
    if args.command != 'train' and args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
