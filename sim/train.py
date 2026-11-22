"""Simulator training on the device learner (sim/learn.py): config.LEARNER's fresh RecurrentPPO over a
batched sim/spec.py environment, with every per-decision tensor on one device.

    .venv-sim/bin/python -m sim.train train --env cpu --workers 4 --pool logs/sim/world/train-8-nc1-derived.npz \
        --output logs/sim/runs/check-s101 --worlds 32 --n-steps 32 --batch-size 256 --decisions 2048 --validate
    .venv-sim/bin/python -m sim.train bench --worlds 64 --updates 5 --out logs/sim/bench/learner-cpu.json
    .venv-sim/bin/python -m sim.train survey --env cpu --pool POOL --worlds 64 --n-steps 32 --rollouts 4

train writes initial.zip, checkpoints/, latest.zip (standard RecurrentPPO zips that baseline/evaluate.py
loads), metrics/ (SB3's json and csv), unsupported-*.json if a world reached unimplemented content (the
run stops before that update), and summary.json: settings read back from the model, the departures from
config.LEARNER, per-goal transitions and episodes, timing and memory, and the policy_digest of the weights the run
started from and of each zip it saved (policy_sha256_at_start, policy_sha256_at_end for latest.zip, checkpoint_digests
by file name under checkpoints/), which ties a checkpoint to its run. bench runs train for a fixed number
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

--task free-play-v1 is the goal-free study: an obs-v6 policy with no goal, lives that end
by death or by their drawn length, the per-tick discount, the cut bootstrap and the normalized value target (sim/learn.py),
with FREE_PLAY_LEARNER's settings unless --half-life, --gae-lambda or --value-norm give others, and --n-steps given (the
rollout length is the study's to set). No goal set, no goal balance and no goal check: --validate runs
validate_free_play. --batch-setting passes the free-play batch's own keyword settings (sim/vec.py CpuBatch); on the CPU
workers it names the world, the muggers and the life-length distribution (FREE_PLAY_NAMED), and the summary records the
batch's settings in effect, its world among them (sim/workers.py BATCH_SETTINGS).

--env real trains free-play-v1 in the real engine's worlds inside scripts/stack.py (sim/real.py): many agents a world,
each deciding when it is due, their segments of n_steps decisions pooled into rollouts of --worlds columns; the
--batch-setting names `starts` (the world whose start records the lives draw) and `life`, and may name `worlds`,
`agents` (a world's; the agents, worlds x agents, are apart from --worlds, the segments of an update), `sites` and
`radius`; the stack runs with RL_SHARED_AGENTS=agents. The summary records engine 'real', the batch's settings in effect
(the stack's facts among them), 'real' with each update's collector record (its rows by age in updates, PPO's
approximate KL and clip fraction for each age before and after the update, the segments dropped as too old) and
'real_finish': at the run's end every live agent leaves and its life's ledger is checked. --validate checks each row
with the weights of its own version. --task free-play-v2 (--env real only, with --death-price 50): free-play-v1 and a
price of 50 levels at the transition that ends a life by the engine's death (sim/real.py, sim/workers.py FREE_PLAY_V2),
checked against the engine's death records at every reply, at each life's end and at the run's end; the summary
('free_play', 'learner', 'lives', 'real') and every checkpoint (VERSION_SAVED) name the version and its price, the lives'
level gain apart from the prices. The simulator engines refuse it by name. --task free-play-v3 (--env real only):
free-play-v1 paid over every stat of the engine's ledger, with a tinderbox in the start kit (sim/workers.py
FREE_PLAY_V3); its life rows, progress rows and summary hold the gain over the seven old skills and over every stat
apart, and its checkpoints save the paid stats and the kit (VERSION_SAVED). The half-life (--half-life) is the
learner's, saved with every checkpoint and repeated by a leg, in any version. --task free-play-v10 (--env real only):
free-play-v3 on the item contract item-v3,
the page of kinds (sim/workers.py FREE_PLAY_V10), whose policy records the item contract beside its action, mask and
feedback versions (baseline/policy.py run_contract); every check of a checkpoint is given the run's contract, so nothing
loads across the item contracts but the declared crossing below. --task free-play-v11 (--env real only): free-play-v10
on the wide contract (sim/workers.py FREE_PLAY_V11; rl/wide_contract.py): the page of 16 kinds (item-v4), act-v4's 128
rows, mask-v6 and obs-v7. A free-play-v3 or v10 checkpoint reaches it by --weights through the tested conversion
(baseline/policy.py widen_state), recorded (`conversion`). --task free-play-v12 (--env real only): free-play-v10 with
each life started at its start record's drawn hitpoints; a free-play-v10 checkpoint starts it by --weights. --task
free-play-v15 (--env real only): free-play-v14 with the pay rare-xp-v2 (sim/real.py RareXp), an inner pay added to the
learner's reward and recounted before each update, the levels' pay apart, whose window every checkpoint saves
(rare_xp_window) and a leg goes on from; a free-play-v12 checkpoint starts it by --weights, and a v15 one a v14 run (the
pay off). --task free-play-v14
(--env real only): free-play-v12 with the start rule
old-life-v1 (sim/workers.py FREE_PLAY_V14: every agent of odd index starts every life with the old-life profile),
recorded in the run's settings and the batch's (start_rule); a free-play-v12 checkpoint starts it by --weights, and a leg
goes on under the same rule. A free-play-v1 checkpoint starts a
free-play-v2 or free-play-v3 run by --weights alone, a free-play-v3 one a free-play-v10 or v11 run, a free-play-v10 one a
free-play-v11 or v12 run, each recorded with both versions and the version's note (load_weights, WEIGHTS_ACROSS);
nothing else loads across the versions. A checkpoint of a removed study arm (sim/workers.py REMOVED: free-play-v4, v5,
v7, v8, v9, v13) is refused by name.
--env real --device cuda runs the update on the GPU and names where the collection's policy calls run (--collect-device
cpu: a copy on the processor that takes the model's weights at the start of each collection, sim/real.py HANDOVER; cuda:
the model's own policy) and cuDNN's TF32 (--cudnn-tf32 off or on; on is refused with the calls on the GPU and
--validate, whose recomputation has other shapes); summary 'devices' and every checkpoint (real_devices) name the
update's and the collection's devices, the GPU and the TF32 settings, and --weights across devices records both sides. A
processor run keeps its path and its records. A real-engine run takes no --resume (its start draws would repeat) and no
--segments; its checkpoints are saved after the update that reached their count (a checkpoint holds the updates of every
rollout it counts). In place of a resume it runs in legs: --leg-from CHECKPOINT --leg N is a new run that starts from a
checkpoint of leg N - 1 with the whole learner state it holds (load_leg: the policy with its value normalization's
statistics, Adam's state, the decision and update counts, the same learner settings and seed), draws its start records,
sites and life lengths from leg N's streams of the run's environment seed (sim/real.py LEG_STREAMS; every agent begins a
new life at the leg's start), seeds torch's and numpy's generators by (seed, N), counts its decisions from the
checkpoint's count up to --decisions, the run's total, and records the parent run, the checkpoint's digest and the leg
in summary 'leg' (LEG_SCOPES labels the whole run's counts apart from the leg's) and, at its start, in lineage.json.
Every real-engine checkpoint saves its leg record (model.real_leg: the leg's number, the first leg's output, the
environment seed, the total, --validate, --checkpoint-every and the batch's world), and a leg repeats each of them, its
learner settings and its devices, or is refused before it trains (leg_parent, check_leg_world); a checkpoint saved
before its update is refused, and one from before the record existed continues only as leg 2 with --unrecorded-parent,
which declares what it cannot check (LEG_UNVERIFIED). From leg 2 on, each row of lives.jsonl and lives-xp.jsonl names
its leg. A checkpoint of a run with self-imitation (self-imitation-v1, --self-imitation, now removed) holds its setting
(self_imitation_setting), so no leg continues it (LEG_SAME).
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
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
from sim.task import FREE_PLAY  # noqa: E402
from item_contract import ITEM_V2  # noqa: E402
from sim.workers import (FREE_PLAY_TASK, FREE_PLAY_TASKS, FREE_PLAY_V2_TASK, FREE_PLAY_V3_TASK,  # noqa: E402
                         FREE_PLAY_V10_TASK, FREE_PLAY_V11_TASK, FREE_PLAY_V12_TASK, FREE_PLAY_V14_TASK,
                         FREE_PLAY_V15_TASK, REAL_WORLDS, death_price, refuse_removed,
                         item_contract, observation_table, refuse_real_only, task_batch)

# 1,024 worlds (256 per goal) x 128 decisions = 131,072 transitions per update; each 4,096-transition
# minibatch is 32 whole world-rollouts, 8 per goal; 4 epochs = 128 gradient steps per update.
BATCHING = dict(worlds=1024, n_steps=128, batch_size=4096, n_epochs=4)
# free-play-v1's learner: the per-tick discount's half-life in game ticks (3,000 is the declared sensitivity value),
# lambda, and PopArt (baseline/policy.py POPART; 'none' is the declared ablation). Defaults of the command, not a
# freeze; config.LEARNER stays the goal tasks'.
FREE_PLAY_LEARNER = dict(half_life_ticks=1000, gae_lambda=.95, value_norm='popart')
# --ent-coef's range (free play only): PPO's entropy coefficient, config.LEARNER's 0.01 unless a run names another
ENT_COEF_RANGE = (0.0, 0.2)
# The free-play batch's settings a run on the CPU workers must name, which have no default it may rely on: muggers, the
# life-length distribution (a proposal, not frozen) and the world (sim/combat.py VILLAGES: the pool's layouts decide it,
# and the batch refuses a pool of another world than the one named); sim/vec.py CpuBatch's names.
FREE_PLAY_NAMED = ('muggers', 'life', 'world')


def free_play_settings(task, half_life_ticks=None, gae_lambda=None, value_norm=None, batch=None, ent_coef=None):
    """A run's goal-free settings, as its summary and recovery state record them: None for a goal task, which takes none
    of them; for FREE_PLAY_TASK each one given or FREE_PLAY_LEARNER's (value_norm 'popart' or 'none'), the per-tick discount
    gamma_tick it gives, and the batch's own settings; for free-play-v2 also its name and death price (sim/workers.py).
    ent_coef (--ent-coef, in ENT_COEF_RANGE): PPO's entropy coefficient, recorded only when given, so a run without it
    records what it did before and a resume refuses a changed one."""
    given = dict(half_life_ticks=half_life_ticks, gae_lambda=gae_lambda, value_norm=value_norm)
    if task not in FREE_PLAY_TASKS:
        if batch or any(v is not None for v in given.values()):
            raise ValueError(f'the half-life, lambda, value normalization and batch settings are {FREE_PLAY_TASK}\'s')
        if ent_coef is not None:
            raise ValueError(f'--ent-coef is the free-play tasks\' ({", ".join(FREE_PLAY_TASKS)}); {task} trains with '
                             f'config.LEARNER\'s {config.LEARNER["ent_coef"]}')
        return None
    low, high = ENT_COEF_RANGE
    if ent_coef is not None and not low <= ent_coef <= high:  # a NaN too
        raise ValueError(f'--ent-coef {ent_coef}: the entropy coefficient is from {low} to {high}')
    chosen = {k: FREE_PLAY_LEARNER[k] if v is None else v for k, v in given.items()}
    if not chosen['half_life_ticks'] > 0 or not 0 <= chosen['gae_lambda'] <= 1 or chosen['value_norm'] not in ('popart', 'none'):
        raise ValueError(f'a positive half-life, lambda in [0, 1] and value normalization popart or none, not {chosen}')
    from baseline.policy import POPART
    return dict(half_life_ticks=float(chosen['half_life_ticks']), gamma_tick=0.5 ** (1 / chosen['half_life_ticks']),
                gae_lambda=float(chosen['gae_lambda']), value_norm=dict(POPART) if chosen['value_norm'] == 'popart' else None,
                bootstrap='a life cut by its drawn length from its last observation; a death none',
                batch=json.loads(json.dumps(batch or {})),  # as the recovery state's JSON holds it
                **(dict(task=task, death_price=death_price(task)) if task == FREE_PLAY_V2_TASK else {}),
                **(dict(task=task, paid='every stat of the ledger', kit_extra=[list(x) for x in task_batch(task)['kit_extra']])
                   if task in EVERY_STAT_TASKS else {}),
                **({} if task not in FREE_PLAY_TASKS or item_contract(task) == ITEM_V2 else dict(items=item_contract(task))),
                **(dict(start_hp='drawn') if task_batch(task).get('start_hp') else {}),
                **({} if 'start_rule' not in task_batch(task) else dict(start_rule=task_batch(task)['start_rule'])),
                **(dict(inner_pay=task_batch(task)['inner'], value='one value estimate learns the levels and the rare-XP '
                        'pay together') if task == FREE_PLAY_V15_TASK else {}),
                **({} if ent_coef is None else dict(ent_coef=float(ent_coef))))


# the versions that pay every stat of the ledger with the tinderbox in the start kit (sim/workers.py FREE_PLAY_V3 and
# later); of them, the one with the inner pay, whose window every checkpoint saves (free-play-v15, rare-xp-v2)
EVERY_STAT_TASKS = (FREE_PLAY_V3_TASK, FREE_PLAY_V10_TASK, FREE_PLAY_V11_TASK, FREE_PLAY_V12_TASK, FREE_PLAY_V14_TASK,
                    FREE_PLAY_V15_TASK)
INNER_TASKS = (FREE_PLAY_V15_TASK,)


def counts_saved(task):
    """The checkpoint attribute that holds an inner pay task's state: rare-xp-v2's window (sim/real.py RareXp)."""
    if task not in INNER_TASKS:
        raise ValueError(f'{task} has no inner pay')
    return 'rare_xp_window'


def free_play_table():
    """(version, table of shapes) of free-play-v1's observations as its batch declares them: sim/task.py FREE_PLAY's
    observation version (obs-v6) and sim/spec.py's table of it, the one place the training command reads them. The fake
    environment declares its own stand-in."""
    from sim.spec import OBSERVATIONS
    return FREE_PLAY['observation'], OBSERVATIONS[FREE_PLAY['observation']]


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
             search_pool=None, engine='cpu', pack=None, batch=None, output=None, leg=1, collect_device=None):
    """cpu: the CPU reference on the generated layouts of `pool` in `workers` processes (sim/workers.py; tables: the
    archive's path, which each worker loads, default logs/sim/data with its pins checked), or with engine 'native' the C
    port (sim/native.py NativeBatch on `pack`); fake: sim/learn.FakeEnv. goals: the declared goal set (sim/spec.py
    goal_set). FREE_PLAY_TASK: no goals; on the CPU workers, with either engine, its obs-v6 table (free_play_table) and
    the batch's own settings `batch`, on the fake environment FakeEnv's free-play stand-in. real (FREE_PLAY_TASK only):
    the real engine's worlds of scripts/stack.py (sim/real.py make_env, `batch` its settings; `worlds` the learner's
    columns; its lives and stop evidence in `output`), keyed by `seed`, its agents on leg `leg`'s draw streams."""
    from sim.learn import DeviceEnv, FakeEnv, HostDeviceEnv
    if name == 'real':
        if task not in FREE_PLAY_TASKS or pool is not None or tables is not None or search_pool is not None:
            raise ValueError(f'the real engine plays {" or ".join(FREE_PLAY_TASKS)} in its own world: no pool, tables or '
                             'search pool')
        from sim.real import make_env as real_env
        return real_env(worlds, seed, output, device, batch, leg=leg, collect_device=collect_device, **task_batch(task))
    refuse_real_only(task, 'fake' if name == 'fake' else engine)
    if task == FREE_PLAY_TASK:
        if name == 'cpu':
            if pool is None:
                raise ValueError('--env cpu trains on generated worlds: give --pool')
            from sim.workers import CpuWorkers
            observation, shapes = free_play_table()
            return HostDeviceEnv(CpuWorkers(worlds, workers or 1, pool, tables=tables, task=task, search_pool=search_pool,
                                            goals=None, engine=engine, pack=pack, observation=observation, shapes=shapes,
                                            batch=batch), seed, device)
        if batch:
            raise ValueError('the fake environment takes no batch settings')
        from names import name_table
        return DeviceEnv(FakeEnv(worlds, id_rows=name_table().spec[0], free_play=True), seed, device)
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
# else is exact. A free-play rollout's rows (validate_free_play) are held to ATOL + RTOL * scale instead.
ATOL = 1e-5
# The free-play recurrent check's relative part (19 November 2026): each row's recomputed log-probability and value
# against the collected ones within ATOL + RTOL * scale, the scale the magnitude the output is computed from, not the
# output. A log-probability is z_a - logsumexp(z) of the masked logits z (baseline/policy.py _masked_logits), so its
# float32 rounding is a few units of the largest legal |z| even when the log-probability is near zero; a value is the
# head's sum (PopArt: std * (W h + b) + mean), whose rounding is a few units of its terms' magnitudes, |std| (|W| |h| +
# |b|) + |mean| (value_scale). A call of another shape sums in another order, and the scales grow as training moves the
# logits (by a common offset too, which the softmax ignores). One real-engine run stopped at a gap just above ATOL after
# five checks at 7.4e-6 to 8.6e-6, four of them on outputs below 8 in magnitude (the gaps are multiples of 2**-21), so
# the outputs were not large; another (free-play-v1) reached 8.1e-6. Measured on this Mac's processor (the final weights
# of two such runs, 4,096 rows of observations drawn from the space, calls of 32 rows against one call): at most
# 4.1 units of 2**-24 of the logit scale and 1.9 of the value scale. RTOL is eight float32 epsilons (2**-23), four times
# the largest measured; a row of other weights or another stored state differs by orders of magnitude more
# (sim/tests/test_learn.py). The logits alone are too small a scale for a row whose legal logits are small beside the
# terms the actor sums on the way to them (a free-play-v14 run stopped at 1.06 times its tolerance on a cloud GPU
# machine's processor, 21 November 2026): a log-probability's scale also holds the actor's largest first-encoder-layer
# term and its LSTM's largest gate term (actor_terms). Measured on this Mac's processor against the same rows recomputed
# in float64 (one run's final weights; 26,880 rows that stage 1 recorded, old-life profile and as drawn): the error
# reaches about 100 units of 2**-24 of the logit scale alone and at most 3.7 of that sum, in both kinds of life, so two
# float32 calls differ by at most 7.5; the value's error is at most 2.1 units of its own scale.
RTOL = 2 ** -20
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
    (sim/combat.py task_spec); the fake environment plays goal-activity-v2 at one horizon. Free play has none."""
    if hasattr(env, 'horizons'):
        return tuple(env.horizons), tuple(env.targets)
    return (env.horizon,) * len(GOALS), tuple(GOAL_TARGETS[g] for g in GOALS)


def recurrent_gaps(model, policy=None, keep=None, scaled=False):
    """validate_rollout's recurrent-state checks on the rollout in the buffer: (gap, reset_gap). gap: the largest
    difference between each step's collected value and log-probability and those recomputed from its stored observation
    and LSTM states. reset_gap: at the episode starts, the largest difference between the outputs from the stored states
    and from zero states (the policy must ignore the stored state there). Every stored row is one single-step sequence
    from its own stored state, so on the CPU VALIDATE_ROWS rows (whole steps) go through the policy at once (a recomputed
    value may then differ from the collected one in its last bits). On a GPU each call keeps the collection's shape, one
    step of every world: there cuDNN's LSTM may run in TF32, whose result depends on the shape. The episode starts of the
    whole rollout are compared together after the steps, VALIDATE_ROWS rows a pair of calls: each zero-state output beside
    the stored-state output of a call of the same shape (a row without a start has the same input in both, so its
    difference is zero by construction). The maxima stay on the device until the end: two reads a rollout. policy: the
    weights to recompute with (default the model's); keep: [n_steps, n_envs] the rows to check (default every row): a
    real-engine rollout checks the rows each set of weights collected with those weights (sim/real.py). The rows are
    recomputed on the device of `policy` (a real-engine collection's device, sim/real.py collect_device), with the GPU's
    one step a call there and VALIDATE_ROWS rows on the processor. scaled (validate_free_play): each row's tolerance
    is ATOL + RTOL * scale, the scale of a log-probability the largest legal |logit| of its row and the actor's largest
    first-encoder-layer and LSTM gate terms (actor_terms), and that of a value its head's terms (value_scale), all from
    the recomputation; returns (gap, reset_gap, worst, scales), worst the row
    whose gap takes the largest share of its tolerance: ratio (gap over tolerance: above 1 fails, a NaN gap's is
    infinite), output (OUTPUTS), gap, collected, scale, tolerance, step and column; scales the largest scale of each
    output over the rows checked (a NaN stays), which validate_free_play requires finite (a third read)."""
    import torch
    b, policy = model.rollout_buffer, model.policy if policy is None else policy
    device = next(policy.parameters()).device
    obs, actions = b.observations, b.actions[..., 0]
    chunk = 1 if device.type == 'cuda' else max(1, VALIDATE_ROWS // b.n_envs)
    flat = lambda store: store.reshape(-1, *store.shape[2:])
    state = lambda store, rows: flat(store)[rows].transpose(0, 1).contiguous().to(device)
    gap = reset_gap = torch.zeros((), device=device)
    candidates, outputs = [], []  # scaled: each call's worst row of each output, [ratio, gap, collected, scale, row]
    largest = torch.zeros(len(OUTPUTS), device=device)  # scaled: each output's largest scale
    with torch.no_grad(), (computed_from(policy) if scaled else nullcontext()) as seen:
        for s0 in range(0, b.buffer_size, chunk):
            rows = slice(s0 * b.n_envs, min(b.buffer_size, s0 + chunk) * b.n_envs)
            o = {k: flat(v)[rows].to(device, b._dtypes[k]) for k, v in obs.items()}
            starts = flat(b.episode_starts)[rows].to(device)
            dist, _ = policy.get_distribution(o, (state(b.hidden_states_pi, rows), state(b.cell_states_pi, rows)), starts)
            values = policy.predict_values(o, (state(b.hidden_states_vf, rows), state(b.cell_states_vf, rows)), starts)
            collected = flat(b.log_probs)[rows].to(device), flat(b._values)[rows].to(device)
            log_prob_gap = (dist.log_prob(flat(actions)[rows].to(device)) - collected[0]).abs()
            value_gap = (values.flatten() - collected[1]).abs()
            if keep is not None:
                log_prob_gap, value_gap = (torch.where(flat(keep)[rows].to(device), g, 0) for g in (log_prob_gap, value_gap))
            gap = torch.maximum(gap, torch.maximum(log_prob_gap.max(), value_gap.max()))
            if scaled:
                scales = (torch.where(o['mask'] != 0, seen['logits'].abs(), 0).amax(-1) + seen['encoder'] + seen['gates'],
                          value_scale(policy.value_net, seen['latent']))
                for output, (g, c, scale) in enumerate(zip((log_prob_gap, value_gap), collected, scales)):
                    if keep is not None:
                        scale = torch.where(flat(keep)[rows].to(device), scale, 0)
                    largest[output] = torch.maximum(largest[output], scale.max())
                    ratio = torch.nan_to_num(g / (ATOL + RTOL * scale), nan=torch.inf)
                    i = ratio.argmax()
                    candidates.append(torch.stack([t.double() for t in (ratio[i], g[i], c[i], scale[i], i + rows.start)]))
                    outputs.append(OUTPUTS[output])
        first = flat(b.episode_starts) != 0
        first = (first if keep is None else first & flat(keep)).nonzero().squeeze(1)
        for r0 in range(0, len(first), VALIDATE_ROWS):
            rows = first[r0:r0 + VALIDATE_ROWS]
            o = {k: flat(v)[rows].to(device, b._dtypes[k]) for k, v in obs.items()}
            starts = flat(b.episode_starts)[rows].to(device)
            pi = (state(b.hidden_states_pi, rows), state(b.cell_states_pi, rows))
            vf = (state(b.hidden_states_vf, rows), state(b.cell_states_vf, rows))
            zero_pi, zero_vf = tuple(torch.zeros_like(t) for t in pi), tuple(torch.zeros_like(t) for t in vf)
            dist, _ = policy.get_distribution(o, pi, starts)
            dist0, _ = policy.get_distribution(o, zero_pi, starts)
            reset_gap = torch.maximum(reset_gap, torch.maximum(
                (dist0.distribution.logits - dist.distribution.logits).abs().max(),
                (policy.predict_values(o, zero_vf, starts) - policy.predict_values(o, vf, starts)).abs().max()))
    if not scaled:
        return float(gap), float(reset_gap)
    table = torch.stack(candidates)
    k = int(table[:, 0].argmax())
    ratio, worst_gap, collected, scale, row = table[k].tolist()
    return float(gap), float(reset_gap), dict(
        ratio=ratio, output=outputs[k], gap=worst_gap, collected=collected, scale=scale, tolerance=ATOL + RTOL * scale,
        step=int(row) // b.n_envs, column=int(row) % b.n_envs), dict(zip(OUTPUTS, largest.tolist()))


# recurrent_gaps' outputs, scaled: each with the magnitude its tolerance takes (RTOL)
OUTPUTS = ('log probability', 'value')
SCALES = {'log probability': "largest legal |logit| of its row and the actor's largest encoder and LSTM gate terms",
          'value': "value head's terms"}


@contextmanager
def computed_from(policy):
    """What each call of `policy` computes its outputs from, recorded as it runs (recurrent_gaps, scaled): 'logits', the
    masked logits before the distribution normalizes them (baseline/policy.py _masked_logits), 'latent', the value
    head's input, and 'encoder' and 'gates', each row's largest term of the actor's first encoder layer and of its
    LSTM's gates (actor_terms)."""
    seen, masked = {}, policy._masked_logits

    def record(latent):
        seen['logits'] = masked(latent)
        return seen['logits']
    policy._masked_logits = record
    hooks = [policy.value_net.register_forward_hook(lambda head, args, output: seen.__setitem__('latent', args[0]))]
    first = policy.pi_features_extractor.encoder[0]
    hooks.append(first.register_forward_hook(lambda layer, args, output: seen.__setitem__(
        'encoder', actor_terms(args[0], first.weight, first.bias))))
    lstm = policy.lstm_actor
    hooks.append(lstm.register_forward_hook(lambda layer, args, output: seen.__setitem__('gates', actor_terms(
        args[0][0], lstm.weight_ih_l0, lstm.bias_ih_l0.abs() + lstm.bias_hh_l0.abs(), args[1][0][0], lstm.weight_hh_l0))))
    try:
        yield seen
    finally:
        for hook in hooks:
            hook.remove()
        del policy._masked_logits


def actor_terms(x, weight, bias, h=None, recurrent=None):
    """Each row's largest term magnitude of a layer, |W| |x| + |b| (with an LSTM's |W_hh| |h| too: x its first step's
    input, h its state then), over the layer's outputs: what that layer's float32 rounding is a few units of."""
    terms = x.abs() @ weight.abs().T + bias.abs()
    if h is not None:
        terms = terms + h.abs() @ recurrent.abs().T
    return terms.amax(-1)


def value_scale(head, latent):
    """The magnitude each row's value is summed from (RTOL): its head's terms |W| |h| + |b|, for a PopArt head
    (baseline/policy.py, std * (W h + b) + mean) times |std| plus |mean|."""
    terms = (latent.abs() @ head.weight.abs().T + head.bias.abs()).flatten()
    if hasattr(head, 'std'):
        terms = terms * head.std.abs() + head.mean.abs()
    return terms


def observation_problems(b):
    """The stored observations' and actions' checks of a rollout buffer: finite inputs, ids in the name and option
    tables, a 0/1 mask with a legal action in every row, and every sampled action legal. Returns (problems, illegal)."""
    import torch
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
    return problems, illegal


def life_pay(paid, rewards, ended, episode_return):
    """Each world's pay in its life so far, summed in float64 from the float32 rewards and carried across rollouts, against
    the batch's episode_return at each life's end. The batch sums a life's pay in float64 and stores the sum as float32,
    and each reward is the float32 of its pay; rewards are never negative, so each of the two roundings is at most 2**-24
    of the return and a life's tolerance is ATOL + 2**-23 * |episode_return|. paid: [worlds] float64; rewards, ended,
    episode_return: [steps, worlds]. Returns (paid at the end, the largest distance, the largest distance less its
    tolerance: above zero, or NaN, when a life's rewards miss its return)."""
    import torch
    worst = torch.zeros((), dtype=torch.float64, device=paid.device)
    excess = torch.full((), -torch.inf, dtype=torch.float64, device=paid.device)
    for step in range(len(rewards)):
        paid = paid + rewards[step].double()
        end, target = ended[step], episode_return[step].double()
        distance = (paid - target).abs()
        worst = torch.maximum(worst, torch.where(end, distance, 0).max())
        excess = torch.maximum(excess, torch.where(end, distance - (ATOL + 2 ** -23 * target.abs()), -torch.inf).max())
        paid = torch.where(end, torch.zeros_like(paid), paid)
    return paid, float(worst), float(excess)


def validate_free_play(model, carried):
    """--validate on a free-play rollout (before its update): validate_rollout's checks without goals or a task clock
    (a life's length is drawn and hidden), with goal-free play's own. The observations and actions (observation_problems);
    no unsupported step; episode starts equal to the previous steps' ends; only running, horizon and death ends; at least
    one game tick a supported step; paid level gains finite and never negative (a fractional level only grows, and a
    death takes nothing back); bootstrap values finite and only at the cut lives; each life's rewards equal to its
    episode_return, the batch's audited level gain of that life (life_pay, in float64), which is less than
    one level a paid skill from its integer gain over the paid skills (outcomes; S, or every stat for free-play-v3: a
    life's paid fractional gain and that integer gain differ by less than one level a skill); the stored recurrent states,
    each row's outputs within ATOL + RTOL x the magnitude they are computed from, a finite one (recurrent_gaps, scaled;
    the counts keep the worst row, recurrent_worst, and each output's largest scale, recurrent_max_scale), and the
    episode starts (recurrent_gaps). `carried` holds each world's pay so far in
    its life (float64) and the previous
    rollout's last ends. A real-engine rollout (sim/real.py: buffer.segments, each column one agent's segment of
    decisions, which no column of another rollout continues) checks each segment by itself in place of what spans
    rollouts: its first episode start, its rewards against its own fractional level gain (segment_pay), its decision ticks
    against its own ticks, and its recurrent outputs with the weights that collected each row (a row of an older version
    with that version's weights, buffer.version_weights loaded into buffer.spare_policy; with the collection's calls on a
    copy, buffer.collection_policy, every version so, the current one with the model's own weights, so a copy that missed
    its hand-over fails); each life's pay is checked against the engine's ledger by the batch. free-play-v2 (the model's death_price D): each death row's reward is its
    level gain less D, so D is added back at the rows whose terminal is a death before the checks of the level gains (a
    price missing at a death, or one at another row, fails them), and the counts hold the levels and the prices apart.
    free-play-v15: each row's inner pay (buffer.segments 'inner', which the batch recounted before this check:
    sim/real.py _recount) is taken out of its reward before the checks of the level gains, its float32 rounding in their
    tolerance.
    Returns the counts; raises CheckFailed on the first violation."""
    import torch
    b = model.rollout_buffer
    segments = getattr(b, 'segments', None)
    price = getattr(model, 'death_price', None) or 0.0
    inner = None if segments is None else segments.get('inner')  # free-play-v15: each row's inner pay (sim/real.py)
    problems, illegal = observation_problems(b)
    infos, rewards, cut_values = b.infos, b.rewards, b.cut_values
    terminal = infos['terminal']
    if bool(infos['unsupported'].any()):
        problems.append('an unsupported world-step reached validation')
    first = carried['done'] if segments is None else segments['first_start']
    expected = torch.cat([first[None], (terminal[:-1] != TERMINAL_RUNNING).float()])
    if not torch.equal(b.episode_starts, expected):
        problems.append('episode starts differ from the previous steps\' terminal flags')
    cut, death = terminal == TERMINAL_HORIZON, terminal == TERMINAL_DEATH
    ended = terminal != TERMINAL_RUNNING
    if not bool((cut | death | ~ended).all()):
        problems.append('a life ended for a reason other than its length or a death')
    if not bool((infos['decision_ticks'][~infos['unsupported']] >= 1).all()):
        problems.append('a supported step took no game tick or went back in game time')
    levels = rewards.double() + price * death.double() if price else rewards  # free-play-v2: the price added back
    if inner is not None:  # free-play-v15: the inner pay taken out, so the rows' levels face the checks below
        levels = levels.double() - inner
    if not bool(torch.isfinite(rewards).all() & (levels >= 0.0).all()):
        problems.append('a paid level gain is negative or not finite' + (' with the death price added back' if price else ''))
    if not bool(torch.isfinite(cut_values).all()) or bool((cut_values[~cut] != 0).any()):
        problems.append('a bootstrap value outside a cut life, or one not finite')
    if segments is None:
        paid, worst_pay, excess = life_pay(carried['paid'], rewards, ended, infos['episode_return'])
        if not excess <= 0:
            problems.append(f'the rewards of a life sum to its episode_return only within {worst_pay:.3g}, beyond its '
                            'tolerance')
    else:
        priced = price * death.double().sum(0) if price else None
        if inner is not None:  # its float32 rows of levels and inner pay round within 2**-23 of the inner pay too
            priced = inner.sum(0) if priced is None else priced + inner.sum(0)
        worst_pay, excess = segment_pay(levels, segments['gain'], priced)
        if not excess <= 0:
            problems.append(f'the rewards of a segment sum to its fractional level gain only within {worst_pay:.3g}, '
                            'beyond its tolerance')
        if not torch.equal(infos['decision_ticks'].long().sum(0), segments['ticks']):
            problems.append('the decision ticks of a segment do not sum to its ticks')
    gain = (infos['episode_return'].double() - infos['outcomes'].double()).abs()
    n_paid = len(getattr(model, 'paid_skills', None) or FREE_PLAY['skills'])  # free-play-v3 pays every stat
    if not bool((gain[ended] < n_paid).all()):
        problems.append(f'a life\'s paid gain is {n_paid} or more levels from its integer gain over the paid skills')
    if segments is None:
        gap, reset_gap, worst, scales = recurrent_gaps(model, scaled=True)
    else:
        versions, gaps = segments['version'], []
        copy = getattr(b, 'collection_policy', model.policy) is not model.policy  # the calls ran on a copy (collect_device)
        for version in versions.unique().tolist():  # each row with the model's weights that collected it
            policy = None
            if version != segments['current'] or copy:  # on the collection's device: the spare policy
                policy = b.spare_policy
                policy.load_state_dict(model.policy.state_dict() if version == segments['current'] else
                                       b.version_weights[version])
                policy.set_training_mode(False)
            found = recurrent_gaps(model, policy=policy, keep=versions == version, scaled=True)
            gaps.append((*found[:2], dict(found[2], version=version, age=segments['current'] - version), found[3]))
        gap, reset_gap = (float(torch.tensor(values).max()) for values in list(zip(*gaps))[:2])  # a NaN stays
        worst = max((found[2] for found in gaps), key=lambda row: row['ratio'])
        scales = {output: float(torch.tensor([found[3][output] for found in gaps]).max()) for output in OUTPUTS}
    if not worst['ratio'] <= 1:  # a NaN gap's ratio is infinite
        problems.append(f'stored LSTM states reproduce the collected outputs only within {gap:.3g}, beyond the tolerance: '
                        + worst_row(worst))
    if not all(math.isfinite(scale) for scale in scales.values()):  # an infinite scale would take any gap
        problems.append(f'a recurrent output\'s scale is not finite (the largest of each output: {scales})')
    if not reset_gap <= ATOL:
        problems.append(f'an episode start depends on the stored state ({reset_gap:.3g})')
    if segments is None:
        carried.update(paid=paid, done=(terminal[-1] != TERMINAL_RUNNING).float())
    counts = dict(decisions=int(rewards.numel()), illegal_actions=illegal, lives_ended=int(ended.sum()),
                  cut_lives=int(cut.sum()), deaths=int(death.sum()), episode_starts=int(b.episode_starts.sum()),
                  paid_sum=float(levels.sum()), native_ticks=int(infos['decision_ticks'].sum()),
                  cut_value_sum=float(cut_values.sum()), life_pay_max_error=worst_pay, recurrent_max_gap=gap,
                  recurrent_worst=worst, recurrent_max_scale=scales, reset_max_gap=reset_gap, problems=problems)
    if segments is not None:
        counts.update(lagged_rows=int((segments['version'] < segments['current']).sum()))
    if price:
        counts.update(prices=float((segments['gain'] - rewards.double().sum(0)).sum()) if segments is not None else
                      float(price * death.sum()), rewards_sum=float(rewards.sum()))  # paid_sum: the level gain alone
    if problems:
        raise CheckFailed('; '.join(problems))
    return counts


def worst_row(row):
    """validate_free_play's failure text for recurrent_gaps' worst row (scaled): which output, where, its collected
    value, gap, tolerance and scale, and for a real-engine row its weights version and age."""
    text = (f"the {row['output']} at step {row['step']}, column {row['column']} ({row['collected']:.7g} collected) differs "
            f"by {row['gap']:.3g}, {row['ratio']:.3g} times its tolerance {row['tolerance']:.3g} (ATOL + RTOL x "
            f"{row['scale']:.4g}, the {SCALES[row['output']]})")
    if 'version' in row:
        text += f"; the row's weights are version {row['version']}, {row['age']} updates old"
    return text


def segment_pay(rewards, gain, priced=None):
    """Each column's float32 rewards summed in float64 against its own fractional level gain `gain` [n_envs] (float64,
    a real-engine segment's, sim/real.py): rewards are never negative and each is the float32 of its pay, so a column's
    tolerance is ATOL + 2**-23 * |gain|. priced (free-play-v2): each column's death prices, already added back to its
    rewards, whose float32 rows of pay less the price round within 2**-23 of the price too. Returns (the largest
    distance, the largest distance less its tolerance)."""
    distance = (rewards.double().sum(0) - gain).abs()
    tolerance = ATOL + 2 ** -23 * (gain.abs() if priced is None else gain.abs() + priced)
    return float(distance.max()), float((distance - tolerance).max())


def validate_rollout(model, carried, horizons, targets=None, goals=GOALS):
    """The checks on the rollout just collected (before its update). `carried` holds what spans rollouts:
    each world's reward paid so far in its episode and the done flags that end the previous rollout. horizons and
    targets: per goal (task_limits; a single horizon applies to every goal, with goal-activity-v2's targets). goals:
    the declared goal set; every step of world slot b must ask for slot_goal(goals, b), in its info and its goal_vec.
    Returns this rollout's counts; raises CheckFailed on the first violation."""
    import torch
    b = model.rollout_buffer
    actions = b.actions[..., 0]
    problems, illegal = observation_problems(b)
    obs = b.observations
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
    rollouts. Not a check: validate_rollout's run first. Free play has no goal: one group, FREE_PLAY_TASK, holds every
    world (its info's goal is never read), and its ended lives' facts are life_facts."""
    import torch
    from act_contract import A_ACT0, A_HELD0, N_ACT, name_id
    from feedback_contract import FAMILIES as ACTION_FAMILIES
    from obs import N_ACTIONS
    from sim.agent import family
    from wide_contract import page_entry
    b = model.rollout_buffer
    actions, obs, infos = b.actions[..., 0], b.observations, b.infos
    if model.free_play:
        labels, goal = (FREE_PLAY_TASK,), torch.zeros_like(infos['terminal'], dtype=torch.long)
    else:
        labels, goal = GOALS, infos['goal'].long()
    G = len(labels)
    device = actions.device

    def by_goal(values, mask=None):  # [steps, worlds] numbers -> per-goal sums
        values = values if mask is None else torch.where(mask, values, torch.zeros_like(values))
        sums = torch.zeros(G, dtype=values.dtype, device=device).index_add_(0, goal.reshape(-1), values.reshape(-1))
        return dict(zip(labels, sums.tolist()))
    names = (*ACTION_FAMILIES, 'other')

    def family_or_other(a):  # the reserved slots (92-95) have no family; act-v4's rows 96..127 are their family's
        try:
            return family(page_entry(a)[0] if a >= N_ACTIONS else a)
        except (ValueError, TypeError):
            return 'other'
    rows = b.observations['mask'].shape[-1]
    table = torch.as_tensor([names.index(family_or_other(a)) for a in range(rows)], device=device)
    kind = table[actions]
    families = torch.zeros(G * len(names), dtype=torch.int64, device=device).index_add_(
        0, (goal * len(names) + kind).reshape(-1), torch.ones_like(kind).reshape(-1)).view(G, len(names)).tolist()
    # each row's pack option entry (rows 29..36, and act-v4's 96..103 for entries 8..15), -1 for any other row
    held_entry = torch.as_tensor([e[1] if (e := page_entry(a)) and e[0] == A_HELD0 else -1 for a in range(rows)],
                                 device=device)[actions]
    is_act, is_held = (actions >= A_ACT0) & (actions < A_ACT0 + N_ACT), held_entry >= 0
    act_op = obs['act_op_ids'].gather(-1, (actions - A_ACT0).clamp(0, N_ACT - 1)[..., None])[..., 0]
    held_op = obs['held_op_ids'].gather(-1, held_entry.clamp(0)[..., None])[..., 0]
    requested = {name: by_goal(((is_act & (act_op == name_id(name))) | (is_held & (held_op == name_id(name)))).long())
                 for name in REQUESTS}
    ended = infos['terminal'] != TERMINAL_RUNNING
    result = dict(transitions=by_goal(torch.ones_like(goal)), episodes_ended=by_goal(ended.long()),
                  horizon_ends=by_goal((infos['terminal'] == TERMINAL_HORIZON).long()),
                  deaths=by_goal((infos['terminal'] == TERMINAL_DEATH).long()),
                  certified_outcomes=by_goal(infos['outcomes'].long(), ended),
                  episode_return_sum=by_goal(infos['episode_return'], ended), reward_sum=by_goal(b.rewards),
                  requested_families={labels[g]: dict(zip(names, row)) for g, row in enumerate(families)},
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
        if model.free_play:  # a life has no group, success, search or supply (sim/vec.py _step_free): its fights' food
            # and healing and its distinct cells, summed over the lives that ended
            result['life_facts'] = {k: by_goal(fact(k), ended) for k in (*COMBAT_FACTS, 'cells')}
            return result
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
    spaced over the batch (so over the CPU workers' slices too); with no goals (free play), count slots so spaced."""
    if not goals:
        if not 0 < count <= worlds:
            raise ValueError(f'{count} segment worlds of {worlds}')
        return [k * worlds // count for k in range(count)]
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


def check_saved_versions(path, policy=None, contract=None, observation=None):
    """baseline/policy.py check_names on a checkpoint as it was saved, before any of its tensors is loaded: the name
    table and the observation, action, mask and feedback versions it records. A checkpoint of the same tensor shapes
    saved under another contract, or with none recorded, is refused, never taken under this run's versions.
    policy: the run's own, whose observation version and value normalization the checkpoint must hold (default obs-v5
    and none, the goal tasks'). contract: the contract the checkpoint must record (baseline/policy.py run_contract:
    the run's own, its policy's, by default; --weights gives the contract of the checkpoint's own version). observation:
    the observation version it must record (the policy's by default; --weights gives its own version's)."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import CONTRACT, check_names, check_value_norm
    from obs import OBSERVATION
    saved = RecurrentPPO.load(path, device='cpu').policy
    contract = contract or (CONTRACT if policy is None else policy.contract)
    observation = observation or (OBSERVATION if policy is None else policy.observation)
    check_names(saved, observation=observation, contract=contract)
    check_value_norm(saved, None if policy is None else policy.value_norm)


# the goal-free settings a free-play checkpoint saves (sim/learn.py DeviceRecurrentPPO), which --init and --weights record
GOAL_FREE_SAVED = ('free_play', 'half_life_ticks')
# a named free-play version's own (sim/workers.py FREE_PLAY_V2 to V7), saved with every checkpoint: its name, and
# free-play-v2's death price, free-play-v3's and later versions' paid stats and kit, or free-play-v15's and its inner
# pay; a free-play-v1 checkpoint holds none of these. A free-play-v15 checkpoint also saves its rare-XP window
# (rare_xp_window), which a leg goes on from.
VERSION_SAVED = ('free_play_task', 'death_price', 'paid_skills', 'kit_extra', 'inner_pay')
WEIGHTS_NOTES = {
    FREE_PLAY_V2_TASK: 'the value statistics (PopArt) in these weights were learned without the death price (free-play-v1)',
    FREE_PLAY_V3_TASK: 'the value statistics (PopArt) in these weights were learned with free-play-v1\'s pay over seven '
                       'skills',
    FREE_PLAY_V10_TASK: 'these weights were learned on item-v2\'s page of slots (free-play-v3): under item-v3 a page position '
                        'is the k-th kind of item and its count input the kind\'s total, a meaning the policy learns again',
    FREE_PLAY_V11_TASK: 'these weights were learned on a page of 8 entries and act-v3\'s 96 rows (free-play-v3: item-v2\'s '
                        'slots; free-play-v10: item-v3\'s kinds) and crossed by the tested conversion (widen_state): the '
                        'page\'s entries 8 to 15 and rows 96 to 127 start at zero weight beside the kept scorers, and '
                        'mask-v6 allows drops that mask-v5 did not, so the play can differ from the source\'s',
    FREE_PLAY_V12_TASK: 'these weights were learned with every life started at full hitpoints (free-play-v10): under v12 '
                        'most lives start 1 to 3 hitpoints under full',
    FREE_PLAY_V14_TASK: 'these weights were learned with every life started as drawn (free-play-v12): under v14 every '
                        'agent of odd index starts every life with the old-life profile (old-life-v1), levels outside '
                        'the range that the policy was trained on',
    FREE_PLAY_V15_TASK: 'these weights were learned with every life started as drawn and without the rare-XP pay '
                        '(free-play-v12): under v15 every agent of odd index starts every life with the old-life profile, '
                        'and the value statistics (PopArt) were learned without the rare-XP pay, whose window starts '
                        'empty'}
# a note of its own for a crossing whose run's version has another source (WEIGHTS_NOTES is by the run's version)
PAIR_NOTES = {(FREE_PLAY_V15_TASK, FREE_PLAY_V14_TASK): 'the value statistics (PopArt) in these weights were learned with '
              'the rare-XP pay (free-play-v15), which this run does not pay'}
# the versions a checkpoint starts by --weights alone (load_weights): a free-play-v1 one free-play-v2 and v3, a
# free-play-v3 one free-play-v10 and v11, a free-play-v10 one v11 and v12, a free-play-v12 one v14 and v15, a
# free-play-v15 one v14 (the pay off)
WEIGHTS_ACROSS = {((FREE_PLAY_TASK, 0.0), (task, death_price(task))) for task in (FREE_PLAY_V2_TASK, FREE_PLAY_V3_TASK)} | \
    {((FREE_PLAY_V3_TASK, 0.0), (FREE_PLAY_V10_TASK, 0.0)), ((FREE_PLAY_V3_TASK, 0.0), (FREE_PLAY_V11_TASK, 0.0)),
     ((FREE_PLAY_V10_TASK, 0.0), (FREE_PLAY_V11_TASK, 0.0)), ((FREE_PLAY_V10_TASK, 0.0), (FREE_PLAY_V12_TASK, 0.0)),
     ((FREE_PLAY_V12_TASK, 0.0), (FREE_PLAY_V14_TASK, 0.0)), ((FREE_PLAY_V12_TASK, 0.0), (FREE_PLAY_V15_TASK, 0.0)),
     ((FREE_PLAY_V15_TASK, 0.0), (FREE_PLAY_V14_TASK, 0.0))}


def saved_version(data):
    """(free-play version, death price) of a model or of a checkpoint's saved data (a dict): (free_play_task, its
    death_price or 0.0) for a named version (free-play-v2, free-play-v3), (free-play-v1, 0.0) for a v1 one, (None, 0.0) for
    a goal task's."""
    get = data.get if isinstance(data, dict) else lambda name: getattr(data, name, None)
    return get('free_play_task') or (FREE_PLAY_TASK if get('free_play') else None), get('death_price') or 0.0


def device_sides(data, model):
    """{'devices': the checkpoint's and the run's real_devices} when either side records them (a real-engine run on a
    GPU, or with a collection copy: sim.train real_devices), for --weights, --init and a leg; {} between processor runs,
    whose records stay as before."""
    sides = data.get('real_devices'), getattr(model, 'real_devices', None)
    return {} if sides == (None, None) else dict(devices=dict(checkpoint=sides[0] or 'processor (none recorded)',
                                                              run=sides[1] or 'processor'))


def refuse_without_names(saved, path):
    """Refuse a checkpoint whose name tables hold no vectors (a public copy, scripts/export_policy.py): a run would train
    with them empty. Its tensors `saved` are the zip's policy.pth."""
    tables = [k for k in saved if k.endswith('embedding.table')]
    if tables and not any(bool(saved[k].any()) for k in tables):
        raise ValueError(f'{path} holds no name vectors (a public copy: scripts/export_policy.py); a run starts from a '
                         'checkpoint that holds its own')


def load_init(model, path, device):
    """Replace a fresh model's policy and optimizer state with a checkpoint's (SB3 set_parameters, exact match);
    return its sha256, the optimizer's step count and whether every policy tensor now equals the checkpoint's, and for
    a free-play checkpoint its goal-free settings, which this run's own replace (a half-life ablation starts from the same
    initialization)."""
    import io
    import zipfile
    import torch
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data').decode())
    refuse_removed(saved_version(data)[0], f'the checkpoint {path}')
    check_saved_versions(path, model.policy)
    if saved_version(data) != saved_version(model):
        raise ValueError(f'{path} is a checkpoint of {saved_version(data)} (version, death price): its optimizer state '
                         f'never starts a run of {saved_version(model)} (a free-play-v1 checkpoint starts a run of another '
                         'version by --weights alone)')
    with zipfile.ZipFile(path) as archive:
        saved = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    refuse_without_names(saved, path)
    model.set_parameters(str(path), exact_match=True, device=device)
    mine = model.policy.state_dict()
    equal = set(saved) == set(mine) and all(torch.equal(saved[k], mine[k].cpu()) for k in saved)
    steps = [int(v['step']) for v in model.policy.optimizer.state_dict()['state'].values() if 'step' in v]
    if not equal:
        raise CheckFailed(f'the initialization {path} did not load exactly')
    goal_free = {k: data[k] for k in GOAL_FREE_SAVED if k in data}
    return dict(path=str(path), sha256=sha256(path), policy_equal=equal, tensors=len(saved),
                optimizer_steps=max(steps) if steps else 0,
                **({'saved_settings_not_imported': goal_free} if goal_free else {}), **device_sides(data, model))


def load_weights(model, path, device):
    """Load only a checkpoint's policy tensors (actor, critic and the frozen name embeddings) into a fresh model built
    with this run's settings: its optimizer, schedules, counters, buffer and recurrent state stay new, so nothing of the
    checkpoint's training settings (n_steps, batching, learning rate) reaches the run. Exact names and shapes, and every
    tensor equal afterwards; returns its sha256 and the saved settings it did not import. A free-play-v1 checkpoint may
    start a free-play-v2 or free-play-v3 run, a free-play-v3 one a free-play-v10 or v11 run, a free-play-v10 one a v11 or
    v12 run, a free-play-v12 one a v14 or v15 run and a free-play-v15 one a v14 run (WEIGHTS_ACROSS), recorded with both
    versions and the version's note (WEIGHTS_NOTES, PAIR_NOTES); no other pair of versions loads, and a checkpoint of a
    removed study arm is refused by name (sim/workers.py refuse_removed). A checkpoint of act-v3's
    96 rows reaches a run of act-v4's 128 (free-play-v11) through baseline/policy.py widen_state alone, recorded
    (`conversion`): its tensors are checked by its own versions, widened, then loaded and compared as the run's."""
    import io
    import zipfile
    import torch
    from baseline.policy import action_rows, run_contract, widen_state
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data').decode())
    (task, price), (mine, my_price) = saved_version(data), saved_version(model)
    refuse_removed(task, f'the checkpoint {path}')
    # the checkpoint holds its own version's contract (its item contract included), whatever the run's; and, crossing
    # from act-v3's 96 rows to act-v4's 128 (free-play-v11), its own observation version, obs-v6, not the run's obs-v7
    own = None if task is None else run_contract(item_contract(task))
    widened = own is not None and action_rows(own) != action_rows(model.policy.contract)
    check_saved_versions(path, model.policy, contract=model.policy.contract if task is None else own,
                         observation=observation_table(item_contract(task))[0] if widened else None)
    with zipfile.ZipFile(path) as archive:
        saved = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    refuse_without_names(saved, path)
    if (task, price) != (mine, my_price) and ((task, price), (mine, my_price)) not in WEIGHTS_ACROSS:
        raise ValueError(f'{path} is a checkpoint of {(task, price)} (version, death price): its weights never start a run '
                         f'of {(mine, my_price)} ({FREE_PLAY_TASK} weights start {FREE_PLAY_V2_TASK} and {FREE_PLAY_V3_TASK} '
                         f'runs, {FREE_PLAY_V3_TASK} weights {FREE_PLAY_V10_TASK} and {FREE_PLAY_V11_TASK} runs, '
                         f'{FREE_PLAY_V10_TASK} weights {FREE_PLAY_V11_TASK} and {FREE_PLAY_V12_TASK} runs, '
                         f'{FREE_PLAY_V12_TASK} weights {FREE_PLAY_V14_TASK} and {FREE_PLAY_V15_TASK} runs, '
                         f'{FREE_PLAY_V15_TASK} weights {FREE_PLAY_V14_TASK} runs; no other version crosses)')
    versions = {} if not {task, mine} - {FREE_PLAY_TASK, None} else dict(
        task_versions=dict(checkpoint=task, run=mine), death_prices=dict(checkpoint=price, run=my_price),
        **({'note': PAIR_NOTES.get((task, mine), WEIGHTS_NOTES[mine])} if task != mine else {}))
    items = (item_contract(task) if task is not None else ITEM_V2, item_contract(mine) if mine is not None else ITEM_V2)
    if items[0] != items[1]:  # the declared crossing of the item contracts (free-play-v3 to v10, v3 or v10 to v11)
        versions['item_contracts'] = dict(checkpoint=items[0], run=items[1])
    if widened:  # free-play-v3 or v10 to v11
        saved = widen_state(saved, model.policy)
        versions['conversion'] = dict(method='baseline/policy.py widen_state: zero first-layer weights for the page\'s '
                                      'entries 8 to 15 and zero action-head rows 96 to 127; every other tensor copied',
                                      checkpoint=dict(observation=observation_table(items[0])[0], contract=own),
                                      run=dict(observation=model.policy.observation, contract=model.policy.contract))
    versions.update(device_sides(data, model))
    model.policy.load_state_dict(saved, strict=True)
    mine = model.policy.state_dict()
    if set(saved) != set(mine) or not all(torch.equal(saved[k], mine[k].cpu()) for k in saved):
        raise CheckFailed(f'the weights of {path} did not load exactly')
    ignored = {k: data.get(k) for k in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                                        'max_grad_norm', 'num_timesteps', '_n_updates', *GOAL_FREE_SAVED, *VERSION_SAVED,
                                        'rare_xp_window')
               if k in data}
    return dict(path=str(path), sha256=sha256(path), tensors=len(saved), policy_equal=True, optimizer='fresh',
                saved_settings_not_imported=ignored, **versions)


# the learner settings a leg shares with the checkpoint it continues, each as saved (SB3's data) and as built
LEG_SAME = ('seed', 'n_envs', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
            'max_grad_norm', 'learning_rate', 'normalize_advantage', 'target_kl', 'clip_range_vf', *GOAL_FREE_SAVED,
            *VERSION_SAVED, 'self_imitation_setting')  # a self-imitation run's checkpoint holds it: no leg continues it


# A real-engine run's own record of its leg, saved with every checkpoint (model.real_leg) and checked by the leg that
# continues it (leg_parent, check_leg_world): its number, the first leg's output, the key of its draws (the
# environment seed), its declared total of decisions, --validate, --checkpoint-every and the batch's world (its settings
# in effect less each launch's own, LEG_LAUNCH). A leg repeats LEG_RUN and the world, and its number is its parent's
# plus one.
LEG_RUN = ('env_seed', 'decisions', 'validate', 'checkpoint_every')
LEG_LAUNCH = ('leg', 'stream_offset', 'world_ids', 'workers', 'policy_threads', 'life_facts', 'hurt_facts')
# what a leg's summary counts over the whole run (every leg so far); every other count, rate and record is the leg's own
LEG_SCOPES = dict(whole_run=['total_timesteps', 'sb3_n_updates', 'leg.decisions_at_start', 'leg.updates_at_start'],
                  this_leg='every other count, rate and record: ppo_updates, gradient_steps, native_ticks, episodes, '
                           'lives, real, updates, checks, checkpoint_digests (named by the whole run\'s counts), the '
                           'speeds, timing and memory; leg.decisions_this_leg beside total_timesteps')
# what a leg from a checkpoint without its leg record (--unrecorded-parent) declares and cannot check
LEG_UNVERIFIED = ('env_seed', 'world', 'validate', 'checkpoint_every')


def leg_run_folder(path):
    """The run folder of a checkpoint: its own folder, or the run's for one in checkpoints/."""
    folder = Path(path).parent
    return folder.parent if folder.name == 'checkpoints' else folder


def leg_world(settings):
    """The world a real-engine batch plays (its settings in effect, sim/real.py), less each launch's own (LEG_LAUNCH)."""
    return {k: v for k, v in settings.items() if k not in LEG_LAUNCH}


def leg_parent(path, leg, unrecorded):
    """The checkpoint leg `leg` continues, read before the leg starts anything: (its data, its leg record or None, its
    sha256). It must hold every update it counts (saved after the update: an early real-engine checkpoint, or
    a simulator one, holds a rollout that no update trained on, and is refused). A checkpoint with its leg record
    (real_leg) is continued by the leg after its own; one without it, a real-engine first leg's before the record was
    saved, only with `unrecorded` (--unrecorded-parent) and as leg 2, which then declares LEG_UNVERIFIED unchecked."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data').decode())
    refuse_removed(saved_version(data)[0], f'the checkpoint {path}')
    count, steps, epochs = data.get('num_timesteps') or 0, data.get('_n_updates') or 0, data.get('n_epochs') or 1
    rollout = (data.get('n_envs') or 0) * (data.get('n_steps') or 0)
    if count <= 0 or not rollout or count % rollout or steps % epochs or count != steps // epochs * rollout:
        raise ValueError(f'{path} counts {count} decisions and {steps // epochs} updates of {rollout}: it was saved before '
                         'its last update (an early real-engine checkpoint, or a simulator one); a leg starts '
                         'only from a checkpoint that holds every update it counts')
    record = data.get('real_leg')
    if record is None and not unrecorded:
        raise ValueError(f'{path} holds no leg record (a checkpoint before the record was saved, or not a real-engine '
                         'one): only a real-engine first leg\'s continues, as leg 2 with --unrecorded-parent, which '
                         f'declares {", ".join(LEG_UNVERIFIED)} unchecked')
    if record is not None and unrecorded:
        raise ValueError(f'{path} holds its leg record (leg {record["number"]}): no --unrecorded-parent')
    before = 1 if record is None else record['number']
    if leg != before + 1:
        raise ValueError(f'{path} is a checkpoint of leg {before}: the leg that continues it is --leg {before + 1}, not '
                         f'--leg {leg}')
    return data, record, sha256(path)


def check_leg_world(path, parent, data, model, environment):
    """A leg's checks once its environment and model exist (leg_parent made the others before it started anything):
    the environment draws from the run's key, and from a recorded parent the same world (leg_world); the device form
    (the update's and the collection's devices and the TF32 settings, sim.train real_devices) equal to the
    checkpoint's, which holds none for a processor run with no collection copy. The GPU's name is recorded, not
    compared."""
    key, mine = environment.batch.key, model.real_leg['env_seed']
    if key != mine:
        raise ValueError(f'the environment draws from key {key}, the leg\'s run from environment seed {mine}')
    if parent is not None and parent['world'] != model.real_leg['world']:
        differ = {k: [parent['world'].get(k), model.real_leg['world'].get(k)]
                  for k in set(parent['world']) | set(model.real_leg['world'])
                  if parent['world'].get(k) != model.real_leg['world'].get(k)}
        raise ValueError(f'leg {model.real_leg["number"]} would continue {path} in another world (the parent\'s, this '
                         f'leg\'s): {differ}')
    form = lambda devices: None if devices is None else {k: devices.get(k) for k in ('update', 'collection', 'cudnn_tf32',
                                                                                    'matmul_tf32')}
    theirs, ours = form(data.get('real_devices')), form(getattr(model, 'real_devices', None))
    if theirs != ours:
        raise ValueError(f'leg {model.real_leg["number"]} would continue {path} on other devices (the parent\'s, this '
                         f'leg\'s): {[theirs, ours]}')


def load_leg(model, path, device, leg, rollout):
    """Leg `leg`'s start (module docstring): the whole learner state of `path`, a checkpoint of an earlier leg, into a
    fresh model built with the same learner settings (LEG_SAME, each equal to the checkpoint's): the policy with its value
    normalization's statistics (PopArt's mean, std and moments are buffers of the policy) and Adam's state (load_init,
    exact), then the decision and update counts (num_timesteps, _n_updates, which the schedules read; the count a multiple
    of `rollout`, the updates whole epochs). Returns summary 'leg': the parent run (the checkpoint's run folder), the
    checkpoint's sha256 and tensor digest, the counts, and the decisions it counts that no update trained on (a
    checkpoint saved between a collection and its update, as every real-engine run before legs saved them)."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data').decode())
    differ = {k: [data.get(k), getattr(model, k, None)] for k in LEG_SAME if data.get(k) != getattr(model, k, None)}
    if differ:
        raise ValueError(f'leg {leg} would continue {path} under other learner settings (saved, this run\'s): {differ}')
    count, steps = data['num_timesteps'], data['_n_updates']
    if count <= 0 or count % rollout or steps % model.n_epochs:
        raise ValueError(f'{path} counts {count} decisions and {steps} epochs: no whole rollouts of {rollout} and updates')
    record = load_init(model, path, device)
    record.pop('saved_settings_not_imported', None)  # LEG_SAME holds them equal
    model.num_timesteps, model._n_updates = count, steps
    updates = steps // model.n_epochs
    return dict(record, number=leg, parent=str(leg_run_folder(path)),
                policy_sha256=policy_digest(model.policy), decisions_at_start=count, updates_at_start=updates,
                collected_not_trained=count - updates * rollout)


def real_devices(model, batch, cuda_real, collect_device):
    """A real-engine run's devices (summary 'devices', and every checkpoint's real_devices): the update's (the model's),
    the collection's policy calls' (the model's own policy, or a copy that takes its weights at the start of each
    collection, sim/real.py HANDOVER), the GPU's name and the TF32 settings; None for a processor run with no collection
    copy, whose records stay as before. A CUDA run's batch must hold the declared copy (collect_device cpu) or none."""
    import torch
    from sim.real import HANDOVER
    copy = batch.collect_setting
    if cuda_real and (copy is not None) != (collect_device == 'cpu'):
        raise ValueError(f'the environment\'s collection copy is {copy}, the run declares --collect-device {collect_device}')
    if not cuda_real and copy is None:
        return None
    cuda = model.device.type == 'cuda'
    return dict(update=str(model.device), collection=str(torch.device(copy)) if copy else str(model.device),
                handover=HANDOVER if copy else 'none: the collection calls the model\'s own policy',
                gpu=torch.cuda.get_device_name(model.device) if cuda else None,
                cudnn_tf32=torch.backends.cudnn.allow_tf32 if cuda else None,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32 if cuda else None)


def check_source_outputs(model, path, device):
    """Before the first update of a weights run: the source checkpoint, loaded as itself, gives the same action
    probabilities and values as the run's model on every stored step of the first rollout (its observations, LSTM
    states and episode starts), so the run starts from the actual saved policy. A source of act-v3's 96 rows in a run of
    act-v4's 128 (baseline/policy.py widen_state) reads its own view of each step, the page's and pack options' first 8
    entries, and both are given one mask, every row of 0..95 legal and rows 96..127 not: the conversion's equality."""
    import torch
    from sb3_contrib import RecurrentPPO
    from obs import N_INV_SEL
    from wide_contract import WIDE_SHAPES
    source = RecurrentPPO.load(path, device=device)
    b, worst = model.rollout_buffer, 0.0
    rows = source.policy.action_space.n
    narrow = rows != model.policy.action_space.n
    with torch.no_grad():
        for step in range(b.buffer_size):
            o = {k: v[step].to(b._dtypes[k]) for k, v in b.observations.items()}
            pi = (b.hidden_states_pi[step].transpose(0, 1).contiguous(), b.cell_states_pi[step].transpose(0, 1).contiguous())
            vf = (b.hidden_states_vf[step].transpose(0, 1).contiguous(), b.cell_states_vf[step].transpose(0, 1).contiguous())
            starts = b.episode_starts[step]
            for p in (model.policy, source.policy):
                p.set_training_mode(False)
            mine = theirs = o
            if narrow:
                legal = torch.ones_like(o['mask'][:, :rows])
                mine = dict(o, mask=torch.cat([legal, torch.zeros_like(o['mask'][:, rows:])], dim=1))
                theirs = dict({k: v[:, :N_INV_SEL] if k in WIDE_SHAPES else v for k, v in o.items()}, mask=legal)
            outputs = [(p.get_distribution(x, pi, starts)[0].distribution.probs[:, :rows].clone(), p.predict_values(x, vf, starts))
                       for p, x in ((model.policy, mine), (source.policy, theirs))]
            worst = max(worst, float((outputs[0][0] - outputs[1][0]).abs().max()), float((outputs[0][1] - outputs[1][1]).abs().max()))
    result = dict(path=str(path), steps=b.buffer_size, max_difference=worst,
                  **(dict(compared='rows 0 to 95 under one mask, the source on the first 8 entries (widen_state)')
                     if narrow else {}))
    if worst > ATOL:
        raise CheckFailed(f'the run\'s first rollout differs from its source checkpoint: {result}')
    return result


RECOVERY_STATE = 'state.pt'


def put_treatment_state(state):
    """A recovery state's treatments: none is left, so a state that holds one is refused (self-distillation, RND, CBP
    and PCGrad, all removed). Returns None."""
    removed = sorted(k for k, v in (state or {}).items() if v is not None)
    if removed:
        raise CheckFailed(f'the recovery state holds removed treatments {removed}')
    return None


def save_recovery(model, directory, record):
    """After a complete update: the policy, Adam, the counters and the random generators' states (no environment or
    recurrent state: a resume starts new episodes), written atomically, with record (a JSON summary) beside it."""
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(policy=model.policy.state_dict(), optimizer=model.policy.optimizer.state_dict(),
                 treatments=None,
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
    treatments = put_treatment_state(state.get('treatments'))
    model.num_timesteps, model._n_updates = state['num_timesteps'], state['n_updates']
    np.random.set_state(state['np_state'])
    torch.set_rng_state(state['torch_state'])
    if state['cuda_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_state'])
    return dict(record, path=str(directory), treatments_restored=treatments)


def check_checkpoint(model, path, device):
    """The saved zip loads as a standard RecurrentPPO with the run's observation version and value normalization and
    gives the same action probabilities and values as the live policy on the last rollout's first step (exact)."""
    import torch
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names, check_value_norm
    loaded = RecurrentPPO.load(path, device=device)
    check_names(loaded.policy, observation=model.policy.observation, contract=model.policy.contract)
    check_value_norm(loaded.policy, model.policy.value_norm)
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


def train(output, *, decisions, seed, worlds, n_steps, batch_size, n_epochs, device='cpu', env='fake',
          checkpoint_every=None, threads=None, verbose=True, pool=None, goals=None, validate=False, tables=None,
          count_syncs=False, environment=None, events=None, callbacks=(), init=None, warmup=1,
          speed=None, ppo_profile=None, replay=None, step_timing=True, workers=None,
          task='goal-activity-v2', weights=None, recovery=None, resume=None, env_seed=None, search_pool=None,
          telemetry=None, segments=None, engine='cpu', pack=None,
          half_life_ticks=None, gae_lambda=None, value_norm=None, batch=None, initial_only=False,
          leg_from=None, leg=None, collect_device=None, cudnn_tf32=None, ent_coef=None, unrecorded_parent=False):
    """One training run; returns summary.json's content. task: the four-goal task version (sim/combat.py task_spec;
    v3 needs env cpu), recorded in the summary and the recovery state: a resume refuses another. goals: the declared
    goal set (sim/spec.py goal_set; default all four), recorded in the summary and the recovery state; a resume refuses
    another. task FREE_PLAY_TASK (module docstring): no goal set and no search pool; half_life_ticks, gae_lambda, value_norm
    and batch are its settings (free_play_settings; on the CPU workers batch names FREE_PLAY_NAMED), recorded with the
    batch's settings in effect (batch_in_effect) in the summary ('free_play') and the recovery state, whose resume refuses
    others; the checkpoints of --init and --weights must hold its observation version and value normalization. leg_from,
    leg: a real-engine run's leg `leg` (2 or more) from `leg_from`, a checkpoint of an earlier leg (module docstring,
    load_leg); `environment`, when given, must be on that leg. collect_device, cudnn_tf32: a real-engine run on CUDA
    names both (module docstring, real_devices). ent_coef (the free-play tasks, --ent-coef): PPO's entropy coefficient
    in the loss, config.LEARNER's unless given (free_play_settings records it when given; changes_from_learner when it
    differs); a leg repeats its checkpoint's (LEG_SAME), --weights records the checkpoint's (saved_settings_not_imported).
    Under PopArt the stock update and the replay probe are refused (they skip its statistics update).
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
    search layouts or v5's supply layouts (sim/search.py). telemetry (default: on for v4, v5 and v11) records
    sim/learn.py Tracker's activation statistics and each update's parameter change in every update row, read-only. Per
    update, summary 'updates' also holds collect_parts (the collector's synchronized parts, sim/learn.py step_timing;
    None with step_timing=False, which leaves only the rollout's own boundary synchronizations) and log_seconds (the
    logger's dump).
    The learner-speed comparison, at fixed minibatch and optimizer settings: speed: (update index, names):
    baseline.policy MaskedRecurrentPolicy's switches (SPEED_SWITCHES) are off before that update and the named ones
    on from its PPO phase onward, so one process times both paths; without it the class defaults (both on) apply.
    Each update records the paths its PPO phase used (summary 'updates' speed_paths). ppo_profile: (update indices,
    directory): those updates' PPO phases run under torch.profiler (CPU and CUDA); its operator table and totals are
    written there. replay: (update index, variants):
    after that update, its PPO phase is replayed from the same inputs under each of REPLAY_VARIANTS (the first, stock,
    is the reference), with the largest parameter difference from it after optimizer steps 1, 2, 4, ... and the
    synchronized seconds, then the real state is put back. Profiled and replayed updates should be warm-up updates:
    their probes' cost falls in their own iteration. initial_only: build the environment and the policy, save
    initial.zip and record policy_sha256_at_start and the learner's settings, then stop with status 'initialized'
    before the first rollout (a plan's frozen start digest of each seed). summary 'tables' is
    the archive a Python reference run trains on (path, sha256, whether its pins were checked); on the C engine the
    archive that every worker held equal to its pack's recorded input (sim/workers.py _native; its pins unchecked, since
    C reads only the pack), so that a plan's tables pin a C-engine run as they pin a Python reference run; None on the
    fake environment. summary 'pack' is the C engine's pack: path, sha256, its manifest's input digests (the archive, the
    pool and the supply pool it was exported from) and the oracle it was exported at. summary 'source' is baseline/telemetry.py source(output): the commit, the changed
    and untracked files and, when the tree differs from the commit, the patch written to output/source.patch ({commit:
    None, error} outside a git checkout); 'validate' whether the run made the --validate checks. Removed treatments (RND,
    CBP, PCGrad) are no longer in the code."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from baseline.policy import NameEmbedding
    from baseline.telemetry import source
    from sim.learn import DeviceRecurrentPPO, build
    free = task in FREE_PLAY_TASKS
    refuse_real_only(task, env if env in ('real', 'fake') else engine)  # free-play-v2 and v3: the real engine only
    learner = free_play_settings(task, half_life_ticks, gae_lambda, value_norm, batch, ent_coef)
    if free and goals is not None:
        raise ValueError(f'{FREE_PLAY_TASK} has no goal: give no goal set')
    rollout, goals = worlds * n_steps, () if free else goal_set(GOALS if goals is None else goals)
    if decisions <= 0 or decisions % rollout:
        raise ValueError(f'decisions must be a positive multiple of worlds x n_steps = {rollout}')
    if (weights or resume) and init:
        raise ValueError('init restores a checkpoint\'s optimizer state: use weights or resume')
    if weights and resume:
        raise ValueError('a resume restores its own weights')
    if not free and worlds % len(goals):
        raise ValueError(f'worlds must be a multiple of {len(goals)}: goals are assigned per world slot')
    if validate and env != 'cpu' and not free:
        raise ValueError('--validate checks the simulator\'s task clock: it needs --env cpu')
    if free and telemetry:
        raise ValueError(f'the activation telemetry is per goal: {FREE_PLAY_TASK} has none')
    if free and search_pool is not None:
        raise ValueError(f'{FREE_PLAY_TASK} plays its own world: it takes no search pool')
    if env == 'real':  # sim/real.py: the real engine's worlds, collected asynchronously in segments
        if engine != 'cpu' or pack is not None:
            raise ValueError('the real engine is its own engine: no --engine or --pack')
        if segments is not None:
            raise ValueError('a real-engine rollout holds rows of several sets of weights: --segments replays one')
        if resume is not None:
            raise ValueError('a resumed real-engine run would draw the same start records, sites and lengths again: run '
                             'a leg (--leg-from, --leg)')
        if workers:
            raise ValueError('--workers counts the simulator\'s CPU workers; the real engine\'s worker processes are '
                             '--batch-setting workers=N')
        engine = 'real'
    real_only = [w for w in ((batch or {}).get(k) for k in ('world', 'starts')) if w in REAL_WORLDS]
    if env != 'real' and free and real_only:  # village-world-v3: a world of the real engine alone
        raise ValueError(f'{real_only[0]} is a world of the real engine only (sim/workers.py REAL_WORLDS): --env real with '
                         '--batch-setting starts; the simulator engines play the worlds of sim/combat.py VILLAGES')
    if (leg_from is None) != (leg is None):
        raise ValueError('a leg names the checkpoint it continues and its number: --leg-from and --leg')
    if unrecorded_parent and leg_from is None:
        raise ValueError('--unrecorded-parent describes the checkpoint of a leg (--leg-from, --leg)')
    parent = parent_data = None
    if leg_from is not None:
        if env != 'real':
            raise ValueError('legs continue a real-engine run (sim/real.py); a simulator run resumes (--recovery, --resume)')
        if type(leg) is not int or leg < 2:
            raise ValueError(f'a continuation is leg 2 or later (the first run is leg 1): {leg!r}')
        if init is not None or weights is not None:
            raise ValueError('a leg starts from its checkpoint\'s whole learner state: no --init or --weights')
        parent_data, parent, parent_sha256 = leg_parent(leg_from, leg, unrecorded_parent)
        if task in INNER_TASKS and not isinstance(parent_data.get(counts_saved(task)), dict):
            raise ValueError(f'{leg_from} holds no inner pay counts ({counts_saved(task)}): a leg of {task} goes on '
                             'from the counts its checkpoint saved')
        if parent is not None:  # the run's key: the leg's draws are its leg's streams of the same key (sim/real.py)
            if env_seed is not None and env_seed != parent['env_seed']:
                raise ValueError(f'{leg_from}\'s run draws from environment seed {parent["env_seed"]}, which its legs '
                                 f'keep: not --env-seed {env_seed}')
            env_seed = parent['env_seed']
            ours = dict(env_seed=env_seed, decisions=decisions, validate=bool(validate),
                        checkpoint_every=checkpoint_every or 8 * rollout)
            differ = {k: [parent[k], ours[k]] for k in LEG_RUN if parent[k] != ours[k]}
            if differ:
                raise ValueError(f'leg {leg} would continue {leg_from} under other run settings (the parent\'s, this '
                                 f'leg\'s): {differ}')
        if decisions != parent_data.get('_total_timesteps'):
            raise ValueError(f'{leg_from} is of a run of {parent_data.get("_total_timesteps")} decisions, which its legs '
                             f'continue to: --decisions {parent_data.get("_total_timesteps")}, not {decisions}')
    if env == 'real' and not re.fullmatch(r'cpu|cuda(:\d+)?', str(device)):  # the device decides the checks below
        raise ValueError(f'a real-engine run updates on cpu, cuda or cuda:N, not --device {device}')
    cuda_real = env == 'real' and str(device).startswith('cuda')
    if not cuda_real and (collect_device is not None or cudnn_tf32 is not None):
        raise ValueError('--collect-device and --cudnn-tf32 are for a real-engine run on CUDA (--env real --device cuda)')
    if cuda_real:
        if collect_device not in ('cpu', 'cuda') or cudnn_tf32 not in ('on', 'off'):
            raise ValueError('a real-engine run on CUDA names where its collection\'s policy calls run (--collect-device cpu '
                             'or cuda) and cuDNN\'s TF32 for its LSTM (--cudnn-tf32 off or on)')
        if collect_device == 'cuda' and cudnn_tf32 == 'on' and validate:
            raise ValueError('--validate recomputes the rows in other shapes than the collection\'s calls, and cuDNN\'s '
                             'TF32 rounds by shape: with the calls on the GPU (--collect-device cuda) give --cudnn-tf32 off')
        torch.backends.cudnn.allow_tf32 = cudnn_tf32 == 'on'
    if free and env == 'cpu' and not set(FREE_PLAY_NAMED) <= set(batch or {}):
        raise ValueError(f'{FREE_PLAY_TASK} on the CPU workers names {", ".join(FREE_PLAY_NAMED)} (--batch-setting): '
                         f'{sorted(set(FREE_PLAY_NAMED) - set(batch or {}))} missing')
    if free and learner['value_norm'] is not None and (replay is not None or not DeviceRecurrentPPO.FAST_UPDATE):
        raise ValueError('a PopArt value target trains with fast_train, which updates its statistics: the stock update and '
                         'the --replay probe, whose variants skip that update, are refused')
    slots = None if segments is None else segment_slots(worlds, goals, segments['worlds'])
    if segments is not None and not 0 < segments['decisions'] <= n_steps:
        raise ValueError(f'a {segments["decisions"]}-decision segment does not fit a {n_steps}-decision rollout')
    if threads:
        torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    if leg_from is not None:  # the leg's lineage at its start, which a killed leg leaves behind
        (output / 'lineage.json').write_text(json.dumps(dict(
            leg=leg, parent=str(leg_from), parent_sha256=parent_sha256,
            first=parent['first'] if parent is not None else str(leg_run_folder(leg_from)), parent_record=parent,
            unverified=None if parent is not None else list(LEG_UNVERIFIED), env_seed=env_seed or seed,
            decisions=decisions, started=time.strftime('%Y-%m-%dT%H:%M:%S%z')), indent=1) + '\n', encoding='utf-8')
    changes = dict(n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs)
    if free:
        changes['gae_lambda'] = learner['gae_lambda']
        if 'ent_coef' in learner:  # --ent-coef: the loss's coefficient (sim/learn.py), saved with every checkpoint
            changes['ent_coef'] = learner['ent_coef']
    telemetry = task in ('goal-activity-v4', 'goal-activity-v5', 'goal-activity-v11') if telemetry is None else telemetry
    archive = None  # the fake environment has no tables
    if env == 'cpu':  # the archive each Python reference worker loads (sim/workers.py load_tables), or on the C engine the
        from sim.build import DATA  # one each worker's open_pack holds equal to the pack's recorded input (sim/workers.py _native)
        archive = dict(path=str(tables or DATA), sha256=sha256(tables or DATA), pins_checked=tables is None and engine == 'cpu')
    packed = None
    if pack is not None:  # what the pack was exported from (its manifest's input digests) and the oracle it was exported at
        from sim.native import read_pack
        meta = read_pack(pack)[1]
        packed = dict(path=str(pack), sha256=sha256(pack), inputs=meta.get('inputs'), oracle_hash=meta.get('oracle_hash'))
    summary = dict(status='started', source=source(output), validate=validate, decisions=decisions, seed=seed,
                   env_seed=env_seed or seed, task=task, worlds=worlds, env=env, engine=engine, pack=packed,
                   goals=list(goals), retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT, tables=archive,
                   pool=None if pool is None else dict(path=str(pool), sha256=sha256(pool)),
                   search_pool=None if search_pool is None else dict(path=str(search_pool), sha256=sha256(search_pool)),
                   changes_from_learner={k: dict(learner=config.LEARNER[k], here=v) for k, v in changes.items()
                                         if config.LEARNER[k] != v},
                   rollout_transitions=rollout, host=versions(device),
                   segments=None if segments is None else dict(segments, slots=slots))
    if free:
        summary['free_play'] = learner
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
                                   engine, pack, batch, output, leg=leg or 1,
                                   collect_device='cpu' if collect_device == 'cpu' else None)
            times['env_compile_seconds'] = time.perf_counter() - t0
        if tuple(environment.env.goals) != goals:
            raise ValueError(f'the environment asks for goals {list(environment.env.goals)}, the run declares {list(goals)}')
        free_kwargs = {}
        if free:
            if not environment.last_obs:
                raise ValueError(f'{FREE_PLAY_TASK} trains on the free-play batch, which gives last_obs: this environment does not')
            if environment.settings_in_effect is None:
                raise ValueError(f'{FREE_PLAY_TASK}\'s environment reports no settings in effect')
            # the batch's settings in effect, recorded with the run's own and compared on a resume
            learner = dict(learner, batch_in_effect=environment.settings_in_effect)
            summary['free_play'] = dict(learner, observation=environment.observation)
            free_kwargs = dict(free_play=True, half_life_ticks=learner['half_life_ticks'], value_norm=learner['value_norm'])
        t0 = time.perf_counter()
        with staged(events, 'policy_build', device):
            model = build(environment, seed=seed, device=device, **changes, **free_kwargs)
            if task == FREE_PLAY_V2_TASK:  # saved with every checkpoint: its version and death price (VERSION_SAVED)
                model.free_play_task, model.death_price = task, death_price(task)
            if task in EVERY_STAT_TASKS:  # saved with every checkpoint: version, paid stats, kit
                model.free_play_task, model.paid_skills = task, list(environment.batch.paid_skills)
                model.kit_extra = [list(x) for x in task_batch(task)['kit_extra']]
            if task in INNER_TASKS:  # and its inner pay (the window too, from the first collection on: sim/real.py)
                model.inner_pay = task_batch(task)['inner']
            if env == 'real':
                devices = real_devices(model, environment.batch, cuda_real, collect_device)
                if devices is not None:  # saved with every checkpoint too; a processor run records none, as before
                    summary['devices'] = model.real_devices = devices
                # saved with every checkpoint: this leg's record, which the leg that continues it checks (leg_parent)
                model.real_leg = dict(number=leg or 1, first=str(output) if leg_from is None else
                                      parent['first'] if parent is not None else str(leg_run_folder(leg_from)),
                                      # leg 1: its batch's key; a later leg: the run's (check_leg_world)
                                      env_seed=environment.batch.key if leg_from is None else env_seed or seed,
                                      decisions=decisions, validate=bool(validate),
                                      checkpoint_every=checkpoint_every or 8 * rollout,
                                      world=leg_world(environment.settings_in_effect))
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
        if env == 'real' and environment.settings_in_effect.get('leg') != (leg or 1):
            raise ValueError(f"the environment draws leg {environment.settings_in_effect.get('leg')}'s streams, the run "
                             f'declares leg {leg or 1}')
        if env == 'real':  # the batch pays the task it declares: its price, paid stats, kit, pays and start hitpoints
            have = dict(death_price=environment.batch.price, all_skills=environment.batch.all_skills,
                        kit_extra=environment.batch.kit_extra, items=environment.batch.item_contract,
                        **({'inner': environment.batch.inner} if environment.batch.inner else {}),
                        **({'start_hp': True} if environment.batch.start_hp else {}),
                        **({'start_rule': environment.batch.start_rule} if environment.batch.start_rule else {}))
            if have != task_batch(task):
                raise ValueError(f"the environment's task settings {have} are not {task}'s {task_batch(task)}")
        if leg_from is not None:
            from stable_baselines3.common.utils import set_random_seed
            from sim.real import LEG_STREAMS
            with staged(events, 'policy_leg', device):
                summary['leg'] = load_leg(model, leg_from, device, leg, rollout)
            check_leg_world(leg_from, parent, parent_data, model, environment)  # after the learner settings' check
            summary['leg'].update(first=model.real_leg['first'], scopes=LEG_SCOPES, lineage='lineage.json',
                                  **({} if parent is not None else dict(unverified=list(LEG_UNVERIFIED))))
            if task in INNER_TASKS:  # the inner pay goes on from the checkpoint's window (sim/real.py RareXp)
                saved = parent_data[counts_saved(task)]
                if saved.get('version') != model.inner_pay:
                    raise ValueError(f"{leg_from} saved counts of {saved.get('version')!r}, not of {model.inner_pay!r}")
                environment.batch.restore_inner(saved)
                setattr(model, counts_saved(task), saved)
                summary['leg'].update(inner_kinds_at_start=len(saved['counts']))
            if decisions <= model.num_timesteps:
                raise ValueError(f'leg {leg} starts at {model.num_timesteps} decisions: --decisions {decisions} leaves none')
            generator = int(np.random.SeedSequence([seed, leg]).generate_state(1)[0])
            set_random_seed(generator, using_cuda=str(device).startswith('cuda'))
            summary['leg'].update(generator_seed=generator, stream_offset=(leg - 1) * LEG_STREAMS,
                                  lives='every agent began a new life at the leg\'s start, from its leg\'s streams')
        if telemetry:
            from sim.learn import attach
            attach(model, telemetry=telemetry)
            summary['treatments'] = dict(telemetry=telemetry)
        if resume is not None:
            saved = json.loads((Path(resume) / 'state.json').read_text()).get('learner')  # none before goal-free play
            if saved != learner:
                raise CheckFailed(f'the recovery state trained with goal-free settings {saved}, this run declares {learner}')
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
        # each periodic checkpoint's policy_digest by its file name, from the weights at its save (the tensors model.save
        # wrote; the digest of the reloaded zip is the same), entered at the save, so a failed run names what it saved
        summary['checkpoint_digests'] = {}
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
        if free:  # read back from the model: the settings it trains with
            summary['learner'].update(free_play=model.free_play, half_life_ticks=model.half_life_ticks,
                                      value_norm=model.policy.value_norm, observation=model.policy.observation,
                                      **{k: getattr(model, k) for k in VERSION_SAVED if hasattr(model, k)})
        if initial_only:  # a plan's frozen start digest: the run stops before its first rollout
            summary['status'] = 'initialized'
            return summary
        steps = [0]

        def counted_step(optimizer, args, kwargs):
            steps[0] += 1
        model.policy.optimizer.register_step_post_hook(counted_step)
        logged, step_counts, waits, memories, collect_seconds, ppo_seconds = [], [], dict(collect=[], update=[]), [], [], []
        treated = []  # per update: telemetry and the parameter change

        def record_losses():
            # RecurrentPPO.train() leaves its train/ values in the logger until the next iteration's dump.
            values = model.logger.name_to_value
            if any(k in values for k in LOSS_KEYS):
                logged.append({k: float(values[k]) for k in LOSS_KEYS + ('train/clip_fraction', 'train/explained_variance')
                               if k in values})
                tracker, sizes = getattr(model, 'tracker', None), getattr(model, 'update_sizes', None)
                if tracker is not None:  # the update's telemetry
                    at = lambda rows: rows[len(treated)] if rows is not None and len(rows) > len(treated) else None
                    treated.append(dict(telemetry=tracker.report(), update_size=at(sizes)))

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
        checks = []
        carried = dict(paid=torch.zeros(worlds, device=device, dtype=torch.float64 if free else None),  # life_pay's
                       done=torch.ones(worlds, device=device))
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
                    checks.append(validate_free_play(model, carried) if free else
                                  validate_rollout(model, carried, *task_limits(environment.env), goals=goals))
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
                if (weights or leg_from) is not None and len(step_counts) == 1:
                    with staged(events, 'source_check', device):
                        summary['source_check'] = check_source_outputs(model, weights or leg_from, device)
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
                summary['checkpoint_digests'][f'ppo_{model.num_timesteps}_steps.zip'] = policy_digest(model.policy)
                row.update(checkpoint=f'checkpoints/ppo_{model.num_timesteps}_steps.zip',
                           checkpoint_seconds=time.perf_counter() - t1)
            record = save_recovery(model, recovery, dict(epoch=time.time(), output=str(output), task=task,
                                                         goals=list(goals), **({'learner': learner} if free else {})))
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
                    summary['checkpoint_digests'][Path(self._checkpoint_path(extension='zip')).name] = \
                        policy_digest(self.model.policy)
                    Checkpoints.seconds += time.perf_counter() - t0
                    Checkpoints.saves += 1
                    index = len(step_counts) - 1
                    Checkpoints.during[index] = Checkpoints.during.get(index, 0) + time.perf_counter() - t0
                    tell('checkpoint_save', 'end', index=len(step_counts) - 1, seconds=time.perf_counter() - t0)
                return result
        class Boundaries(BaseCallback):  # the real engine's checkpoints: after the update, at the next rollout's start
            def _on_step(self):
                return True

            def _on_rollout_start(self):
                if len(step_counts) > 1:
                    boundary_checkpoint()

        def boundary_checkpoint():
            """A complete update whose count falls on `every`: its checkpoint holds that update (Checkpoints' timing)."""
            if model.num_timesteps % every:
                return
            t0, index = time.perf_counter(), len(step_counts) - 1
            tell('checkpoint_save', 'start', index=index)
            name = f'ppo_{model.num_timesteps}_steps.zip'
            (output / 'checkpoints').mkdir(exist_ok=True)
            model.save(output / 'checkpoints' / name)
            summary['checkpoint_digests'][name] = policy_digest(model.policy)
            seconds = time.perf_counter() - t0
            Checkpoints.seconds, Checkpoints.saves = Checkpoints.seconds + seconds, Checkpoints.saves + 1
            Checkpoints.during[index] = Checkpoints.during.get(index, 0) + seconds
            tell('checkpoint_save', 'end', index=index, seconds=seconds)
        every = checkpoint_every or 8 * rollout
        if (recovery is not None or env == 'real') and every % rollout:
            raise ValueError('checkpoints fall on complete updates: checkpoint_every must be a multiple of worlds x n_steps')
        learn_start = time.perf_counter()
        saving = [Recovery()] if recovery is not None else [Boundaries()] if env == 'real' else \
            [Checkpoints(max(1, every // worlds), str(output / 'checkpoints'), name_prefix='ppo')]
        callbacks = [Clock(), *saving, *callbacks]
        try:
            model.learn(decisions - start_timesteps, reset_num_timesteps=resume is None and leg_from is None,
                        callback=CallbackList(callbacks + ([Validation()] if validate else []) +
                                              ([Segments()] if segments is not None else []) + [Probe()]))
        finally:
            if validate:
                (output / 'checks.json').write_text(json.dumps(dict(rollouts=checks), indent=1) + '\n', encoding='utf-8')
        update_ended()  # the last update, whose losses no dump wrote
        after_update(len(step_counts) - 1)
        if recovery is not None:
            saved_update(len(step_counts) - 1)
        elif env == 'real':
            boundary_checkpoint()
        for a, on in switches.items():  # the switches are the class's: leave them as they were for whatever runs next
            setattr(type(model.policy), a, on)
        marks.append(('rollout', time.perf_counter()))
        t0 = time.perf_counter()
        with staged(events, 'final_save', device):
            model.save(output / 'latest.zip')
            summary['policy_sha256_at_end'] = policy_digest(model.policy)  # latest.zip's, from the weights at the save
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
        if free:  # lives, no goals (DeviceRolloutBuffer.episodes)
            episodes = [e for r in model.rollouts for e in zip(*(r['episodes'][k] for k in
                                                                  ('terminal', 'episode_return', 'episode_ticks', 'cut_value')))]
            cut = [e for e in episodes if e[0] == TERMINAL_HORIZON]
            summary['lives'] = dict(transitions=sum(r['transitions'][0] for r in model.rollouts), lives=len(episodes),
                                    deaths=sum(1 for e in episodes if e[0] == TERMINAL_DEATH), cut=len(cut),
                                    mean_return=float(np.mean([e[1] for e in episodes])) if episodes else None,
                                    mean_ticks=float(np.mean([e[2] for e in episodes])) if episodes else None,
                                    mean_cut_value=float(np.mean([e[3] for e in cut])) if cut else None)
            if getattr(model, 'paid_skills', None):  # free-play-v3: mean_return is every stat's; the seven old skills' apart
                c = environment.batch.counts  # the lives settled by the end of the run's last collection
                summary['lives'].update(paid_skills=list(model.paid_skills), settled=dict(
                    levels_seven=c['levels_seven'], levels_all=c['levels_all'],
                    **({} if 'inner_paid' not in c else dict(inner_lives=c['inner_lives'], inner_rows=c['inner_paid'])),
                    note='the lives settled by the last collection: their level gain over the seven old skills and over '
                         'every stat; the run\'s end (real_finish) settles the rest'
                         + ('; free-play-v15: their inner pay apart (inner_lives), and the inner pay of the rows handed '
                            'to the learner (inner_rows); mean_return is the levels alone' if 'inner_paid' in c else '')))
            if task in INNER_TASKS and hasattr(model, counts_saved(task)):  # the counts the last checkpoint saves
                summary['lives'].update(inner_kinds_counted=len(getattr(model, counts_saved(task))['counts']))
            price = getattr(model, 'death_price', None)
            if price:  # free-play-v2: mean_return is the level gain alone; the prices and the return with them apart
                lives = summary['lives']
                lives.update(death_price=price, mean_levels=lives['mean_return'], prices=price * lives['deaths'],
                             mean_return_with_price=lives['mean_return'] - price * lives['deaths'] / lives['lives']
                             if episodes else None)
            per_goal = None
        else:
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
        if leg_from is not None:  # beside total_timesteps and sb3_n_updates, the whole run's (LEG_SCOPES)
            summary['leg'].update(decisions_this_leg=model.num_timesteps - summary['leg']['decisions_at_start'],
                                  updates_whole_run=model._n_updates // model.n_epochs)
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
        if env == 'real':  # after the last update's record and latest.zip: every live agent leaves and the ledger of each
            summary['status'] = 'leaving'  # life it ends is checked; the run is complete only when that passes
            summary['real_finish'] = environment.batch.finish()
            summary['status'] = 'completed'
    except BaseException as error:
        stop = terminated_in(error)
        if stop is None or stop is error:
            summary.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        # the SIGTERM landed inside a write (torch.save into a checkpoint's zip entry), whose cleanup error replaced it
        summary.update(status='failed', error=f'{type(stop).__name__}: {stop}', cut_write=f'{type(error).__name__}: {error}')
        raise SystemExit(stop.code) from error
    finally:
        summary['wall_seconds'] = time.perf_counter() - started
        if env == 'real' and environment is not None:  # the batch's record (sim/real.py)
            summary['real'] = environment.batch.report()
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        if env == 'real' and environment is not None:  # then its worlds are left and its workers end
            environment.close()
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


def batch_settings(items):
    """--batch-setting KEY=JSON ... as a dict (None when none is given); a value that is not JSON is refused."""
    if not items:
        return None
    out = {}
    for item in items:
        key, sep, value = item.partition('=')
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            sep = ''
        if not (key and sep):
            raise SystemExit(f'--batch-setting {item!r}: KEY=VALUE with a JSON value (a string in quotes)')
    return out


def terminated(signum, frame):
    """The train command's SIGTERM handler: the signal ends the run as an exit of 128 + its number (143), which train's
    `finally` meets, so summary.json records status 'failed' (error 'SystemExit: 143') with its source, run facts and
    pack. A supervisor's stop before its deadline is a SIGTERM; Python's
    default action ends the process with no summary, which no entry point can then describe (sim/hour.py policy_run).
    A signal that lands inside a checkpoint's write is still this stop (terminated_in), and the cut file is never read as
    a checkpoint: its zip entry fails to load, and only a save that returned records the file's digest."""
    raise SystemExit(128 + signum)


def terminated_in(error):
    """The SIGTERM's SystemExit (terminated) that `error` is or was raised while unwinding, else None: inside torch.save
    writing a checkpoint into a zip entry (stable-baselines3's save), the stop cuts the write and the writer's cleanup
    raises its own error ("unexpected pos ..."), which replaces it."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, SystemExit) and error.code == 128 + signal.SIGTERM:
            return error
        seen.add(id(error))
        error = error.__context__
    return None


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
        command.add_argument('--env', default='fake', choices=('fake', 'cpu', 'real') if name == 'train' else ('fake', 'cpu'),
                             help='train --env real: the real engine\'s worlds inside scripts/stack.py (sim/real.py; '
                             '--task free-play-v1, --batch-setting starts and life)')
        command.add_argument('--workers', type=int, help='--env cpu: worker processes (sim/workers.py)')
        if name == 'train':
            command.add_argument('--engine', default='cpu', choices=('cpu', 'native'),
                                 help='--env cpu: the Python reference (cpu) or the C port (native, needs --pack)')
            command.add_argument('--pack', type=Path, help='--engine native: the exported pack (python -m sim.native export)')
        command.add_argument('--tables', type=Path, help='--env cpu: the simulator archive (default logs/sim/data, pins '
                             'checked); with --engine native the pack must come from it (default logs/sim/data\'s archive '
                             'too, held equal to the pack\'s recorded input)')
        command.add_argument('--worlds', type=int, default=BATCHING['worlds'])
        command.add_argument('--pool', type=Path, help='--env cpu: a sim/world.py layout pool saved with --derived')
        command.add_argument('--n-steps', type=int, default=None if name == 'train' else BATCHING['n_steps'],
                             help=f'decisions per world per update (default {BATCHING["n_steps"]}; {FREE_PLAY_TASK} needs it given)')
        if name in ('train', 'bench'):
            command.add_argument('--batch-size', type=int, default=BATCHING['batch_size'])
            command.add_argument('--epochs', type=int, default=BATCHING['n_epochs'])
            command.add_argument('--threads', type=int, help='torch CPU threads (default: torch\'s own)')
            command.add_argument('--goals', nargs='+', choices=GOALS,
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
                                          'goal-activity-v11', *FREE_PLAY_TASKS))
            command.add_argument('--death-price', type=float, help=f'{FREE_PLAY_V2_TASK} only, and then required: its '
                                 f'death price in levels, {death_price(FREE_PLAY_V2_TASK):g} (a constant of the version that '
                                 'the command, and so its launch record, names; any other value is refused)')
            command.add_argument('--half-life', type=float, help=f'{FREE_PLAY_TASK}: the per-tick discount\'s half-life in game ticks '
                                 f'(default {FREE_PLAY_LEARNER["half_life_ticks"]}; 3000 is the declared sensitivity value)')
            command.add_argument('--gae-lambda', type=float, help=f'{FREE_PLAY_TASK}: lambda (default {FREE_PLAY_LEARNER["gae_lambda"]})')
            command.add_argument('--ent-coef', type=float, help=f'the free-play tasks: PPO\'s entropy coefficient '
                                 f'(default config.LEARNER\'s {config.LEARNER["ent_coef"]}, then unrecorded; from '
                                 f'{ENT_COEF_RANGE[0]} to {ENT_COEF_RANGE[1]}; a leg repeats its checkpoint\'s, --weights '
                                 'records the checkpoint\'s beside it)')
            command.add_argument('--value-norm', choices=('popart', 'none'), help=f'{FREE_PLAY_TASK}: the value target\'s '
                                 f'normalization (default {FREE_PLAY_LEARNER["value_norm"]}; none is the declared ablation)')
            command.add_argument('--batch-setting', nargs='+', metavar='KEY=JSON', help=f'{FREE_PLAY_TASK}: the batch\'s own '
                                 'keyword settings, each value JSON (sim/vec.py CpuBatch); --env cpu needs '
                                 f'{", ".join(FREE_PLAY_NAMED)} (world="village-world-v1" or "village-world-v2")')
            command.add_argument('--search-pool', type=Path, help='goal-activity-v4: the search layouts; v5 and v11: the supply '
                                                                 'layouts (python -m sim.search pool / supply-pool)')
            command.add_argument('--replay', nargs='+', metavar='INDEX VARIANT', help='after update INDEX, replay its PPO phase '
                                 'under each REPLAY_VARIANTS name (the first is the reference): timing and parameter differences '
                                 '(not under PopArt)')
            command.add_argument('--weights', type=Path, help='start from this checkpoint\'s policy tensors only (fresh optimizer)')
            command.add_argument('--recovery', type=Path, help='leave a resumable state here after every complete update')
            command.add_argument('--resume', type=Path, help='continue from this recovery directory')
            command.add_argument('--leg-from', type=Path, help='--env real: a checkpoint of an earlier leg, whose whole '
                                 'learner state this leg continues (with --leg)')
            command.add_argument('--leg', type=int, help='--env real: this leg\'s number, its checkpoint\'s leg plus one '
                                 '(its draw streams)')
            command.add_argument('--unrecorded-parent', action='store_true', help='--env real: --leg-from is a first '
                                 'leg\'s checkpoint saved before checkpoints held their leg record; this leg 2 then '
                                 f'declares {", ".join(LEG_UNVERIFIED)} unchecked (its summary and lineage.json say so)')
            command.add_argument('--collect-device', choices=('cpu', 'cuda'), help='--env real --device cuda, and then '
                                 'required: where the collection\'s policy calls run (cpu: a copy that takes the weights '
                                 'after each update, sim/real.py HANDOVER; cuda: the model\'s own policy)')
            command.add_argument('--cudnn-tf32', choices=('off', 'on'), help='--env real --device cuda, and then required: '
                                 'torch.backends.cudnn.allow_tf32 for the run (on is refused with --collect-device cuda and '
                                 '--validate)')
            command.add_argument('--env-seed', type=int, help='the environment\'s seed (default --seed; a resume declares a new one)')
            command.add_argument('--segments', action='store_true', help='per-update evidence: save and replay a fixed '
                                 'sample of world segments every update (segments/)')
            command.add_argument('--segment-worlds', type=int, default=SEGMENTS['worlds'],
                                 help='world slots in the sample, split equally between the goals')
            command.add_argument('--segment-decisions', type=int, default=SEGMENTS['decisions'],
                                 help='decisions of each slot, from its rollout\'s first')
            command.add_argument('--initial-only', action='store_true', help='save initial.zip and summary.json\'s '
                                 'policy_sha256_at_start, then stop before the first rollout (a plan\'s start digests)')
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
        signal.signal(signal.SIGTERM, terminated)
        if args.n_steps is None and args.task in FREE_PLAY_TASKS:
            raise SystemExit(f'{args.task}: give the rollout length (--n-steps); the study sets it')
        if (args.task == FREE_PLAY_V2_TASK) != (args.death_price is not None) or \
                (args.death_price is not None and args.death_price != death_price(args.task)):
            raise SystemExit(f'--death-price {death_price(FREE_PLAY_V2_TASK):g} goes with --task {FREE_PLAY_V2_TASK}, and '
                             f'only with it: not {args.death_price} with {args.task}')
        result = train(args.output, decisions=args.decisions, seed=args.seed, worlds=args.worlds,
                       n_steps=BATCHING['n_steps'] if args.n_steps is None else args.n_steps,
                       batch_size=args.batch_size, n_epochs=args.epochs, device=args.device, env=args.env,
                       checkpoint_every=args.checkpoint_every, threads=args.threads, pool=args.pool,
                       goals=tuple(args.goals) if args.goals else None,
                       validate=args.validate, workers=args.workers, tables=args.tables, task=args.task, weights=args.weights,
                       recovery=args.recovery, resume=args.resume, env_seed=args.env_seed, search_pool=args.search_pool,
                       segments=dict(worlds=args.segment_worlds, decisions=args.segment_decisions) if args.segments else None,
                       replay=(int(args.replay[0]), tuple(args.replay[1:])) if args.replay else None,
                       ppo_profile=(set(args.ppo_profile), args.output / 'ppo-profile') if args.ppo_profile else None,
                       engine=args.engine, pack=args.pack, half_life_ticks=args.half_life,
                       gae_lambda=args.gae_lambda, value_norm=args.value_norm, batch=batch_settings(args.batch_setting),
                       initial_only=args.initial_only, leg_from=args.leg_from, leg=args.leg,
                       unrecorded_parent=args.unrecorded_parent,
                       collect_device=args.collect_device, cudnn_tf32=args.cudnn_tf32, ent_coef=args.ent_coef)
    elif args.command == 'bench':
        result = bench(worlds=args.worlds, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
                       updates=args.updates, seed=args.seed, device=args.device, env=args.env, threads=args.threads,
                       pool=args.pool, goals=tuple(args.goals) if args.goals else GOALS, validate=args.validate)
    else:
        result = survey(worlds=args.worlds, n_steps=args.n_steps, rollouts=args.rollouts, seed=args.seed,
                        device=args.device, env=args.env, pool=args.pool)
    print(json.dumps(result, indent=1))
    if args.command != 'train' and args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
