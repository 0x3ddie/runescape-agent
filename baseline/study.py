"""The native transfer study runner.

    python -m baseline.study freeze --out MANIFEST --campaign-start EPOCH --slots N --eval-workers P --hardware JSON
    python -m baseline.study execute --manifest MANIFEST --run pretrained-101 --work DIR [--resume ZIP --transitions T]
    python -m baseline.study analyze --manifest MANIFEST --runs DIR --out analysis.json

freeze writes the frozen manifest: source revision and code digest, library versions, both engine builds, the name
table, the six initializations (each simulator policy and its actual saved initialization), the start records, the
learner and slot settings, the checkpoints, the evaluation schedule, the time and recovery limits, the hardware and
this file's digest (the analysis). attempt() is one container's attempt at one run: it applies the recovery rule
(one resume from a verified complete-update state; a third start refuses), then runs execute inside a stack whose
world service has both engine builds. execute trains (baseline.train --study arena, weights only, --validate) and
then evaluates every checkpoint the run reached with greedy actions, each trial in a fresh world: the first 50
evaluation records of the four arena goals at 0, 262,144, 524,288 and 1,048,576 decisions, and all 200 records of
every site-goal at 2,097,152. analyze computes the frozen metrics from the evaluation results.

The run decision cap counts kept and discarded native decisions: after a resume, training continues only while
kept plus discarded decisions stay within it, so a checkpoint past it is reported missing, never substituted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from baseline import config
from baseline.resource_task import GOALS

ROOT = Path(__file__).resolve().parents[1]
GROUPS = dict(pretrained='latest.zip', fresh='initial.zip')  # the simulator policy; its saved initialization
SEEDS = (101, 202, 303)
DECISIONS = 2_097_152
CHECKPOINTS = (0, 262_144, 524_288, 1_048_576, DECISIONS)
TRAIN_SET = 'arena'
ARENA = tuple(f'study_{TRAIN_SET}_{goal}' for goal in GOALS)
SITES = tuple(config.STUDY_FIXTURES)  # every applicable site-goal, arena first
EVAL_SEED = 41001
CHUNK = 25  # evaluation trials per evaluate process
SAMPLED_SEEDS = (EVAL_SEED, 909001)  # the final checkpoint's two sampled passes
RUN_SECONDS, CAMPAIGN_SECONDS = 24 * 3600, 72 * 3600
# Code the container runs, for the code digest; tests and caches excluded.
CODE = ('baseline', 'rl', 'env-ts', 'scripts/stack.py', 'sim/__init__.py', 'sim/spec.py')
CODE_SKIPPED = ('__pycache__', '.DS_Store', 'tests')
# The frozen analysis: bootstrap resamples and seed for paired-start intervals; the pilot thresholds; competence.
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 17_017
PASS = dict(mean_advantage=0.10, each_pair_positive=True, goal_drop=0.05)
COMPETENCE = 160  # controlled successes of 200 in every seed and site-goal
TARGET = 40  # controlled successes of 50: the target-crossing decision count (secondary)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def code_digest(root=ROOT):
    total = hashlib.sha256()
    for relative in CODE:
        top = root / relative
        files = [top] if top.is_file() else sorted(
            Path(d) / f for d, dirs, names in os.walk(top) if not set(Path(d).relative_to(top).parts) & set(CODE_SKIPPED)
            for f in names if f not in CODE_SKIPPED and not f.endswith('.pyc'))
        for path in files:
            total.update(path.relative_to(root).as_posix().encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return total.hexdigest()


def records_digest(name, kind):
    return hashlib.sha256(json.dumps(config.study_records(name, kind), sort_keys=True).encode()).hexdigest()


def runs():
    """The six runs, as two chains of three: each seed's pair trains at the same time on matched containers."""
    return [f'{group}-{seed}' for seed in SEEDS for group in GROUPS]


def batch(slots):
    return config.LEARNER['n_steps'] * slots


def evaluations(checkpoint, decisions=DECISIONS):
    """(fixture, first trial, trials) jobs of the frozen schedule at one checkpoint."""
    final = checkpoint == decisions
    fixtures, trials = (SITES, config.STUDY_EVAL_RECORDS) if final else (ARENA, config.STUDY_INTERMEDIATE_TRIALS)
    return [(f, k, min(CHUNK, trials - k)) for f in fixtures for k in range(0, trials, CHUNK)]


def freeze(out, *, campaign_start, slots, eval_workers, hardware, initializations, names, engines):
    """The frozen study manifest. `initializations`: {run: path}; `engines`: {name: rs-sdk checkout}."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from stack import engine_identity
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    changed = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no', '--', *CODE],
                             capture_output=True, text=True, check=True).stdout.strip()
    if changed:
        raise RuntimeError(f'study code has uncommitted changes: {changed}')
    if slots % len(GOALS) or any(c % batch(slots) for c in CHECKPOINTS):
        raise ValueError(f'{slots} slots do not give four equal goal quotas and complete-update checkpoints')
    from importlib.metadata import version
    manifest = dict(
        frozen_epoch=time.time(), revision=revision, code_digest=code_digest(), analysis_sha256=sha256(__file__),
        design='native transfer study',
        versions=dict(python=platform.python_version(),
                      **{p: version(p) for p in ('torch', 'stable-baselines3', 'sb3-contrib', 'gymnasium', 'numpy')}),
        engines={name: dict(path=str(path), **engine_identity(Path(path))) for name, path in engines.items()},
        names=dict(path=str(names), sha256=sha256(names)),
        initializations={run: dict(path=str(path), sha256=sha256(path)) for run, path in initializations.items()},
        records={f'{name}-{kind}': records_digest(name, kind) for name in config.START_SETS for kind in ('eval', 'train')
                 if kind == 'eval' or name == TRAIN_SET},
        learner=dict(config.LEARNER, policy_kwargs=str(config.LEARNER['policy_kwargs'])),
        training=dict(start_set=TRAIN_SET, slots=slots, batch=batch(slots), decisions=DECISIONS,
                      checkpoints=list(CHECKPOINTS), goal_of_slot='GOALS[i % 4]', kl_coef=0, validate=True,
                      device='cpu', xp_rate=config.XP_RATE, horizon_ticks=240, seeds=list(SEEDS),
                      groups=dict(GROUPS)),
        evaluation=dict(decoding='greedy', seed=EVAL_SEED, intermediate=dict(fixtures=list(ARENA),
                        trials=config.STUDY_INTERMEDIATE_TRIALS, checkpoints=list(CHECKPOINTS[:-1])),
                        final=dict(fixtures=list(SITES), trials=config.STUDY_EVAL_RECORDS, checkpoint=DECISIONS,
                                   sampled_passes=[dict(seed=s) for s in SAMPLED_SEEDS]),
                        fresh_world_per_trial=True, workers=eval_workers, chunk=CHUNK),
        runs=runs(), chains=[[f'{g}-{s}' for s in SEEDS] for g in GROUPS],
        limits=dict(campaign_start_epoch=campaign_start, campaign_deadline_epoch=campaign_start + CAMPAIGN_SECONDS,
                    run_seconds=RUN_SECONDS, concurrent_training_jobs=2, gpus_per_job=0,
                    shutdown_reserve_seconds=900, evaluation_reserve_seconds=3600),
        recovery=('At most one infrastructure-only resume per run, from the last persisted complete-update state '
                  '(model, optimizer, counters, random states; its sha256 verified), with fresh episode worlds and '
                  'recurrent state and a new rollout. Discarded native decisions (the last persisted progress minus '
                  'the recovery point, plus one batch for collection not yet persisted) count against the run cap '
                  'of 2,097,152; training stops at the last complete update within it; a checkpoint not reached '
                  'is missing. The run deadline is 24 h from its first attempt, whatever happens between. A third '
                  'start of a run refuses. No new seed, restart from zero or extension.'),
        analysis=dict(primary='normalized trapezoidal area under the arena controlled-success curve over the five '
                              'checkpoints, records 0-49 at every checkpoint, per goal and run; paired advantage '
                              'pretrained minus fresh per seed and goal, averaged equally',
                      bootstrap=dict(resamples=BOOTSTRAP, seed=BOOTSTRAP_SEED,
                                     unit='evaluation record (shared by goals, groups, seeds and checkpoints)'),
                      passes=PASS, competence=dict(threshold=COMPETENCE, of=config.STUDY_EVAL_RECORDS),
                      secondary=dict(target_crossing=f'first checkpoint with >= {TARGET}/50 arena controlled successes')),
        hardware=hardware)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    return manifest


def verify(manifest, *, names, initialization, worlds=None):
    """Refuse to run on anything other than the frozen inputs: code, analysis, records, name table, the run's
    initialization and (with a world service) both engine builds."""
    problems = []
    if code_digest() != manifest['code_digest']:
        problems.append('code digest')
    if sha256(__file__) != manifest['analysis_sha256']:
        problems.append('analysis digest')
    for key, digest in manifest['records'].items():
        name, kind = key.rsplit('-', 1)
        if records_digest(name, kind) != digest:
            problems.append(f'records {key}')
    if sha256(names) != manifest['names']['sha256']:
        problems.append('name table')
    if initialization is not None and sha256(initialization['path']) != initialization['sha256']:
        problems.append('initialization')
    if worlds is not None:
        found = worlds.identity()['engines']
        for name, engine in manifest['engines'].items():
            if {k: found.get(name, {}).get(k) for k in ('tree', 'digest')} != {k: engine[k] for k in ('tree', 'digest')}:
                problems.append(f'engine {name}')
    if problems:
        raise RuntimeError(f'the study inputs differ from the manifest: {problems}')


def evaluate_command(checkpoint_path, target, fixture, first, trials, policy=None, sampled=False, seed=EVAL_SEED):
    """One baseline.evaluate process of the frozen schedule: the fixture's trials from `first`, greedy or, with `sampled`,
    sampled under the declared seed base. The records are the fixture's own either way."""
    return [sys.executable, '-m', 'baseline.evaluate', '--output', str(target),
            *(['--policy', policy] if policy else ['--checkpoint', str(checkpoint_path)]),
            '--fixtures', fixture, '--trials', str(trials), '--first-trial', str(first), '--seed', str(seed),
            *(['--sampled'] if sampled else []), '--final', '--world-service']


def evaluate_checkpoint(checkpoint_path, decisions, out, workers, timeout, budget=DECISIONS, policy=None, limit=0,
                        sampled=False, seed=EVAL_SEED):
    """The frozen evaluation of one checkpoint: chunks of trials as parallel baseline.evaluate processes, each
    trial in a fresh world from the world service. Returns {fixture: rows}; refuses an incomplete chunk.
    policy: a control (reference or wait) instead of the checkpoint, and limit: the first trials only (preflight);
    sampled and seed: one of the final checkpoint's sampled passes."""
    jobs = [(f, k, n if not limit else min(n, limit - k)) for f, k, n in evaluations(decisions, budget)
            if not limit or k < limit]
    def run(job):
        fixture, first, trials = job
        target = out / f'{fixture}-{first:03d}'
        if (target / 'results.json').exists() and json.loads((target / 'results.json').read_text())['status'] == 'completed':
            return target
        argv = evaluate_command(checkpoint_path, target, fixture, first, trials, policy, sampled, seed)
        with open(out / f'{fixture}-{first:03d}.log', 'w', encoding='utf-8') as log:
            code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        if code:
            raise RuntimeError(f'evaluation {target.name} of {checkpoint_path} failed ({code})')
        return target
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        targets = list(pool.map(run, jobs))
    rows = {}
    for target in targets:
        result = json.loads((target / 'results.json').read_text())
        for trial in result['trials']:
            rows.setdefault(trial['fixture'], []).append({k: trial.get(k) for k in TRIAL_FIELDS})
    return rows


TRIAL_FIELDS = ('fixture', 'trial', 'start', 'goal', 'success', 'controlled_success', 'first_completed', 'survived',
                'ticks_to_outcome', 'completion_ticks', 'non_goal_before_outcome', 'refusals_before_outcome',
                'decisions', 'reward', 'reward_bounds', 'kills', 'produced_by_activity')


def execute(manifest, run, work, *, resume=None, transitions=None, deadline):
    """One attempt's work inside a stack with both engines' world service: verify, train, evaluate what the run
    reached. Writes work/result.json; returns it."""
    from baseline.world import FreshWorlds
    group, seed = run.rsplit('-', 1)
    seed, slots = int(seed), manifest['training']['slots']
    budget, checkpoints = manifest['training']['decisions'], manifest['training']['checkpoints']
    transitions = budget if transitions is None else transitions
    worlds = FreshWorlds()
    init = manifest['initializations'][run]
    names = Path(os.environ.get('RL_NAME_TABLE') or manifest['names']['path'])
    # A container finds the initializations by content (STUDY_INPUTS/<sha256>.zip); a checkout by their paths.
    inputs = os.environ.get('STUDY_INPUTS')
    init_path = Path(inputs) / f"{init['sha256']}.zip" if inputs else ROOT / init['path']
    verify(manifest, names=names, initialization=dict(init, path=init_path), worlds=worlds)
    result = dict(run=run, group=group, seed=seed, resume=str(resume) if resume else None, transitions=transitions,
                  started=time.time(), deadline=deadline)
    write = lambda: (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    work.mkdir(parents=True, exist_ok=True)
    write()
    reserve = manifest['limits']['evaluation_reserve_seconds']
    argv = [sys.executable, '-m', 'baseline.train', '--output', str(work / 'train'), '--study', TRAIN_SET,
            '--kl-coef', '0', '--agents', str(slots), '--transitions', str(transitions), '--seed', str(seed),
            '--checkpoints', *map(str, checkpoints), '--validate', '--device', 'cpu',
            *(['--resume', str(resume)] if resume else ['--weights', str(init_path)])]
    try:
        with open(work / 'train.log', 'w', encoding='utf-8') as log:
            code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=max(60, deadline - reserve - time.time())).returncode
        result['train'] = dict(code=code)
    except subprocess.TimeoutExpired:
        result['train'] = dict(code=None, error='training reached its deadline')
    summary = work / 'train' / 'summary.json'
    if summary.exists():
        s = json.loads(summary.read_text())
        result['train'].update(status=s['status'], error=s.get('error'), total_timesteps=s.get('total_timesteps'),
                               ppo_updates=s.get('ppo_updates'), learner=s.get('learner'), seconds=s.get('seconds'),
                               weights_source=s.get('weights_source'), checkpoints_saved=s.get('checkpoints_saved'))
    result['trained'] = time.time()
    write()
    # Evaluate every checkpoint this attempt saved, in order, with the frozen schedule.
    workers = manifest['evaluation']['workers']
    result['evaluations'] = {}
    for decisions in checkpoints if manifest['evaluation'].get('checkpoints', True) else ():
        path = work / 'train' / 'checkpoints' / f'decisions-{decisions}.zip'
        if not path.exists():
            continue
        final = decisions == budget
        worlds.target(TRAIN_SET, workers // 2 + 1 if final else workers + 1)
        worlds.target('pinned', workers // 2 + 1 if final else 0)
        started = time.time()
        rows = evaluate_checkpoint(path, decisions, work / 'eval' / f'decisions-{decisions}', workers,
                                   timeout=max(60, deadline - time.time()), budget=budget)
        result['evaluations'][str(decisions)] = dict(
            sha256=sha256(path), decoding='greedy', seconds=round(time.time() - started, 1),
            controlled={f: sum(bool(r['controlled_success']) for r in v) for f, v in rows.items()},
            trials={f: len(v) for f, v in rows.items()})
        (work / 'eval' / f'decisions-{decisions}' / 'trials.json').write_text(json.dumps(rows) + '\n', encoding='utf-8')
        write()
        # the final checkpoint's sampled passes: the same records beside the greedy pass,
        # under eval-sampled{p}, outside load_trials' greedy glob
        for p, pass_ in enumerate(manifest['evaluation']['final'].get('sampled_passes', []) if final else [], 1):
            started, folder = time.time(), work / f'eval-sampled{p}' / f'decisions-{decisions}'
            rows = evaluate_checkpoint(path, decisions, folder, workers, timeout=max(60, deadline - time.time()),
                                       budget=budget, sampled=True, seed=pass_['seed'])
            result['evaluations'][f'{decisions}-sampled{p}'] = dict(
                sha256=sha256(path), decoding='sampled', seed=pass_['seed'], seconds=round(time.time() - started, 1),
                controlled={f: sum(bool(r['controlled_success']) for r in v) for f, v in rows.items()},
                trials={f: len(v) for f, v in rows.items()})
            (folder / 'trials.json').write_text(json.dumps(rows) + '\n', encoding='utf-8')
            write()
    result['ended'] = time.time()
    write()
    return result


# ---- recovery rule and one container's attempt (the remote runner calls attempt) ----

def plan(manifest, state, root, now, call_id=None):
    """What this attempt may do: start, resume once from a verified complete-update state, or refuse. Only a
    restart of the call that began the last attempt (a cloud runner may re-run a lost container's input) may resume it."""
    attempts, slots, cap = state['attempts'], manifest['training']['slots'], manifest['training']['decisions']
    if state.get('final'):
        return dict(kind='refuse', reason=f"the run has ended ({state['final']})")
    if not attempts:
        return dict(kind='start', transitions=cap)
    if 'ended' not in attempts[-1] and attempts[-1].get('call_id') != call_id:
        return dict(kind='refuse', reason='another call began this run and has not ended; no second worker')
    if len(attempts) >= 2:
        return dict(kind='refuse', reason='a resume was already used; a third start refuses')
    previous = root / f"attempt-{attempts[-1]['attempt']}"
    records = sorted((previous / 'train' / 'recovery').glob('update-*.json'))
    if not records:
        return dict(kind='refuse', reason='no persisted complete-update state to resume from')
    record = json.loads(records[-1].read_text())
    model = records[-1].with_suffix('.zip')
    if not model.exists() or sha256(model) != record['sha256'] or record['num_timesteps'] % batch(slots):
        return dict(kind='refuse', reason='the persisted recovery state does not verify')
    progress_path = previous / 'train' / 'progress.json'
    progress = json.loads(progress_path.read_text())['num_timesteps'] if progress_path.exists() else 0
    discarded = max(0, progress - record['num_timesteps']) + batch(slots)
    kept = record['num_timesteps']
    remaining = (cap - kept - discarded) // batch(slots) * batch(slots)
    if remaining <= 0:
        return dict(kind='refuse', reason='the decision cap leaves no complete update after the discarded collection')
    return dict(kind='resume', recovery=str(model), recovery_timesteps=kept, discarded_persisted=progress - kept,
                discarded_counted=discarded, transitions=remaining)


def attempt(manifest, run, root, work, *, bun, engines, blocks, persist, now=time.time, call_id=None):
    """One container's attempt at `run`: root is its persistent folder, work this container's disk; persist()
    copies work to root. Applies the recovery rule, then runs execute inside a stack with a world service."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from stack import run as stack_run
    root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(run=run, attempts=[])
    decision = plan(manifest, state, root, now(), call_id)
    first = state['attempts'][0]['started'] if state['attempts'] else now()
    deadline = min(first + manifest['limits']['run_seconds'], manifest['limits']['campaign_deadline_epoch']) - \
        manifest['limits']['shutdown_reserve_seconds']
    if decision['kind'] != 'refuse' and deadline - now() < 3600:
        decision = dict(kind='refuse', reason='less than an hour remains before the run or campaign deadline')
    save = lambda: state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    if decision['kind'] == 'refuse':
        state.setdefault('refusals', []).append(dict(time=now(), reason=decision['reason']))
        if state['attempts'] and not state.get('final'):
            state['final'] = 'incomplete'
        save()
        return state
    k = len(state['attempts']) + 1
    state['attempts'].append(dict(attempt=k, started=now(), plan=decision, deadline=deadline, call_id=call_id))
    save()
    out = work / f'attempt-{k}'
    command = [sys.executable, '-m', 'baseline.study', 'execute', '--manifest', str(work / 'manifest.json'),
               '--run', run, '--work', str(out), '--deadline', str(deadline), '--transitions', str(decision['transitions']),
               *(['--resume', decision['recovery']] if decision['kind'] == 'resume' else [])]
    (work / 'manifest.json').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
    try:
        code = stack_run(bun, out / 'stack', max(60, deadline - now()), command, worlds=0, lockstep=True,
                         xp_rate=config.XP_RATE, engines=engines, blocks=blocks)
    except Exception as error:
        code = f'{type(error).__name__}: {error}'
    persist()
    result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
    state['attempts'][-1].update(ended=now(), code=code, result=result)
    trained = result.get('train', {})
    reached = trained.get('total_timesteps') == decision['recovery_timesteps'] + decision['transitions'] \
        if decision['kind'] == 'resume' else trained.get('total_timesteps') == manifest['training']['decisions']
    if trained.get('status') == 'completed' and reached and code == 0:
        state['final'] = 'completed' if decision['kind'] == 'start' else 'completed after resume (cap reduced)'
    elif trained.get('status') == 'failed' and 'CheckFailed' in str(trained.get('error')):
        # A validity defect stops collection everywhere until it is diagnosed.
        state['final'] = 'invalid'
        (root.parent / 'STOP').write_text(f'{run}: {trained.get("error")}\n', encoding='utf-8')
    else:
        # Not resumed automatically: only a restarted container (the attempt never ended) resumes by itself.
        state['final'] = 'incomplete'
    save()
    persist()
    return state


# ---- the frozen analysis ----

def load_trials(runs_dir, run):
    """{decisions: {fixture: [rows]}} from every attempt's evaluations of `run` (an evaluation is never repeated)."""
    out = {}
    for path in sorted((runs_dir / run).glob('attempt-*/eval/decisions-*/trials.json')):
        decisions = int(path.parent.name.split('-')[1])
        if decisions in out:
            raise RuntimeError(f'{run} has two evaluations at {decisions}')
        out[decisions] = json.loads(path.read_text())
    return out


def success_matrix(trials, fixtures, records):
    """[checkpoint, fixture, record] controlled successes (NaN where not evaluated)."""
    m = np.full((len(CHECKPOINTS), len(fixtures), records), np.nan)
    for c, decisions in enumerate(CHECKPOINTS):
        for f, fixture in enumerate(fixtures):
            for row in trials.get(decisions, {}).get(fixture, []):
                if row['trial'] < records:
                    m[c, f, row['trial']] = float(bool(row['controlled_success']))
    return m


def area(curve):
    """Normalized trapezoidal area over the checkpoints (fractions of the run's decisions)."""
    x = np.asarray(CHECKPOINTS, float) / DECISIONS
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(x)))


def analyze(manifest, runs_dir):
    n = config.STUDY_INTERMEDIATE_TRIALS
    trials = {run: load_trials(runs_dir, run) for run in manifest['runs']}
    arena = {run: success_matrix(trials[run], ARENA, n) for run in trials}
    complete = {run: not np.isnan(m).any() for run, m in arena.items()}
    report = dict(cells={}, areas={}, complete=complete)
    for run, m in arena.items():
        report['cells'][run] = {f'{d}': {g: None if np.isnan(m[c, i]).any() else int(m[c, i].sum())
                                         for i, g in enumerate(GOALS)} for c, d in enumerate(CHECKPOINTS)}
        report['areas'][run] = {g: None if np.isnan(m[:, i]).any() else area(m[:, i].mean(-1)) for i, g in enumerate(GOALS)}
    pairs = [(f'pretrained-{s}', f'fresh-{s}') for s in SEEDS]
    if all(complete[p] and complete[f] for p, f in pairs):
        def advantages(index):
            """[seed, goal] paired area advantage on the records `index`."""
            return np.asarray([[area(arena[p][:, g, index].mean(-1)) - area(arena[f][:, g, index].mean(-1))
                                for g in range(len(GOALS))] for p, f in pairs])
        point = advantages(np.arange(n))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = np.stack([advantages(rng.integers(0, n, n)) for _ in range(BOOTSTRAP)])
        interval = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        mean, per_pair, per_goal = point.mean(), point.mean(1), point.mean(0)
        report['pretraining'] = dict(
            mean_advantage=float(mean), mean_interval=interval(samples.mean((1, 2))),
            pair_advantage={s: float(v) for s, v in zip(SEEDS, per_pair)},
            pair_interval={s: interval(samples.mean(2)[:, k]) for k, s in enumerate(SEEDS)},
            goal_advantage={g: float(v) for g, v in zip(GOALS, per_goal)},
            goal_interval={g: interval(samples.mean(1)[:, k]) for k, g in enumerate(GOALS)},
            cells={s: {g: float(point[k, i]) for i, g in enumerate(GOALS)} for k, s in enumerate(SEEDS)},
            passes=bool(mean >= PASS['mean_advantage'] and (per_pair > 0).all() and (per_goal >= -PASS['goal_drop']).all()))
    else:
        report['pretraining'] = dict(passes=None, reason='a run lacks a checkpoint evaluation: incomplete, not computed')
    # Native competence: all 200 records of every site-goal at the final checkpoint, per seed and recipe.
    report['competence'] = {}
    for group in GROUPS:
        cells = {}
        for seed in SEEDS:
            final = trials[f'{group}-{seed}'].get(DECISIONS, {})
            cells[seed] = {f: (sum(bool(r['controlled_success']) for r in final[f]) if len(final.get(f, [])) ==
                               config.STUDY_EVAL_RECORDS else None) for f in SITES}
        values = [v for c in cells.values() for v in c.values()]
        report['competence'][group] = dict(cells=cells, passes=None if None in values else all(v >= COMPETENCE for v in values))
    # Secondary: first checkpoint reaching the target; deaths and wrong first activities at each evaluation.
    report['target_crossing'] = {run: {g: next((d for c, d in enumerate(CHECKPOINTS)
                                                if not np.isnan(m[c, i]).any() and m[c, i].sum() >= TARGET), None)
                                       for i, g in enumerate(GOALS)} for run, m in arena.items()}
    report['secondary'] = {run: {str(d): {f: dict(deaths=sum(not r['survived'] for r in rows),
                                                  wrong_first=sum(r['first_completed'] not in (None, r['goal']) for r in rows),
                                                  nothing=sum(r['first_completed'] is None for r in rows),
                                                  refusals=float(np.mean([r['refusals_before_outcome'] for r in rows])),
                                                  ticks_to_outcome=float(np.median([r['ticks_to_outcome'] for r in rows
                                                                                    if r['ticks_to_outcome'] is not None]))
                                                  if any(r['ticks_to_outcome'] is not None for r in rows) else None)
                                          for f, rows in t.items()} for d, t in trials[run].items()}
                           for run in trials}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--campaign-start', type=float, required=True, help='epoch of the first remote study preflight')
    f.add_argument('--slots', type=int, required=True)
    f.add_argument('--eval-workers', type=int, required=True)
    f.add_argument('--hardware', required=True, help='JSON: provider, workspace, cpu, memory, gpu')
    f.add_argument('--campaign', type=Path, default=ROOT / 'logs/sim/campaign')
    f.add_argument('--names', type=Path, default=ROOT / 'logs/names/name-vectors.npz')
    f.add_argument('--engine', action='append', default=[], metavar='NAME=CHECKOUT')
    e = sub.add_parser('execute')
    e.add_argument('--manifest', type=Path, required=True)
    e.add_argument('--run', required=True)
    e.add_argument('--work', type=Path, required=True)
    e.add_argument('--deadline', type=float, required=True)
    e.add_argument('--transitions', type=int, default=DECISIONS)
    e.add_argument('--resume', type=Path)
    a = sub.add_parser('analyze')
    a.add_argument('--manifest', type=Path, required=True)
    a.add_argument('--runs', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        inits = {f'{g}-{s}': args.campaign / f's{s}' / 'run' / name for s in SEEDS for g, name in GROUPS.items()}
        engines = dict(spec.split('=', 1) for spec in args.engine)
        print(json.dumps(freeze(args.out, campaign_start=args.campaign_start, slots=args.slots,
                                eval_workers=args.eval_workers, hardware=json.loads(args.hardware),
                                initializations=inits, names=args.names, engines=engines), indent=2))
    elif args.command == 'execute':
        manifest = json.loads(args.manifest.read_text())
        result = execute(manifest, args.run, args.work, resume=args.resume, transitions=args.transitions,
                         deadline=args.deadline)
        sys.exit(0 if result.get('train', {}).get('status') == 'completed' else 1)
    else:
        manifest = json.loads(args.manifest.read_text())
        report = analyze(manifest, args.runs)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('pretraining', 'competence', 'complete')}, indent=2))


if __name__ == '__main__':
    main()
