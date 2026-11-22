"""Throughput of the native simulator (sim/native.py NativeBatch): `python -m sim.native bench`.

World ticks are every world cycle a step runs, the setup cycles of the episodes it begins included (RsSlot.cycles);
reset's own cycles fall before the timed loop. Game ticks and
policy decisions are reported separately. A case is an activity (a task's four-goal mix, or one goal in every slot)
and an action source: the equivalence harness's keyed random legal play or its biased play (sim/native_harness.py
RandomLegal, Biased), or a checkpoint's sampled actions (policy:PATH[:seed], PolicySource). Each case first records
its actions: one batch plays `--decisions` steps from reset(key) under the source. Then each repeat resets a batch
(untimed) and replays that list, timing only the loop of step calls (NativeBatch.step_into with or without its
check_batch), so the source's own cost is left out and every repeat and thread count plays the same trajectory (a
replay that ran another number of cycles raises). The rate of a repeat is its ticks over its loop time; a case reports
the median repeat, the spread, the load average (1 minute) before and after each repeat (the host is shared, so a
number is read with its load) and, on macOS, the instructions the process retired per tick in the loop, which do not
depend on the load or the core type and so compare two builds.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native bench --pack logs/native/train-8-nc1-a6c325a-ids3-item3.pack \\
        --cases v5,v5:wood,v2 --sources random,biased --worlds 64 --decisions 200 --threads 1,4 --check off,on

A pack opens only against the archive and pools it was exported from (--tables, --pool, --supply-pool; by default the
default pack's), as the training workers open theirs (sim/native.py open_pack).

`python -m sim.native env` plays the same cases through sim/workers.py CpuWorkers, the path sim.train collects through,
with either engine (env_run): uniformly random legal actions from one numpy generator, timing only the reset and step
calls. Given both engines it also compares them (compare_engines: the per-step tick sums, the episode ends and, with
--digest, a hash of every step's observations, reward, done and info) and exits 1 unless they agree; the local machine's
1,024-world v2 run at key 11 and seed 20261027 gives 642,784 ticks and 2,221 episodes with either engine.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native env --cases v2,v5:melee --engines cpu,native --workers 4 \\
        --worlds 1024 --decisions 64 --key 11 --seed 20261027 --digest
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

LOGS = Path(__file__).resolve().parents[1] / 'logs'

TASKS = dict(v2='goal-activity-v2', v3='goal-activity-v3', v5='goal-activity-v5')
CASES = ('v5', 'v5:wood', 'v5:fish', 'v5:mine', 'v5:melee', 'v3', 'v2', 'v2:wood', 'v2:fish', 'v2:mine', 'v2:melee')
# free-play-v1 cases: the batch settings of each (no muggers, so a village-world-v1 and a village-world-v2 pack play the
# same case); the pack's records decide the world
FREE_CASES = {'free': dict(xp_rate=1, muggers=False, life=('uniform', 100, 1000)),
              'free-x25': dict(xp_rate=25, muggers=False, life=('uniform', 100, 1000)),
              'free-long': dict(xp_rate=1, muggers=False, life=('fixed', 36000))}


def parse_case(case):
    """'v5' (the four-goal mix) or 'v5:wood' (one goal in every slot) -> (task, goals or None)."""
    version, _, goal = case.partition(':')
    if version not in TASKS:
        raise ValueError(f'unknown case {case}: a task of {sorted(TASKS)}, optionally :goal')
    return TASKS[version], ((goal,) if goal else None)


def batch_of(pack, case, n, threads=1, inputs=None):
    """The NativeBatch of a case: a goal task's (parse_case) or a free-play one's (FREE_CASES). inputs: the archive and
    pools the pack must be exported from ({name: path}, open_pack), as the training workers name them."""
    from sim.native import FREE_TASK, NativeBatch
    if case in FREE_CASES:
        return NativeBatch(pack, n, task=FREE_TASK, threads=threads, inputs=inputs, **FREE_CASES[case])
    task, goals = parse_case(case)
    return NativeBatch(pack, n, task=task, goals=goals, threads=threads, inputs=inputs)


def source_of(spec, key):
    from sim import native_harness
    return native_harness.make_source(spec, key)


def record(pack, case, source, n, decisions, key, inputs=None):
    """The actions [decisions][n] the source plays from reset(key), and the world cycles their steps run."""
    batch = batch_of(pack, case, n, inputs=inputs)
    play = source_of(source, key)
    batch.reset_into(key, check=False)
    slots, starts, actions = np.arange(n), np.ones(n, bool), []
    before = batch.cycles
    for step in range(decisions):
        obs = batch.observations() if source.startswith('policy:') else {'mask': batch.views['obs/mask']}
        action = np.asarray(play(step, obs, slots, starts), np.int32)
        actions.append(action)
        batch.step_into(action, check=False)
        starts = batch.views['done'].copy()
        if batch.views['info/unsupported'].any():
            raise RuntimeError(f'{case} {source}: unsupported content at step {step} (the benchmark plays none)')
    return np.stack(actions), batch.cycles - before


class _Usage(ctypes.Structure):
    """macOS struct rusage_info_v4 (sys/resource.h): a uuid, then 39 counters; ri_instructions is the 30th."""
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('counters', ctypes.c_uint64 * 39)]


def instructions():
    """Instructions this process has retired (macOS proc_pid_rusage RUSAGE_INFO_V4), None elsewhere: a count that does
    not depend on the load or the core type, for comparing builds on the shared host."""
    if sys.platform != 'darwin':
        return None
    usage = _Usage()
    if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) != 0:
        return None
    return usage.counters[29]


def replay(pack, case, actions, threads, check, key, inputs=None):
    """One timed replay: (world ticks, seconds of the step loop, episodes ended, instructions retired or None)."""
    n = actions.shape[1]
    batch = batch_of(pack, case, n, threads, inputs)
    batch.reset_into(key, check=False)
    before, episodes = batch.cycles, sum(batch.episodes)
    count = instructions()
    t0 = time.perf_counter()
    for action in actions:
        batch.step_into(action, check)
    seconds = time.perf_counter() - t0
    after = instructions()
    return batch.cycles - before, seconds, sum(batch.episodes) - episodes, None if count is None else after - count


def load1():
    return round(os.getloadavg()[0], 2)


def bench_case(pack, case, source, n, decisions, key, threads, checks, repeats, inputs=None):
    """Rows of one (case, source): one per (threads, check), each with its repeats."""
    from sim.native import load
    load()  # build outside any timing
    actions, ticks = record(pack, case, source, n, decisions, key, inputs)
    rows = []
    for t in threads:
        for check in checks:
            runs = []
            for _ in range(repeats):
                load_before = load1()
                got, seconds, ended, count = replay(pack, case, actions, t, check, key, inputs)
                if got != ticks:
                    raise RuntimeError(f'{case} {source}: a replay ran {got} cycles, the recording {ticks}')
                runs.append(dict(seconds=seconds, instructions=count, load_before=load_before, load_after=load1()))
            rates = sorted(ticks / r['seconds'] for r in runs)
            median = statistics.median(rates)
            counts = [r['instructions'] for r in runs if r['instructions'] is not None]
            rows.append(dict(case=case, world=batch_of(pack, case, 1, inputs=inputs).world, source=source, worlds=n,
                             decisions=decisions * n, key=key, threads=t,
                             check=check, ticks=ticks, ticks_per_decision=ticks / (decisions * n), episodes=ended,
                             ticks_per_second=median, us_per_tick=1e6 / median,
                             decisions_per_second=median * decisions * n / ticks,
                             instructions_per_tick=statistics.median(counts) / ticks if counts else None,
                             spread=[rates[0], rates[-1]], repeats=runs))
    return rows


def line(row):
    loads = ' '.join(f'{r["load_before"]:.1f}-{r["load_after"]:.1f}' for r in row['repeats'])
    counted = row['instructions_per_tick']
    counted = '' if counted is None else f'{counted:,.0f} instructions/tick; '
    return (f'{row["case"]:9s} {row["source"]:8s} threads {row["threads"]} check {"on " if row["check"] else "off"} '
            f'{row["ticks_per_second"]:>11,.0f} ticks/s {row["us_per_tick"]:6.2f} us/tick '
            f'{row["decisions_per_second"]:>9,.0f} decisions/s ({row["ticks"]:,} ticks, '
            f'{row["ticks_per_decision"]:.2f}/decision, spread {row["spread"][0]:,.0f}-{row["spread"][1]:,.0f}, best '
            f'{1e6 / row["spread"][1]:.2f} us/tick; {counted}load {loads})')


def add_arguments(parser):
    parser.add_argument('--pack', required=True, help='a pack (logs/native/NAME.pack)')
    parser.add_argument('--cases', default=','.join(CASES),
                        help=f'comma separated, of {", ".join((*CASES, *FREE_CASES))}')
    parser.add_argument('--sources', default='random,biased', help='random, biased, policy:PATH[:seed], comma separated')
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--decisions', type=int, default=200, help='decisions per world')
    parser.add_argument('--key', type=int, default=1)
    parser.add_argument('--threads', default='1', help='thread counts, comma separated')
    parser.add_argument('--check', default='off,on', help='check_batch off, on or both (off,on)')
    parser.add_argument('--repeats', type=int, default=3)
    pack_inputs(parser)
    parser.add_argument('--out', type=Path, help='append the rows to this JSON lines file')


def pack_inputs(parser):
    """The archive and pools a pack must be exported from (open_pack's inputs, as the training workers pass them); the
    defaults are the default pack's. An empty --supply-pool leaves the supply pool unchecked."""
    parser.add_argument('--pool', default=str(LOGS / 'sim/world/train-8-nc1-derived.npz'))
    parser.add_argument('--supply-pool', default=str(LOGS / 'supply/pools/train-128.npz'))
    parser.add_argument('--tables', default=str(LOGS / 'sim/data/draynor-nc1.npz'), help='the archive (pins unchecked)')


def main(args):
    threads = [int(t) for t in args.threads.split(',')]
    checks = [{'off': False, 'on': True}[c] for c in args.check.split(',')]
    inputs = {name: path for name, path in (('tables', args.tables), ('pool', args.pool), ('supply_pool', args.supply_pool))
              if path}
    for case in args.cases.split(','):
        for source in args.sources.split(','):
            for row in bench_case(args.pack, case, source, args.worlds, args.decisions, args.key, threads, checks,
                                  args.repeats, inputs):
                print(line(row), flush=True)
                if args.out:
                    with args.out.open('a') as f:
                        f.write(json.dumps(row) + '\n')


# ---- the environment through the training workers: python -m sim.native env ----------------------------------------
def fit_worlds(worlds, workers, goals):
    """The least world count of at least `worlds` that the workers split evenly and the goal set divides (CpuWorkers'
    rule): 1,024 for 1 to 32 workers and four goals, 1,056 for 48."""
    step = math.lcm(workers, goals)
    return -(-worlds // step) * step


def env_run(case, engine, workers, worlds, decisions, key, seed, pool, supply_pool, tables, pack=None, digest=False):
    """One run through CpuWorkers with `engine` ('cpu', the Python reference, or 'native'): reset(key), then `decisions`
    steps, each world's action drawn uniformly over its returned mask (one numpy generator of `seed` scores every
    action; the legal one scored highest is played), so two engines that agree play the same trajectories. Only the
    reset and step calls are timed. Ticks are info['decision_ticks'] summed, the native ticks the decisions advanced
    (episode setup cycles not counted). The row keeps the per-step tick sums and episode ends; with `digest`,
    outputs_sha256 hashes every observation (reset's included), reward, done and info array in a fixed order."""
    from baseline.resource_task import GOALS
    from sim.workers import CpuWorkers
    task, goals = parse_case(case)
    goals = goals or tuple(GOALS)
    n = fit_worlds(worlds, workers, len(goals))
    load_before = load1()
    t0 = time.perf_counter()
    env = CpuWorkers(n, workers, pool, tables=tables, task=task, goals=goals, engine=engine,
                     search_pool=supply_pool if task == TASKS['v5'] else None, pack=pack if engine == 'native' else None)
    startup = time.perf_counter() - t0
    rng, outputs = np.random.default_rng(seed), hashlib.sha256() if digest else None

    def update(arrays):
        for name in sorted(arrays):
            outputs.update(np.ascontiguousarray(arrays[name]).tobytes())
    try:
        t = time.perf_counter()
        state, obs = env.reset(key)
        reset_seconds, step_seconds, ticks, ends = time.perf_counter() - t, 0.0, [], []
        if digest:
            update(obs)
        for _ in range(decisions):
            mask = obs['mask'].astype(bool)
            score = rng.random(mask.shape)
            score[~mask] = -1.0
            action = score.argmax(1).astype(np.int32)
            t = time.perf_counter()
            state, obs, reward, done, info = env.step(state, action)
            step_seconds += time.perf_counter() - t
            if info['unsupported'].any():
                raise RuntimeError(f'{case} {engine}: unsupported content at step {len(ticks)} (random play reaches none)')
            ticks.append(int(info['decision_ticks'].sum()))
            ends.append(int(done.sum()))
            if digest:
                update(dict(obs, **{f'info/{k}': v for k, v in info.items()}, reward=reward, done=done))
    finally:
        env.close()
    total = sum(ticks)
    return dict(case=case, engine=engine, workers=workers, worlds=n, decisions_per_world=decisions, decisions=n * decisions,
                key=key, seed=seed, startup_seconds=startup, reset_seconds=reset_seconds, step_seconds=step_seconds,
                decisions_per_second=n * decisions / step_seconds, ticks=total, ticks_per_second=total / step_seconds,
                ticks_per_decision=total / (n * decisions), episodes=sum(ends),
                tick_sha256=hashlib.sha256(np.asarray([ticks, ends], '<i8').tobytes()).hexdigest(),
                outputs_sha256=outputs.hexdigest() if digest else None, ticks_by_step=ticks, ends_by_step=ends,
                load_before=load_before, load_after=load1())


def compare_engines(rows):
    """Per case and world count with both engines, the Python reference's first row against the native one's: exact
    when the per-step tick sums, the episode ends and (when both hashed them) the outputs agree; first_step is the
    first step whose ticks or ends differ (None when they agree)."""
    out = []
    for case, n in dict.fromkeys((r['case'], r['worlds']) for r in rows):
        cpu, nat = (next((r for r in rows if (r['case'], r['worlds'], r['engine']) == (case, n, e)), None) for e in ('cpu', 'native'))
        if cpu is None or nat is None:
            continue
        steps = list(zip(cpu['ticks_by_step'], cpu['ends_by_step'])), list(zip(nat['ticks_by_step'], nat['ends_by_step']))
        first = next((k for k, (a, b) in enumerate(zip(*steps)) if a != b), None)
        if first is None and len(steps[0]) != len(steps[1]):
            first = min(len(steps[0]), len(steps[1]))
        hashed = cpu['outputs_sha256'] is not None and nat['outputs_sha256'] is not None
        outputs_equal = cpu['outputs_sha256'] == nat['outputs_sha256'] if hashed else None
        out.append(dict(case=case, worlds=n, decisions_per_world=cpu['decisions_per_world'], key=cpu['key'], seed=cpu['seed'],
                        ticks=dict(cpu=cpu['ticks'], native=nat['ticks']), episodes=dict(cpu=cpu['episodes'], native=nat['episodes']),
                        tick_sha256=dict(cpu=cpu['tick_sha256'], native=nat['tick_sha256']), first_step=first,
                        outputs_sha256=dict(cpu=cpu['outputs_sha256'], native=nat['outputs_sha256']), outputs_equal=outputs_equal,
                        exact=first is None and outputs_equal is not False))
    return out


def env_line(row):
    return (f'{row["case"]:9s} {row["engine"]:6s} workers {row["workers"]:>3} worlds {row["worlds"]} '
            f'{row["decisions_per_second"]:>11,.0f} decisions/s {row["ticks_per_second"]:>13,.0f} ticks/s ({row["ticks"]:,} ticks, '
            f'{row["episodes"]:,} episodes, step {row["step_seconds"]:.2f} s, reset {row["reset_seconds"]:.2f} s, startup '
            f'{row["startup_seconds"]:.1f} s; ticks {row["tick_sha256"][:12]}'
            f'{"" if row["outputs_sha256"] is None else " outputs " + row["outputs_sha256"][:12]}; load {row["load_before"]:.1f}-'
            f'{row["load_after"]:.1f})')


def env_arguments(parser):
    from sim.native import PACK
    parser.add_argument('--cases', default='v2,v5:melee', help=f'comma separated, of {", ".join(CASES)}')
    parser.add_argument('--engines', default='cpu,native', help='cpu (the Python reference), native, or both (compared)')
    parser.add_argument('--workers', default='1', help='worker counts, comma separated; host is this machine\'s CPU count')
    parser.add_argument('--worlds', type=int, default=1024, help='at least this many (fit_worlds rounds up to split evenly)')
    parser.add_argument('--decisions', type=int, default=64, help='decisions per world')
    parser.add_argument('--key', type=int, default=11)
    parser.add_argument('--seed', type=int, default=20261027, help='the action generator\'s seed')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--digest', action='store_true', help='hash every step\'s outputs (outputs_sha256)')
    pack_inputs(parser)
    parser.add_argument('--pack', default=str(PACK))
    parser.add_argument('--out', type=Path, help='append each row, then each comparison, to this JSON lines file')


def env_main(args):
    """Every case, engine, worker count and repeat in that order; then compare_engines when both engines ran. Exits 1
    if a comparison is not exact."""
    workers = [os.cpu_count() if w == 'host' else int(w) for w in args.workers.split(',')]
    workers = list(dict.fromkeys(workers))  # 'host' may repeat a listed count
    engines = args.engines.split(',')
    rows = []

    def emit(row):
        if args.out:
            with args.out.open('a') as f:
                f.write(json.dumps(row) + '\n')
    for case in args.cases.split(','):
        for engine in engines:
            for w in workers:
                for _ in range(args.repeats):
                    row = env_run(case, engine, w, args.worlds, args.decisions, args.key, args.seed, args.pool,
                                  args.supply_pool, args.tables, args.pack, args.digest)
                    print(env_line(row), flush=True)
                    emit(row)
                    rows.append(row)
    compared = compare_engines(rows)
    for row in compared:
        print(f'[env] {row["case"]} {row["worlds"]} worlds: {"EXACT" if row["exact"] else "DIFFERS"} (ticks {row["ticks"]}, '
              f'episodes {row["episodes"]}, first differing step {row["first_step"]}, outputs equal {row["outputs_equal"]})',
              flush=True)
        emit(dict(row, comparison=True))
    if not all(row['exact'] for row in compared):
        sys.exit(1)
