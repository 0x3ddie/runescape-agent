"""Throughput of the simulated goal task's environment, each stage measured on its own.

    .venv-sim/bin/python -m sim.bench cpu-env --workers 8 --worlds 4 --decisions 200 --pool logs/sim/world/dev-mixed-4-derived.npz

cpu-env: the CPU reference of the four-goal task (sim/vec.py CpuBatch) on a compact-world pool, one
process per worker, each with `--worlds` worlds (the first value) stepped by uniformly random legal
actions for `--decisions` decisions per world; it stops at a world's unsupported content (CpuBatch
refuses that world). Complete training (policy, collection, updates) is sim.train's own benchmark
(`python -m sim.train bench`), not here. The JAX kernel and environment stages were removed with the JAX
simulator. Startup (tables, worker processes) is reported separately and excluded from the rates. The
host, versions and peak memory go with every result.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
from queue import Empty
import resource
import subprocess
import sys
import time

import numpy as np


def host():
    import torch
    import stable_baselines3
    import sb3_contrib
    cpu = platform.processor()
    if sys.platform == 'darwin':
        cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or cpu
    elif Path('/proc/cpuinfo').exists():  # Linux (a cloud host)
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    from sim import paths
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    return dict(cpu=cpu, cores=mp.cpu_count(), memory_gb=round(memory / 2 ** 30, 1), os=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                sb3=stable_baselines3.__version__, sb3_contrib=sb3_contrib.__version__, device='cpu',
                sim_paths=paths.IMPL)


def peak_rss_mb(children=False):
    usage = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10), 1)  # bytes on macOS, KiB on Linux


def load_tables(path=None):
    """logs/sim/data's archive with its pins checked, or `path` as given (a copy away from the checkout, where the pins'
    inputs are absent, as a remote machine loads it)."""
    from sim.data import Tables
    return Tables.load() if path is None else Tables.load(path, check_pins=False)


# ---- the CPU reference --------------------------------------------------------------------------------
def _cpu_env_worker(index, worlds, decisions, seed, pool, queue, tables=None):
    from sim.engine import World
    from sim.vec import CpuBatch
    from sim.world import load_pool
    t0 = time.perf_counter()
    layouts, derived = load_pool(pool)
    env = CpuBatch(load_tables(tables), layouts, worlds, derived=derived)
    startup = time.perf_counter() - t0
    ticks = [0]
    cycle = World.cycle

    def counted(self):
        ticks[0] += 1
        return cycle(self)
    World.cycle = counted
    rng = np.random.default_rng([seed, index])
    state, obs = env.reset(seed * 1000 + index)
    steps = ended = unsupported = 0
    start, before = time.perf_counter(), ticks[0]
    while steps < decisions:
        actions = np.asarray([rng.choice(np.flatnonzero(obs['mask'][b])) for b in range(worlds)], np.int32)
        state, obs, _, done, info = env.step(state, actions)
        steps += 1
        ended += int(done.sum())
        unsupported += int(info['unsupported'].sum())
        if unsupported:  # CpuBatch refuses a world after unsupported content: the run stops there
            break
    elapsed = time.perf_counter() - start
    queue.put(dict(startup=startup, seconds=elapsed, decisions=steps * worlds, episodes=ended, unsupported=unsupported,
                   ticks=ticks[0] - before, peak_rss_mb=peak_rss_mb()))


def bench_cpu_env(workers, worlds, decisions, seed, pool, tables=None):
    context = mp.get_context('spawn')
    queue = context.Queue()
    processes = [context.Process(target=_cpu_env_worker, args=(i, worlds, decisions, seed, pool, queue, tables))
                 for i in range(workers)]
    for p in processes:
        p.start()
    rows = []
    while len(rows) < workers:
        try:
            rows.append(queue.get(timeout=5))
        except Empty:
            if any(not p.is_alive() and p.exitcode for p in processes):
                raise RuntimeError('a cpu-env worker failed')
    for p in processes:
        p.join()
    total = lambda key: sum(r[key] for r in rows)
    wall = max(r['seconds'] for r in rows)
    return dict(stage='cpu-env', label='env-only throughput of the CPU reference (sim/vec.py CpuBatch), not training',
                world=f'compact pool {pool}', workers=workers, worlds_per_worker=worlds,
                policy='uniformly random legal actions', wall_seconds=wall, decisions=total('decisions'),
                decisions_per_second=total('decisions') / wall, world_ticks_per_second=total('ticks') / wall,
                ticks_per_decision=total('ticks') / max(1, total('decisions')), episodes=total('episodes'),
                unsupported=total('unsupported'), startup_seconds=max(r['startup'] for r in rows),
                worker_peak_rss_mb=max(r['peak_rss_mb'] for r in rows), workers_detail=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('cpu-env',))
    parser.add_argument('--workers', type=int, default=1, help='worker processes')
    parser.add_argument('--decisions', type=int, default=64, help='decisions per world')
    parser.add_argument('--worlds', type=int, default=4, help='worlds per worker')
    parser.add_argument('--seed', type=int, default=909)
    parser.add_argument('--pool', required=True, help='a compact-world pool file with derived tables')
    parser.add_argument('--tables', help='the simulator archive, its pins unchecked (default: logs/sim/data, checked)')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = bench_cpu_env(args.workers, args.worlds, args.decisions, args.seed, args.pool, args.tables)
    result['host'] = host()
    print(json.dumps({k: v for k, v in result.items() if k != 'workers_detail'}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
