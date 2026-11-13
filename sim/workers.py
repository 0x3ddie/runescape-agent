"""CPU reference workers behind preallocated shared buffers: sim/spec.py's batched interface over
B worlds of sim/vec.py CpuBatch (or, with engine='native', sim/native.py NativeBatch writing its slice in place), split
into contiguous slices across N processes.

Each world belongs to one slice for the whole run and keeps its global slot: its goal (GOALS[slot % 4], or the one goal
of a single-goal set, sim/spec.py goal_set), its keyed starts (sim/task.py draw_start), its episode sequence and its memory never move, so the batch draws what one CpuBatch
of B worlds draws and every rollout holds the same transitions per goal. The observations, rewards, done flags, info
and actions live in one shared memory block allocated before the workers start; a worker writes only its slice.

Synchronous: step() writes the actions, wakes every worker and returns when the last has written its slice. Each step's
wall time and every worker's busy time (its CpuBatch call and its copy into the buffer) are kept, so the waiting for the
slowest worker and the communication are measured, not assumed (timing()). A worker that raises (CpuBatch refuses a
world it reported unsupported) answers with the error and its evidence, and step() raises once every worker has answered.
The returned arrays are views of the shared buffer, valid until the next call: a caller that keeps them copies them.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import weakref

import numpy as np

from baseline.resource_task import GOALS
from sim.spec import HORIZON_TICKS, INFO_FIELDS, OBSERVATION_SHAPES, dtype, goal_set
from sim.task import FAMILIES


def fields(n):
    """The shared buffer's arrays for n worlds: name -> (shape, dtype)."""
    out = {f'obs/{k}': ((n, *shape), np.dtype(dtype(k))) for k, shape in OBSERVATION_SHAPES.items()}
    out.update({f'info/{k}': ((n,), np.dtype(kind)) for k, kind in INFO_FIELDS.items()})
    from sim.vec import EPISODE_FACTS
    out.update({'info/made': ((n, len(FAMILIES)), np.dtype(np.int32)), 'info/kills': ((n,), np.dtype(np.int32)),
                'info/facts': ((n, len(EPISODE_FACTS)), np.dtype(np.int32)),
                'reward': ((n,), np.dtype(np.float32)), 'done': ((n,), np.dtype(bool)), 'action': ((n,), np.dtype(np.int32))})
    return out


def views(buffer, n):
    """The arrays of fields(n) over `buffer` (None: only the size), each 64-byte aligned; returns (views, size)."""
    out, offset = {}, 0
    for name, (shape, kind) in fields(n).items():
        offset = -(-offset // 64) * 64
        if buffer is not None:
            out[name] = np.ndarray(shape, kind, buffer=buffer, offset=offset)
        offset += int(np.prod(shape)) * kind.itemsize
    return out, offset


def load_tables(path):
    from sim.data import Tables
    return Tables.load() if path is None else Tables.load(path, check_pins=False)


def _native(first, count, tables, pool, horizon, task, search_pool, goals, pack, v, mine):
    """The worker's NativeBatch (sim/native.py) over the pack exported from these inputs, writing its slice in place."""
    from sim.native import NativeBatch
    inputs = {name: path for name, path in (('tables', tables), ('pool', pool), ('supply_pool', search_pool)) if path}
    env = NativeBatch(pack, count, first=first, task=task, goals=goals, horizon=horizon, inputs=inputs)
    env.bind({name: array[mine] for name, array in v.items()})
    return env


def _worker(first, count, n, name, tables, pool, horizon, task, conn, search_pool=None, goals=GOALS, engine='cpu',
            pack=None):
    from sim.vec import CpuBatch
    from sim.world import load_pool
    block = shared_memory.SharedMemory(name=name)
    v = env = None
    try:
        v, _ = views(block.buf, n)
        mine = slice(first, first + count)
        t0 = time.perf_counter()
        if engine == 'native':
            env = _native(first, count, tables, pool, horizon, task, search_pool, goals, pack, v, mine)
        else:
            layouts, derived = load_pool(pool)
            search = None if search_pool is None else load_pool(search_pool)
            env = CpuBatch(load_tables(tables), layouts, count, horizon=horizon, derived=derived, first=first, task=task,
                           search=search, goals=goals)
        conn.send(('ready', time.perf_counter() - t0))
        state, reported = None, 0
        while True:
            command, argument = conn.recv()
            if command == 'close':
                break
            t = time.perf_counter()
            obs = info = None  # the native batch writes its slice of the views itself
            try:
                if engine == 'native':
                    if command == 'reset':
                        env.reset_into(argument)
                    else:
                        env.step_into(v['action'][mine])
                elif command == 'reset':
                    state, obs = env.reset(argument)
                else:
                    state, obs, reward, done, info = env.step(state, v['action'][mine])
            except Exception as error:
                conn.send(('error', f'worker for slots {first}-{first + count - 1}: {type(error).__name__}: {error}',
                           env.evidence))
                continue
            for k, value in (obs or {}).items():
                v[f'obs/{k}'][mine] = value
            if info is not None:
                v['reward'][mine], v['done'][mine] = reward, done
                for k, value in info.items():
                    v[f'info/{k}'][mine] = value
            # a world flagged unsupported in this step: its evidence (the reason, the start, the action), without the
            # observation, travels with the reply so the collector's stop record names it
            fresh = [{k: x for k, x in row.items() if k != 'observation'} for row in env.evidence[reported:]]
            reported = len(env.evidence)
            conn.send(('ok', time.perf_counter() - t, fresh))
    finally:
        del v, env  # the native batch holds views of the block
        block.close()


def _release(block, processes, conns):
    for conn in conns:
        try:
            conn.send(('close', None))
        except (BrokenPipeError, OSError):
            pass
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
    try:
        block.close()
    except BufferError:  # a view still open in this process: the unlink below frees the block all the same
        pass
    block.unlink()


class CpuWorkers:
    """sim/spec.py's interface over n CPU reference worlds in `workers` processes (module docstring). pool: a
    sim/world.py pool file with its derived tables; tables: the simulator archive's path (None: logs/sim/data,
    its pins checked)."""

    def __init__(self, n, workers, pool, tables=None, horizon=HORIZON_TICKS, context='spawn', task='goal-activity-v2',
                 search_pool=None, goals=GOALS, engine='cpu', pack=None):
        """task: the four-goal task version (sim/combat.py task_spec); `horizons` and `targets` give each goal's.
        search_pool: goal-activity-v4's search layouts (sim/search.py), a pool file each worker loads beside `pool`.
        goals: the declared goal set (sim/spec.py goal_set): all four, or one goal in every world slot.
        engine: 'cpu' (CpuBatch, the default) or 'native' (sim/native.py NativeBatch on `pack`, the pack exported
        from these pool and archive files, which open_pack checks; it writes its slice of the buffer in place)."""
        if engine not in ('cpu', 'native') or (engine == 'native') != (pack is not None):
            raise ValueError(f"engine 'cpu', or 'native' with a pack, not {engine!r} with pack {pack!r}")
        from sim.combat import task_spec
        goals = goal_set(goals)
        if n % len(goals) or n % workers:
            raise ValueError(f'{n} worlds: a multiple of {len(goals)} goals, split evenly over {workers} workers')
        self.n, self.workers, self.horizon, self.goals = n, workers, horizon, goals
        spec = task_spec(task)
        self.task, self.horizons, self.targets = task, spec['horizons'], spec['targets']
        _, size = views(None, n)
        self._block = shared_memory.SharedMemory(create=True, size=size)
        self.v, _ = views(self._block.buf, n)
        ctx = mp.get_context(context)
        per, self._conns, self._processes = n // workers, [], []
        for k in range(workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(k * per, per, n, self._block.name, tables, str(pool), horizon, task, child,
                                                        None if search_pool is None else str(search_pool), goals, engine,
                                                        None if pack is None else str(pack)), daemon=True)
            process.start()
            child.close()
            self._conns.append(parent)
            self._processes.append(process)
        self._release = weakref.finalize(self, _release, self._block, self._processes, self._conns)
        self.evidence = []
        self.startup_seconds = max(self._gather('ready'))
        self.state, self._steps = None, []

    def _gather(self, expected='ok'):
        replies = [conn.recv() for conn in self._conns]
        errors = [r for r in replies if r[0] == 'error']
        if errors:
            self.evidence = [row for r in errors for row in r[2]]
            raise RuntimeError('; '.join(r[1] for r in errors))
        if any(r[0] != expected for r in replies):
            raise RuntimeError(f'unexpected worker replies: {[r[0] for r in replies]}')
        self.evidence += [row for r in replies if len(r) > 2 for row in r[2]]
        return [r[1] for r in replies]

    def _obs(self):
        return {k: self.v[f'obs/{k}'] for k in OBSERVATION_SHAPES}

    def reset(self, key):
        for conn in self._conns:
            conn.send(('reset', int(key)))
        self._gather()
        self.state = dict(key=int(key), steps=0)
        return dict(self.state), self._obs()

    def step(self, state, action):
        if state != self.state:
            raise ValueError('a state of another reset or step')
        self.v['action'][:] = np.asarray(action, np.int32)
        t = time.perf_counter()
        for conn in self._conns:
            conn.send(('step', None))
        busy = self._gather()
        self._steps.append((time.perf_counter() - t, busy))
        self.state = dict(self.state, steps=self.state['steps'] + 1)
        info = {k.split('/', 1)[1]: v for k, v in self.v.items() if k.startswith('info/')}
        return dict(self.state), self._obs(), self.v['reward'], self.v['done'], info

    def timing(self):
        """The steps since the last call: wall seconds, the slowest and the mean worker's busy seconds, and the shares of
        the wall time the workers waited for the slowest (1 - mean busy / wall) and spent outside every worker
        (communication: 1 - slowest busy / wall). Clears the record."""
        steps, self._steps = self._steps, []
        if not steps:
            return None
        wall = sum(w for w, _ in steps)
        slowest = sum(max(b) for _, b in steps)
        mean = sum(sum(b) / len(b) for _, b in steps)
        return dict(steps=len(steps), workers=self.workers, wall_seconds=wall, slowest_busy_seconds=slowest,
                    mean_busy_seconds=mean, waiting_share=1 - mean / wall, communication_share=1 - slowest / wall)

    def close(self):
        self.v = {}
        self._release()
