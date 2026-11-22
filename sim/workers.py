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
world it reported unsupported, or its settings while it is built) answers with the error and its evidence, and the call
raises once every worker has answered. Each reply is checked against the buffer before it is written: every field
present, in the field's shape and dtype, so no field keeps an earlier step's value.
The returned arrays are views of the shared buffer, valid until the next call: a caller that keeps them copies them.

The observations follow the table of shapes the batch declares (sim/spec.py): today's obs-v5 table for the goal tasks, and
for the goal-free task (FREE_PLAY_TASK) the table given to CpuWorkers, obs-v6's. A worker
refuses a batch whose observations depart from the table the buffer was allocated for, or that declares another
observation version. The free-play batch has no goal and adds info['last_obs'], each ended life's observation at its end
(sim/spec.py), which travels through the buffer as `last_obs/NAME` fields of the same table; a cut life's must offer a
legal action, since the learner values it. Each free-play worker reports its batch's settings in effect (BATCH_SETTINGS)
when it is ready, and CpuWorkers keeps them (settings_in_effect) for the run's record. On either engine: the C engine's
free-play batch (sim/native.py NativeBatch) takes the same settings, writes the same fields of its slice itself and
checks its observations and last_obs as _write checks a CpuBatch reply (NativeBatch.check).
"""
from copy import deepcopy
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import weakref

import numpy as np

from baseline.resource_task import GOALS
from item_contract import ITEM_V2, ITEM_V3
from obs import OBSERVATION
from sim.spec import HORIZON_TICKS, INFO_FIELDS, OBSERVATION_SHAPES, OBSERVATIONS, TERMINAL_HORIZON, dtype, goal_set
from wide_contract import ITEM_V4, OBSERVATION_V7, WIDE_SHAPES
from sim.task import FAMILIES, FREE_PLAY

# The goal-free task's name (sim/task.py FREE_PLAY): no goal, and lives cut by their drawn length bootstrap
# (sim/learn.py).
FREE_PLAY_TASK = FREE_PLAY['version']
# free-play-v2: free-play-v1, and the transition that ends a life by the engine's death also pays -death_price levels,
# a declared constant of the version. free-play-v3: free-play-v1 paid over every stat of the engine's ledger (the game's
# total level; `paid` 'all'), with one tinderbox more in the start kit (`kit_extra`). Defined here, not in sim/task.py,
# whose text the C engine's packs pin (sim/native.py ORACLE). Built for the engines named (the real engine,
# sim/real.py); every other engine refuses them by name (refuse_real_only).
FREE_PLAY_V2 = dict(FREE_PLAY, version='free-play-v2', death_price=50.0, engines=('real',),
                    rule='a price of 50 levels at each death')
FREE_PLAY_V3 = dict(FREE_PLAY, version='free-play-v3', skills="every stat of the engine's ledger", paid='all',
                    seven=FREE_PLAY['skills'], kit_extra=(('tinderbox', 1),), engines=('real',),
                    rule="every stat of the engine's ledger paid and a tinderbox in the start kit")
# free-play-v10: free-play-v3 (every stat paid, the tinderbox kit, no inner pay) on the item contract item-v3, the page
# of kinds: the pack's page lists one entry for each kind of item with the kind's total. The item contract is an
# attribute of a version (`items`; item-v2 when none is named), which the reset cfg of each life carries.
FREE_PLAY_V10 = dict(FREE_PLAY_V3, version='free-play-v10', items=ITEM_V3,
                     rule='free-play-v3 on the page of kinds (item-v3): one page entry for each kind of item in the pack, '
                          'with its total')
# free-play-v11: free-play-v10 on the wide contract (rl/wide_contract.py): the page of 16 kinds (item-v4), act-v4's 128
# rows, mask-v6 and obs-v7. The real engine's alone, as v10 (`engines`); a free-play-v3 or v10 policy reaches it by the
# tested conversion alone (baseline/policy.py widen_state).
FREE_PLAY_V11 = dict(FREE_PLAY_V10, version='free-play-v11', items=ITEM_V4,
                     rule='free-play-v10 on the page of 16 kinds (item-v4) with 128 action rows (act-v4, mask-v6, obs-v7)')
# free-play-v12: free-play-v10 with each life started at its start record's drawn hitpoints (`start_hp`; the reset's
# startHp, as the simulator's worlds start a life). Real engine only, the page of kinds (item-v3).
FREE_PLAY_V12 = dict(FREE_PLAY_V10, version='free-play-v12', start_hp='drawn',
                     rule="free-play-v10 with each life started at its start record's drawn hitpoints")
# free-play-v14 (longer lives, stage 2): free-play-v12 with the start rule old-life-v1. Every agent of odd index
# (sim/real.py: agent j is slot j mod N of world j // N, fixed for the run and its legs) starts every life with the
# old-life profile, the levels that the routine reaches in a life of about 100,000 ticks; every other draw is its
# record's own, by stage 1's profile function, which sim/real.py applies (the tile, the weapon, the food, the worn draw,
# the levels of every stat not named; the hitpoints at the arm the profile's hitpoints level less the record's own
# deficit, never under 1). The agents of even index start as drawn, as in free-play-v12, from the same record streams.
# The pay and the checks are v12's. Real engine only, the page of kinds (item-v3).
OLD_LIFE = dict(name='old-life-v1', agents='odd', levels=dict(attack=31, defence=30, strength=37, hitpoints=34,
                                                             woodcutting=50, firemaking=52))
FREE_PLAY_V14 = dict(FREE_PLAY_V12, version='free-play-v14', start_rule=OLD_LIFE,
                     rule='free-play-v12 with the start rule old-life-v1: every agent of odd index starts every life with '
                          'woodcutting 50, firemaking 52, attack 31, strength 37, defence 30 and hitpoints 34')
# free-play-v15 (directed exploration: a pay for XP of a rare kind): free-play-v14 with the pay rare-xp-v2 added to the
# learner's reward (sim/real.py RareXp): a decision whose XP rose in a stat s pays 1 / (1 + R / 10) levels for s, R the
# decisions of the run with XP in s among its last 131,072 decisions before it. Stored apart from the levels, as the
# inner pays are; the levels' pay and every read of it stay v14's. Real engine only.
RARE_XP = 'rare-xp-v2'
FREE_PLAY_V15 = dict(FREE_PLAY_V14, version='free-play-v15', inner=RARE_XP,
                     rule='free-play-v14 with the pay rare-xp-v2: 1 / (1 + R / 10) levels for each stat whose XP a '
                          'decision raised, R the decisions with XP in it among the run\'s last 131,072')
(FREE_PLAY_V2_TASK, FREE_PLAY_V3_TASK, FREE_PLAY_V10_TASK, FREE_PLAY_V11_TASK, FREE_PLAY_V12_TASK, FREE_PLAY_V14_TASK,
 FREE_PLAY_V15_TASK) = (v['version'] for v in (FREE_PLAY_V2, FREE_PLAY_V3, FREE_PLAY_V10, FREE_PLAY_V11, FREE_PLAY_V12,
                                               FREE_PLAY_V14, FREE_PLAY_V15))
# The study arms removed from the code after their studies closed (21 November 2026): each version and what it played.
# A checkpoint, a launch or a leg that names one is refused by name (refuse_removed), and no kept version plays it.
# free-play-v6 (v4 with the health pay), which no run used, was removed before.
REMOVED = {'free-play-v4': dict(rule='free-play-v3 with the inner pay first-item-v1'),
           'free-play-v5': dict(rule='free-play-v3 with the health pay health-drive-v1'),
           'free-play-v7': dict(rule='free-play-v3 with 3 raw chicken in the start kit'),
           'free-play-v8': dict(rule='free-play-v7 with the use pay first-use-v1'),
           'free-play-v9': dict(rule='free-play-v3 with the use pay first-use-v1'),
           'free-play-v13': dict(rule='free-play-v12 with the cost of time hurt health-cost-v1')}


def refuse_removed(task, what):
    """Refuse a removed study arm (REMOVED) by name: `what` is the thing (a checkpoint, a launch) that names `task`."""
    if task in REMOVED:
        arm = REMOVED[task]
        raise ValueError(f"{what} is of {task} ({arm['rule']}), a study arm removed from the code after its study "
                         'closed: no kept version plays it')
# village-world-v3 (19 November 2026): a world of the real engine only, village-world-v2's rules and record draw
# (`based_on`) with two shore sites beside its six sites, each shore site with its own start radius (8 and 3 tiles,
# chosen from the start count with the reset's jitter: no start within 15 tiles of an npc that attacks first; every
# other site takes the batch's radius). Sites are names here; their tiles are the site table's, which the real
# engine's batch reads (sim/real.py). Defined here, not among
# sim/combat.py VILLAGES (an ORACLE file, the worlds the simulator engines play), so the training command refuses it by
# name on those engines. VILLAGE_SITES: the six, in the site table's order (which a real-engine batch of village-world-v2
# drew from before the shore sites were added to that table).
VILLAGE_SITES = ('port_sarim_north', 'draynor_west', 'falador_farm_west', 'rimmington_coast', 'rimmington_mine_west',
                 'draynor_manor_south')
SHORE_SITES = ('rimmington_shore_east', 'rimmington_shore_west')
VILLAGE_V3 = dict(version='village-world-v3', based_on='village-world-v2', engines=('real',),
                  sites=VILLAGE_SITES + SHORE_SITES, site_radius=dict(rimmington_shore_east=8, rimmington_shore_west=3),
                  rule='village-world-v2 with two shore sites, rimmington_shore_east (radius 8) and rimmington_shore_west '
                       '(radius 3)')
REAL_WORLDS = {VILLAGE_V3['version']: VILLAGE_V3}
FREE_PLAY_TASKS = (FREE_PLAY_TASK, FREE_PLAY_V2_TASK, FREE_PLAY_V3_TASK, FREE_PLAY_V10_TASK, FREE_PLAY_V11_TASK,
                   FREE_PLAY_V12_TASK, FREE_PLAY_V14_TASK, FREE_PLAY_V15_TASK)
FREE_PLAY_VERSIONS = {v['version']: v for v in (FREE_PLAY, FREE_PLAY_V2, FREE_PLAY_V3, FREE_PLAY_V10, FREE_PLAY_V11,
                                                FREE_PLAY_V12, FREE_PLAY_V14, FREE_PLAY_V15)}
ENGINE_NAMES = dict(real='real engine', cpu='Python reference', native='C engine', fake='fake environment')


def death_price(task):
    """The levels a free-play version takes at a death: 0 for free-play-v1 (nothing taken back), FREE_PLAY_V2's for v2."""
    return FREE_PLAY_VERSIONS[task].get('death_price', 0.0)


def task_batch(task):
    """A free-play version's own settings of the real engine's batch (sim/real.py RealBatch): its death price, whether
    every stat of the ledger is paid, the items its start kit holds beside each start record's (free-play-v3 and every
    later version a tinderbox), its item contract (item-v2 unless the version names another: free-play-v10's item-v3),
    free-play-v15's inner pay rare-xp-v2 (inner), free-play-v12's, v14's and v15's drawn start hitpoints (start_hp) and
    free-play-v14's and v15's start rule old-life-v1 (start_rule); the other versions name none of these, so their
    settings are as before."""
    version = FREE_PLAY_VERSIONS[task]
    return dict(death_price=death_price(task), all_skills=version.get('paid') == 'all', items=item_contract(task),
                kit_extra=tuple(version.get('kit_extra', ())), **({'inner': version['inner']} if 'inner' in version else {}),
                **({'start_hp': True} if version.get('start_hp') == 'drawn' else {}),
                **({'start_rule': deepcopy(version['start_rule'])} if 'start_rule' in version else {}))


def item_contract(task):
    """The item contract a free-play version plays under: its `items`, item-v2 when it names none."""
    return FREE_PLAY_VERSIONS[task].get('items', ITEM_V2)


def observation_table(items):
    """(version, table of shapes) of a free-play batch's observations under its item contract: obs-v7, obs-v6's table
    with WIDE_SHAPES, under item-v4 (free-play-v11); free-play's own (sim/task.py FREE_PLAY, obs-v6) otherwise."""
    if items == ITEM_V4:
        return OBSERVATION_V7, {**OBSERVATIONS[FREE_PLAY['observation']], **WIDE_SHAPES}
    return FREE_PLAY['observation'], OBSERVATIONS[FREE_PLAY['observation']]


def refuse_real_only(task, engine):
    """A free-play version on an engine its definition does not name (FREE_PLAY_V2, FREE_PLAY_V3 `engines`; `engine` a
    key of ENGINE_NAMES): refused by name, for no other engine pays it."""
    version = FREE_PLAY_VERSIONS.get(task, {})
    if 'engines' in version and engine not in version['engines']:
        raise ValueError(f'{task} ({version["rule"]}) is built for the '
                         f'{" and ".join(ENGINE_NAMES[e] for e in version["engines"])} only (sim.train --env real); the '
                         f'{ENGINE_NAMES[engine]} does not pay it')


# The free-play batch's settings in effect, as sim/vec.py CpuBatch and sim/native.py NativeBatch hold them: its world
# (sim/combat.py VILLAGES, the one its pool's layouts declare), observation version, XP rate, muggers, life-length
# distribution and agent settings with their limits (cfg). Each worker reports them when it is ready.
BATCH_SETTINGS = ('world', 'observation', 'xp_rate', 'muggers', 'life', 'cfg')


def fields(n, shapes=OBSERVATION_SHAPES, last_obs=False):
    """The shared buffer's arrays for n worlds: name -> (shape, dtype). shapes: the batch's table of observation shapes;
    last_obs: also each ended life's last observation (the free-play batch's info['last_obs']) in that table."""
    out = {f'obs/{k}': ((n, *shape), np.dtype(dtype(k))) for k, shape in shapes.items()}
    if last_obs:
        out.update({f'last_obs/{k}': ((n, *shape), np.dtype(dtype(k))) for k, shape in shapes.items()})
    out.update({f'info/{k}': ((n,), np.dtype(kind)) for k, kind in INFO_FIELDS.items()})
    from sim.vec import EPISODE_FACTS
    out.update({'info/made': ((n, len(FAMILIES)), np.dtype(np.int32)), 'info/kills': ((n,), np.dtype(np.int32)),
                'info/facts': ((n, len(EPISODE_FACTS)), np.dtype(np.int32)),
                'reward': ((n,), np.dtype(np.float32)), 'done': ((n,), np.dtype(bool)), 'action': ((n,), np.dtype(np.int32))})
    return out


def views(buffer, n, shapes=OBSERVATION_SHAPES, last_obs=False):
    """The arrays of fields(n, shapes, last_obs) over `buffer` (None: only the size), each 64-byte aligned; returns
    (views, size)."""
    out, offset = {}, 0
    for name, (shape, kind) in fields(n, shapes, last_obs).items():
        offset = -(-offset // 64) * 64
        if buffer is not None:
            out[name] = np.ndarray(shape, kind, buffer=buffer, offset=offset)
        offset += int(np.prod(shape)) * kind.itemsize
    return out, offset


def load_tables(path):
    from sim.data import Tables
    return Tables.load() if path is None else Tables.load(path, check_pins=False)


def _native(first, count, tables, pool, horizon, task, search_pool, goals, pack, v, mine, batch=None):
    """The worker's NativeBatch (sim/native.py) over the pack exported from these inputs, writing its slice in place:
    open_pack holds the pack's recorded inputs equal to the archive (`tables`, or with none given the one the Python
    reference would load, sim/build.py DATA), the pool and any supply pool, so a C-engine run trains on the inputs it
    names, as a Python reference run does. FREE_PLAY_TASK: no goal and no horizon of its own, and the batch's keyword
    settings `batch` as CpuBatch takes them; a setting that CpuBatch does not take (NativeBatch's allow_stale or threads)
    is refused, so both engines run a batch under the same settings."""
    from sim.build import DATA
    from sim.native import NativeBatch
    inputs = {name: path for name, path in (('tables', tables or DATA), ('pool', pool), ('supply_pool', search_pool)) if path}
    if task == FREE_PLAY_TASK:
        import inspect
        from sim.vec import CpuBatch
        given = dict(batch or {})
        foreign = sorted(set(given) - set(inspect.signature(CpuBatch).parameters))
        if foreign:
            raise ValueError(f'batch settings that the Python reference does not take: {foreign}')
        env = NativeBatch(pack, count, first=first, task=task, inputs=inputs, **given)
    else:
        env = NativeBatch(pack, count, first=first, task=task, goals=goals, horizon=horizon, inputs=inputs)
    env.bind({name: array[mine] for name, array in v.items()})
    return env


def check_layout(arrays, shapes, count, what):
    """Raise unless `arrays` holds exactly the table's keys, each [count, *shape] in its sim/spec.py dtype."""
    if set(arrays) != set(shapes):
        raise ValueError(f'{what} departs from the declared table: {sorted(set(arrays) ^ set(shapes))}')
    for k, shape in shapes.items():
        value = arrays[k]
        if value.shape != (count, *shape) or value.dtype != dtype(k):
            raise ValueError(f'{what} {k}: {value.dtype}{value.shape}, the table has {np.dtype(dtype(k))}{(count, *shape)}')


def _write(v, mine, shapes, last_obs, obs, step=None):
    """A CpuBatch reply into the worker's slice of the buffer: its observations, and after a step (reward, done, info) its
    rewards, done flags and info, the free-play batch's last_obs as last_obs/NAME. All of it is checked first: the
    observation tables, every info field of the buffer and no other, each array in its field's shape and dtype, and a
    legal action in the last observation of each life cut by its length (the learner values it)."""
    count = mine.stop - mine.start
    check_layout(obs, shapes, count, 'the observation')
    if step is not None:
        info = dict(step[2])
        ended = info.pop('last_obs', None)
        if (ended is not None) != last_obs:
            raise ValueError('info carries last_obs exactly when the batch is the free-play one: '
                             f'{"missing" if last_obs else "unexpected"}')
        fields = {k.split('/', 1)[1] for k in v if k.startswith('info/')}
        if set(info) != fields:
            raise ValueError(f'info departs from the buffer\'s fields: {sorted(set(info) ^ fields)}')
        for name, value in dict(reward=step[0], done=step[1], **{f'info/{k}': x for k, x in info.items()}).items():
            value, slot = np.asarray(value), v[name][mine]
            if value.shape != slot.shape or value.dtype != slot.dtype:
                raise ValueError(f'{name}: {value.dtype}{value.shape}, the buffer has {slot.dtype}{slot.shape}')
        if ended is not None:
            check_layout(ended, shapes, count, 'last_obs')
            cut = np.asarray(info['terminal']) == TERMINAL_HORIZON
            if not ended['mask'][cut].any(-1).all():
                raise ValueError('the last observation of a life cut by its length offers no legal action')
    for k, value in obs.items():
        v[f'obs/{k}'][mine] = value
    if step is not None:
        v['reward'][mine], v['done'][mine] = step[0], step[1]
        for k, value in info.items():
            v[f'info/{k}'][mine] = value
        for k, value in (ended or {}).items():
            v[f'last_obs/{k}'][mine] = value


def in_effect(env, observation, free_play):
    """The batch's settings in effect (BATCH_SETTINGS, in JSON's types) for the free-play batch, None for a goal task's;
    a batch that declares another observation version than the buffer's (`observation`) is refused."""
    if getattr(env, 'observation', observation) != observation:
        raise ValueError(f'the batch declares observation {env.observation}, the learner records {observation}')
    return json.loads(json.dumps({k: getattr(env, k) for k in BATCH_SETTINGS})) if free_play else None


def _worker(first, count, n, name, tables, pool, horizon, task, conn, search_pool=None, goals=GOALS, engine='cpu',
            pack=None, shapes=OBSERVATION_SHAPES, batch=None, observation=OBSERVATION):
    """One worker's batch over its slice. shapes, observation: the table the buffer holds and its version; batch: the
    free-play batch's own keyword settings (CpuBatch takes horizon and goals for the goal tasks only). A batch that cannot
    be built answers 'error' with the reason; else 'ready' with its settings in effect (in_effect)."""
    from sim.vec import CpuBatch
    from sim.world import load_pool
    block = shared_memory.SharedMemory(name=name)
    free_play = task == FREE_PLAY_TASK
    v = env = None
    try:
        v, _ = views(block.buf, n, shapes, free_play)
        mine = slice(first, first + count)
        try:
            if engine == 'native':
                env = _native(first, count, tables, pool, horizon, task, search_pool, goals, pack, v, mine, batch)
            else:
                layouts, derived = load_pool(pool)
                search = None if search_pool is None else load_pool(search_pool)
                given = dict(batch or {}) if free_play else dict(horizon=horizon, goals=goals)
                env = CpuBatch(load_tables(tables), layouts, count, derived=derived, first=first, task=task, search=search,
                               **given)
            settings = in_effect(env, observation, free_play)
        except Exception as error:
            conn.send(('error', f'worker for slots {first}-{first + count - 1} could not build its batch: '
                                f'{type(error).__name__}: {error}', []))
            return
        conn.send(('ready', dict(settings=settings)))
        state, reported = None, 0
        while True:
            command, argument = conn.recv()
            if command == 'close':
                break
            t = time.perf_counter()
            try:
                if engine == 'native':  # the native batch writes its slice of the views itself
                    if command == 'reset':
                        env.reset_into(argument)
                    else:
                        env.step_into(v['action'][mine])
                elif command == 'reset':
                    state, obs = env.reset(argument)
                    _write(v, mine, shapes, free_play, obs)
                else:
                    state, obs, reward, done, info = env.step(state, v['action'][mine])
                    _write(v, mine, shapes, free_play, obs, (reward, done, info))
            except Exception as error:
                conn.send(('error', f'worker for slots {first}-{first + count - 1}: {type(error).__name__}: {error}',
                           env.evidence))
                continue
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
                 search_pool=None, goals=GOALS, engine='cpu', pack=None, *, observation=None, shapes=None, batch=None):
        """task: the four-goal task version (sim/combat.py task_spec); `horizons` and `targets` give each goal's.
        search_pool: goal-activity-v4's search layouts (sim/search.py), a pool file each worker loads beside `pool`.
        goals: the declared goal set (sim/spec.py goal_set): all four, or one goal in every world slot.
        engine: 'cpu' (CpuBatch, the default) or 'native' (sim/native.py NativeBatch on `pack`, the pack exported
        from these pool and archive files, the archive's default too, which open_pack checks; it writes its slice of the
        buffer in place).
        observation, shapes: the version and table of shapes the batch's observations follow (default: obs-v5's,
        OBSERVATION_SHAPES, the goal tasks'). task FREE_PLAY_TASK needs both (obs-v6's), takes no goals and no horizon of
        its own, gives last_obs, and passes `batch` (its keyword settings) to each worker's batch, CpuBatch or
        NativeBatch (_native); every worker must report the same settings in effect (settings_in_effect; None for a goal
        task)."""
        if engine not in ('cpu', 'native') or (engine == 'native') != (pack is not None):
            raise ValueError(f"engine 'cpu', or 'native' with a pack, not {engine!r} with pack {pack!r}")
        refuse_real_only(task, engine)
        free_play = task == FREE_PLAY_TASK
        if free_play:
            if observation is None or shapes is None or goals not in (None, ()):
                raise ValueError(f'{FREE_PLAY_TASK} runs with its declared observation version and table, and no goals')
            goals, self.horizons, self.targets = (), None, None
            if n % workers:
                raise ValueError(f'{n} worlds split evenly over {workers} workers')
        else:
            if batch:
                raise ValueError(f'batch settings are the {FREE_PLAY_TASK} batch\'s')
            from sim.combat import task_spec
            goals = goal_set(goals)
            if n % len(goals) or n % workers:
                raise ValueError(f'{n} worlds: a multiple of {len(goals)} goals, split evenly over {workers} workers')
            spec = task_spec(task)
            self.horizons, self.targets = spec['horizons'], spec['targets']
        if (observation is None) != (shapes is None):
            raise ValueError('an observation version comes with its table of shapes')
        self.observation = OBSERVATION if observation is None else observation
        self.shapes = OBSERVATION_SHAPES if shapes is None else dict(shapes)
        self.last_obs = free_play
        self.n, self.workers, self.horizon, self.goals, self.task = n, workers, horizon, goals, task
        _, size = views(None, n, self.shapes, free_play)
        self._block = shared_memory.SharedMemory(create=True, size=size)
        self.v, _ = views(self._block.buf, n, self.shapes, free_play)
        ctx = mp.get_context(context)
        per, self._conns, self._processes = n // workers, [], []
        for k in range(workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(k * per, per, n, self._block.name, tables, str(pool), horizon, task, child,
                                                        None if search_pool is None else str(search_pool), goals, engine,
                                                        None if pack is None else str(pack), self.shapes, batch,
                                                        self.observation), daemon=True)
            process.start()
            child.close()
            self._conns.append(parent)
            self._processes.append(process)
        self._release = weakref.finalize(self, _release, self._block, self._processes, self._conns)
        self.evidence = []
        try:
            ready = self._gather('ready')
            settings = [r['settings'] for r in ready]
            if any(s != settings[0] for s in settings):
                raise RuntimeError(f'the workers\' batches run under different settings: {settings}')
        except BaseException:
            self._release()
            raise
        self.settings_in_effect = settings[0]
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

    def _obs(self, part='obs'):
        return {k: self.v[f'{part}/{k}'] for k in self.shapes}

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
        if self.last_obs:
            info['last_obs'] = self._obs('last_obs')
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
