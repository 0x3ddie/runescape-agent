"""free-play-v1 in the C engine (sim/native.py NativeBatch, sim/native/task.c): the native batch reproduces CpuBatch's
free-play batch step by step and tick by tick (the harness, sim/native_harness.py) at XP rates 1 and 25, with and
without muggers, for lives cut by their length and ended by a death; a life with more events than the C buffer holds
settles in steps; late log1p features fill obs and last_obs alike (a test build whose tables stop at count 1); the
batch has CpuBatch's interface, takes and refuses CpuBatch's settings, and checks each ended life's last observation;
a divergence in a scalar info field is reported; thread counts change nothing; and the C XpCheck is the oracle's
(sim/audit.py) on generated event streams, its refusals and the XP cap included. Private data: the tests read the
archive and the supply pool in place and skip when one is absent (a skip is not a pass); their pack is exported here
from the training pool's 8 compact layouts with the suite's synthetic name table (as the harness tests' packs are), so
both engines project the same name rows."""
from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import native, native_harness as H
from sim.build import DATA

ROOT = DATA.parents[3]
SUPPLY = ROOT / 'logs/supply/pools/train-128.npz'
FREE = 'free-play-v1'
SHORT = ('uniform', 100, 800)


@pytest.fixture(scope='module')
def exported(name_table_path, tmp_path_factory):
    """(pool, pack) files: the training pool's 8 compact layouts as a plain pool and its pack without trees (C searches
    routes live, Python computes them as queried) under the synthetic name table."""
    if not (DATA.exists() and SUPPLY.exists()):
        pytest.skip('private archive or supply pool absent')
    from names import NameTable
    from sim import search, world as W
    from sim.native_export import export
    from sim.workers import load_tables
    tables, work = load_tables(DATA), tmp_path_factory.mktemp('free-play-pack')
    layouts = [W.generate(seed, tables) for seed in W.split_seeds('train', 8)]
    pool = W.save_pool(work / 'pool.npz', layouts)
    supply = W.save_pool(work / 'supply.npz', W.load_pool(SUPPLY)[0][:1], version=search.SUPPLY['world'])
    return pool, export(work / 'free.pack', pool, supply, DATA, names=NameTable(name_table_path), trees=False,
                        variants=2, log=lambda _: None)


@pytest.fixture(scope='module')
def short_pack(name_table_path, tmp_path_factory):
    """(tables, layouts, pack): a training layout and a copy of it whose mugger draw always fails, exported as the
    tests' packs are. The copy keeps one start tile and the walkable tiles within 5 of the goblins' centre, so no tile
    of its region is 6 to 12 from the centre and 7 from the start (sim/combat.py draw_combat): 0 tiles for 1 mugger."""
    if not (DATA.exists() and SUPPLY.exists()):
        pytest.skip('private archive or supply pool absent')
    from dataclasses import replace
    from names import NameTable
    from sim import combat, search, world as W
    from sim.native_export import export
    from sim.workers import load_tables
    tables, work = load_tables(DATA), tmp_path_factory.mktemp('short-pack')
    good = W.generate(W.split_seeds('train', 1)[0], tables)
    cx, cz = np.mean(combat.camp(good, tables), axis=0)
    x, z = np.indices(good.walkable.shape)
    start = np.zeros_like(good.start)
    start[tuple(np.argwhere(good.start)[0])] = True
    near = np.maximum(np.abs(x - cx), np.abs(z - cz)) <= 5
    short = replace(good, seed=good.seed + 900_000, walkable=good.walkable & (near | start), start=start)
    pool = W.save_pool(work / 'pool.npz', [good, short])
    supply = W.save_pool(work / 'supply.npz', W.load_pool(SUPPLY)[0][:1], version=search.SUPPLY['world'])
    pack = export(work / 'short.pack', pool, supply, DATA, names=NameTable(name_table_path), trees=False, variants=1,
                  log=lambda _: None)
    return tables, W.load_pool(pool)[0], native.open_pack(pack, names=NameTable(name_table_path))


@pytest.fixture(scope='module')
def inputs(exported, name_table_path):
    """(tables, layouts, derived None, pack) of the exported files."""
    from names import NameTable
    from sim import world as W
    from sim.workers import load_tables
    pool, pack = exported
    return load_tables(DATA), W.load_pool(pool)[0], None, native.open_pack(pack, names=NameTable(name_table_path))


def sides(inputs, n, scenario=None, first=0, **settings):
    from sim.vec import CpuBatch
    tables, layouts, derived, pack = inputs
    env = CpuBatch(tables, layouts, n, derived=derived, first=first, task=FREE, **settings)
    a = H.PythonSide(env, canon=H.Canon(H.PackStrings(pack)), scenario=scenario)
    return a, H.NativeSide(native.NativeBatch(pack, n, first=first, task=FREE, **settings), scenario=scenario, python=a)


class Ends:
    """observe(): each ended life's terminal code, ticks and the Python side's events from arm (len(world.events)
    at the step before its end, a lower bound)."""

    def __init__(self, side):
        self.side, self.rows, self.events = side, [], [0] * side.n

    def __call__(self, step, out_a, out_b):
        if step < 0:
            return
        _, _, _, done, info = out_a
        for b in range(self.side.n):
            if done[b]:
                self.rows.append((int(info['terminal'][b]), int(info['episode_ticks'][b]), self.events[b]))
        for b, task in enumerate(self.side.env.tasks):
            self.events[b] = len(task.world.events) - task.event_start


@pytest.mark.parametrize('xp_rate,muggers', [(1, False), (1, True), (25, False), (25, True)])
def test_free_play_is_equal_step_by_step_and_tick_by_tick(inputs, xp_rate, muggers):
    """Random legal play on short lives: observations (obs-v6), reward bits, done, every info field (last_obs, goal -1,
    made, kills, facts), each start and every tick's record, draws and events (xp and hit) alike."""
    a, b = sides(inputs, 4, xp_rate=xp_rate, muggers=muggers, life=SHORT)
    ends = Ends(a)
    result = H.run(a, b, H.RandomLegal(11), 11, 160, observe=ends)
    assert result.divergence is None, result.divergence.text()
    assert not result.gap and result.traced and result.steps == 160, (result.gap, result.stopped)
    assert len(ends.rows) >= 4 and any(code == 1 for code, _, _ in ends.rows), ends.rows


def test_a_shared_check_failure_in_a_scenario_step_where_a_lower_life_ended_stops_alike(inputs):
    """Scenario sides (the C side begins each new life from the Python side's start): both engines raise one check
    failure at world 3 of a step in which a lower world's life ended (the oracle's text and its cut, as C words it). The
    run stops alike: the C side begins that lower world's next life before it raises, as the Python side's step began it,
    so the step's tick rows agree (before, the C side held no setup rows for it: a false 'ticks' divergence)."""
    from sim.audit import AuditError
    oracle, cut = "hit at tick 9 ({'npc': 1}) has no xp event for stats [0, 3]", 'hit at tick 9 has no xp event'
    settings = dict(scenario='freehurt', xp_rate=1, muggers=False, life=SHORT)
    a, b, ends = *sides(inputs, 4, **settings), []
    H.run(a, b, H.RandomLegal(13), 13, 150,
          observe=lambda step, out_a, out_b: step >= 0 and ends.append(np.asarray(out_a[3], bool)))
    step = next(k for k, done in enumerate(ends) if done[:3].any() and not done[3])  # a lower life ends, world 3's not
    a, b = sides(inputs, 4, **settings)
    py_step, c_step = a.step, b.batch.step_into

    def oracle_step(state, action):
        if a.step_index == step:  # world 3's life raises after its cycles, the lower worlds stepped before it
            task = a.env.tasks[3]
            inner = task.step

            def raising(act):
                inner(act)
                raise AuditError(oracle)
            task.step = raising
        return py_step(state, action)

    def native_step(action, check=True):
        c_step(action, check=check)
        if b.step_index == step:
            error = AuditError(cut)
            error.world = 3  # as NativeBatch names a C error's failing world
            raise error
    a.step, b.batch.step_into = oracle_step, native_step
    result = H.run(a, b, H.RandomLegal(13), 13, 150)
    assert result.divergence is None, result.divergence.text()
    assert result.steps == step and result.stopped.endswith(f'(C text: {cut})'), (step, result.stopped)


def test_deaths_and_cuts_end_lives_alike(inputs):
    """A hurt start among muggers under idle play: lives end by a death (terminal 2) and by their length (terminal 1),
    alike, and the next life begins alike."""
    a, b = sides(inputs, 4, scenario='freehurt', xp_rate=1, muggers=True, life=('uniform', 40, 300))
    ends = Ends(a)
    result = H.run(a, b, H.make_source('idle', 5), 5, 300, observe=ends)
    assert result.divergence is None, result.divergence.text()
    codes = {code for code, _, _ in ends.rows}
    assert codes == {1, 2}, ends.rows


def test_the_controls_play_alike(inputs):
    """sim/free.py's gatherers and fighter (their drops, page turns, fights and style changes) at 25x, where levels and
    unlocks come fast: every production's and hit's XP event with its cause (a mined rock's type before it depletes,
    the tree's, the fish's) alike, tick by tick."""
    a, b = sides(inputs, 4, xp_rate=25, muggers=False, life=('fixed', 2500))
    result = H.run(a, b, H.ControlSource(a, ['mine', 'wood', 'fish', 'melee'], 9), 9, 350)
    assert result.divergence is None, result.divergence.text()
    assert result.traced and result.steps == 350, (result.gap, result.stopped)


def test_a_life_past_the_event_buffer_settles_in_steps(inputs):
    """The chickens control fills a life with events: one life of 9,000 ticks holds more than RS_MAX_EVENTS events,
    which the C engine settles step by step (the XP check and the pay) and drops; every tick's record (the event count
    and the last event included) stays the oracle's and the life is cut at its length."""
    a, b = sides(inputs, 1, xp_rate=1, muggers=False, life=('fixed', 9000))
    ends = Ends(a)
    result = H.run(a, b, H.ControlSource(a, ['chickens'], 3), 3, 2200, observe=ends)
    assert result.divergence is None, result.divergence.text()
    cap = native.load().constants()['RS_MAX_EVENTS']
    assert ends.rows and ends.rows[0][:2] == (1, 9000) and ends.rows[0][2] > cap, (ends.rows, cap)


def outcome(make):
    """'ok', or the class and text of what building a batch, reset(4) and three steps of legal play raised."""
    try:
        env = make()
        state, obs = env.reset(4)
        for _ in range(3):
            state, obs, *_ = env.step(state, np.array([np.flatnonzero(row)[0] for row in obs['mask']], np.int32))
    except Exception as error:
        return type(error).__name__, str(error)
    return 'ok'


def test_the_native_free_play_batch_has_cpubatch_s_interface(inputs):
    from baseline import config
    from sim.engine import XP_RATE
    from sim.spec import INFO_FIELDS, OBSERVATION_SHAPES_V6
    from sim.task import free_play_cfg
    from sim.vec import CpuBatch
    tables, layouts, derived, pack = inputs
    # each setting taken or refused alike (NativeBatch refuses at build what CpuBatch's World and FreePlayEpisode
    # refuse at its first life); TASK_CFG's limits (4,000 decisions, 12,000 ticks) can cut a 13,000-tick life
    for case in (dict(xp_rate=True), dict(xp_rate=np.int64(25)), dict(xp_rate=1.0), dict(xp_rate=25),
                 dict(cfg=config.TASK_CFG, life=('fixed', 13000)), dict(cfg=config.TASK_CFG), dict(goals=None),
                 dict(goals=()), dict(goals=('wood', 'mine', 'fish', 'melee')), dict(horizon=None), dict(horizon=600),
                 dict(world=None), dict(world='village-world-v1'), dict(world='village-world-v2'), dict(world='nowhere'),
                 dict(world='village-world-v2', muggers=True)):
        settings = dict(dict(muggers=False, life=('fixed', 120)), **case)
        got = (outcome(lambda: native.NativeBatch(pack, 2, task=FREE, **settings)),
               outcome(lambda: CpuBatch(tables, layouts, 2, derived=derived, task=FREE, **settings)))
        assert got[0] == got[1], (case, got)
    with pytest.raises(ValueError, match='n >= 1 worlds'):
        native.NativeBatch(pack, 0, task=FREE, muggers=False, life=SHORT)
    with pytest.raises(ValueError):  # CpuBatch refuses it at reset, stacking no world
        CpuBatch(tables, layouts, 0, derived=derived, task=FREE, muggers=False, life=SHORT).reset(4)
    with pytest.raises(ValueError, match='first >= 0'):  # CpuBatch takes first -1 (reported: sim/vec.py is an oracle)
        native.NativeBatch(pack, 2, first=-1, task=FREE, muggers=False, life=SHORT)
    for bad in (dict(muggers=None, life=SHORT), dict(muggers=False, life=None), dict(muggers=1, life=SHORT),
                dict(muggers=False, life=SHORT, xp_rate=5), dict(muggers=False, life=('uniform', 9, 3)),
                dict(muggers=False, life=SHORT, horizon=60), dict(muggers=False, life=SHORT, goals=('wood',))):
        with pytest.raises(ValueError):
            native.NativeBatch(pack, 2, task=FREE, **bad)
        with pytest.raises(ValueError):
            CpuBatch(tables, layouts, 2, derived=derived, task=FREE, **bad)
    for bad in (dict(xp_rate=1), dict(muggers=False), dict(life=SHORT), dict(world='village-world-v1')):
        with pytest.raises(ValueError) as refused:
            native.NativeBatch(pack, 2, task='goal-activity-v5', **bad)
        with pytest.raises(ValueError) as refused_cpu:  # CpuBatch's first check of a goal task, before its pools
            CpuBatch(tables, layouts, 2, derived=derived, task='goal-activity-v5', **bad)
        assert str(refused.value) == str(refused_cpu.value) == \
            "xp_rate, muggers, life and world are free-play-v1 settings, not goal-activity-v5's", bad
    settings = dict(muggers=True, life=('fixed', 120))
    nb = native.NativeBatch(pack, 3, task=FREE, **settings)
    cb = CpuBatch(tables, layouts, 3, derived=derived, task=FREE, **settings)
    for env in (nb, cb):
        assert (env.observation, env.shapes, env.xp_rate, env.muggers, env.life, env.cfg, env.world) == \
            ('obs-v6', OBSERVATION_SHAPES_V6, 1, True, ('fixed', 120), free_play_cfg(), 'village-world-v1')
    assert native.NativeBatch(pack, 1, task=FREE, muggers=False, life=SHORT, world='village-world-v1').world == \
        CpuBatch(tables, layouts, 1, derived=derived, task=FREE, muggers=False, life=SHORT, world='village-world-v1').world
    goal_batch = native.NativeBatch(pack, 1, task='goal-activity-v5')
    assert (goal_batch.observation, goal_batch.world) == ('obs-v5', None)
    assert native.NativeBatch(pack, 1, task=FREE, muggers=False, life=SHORT, xp_rate=25).xp_rate == 25 != XP_RATE - 1
    state_n, obs_n = nb.reset(4)
    state_c, obs_c = cb.reset(4)
    for k in range(80):
        action = np.array([np.flatnonzero(obs_c['mask'][w])[0] for w in range(3)], np.int32)
        out_n, out_c = nb.step(state_n, action), cb.step(state_c, action)
        assert len(out_n) == len(out_c) == 5
        (state_n, obs_n, reward, done, info), (state_c, obs_c, _, _, info_c) = out_n, out_c
        assert set(info) == set(info_c) == {*INFO_FIELDS, 'made', 'kills', 'facts', 'last_obs'}
        for name in info:
            if name == 'last_obs':
                for key, value in info['last_obs'].items():
                    assert (value.dtype, value.shape) == (info_c['last_obs'][key].dtype, info_c['last_obs'][key].shape)
                    assert not value[~done].any()  # zero in every row whose life did not end
            else:
                assert (info[name].dtype, info[name].shape) == (info_c[name].dtype, info_c[name].shape), name
        assert (info['goal'] == -1).all() and reward.dtype == np.float32 and done.dtype == bool
        assert set(obs_n) == set(OBSERVATION_SHAPES_V6)
        if done.any():
            assert any(info['last_obs']['mask'][w].any() for w in np.flatnonzero(done))
    assert nb.episodes == cb.episodes and max(nb.episodes) >= 1


def test_threads_change_no_free_play_output(inputs):
    pack = inputs[3]
    runs = []
    for threads in (1, 4):
        batch, source = native.NativeBatch(pack, 12, task=FREE, muggers=True, life=('uniform', 60, 300),
                                           threads=threads), H.Biased(2)
        state, obs = batch.reset(2)
        rows = [H.world_digest(obs, None, None, None, w) for w in range(12)]
        for step in range(120):
            state, obs, reward, done, info = batch.step(state, source(step, obs, range(12), None))
            rows += [H.world_digest(obs, reward, done, info, w) for w in range(12)]
        runs.append(rows)
    assert runs[0] == runs[1]


@pytest.mark.parametrize('name,changed', [('goal', np.int32(0)), ('episode_return', np.float32(0.5)), ('kills', np.int32(7))])
def test_a_scalar_info_field_that_differs_is_a_divergence_in_info(inputs, name, changed):
    """One world's value of a scalar info field changed on the native side at one decision: the run reports an info
    divergence there, naming the field, and its report says the world has no goal."""
    a, b = sides(inputs, 4, xp_rate=1, muggers=False, life=SHORT)
    step = b.step

    def changed_at(state, action):
        out = step(state, action)
        if b.step_index == 6:
            out[4][name][2] = changed
        return out
    b.step = changed_at
    d = H.run(a, b, H.RandomLegal(5), 5, 20).divergence
    assert d is not None and (d.where, d.local, d.step, d.goal) == ('info', 2, 6, 'no goal'), d and d.text()
    assert d.detail[0].startswith(f'info.{name}[]: ') and 'goal no goal,' in d.text(), d.detail


def test_the_native_batch_checks_each_ended_life_s_last_observation(inputs):
    """step_into's check covers last_obs as sim/workers.py _write checks a CpuBatch reply: a non-finite value in an
    ended life's row, and a mask with no legal action in a life cut by its length, are refused."""
    from sim.spec import TERMINAL_HORIZON
    pack = inputs[3]
    nb = native.NativeBatch(pack, 4, task=FREE, muggers=False, life=('fixed', 60))
    state, obs = nb.reset(3)
    for _ in range(200):
        state, obs, _, done, info = nb.step(state, np.array([np.flatnonzero(row)[0] for row in obs['mask']], np.int32))
        if (info['terminal'] == TERMINAL_HORIZON).any():
            break
    w = int(np.flatnonzero(info['terminal'] == TERMINAL_HORIZON)[0])
    nb.check(step=True)
    for key, change, why in (('mask', 0, 'offers no legal action'), ('prog_vec', np.nan, 'last_obs prog_vec: non-finite')):
        kept = nb.views[f'last_obs/{key}'][w].copy()
        nb.views[f'last_obs/{key}'][w] = change
        with pytest.raises(ValueError, match=why):
            nb.check(step=True)
        nb.views[f'last_obs/{key}'][w] = kept
    nb.check(step=True)
    fill = nb._fill_late

    def emptied(last=False):  # every last_obs row with no legal action, after the call's own late fill
        fill(last)
        if last:
            nb.views['last_obs/mask'][:] = 0
    nb._fill_late = emptied
    with pytest.raises(ValueError, match='offers no legal action'):
        for _ in range(200):
            state, obs, *_ = nb.step(state, np.array([np.flatnonzero(row)[0] for row in obs['mask']], np.int32))


LATE = """
import json, sys
import numpy as np
import pytest
from names import NameTable
from sim import native, native_harness as H, world as W
from sim.build import DATA
from sim.vec import CpuBatch
from sim.workers import load_tables
pool, pack_path, names, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
assert native.load().constants()['RS_LOG1P_COUNTS'] == 1
tables, layouts = load_tables(DATA), W.load_pool(pool)[0]
with pytest.raises(ValueError, match='other ids'):  # this build's rs.h contract is not the pack's
    native.open_pack(pack_path, names=NameTable(names))
pack = native.load().Pack(pack_path)  # mapped as is: RS_LOG1P_COUNTS, the one change, is in no pack row
out = dict(late=0, late_last=0, steps=0, divergence=None)
for settings, source in ((dict(xp_rate=25, muggers=True, life=('uniform', 100, 800)), 'random'),
                         (dict(xp_rate=25, muggers=False, life=('fixed', 1500)), 'controls')):
    env = CpuBatch(tables, layouts, 4, task='free-play-v1', **settings)
    a = H.PythonSide(env, canon=H.Canon(H.PackStrings(pack)))
    batch = native.NativeBatch(pack, 4, task='free-play-v1', **settings)
    b = H.NativeSide(batch, python=a)

    def observe(step, out_a, out_b):
        out['late'] += len(batch._batch.late())
        out['late_last'] += len(batch._batch.late_last()) if step >= 0 else 0
    play = H.RandomLegal(9) if source == 'random' else H.ControlSource(a, ['mine', 'wood', 'fish', 'melee'], 9)
    result = H.run(a, b, play, 9, steps, observe=observe)
    out['steps'] += result.steps
    if result.divergence is not None or result.gap or result.stopped:
        out['divergence'] = result.divergence.text() if result.divergence else result.gap or result.stopped
        break
print(json.dumps(out))
"""


def test_a_decision_where_three_skills_gain_pays_alike_in_python_and_c(exported):
    """free-play-v1 at 25x XP with muggers, world slot 6 of the training pool, key 82, actions 12, 9, 11, 0 (a
    divergence an earlier full gate found): its third decision raises three skills at once, and Python's sum() of the
    three fractional level gains (compensated) differs in its last bit from a plain left-to-right sum. Both engines pay
    it alike: paid_bits in every tick record, reward and return equal (the C twin sums as sum() does, pysum.h)."""
    import json
    pool, pack = exported
    inputs = H.Inputs(pool=str(pool), archive=str(DATA), task=FREE, against='native', pack=str(pack), xp_rate=25,
                      muggers=True, life=json.dumps(['uniform', 100, 1000]))
    a, b = H.build(inputs, 1, first=6)
    found = []

    def observe(step, out_a, out_b):
        task = a.env.tasks[0]
        if step >= 0:
            terms = [after - was for was, after in zip(seen[0], task.level)]
            plain = 0.0
            for term in terms:
                plain += term
            found.append((sum(1 for term in terms if term), sum(terms) != plain))
        seen[0] = list(task.level)
    seen = [None]
    result = H.run(a, b, H.Replay([[12], [9], [11], [0]]), 82, 4, observe=observe)
    assert result.divergence is None and result.steps == 4 and result.traced, result.divergence
    assert found[2] == (3, True), found  # three skills, and sum() is not the plain sum there


def test_late_log1p_features_fill_obs_and_last_obs_alike(exported, name_table_path, tmp_path):
    """A test build (SIM_NATIVE_SOURCE: a copy of sim/native/ whose log1p tables stop at count 1, in its own
    SIM_BUILD) makes every count of 1 or more a late feature, which NativeBatch writes after each call into obs and
    last_obs (_fill_late): random play among muggers and the controls at 25x, in lockstep with the oracle, with late
    features in both. No committed C file changes. The copy's rs.h contract differs from the pack's, so open_pack
    refuses the pack there (ids_hash covers every rs.h value) and the script maps it as it is."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    source = tmp_path / 'native'
    shutil.copytree(native.SOURCE, source)
    header = (source / 'rs.h').read_text()
    assert header.count('#define RS_LOG1P_COUNTS 65536\n') == 1
    (source / 'rs.h').write_text(header.replace('#define RS_LOG1P_COUNTS 65536\n', '#define RS_LOG1P_COUNTS 1\n'))
    (tmp_path / 'late.py').write_text(LATE)
    env = dict(os.environ, SIM_NATIVE_SOURCE=str(source), SIM_BUILD=str(tmp_path / 'build'), SIM_NATIVE_LTO='0',
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'rl')]))
    pool, pack = exported
    done = subprocess.run([sys.executable, str(tmp_path / 'late.py'), str(pool), str(pack), str(name_table_path), '300'],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['divergence'] is None, out['divergence']
    assert out['steps'] == 600 and out['late'] > 1000 and out['late_last'] > 10, out


def test_a_failed_draw_leaves_no_start_as_cpubatch_s(short_pack):
    """A world whose draw fails (the copy's mugger tiles run short: ValueError in both engines, the same text) has no
    start after it, as CpuBatch's starts[b] is None: at a reset (where every world's start is cleared first, so a world
    the failed reset never reached keeps no start of the reset before), and at the auto-reset after a life on the other
    layout (the C slot kept the ended life's start before)."""
    from sim.task import draw_start
    from sim.vec import CpuBatch
    tables, layouts, pack = short_pack
    drawn = lambda k, slot, e: draw_start(k, slot, e, layouts)['layout']
    good = next(k for k in range(10_000) if drawn(k, 0, 0) == drawn(k, 1, 0) == 0)
    first = next(k for k in range(10_000) if drawn(k, 0, 0) == 1)
    later = next(k for k in range(10_000) if (drawn(k, 0, 0), drawn(k, 0, 1), drawn(k, 1, 0)) == (0, 1, 0))
    settings = dict(muggers=True, life=('fixed', 40))
    for key in (first, later):
        got = []
        for env in (native.NativeBatch(pack, 2, task=FREE, **settings), CpuBatch(tables, layouts, 2, task=FREE, **settings)):
            env.reset(good)
            with pytest.raises(ValueError) as caught:
                state, obs = env.reset(key)
                for _ in range(400):  # until world 0's life ends and its next draw fails
                    state, obs, *_ = env.step(state, np.array([np.flatnonzero(row)[0] for row in obs['mask']], np.int32))
            got.append((str(caught.value), H._plain(env.starts), env.episodes))
        assert got[0] == got[1], (key, got)
        assert got[0][1][0] is None and 'tiles for 1 muggers' in got[0][0], got
        assert (got[0][1][1] is None) == (key == first) and got[0][2][0] == int(key == later), got


LIVES = (('fixed', 1), ('fixed', 36000), ('uniform', 100, 1000), ('uniform', 6000, 36000), ('uniform', 1, 150000))


def test_pack_episode_draw_of_free_play_equals_the_oracle(inputs, short_pack):
    """Pack.episode_draw's free-play branch (rs_draw_village, rs_village_npcs and the kit) against CpuBatch's draws
    (draw_village on the layout's region, draw_life, village_npcs, kit): the start, its length, the kit and the npc
    spawns alike for keyed (key, slot, episode), both mugger forms and five life forms, on the training pack and on the
    short pack, whose copy layout raises the same mugger shortage in both; muggers other than 0 and 1 are refused."""
    from sim.combat import draw_village, kit, outside, village_npcs
    from sim.task import draw_life, draw_start
    packs = [(inputs[0], inputs[1], inputs[3]), short_pack]
    regions = [[outside(layout) for layout in layouts] for _, layouts, _ in packs]
    seen = dict(drawn=0, refused=0)

    @settings(max_examples=300, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    @given(st.integers(0, 1), st.integers(-2 ** 63, 2 ** 63 - 1), st.integers(0, 2 ** 31 - 1),
           st.integers(0, 2 ** 31 - 1), st.booleans(), st.sampled_from(LIVES))
    def check(which, key, slot, episode, muggers, life):
        tables, layouts, pack = packs[which]
        k = draw_start(key, slot, episode, layouts)['layout']
        try:
            start = draw_village(key, slot, episode, layouts, tables, muggers, region=regions[which][k])
            start['length'] = draw_life(key, slot, episode, life)
            v = start['village']
            want = (start, [(tables.obj_id(name), count) for name, count in kit(start['weapon'], v['food'])],
                    [(n['nid'], n['type'], n['x'], n['z'], n['level'], n['size'])
                     for n in village_npcs(layouts[k], tables, v['muggers'])])
        except ValueError as error:
            want = ('ValueError', str(error))
        try:
            raw, kit_ids, spawns = pack.episode_draw(native.TASKS[FREE], 0, key, slot, episode, int(muggers),
                                                     int(life[0] == 'uniform'), life[1], life[-1])
            assert raw['kind'] == native.START_KINDS.index('village') and raw['rec'] == k
            got = (native.decode_start(raw), kit_ids, spawns)
        except native.load().Error as error:
            failed = native.as_python(error, pack)
            got = (type(failed).__name__, str(failed))
        assert got == want, (which, key, slot, episode, muggers, life)
        seen['refused' if want[0] == 'ValueError' else 'drawn'] += 1
    check()
    assert seen['drawn'] >= 100 and seen['refused'] >= 20, seen
    with pytest.raises(ValueError, match='muggers: 0 or 1'):
        inputs[3].episode_draw(native.TASKS[FREE], 0, 1, 0, 0, 2, 0, 50, 0)


def test_a_run_reports_the_most_ground_objs_of_a_world(inputs):
    """run() reports the most ground objs one Python world held at a tick's end (Result.most_objs), at least the most
    counted after each decision, and a gate case row carries it."""
    a, b = sides(inputs, 4, xp_rate=25, muggers=False, life=('fixed', 2500))
    most = [0]

    def observe(step, out_a, out_b):
        most[0] = max([most[0], *(len(task.world.objs) for task in a.env.tasks)])
    result = H.run(a, b, H.ControlSource(a, ['chickens', 'melee', 'wood', 'fish'], 4), 4, 300, observe=observe)
    assert result.divergence is None and result.steps == 300, result.divergence and result.divergence.text()
    assert result.most_objs >= most[0] >= 3, (result.most_objs, most)
    case = H.CaseResult('x', FREE, 'x25-none', 0, 'control:chickens', None)
    case.add(result, 4)
    assert case.most_objs == result.most_objs and f'at most {result.most_objs} ground objs' in case.line()


# ---- XpCheck in C against sim/audit.py XpCheck -----------------------------------------------------------------
AGENT = 'wd1a0'


@pytest.fixture(scope='module')
def xp_world(inputs):
    """The archive's rows a free-play life's XP comes from: trees and compact rocks by loc type, fish, chickens,
    goblins and muggers by npc type, bones."""
    tables, _, _, pack = inputs
    from sim.audit import XpCheck
    content = tables.meta['content']
    locs = {v['debugname']: k for k, v in tables.loc_types.items()}
    npcs = {v['debugname']: k for k, v in tables.npc_types.items()}
    trees = [(locs[name], tables.obj_id(row['product'])) for name, row in sorted(content['trees'].items()) if name in locs]
    rocks = [(locs[name], tables.obj_id(row['output'])) for name, row in sorted(content['compact']['rocks'].items())
             if name in locs]  # the archive's loc types (a world holds no other)
    fish = [tables.obj_id(name) for name in ('raw_shrimp', 'raw_anchovies')]
    bones = [tables.obj_id('bones')]
    return dict(check=lambda rate, stats: XpCheck(tables, rate, AGENT, stats), canon=H.Canon(H.PackStrings(pack)),
                trees=trees, rocks=rocks, fish=fish, bones=bones, pack=pack,
                npcs=[npcs[name] for name in ('chicken', 'goblin', 'mugger')])


@st.composite
def xp_streams(draw, world):
    """(rate, stats, events): ticks of productions, hits, burials and npc lives' respawns with their xp events, each
    event as the engine writes it (amounts from the oracle's own tables, the XP capped at 2,000,000,000 tenths; a hit on
    an npc life in the tick of its respawn gives none) or broken one way: another amount, before or after, stat or
    cause, a missing or an extra event, a hit by nobody, another life, XP for a hit in its life's respawn cycle, no XP for
    a hit a later tick, a second respawn of one life."""
    from sim.audit import HIT_PERCENT, PRODUCE_SKILL
    from sim.engine import HITPOINTS, PRAYER
    rate = draw(st.sampled_from((1, 25)))
    stats = [draw(st.one_of(st.integers(0, 40000), st.integers(2_000_000_000 - 20_000, 2_000_000_000)))
             for _ in range(21)]  # some near the XP cap, so that it applies
    check = world['check'](rate, list(stats))
    xp, events, tick = list(stats), [], draw(st.integers(0, 50))
    bad = draw(st.sampled_from(('none',) * 6 + ('amount', 'before', 'after', 'stat', 'cause', 'missing', 'extra',
                                                 'nobody', 'life', 'respawn_xp', 'late', 'second')))
    respawned = {}  # (npc, life) -> the tick of its npc_respawn event
    broken = draw(st.integers(0, 7))  # the index of the xp event a break lands on
    n_xp = 0

    def add(stat, amount, cause):
        nonlocal n_xp
        before, after = xp[stat], min(xp[stat] + amount * rate, 2_000_000_000)
        data = dict(stat=stat, xp=amount, before=before, after=after, **cause)
        here = n_xp == broken
        n_xp += 1
        if here and bad == 'missing':
            return
        if here and bad == 'amount':
            data['xp'] += 1
        if here and bad == 'before':
            data['before'] += 10
        if here and bad == 'after':
            data['after'] += 10
        if here and bad == 'stat':
            data['stat'] = (stat + 1) % 21
        if here and bad == 'cause':
            data = dict(data, **({'obj': cause.get('obj', 0) + 1} if 'obj' in cause else {'life': cause['life'] + 1}))
        xp[data['stat']] = data['after'] if data['before'] == xp[data['stat']] else xp[data['stat']]
        events.append((tick, 'xp', data))
        if here and bad == 'extra':
            events.append((tick, 'xp', dict(data, before=data['after'], after=min(data['after'] + data['xp'] * rate,
                                                                                    2_000_000_000))))
            xp[data['stat']] = events[-1][2]['after']
    for _ in range(draw(st.integers(2 if bad == 'late' else 1, 6))):
        tick += draw(st.integers(1 if bad == 'late' else 0, 3))
        for k in range(draw(st.integers(1, 3)) + 1):
            if k == 0:  # each tick may open with a respawn (always under the faults that need one)
                kind = 'respawn' if bad in ('respawn_xp', 'second', 'late') or draw(st.booleans()) else 'none'
            elif k == 1 and bad in ('second', 'late', 'respawn_xp'):  # the faults: a life respawned again; a hit
                kind = 'respawn' if bad == 'second' else 'hit'               # with no XP a tick later, or XP in the cycle
            else:
                kind = draw(st.sampled_from(('wood', 'ore', 'fish', 'hit', 'bury', 'gem', 'respawn')))
            if kind == 'none':
                continue
            if kind == 'respawn':  # World.npc_respawn's record; a second one of a life only as the fault
                npc, life = draw(st.sampled_from(sorted(respawned))) if bad == 'second' and respawned else \
                    (draw(st.integers(0, 15)), draw(st.integers(0, 3)))
                if (npc, life) in respawned and bad != 'second':
                    continue
                respawned.setdefault((npc, life), tick)
                events.append((tick, 'npc_respawn', dict(npc=npc, type=draw(st.sampled_from(world['npcs'])), life=life)))
            elif kind in ('wood', 'ore', 'fish', 'gem'):
                script = {'wood': '[label,get_logs]', 'ore': draw(st.sampled_from(('[label,get_ore_normal]',
                                                                                  '[label,get_ore_fast]'))),
                          'fish': '[proc,fish_roll]', 'gem': '[label,get_ore_normal]'}[kind]
                src, obj = draw(st.sampled_from(world['trees'] if kind == 'wood' else world['rocks'])) \
                    if kind in ('wood', 'ore') else (None, draw(st.sampled_from(world['fish'])))
                if kind == 'gem':  # a gem: a production of get_ore that owes no XP
                    src, obj = draw(st.sampled_from(world['rocks']))[0], world['bones'][0]
                events.append((tick, 'produce', dict(obj=obj, count=1, script=script)))
                if kind != 'gem':
                    amount = check._product(tick, dict(obj=obj, src=src), script)
                    add(PRODUCE_SKILL[script], amount, dict(cause='produce', obj=obj, src=src))
            elif kind == 'hit':
                now = [key for key, at in respawned.items() if at == tick]  # often a life that respawned in this tick
                earlier = sorted(key for key, at in respawned.items() if at < tick)  # or one that respawned before it
                if now and (bad == 'respawn_xp' or draw(st.booleans())):
                    npc, life = draw(st.sampled_from(now))
                elif earlier and (bad == 'late' or draw(st.booleans())):
                    npc, life = draw(st.sampled_from(earlier))
                else:
                    npc, life = draw(st.integers(0, 15)), draw(st.integers(0, 3))
                style = draw(st.sampled_from((0, 1, 2, 3)))
                hit = dict(npc=npc, type=draw(st.sampled_from(world['npcs'])), life=life,
                           source=None if bad == 'nobody' else AGENT, damage=draw(st.integers(0, 12)),
                           hp=draw(st.integers(1, 12)), style=style)
                events.append((tick, 'hit', hit))
                cycle = respawned.get((npc, life)) == tick
                if (cycle and bad != 'respawn_xp') or ((npc, life) in respawned and not cycle and bad == 'late'):
                    continue  # its life's respawn cycle gives none (the faults: XP there, none a later tick)
                for stat in (*HIT_PERCENT[style], HITPOINTS):
                    amount = check._hit_amount(dict(hit, used=set()), stat)
                    if amount:
                        add(stat, amount, dict(cause='hit', npc=npc, life=life))
            else:
                obj = world['bones'][0]
                events.append((tick, 'remove', dict(obj=obj, count=1, why='bury')))
                from sim.audit import _param
                add(PRAYER, _param(check.t, 'obj', obj, 'bone_exp'), dict(cause='bury', obj=obj))
    return rate, stats, events


def checks(xp_world, events, stats, rate=1):
    """(oracle, C) of one event stream: XpCheck.observe's (xp, gains) or AuditError, and Pack.xp_check's or its error."""
    from sim.audit import AuditError
    canon, pack = xp_world['canon'], xp_world['pack']
    rows = [(e[1], e[0], *e[2:]) for e in (canon.event(event, AGENT) for event in events)]
    oracle = xp_world['check'](rate, list(stats))
    try:
        gained = oracle.observe(events)
        want = (oracle.xp, gained)
    except AuditError as error:
        want = error
    try:
        got = pack.xp_check(rows, stats, rate)
        got = (list(got[0]), list(got[1]))
    except native.load().Error as error:
        got = native.as_python(error, pack)
    return want, got


def test_xp_check_in_c_is_the_oracle_s(xp_world):
    """The C XpCheck (rs_free_xp_check) is the oracle's on generated streams: the same XP and gains, or a refusal alike
    (raised_alike); the sample passes streams with a damaging hit in its npc life's respawn cycle and refuses XP
    there and a second respawn of one life."""
    from sim.audit import AuditError
    seen = dict(passed=0, refused=0, cycle_passed=0, cycle_refused=0, second=0, later_passed=0, later_refused=0)

    @settings(max_examples=400, deadline=None, derandomize=True, database=None, suppress_health_check=list(HealthCheck))
    @given(xp_streams(xp_world))
    def check(stream):
        rate, stats, events = stream
        want, got = checks(xp_world, events, stats, rate)
        if isinstance(want, AuditError):  # the C text is the oracle's, or its cut to what the C reason carries
            assert H.raised_alike(want, got), (got, want, events)
        else:
            assert got == want, (got, want, events)
        seen['passed' if not isinstance(want, AuditError) else 'refused'] += 1
        respawns = {(d['npc'], d['life']): t for t, kind, d in events if kind == 'npc_respawn'}
        cycle = any(kind == 'hit' and d['damage'] and respawns.get((d['npc'], d['life'])) == t for t, kind, d in events)
        seen['cycle_passed'] += cycle and not isinstance(want, AuditError)
        seen['cycle_refused'] += isinstance(want, AuditError) and "respawn cycle gives no XP" in str(want)
        seen['second'] += isinstance(want, AuditError) and str(want).startswith('a second npc_respawn')
        later = [(d['npc'], d['life'], t) for t, kind, d in events if kind == 'hit' and d['damage'] and d['source'] == AGENT
                 and respawns.get((d['npc'], d['life']), t) < t]  # a damaging hit on a life that respawned a tick before
        seen['later_passed'] += bool(later) and not isinstance(want, AuditError)
        seen['later_refused'] += bool(later) and isinstance(want, AuditError) and 'has no xp event for stats' in str(want)
    check()
    assert seen['passed'] >= 50 and seen['refused'] >= 50, seen
    # the respawn paths, in quantity (their mix varies with what ran before in the process; each is fixed in
    # test_a_hit_in_its_npc_life_s_respawn_cycle_owes_no_xp_in_c_as_in_the_oracle)
    assert sum(seen[k] for k in ('cycle_passed', 'cycle_refused', 'second', 'later_passed', 'later_refused')) >= 20, seen


def test_a_hit_in_its_npc_life_s_respawn_cycle_owes_no_xp_in_c_as_in_the_oracle(xp_world):
    """The respawn-cycle state, built as events: a chicken life respawns at tick 62 and the agent hits it there for 1
    (style 0, no xp event): both checks accept it, as they accept the same hit a tick later with its XP. An XP event for
    the hit in the respawn cycle, the hit a tick later with no XP (an engine that kept the multiplier at 0 a cycle
    longer: the check is no weaker) and a second respawn of that life are refused by both alike."""
    from sim.audit import AuditError
    from sim.engine import ATTACK, HITPOINTS
    chicken, stats = xp_world['npcs'][0], [1000] * 21
    respawn = (62, 'npc_respawn', dict(npc=8, type=chicken, life=1))
    hit = dict(npc=8, type=chicken, life=1, source=AGENT, damage=1, hp=3, style=0)
    want, got = checks(xp_world, [respawn, (62, 'hit', hit)], stats)
    assert want == got and want == (stats, [0] * 21), (want, got)
    amount = xp_world['check'](1, stats)._hit_amount(dict(hit, used=set()), ATTACK)
    given = (62, 'xp', dict(stat=ATTACK, xp=amount, before=1000, after=1000 + amount, cause='hit', npc=8, life=1))
    hp = xp_world['check'](1, stats)._hit_amount(dict(hit, used=set()), HITPOINTS)
    later = [(63, 'hit', hit), (63, 'xp', dict(given[2])),
             (63, 'xp', dict(stat=HITPOINTS, xp=hp, before=1000, after=1000 + hp, cause='hit', npc=8, life=1))]
    want, got = checks(xp_world, [respawn, *later], stats)
    assert want == got and want[1][ATTACK] == amount and want[1][HITPOINTS] == hp, (want, got)
    for events, words in (([respawn, (62, 'hit', hit), given], "a hit in its npc life's respawn cycle gives no XP"),
                          ([respawn, (63, 'hit', hit)], 'has no xp event for stats [0, 3]'),
                          ([respawn, (62, 'hit', hit), (110, 'npc_respawn', dict(npc=8, type=chicken, life=1))],
                           'a second npc_respawn of npc 8 life 1 at tick 110')):
        want, got = checks(xp_world, events, stats)
        assert isinstance(want, AuditError) and words in str(want) and H.raised_alike(want, got), (want, got)


# ---- village-world-v2 (the pool's declared world decides) ----------------------------------------------------------
DATA_NC2 = DATA.with_name('draynor-nc2.npz')
POOL_V2 = ROOT / 'logs/sim/world/train-128-20000-v2.npz'
# the v2 training pool's first seven layouts and 20014: between them a door pen (with and without the highwayman) and
# gate pens, the highwayman, each field kind, a willow and a yew
V2_LAYOUTS = (0, 1, 2, 3, 4, 5, 6, 14)
HIGHWAYMAN = 180


@pytest.fixture(scope='module')
def exported_v2(name_table_path, tmp_path_factory):
    """(pool, pack) files of village-world-v2: V2_LAYOUTS as a plain pool (VillageLayouts) and their pack without trees
    over draynor-nc2 (one supply layout, one variant) under the synthetic name table."""
    if not (DATA_NC2.exists() and POOL_V2.exists() and SUPPLY.exists()):
        pytest.skip('private v2 archive, v2 pool or supply pool absent')
    from names import NameTable
    from sim import search, world as W
    from sim.native_export import export
    work = tmp_path_factory.mktemp('free-play-v2')
    layouts = W.load_pool(POOL_V2)[0]
    pool = W.save_pool(work / 'pool.npz', [layouts[k] for k in V2_LAYOUTS])
    supply = W.save_pool(work / 'supply.npz', W.load_pool(SUPPLY)[0][:1], version=search.SUPPLY['world'])
    return pool, export(work / 'v2.pack', pool, supply, DATA_NC2, names=NameTable(name_table_path), trees=False,
                        variants=1, log=lambda _: None)


@pytest.fixture(scope='module')
def inputs_v2(exported_v2, name_table_path):
    """(tables, layouts, derived None, pack) of the v2 files."""
    from names import NameTable
    from sim import world as W
    from sim.workers import load_tables
    pool, pack = exported_v2
    layouts = W.load_pool(pool)[0]
    assert {W.world_of(layout)['version'] for layout in layouts} == {'village-world-v2'}
    return load_tables(DATA_NC2), layouts, None, native.open_pack(pack, names=NameTable(name_table_path))


class Census:
    """observe(): what the Python side's lives reached of village-world-v2 (a run's coverage, read after it): picks,
    willow and yew logs, steps with a pen door open, the highwayman's hits on the player and the player's on it, its
    defeats, and the lives ended by a death."""

    def __init__(self, side, tables):
        self.side, self.counts, self.slots = side, {}, {}
        self.objs = {name: tables.obj_ids[name] for name in ('willow_logs', 'yew_logs')}
        self.cross = next(k for k, v in tables.loc_types.items() if v['debugname'] == 'poordoor_cross')

    def add(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + n

    def __call__(self, step, out_a, out_b):
        for b, task in enumerate(self.side.env.tasks):
            w = task.world
            last, seen = self.slots.get(b, (None, 0))
            if last is not None and last is not w:  # a life that ended in the step (its last decision's events)
                self.events(last, seen)
                seen = 0
            self.events(w, seen)
            self.slots[b] = (w, len(w.events))
            if any(loc.active and loc.type == self.cross for row in w.loc_at.values() for loc in row):  # an added loc
                self.add('door open steps')
        if step >= 0:
            self.add('deaths', int((out_a[4]['terminal'] == 2).sum()))

    def events(self, w, seen):
        hunter = {n.nid for n in w.npcs if n.type == HIGHWAYMAN}
        for _, kind, d in w.events[seen:]:
            if kind == 'produce' and d.get('script') == '[proc,pickup_loc_floor]':
                self.add('picks')
            if kind == 'produce' and d.get('obj') in self.objs.values():
                self.add('willow and yew logs')
            if kind == 'hit' and d.get('type') == HIGHWAYMAN:
                self.add('hits on the highwayman')
            if kind == 'npc_defeated' and d.get('type') == HIGHWAYMAN:
                self.add('highwayman defeated')
            if kind == 'agent_hp' and d.get('cause') == 'damage' and d.get('source') in hunter:
                self.add('highwayman damage')
                if d.get('after') == 0:
                    self.add('highwayman kills')


@pytest.mark.parametrize('xp_rate', [1, 25])
def test_village_world_v2_is_equal_step_by_step_and_tick_by_tick(inputs_v2, xp_rate):
    """Random legal play on village-world-v2 (no muggers): observations, reward bits, done, every info field, each
    start and every tick's record, draws and events alike; the run picks plants and meets the highwayman (its hunt,
    approach and hits, and the player's hits on it)."""
    a, b = sides(inputs_v2, 8, xp_rate=xp_rate, muggers=False, life=('uniform', 100, 1000))
    census = Census(a, inputs_v2[0])
    result = H.run(a, b, H.RandomLegal(5), 5, 300, observe=census)
    assert result.divergence is None, result.divergence.text()
    assert not result.gap and result.traced and result.steps == 300, (result.gap, result.stopped)
    c = census.counts
    assert c.get('picks', 0) > 0 and c.get('highwayman damage', 0) > 0 and c.get('hits on the highwayman', 0) > 0, c


def test_village_world_v2_controls_play_alike(inputs_v2):
    """sim/free.py's controls on village-world-v2 at 25x: the gatherers, the fighter and the chickens control (the
    highwayman among their attackers, the pens' gates and doors) and the rotation, with walks to targets past the
    client scene of a 64-tile layout, alike tick by tick."""
    a, b = sides(inputs_v2, 6, xp_rate=25, muggers=False, life=('fixed', 2500))
    source = H.ControlSource(a, ['mine', 'wood', 'fish', 'melee', 'chickens', 'rotation'], 9, period=600)
    result = H.run(a, b, source, 9, 350)
    assert result.divergence is None, result.divergence.text()
    assert result.traced and result.steps == 350, (result.gap, result.stopped)


def test_the_highwayman_kills_and_is_killed_alike(inputs_v2):
    """Lives that start on a highwayman layout at combat level 3 (attack, strength and defence 1): waiting, the player is
    hunted, approached and hit by it, and lives end in its kills; the fighter kills it, its death drop (bones and the
    black cape) falls and it respawns. Alike step by step and tick by tick."""
    k = next(i for i, layout in enumerate(inputs_v2[1]) if any(int(r[1]) == HIGHWAYMAN for r in layout.npcs)
             and layout.gate[1] >= 0)
    scenario = dict(layout=k, levels={'attack': 1, 'strength': 1, 'defence': 1})
    a, b = sides(inputs_v2, 8, scenario=scenario, xp_rate=25, muggers=False, life=('fixed', 2000))
    census = Census(a, inputs_v2[0])
    result = H.run(a, b, H.ControlSource(a, ['wait', 'melee'], 4), 4, 300, observe=census)
    assert result.divergence is None, result.divergence.text()
    assert result.traced and result.steps == 300, (result.gap, result.stopped)
    c = census.counts
    assert c.get('highwayman kills', 0) > 0 and c.get('highwayman defeated', 0) > 0, c


def test_a_pen_door_opens_through_the_chickens_control_alike(inputs_v2):
    """The chickens control on a door pen with no highwayman: the door's Open (its cross for 500 ticks), the chickens
    behind it, alike tick by tick."""
    k = next(i for i, layout in enumerate(inputs_v2[1]) if layout.gate[0] >= 0 and layout.gate[1] < 0
             and not any(int(r[1]) == HIGHWAYMAN for r in layout.npcs))
    a, b = sides(inputs_v2, 4, scenario=dict(layout=k), xp_rate=25, muggers=False, life=('fixed', 3000))
    census = Census(a, inputs_v2[0])
    result = H.run(a, b, H.ControlSource(a, ['chickens'], 6), 6, 250, observe=census)
    assert result.divergence is None, result.divergence.text()
    assert result.traced and result.steps == 250, (result.gap, result.stopped)
    assert census.counts.get('door open steps', 0) > 0, census.counts


def test_the_village_world_v2_batch_has_cpubatch_s_interface(inputs_v2):
    """A v2 pack's free-play batch holds village-world-v2, as CpuBatch on the v2 pool, and both refuse alike (same
    class and text): muggers, another world, a muggers form of no bool. A goal task on the v2 pack is refused natively
    (a declared difference: CpuBatch plays the goal tasks on v2 layouts; the C engine on compact-world-v1's only)."""
    from sim.vec import CpuBatch
    tables, layouts, derived, pack = inputs_v2
    for case in (dict(), dict(world='village-world-v2'), dict(world='village-world-v1'), dict(muggers=True),
                 dict(muggers=None), dict(xp_rate=25)):
        settings = dict(dict(muggers=False, life=('fixed', 120)), **case)
        got = (outcome(lambda: native.NativeBatch(pack, 2, task=FREE, **settings)),
               outcome(lambda: CpuBatch(tables, layouts, 2, derived=derived, task=FREE, **settings)))
        assert got[0] == got[1], (case, got)
    assert native.NativeBatch(pack, 2, task=FREE, muggers=False, life=SHORT).world == \
        CpuBatch(tables, layouts, 2, derived=derived, task=FREE, muggers=False, life=SHORT).world == 'village-world-v2'
    assert set(native.compact_layouts(pack)) == {(layout.seed, 'village-world-v2') for layout in layouts}
    with pytest.raises(ValueError, match='^the native engine plays the goal tasks on compact-world-v1 layouts only$'):
        native.NativeBatch(pack, 2, task='goal-activity-v5')
    from sim.engine import XP_RATE
    from sim.spec import HORIZON_TICKS
    with pytest.raises(native.load().Error, match='compact-world-v1 layouts only'):  # the C rule, past NativeBatch
        native.load().Batch(pack, dict(task=native.TASKS['goal-activity-v5'], n=1, first=0, n_goals=4, goal=0,
                                       horizon=HORIZON_TICKS, pace_jitter=0, decision_points=0, max_decisions=100,
                                       max_episode_ticks=100, threads=1, xp_rate=XP_RATE, muggers=0, life_kind=0, life_lo=0,
                                       life_hi=0), native.log1p_table())
    with pytest.raises(native.load().Error, match='village-world-v2 holds no muggers: pass False'):
        native.load().Batch(pack, dict(task=native.TASKS[FREE], n=1, first=0, n_goals=4, goal=0, horizon=HORIZON_TICKS,
                                       pace_jitter=0, decision_points=0, max_decisions=100, max_episode_ticks=10000,
                                       threads=1, xp_rate=1, muggers=1, life_kind=0, life_lo=120, life_hi=120),
                            native.log1p_table())


def test_pack_episode_draw_of_village_world_v2_equals_the_oracle(inputs_v2):
    """Pack.episode_draw's free-play branch on the v2 pack against CpuBatch's draws (draw_village: no mugger tiles drawn,
    draw_life, village_npcs: the highwayman after the fishing spots and before the goblins, kit): the start, its length,
    the kit and the spawns alike for keyed (key, slot, episode) and five life forms; muggers refused alike (village-world-v2
    holds none)."""
    from sim.combat import draw_village, kit, outside, village_npcs
    from sim.task import draw_life, draw_start
    tables, layouts, _, pack = inputs_v2
    regions = [outside(layout) for layout in layouts]
    seen = dict(drawn=0, refused=0, hunters=0)

    @settings(max_examples=300, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    @given(st.integers(-2 ** 63, 2 ** 63 - 1), st.integers(0, 2 ** 31 - 1), st.integers(0, 2 ** 31 - 1),
           st.booleans(), st.sampled_from(LIVES))
    def check(key, slot, episode, muggers, life):
        k = draw_start(key, slot, episode, layouts)['layout']
        try:
            start = draw_village(key, slot, episode, layouts, tables, muggers, region=regions[k])
            start['length'] = draw_life(key, slot, episode, life)
            v = start['village']
            want = (start, [(tables.obj_id(name), count) for name, count in kit(start['weapon'], v['food'])],
                    [(n['nid'], n['type'], n['x'], n['z'], n['level'], n['size'])
                     for n in village_npcs(layouts[k], tables, v['muggers'])])
        except ValueError as error:
            want = ('ValueError', str(error))
        try:
            raw, kit_ids, spawns = pack.episode_draw(native.TASKS[FREE], 0, key, slot, episode, int(muggers),
                                                     int(life[0] == 'uniform'), life[1], life[-1])
            assert raw['kind'] == native.START_KINDS.index('village') and raw['rec'] == k
            got = (native.decode_start(raw), kit_ids, spawns)
        except native.load().Error as error:
            failed = native.as_python(error, pack)
            got = (type(failed).__name__, str(failed))
        assert got == want, (key, slot, episode, muggers, life)
        seen['refused' if want[0] == 'ValueError' else 'drawn'] += 1
        seen['hunters'] += want[0] != 'ValueError' and any(s[1] == HIGHWAYMAN for s in want[2])
    check()
    assert seen['drawn'] >= 100 and seen['refused'] >= 100 and seen['hunters'] >= 30, seen


def test_the_training_workers_name_their_world_and_collect_alike_on_both_engines(exported, exported_v2, inputs_v2,
                                                                               tmp_path):
    """The training workers (sim/workers.py CpuWorkers) as a free-play run builds them (the world among its named batch
    settings): on the v2 pool the Python reference in one worker and the C engine on the v2 pack in 1, 2 and 4 workers
    report the same settings in effect, village-world-v2 among them, and give the same outputs (observations, reward
    bits, done, every info field and each ended life's last observation) world by world and step by step under keyed
    random legal play; on a v1 pool and pack both engines report village-world-v1; a named world that is not the pool's
    is refused by both in the same words; and a training command (sim/train.py train) that names no world is refused on
    the C engine as on the Python reference, before any worker starts."""
    from sim.engine import Rng
    from sim.spec import OBSERVATIONS
    from sim.task import FREE_PLAY
    from sim.vec import R_ACTION
    from sim.workers import CpuWorkers
    obs_v = FREE_PLAY['observation']

    def workers(pool, pack, tables, engine, n, world):
        return CpuWorkers(8, n, pool, tables=tables, task=FREE, goals=None, engine=engine,
                          pack=pack if engine == 'native' else None, observation=obs_v, shapes=OBSERVATIONS[obs_v],
                          batch=dict(muggers=False, xp_rate=1, life=('uniform', 100, 300), world=world))
    pool, pack = exported_v2
    runs, settings = {}, {}
    for engine, n in (('cpu', 1), ('native', 1), ('native', 2), ('native', 4)):
        env = workers(pool, pack, DATA_NC2, engine, n, 'village-world-v2')
        try:
            settings[(engine, n)] = env.settings_in_effect
            rng, rows = Rng(3), []
            state, obs = env.reset(3)
            for step in range(120):
                legal = [np.flatnonzero(obs['mask'][b]) for b in range(8)]
                action = np.array([row[int(rng(R_ACTION, step, b) * len(row))] for b, row in enumerate(legal)], np.int32)
                state, obs, reward, done, info = env.step(state, action)
                rows.append([H.world_digest(obs, reward, done, info, b) for b in range(8)])
        finally:
            env.close()
        runs[(engine, n)] = rows
    assert all(rows == runs[('cpu', 1)] for rows in runs.values())
    assert all(s == settings[('cpu', 1)] for s in settings.values()) and settings[('cpu', 1)]['world'] == 'village-world-v2'
    v1_pool, v1_pack = exported
    for engine in ('cpu', 'native'):
        env = workers(v1_pool, v1_pack, DATA, engine, 1, 'village-world-v1')
        try:
            assert env.settings_in_effect['world'] == 'village-world-v1', engine
        finally:
            env.close()
    refused = {}
    for engine in ('cpu', 'native'):
        with pytest.raises(RuntimeError) as caught:
            workers(pool, pack, DATA_NC2, engine, 1, 'village-world-v1')
        refused[engine] = str(caught.value)
    assert refused['cpu'] == refused['native'] and "the pool holds village-world-v2 layouts, not 'village-world-v1'" in \
        refused['cpu'], refused
    from sim.train import train
    unnamed = dict(decisions=8 * 16, seed=1, worlds=8, n_steps=16, batch_size=64, n_epochs=1, verbose=False, task=FREE,
                   env='cpu', pool=pool, tables=DATA_NC2, batch=dict(muggers=False, life=['uniform', 100, 300]))
    for engine, packed in (('cpu', {}), ('native', dict(pack=pack))):
        with pytest.raises(ValueError, match=r"names muggers, life, world \(--batch-setting\): \['world'\] missing"):
            train(tmp_path / engine, engine=engine, **unnamed, **packed)
    assert not any(tmp_path.iterdir())


def test_a_layout_record_of_no_declared_world_is_refused_at_open(exported, tmp_path):
    """A layout record names its world as 1 + its sim/world.py WORLDS index (rs.h RsLayoutRec.world); a pack whose
    record names world 0 or one past the declared worlds is refused when it is mapped, before any batch reads it."""
    from sim.world import WORLDS
    module = native.load()
    _, pack = exported
    section = module.constants()['sections']['LAYOUTS']
    offset = next(row[5] for row in module.Pack(str(pack)).sections() if row[0] == section)
    dtype, field = native.record_dtype('RsLayoutRec').fields['world'][:2]
    data = bytearray(pack.read_bytes())
    for value in (0, len(WORLDS) + 1):
        data[offset + field:offset + field + dtype.itemsize] = np.asarray(value, dtype).tobytes()
        (tmp_path / 'bad.pack').write_bytes(data)
        with pytest.raises(module.Error) as caught:
            module.Pack(str(tmp_path / 'bad.pack'))
        assert str(native.as_python(caught.value)) == "native pack: a layout's world is not a declared one", value


def test_a_village_world_v2_gate_case_and_thread_check_run_on_its_own_inputs(exported_v2, tmp_path, monkeypatch):
    """The gate's village-world-v2 set on its own inputs (the v2 files, a GateContext of that world): one quick case of
    its configurations (gate_case) equal throughout, with the decisions it compared, the census of its Python lives and
    their deaths in its row; its thread check (1, 3 and 8 threads) identical; the rows name the world set's case ids and
    carry the process's provenance (the files it played and their sha256, the pack's recorded inputs, the library). Not a
    gate run: one case and the thread part, on 8 layouts."""
    pool, pack = exported_v2
    supply = pool.parent / 'supply.npz'
    ctx = H.GateContext(str(DATA_NC2), str(pool), str(supply), str(pack), tmp_path, 1, 'village-world-v2').load(
        policies=False)
    assert ctx.base(FREE, 'w2-x1').world == ctx.batch(FREE, 'w2-x1', 2).world == 'village-world-v2'
    rows, lines, reads = [], [], [0]
    read = H.FreeCensus.read

    def counted(self, b, task):
        reads[0] += 1
        return read(self, b, task)
    monkeypatch.setattr(H.FreeCensus, 'read', counted)
    want = 'free-play-v1|w2-x1|0|random|-'
    H.gate_cases_part(ctx, 'quick', (0, 1), lambda name: name != want, rows.append, lines.append)
    assert [row.id for row in rows] == [want] and want in H.gate_expected('quick', world='village-world-v2')
    row = rows[0]
    assert row.divergences == 0 and row.gaps == 0 and not row.stopped and row.runs > 0 and row.ticks > 0, row
    assert row.compared == row.decisions > 0 and isinstance(row.deaths, int), row
    assert reads[0] == (row.decisions + row.runs) * H.GATE_N, reads  # every world at each reset and compared decision
    assert set(row.census) == {'picks', 'pick_full', 'willow_yew_logs', 'door_openings', 'door_open_steps',
                               'highwayman_damage', 'highwayman_hits', 'highwayman_defeated', 'cape_pickups', 'cape_worn'}
    threads = H.gate_threads(ctx, 'quick', lines.append)
    assert threads['failures'] == [] and 'free play (w2-x1) at 1, 3 and 8 threads: identical' in lines[-1], lines[-1]
    given = H.gate_provenance(ctx)
    assert given['world'] == 'village-world-v2' and given['library'] == native.library()[0].name
    assert given['files']['pack']['sha256'] == native.sha256_file(pack) and not H.provenance_problems(
        [H.gate_row(row, 'c' * 40, 'quick', None, 'village-world-v2', given)])


def test_a_village_world_v2_gate_command_runs_resumes_and_records_its_inputs(exported_v2, tmp_path, monkeypatch):
    """gate_main end to end on the v2 files, as one process of a run plan: one quick case and the thread check (--only),
    at a clean commit (stood in: the suite may run on a changed tree) with SIM_NATIVE_LTO named. It exits 0 and every row
    is of that commit and carries one provenance: the files' sha256, the library, SIM_NATIVE_LTO. A resume runs nothing
    again; after an interruption cut its last row, a resume runs that part alone again; a resume at another commit is
    refused and leaves the file as it was."""
    import argparse
    import json
    pool, pack = exported_v2
    supply, out = pool.parent / 'supply.npz', tmp_path / 'gate.jsonl'
    monkeypatch.setattr(H, '_commit', lambda: 'a' * 40)
    monkeypatch.setenv('SIM_NATIVE_LTO', '1')  # LTO on, as unset: the same library

    def gate(*argv):
        parser = argparse.ArgumentParser()
        H.gate_arguments(parser)
        args = parser.parse_args(['--quick', '--world', 'village-world-v2', '--archive', str(DATA_NC2), '--pool', str(pool),
                                  '--supply', str(supply), '--pack', str(pack), '--work', str(tmp_path), '--out', str(out),
                                  '--parts', 'cases,threads', '--only', r'^free-play-v1\|w2-x1\|0\|idle\||^threads$', *argv])
        with pytest.raises(SystemExit) as code:
            H.gate_main(args)
        return code.value.code
    assert gate() == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row['id'] for row in rows] == ['free-play-v1|w2-x1|0|idle|freehurt', 'threads'], rows
    given = rows[0]['provenance']
    assert all(row['commit'] == 'a' * 40 and row['provenance'] == given for row in rows)
    assert given['env'] == {'SIM_NATIVE_LTO': '1'} and given['library'] == native.library()[0].name
    assert given['files']['pack']['sha256'] == native.sha256_file(pack) and given['pack_inputs']['pool'] == \
        given['files']['pool']['sha256'] == native.sha256_file(pool)
    assert rows[0]['compared'] > 0 and rows[0]['census'] is not None
    text = out.read_text()
    assert gate('--resume') == 0 and out.read_text() == text            # nothing to run again
    out.write_text(text[:-12])                                          # the thread check's row cut short
    assert gate('--resume') == 0
    again = [json.loads(line) for line in out.read_text().splitlines()]
    assert again[0] == rows[0] and [row['id'] for row in again] == ['free-play-v1|w2-x1|0|idle|freehurt', 'threads']
    text = out.read_text()
    monkeypatch.setattr(H, '_commit', lambda: 'b' * 40)
    with pytest.raises(ValueError, match='give a new --out'):
        gate('--resume')
    assert out.read_text() == text


def test_a_gate_world_set_refuses_another_world_s_inputs(exported, exported_v2, name_table_path, tmp_path):
    """A gate world set plays its own world's pool: village-world-v2's set on village-world-v1's files (and v1's on v2's)
    is refused at load, before any case could run v1 layouts under v2's name; and village-world-v2's set plays layouts of
    the train split only: a v2 pool holding a development seed is refused at load."""
    from dataclasses import replace
    from names import NameTable
    from sim import world as W
    from sim.native_export import export
    for (pool, pack), archive, world in ((exported, DATA, 'village-world-v2'), (exported_v2, DATA_NC2, 'village-world-v1')):
        ctx = H.GateContext(str(archive), str(pool), str(pool.parent / 'supply.npz'), str(pack), tmp_path, 1, world)
        with pytest.raises(ValueError, match="plays that world's pool"):
            ctx.load(policies=False)
    pool, _ = exported_v2
    supply = pool.parent / 'supply.npz'
    layouts = W.load_pool(pool)[0]
    held = W.save_pool(tmp_path / 'held.npz', [*layouts[:-1], replace(layouts[-1], seed=W.SPLITS['development'][0])])
    pack = export(tmp_path / 'held.pack', held, supply, DATA_NC2, names=NameTable(name_table_path), trees=False,
                  variants=1, log=lambda _: None)
    ctx = H.GateContext(str(DATA_NC2), str(held), str(supply), str(pack), tmp_path, 1, 'village-world-v2')
    with pytest.raises(ValueError, match='train split only'):
        ctx.load(policies=False)


def test_the_wider_world_s_workers_part_compares_both_engines_and_finds_a_difference(exported_v2, tmp_path, monkeypatch):
    """The gate's training-workers part of village-world-v2 (gate_workers, quick): the C engine in 1, 2 and 4 workers
    equals the Python reference in one on both configurations, lives ending within the run; with one world output
    changed in the 2-worker run (a stand-in for an engine difference) the part names that run, its decision and world."""
    pool, pack = exported_v2
    ctx = H.GateContext(str(DATA_NC2), str(pool), str(pool.parent / 'supply.npz'), str(pack), tmp_path, 1,
                        'village-world-v2').load(policies=False)
    lines = []
    row = H.gate_workers(ctx, 'quick', lines.append)
    n, steps = H.GATE['quick']['workers']
    assert row['failures'] == [] and row['compared'] == len(H.WORKERS_CONFIGS) * len(H.WORKERS_COUNTS) * n * steps, row
    assert row['episodes'] > 0 and 'identical' in lines[-1], lines[-1]
    digest, calls = H.world_digest, [0]
    target = 2 * n * (steps + 1) + n * 3 + 5  # the 2-worker run of w2-x1: its third decision's world 5

    def changed(*args):
        calls[0] += 1
        return 'changed' if calls[0] - 1 == target else digest(*args)
    monkeypatch.setattr(H, 'world_digest', changed)
    row = H.gate_workers(ctx, 'quick', lines.append)
    assert row['failures'] == ['w2-x1: the C engine in 2 workers differs from the Python reference first at decision 2 '
                               'world 5 (1 world outputs)'], row['failures']


@pytest.fixture(scope='module')
def free_start(exported_v2, name_table_path, tmp_path_factory):
    """A free-play-v1 start (sim.train train --initial-only on the v2 pool, under the suite's synthetic name table): an
    obs-v6 checkpoint for the policy sources."""
    import sys
    import names
    from sim.train import main
    pool, _ = exported_v2
    out = tmp_path_factory.mktemp('free-start')
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('RL_NAME_TABLE', str(name_table_path))
        patch.setattr(sys, 'argv', ['sim.train', 'train', '--task', FREE, '--env', 'cpu', '--workers', '1', '--pool',
                                    str(pool), '--tables', str(DATA_NC2), '--worlds', '2', '--n-steps', '16',
                                    '--batch-size', '32', '--epochs', '1', '--decisions', '32', '--seed', '3',
                                    '--batch-setting', 'world="village-world-v2"', 'muggers=false',
                                    'life=["fixed",200]', '--output', str(out / 'run'), '--initial-only'])
        names.name_table.cache_clear()
        try:
            main()
        finally:
            names.name_table.cache_clear()
    return out / 'run' / 'initial.zip'


def test_a_free_play_checkpoint_is_a_diff_source_that_plays_both_engines_alike(inputs_v2, free_start):
    """policy:PATH of a free-play-v1 start loads through sim/hour.py load_free_policy's obs-v6 checks, where
    check_names' obs-v5 refused it; its sampled actions play village-world-v2 alike in both engines, and it refuses an
    observation that is not exactly its own arrays."""
    source = H.make_source(f'policy:{free_start}', 5)
    assert isinstance(source, H.PolicySource) and source.exact
    a, b = sides(inputs_v2, 4, xp_rate=1, muggers=False, life=('uniform', 100, 400))
    result = H.run(a, b, source, 5, 60)
    assert result.divergence is None, result.divergence.text()
    assert not result.gap and result.traced and result.steps == 60 and result.episodes > 0, (result.gap, result.stopped)
    _, obs = a.env.reset(5)
    with pytest.raises(ValueError, match="unlike the policy's own arrays"):
        source(0, {k: v for k, v in obs.items() if k != 'prog_vec'}, range(4), np.ones(4, bool))


def test_the_policy_part_plays_its_pinned_checkpoint_and_the_found_life_in_both_engines(exported_v2, free_start, tmp_path,
                                                                                       monkeypatch):
    """The gate's policy part (gate_policy) on the suite's v2 pack, its pins pointed at a free-play start and the suite's
    pool: the checkpoint sampled under the wider pilot's batch in both engines, every decision of each run compared, what
    its lives reached counted, and the found life (here waits on the suite's pool) played to its last action: no failure.
    A checkpoint without its pinned sha256, or a pool or archive that is not the plan's, fails the part and plays
    nothing."""
    import copy
    from sim import gate
    pool, pack = exported_v2
    plan = copy.deepcopy(gate.FREE['plans']['wider_pilot'])
    plan['run'].update(pool=native.sha256_file(pool), tables=native.sha256_file(DATA_NC2))
    monkeypatch.setitem(gate.FREE['plans'], 'wider_pilot', plan)
    pinned = dict(H.GATE_POLICY['village-world-v2'], checkpoint=free_start, sha256=native.sha256_file(free_start))
    monkeypatch.setitem(H.GATE_POLICY, 'village-world-v2', pinned)
    monkeypatch.setitem(H.GATE['quick'], 'free_policy', ((0, 2), 2, 20))
    monkeypatch.setattr(H, 'RESPAWN_LIFE', dict(H.RESPAWN_LIFE, key=5, slot=1, episode=0, actions=(0,) * 12))
    ctx = H.GateContext(str(DATA_NC2), str(pool), str(pool.parent / 'supply.npz'), str(pack), tmp_path, 1,
                        'village-world-v2').load(policies=False)
    lines = []
    row = H.gate_policy(ctx, 'quick', lines.append)
    assert row['failures'] == [] and (row['runs'], row['compared'], row['world_decisions']) == (2, 40, 80), row
    assert row['ticks'] > 0 and row['files']['checkpoint']['sha256'] == pinned['sha256']
    assert {'agent_hits', 'agent_defeats', 'productions', 'xp_events', 'level_ups', 'picks'} <= set(row['census'])
    assert row['found_life'] == dict(key=5, slot=1, episode=0, actions=12, steps=12, ticks=row['found_life']['ticks'],
                                     stopped='', gap='', divergence=None)
    assert H.gate_clean([row])[0] and lines[0].startswith('[gate] policy: wider_pilot initial.zip')
    # the found life is played from its slot, key and episode, and a stop alike in it fails the part
    from sim.audit import AuditError
    played, build = [], H.build

    def stopping(inputs, n, first=0, tables_=None):
        a, b = build(inputs, n, first=first, tables_=tables_)
        if n == 1:
            for side in (a, b):
                step = side.step

                def raising(state, action, step=step, side=side):
                    out = step(state, action)
                    if side.step_index == 10:
                        raise AuditError('a check failure alike')
                    return out
                side.step = raising
            start = a.start_episode
            a.start_episode = lambda w, episode: played.append((a.first, episode)) or start(w, episode)
        return a, b
    monkeypatch.setattr(H, 'build', stopping)
    monkeypatch.setattr(H, 'RESPAWN_LIFE', dict(H.RESPAWN_LIFE, key=5, slot=1, episode=2, actions=(0,) * 12))
    row = H.gate_policy(ctx, 'quick', lines.append)
    assert played == [(1, 2)] and row['found_life']['steps'] == 10 and row['found_life']['stopped'], row['found_life']
    assert row['failures'] == ['the respawn-cycle life (key 5, slot 1, episode 2) played 10 of its 12 actions: both '
                               'sides raised AuditError at step 10: a check failure alike']
    assert not H.gate_clean([row])[0]
    monkeypatch.setattr(H, 'build', build)
    for change, plan_change, words in ((dict(pinned, sha256='0' * 64), None, 'checkpoint'),
                                       (pinned, dict(plan, run=dict(plan['run'], pool='another pool')), 'pool'),
                                       (pinned, dict(plan, run=dict(plan['run'], tables='0' * 64)), 'archive'),
                                       (pinned, dict(plan, world='village-world-v1'), "not village-world-v2's")):
        monkeypatch.setitem(H.GATE_POLICY, 'village-world-v2', change)
        monkeypatch.setitem(gate.FREE['plans'], 'wider_pilot', plan_change or plan)
        row = H.gate_policy(ctx, 'quick', lines.append)
        assert (row['runs'], row['compared']) == (0, 0) and words in row['failures'][0], (words, row['failures'])
        assert not H.gate_clean([row])[0]
    monkeypatch.setitem(H.GATE_POLICY, 'village-world-v2', pinned)
    monkeypatch.setitem(gate.FREE['plans'], 'wider_pilot', plan)
    stale = tmp_path / 'stale.pack'  # a pack that does not open: the part fails with its words, it does not crash
    stale.write_bytes(b'not a pack')
    monkeypatch.setattr(ctx, 'pack_path', str(stale))
    row = H.gate_policy(ctx, 'quick', lines.append)
    assert row['runs'] == 0 and "does not open on the plan's inputs" in row['failures'][0] and not H.gate_clean([row])[0]


def test_the_free_play_census_counts_what_its_lives_reached(exported_v2):
    """FreeCensus over a life's records (and PolicyCensus: also the agent's hits and defeats, its productions, XP events and
    level-ups): a pick, a Pick refused on a full inventory (its message), a pen door that opens
    (counted once while it stays open, again after it closes and opens), the cape picked up and worn, the highwayman's hit
    on the player and its defeat; a life that ends is read to its end before the next one's records."""
    from types import SimpleNamespace as NS
    from sim.engine import INV, WORN
    from sim.workers import load_tables
    tables = load_tables(DATA_NC2)
    census, policy = H.FreeCensus(tables), H.PolicyCensus(tables)
    cape, logs = tables.obj_ids['black_cape'], tables.obj_ids['yew_logs']
    cross = next(k for k, v in tables.loc_types.items() if v['debugname'] == 'poordoor_cross')
    hunter = next(k for k, v in tables.npc_types.items() if v['debugname'] == 'highwayman')
    door = NS(active=False, type=cross)

    def life():
        return NS(t=tables, events=[], chat_log=[], npcs=[NS(nid=7, type=hunter)], loc_at={(0, 1, 1): [door]},
                  text=lambda key: f'text of {key}')
    w = life()
    task = NS(world=w)
    w.events += [(1, 'produce', dict(obj=3, script='[proc,pickup_loc_floor]')), (1, 'produce', dict(obj=logs, script='x'))]
    w.chat_log += [(1, 'text of pick_full_cabbage'), (1, 'You pick a cabbage.')]
    census.read(0, task), policy.read(0, task)
    door.active = True
    w.events += [(2, 'pickup', dict(obj=cape, count=1)), (2, 'transfer', dict(src=INV, dst=WORN, obj=cape, to=cape, count=1)),
                 (2, 'agent_hp', dict(cause='damage', source=7)), (2, 'npc_defeated', dict(type=hunter, source=AGENT)),
                 (2, 'xp', dict(stat=0, xp=830, before=0, after=830)),  # attack 1 -> 2: a level-up
                 (2, 'xp', dict(stat=3, xp=13, before=11540, after=11553))]
    census.read(0, task), policy.read(0, task)
    census.read(0, task), policy.read(0, task)  # the door still open: one opening, two open steps
    door.active = False
    census.read(0, task), policy.read(0, task)
    door.active = True
    w.events.append((3, 'hit', dict(type=hunter, source=AGENT)))  # read at the next life's first read: its last records
    task.world = life()
    census.read(0, task), policy.read(0, task)
    assert census.counts == dict(picks=1, pick_full=1, willow_yew_logs=1, door_openings=2, door_open_steps=3,
                                 highwayman_damage=1, highwayman_hits=1, highwayman_defeated=1, cape_pickups=1,
                                 cape_worn=1), census.counts
    assert policy.counts == dict(census.counts, agent_hits=1, agent_defeats=1, productions=2, xp_events=2, level_ups=1)


def test_the_respawn_cycle_life_plays_all_its_actions_in_both_engines(monkeypatch):
    """The respawn-cycle life (sim/native_harness.py RESPAWN_LIFE: the wider pilot's update-24 checkpoint reached it on the
    C engine) on its training pool and pack under the real name table: both engines play all 319 actions alike. Its last
    decision's tick holds the chicken life's npc_respawn event and the agent's hit on that life, with no xp event: the
    hit now owes none."""
    import names
    life = H.RESPAWN_LIFE
    if not (DATA_NC2.exists() and POOL_V2.exists() and native.PACK_V2.exists() and names.DEFAULT_TABLE.exists()):
        pytest.skip('private v2 archive, pool, pack or name table absent')
    monkeypatch.delenv('RL_NAME_TABLE')
    names.name_table.cache_clear()
    inputs = H.Inputs(pool=str(POOL_V2), archive=str(DATA_NC2), task=FREE, against='native', pack=str(native.PACK_V2),
                      xp_rate=1, muggers=False, life='["uniform", 6000, 36000]')
    a, b = H.build(inputs, 1, first=life['slot'])
    result = H.run(a, b, H.Replay([[x] for x in life['actions']]), life['key'], len(life['actions']),
                   episode=life['episode'])
    assert result.divergence is None, result.divergence.text()
    assert (result.steps, result.stopped, result.gap) == (len(life['actions']), '', ''), (result.steps, result.stopped)
    last = [(kind, d) for t, kind, d in a.env.tasks[0].world.events if t == 5782]
    assert ('npc_respawn', dict(npc=11, type=41, life=18)) in last and any(
        kind == 'hit' and (d['npc'], d['life'], d['damage']) == (11, 18, 1) for kind, d in last), last
    assert not any(kind == 'xp' for kind, _ in last), last


def test_village_world_v2_draws_no_mugger_tiles(name_table_path, tmp_path):
    """draw_combat in village-world-v2 places no muggers, so it draws no mugger tile: on a copy of a v2 layout whose
    walkable tiles lie within 5 of its camp centre (where v1's draw raises 'tiles for 1 muggers'), both engines draw the
    same starts and spawns, with no mugger."""
    if not (DATA_NC2.exists() and POOL_V2.exists() and SUPPLY.exists()):
        pytest.skip('private v2 archive, v2 pool or supply pool absent')
    from dataclasses import replace
    from names import NameTable
    from sim import combat, search, world as W
    from sim.native_export import export
    from sim.task import draw_life, draw_start
    from sim.workers import load_tables
    tables = load_tables(DATA_NC2)
    good = W.load_pool(POOL_V2)[0][0]
    cx, cz = np.mean(combat.camp(good, tables), axis=0)
    x, z = np.indices(good.walkable.shape)
    start = np.zeros_like(good.start)
    start[tuple(np.argwhere(good.start)[0])] = True
    near = np.maximum(np.abs(x - cx), np.abs(z - cz)) <= 5
    short = replace(good, seed=good.seed + 900_000, walkable=good.walkable & (near | start), start=start)
    layouts = [good, short]
    with pytest.raises(ValueError, match='tiles for 1 muggers'):   # v1's draw on this layout
        combat.draw_combat(5, 0, 0, short, draw_start(5, 0, 0, layouts, layout=1), tables, group='danger')
    pool = W.save_pool(tmp_path / 'pool.npz', layouts)
    supply = W.save_pool(tmp_path / 'supply.npz', W.load_pool(SUPPLY)[0][:1], version=search.SUPPLY['world'])
    pack = native.open_pack(export(tmp_path / 'short.pack', pool, supply, DATA_NC2, names=NameTable(name_table_path),
                                   trees=False, variants=1, log=lambda _: None), names=NameTable(name_table_path))
    layouts = W.load_pool(pool)[0]
    drawn = 0
    for key in range(200):
        k = draw_start(key, 0, 0, layouts)['layout']
        want = combat.draw_village(key, 0, 0, layouts, tables, False)
        want['length'] = draw_life(key, 0, 0, ('fixed', 300))
        raw, _, spawns = pack.episode_draw(native.TASKS[FREE], 0, key, 0, 0, 0, 0, 300, 300)
        assert native.decode_start(raw) == want and want['village']['muggers'] == [], key
        assert [s[1] for s in spawns] == [n['type'] for n in combat.village_npcs(layouts[k], tables, [])], key
        drawn += k == 1
    assert drawn >= 50, drawn


def test_a_village_world_v2_soak_run_reports_its_world_and_what_it_reached(exported_v2):
    """free_soak_run on the v2 files: equal throughout, and its row names the world, the layouts its lives played and the
    census of what they reached (picks and the highwayman among it)."""
    pool, pack = exported_v2
    row = H.free_soak_run((str(pack), str(pool), str(DATA_NC2), False, 1, ['uniform', 100, 800], 'random', 3, 8, 200))
    assert row['divergence'] is None and row['gap'] is None and row['stopped'] is None, row
    assert row['world'] == 'village-world-v2' and len(row['layouts']) >= 4, row['layouts']
    assert row['census']['picks'] > 0 and row['census']['highwayman_damage'] > 0, row['census']
