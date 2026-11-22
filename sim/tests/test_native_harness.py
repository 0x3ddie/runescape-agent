"""The equivalence harness (sim/native_harness.py): a CpuBatch against its twin reports no divergence under
every action source and scenario start; a perturbed copy is reported at the decision and tick of the perturbation, in
the layer it touched (state record, draw log, observation); the repro line rebuilds CpuBatch(n=1, first=slot) at the
reported episode and replays it to the same ticks; and the canonical record reads the oracle's own state (varp order,
script frames and their suspension sites) as rs.h names it."""
import argparse
import re
import shlex

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import native_harness as H
from sim.build import DATA

ROOT = DATA.parents[3]
POOL, SUPPLY = ROOT / 'logs/sim/world/train-8-nc1-derived.npz', ROOT / 'logs/supply/pools/train-128.npz'
KEY = 7


@pytest.fixture(scope='module')
def tables():
    if not DATA.exists():
        pytest.skip('simulator tables logs/sim/data/draynor-nc1.npz absent (docs/RUNNING.md says how to build them)')
    from sim.workers import load_tables
    return load_tables(DATA)  # check_pins=False: the archive's pin check fails without vendor/rs-sdk


@pytest.fixture(scope='module')
def pools():
    if not POOL.exists() or not SUPPLY.exists():
        pytest.skip('training pools absent (docs/RUNNING.md)')
    from sim.world import load_pool
    layouts, derived = load_pool(POOL)
    supply, none = load_pool(SUPPLY)
    return layouts, derived, (supply[:4], none[:4])


@pytest.fixture(scope='module')
def v2(tables, pools):
    """A v2 batch with a short horizon, so episodes end and auto-reset within a few decisions."""
    from sim.vec import CpuBatch
    layouts, derived, _ = pools
    return CpuBatch(tables, layouts, 4, horizon=40, derived=derived)


@pytest.fixture(scope='module')
def v5(tables, pools):
    from sim.vec import CpuBatch
    layouts, derived, supply = pools
    return CpuBatch(tables, layouts, 4, derived=derived, task='goal-activity-v5', search=supply)


def sides(env, scenario=None, perturb=None):
    return H.PythonSide(env, scenario=scenario), H.PythonSide(H.twin(env), scenario=scenario, perturb=perturb,
                                                              name='perturbed' if perturb else 'twin')


@pytest.mark.parametrize('source', ['random', 'biased'])
def test_a_batch_has_no_divergence_from_its_twin(v2, v5, source):
    for env in (v2, v5):
        a, b = sides(env)
        result = H.run(a, b, H.make_source(source, KEY), KEY, 40)
        assert result.divergence is None, result.divergence.text()
        assert result.traced and result.ticks > 40 * env.n and (result.steps == 40 or 'unsupported' in result.stopped)


def test_episodes_end_and_restart_alike(v2):
    a, b = sides(v2)
    result = H.run(a, b, H.RandomLegal(3), 3, 30)
    assert result.divergence is None, result.divergence.text()
    assert result.episodes >= v2.n and max(a.episodes) >= 2


@settings(max_examples=6, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(0, 2 ** 31 - 1), st.lists(st.integers(0, 95), min_size=1, max_size=40))
def test_recorded_action_lists_have_no_divergence(v2, key, values):
    a, b = sides(v2)
    result = H.run(a, b, H.Listed(values), key, -(-len(values) // v2.n) + 2)
    assert result.divergence is None, result.divergence.text()


@pytest.mark.parametrize('scenario', ['danger', 'hurt', 'under', 'scarce'])
def test_scenario_starts_reach_their_start_on_both_sides(v5, scenario):
    a, b = sides(v5, scenario=scenario)
    result = H.run(a, b, H.Biased(KEY), KEY, 12)
    assert result.divergence is None, result.divergence.text()
    starts = dict(zip(v5.goals, b.starts))
    if scenario in ('danger', 'hurt'):
        assert starts['melee']['combat']['group'] == H.SCENARIOS[scenario]['group']
        assert scenario != 'danger' or starts['melee']['combat']['muggers']
        assert scenario != 'hurt' or starts['melee']['combat']['hp'] == 2
    elif scenario == 'under':
        assert all(s['levels']['woodcutting'] == 14 for s in b.starts)
    else:
        assert starts['wood']['variant'] == 'scarce'


def perturbed_at(world_index, decision, change):
    """A perturbation of world `world_index`'s second cycle of its decision `decision` (task.decisions, counted in
    the episode that runs it); it records the tick and episode it fired at."""
    fired = {}

    def perturb(trace, step, world, agent, task):
        if trace.b != world_index or getattr(task, 'decisions', 0) != decision or fired.get('cycles', 0) > 1:
            return
        fired['cycles'] = fired.get('cycles', 0) + 1
        if fired['cycles'] == 2:
            change(world, agent)
            fired.update(tick=world.tick, episode=trace.episode, step=step)  # Row.tick: the tick after the cycle
    return perturb, fired


def test_a_state_change_is_reported_at_its_decision_and_tick(v2):
    from sim.engine import ATTACK
    perturb, fired = perturbed_at(1, 4, lambda world, agent: world.player.stats.__setitem__(
        ATTACK, world.player.stats[ATTACK] + 10))
    a, b = sides(v2, perturb=perturb)
    result = H.run(a, b, H.RandomLegal(KEY), KEY, 30)
    d = result.divergence
    assert d is not None and fired['tick'] > 0
    assert (d.where, d.local, d.step, d.decision, d.tick, d.episode) == ('tick', 1, fired['step'], 4, fired['tick'],
                                                                         fired['episode'])
    assert any(line.startswith(f'RS_CF_PLAYER_STATS[{ATTACK}].xp:') for line in d.detail), d.detail
    assert len(d.actions) == min(H.HISTORY, fired['step'] + 1) and d.tails['perturbed'][-1].tick == fired['tick']


def test_a_draw_without_effect_is_found_in_the_draw_log(v2):
    perturb, fired = perturbed_at(2, 3, lambda world, agent: world.rng(99, world.tick))
    a, b = sides(v2, perturb=perturb)
    d = H.run(a, b, H.RandomLegal(KEY), KEY, 30).divergence
    assert d is not None and (d.where, d.local, d.step, d.tick) == ('draws', 2, fired['step'], fired['tick'])
    assert f'(99, {fired["tick"]})' in d.detail[0]


def test_an_observation_difference_with_equal_ticks_is_placed_in_the_observation_layer(v2):
    a, b = sides(v2)
    step = b.step

    def flipped(state, action):  # one bit of one feature of world 3, at the sixth decision
        out = step(state, action)
        if b.step_index == 5:
            out[1]['prog_vec'][3, 2] = np.nextafter(out[1]['prog_vec'][3, 2], np.float32(2))
        return out
    b.step = flipped
    d = H.run(a, b, H.RandomLegal(KEY), KEY, 30).divergence
    assert d is not None and (d.where, d.local, d.step) == ('observation', 3, 5)
    assert d.detail[0].startswith('prog_vec[2]:') and d.tick > 0


def test_the_repro_rebuilds_the_world_at_its_episode_and_replays_it(tables, pools, v2):
    """The report's repro line names the world slot, key, episode and its actions; CpuBatch(n=1, first=slot) rebuilt
    there and replayed ends on the tick hashes the batch's world had (the oracle's starts are keyed by slot and
    episode, so one world reproduces alone)."""
    from sim.engine import ATTACK
    from sim.vec import CpuBatch
    perturb, fired = perturbed_at(1, 3, lambda world, agent: world.player.levels.__setitem__(ATTACK, 0))

    def later(trace, step, world, agent, task):  # in a later episode than the first
        if trace.episode >= 1:
            perturb(trace, step, world, agent, task)
    a, b = sides(v2, perturb=later)
    d = H.run(a, b, H.RandomLegal(KEY), KEY, 60, repro=H.Inputs(pool=str(POOL), horizon=40).repro(KEY)).divergence
    assert d is not None and d.where == 'tick' and d.episode == fired['episode'] >= 1 and d.decision == 3
    parser = argparse.ArgumentParser()
    H.add_arguments(parser)
    args = parser.parse_args(shlex.split(d.repro.split(' diff ', 1)[1]))
    assert (args.slot, args.key, args.episode, args.horizon) == (d.slot, KEY, d.episode, 40)
    actions = [int(v) for v in args.actions.split(',')]
    assert len(actions) == 3 and actions == d.actions[-3:]
    layouts, derived, _ = pools
    one = H.PythonSide(CpuBatch(tables, layouts, 1, horizon=40, derived=derived, first=args.slot))
    one.reset(KEY)
    one.take()
    state, obs = one.start_episode(0, args.episode)
    for action in actions:
        state, obs, *_ = one.step(state, np.asarray([action], np.int32))
    replayed = {row.tick: row.crc for row in one.traces[0].tail}
    batch = {row.tick: row.crc for row in a.tail(d.local) if row.episode == d.episode}
    assert d.tick in replayed and d.tick in batch and all(replayed[t] == crc for t, crc in batch.items() if t in replayed)


def test_the_tick_record_is_taken_after_the_episodes_watch(v2):
    """rs.h's hash point: after the watched cycle, so the terminal cycle's Row shows the terminal at the end tick (a
    record taken inside the watch, before its check, would never show one: no cycle follows the terminal)."""
    a, b = sides(v2)
    a.reset(KEY)
    rows, state, obs, source = [], a.env._state(), a.obs, H.RandomLegal(KEY)
    for step in range(12):
        state, obs, _, done, _ = a.step(state, source(step, obs, a.first + np.arange(a.n), None))
        rows += [row for world in a.take() for row in world]
    ends = [H.decode(row.record)['RS_CF_EPISODE'] for row in rows]
    terminal = [(row.tick, int(e[3])) for row, e in zip(rows, ends) if e[5]]
    assert terminal and all(tick == end for tick, end in terminal)


def test_the_varps_follow_the_players_own_order(v2):
    from sim.engine import Player
    p = Player(v2.tasks[0].world, 'x')
    assert list(p.varps) == list(H.header()[1])


def test_state_outside_the_contract_is_refused(v2):
    a, _ = sides(v2)
    a.reset(KEY)
    task = v2.tasks[0]
    task.agent.player.varps['unlisted'] = 1
    with pytest.raises(H.CanonError, match='unlisted'):
        a.canon(task.world, task.agent, task)
    del task.agent.player.varps['unlisted']


def test_script_frames_name_their_scripts_and_suspension_sites(v2):
    """Generators of the oracle's own scripts, suspended: prospect at its two delays, a mining closure (first swing on
    a normal rock: its free variables pick the rs.h id) inside its level refusal's mesbox."""
    from sim import engine
    a, _ = sides(v2)
    a.reset(KEY)
    ids = H.contract()[1]
    for task in v2.tasks:
        world, p = task.world, task.agent.player
        rocks = [loc for loc in world.locs if world.t.loc_types[loc.type]['triggers'].get('OPLOC1', '').startswith(
            '[oploc1,_mining_rock_')]
        if not rocks:
            continue
        loc = rocks[0]
        prospect = engine.SCRIPTS[world.t.loc_types[loc.type]['triggers']['OPLOC2']](world, p, loc)
        assert H.frame_chain(prospect) == [(ids['RS_S_PROSPECT'], 0)]
        next(prospect)
        assert H.frame_chain(prospect) == [(ids['RS_S_PROSPECT'], 1)]
        next(prospect)
        assert H.frame_chain(prospect) == [(ids['RS_S_PROSPECT'], 2)]
        trigger = world.t.loc_types[loc.type]['triggers']['OPLOC1']
        mining = engine.SCRIPTS[trigger](world, p, loc)
        p.levels[engine.MINING] = 0  # below every rock: [proc,mesbox](mine_level), the script's second site
        assert next(mining) == ('pausebutton',)
        kind = 'FAST' if trigger.endswith('fast]') else 'NORMAL'
        assert H.frame_chain(mining) == [(ids[f'RS_S_MINE_FIRST_{kind}'], 2), (ids['RS_S_MESBOX'], 1)]
        return
    pytest.skip('no rock in these layouts')


def test_policy_actions_are_sampled_alike_and_fed_to_both_sides(v2):
    """A MaskedRecurrentPolicy (config.LEARNER's network, untrained, seeded) as the action source: its seeded draws
    repeat, it acts only legally, and both sides stay equal under it."""
    torch = pytest.importorskip('torch')
    from baseline import config
    from baseline.adapter import observation_space
    from baseline.policy import MaskedRecurrentPolicy, load_name_vectors
    from gymnasium import spaces
    from names import name_table
    from obs import OBSERVATION
    a, b = sides(v2)
    _, obs = a.reset(KEY)
    torch.manual_seed(0)
    space = observation_space({k: v[0] for k, v in obs.items()})
    policy = MaskedRecurrentPolicy(space, spaces.Discrete(96), lambda _: 3e-4, names=name_table().spec,
                                   observation=OBSERVATION, **config.LEARNER['policy_kwargs'])
    load_name_vectors(policy, name_table().vectors)
    starts = np.ones(v2.n, bool)
    first, again = (H.PolicySource(policy, space.spaces, seed=5)(0, obs, np.arange(v2.n), starts) for _ in range(2))
    assert np.array_equal(first, again) and all(obs['mask'][w, first[w]] for w in range(v2.n))
    result = H.run(a, b, H.PolicySource(policy, space.spaces, seed=5), KEY, 15)
    assert result.divergence is None and result.steps == 15, result.divergence and result.divergence.text()


def test_a_native_side_that_has_not_ported_a_path_is_a_gap_not_a_divergence(tmp_path, v2):
    """A native call that fails with RS_NOT_PORTED (the module's Error for an unported C path, here raised by the
    batch's reset) stops the run as a gap, never a divergence."""
    from sim import native
    from sim.tests.test_native import Names, _pack
    path, _ = _pack(tmp_path / 'batch.pack', rows=True)
    batch = native.NativeBatch(native.open_pack(path, names=Names('d' * 16)), v2.n, task='goal-activity-v2')
    side, m = H.NativeSide(batch), native.load()
    error = m.Error(3, m.exceptions().index('EXC_NOT_PORTED'), [row[0] for row in m.reasons()].index('RS_E_NOT_PORTED'),
                    (0,) * 6, 'rs_batch_reset', -1)
    real = batch._batch

    class Unported:
        def __getattr__(self, name):
            return getattr(real, name)

        def reset(self, key):
            raise error
    batch._batch = Unported()
    result = H.run(H.PythonSide(v2), side, H.RandomLegal(KEY), KEY, 5)
    assert result.divergence is None and 'native' in result.gap and 'rs_batch_reset' in result.gap


def test_a_v11_batch_on_its_bound_pool_has_no_divergence_from_c(tables, pools, name_table_path, tmp_path):
    """goal-activity-v11 (mine only) on its bound pool, compact-world-v1 train seeds 1,000-1,127, held as a plain pool
    (no derived tables) and exported without trees, so Python computes each route tree as queried and C searches it
    live; one supply layout, which v11 needs and never plays. Its 8 worlds under random and biased play, episodes ending
    and restarting, report no divergence (the training equality run is this at 1,024 worlds)."""
    from names import NameTable
    from sim import search, world as W
    from sim.native_export import export
    pool = W.save_pool(tmp_path / 'v11.npz', [W.generate(seed, tables) for seed in W.split_seeds('train', 128, 1000)])
    supply = W.save_pool(tmp_path / 'supply.npz', pools[2][0][:1], version=search.SUPPLY['world'])
    pack = export(tmp_path / 'v11.pack', pool, supply, DATA, names=NameTable(name_table_path), trees=False, variants=2,
                  log=lambda _: None)
    inputs = H.Inputs(pool=str(pool), archive=str(DATA), supply=str(supply), task='goal-activity-v11', goals='mine',
                      against='native', pack=str(pack))
    a, b = H.build(inputs, 8, tables_=tables)
    for source in ('random', 'biased'):
        result = H.run(a, b, H.make_source(source, KEY), KEY, 60)
        assert result.divergence is None and not result.gap and not result.stopped, (result.divergence, result.gap,
                                                                                      result.stopped)
        assert result.traced and result.episodes >= 8 and all(goal == 'mine' for goal in a.env.goals)


def wood_pair(tables, pools, name_table_path, tmp_path):
    """Python and C sides of wood specialists' worlds on 8 training layouts, exported without trees."""
    from names import NameTable
    from sim import search, world as W
    from sim.native_export import export
    pool = W.save_pool(tmp_path / 'wood.npz', [W.generate(seed, tables) for seed in W.split_seeds('train', 8)])
    supply = W.save_pool(tmp_path / 'supply.npz', pools[2][0][:1], version=search.SUPPLY['world'])
    pack = export(tmp_path / 'wood.pack', pool, supply, DATA, names=NameTable(name_table_path), trees=False, variants=2,
                  log=lambda _: None)
    inputs = H.Inputs(pool=str(pool), archive=str(DATA), task='goal-activity-v2', goals='wood', against='native',
                      pack=str(pack))
    return H.build(inputs, 8, tables_=tables)


def test_held_v2_an_inventory_option_ends_a_chop_alike_in_python_and_c(tables, pools, name_table_path, tmp_path):
    """held-v2 in both engines: wood specialists' worlds on 8 training layouts, exported without trees, under
    keyed play weighted toward ACT, which takes a legal HELD row (no item selected) at every decision where the Python
    player is not delayed and its interaction is a loc's, a chop that goes on. Each such option ends that interaction
    on the Python side, and the native side's tick records (the player's target among them) equal Python's."""
    from sim.agent import A_HELD0, N_HELD
    from sim.engine import Loc
    a, b = wood_pair(tables, pools, name_table_path, tmp_path)
    biased, taken, ended = H.Biased(KEY, dict(act=30.0, dialog=1e6), 'chop'), [], []

    def source(step, obs, slots, starts):
        actions = biased(step, obs, slots, starts)
        for w in range(len(actions)):
            agent, held = a.env.tasks[w].agent, np.flatnonzero(obs['mask'][w, A_HELD0:A_HELD0 + N_HELD])
            if len(held) and agent.using is None and not agent.player.delayed and isinstance(agent.player.target, Loc):
                actions[w] = A_HELD0 + held[step % len(held)]
                taken.append((step, w, agent))
        return actions

    def observe(step, out_a, out_b):
        ended.extend(not isinstance(agent.player.target, Loc) for s, w, agent in taken
                     if s == step and a.env.tasks[w].agent is agent)
    result = H.run(a, b, source, KEY, 300, observe=observe)
    assert result.divergence is None and not result.gap and not result.stopped, (result.divergence, result.gap,
                                                                                  result.stopped)
    assert len(ended) >= 40 and all(ended), (len(taken), ended)


def test_a_use_on_a_held_item_with_no_script_says_so_alike_in_python_and_c(tables, pools, name_table_path, tmp_path):
    """A use with no script in both engines: keyed play weighted toward USE and HELD (with an item selected, every HELD row is a use on
    another held item) makes uses with no script, after which OpHeldUHandler's debug line and "Nothing interesting
    happens." reach the chat; the native side's tick records (chat included) equal Python's."""
    from sim.agent import NOTHING_INTERESTING
    a, b = wood_pair(tables, pools, name_table_path, tmp_path)
    said, seen = [], {}

    def observe(step, out_a, out_b):
        for task in a.env.tasks:
            log = task.world.chat_log
            said.extend(text for _, text in log[seen.get(id(task.world), 0):])
            seen[id(task.world)] = len(log)
    result = H.run(a, b, H.Biased(KEY, dict(use=30.0, held=30.0, dialog=1e6), 'use'), KEY, 120, observe=observe)
    assert result.divergence is None and not result.gap and not result.stopped, (result.divergence, result.gap,
                                                                                  result.stopped)
    pairs = sum(t.startswith('No trigger for [opheldu,') and u == NOTHING_INTERESTING for t, u in zip(said, said[1:]))
    assert pairs >= 10, pairs


# ---- the gate's verdict ----------------------------------------------------------------------------------------
COMMIT = 'c0ffee' * 6 + 'c0ff'  # 40 hex digits: a clean commit
PROVENANCE = dict(files={k: dict(path=f'/inputs/{k}', sha256=f'{k}-sha256') for k in ('pack', 'archive', 'pool', 'supply')},
                  pack_inputs=dict(tables='archive-sha256', pool='pool-sha256', supply_pool='supply-sha256'),
                  pack_names=None, library='_native-0123456789abcdef.so', env={})


def part_row(name):
    """A passed part's row (gate_parts): the soak's counts, the policy part's, or no failures."""
    if name == 'soak':
        return dict(id='soak', target=1, ticks=1, free_target=1, free_ticks=1, errors=[], divergences=0, mismatches=0,
                    gaps=0, stopped=[])
    if name == 'policy':
        return dict(id='policy', failures=[], compared=10, divergences=0, gaps=0, stopped=[])
    return dict(id=name, failures=[])


def gate_rows(mode, only=None, world=None):
    """A clean gate's result rows as gate_main writes them (gate_row): one row a case of `mode` that compared its
    decisions and the world set's parts (gate_parts), those `only` matches, of one clean commit and one set of inputs;
    with `world` (village-world-v2's set) each row names it."""
    import re
    world = world or H.GATE_WORLDS[0]
    rows = [dict(id=H.case_id(task, goals, first, source, scenario), task=task, goals=goals, first=first, source=source,
                 scenario=scenario, runs=1, decisions=10, compared=10, world_decisions=80, ticks=100, episodes=1,
                 divergences=0, gaps=0, stopped=[])
            for task, goals, first in H.gate_configs(mode, world) for source, scenario in H.gate_cases(task, goals)]
    rows += [part_row(name) for name in H.gate_parts(mode, world)]
    rows = [row for row in rows if only is None or re.search(only, row['id'])]
    return [H.gate_row(row, COMMIT, mode, only, world, dict(PROVENANCE, world=world)) for row in rows]


def summary(rows, expected=None, parts=None):
    """(clean, the verdict line, every line) of gate_summary."""
    lines = []
    return H.gate_summary(rows, lines.append, expected, parts), lines[-1], lines


def test_a_case_that_stops_alike_on_an_exception_fails_the_gate(v2):
    """A run that stops because both sides raised the same exception compared none of its remaining decisions: the
    gate is not clean. A run stopped by an unsupported world, flagged or raised alike, passes."""
    a, b = sides(v2)
    for side in (a, b):
        step = side.step

        def cut(state, action, step=step, side=side):
            if side.step_index == 3:
                raise RuntimeError("the agent's limits cut a free-play life (free_play_cfg)")
            return step(state, action)
        side.step = cut
    result = H.run(a, b, H.RandomLegal(KEY), KEY, 20)
    assert result.divergence is None and result.steps == 3 and result.stopped, result
    case = H.CaseResult('x', 'goal-activity-v2', 'all', 0, 'random', None)
    case.add(result, KEY)
    assert not H.unsupported_stop(case.stopped[0]) and case.compared == 3
    rows = gate_rows('quick')
    assert summary(rows)[0]
    for text, clean in ((case.stopped[0], False), ('key 1: both sides raised AuditError at step 4: xp', False),
                        ('key 1: world slot 3 reached unsupported content at step 4, alike on both sides', True),
                        ('key 1: both sides raised Unsupported at step -1: a setup', True)):
        rows[0]['stopped'] = [text]
        assert summary(rows)[0] is clean, text
    rows = gate_rows('full')
    soak = next(row for row in rows if row['id'] == 'soak')
    soak['stopped'] = ['free-play-v1 key 1 slot 2 episode 0: both sides raised RuntimeError at step 9: closed']
    assert not summary(rows)[0] and '1 stopped on an exception' in summary(rows)[1]


HIT = "{'npc': 11, 'type': 41, 'life': 18, 'source': 'wd1a0', 'damage': 1, 'hp': 3, 'style': 1, 'used': set()}"
XP = "{'stat': 2, 'xp': 40, 'before': 4640, 'after': 4680, 'cause': 'hit', 'npc': 11, 'life': 17}"


@pytest.mark.parametrize('oracle,c_text', [
    (f'hit at tick 5782 ({HIT}) has no xp event for stats [2, 3]', 'hit at tick 5782 has no xp event'),
    ("production at tick 9 ({'obj': 1511, 'count': 1, 'script': '[label,get_logs]', 'src': 1276}) has no xp event",
     'production at tick 9 has no xp event'),
    ("bones buried at tick 3 ({'obj': 526, 'count': 1, 'why': 'bury'}) have no xp event",
     'bones buried at tick 3 has no xp event'),
    (f'xp event at tick 5724 ({XP}): the tables give 80 at rate 1',
     'xp event at tick 5724 (stat 2): the tables give another amount at its rate'),
    (f'xp event at tick 5724 ({XP}): the stat held 4600, not 4640',
     "xp event at tick 5724 (stat 2): the stat's XP before is not its checked XP"),
    (f'xp event at tick 5724 ({XP}): production by [label,get_logs] trains no such skill',
     'xp event at tick 5724 (stat 2): production by a script that trains no such skill'),
    (f'xp event at tick 5724 ({XP}): a hit in style 3 gives that stat no more XP',
     'xp event at tick 5724 (stat 2): a hit in its style gives that stat no more XP'),
    (f'xp event at tick 5724 ({XP}): no hit by this agent on that npc life in its tick',
     'xp event at tick 5724 (stat 2): no hit by this agent on that npc life in its tick'),
    ('inventory holdings not explained by the events: {1511: 2} != {1511: 3}',
     'inventory holdings not explained by the events'),
    ('paid 1.25, the fractional level gain 1.2500021', 'paid differs from the fractional level gain'),
])
def test_a_check_failure_reads_alike_when_the_c_text_is_the_oracle_s_cut(oracle, c_text):
    """The C text of a check failure whose rs.h reason carries less than the oracle's is the oracle's text cut to the
    reason's fields (audit_cut): the two raise alike ('cut'). Another tick, cause, stat, place or reason, another class,
    or a cut of anything but a check failure is not alike; one class and one text is ('text')."""
    from sim.audit import AuditError
    assert H.audit_cut(oracle) == c_text and H.raised_alike(AuditError(oracle), AuditError(c_text)) == 'cut'
    assert H.raised_alike(AuditError(c_text), AuditError(c_text)) == 'text'
    assert H.raised_alike(AuditError(oracle), RuntimeError(c_text)) is None
    assert H.raised_alike(RuntimeError(oracle), RuntimeError(c_text)) is None
    another_cause = re.sub(r'^(hit|production|bones buried) at',
                           lambda m: {'hit': 'production', 'production': 'bones buried', 'bones buried': 'hit'}[m[1]] + ' at',
                           c_text)
    for other in (re.sub(r'tick (\d+)', lambda m: f'tick {int(m[1]) + 1}', c_text), another_cause,
                  c_text.replace('(stat 2)', '(stat 3)'), c_text.replace('inventory', 'bank'), c_text + '.'):
        if other != c_text:
            assert H.raised_alike(AuditError(oracle), AuditError(other)) is None, other


def test_a_check_failure_reads_alike_against_the_text_the_c_engine_renders():
    """The cuts against the texts the C module renders for each reason that carries less than the oracle (rs.h
    RS_E_AUDIT_XP with each XpCheck reason C words, RS_E_AUDIT_XP_OWED, RS_E_AUDIT_HOLDINGS, RS_E_AUDIT_PAID), so a
    change of a C form or of an oracle text breaks the comparison here."""
    from sim import native
    from sim.audit import AuditError
    module = native.load()
    reason = {name: k for k, (name, *_) in enumerate(module.reasons())}

    def rendered(name, args, text=None):
        return native.as_python(module.Error(1, module.exceptions().index('EXC_AUDIT'), reason[name],
                                             tuple(args) + (0,) * (6 - len(args)), text, 0))
    pairs = [(f'xp event at tick 5724 ({XP}): {why}', rendered('RS_E_AUDIT_XP', (5724, 2), c_why))
             for why, c_why in (('the tables give 80 at rate 1', 'the tables give another amount at its rate'),
                                ('the stat held 4600, not 4640', "the stat's XP before is not its checked XP"),
                                ('production by [label,get_logs] trains no such skill',
                                 'production by a script that trains no such skill'),
                                ('a hit in style 3 gives that stat no more XP',
                                 'a hit in its style gives that stat no more XP'),
                                ('no cause', 'no cause'))]
    pairs += [(f'hit at tick 5782 ({HIT}) has no xp event for stats [2, 3]', rendered('RS_E_AUDIT_XP_OWED', (5782,), 'hit')),
              ("bones buried at tick 3 ({'obj': 526}) have no xp event",
               rendered('RS_E_AUDIT_XP_OWED', (3,), 'bones buried')),
              ('paid 1.25, the fractional level gain 1.2500021', rendered('RS_E_AUDIT_PAID', ())),
              *((f'{place} holdings not explained by the events: {{1: 2}} != {{1: 3}}',
                 rendered('RS_E_AUDIT_HOLDINGS', (), place)) for place in ('inventory', 'bank', 'worn'))]
    for oracle, c_error in pairs:
        assert type(c_error) is AuditError and H.raised_alike(AuditError(oracle), c_error) == 'cut', (oracle, c_error)


def test_gate_only_policy_selects_the_part_alone_when_anchored():
    """gate --only searches row ids: the policy part's id is also in the village-world-v1 cases whose source is a policy
    (policy:A, policy:S), so 'policy' selects those cases and the part, and '^policy$' the part alone."""
    assert H.gate_expected('full', only='^policy$') == {'policy'}
    assert H.gate_expected('quick', only='^threads$|^policy$') == {'threads', 'policy'}
    both = H.gate_expected('quick', only='policy')
    assert 'policy' in both and len(both) > 1 and all(name == 'policy' or '|policy:' in name for name in both)


def raising_at(side, step_index, text, world=None):
    """`side` with its step raising AuditError(text) after the cycles of its step `step_index` (its rows taken); with
    `world`, the error names it as a C error names its failing world."""
    from sim.audit import AuditError
    step = side.step

    def raise_after(state, action):
        out = step(state, action)
        if side.step_index == step_index:
            error = AuditError(text)
            if world is not None:
                error.world = world
            raise error
        return out
    side.step = raise_after
    return side


def test_a_check_failure_cut_alike_stops_the_run_only_when_the_step_s_rows_agree(v2):
    """Both sides raise a check failure in one step, the oracle's text and its cut (as the C engine words it): the run
    stops alike, the C text kept in the stop, when the raising step's tick rows (records, draws, events, chat) agree; a
    draw that only one side made in that step is reported as the divergence it is, and a C text that is not the cut is
    an exception divergence."""
    oracle, cut = f'hit at tick 40 ({HIT}) has no xp event for stats [2, 3]', 'hit at tick 40 has no xp event'
    a, b = sides(v2)
    result = H.run(raising_at(a, 3, oracle), raising_at(b, 3, cut), H.RandomLegal(KEY), KEY, 20)
    assert result.divergence is None and result.steps == 3, result.divergence and result.divergence.text()
    assert result.stopped.endswith(f'{oracle} (C text: {cut})') and not H.unsupported_stop(result.stopped)
    a, b = sides(v2)
    result = H.run(raising_at(a, 3, oracle), raising_at(b, 3, 'hit at tick 41 has no xp event'), H.RandomLegal(KEY),
                   KEY, 20)
    assert result.divergence is not None and result.divergence.where == 'exception' and not result.stopped
    perturb, fired = perturbed_at(2, 4, lambda world, agent: world.rng(99, world.tick))
    a, b = sides(v2, perturb=perturb)
    result = H.run(raising_at(a, 3, oracle), raising_at(b, 3, cut), H.RandomLegal(KEY), KEY, 20)
    d = result.divergence
    assert fired.get('step') == 3 and d is not None and not result.stopped, (fired, result)
    assert (d.where, d.local, d.step, d.tick) == ('draws', 2, 3, fired['tick']), d.text()
    a, b = sides(v2)  # the C side names world 1, while the oracle stepped every world (it raised at a later one)
    result = H.run(raising_at(a, 3, oracle), raising_at(b, 3, cut, world=1), H.RandomLegal(KEY), KEY, 20)
    assert result.divergence is not None and result.divergence.where == 'exception' and not result.stopped


def test_a_policy_part_that_failed_or_compared_nothing_fails_the_gate():
    """The policy part (gate_policy) is clean only with no failure (its pins, the respawn-cycle life), no divergence,
    gap or stop on an exception, and some decision compared; its line and census are in the summary."""
    for world in H.GATE_WORLDS:
        rows = gate_rows('quick', world=world)
        assert summary(rows)[0]
        policy = next(row for row in rows if row['id'] == 'policy')
        for change, words in ((dict(failures=['respawn-cycle life: 318 of 319 actions']), '0 divergences'),
                              (dict(divergences=1), '1 divergences'), (dict(gaps=1), '1 gaps'),
                              (dict(stopped=['key 1 first 0: both sides raised AuditError at step 4: x']),
                               '1 stopped on an exception'), (dict(compared=0), '1 compared nothing')):
            changed = [dict(row, **change) if row is policy else row for row in rows]
            clean, line, _ = summary(changed)
            assert not clean and words in line, (world, change, line)
    rows = gate_rows('quick', world=H.GATE_WORLDS[1])
    policy = next(row for row in rows if row['id'] == 'policy')
    policy.update(plan='wider_pilot', worlds=4, decisions=200, runs=1, world_decisions=800, ticks=12000, episodes=2,
                  deaths=2, files=dict(checkpoint=dict(path='/p/c.zip', sha256='ffd81cf4' * 8),
                                       pool=dict(path='/p/pool.npz', sha256='x')),
                  census=dict(agent_hits=9), found_life=dict(steps=319, actions=319, stopped='', divergence=None, gap=''))
    clean, _, lines = summary(rows)
    assert clean and any(line.startswith('[gate] policy: wider_pilot c.zip (ffd81cf4ffd81cf4) on pool.npz: 1 runs, 800 '
                                         "world decisions (10 of 200 decisions compared)")
                         and "respawn-cycle life: 319 of 319" in line for line in lines), lines
    assert '[gate] policy reached: agent_hits 9; deaths 2' in lines


def test_a_case_that_compared_no_decision_fails_the_gate(v2):
    """Every run of a case stopped alike before its first decision (an unsupported start, say) compared nothing: the case
    fails the gate however its stops read, and its rows say how many decisions each case compared (compared_steps: a
    divergence's own decision and a reset's divergence compare none)."""
    from sim.engine import Unsupported
    a, b = sides(v2)

    def unsupported(key):
        raise Unsupported('a start')
    for side in (a, b):
        side.reset = unsupported
    result = H.run(a, b, H.RandomLegal(KEY), KEY, 20)
    assert result.steps == 0 and H.unsupported_stop(result.stopped), result
    case = H.CaseResult('x', 'goal-activity-v2', 'all', 0, 'random', None)
    case.add(result, KEY)
    assert case.compared == 0 and case.decisions == 0
    for steps, step, compared in ((5, 4, 4), (5, 5, 5), (0, -1, 0), (3, 3, 3)):
        found = H.Result(steps=steps, divergence=H.Divergence('tick', 0, 0, KEY, 'wood', 0, step, 0, 0, [], []))
        assert H.compared_steps(found) == compared, (steps, step)
    # a world flagged unsupported at step 0 still compared that decision (every world's outputs, its flag among them):
    # 1 a run; a step that raised Unsupported alike compared none: 0 a run, the case fails
    flagged = H.Result(steps=1, stopped='world slot 3 reached unsupported content at step 0, alike on both sides')
    raised = H.Result(steps=0, stopped='both sides raised Unsupported at step 0: a fight')
    assert (H.compared_steps(flagged), H.compared_steps(raised)) == (1, 0)
    rows = gate_rows('quick')
    rows[0].update(compared=0, stopped=['key 1: both sides raised Unsupported at step -1: a start'])
    clean, line, lines = summary(rows)
    assert not clean and '1 compared nothing' in line and f'[gate] compared no decision: {rows[0]["id"]}' in lines
    del rows[0]['compared']  # a row without the count (written before it) compared nothing it can show
    assert not summary(rows)[0]


def test_the_gate_holds_every_case_it_declares():
    """A merged gate is clean only with a row for every case of its mode's configurations (gate_configs x gate_cases)
    and every part of its world set (gate_parts), and none outside them; a rerun (--only) claims the cases and parts its
    pattern matches; a shard's own summary its shard's."""
    for mode in ('quick', 'full'):
        rows = gate_rows(mode)
        clean, line, _ = summary(rows)
        assert clean and f' 0 of {len(rows)} rows missing' in line, line
        free = [k for k, row in enumerate(rows) if row.get('task') == 'free-play-v1']
        clean, line, _ = summary([row for k, row in enumerate(rows) if k != free[-1]])
        assert not clean and f'1 of {len(rows)} rows missing' in line, line
        assert not summary(rows + [dict(rows[0], id='goal-activity-v2|all|99|random|-')])[0]
        assert not summary(rows + [dict(rows[-1], mode='quick' if mode == 'full' else 'full')])[0]
    rerun = gate_rows('full', only='x25-none-long')
    assert len(rerun) == 6 and summary(rerun)[0] and not summary(rerun[1:])[0]
    assert summary(gate_rows('full') + rerun)[0]  # a whole gate with a rerun's rows beside it
    assert summary(gate_rows('full', only='soak|threads'))[0]
    shards = [H.gate_expected('quick', (i, 4), parts=['cases']) for i in range(4)]
    assert set().union(*shards) == H.gate_expected('quick', parts=['cases'])
    assert sum(map(len, shards)) == len(H.gate_expected('quick', parts=['cases']))
    rows = gate_rows('quick')
    mine = [row for row in rows if row['id'] in shards[1]]
    assert summary(mine, shards[1])[0] and not summary(mine[1:], shards[1])[0] and summary([], set())[0]


def test_a_gate_verdict_needs_every_part_one_clean_commit_and_one_set_of_inputs():
    """Each world set's gate claims its parts for its mode (village-world-v1: digests, threads, in the full gate the soak,
    and the policy part; village-world-v2: the training workers, threads, the full gate's soak and the policy part): a
    merge without one reads NOT CLEAN
    unless its summary names the parts it claims (--parts). Its rows must all be of one commit of 40 hex digits (not
    '-dirty', not 'unknown', not empty, not two) and carry one provenance whose pack records the archive, pool and supply
    pool they played and whose world is theirs."""
    import copy
    for world in H.GATE_WORLDS:
        for mode in ('quick', 'full'):
            rows = gate_rows(mode, world=world)
            parts = H.gate_parts(mode, world)
            assert parts == [('digests' if world == H.GATE_WORLDS[0] else 'workers'), 'threads',
                             *(['soak'] if mode == 'full' else []), 'policy']
            assert summary(rows)[0]
            for part in parts:
                rest = [row for row in rows if row['id'] != part]
                clean, _, lines = summary(rest)
                assert not clean and f'[gate] missing: {part}' in lines, (world, mode, part)
                named = ['cases', *(p for p in parts if p != part)]
                assert summary(rest, parts=named)[0], (world, mode, part)  # a merge that names what it claims
    rows = gate_rows('quick')
    for commits in (['', ''], [COMMIT + '-dirty'] * 2, ['unknown'] * 2, [COMMIT, 'f' * 40], [None, None]):
        changed = [dict(row, commit=commits[k % 2]) for k, row in enumerate(rows)]
        clean, _, lines = summary(changed)
        assert not clean and any('not of one clean commit' in line for line in lines), commits
    other = copy.deepcopy(PROVENANCE)
    other['files']['pack']['sha256'] = 'another pack'
    for changed, words in (([dict(rows[0], provenance=dict(other, world='village-world-v1'))] + rows[1:], 'sets of inputs'),
                           ([dict(row, provenance=None) for row in rows], 'no inputs'),
                           ([dict(row, provenance=dict(row['provenance'], pack_inputs=dict(PROVENANCE['pack_inputs'],
                                                                                         pool='x'))) for row in rows],
                            'another pool'),
                           ([dict(row, provenance=dict(row['provenance'], world='village-world-v2')) for row in rows],
                            'the inputs are of village-world-v2')):
        clean, _, lines = summary(changed)
        assert not clean and any(words in line for line in lines), words


def test_a_summary_that_names_its_parts_claims_them_alone_and_says_so():
    """gate --summary --parts narrows a merge's claim to the parts it names: it must name some, each a part of the rows'
    world set and mode ('cases' or gate_parts), and the verdict line says what it claims and leaves out. Free play's census
    and deaths are summed in the summary."""
    rows = [row for row in gate_rows('quick') if row['id'] != 'digests']
    clean, line, _ = summary(rows, parts=['cases', 'threads'])
    assert clean and line.endswith('; claims cases, threads (leaves out digests, policy)'), line
    for parts in (['cases', 'workers'], ['cases', 'soak']):  # not parts of village-world-v1's quick gate
        clean, _, lines = summary(rows, parts=parts)
        assert not clean and any('does not have' in text for text in lines), parts
    for text in ('', ',', 'cases,soaks'):
        with pytest.raises(ValueError):
            H.summary_parts(text)
    free = [k for k, row in enumerate(rows) if row.get('task') == 'free-play-v1'][:2]
    for k in free:
        rows[k] = dict(rows[k], census=dict(picks=2, door_openings=1), deaths=3)
    _, _, lines = summary(rows, parts=['cases', 'threads'])
    assert '[gate] free play reached: picks 4, door_openings 2; deaths 6' in lines


def test_a_resume_keeps_one_gate_in_its_file(tmp_path):
    """gate --resume: a last line that the interruption cut short is dropped and its case runs again; any other unreadable
    line, or a row of another commit, mode, world set or inputs, refuses the resume (the file stays one gate's)."""
    import json
    rows = gate_rows('quick')[:3]
    out = tmp_path / 'gate.jsonl'
    out.write_text(''.join(json.dumps(row) + '\n' for row in rows[:2]) + json.dumps(rows[2])[:40])
    said = []
    before = H.resume_rows(out, said.append)
    assert before == rows[:2] and out.read_text().count('\n') == 2 and 'left unfinished' in said[0]
    provenance = rows[0]['provenance']
    assert H.resume_done(before, out, COMMIT, 'quick', H.GATE_WORLDS[0], provenance) == {r['id'] for r in rows[:2]}
    for commit, mode, world, inputs in ((COMMIT + '-dirty', 'quick', H.GATE_WORLDS[0], provenance),
                                        (COMMIT, 'full', H.GATE_WORLDS[0], provenance),
                                        (COMMIT, 'quick', 'village-world-v2', provenance),
                                        (COMMIT, 'quick', H.GATE_WORLDS[0], dict(provenance, library='other.so'))):
        with pytest.raises(ValueError, match='give a new --out'):
            H.resume_done(before, out, commit, mode, world, inputs)
    moved = H.copy.deepcopy(provenance)  # the same files elsewhere (another checkout): the same inputs
    moved['files']['pool']['path'] = '/elsewhere/pool'
    assert H.resume_done(before, out, COMMIT, 'quick', H.GATE_WORLDS[0], moved) == {r['id'] for r in rows[:2]}
    assert summary(gate_rows('quick')[:-1] + [dict(gate_rows('quick')[-1], provenance=moved)])[0]
    out.write_text('{"id": 1}\nnot a row\n{"id": 2}\n')
    with pytest.raises(ValueError, match='line 2'):
        H.resume_rows(out, said.append)


def test_a_resume_after_any_cut_keeps_the_whole_rows_and_a_file_of_lines(tmp_path):
    """Wherever an interruption cut the last row (none of it written, any part of it, all but its newline, all of it:
    every offset), a resume reads exactly the whole rows before the cut, and a row appended after it gives a file whose
    every line is a row."""
    import json
    rows = gate_rows('quick')[:3]
    text = ''.join(json.dumps(row) + '\n' for row in rows)
    out = tmp_path / 'gate.jsonl'
    for cut in range(len(text) - len(json.dumps(rows[2])) - 1, len(text) + 1):
        out.write_text(text[:cut])
        before = H.resume_rows(out, lambda line: None)
        whole = rows if cut >= len(text) - 1 else rows[:2]
        assert before == whole, cut
        with out.open('a') as stream:
            stream.write(json.dumps(dict(id='next')) + '\n')
        assert [json.loads(line) for line in out.read_text().splitlines()] == [*whole, dict(id='next')], cut


def test_the_commit_outside_a_checkout_is_unknown(tmp_path, monkeypatch):
    """_commit names HEAD (with -dirty for a changed tracked file); outside a checkout it is 'unknown', which no
    verdict accepts."""
    assert H.COMMIT.fullmatch(H._commit().removesuffix('-dirty'))
    monkeypatch.setattr(H.native, 'ROOT', tmp_path)
    assert H._commit() == 'unknown'


def test_village_world_v2_is_a_gate_set_of_its_own():
    """The gate's village-world-v2 set (gate --world village-world-v2) is opt-in: the default set's cases are the 636
    (full) and 208 (quick) the recorded gates hold, its rows name no world and stay clean as before; v2's set is free play
    alone (no goal task, no muggers: the four FREE_CONFIGS_V2, long lives in the full gate only), shares no case with v1,
    and no --only pattern of v1's long or short configurations matches a v2 case. Rows that name v2 claim v2's set; rows
    of both sets are not one gate."""
    import re
    v2 = 'village-world-v2'
    assert H.GATE_WORLDS[0] == 'village-world-v1'
    cases = ['cases']
    assert (len(H.gate_expected('full', parts=cases)), len(H.gate_expected('quick', parts=cases))) == (636, 208)
    full, quick = H.gate_expected('full', world=v2, parts=cases), H.gate_expected('quick', world=v2, parts=cases)
    assert (len(full), len(quick)) == (60, 16) and quick < full and not full & H.gate_expected('full')
    assert {(t, g) for t, g, _ in H.gate_configs('full', v2)} == {('free-play-v1', name) for name in H.FREE_CONFIGS_V2}
    assert all(not c['muggers'] for c in H.FREE_CONFIGS_V2.values())
    for pattern in (*H.FREE_CONFIGS, 'none', 'muggers'):
        assert not [name for name in full if re.search(pattern, name)], pattern
    for mode in ('quick', 'full'):
        rows = gate_rows(mode, world=v2)
        assert summary(rows)[0] and f'0 of {len(rows)} rows missing' in summary(rows)[1]
        assert H.claimed_cases(rows) == H.gate_expected(mode, world=v2) == {row['id'] for row in rows}
        assert not summary(rows[1:])[0]                                   # a v2 case missing
        assert H.claimed_cases(rows + gate_rows(mode)) is None             # two world sets are not one gate
        assert not summary(rows + [dict(rows[0], id='free-play-v1|x1-none|0|random|-')])[0]   # a v1 case in v2's rows
    assert H.claimed_cases(gate_rows('full')) == H.gate_expected('full')  # rows that name no world: v1's, as before
    with pytest.raises(ValueError, match='world sets'):
        H.gate_configs('full', 'compact-world-v1')


# The default set's case ids in run order and its shards (k = 4), as the gate before the world sets wrote them (sha256
# prefixes, computed earlier): a reorder or a renamed case changes the shard a case runs in and the recorded rows.
V1_ORDER = {'full': (636, 'f7a541cb337a32ca', '2922a1dd309bce70'), 'quick': (208, '8845617be3d5ea0c', '1f8696b902e8397a')}


def test_the_default_gate_set_keeps_its_order_rows_and_command():
    """village-world-v1's set keeps its case ids in run order and its shard assignment; its rows name no world (v2's
    do) and carry their provenance; the command's defaults are its inputs and its parts as before (every part but the
    training workers, which it refuses), and village-world-v2's command defaults to its own inputs and every part but the
    digests, refuses a digests part named, keeps inputs given and names its results file by the world."""
    import hashlib
    for mode, (n, order, shards) in V1_ORDER.items():
        ids = [H.case_id(t, g, f, s, sc) for t, g, f in H.gate_configs(mode) for s, sc in H.gate_cases(t, g)]
        split = [sorted(H.gate_expected(mode, (i, 4), parts=['cases'])) for i in range(4)]
        assert (len(ids), hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:16],
                hashlib.sha256(repr(split).encode()).hexdigest()[:16]) == (n, order, shards), mode
    row = H.gate_row(dict(id='threads', failures=[]), 'c', 'full', None, 'village-world-v1', PROVENANCE)
    assert 'world' not in row and row['provenance'] == PROVENANCE
    assert H.gate_row(dict(id='threads', failures=[]), 'c', 'full', None, 'village-world-v2')['world'] == 'village-world-v2'

    def settings(*argv):
        parser = argparse.ArgumentParser()
        H.gate_arguments(parser)
        args = parser.parse_args(list(argv))
        return H.gate_settings(args), args
    (mode, world, parts, shard, out), args = settings()
    assert (mode, world, parts, shard, out.name) == ('full', 'village-world-v1',
                                                     ['cases', 'digests', 'threads', 'soak', 'policy'], (0, 1),
                                                     'gate-full.jsonl')
    assert (args.archive, args.pool, args.pack) == tuple(str(H.GATE_INPUTS[k]) for k in ('archive', 'pool', 'pack'))
    (mode, world, parts, shard, out), args = settings('--quick', '--world', 'village-world-v2', '--shard', '1/4')
    assert (mode, world, parts, out.name) == ('quick', 'village-world-v2',
                                              ['cases', 'threads', 'soak', 'workers', 'policy'],
                                              'gate-quick-village-world-v2-1of4.jsonl')
    assert (args.archive, args.pool, args.pack) == tuple(str(H.GATE_INPUTS_V2[k]) for k in ('archive', 'pool', 'pack'))
    assert settings('--world', 'village-world-v2', '--pool', 'mine.npz')[1].pool == 'mine.npz'
    for parts in ('cases,digests,threads,soak', 'digests'):
        with pytest.raises(ValueError, match='no digests part'):
            settings('--world', 'village-world-v2', '--parts', parts)
    with pytest.raises(ValueError, match='no workers part'):
        settings('--parts', 'cases,workers')
    with pytest.raises(ValueError, match='unknown parts'):
        settings('--parts', 'cases,soaks')