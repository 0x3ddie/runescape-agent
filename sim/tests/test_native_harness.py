"""The equivalence harness (sim/native_harness.py): a CpuBatch against its twin reports no divergence under
every action source and scenario start; a perturbed copy is reported at the decision and tick of the perturbation, in
the layer it touched (state record, draw log, observation); the repro line rebuilds CpuBatch(n=1, first=slot) at the
reported episode and replays it to the same ticks; and the canonical record reads the oracle's own state (varp order,
script frames and their suspension sites) as rs.h names it."""
import argparse
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
