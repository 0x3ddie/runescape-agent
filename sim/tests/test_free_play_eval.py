"""free-play-v1's evaluation in the Python reference: the loader of
a policy without a goal, the play command (sim/hour.py free) and the statistics and verdict from its rows alone
(sim/gate.py free). The claims are tested on rows with known answers under a test plan; the wiring on a tiny checkpoint
that the training command saves, on development records with short lives."""
import fcntl
import json
import os
import re
import sys

from hypothesis import given, settings, strategies as st
import pytest

from baseline.compare import paired_bootstrap
from sim import free
from sim.engine import Unsupported
from sim.gate import (FREE as RULES, FREE_PLAN, FREE_PLAN_SETTINGS, FREE_SHARED, free_json, free_plan_differs, free_plan_run,
                      free_plan_world,
                      main as gate_main)
from sim.hour import (FREE, FreePolicyActor, FreeStop, free_jobs, free_life, free_run, load_free_policy, main as hour_main,
                      trial_seed)
from sim.task import FREE_PLAY, FreePlayEpisode

S = FREE_PLAY['skills']
TINY = 1  # the tiny checkpoint's training seed
CONTROLS = ('wait', 'random', 'wood', 'fish', 'mine', 'melee', 'chickens', 'once', 'eat', 'rotation')
SEED = 101
PASS_SEEDS = dict(sampled1=11, sampled2=22, greedy=11)  # greedy plays sampled1's seed (sim/hour.py free_run)
CLEAN = dict(commit='c', changed=[], untracked=[], patch=None)  # one commit with no change: the outputs' and the runs'
TABLES = '307b77a7a58327782bc25654127048968d12c9e88ef1a74c221ca60b3d6c5771'  # sha256 of sim/build.py DATA at the freeze
# The pilot's plan as frozen on 12 November 2026, with the digests computed at the freeze: a known answer for
# sim/gate.py FREE['plans']['pilot'].
PILOT = dict(split='development', first=4000, records=64, key=223, xp_rate=1, muggers=False, ticks=36000, read=6000,
             period=4500, control_seed=4253, tables=TABLES,
             run=dict(decisions=4194304, worlds=1024, n_steps=64, batch_size=1024, n_epochs=4, learning_rate=3e-4,
                      gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lstm_hidden_size=128,
                      half_life_ticks=1000, value_norm=dict(method='popart', beta=0.1, min_std=0.01), muggers=False,
                      life=['uniform', 6000, 36000], pool='1cccffbede7f2613b5baf31e9f436a22bc84202eb3ed99023c33e0cd553d9469',
                      tables=TABLES, validate=True),
             start={77: '96b1f9872b62bef7034b8502b227273ece84f6c0d988d50e3ccc6b8223105d65'},
             pass_seeds=dict(sampled1=229, sampled2=233), policy_seeds=[77], claims=['L'])
# The wider world's pilot plan as frozen on 14 November 2026, with the digests computed at the freeze: a known answer
# for sim/gate.py FREE['plans']['wider_pilot']. Its learner values are the first pilot's; its run pins the C engine
# and its pack.
V2_TABLES = '7dee885e663b89956f2416262582d34f8438caa3eeb6a3421b57bdad862eba19'  # sha256 of logs/sim/data/draynor-nc2.npz
WIDER = dict(world='village-world-v2', split='development', first=7000, records=64, key=239, xp_rate=1, muggers=False,
             ticks=36000, read=6000, period=4500, control_seed=4253, tables=V2_TABLES,
             run=dict(PILOT['run'], pool='56e2607b7a0e5a7e4a19797a13b6b61774709adf69523d7ba6faf3ced8df69bb',
                      tables=V2_TABLES, engine='native',
                      # pack was 97640806ca29025e565bafd37bdd2fe6e3160700a47b02339761659fc08b683f, moved
                      # on 20 November 2026; the 34 sections are byte-equal
                      pack='21f147bfa83868ec5d5de456502cec7c97e2e72d875a45f337319dc5900cd04c'),
             start={79: 'e355680184117f730fef1d9f3c6050c60c21922ca33218c5b5c4aad35410eb6c'},
             pass_seeds=dict(sampled1=251, sampled2=257), policy_seeds=[79], claims=['L'])
# The five seeds' plan as frozen on 16 November 2026, with the start digests made at the freeze: a known answer for
# sim/gate.py FREE['plans']['wider_final']. Its run is the wider pilot's.
WIDER_FINAL = dict(WIDER, split='final', first=8000, records=256, key=241,
                   start={101: '5dc9abefeb3349d6c37cb3dcb3d97bfa52e26df027bb00ab8efd1e00d67695e5',
                          202: '18ce6031a43d9207acd86220c12bbacdd075e728c051a8c8891058a99a011642',
                          303: '1be77d8c94ee4f97a6221cb90e78a1f5ece4e109c278a154ef86e522dbb65293',
                          404: '8be523c0832a682e9278ef15860e8e7ea06068d1754caf148ca43f85677f09ba',
                          505: '7e71409a474ed3d76ec45463c0b92f6d7558b2a175348d476a93573d1dff04e9'},
                   policy_seeds=[101, 202, 303, 404, 505], claims=['L', 'B', 'R', 'no_rescue'])
# A training run's facts in the builders' manifests, in summary.json's JSON form (a plan writes them as Python literals)
RUN = dict({k: 1 for k in FREE['run_facts']}, life=['uniform', 6000, 36000], half_life_ticks=1000.0,
           value_norm=dict(method='popart', beta=0.1, min_std=0.01))


# ---- rows with known answers (no tables) ----------------------------------------------------------------------------
def life(actor, record, gain, pass_name=None, policy_seed=None, death=None, read=None, unsupported=None, **reads):
    """A row as sim/hour.py free_life writes it, with only what the verdict reads."""
    gain = {k: v for k, v in gain.items() if v}
    seed = [5, record] if pass_name is None else PASS_SEEDS[pass_name] * 1000 + record
    return dict(dict(actor=actor, record=record, pass_name=pass_name, policy_seed=policy_seed, seed=seed, gain=gain,
                     gain_s=sum(v for k, v in gain.items() if k in S), death=death, unsupported=unsupported, ticks=36000,
                     read=read or dict(gain_s=0, died=False), made={}, kills={}, damage={}, eat_log=[], eats_full=0,
                     activity=[]), **reads)


def manifest(records, seed=None, period=None, status='completed', run=None, **changes):
    """A sim.hour free manifest: a policy seed's (latest.zip of a completed run whose summary records its digest) or,
    with no seed, the controls'. `run` changes the policy's run facts."""
    policy = None if seed is None else dict(
        seed=seed, policy=dict(sha256=f'p{seed}', policy_sha256=f'd{seed}'), initial=dict(sha256=f'i{seed}'),
        run=dict(dict(status=status, checkpoint='latest.zip', policy_sha256_at_start=f's{seed}', policy_sha256_at_end=f'd{seed}',
                      facts=RUN, env_seed=seed, weights=None, init=None, env='cpu', engine='cpu', pack=None,
                      source=CLEAN, batch_in_effect=dict(world='village-world-v1')), **(run or {})))
    passes = [] if seed is None else [dict(name=p, seed=s, sampled=p != 'greedy') for p, s in PASS_SEEDS.items()]
    return dict(dict(task='free-play-v1', rows=FREE['rows'], world='village-world-v1', observation='obs-v6',
                     source=CLEAN, split='development', first=4000,
                     records=records, key=223, muggers=False, xp_rate=1, ticks=36000, read=6000, control_seed=5,
                     tables=dict(path='t', sha256='h'), parts=1, period=period, passes=passes, policy=policy), **changes)


def output(folder, lives, records, seed=None, period=None, order=None, **changes):
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps(manifest(records, seed, period, **changes)))
    rows = [dict({k: v for k, v in r.items() if k != 'pass_name'}, **{'pass': r['pass_name']}) for r in lives]
    if order is not None:
        order.shuffle(rows)
    (folder / 'lives-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return folder


GAINS = dict(wait={}, random=dict(woodcutting=10, attack=5), wood=dict(woodcutting=45), fish=dict(fishing=43),
             mine=dict(mining=42), melee=dict(attack=20, strength=20, defence=20, hitpoints=14),
             chickens=dict(attack=25, strength=25, defence=25, hitpoints=15), once=dict(woodcutting=1), eat={},
             rotation=dict(attack=10, strength=10, defence=10, hitpoints=10, woodcutting=20, fishing=15, mining=15))


def frozen(**changes):
    """A plan at the builders' settings, written as Python literals (tuples, an int half-life): the tests' own values
    (the study's plans are frozen in sim/gate.py FREE)."""
    seeds = changes.get('policy_seeds', (SEED,))
    start = {s: f's{s}' for s in seeds} if isinstance(seeds, (tuple, list)) else {}
    run = dict(RUN, life=tuple(RUN['life']), half_life_ticks=1000)
    return dict(dict(split='development', first=4000, records=4, key=223, xp_rate=1, muggers=False, ticks=36000,
                     read=6000, period=4500, control_seed=5, tables='h', run=run, start=start,
                     pass_seeds=dict(sampled1=11, sampled2=22), policy_seeds=(SEED,), claims=('L', 'B', 'R', 'no_rescue')),
                **changes)


def evaluation(tmp, policy, records=4, controls=None, initial=None, order=None, plans=None, status='completed', run=None,
               **changes):
    """A policy seed's output (or one a seed when `policy` maps policy seeds to passes) and the controls', read under
    `plans` (default: the final plan frozen at these settings and seeds; {} for FREE's own plans): a pass maps to a
    function of the record giving the row's keyword arguments (with 'gain'), `controls` a control to one (default GAINS),
    `initial` a sampled pass to one (default: the initialization gains nothing in both)."""
    by_seed = policy if all(type(k) is int for k in policy) else {SEED: policy}
    initial = dict(sampled1=constant({}), sampled2=constant({})) if initial is None else initial
    folders = []
    for seed, passes in by_seed.items():
        rows = [life(a, r, **p(r), pass_name=pass_name, policy_seed=seed) for a, by in (('policy', passes),
                                                                                       ('initial', initial))
                for pass_name, p in by.items() for r in range(records)]
        folders.append(output(tmp / f'policy{seed}', rows, records, seed=seed, order=order, status=status, run=run,
                              **changes))
    made = controls or {c: (lambda g: lambda r: dict(gain=g))(GAINS[c]) for c in CONTROLS}
    folders.append(output(tmp / 'controls', [life(c, r, **f(r)) for c, f in made.items() for r in range(records)], records,
                          period=4500, order=order, **changes))
    plans = dict(final=frozen(records=records, policy_seeds=tuple(by_seed))) if plans is None else plans
    with pytest.MonkeyPatch.context() as patch:
        for name, plan in plans.items():
            patch.setitem(RULES['plans'], name, plan)
        return free_json(folders, tmp / 'free.json')


def constant(gain, **extra):
    return lambda r: dict(gain=gain, **extra)


def words(result):
    return {k: result['verdict'][k] for k in ('L', 'B', 'R', 'no_rescue') if k in result['verdict']}


ALL = dict(woodcutting=30, mining=30, fishing=30, attack=30, strength=30, defence=30, hitpoints=30)
FIGHTS = dict(attack=60, strength=60, defence=60, hitpoints=50)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 120), st.integers(0, 120)), min_size=2, max_size=12), st.randoms())
def test_each_difference_is_the_projects_paired_bootstrap_by_record_whatever_the_rows_order(tmp_path_factory, pairs, rng):
    """Policy - random legal play is baseline/compare.py's paired bootstrap of the differences taken record by record
    (10,000 resamples, seed 17017), however the rows lie in the files, and Claim L reads its lower bound."""
    tmp = tmp_path_factory.mktemp('pairs')
    controls = {c: constant(GAINS[c]) for c in CONTROLS}
    controls['random'] = lambda r: dict(gain=dict(woodcutting=pairs[r][1]))
    by = lambda r: dict(gain=dict(woodcutting=pairs[r][0]))
    result = evaluation(tmp, dict(sampled1=by, sampled2=by, greedy=by), records=len(pairs), controls=controls, order=rng)
    expect = paired_bootstrap([p - c for p, c in pairs], draws=10_000, seed=17017)
    for p in ('sampled1', 'sampled2'):
        got = result['seeds'][str(SEED)]['passes'][p]
        assert got['differences']['random'] == expect == got['claims']['L']['difference']
        assert got['claims']['L']['holds'] is (expect['low'] > 0) and got['claims']['L']['threshold'] == 0
    assert result['seeds'][str(SEED)]['verdict']['L'] == ('pass' if expect['low'] > 0 else 'fail')
    assert RULES['statistics'] == dict(draws=10_000, seed=17017)


def test_a_policy_that_only_fights_passes_l_and_r_and_cannot_pass_claim_b(tmp_path):
    """Four of the seven skills: above every control in all, and 0 outside the fighter's own skills, where the fighter
    gains 0 too, so the second half of Claim B has a lower bound of 0. A policy that gains in every skill passes B; its
    prayer, outside S, counts in neither half."""
    fights = constant(FIGHTS)
    result = evaluation(tmp_path / 'a', dict(sampled1=fights, sampled2=fights, greedy=fights))
    claims = result['seeds'][str(SEED)]['passes']['sampled1']['claims']
    melee = claims['B']['references']['melee']
    assert melee['total']['holds'] and melee['outside']['difference'] == dict(
        mean=0.0, low=0.0, high=0.0, pairs=4, draws=10_000, seed=17017) and melee['outside']['holds'] is False
    assert all(claims['B']['references'][g]['holds'] for g in ('wood', 'fish', 'mine')) and claims['B']['holds'] is False
    assert words(result) == dict(L='pass', B='fail', R='pass', no_rescue='no match')
    assert result['verdict']['readings'] == ['learns, not shown beyond one activity'] == \
        result['seeds'][str(SEED)]['verdict']['readings']
    assert claims['L']['difference']['mean'] == 230 - 15 and claims['R']['difference']['mean'] == 230 - 90
    assert claims['no_rescue']['exploits']['chickens']['difference']['mean'] == 230 - 90
    every = constant(dict(ALL, prayer=40))
    result = evaluation(tmp_path / 'b', dict(sampled1=every, sampled2=every, greedy=every))
    assert words(result) == dict(L='pass', B='pass', R='pass', no_rescue='no match') and result['verdict']['readings'] == []
    p = result['seeds'][str(SEED)]['passes']['sampled2']
    melee = p['claims']['B']['references']['melee']
    assert melee['own'] == list(RULES['references']['melee']) and melee['outside']['difference']['mean'] == 90.0
    assert p['policy']['mean_gain'] == 210 and p['claims']['B']['references']['wood']['outside']['difference']['mean'] == 180
    timer = dict({c: constant(GAINS[c]) for c in CONTROLS}, rotation=constant(ALL))  # a rotation as good as the policy
    result = evaluation(tmp_path / 'c', dict(sampled1=every, sampled2=every, greedy=every), controls=timer)
    assert words(result) == dict(L='pass', B='pass', R='fail', no_rescue='no match')
    assert result['verdict']['readings'] == ['not shown better than a timer']


@pytest.mark.parametrize('matching', ('sampled1', 'sampled2'))
def test_a_policy_equal_to_an_exploit_script_in_either_pass_matches_under_no_rescue(tmp_path, matching):
    better = constant(ALL)
    policy = {p: constant(GAINS['chickens']) if p == matching else better for p in ('sampled1', 'sampled2', 'greedy')}
    result = evaluation(tmp_path, policy)
    rescue = result['seeds'][str(SEED)]['passes'][matching]['claims']['no_rescue']
    assert rescue['matches'] is True and rescue['exploits']['chickens']['matches'] is True
    assert rescue['exploits']['chickens']['difference']['low'] == 0.0
    assert not rescue['exploits']['eat']['matches'] and not rescue['exploits']['once']['matches']
    assert result['seeds'][str(SEED)]['verdict']['no_rescue'] == 'match' == result['verdict']['no_rescue']
    assert 'the return is cheap in this world' in result['verdict']['readings']


def test_a_claim_is_read_in_each_pass_and_never_pooled(tmp_path):
    """L holds in the first pass and fails in the second: the seed fails L. Greedy plays no part in a claim."""
    policy = dict(sampled1=constant(ALL), sampled2=constant(GAINS['random']), greedy=constant(ALL))
    result = evaluation(tmp_path, policy)
    passes = result['seeds'][str(SEED)]['passes']
    assert passes['sampled1']['claims']['L']['holds'] and passes['sampled2']['claims']['L']['holds'] is False
    assert result['seeds'][str(SEED)]['verdict']['L'] == 'fail' and 'claims' not in passes['greedy']
    assert passes['greedy']['differences']['random']['mean'] == 210 - 15 and 'initial' not in passes['greedy']['differences']


def test_a_missing_or_unsupported_life_leaves_a_claim_incomplete_never_a_result(tmp_path):
    every = constant(ALL)
    controls = {c: constant(GAINS[c]) for c in CONTROLS}
    controls['random'] = lambda r: dict(gain=GAINS['random'], unsupported=dict(file='sim/engine.py', line=1, message='x')
                                        if r == 2 else None)
    result = evaluation(tmp_path / 'a', dict(sampled1=every, sampled2=every, greedy=every), controls=controls)
    got = result['seeds'][str(SEED)]
    assert got['passes']['sampled1']['claims']['L'] == dict(difference=None, threshold=0, holds=None)
    assert got['verdict']['L'] == 'incomplete' and got['verdict']['B'] == 'pass'
    assert result['controls']['random']['complete'] is False and result['controls']['random']['unsupported'] == 1
    assert result['evidence']['unsupported'] == [dict(actor='random', policy_seed=None, pass_name=None, record=2,
                                                      site='sim/engine.py', line=1, message='x')]
    stopped = lambda r: dict(gain=ALL, unsupported=dict(file='f', line=2, message='y') if r == 0 else None)
    result = evaluation(tmp_path / 'b', dict(sampled1=every, sampled2=stopped, greedy=every))
    assert words(result) == dict(L='incomplete', B='incomplete', R='incomplete', no_rescue='incomplete')
    del controls['rotation']  # a control not played: its claim cannot be read
    result = evaluation(tmp_path / 'c', dict(sampled1=every, sampled2=every, greedy=every),
                        controls=dict(controls, random=constant(GAINS['random'])))
    assert result['verdict']['R'] == 'incomplete' and result['verdict']['L'] == 'pass'


def test_the_verdict_over_the_seeds_needs_every_declared_seed(tmp_path):
    """A claim passes only when it passes in every declared seed: one failing seed fails it, one incomplete seed or a
    declared seed with no output leaves it incomplete. Each verdict carries the policy's deaths a life."""
    every, weak = constant(ALL, death=dict(tick=9, killer='goblin')), constant(GAINS['random'])
    both = lambda second: {101: dict(sampled1=every, sampled2=every, greedy=every), 202: second}
    result = evaluation(tmp_path / 'a', both(dict(sampled1=every, sampled2=every, greedy=every)))
    assert result['verdict']['seeds'] == [101, 202]
    assert words(result) == dict(L='pass', B='pass', R='pass', no_rescue='no match')
    assert result['verdict']['deaths_a_life']['101'] == dict(sampled1=1, sampled2=1, greedy=1)
    result = evaluation(tmp_path / 'b', both(dict(sampled1=every, sampled2=weak, greedy=every)))
    assert result['seeds']['202']['verdict']['L'] == 'fail' and result['verdict']['L'] == 'fail'
    assert result['seeds']['202']['verdict']['deaths_a_life'] == dict(sampled1=1, sampled2=0, greedy=1)
    assert result['seeds']['101']['verdict']['no_rescue'] == 'no match' and result['verdict']['no_rescue'] == 'match'
    assert result['verdict']['readings'] == ['the return is cheap in this world']  # one matching seed
    result = evaluation(tmp_path / 'c', both(dict(sampled1=every)))  # no sampled2 and no greedy rows
    assert result['seeds']['202']['verdict']['L'] == 'incomplete' and words(result)['L'] == 'incomplete'
    assert result['complete']['greedy'] == {'101': True, '202': False}
    result = evaluation(tmp_path / 'd', {101: dict(sampled1=every, sampled2=every, greedy=every)},
                        plans=dict(final=frozen(policy_seeds=(101, 202))))
    assert result['verdict']['seeds'] == [101, 202] and result['seeds']['202']['missing'] is True
    assert words(result) == dict(L='incomplete', B='incomplete', R='incomplete', no_rescue='incomplete')
    assert result['complete'] == dict(wait=True, initial={'101': dict(sampled1=True, sampled2=True),
                                                          '202': dict(sampled1=False, sampled2=False)},
                                      greedy={'101': True, '202': False})
    full = dict(sampled1=every, sampled2=every, greedy=every)
    for name, extra, gaps in (('nowait', dict(controls={c: constant(GAINS[c]) for c in CONTROLS if c != 'wait'}), ['wait']),
                              ('noinit', dict(initial=dict(sampled1=constant({}))), ['initial sampled2']),
                              ('nogreedy', {}, ['greedy'])):
        result = evaluation(tmp_path / name, {101: full if gaps != ['greedy'] else dict(sampled1=every, sampled2=every)},
                            plans=dict(final=frozen(policy_seeds=(101,))), **extra)
        assert result['seeds']['101']['verdict']['incomplete_controls'] == gaps  # its claims hold, but its controls lack
        assert words(result) == dict(L='incomplete', B='incomplete', R='incomplete', no_rescue='incomplete')
        assert result['verdict']['readings'] == [] and result['seeds']['101']['passes']['sampled1']['claims']['L']['holds']
    for missing, claim in (('random', 'L'), ('rotation', 'R'), ('melee', 'B')):  # a claim's own control, named
        result = evaluation(tmp_path / f'no-{missing}', {101: full}, plans=dict(final=frozen(policy_seeds=(101,))),
                            controls={c: constant(GAINS[c]) for c in CONTROLS if c != missing})
        assert result['seeds']['101']['verdict']['incomplete_controls'] == [missing] and words(result)[claim] == 'incomplete'


def test_a_verdict_is_written_only_for_outputs_that_match_a_frozen_plan(tmp_path, monkeypatch):
    """The pilots' plans hold the values as frozen, each one a known answer; the final read's holds none
    yet. With every plan unfrozen (patched), every output is a description: the same numbers, every claim described and
    no claim word. A plan is matched only by its own settings, tables, seeds and pass seeds, from one commit with no
    change and the final checkpoint of a completed run with the plan's facts, started from the plan's start digest (its
    own start, the frozen file at its own seed, or the one relaunch; never other tensors); the pilot reads Claim L alone and
    describes the others; an XP rate other than 1 gives the differences only; a malformed plan is refused."""
    plain = lambda value: json.loads(json.dumps(value))  # the JSON form that a plan is compared in
    assert set(RULES['plans']) == {'final', 'pilot', 'wider_pilot', 'wider_final'}
    assert set(RULES['plans']['final']) == set(FREE_PLAN)
    for name, known in (('wider_pilot', WIDER), ('wider_final', WIDER_FINAL)):
        wider = RULES['plans'][name]
        assert set(wider) == set(known) == set(FREE_PLAN) and free_plan_world(wider) == 'village-world-v2'
        assert set(wider['run']) == set(known['run']) == set(FREE['run_facts']) | set(FREE['engine_facts'])
        for key, value in known.items():
            assert plain(wider[key]) == plain(value), (name, key)
        for fact, value in known['run'].items():
            assert plain(wider['run'][fact]) == plain(value), (name, fact)
    assert 'world' in FREE_PLAN  # a plan frozen from now names its world
    assert set(RULES['plans']['final'].values()) == {None}  # its decision budget and start digests come later
    pilot = RULES['plans']['pilot']
    assert set(pilot) == set(PILOT) == set(FREE_PLAN) - {'world'}  # frozen before the world was a plan setting: v1's
    assert free_plan_world(pilot) == 'village-world-v1'
    assert set(pilot['run']) == set(PILOT['run']) == set(FREE['run_facts'])
    assert FREE['first_world_archive'] == pilot['tables'] == pilot['run']['tables']  # the old records' archive, pinned
    for key, value in PILOT.items():  # value by value, and each run fact by itself
        assert plain(pilot[key]) == plain(value), key
    for fact, value in PILOT['run'].items():
        assert plain(pilot['run'][fact]) == plain(value), fact
    assert set(frozen()) == set(FREE_PLAN) - {'world'} and free_plan_world(frozen()) == 'village-world-v1'
    every = constant(ALL)
    policy = dict(sampled1=every, sampled2=every, greedy=every)
    result = evaluation(tmp_path / 'own', policy, plans=dict.fromkeys(RULES['plans'], dict.fromkeys(FREE_PLAN)))
    assert result['verdict']['kind'] == 'description' and words(result) == {} and result['verdict']['plan'] is None
    assert result['verdict']['why'] == ['plan final is not frozen', 'plan pilot is not frozen', 'plan wider_pilot is not frozen',
                                        'plan wider_final is not frozen']
    assert result['seeds'][str(SEED)]['verdict'] == dict(kind='description', deaths_a_life=dict(sampled1=0, sampled2=0,
                                                                                                 greedy=0))
    p = result['seeds'][str(SEED)]['passes']['sampled1']
    assert 'claims' not in p and p['described']['L']['would_hold'] is True  # the numbers stay, described
    named = lambda d: {k for k, v in d.items() for k in (k, *(named(v) if isinstance(v, dict) else ()))}
    assert not {'holds', 'matches'} & named(p['described']) and 'would_match' in named(p['described'])
    assert result['statement'] == RULES['statement'] and 'generalization' in result['statement']
    changed = dict(world='village-world-v2', split='final', first=1, records=5, key=7, xp_rate=25, muggers=True, ticks=1,
                   read=2, period=4000, control_seed=6)
    assert set(changed) == set(FREE_PLAN_SETTINGS)
    other = {'world': f"world, the run's world (seeds [{SEED}])"}  # a village-world-v1 run's outputs under a v2 plan
    cases = [*((k, {k: v}, other.get(k, k)) for k, v in changed.items()), ('tables', dict(tables='x'), 'tables'),
             ('run', dict(run=dict(RUN, decisions=2, life=['fixed', 9])), 'run (decisions, life)'),
             ('swapped', dict(pass_seeds=dict(sampled1=22, sampled2=11)), 'pass_seeds'),
             ('other', dict(pass_seeds=dict(sampled1=11, sampled2=23)), 'pass_seeds'),
             ('seeds', dict(policy_seeds=(202,), start={202: 's202'}), f'policy_seeds, start (seeds [{SEED}])'),
             ('start', dict(start={SEED: 'another'}), f'start (seeds [{SEED}])')]
    assert isinstance(frozen()['run']['life'], tuple)  # every verdict here reads a plan of Python literals
    for name, change, differ in cases:
        result = evaluation(tmp_path / name, policy, plans=dict(final=frozen(**change)))
        assert result['verdict']['kind'] == 'description' and result['verdict']['why'][0] == f'not plan final: {differ} differ'
    result = evaluation(tmp_path / 'sixth', {101: policy, 202: policy}, plans=dict(final=frozen()))  # one more seed
    assert result['verdict']['why'][0] == 'not plan final: policy_seeds, start (seeds [202]) differ'
    for name, run, why in (('unfinished', dict(status='started'), 'a checkpoint of a run that has not completed (seeds [101])'),
                           ('later', dict(checkpoint='checkpoints/ppo_16_steps.zip'),
                            "a checkpoint that is not its run's final (seed 101: checkpoints/ppo_16_steps.zip)"),
                           ('otherfinal', dict(policy_sha256_at_end='x'), "a checkpoint that is not its run's final (seed 101: latest.zip)"),
                           ('nodigest', dict(policy_sha256_at_end=None),
                            'a run summary that records no final checkpoint digest (policy_sha256_at_end; seeds [101])'),
                           ('envseed', dict(env_seed=5), 'seed 101: environment seed 5'),
                           ('noweights', dict(env_seed=SEED + 1_000_000), 'seed 101: environment seed 1000101'),
                           ('init', dict(init=dict(path='i')), 'seed 101: environment seed 101'),
                           ('nosource', dict(source=None), 'a run summary that records no source, as an older run'),
                           ('dirtyrun', dict(source=dict(CLEAN, changed=[' M sim/gate.py'], patch=dict(sha256='p'))),
                            'a run trained from a source that is not one commit with no change (seed 101:'),
                           ('elsewhere', dict(source=dict(CLEAN, commit='e')),
                            "a run trained at another commit than the outputs' c (seed 101: e)"),
                           ('notables', dict(facts=dict(RUN, tables=None), engine='c', pack=dict(sha256='k')),
                            "a run summary that records no training tables, as the fake environment's and an older C-engine "
                            "run's: a plan pins the archive a run trained on (seed 101: env cpu, engine c)")):
        result = evaluation(tmp_path / name, policy, run=run)
        assert result['verdict']['kind'] == 'description' and why in result['verdict']['why'][0]
    relaunch = evaluation(tmp_path / 'relaunch', policy, run=dict(env_seed=SEED + 1_000_000, weights=dict(sha256='w')))
    assert relaunch['verdict']['kind'] == 'verdict'  # the one relaunch from the frozen initialization (its start digest)
    fromfile = evaluation(tmp_path / 'fromfile', policy, run=dict(weights=dict(sha256='w')))
    assert fromfile['verdict']['kind'] == 'verdict'  # the frozen initialization's file at its own seed (its start digest)
    for name, run, why in (('trained', dict(env_seed=SEED + 1_000_000, weights=dict(sha256='w'), policy_sha256_at_start='o'),
                            f'not plan final: start (seeds [{SEED}]) differ'),  # a relaunch from trained weights
                           ('otherfile', dict(weights=dict(sha256='w'), policy_sha256_at_start='o'),
                            f'not plan final: start (seeds [{SEED}]) differ'),  # a file of other tensors at its own seed
                           ('tables', dict(facts=dict(RUN, tables='other')), 'not plan final: run (tables) differ'),
                           ('novalidate', dict(facts=dict(RUN, validate=None)), 'not plan final: run (validate) differ'),
                           ('world', dict(batch_in_effect=dict(world='village-world-v2')),
                            f"not plan final: the run's world (seeds [{SEED}]) differ"),
                           ('noworld', dict(batch_in_effect={}),  # names no world, trained over archive 1: no world's
                            f"not plan final: the run's world (seeds [{SEED}]) differ")):
        result = evaluation(tmp_path / f'other-{name}', policy, run=run)
        assert result['verdict']['kind'] == 'description' and result['verdict']['why'][0] == why
    from sim import hour
    pinned = FREE['first_world_archive']
    with monkeypatch.context() as patch:  # a record that names no world, over the first world's archive (the pilot's)
        patch.setattr(free, 'world_tables', lambda world: tmp_path / 'absent.npz')  # this checkout without that file
        hour.world_archive.cache_clear()
        old = evaluation(tmp_path / 'old-record', policy, run=dict(batch_in_effect={}, facts=dict(RUN, tables=pinned)),
                         plans=dict(final=frozen(run=dict(frozen()['run'], tables=pinned))))
    hour.world_archive.cache_clear()
    assert old['verdict']['kind'] == 'verdict' and old['verdict']['plan'] == 'final', old['verdict']
    unset = [dict(policy_seeds=()), dict(policy_seeds=[]), dict(claims=()), dict(claims=[]),  # no seed or no claim
             dict(run=dict(RUN, decisions=None)), dict(start={SEED: None}),  # a None at any depth
             dict(pass_seeds=dict(sampled1=11, sampled2=None)), dict(claims=('L', None))]
    for k, empty in enumerate(unset):
        result = evaluation(tmp_path / f'unset-{k}', policy, plans=dict(final=frozen(**empty)))
        assert result['verdict']['why'][0] == 'plan final is not frozen' and words(result) == {}
    controls = output(tmp_path / 'alone', [life(c, r, {}) for c in CONTROLS for r in range(4)], 4, period=4500)
    mine = output(tmp_path / 'mine', [life('policy', r, ALL, pass_name=p, policy_seed=SEED) for p in PASS_SEEDS for r in range(4)],
                  4, seed=SEED)
    monkeypatch.setitem(RULES['plans'], 'final', frozen())
    assert free_json([controls], tmp_path / 'f.json')['verdict']['why'][0] == 'no policy output: a plan reads one at least'
    why = free_json([mine], tmp_path / 'f.json')['verdict']['why']
    assert why[0] == 'no control output: a plan reads the controls of its records' and not any('period' in w for w in why)
    monkeypatch.undo()
    for name, source in (('dirty', dict(commit='c', changed=[' M sim/gate.py'], untracked=[], patch=dict(sha256='a'))),
                         ('nogit', dict(commit=None, error='not a git checkout'))):
        for plan in ('final', 'pilot'):
            if source['commit'] is None:  # outputs of no commit: no run's commit equals theirs, so a description says it
                why = evaluation(tmp_path / f'{name}-{plan}', policy, source=source,
                                 plans={plan: frozen(claims=('L',))})['verdict']['why'][0]
                assert why == "a run trained at another commit than the outputs' None (seed 101: c)"
                continue
            with pytest.raises(ValueError, match='one commit with no change'):
                evaluation(tmp_path / f'{name}-{plan}', policy, source=source, plans={plan: frozen(claims=('L',))})
        assert evaluation(tmp_path / f'{name}-own', policy, plans={}, source=source)['verdict']['kind'] == 'description'
    for name, change, why in (('claims', dict(claims=('L', 'X')), 'claims a tuple or list'),
                              ('string', dict(claims='L'), 'claims a tuple or list'),
                              ('seedint', dict(policy_seeds=77), 'policy_seeds a tuple or list'),
                              ('half', dict(pass_seeds=dict(sampled1=11)), 'pass_seeds of'),
                              ('same', dict(pass_seeds=dict(sampled1=11, sampled2=11)), 'two different whole numbers'),
                              ('startseeds', dict(start={202: 's202'}), 'start {policy seed'),
                              ('facts', dict(run={k: 1 for k in FREE['run_facts'] if k != 'pool'}), "run without the facts ['pool']"),
                              ('extra', dict(run=dict(RUN, extra=1)), "with ['extra'] beside them"),
                              ('json', dict(run=dict(RUN, life={1, 2})), 'run.life of a form that JSON holds'),
                              ('world', dict(world='village-world-v9'),
                               "world one of ['village-world-v1', 'village-world-v2']")):
        with pytest.raises(ValueError, match=f'plan final is malformed: it needs .*{re.escape(why)}'):
            evaluation(tmp_path / f'malformed-{name}', policy, plans=dict(final=frozen(**change)))
    with pytest.raises(ValueError, match="match the plans \\['final', 'pilot'\\]"):
        evaluation(tmp_path / 'both', policy, plans=dict(final=frozen(), pilot=frozen(claims=('L',))))
    result = evaluation(tmp_path / 'pilot', policy, plans=dict(pilot=frozen(claims=('L',))))
    assert result['verdict']['plan'] == 'pilot' and words(result) == dict(L='pass') and result['verdict']['readings'] == []
    p = result['seeds'][str(SEED)]['passes']['sampled2']
    assert set(p['claims']) == {'L'} and set(p['described']) == {'B', 'R', 'no_rescue'}
    result = evaluation(tmp_path / 'x25', policy, xp_rate=25, plans=dict(final=frozen(xp_rate=25)))
    assert result['verdict']['kind'] == 'description (25x, out of the training distribution)' and words(result) == {}
    passes = result['seeds'][str(SEED)]['passes']
    assert not {'claims', 'described'} & set(passes['sampled1']) and passes['sampled1']['differences']['random']['mean'] == 195


def test_a_plan_of_the_c_engine_pins_its_pack_and_the_inputs_the_pack_was_exported_from(tmp_path):
    """A plan whose run holds the engine facts pins the C engine and its pack's sha256 beside the run facts; one that
    holds the run facts alone (the pilot's form) pins the Python reference. Each plan reads only a run of its own engine
    and pack. A C-engine run is read only when its pack records the run's own tables and pool as the inputs it was
    exported from: an older summary that records no tables, or a pack exported from another archive or pool, is
    refused whatever the plan's values. A plan that names the engine facts in part, or the Python reference by them,
    is malformed."""
    every = constant(ALL)
    policy = dict(sampled1=every, sampled2=every, greedy=every)
    inputs = dict(tables=RUN['tables'], pool=RUN['pool'], supply_pool='q')  # a C run's pack, exported from its own inputs
    c_run = dict(engine='native', pack=dict(path='p', sha256='k', inputs=inputs, oracle_hash='o'))
    c_plan = lambda **changes: frozen(run=dict(frozen()['run'], engine='native', pack='k', **changes))
    assert evaluation(tmp_path / 'c', policy, run=c_run, plans=dict(final=c_plan()))['verdict']['kind'] == 'verdict'
    assert evaluation(tmp_path / 'cpu', policy, plans=dict(final=frozen()))['verdict']['kind'] == 'verdict'
    for name, run, plan, why in (
            ('python-under-c', None, c_plan(), 'not plan final: run (engine, pack) differ'),
            ('c-under-python', c_run, frozen(), 'not plan final: run (engine, pack) differ'),
            ('other-pack', dict(c_run, pack=dict(c_run['pack'], sha256='j')), c_plan(), 'not plan final: run (pack) differ'),
            ('other-pool', dict(c_run, pack=dict(c_run['pack'], inputs=dict(inputs, pool='x'))), c_plan(),
             "a C-engine run whose pack does not record its tables and pool as the inputs it was exported from "
             "(seed 101: the pack records {'tables': 1, 'pool': 'x'})"),
            ('other-tables', dict(c_run, pack=dict(c_run['pack'], inputs=dict(inputs, tables='x'))), c_plan(),
             "(seed 101: the pack records {'tables': 'x', 'pool': 1})"),
            ('no-inputs', dict(c_run, pack=dict(path='p', sha256='k')), c_plan(),
             "(seed 101: the pack records {'tables': None, 'pool': None})"),
            ('older', dict(c_run, facts=dict(RUN, tables=None)), c_plan(),
             'a run summary that records no training tables')):
        result = evaluation(tmp_path / name, policy, run=run, plans=dict(final=plan))
        assert result['verdict']['kind'] == 'description' and why in result['verdict']['why'][0], name
    for name, run, why in (('half', dict(frozen()['run'], engine='native'), "run without the facts ['pack']"),
                           ('cpu', dict(frozen()['run'], engine='cpu', pack='k'), "engine facts of the C engine ('native')"),
                           ('nopack', dict(frozen()['run'], engine='native', pack=7), "engine facts of the C engine ('native')")):
        with pytest.raises(ValueError, match=f'plan final is malformed: it needs .*{re.escape(why)}'):
            evaluation(tmp_path / f'malformed-{name}', policy, run=c_run, plans=dict(final=frozen(run=run)))


def test_each_pilots_outputs_find_their_own_frozen_plan_and_never_the_other_pilots(tmp_path, monkeypatch):
    """FREE's own plans, as frozen, over outputs at each pilot's values: the wider world's (village-world-v2 records 7,000
    to 7,063 under key 239, pass seeds 251 and 257, seed 79 trained on the C engine from the plan's start as a file
    (--weights at its own environment seed, the cloud GPU run's form), its pack exported from the run's own archive and
    pool) find wider_pilot and never pilot; the first world's (the pilot's values, its run on the Python reference
    from its own start) find pilot alone, as before. With its own plan unfrozen, each finds no
    plan, and the other pilot's plan names the world among what differs."""
    every = constant(ALL)
    policy = dict(sampled1=every, sampled2=every, greedy=every)
    pilots = {}
    for name, plan in (('wider_pilot', WIDER), ('pilot', PILOT)):  # outputs at the known answers' values
        plain = json.loads(json.dumps(plan))
        (seed, start), = plain['start'].items()
        seed, run = int(seed), plain['run']
        facts = {k: run[k] for k in FREE['run_facts']}
        engine = dict(engine='cpu', pack=None) if 'engine' not in run else dict(
            engine=run['engine'], pack=dict(path='p', sha256=run['pack'], oracle_hash='o',
                                            inputs=dict(tables=run['tables'], pool=run['pool'], supply_pool='q')))
        world = free_plan_world(plan)
        # a cloud GPU machine trains from this Mac's file
        weights = None if 'engine' not in run else dict(path='init.zip', sha256='f')
        record = dict(facts=facts, policy_sha256_at_start=start, batch_in_effect=dict(world=world), weights=weights, **engine)
        settings = dict(records=plain['records'], world=world, first=plain['first'], key=plain['key'],
                        control_seed=plain['control_seed'], tables=dict(path='t', sha256=plain['tables']))
        pilots[name] = (seed, policy, record, settings, plain['pass_seeds'])
    for name, (seed, passes, record, settings, pass_seeds) in pilots.items():
        with monkeypatch.context() as patch:
            for p, s in dict(pass_seeds, greedy=pass_seeds['sampled1']).items():
                patch.setitem(PASS_SEEDS, p, s)
            result = evaluation(tmp_path / name, {seed: passes}, plans={}, run=record, **settings)
            assert result['verdict']['kind'] == 'verdict' and result['verdict']['plan'] == name, result['verdict']
            assert words(result) == dict(L='pass') and set(result['seeds']) == {str(seed)}
            unfrozen = evaluation(tmp_path / f'{name}-unfrozen', {seed: passes}, run=record, **settings,
                                  plans={name: dict.fromkeys(FREE_PLAN)})
        other = ({'pilot', 'wider_pilot'} - {name}).pop()
        why = unfrozen['verdict']['why']
        assert unfrozen['verdict']['plan'] is None and f'plan {name} is not frozen' in why
        assert [w for w in why if w.startswith(f'not plan {other}: world, ')], why


def test_outputs_of_another_evaluation_a_second_rotation_or_a_life_twice_are_refused(tmp_path):
    rows = [life('wait', r, {}) for r in range(2)]
    a = output(tmp_path / 'a', rows, 2)
    changed = dict(task='x', rows=0, world='w', observation='o', source=dict(commit='d'), split='final', first=1, records=3,
                   key=7, muggers=True, xp_rate=25, ticks=1, read=2, control_seed=6)
    assert set(changed) == set(FREE_SHARED)
    for name, changes in [*((k, {k: v}) for k, v in changed.items()), ('tables', dict(tables=dict(path='t', sha256='x')))]:
        records = changes.pop('records', 2)
        with pytest.raises(ValueError, match='another evaluation|holds rows of revision 0'):
            free_json([a, output(tmp_path / name, [life('eat', r, {}) for r in range(2)], records, **changes)],
                      tmp_path / 'f.json')
    with pytest.raises(ValueError, match='twice'):
        free_json([a, output(tmp_path / 'again', rows[:1], 2)], tmp_path / 'f.json')
    rotations = [output(tmp_path / f'rotation{p}', [life('rotation', r, {}) for r in range(2)], 2, period=p) for p in (2, 3)]
    with pytest.raises(ValueError, match='one rotation'):
        free_json(rotations, tmp_path / 'f.json')
    with pytest.raises(ValueError, match='declares 0 records'):
        free_json([output(tmp_path / 'none', [], 0)], tmp_path / 'f.json')
    mine = lambda policy, p, **row: [life('policy', r, {}, pass_name=p, policy_seed=policy, **row) for r in range(2)]
    first = output(tmp_path / 's1', mine(SEED, 'sampled1'), 2, seed=SEED)
    other = manifest(2, SEED)['passes']
    other[1]['seed'] = 33
    with pytest.raises(ValueError, match='other pass seeds'):
        free_json([first, output(tmp_path / 's2', mine(SEED, 'sampled2'), 2, seed=SEED, passes=other)], tmp_path / 'f.json')
    twice = manifest(2, 202)['policy']
    twice['policy']['sha256'] = f'p{SEED}'
    with pytest.raises(ValueError, match='one checkpoint'):
        free_json([first, output(tmp_path / 'p202', mine(202, 'sampled1'), 2, seed=202, policy=twice)], tmp_path / 'f.json')
    twice = manifest(2, 202)['policy']
    twice['initial']['sha256'] = f'i{SEED}'
    with pytest.raises(ValueError, match='one checkpoint'):  # one initialization under two seeds
        free_json([first, output(tmp_path / 'i202', mine(202, 'sampled1'), 2, seed=202, policy=twice)], tmp_path / 'f.json')
    same = output(tmp_path / 'same', mine(SEED, 'sampled1') + mine(SEED, 'sampled2', seed=11000), 2, seed=SEED)
    with pytest.raises(ValueError, match=r'records \[0\] play one seed in both sampled passes'):
        free_json([same], tmp_path / 'f.json')
    (a / 'manifest.json').write_text(json.dumps(manifest(2, parts=2)))
    assert free_json([a], tmp_path / 'f.json')['evidence']['missing'] == ['a/lives-1.jsonl']


def test_the_read_and_greedy_are_described_and_the_unpaid_reads_have_known_answers(tmp_path):
    runs = [['woodcutting', 10, 50, 5], ['fight:chicken', 60, 100, 4], ['woodcutting', 120, 130, 2]]
    policy = lambda r: dict(gain=dict(woodcutting=4, attack=1, prayer=2), read=dict(gain_s=5, died=r == 0),
                            death=dict(tick=200, killer='goblin') if r == 0 else None, activity=runs if r == 0 else runs[:1],
                            eat_log=[[5, 3, 3], [9, 10, 0]], eats_full=1, made=dict(wood=7), kills=dict(chicken=2),
                            damage=dict(goblin=3))
    result = evaluation(tmp_path, dict(sampled1=policy, sampled2=policy, greedy=policy),
                        controls={c: constant(GAINS[c], read=dict(gain_s=2, died=False)) for c in CONTROLS})
    p = result['seeds'][str(SEED)]['passes']['sampled1']
    assert p['read_differences']['random']['mean'] == 3 and p['policy']['read']['deaths'] == 1
    assert p['policy']['mean_gain'] == 5 and p['policy']['deaths_a_life'] == .25
    reads, totals = p['policy']['reads']['a_life'], p['policy']['reads']['totals']
    assert reads['levels'] == dict(attack=1, prayer=2, woodcutting=4) and reads['made'] == dict(wood=7)
    assert reads['time'] == {'woodcutting': (80 + 3 * 50) / 4, 'fight:chicken': 50 / 4,
                             'none': (36000 - 130 + 3 * (36000 - 50)) / 4}
    assert reads['switches'] == .5 and reads['eats'] == 2 and reads['eats_full'] == 1
    assert reads['kills'] == dict(chicken=2) and reads['damage'] == dict(goblin=3)
    assert totals == dict(deaths=dict(count=1, ticks=[200], killers=dict(goblin=1)), eats_by_hp={'3': 4, '10': 4},
                          switches_max=2)  # the one life with two switches
    assert 'claims' not in result['seeds'][str(SEED)]['passes']['greedy']


# ---- the policy, the command and the rows in the Python reference (private tables) -------------------------------
def train_tiny(tmp, tables, name_table_path, *extra, world='village-world-v1'):
    """The training command on 4 development layouts of `world` (over `tables`, its archive; another world than the
    first names its archive with --tables): 2 worlds x 16 decisions, 1 worker, under the tests' synthetic name table
    (conftest.py), which every test function reads; `extra` its added arguments."""
    import names
    from sim import world as W
    from sim.combat import VILLAGES
    from sim.train import main
    spec = W.WORLDS[VILLAGES[world]['layouts']]
    W.save_pool(tmp / 'pool.npz', [W.generate(s, tables, spec) for s in W.split_seeds('development', 4)])
    archive = [] if world == 'village-world-v1' else ['--tables', str(free.world_tables(world))]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('RL_NAME_TABLE', str(name_table_path))
        patch.setattr(sys, 'argv', ['sim.train', 'train', '--task', 'free-play-v1', '--env', 'cpu', '--workers', '1',
                                    '--pool', str(tmp / 'pool.npz'), '--worlds', '2', '--n-steps', '16', '--batch-size',
                                    '32', '--epochs', '1', '--decisions', '32', '--seed', '1', '--validate',
                                    '--batch-setting', f'world="{world}"', 'muggers=false', 'life=["fixed",200]',
                                    *archive, '--output', str(tmp / 'run'),
                                    *extra])
        names.name_table.cache_clear()
        try:
            main()
        finally:
            names.name_table.cache_clear()
    return tmp / 'run'


@pytest.fixture(scope='module')
def tiny(tables, tmp_path_factory, name_table_path):
    """A tiny free-play-v1 checkpoint through the training command: one update."""
    return train_tiny(tmp_path_factory.mktemp('tiny'), tables, name_table_path)


def saved_as(source, path, **kwargs):
    """`source` saved again with policy_kwargs changed: a checkpoint that records other versions than its table's."""
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load(str(source), device='cpu')
    model.policy_kwargs.update(kwargs)
    model.save(path)
    return path


def run_copy(tiny, folder, checkpoint=False, **summary):
    """The tiny run's files linked into `folder` with its summary.json changed by `summary` (top-level keys, or the batch
    in effect's under 'batch'); `checkpoint`: also one of checkpoints/ with other weights, whose digest the summary
    records as sim.train does (checkpoint_digests), unless `summary` gives checkpoint_digests."""
    folder.mkdir(parents=True)
    for name in ('initial.zip', 'latest.zip'):
        os.link(tiny / name, folder / name)
    run = json.loads((tiny / 'summary.json').read_text())
    run['free_play']['batch_in_effect'].update(summary.pop('batch', {}))
    (folder / 'checkpoints').mkdir()
    if checkpoint:
        import torch
        from sb3_contrib import RecurrentPPO
        from sim.train import policy_digest
        model = RecurrentPPO.load(str(tiny / 'latest.zip'), device='cpu')
        with torch.no_grad():
            next(model.policy.parameters()).add_(0.01)
        model.save(folder / 'checkpoints' / 'ppo_16_steps.zip')
        run['checkpoint_digests'] = dict(run['checkpoint_digests'], **{'ppo_16_steps.zip': policy_digest(model.policy)})
    (folder / 'summary.json').write_text(json.dumps(dict(run, **summary)))
    return folder


def test_the_loader_takes_a_free_play_checkpoint_and_refuses_another_contract(tiny, tmp_path):
    from baseline.policy import POPART
    from sim.train import train
    model, facts = load_free_policy(tiny / 'latest.zip')
    assert facts['observation'] == 'obs-v6' and facts['value_norm'] == POPART and facts['free_play'] is True
    assert facts['lstm'] == 128 and 'goal_vec' not in model.observation_space.spaces
    assert model.policy.training is False  # evaluation mode, as collection plays it
    common = dict(seed=1, worlds=4, n_steps=8, batch_size=16, n_epochs=1, decisions=32, verbose=False)
    train(tmp_path / 'goal', **common)  # obs-v5, goal-conditioned
    train(tmp_path / 'fake', task='free-play-v1', **common)  # goal-free, the stand-in's table
    with pytest.raises(ValueError, match='reads obs-v5 inputs, not obs-v6'):
        load_free_policy(tmp_path / 'goal' / 'latest.zip')
    with pytest.raises(ValueError, match='reads fake-obs-v6 inputs, not obs-v6'):
        load_free_policy(tmp_path / 'fake' / 'latest.zip')
    with pytest.raises(ValueError, match='reads obs-v5 inputs'):  # obs-v6's table under another recorded version
        load_free_policy(saved_as(tiny / 'latest.zip', tmp_path / 'v5.zip', observation='obs-v5'))
    with pytest.raises(ValueError, match="another observation table than obs-v6's"):  # another table recorded as obs-v6
        load_free_policy(saved_as(tmp_path / 'fake' / 'latest.zip', tmp_path / 'table.zip', observation='obs-v6'))
    actor = FreePolicyActor(model, deterministic=False)
    obs = {k: s.sample() for k, s in model.observation_space.spaces.items()}
    actor.reset(1)
    for wrong in ({k: v for k, v in obs.items() if k != 'prog_vec'}, dict(obs, goal_vec=obs['task_vec'])):
        with pytest.raises(ValueError, match="unlike the policy's own arrays"):
            actor.act(wrong)


def settings_of(**changes):
    return dict(dict(world='village-world-v1', split='development', first=4000, key=223, muggers=False, xp_rate=1,
                     ticks=400, read=150, period=60,
                     control_seed=5, passes=[dict(name='sampled1', seed=11, sampled=True)],
                     policy=dict(seed=SEED, policy=dict(sha256='p'), initial=dict(sha256='i'))), **changes)


@pytest.mark.parametrize('actor', ('random', 'chickens', 'policy'))
def test_the_6000_tick_read_is_a_life_of_that_length(tables, tiny, actor):
    """The read of a 400-tick life at 150 ticks is the 150-tick life of the same record, actor and seed: obs-v6 shows
    neither length nor time, and the read ends at the cycle in which the shorter life ends."""
    models = dict(policy=load_free_policy(tiny / 'latest.zip')[0])
    job = (actor, 'sampled1' if actor == 'policy' else None, 1)
    longer = free_life(job, settings_of(), tables, models)['read']
    short = free_life(job, settings_of(ticks=150, read=149), tables, models)
    assert longer['tick'] == short['ticks'] and longer['died'] == (short['death'] is not None)
    assert longer['gain'] == {k: v for k, v in short['gain'].items() if k in S} and longer['gain_s'] == short['gain_s']
    assert longer['fractional'] == pytest.approx(short['paid'], abs=1e-6)


def test_the_read_of_a_life_that_dies_before_or_after_it(tables):
    """Random legal play with muggers on development record 0 dies at tick 451: a read at 200 ticks is before the death
    and does not count it; a read at 700 is the life itself, read at its death."""
    run = settings_of(muggers=True, ticks=800)
    before = free_life(('random', None, 0), dict(run, read=200), tables, {})
    after = free_life(('random', None, 0), dict(run, read=700), tables, {})
    assert before['death'] == after['death'] and before['death']['tick'] == 451
    assert before['read']['died'] is False and 200 <= before['read']['tick'] < 451
    assert after['read']['died'] is True and after['read']['tick'] == 451 and after['read']['gain_s'] == after['gain_s']


def test_greedy_play_is_one_life_whatever_its_seed_and_sampled_play_is_not(tables, tiny):
    models = {a: load_free_policy(tiny / f)[0] for a, f in (('policy', 'latest.zip'), ('initial', 'initial.zip'))}
    rows = {}
    for actor, p in (('policy', 'greedy'), ('initial', 'sampled1')):
        for seed in (11, 12):
            run = settings_of(ticks=300, passes=[dict(name=p, seed=seed, sampled=p != 'greedy')])
            row = free_life((actor, p, 1), run, tables, models)
            rows[actor, seed] = {k: v for k, v in row.items() if k not in ('seed', 'seconds')}
    assert rows['policy', 11] == rows['policy', 12] and rows['initial', 11] != rows['initial', 12]


def test_the_unpaid_reads_of_a_life_follow_its_events(tables):
    """Each eat has the hitpoints before it and what it healed, an eat at full hitpoints heals 0, the eats are the soak
    row's, and the activity runs hold every XP event once, a fight's by the npc that was hit. Kills and fights go by npc
    name: chickens record 0 hits both chicken types (chicken and chicken_brown, both named Chicken)."""
    for actor in ('eat', 'chickens', 'random'):
        for record in range(3):
            row = free_life((actor, None, record), settings_of(ticks=600), tables, {})
            assert len(row['eat_log']) == row['eats'] and row['eats_full'] == sum(e[2] == 0 for e in row['eat_log'])
            base = row['levels0']['hitpoints'] + row['gain'].get('hitpoints', 0)
            assert all(0 < hp <= base and (healed > 0) == (hp < base) for _, hp, healed in row['eat_log'] if actor == 'eat')
            combat = {'attack', 'strength', 'defence', 'hitpoints'}
            done = {run[0] for run in row['activity']}
            assert {a for a in done if not a.startswith('fight:')} == set(row['xp']) - combat
            assert any(a.startswith('fight:') for a in done) == bool(set(row['xp']) & combat)
            ticks = [t for run in row['activity'] for t in run[1:3]]
            assert ticks == sorted(ticks) and all(t <= row['ticks'] for t in ticks)
            assert set(row['kills']) <= {'chicken', 'goblin'}
            assert {a for a in done if a.startswith('fight:')} <= {'fight:chicken', 'fight:goblin'}
            if actor == 'chickens':
                assert all(a.startswith('fight:') for a in done) and sum(row['kills'].values()) > 0
                assert 'fight:chicken' in done and 'chicken' in row['kills']


def command(tmp_path, monkeypatch, tiny, out, *extra, records=2, ticks=200, read=60):
    monkeypatch.setattr(sys, 'argv', ['sim.hour', 'free', '--output', str(out), '--world', 'village-world-v1', '--first',
                                      '4000', '--records', str(records),
                                      '--key', '223', '--muggers', '0', '--xp-rate', '1', '--ticks', str(ticks),
                                      '--read-ticks', str(read), '--checkpoint', str(tiny / 'latest.zip'), '--initial',
                                      str(tiny / 'initial.zip'), '--policy-seed', str(TINY), '--seeds', '11', '22',
                                      '--controls', *CONTROLS, '--period', '40', '--control-seed', '5', *extra])
    hour_main()


def rows_of(folder, parts=1):
    from sim.hour import free_rows
    rows = [r for k in range(parts) for r in free_rows(folder / f'lives-{k}.jsonl')[0]]
    return sorted(({k: v for k, v in r.items() if k != 'seconds'} for r in rows),
                  key=lambda r: (r['record'], r['actor'], r['pass'] or ''))


def test_the_command_plays_every_actor_and_the_verdict_step_reads_every_number_from_the_rows(tables, tiny, tmp_path,
                                                                                             monkeypatch):
    """sim.hour free on two development records with short lives: the policy sampled in two passes and greedy, its
    initialization in both sampled passes, every control once; each sampled policy life plays again from its row's seed
    alone, and the two passes differ; parts in two processes' shares give the same rows, a part cut in a line resumes to
    them, and sim.gate free computes every number of the claims from the rows alone."""
    import torch
    for name in ('pilot', 'wider_pilot', 'wider_final'):  # no plan frozen: a real run's own reasons
        monkeypatch.setitem(RULES['plans'], name, dict.fromkeys(FREE_PLAN))
    command(tmp_path, monkeypatch, tiny, tmp_path / 'one')
    rows, manifest = rows_of(tmp_path / 'one'), json.loads((tmp_path / 'one' / 'manifest.json').read_text())
    assert len(rows) == 2 * (5 + len(CONTROLS)) and manifest['controls'] == list(CONTROLS) and manifest['parts'] == 1
    assert {(r['actor'], r['pass']) for r in rows} == {(a, p) for a, p, _ in free_jobs(1, True, CONTROLS)}
    assert manifest['policy']['policy']['sha256'] != manifest['policy']['initial']['sha256']
    assert [p['seed'] for p in manifest['passes']] == [11, 22, 11] and manifest['torch_threads'] == FREE['torch_threads']
    run, summary = manifest['policy']['run'], json.loads((tiny / 'summary.json').read_text())
    assert run['checkpoint'] == 'latest.zip' and run['status'] == 'completed' and run['seed'] == TINY
    assert run['batch_in_effect']['muggers'] is False and run['unfinished'] is False
    assert set(run['facts']) == set(FREE['run_facts']) and run['facts']['pool'] == summary['pool']['sha256']
    assert {k: run['facts'][k] for k in ('decisions', 'worlds', 'n_steps', 'batch_size', 'muggers', 'life')} == dict(
        decisions=32, worlds=2, n_steps=16, batch_size=32, muggers=False, life=['fixed', 200])
    assert run['facts']['value_norm'] == summary['learner']['value_norm'] and run['facts']['tables'] == summary['tables']['sha256']
    assert run['policy_sha256_at_end'] == summary['policy_sha256_at_end'] == manifest['policy']['policy']['policy_sha256']
    assert run['env_seed'] == TINY and run['weights'] is None and run['engine'] == 'cpu'  # its own start
    assert run['facts']['validate'] is True is summary['validate']  # the tiny run trained with --validate
    assert run['source'] == summary['source'] and summary['source'].get('commit')  # the source it trained from
    for r in rows:
        mine = r['actor'] in ('policy', 'initial')
        assert r['reason'] in ('horizon', 'death') and r['unsupported'] is None and r['read']['ticks'] == 60
        assert r['split'] == 'development' and r['first'] == 4000 and r['key'] == 223 and r['muggers'] is False
        assert (r['policy_seed'] == TINY) == mine and r['checkpoint'] == (manifest['policy'][r['actor']]['sha256'] if mine
                                                                         else None)
        assert r['deterministic'] is (r['pass'] == 'greedy' if mine else None) and (r['period'] == 40) == (r['actor'] == 'rotation')
        assert r['seed'] == ([5, r['record']] if not mine else r['seed']) and r['layout'] == 1_004_000 + r['record']
    seeds = {(r['pass'], r['record']): r['seed'] for r in rows if r['actor'] == 'policy'}
    assert len(set(seeds.values())) == 4 and seeds[('greedy', 0)] == seeds[('sampled1', 0)]
    model = load_free_policy(tiny / 'latest.zip')[0]
    sampled = [r for r in rows if r['actor'] == 'policy' and r['pass'] != 'greedy']
    threads = torch.get_num_threads()
    torch.set_num_threads(FREE['torch_threads'])
    try:
        for r in sampled:  # the life again from its row's seed, outside free_life
            again = free.Life(4000, 223, r['record'], False, 1, 200, tables, 'development')
            actor = FreePolicyActor(model, deterministic=False)
            got = again.row('policy', None, actor, free.play(again.task, actor, r['seed']), None, 0)
            assert {k: v for k, v in got.items() if k != 'seconds'} == {k: r[k] for k in got if k != 'seconds'}
    finally:
        torch.set_num_threads(threads)
    unlike = lambda a, b: {k: v for k, v in a.items() if k not in ('seed', 'pass')} != \
        {k: v for k, v in b.items() if k not in ('seed', 'pass')}
    assert any(unlike(a, b) for a in sampled for b in sampled if a['record'] == b['record'] and a['pass'] != b['pass'])
    monkeypatch.setattr(sys, 'argv', ['sim.gate', 'free', str(tmp_path / 'one'), '--output', str(tmp_path / 'free.json')])
    gate_main()
    result = json.loads((tmp_path / 'free.json').read_text())
    assert result['verdict']['kind'] == 'description' and result['verdict']['seeds'] == [TINY] and words(result) == {}
    # a real run fits: the plans' own lines only, beside the run's source when this tree has changes (it is one commit
    # with no change only at a commit, as the outputs' then are)
    changed = 'a run trained from a source that is not one commit with no change'
    assert [w for w in result['verdict']['why'] if not w.startswith(changed)] == ['plan final is not frozen',
                                                                                   'plan pilot is not frozen',
                                                                                   'plan wider_pilot is not frozen',
                                                                                   'plan wider_final is not frozen']
    assert result['settings']['period'] == 40 and result['settings']['control_seed'] == 5
    assert result['seeds'][str(TINY)]['pass_seeds'] == PASS_SEEDS and result['statement'] == RULES['statement']
    assert result['complete'] == dict(wait=True, initial={str(TINY): dict(sampled1=True, sampled2=True)},
                                      greedy={str(TINY): True})
    for p in RULES['passes']:
        claims = result['seeds'][str(TINY)]['passes'][p]['described']
        tests = [claims['L'], claims['R'], *(claims['B']['references'][g][h] for g in RULES['references']
                                            for h in ('total', 'outside')), *claims['no_rescue']['exploits'].values()]
        assert len(tests) == 13 and all(t['difference']['pairs'] == 2 and t['difference']['draws'] == 10_000 for t in tests)
    assert set(result['controls']) == set(CONTROLS) and all(c['complete'] and c['reads'] for c in result['controls'].values())
    shares = tmp_path / 'shares'
    for k in range(2):  # each part as its own process would play it
        command(tmp_path, monkeypatch, tiny, shares, '--part', str(k), '2')
    assert rows_of(shares, 2) == rows
    lines = (shares / 'lives-1.jsonl').read_text().splitlines(keepends=True)
    (shares / 'lives-1.jsonl').write_text(''.join(lines[:5]) + lines[5][:40])  # a part stopped in the middle of a line
    got = free_run(shares, world='village-world-v1', split='development',
                           first=4000, records=2, key=223, muggers=False, xp_rate=1, ticks=200,
                   read=60, checkpoint=tiny / 'latest.zip', initial=tiny / 'initial.zip', policy_seed=TINY, seeds=(11, 22),
                   controls=CONTROLS, period=40, control_seed=5, part=(1, 2))
    assert got['cut'] and got['kept'] == 5 and got['played'] == len(lines) - 5 and rows_of(shares, 2) == rows


def test_a_part_refuses_another_checkpoint_other_pass_seeds_and_a_second_process(tables, tiny, tmp_path):
    """A resume (or another part) of an output plays the same checkpoint with the same pass seeds, and one process plays
    a part at a time. A checkpoint of checkpoints/ of a completed run plays like latest.zip."""
    folder = run_copy(tiny, tmp_path / 'run', checkpoint=True)
    later = folder / 'checkpoints' / 'ppo_16_steps.zip'
    played = dict(world='village-world-v1', split='development',
                  first=4000, records=1, key=223, muggers=False, xp_rate=1, ticks=50, read=10,
                  initial=folder / 'initial.zip', policy_seed=TINY, control_seed=1, tables=tables)
    free_run(tmp_path / 'o', checkpoint=folder / 'latest.zip', seeds=(11, 22), **played)
    for checkpoint, seeds, differ in ((later, (11, 22), 'policy'), (folder / 'latest.zip', (11, 33), 'passes')):
        with pytest.raises(ValueError, match=f"another run's settings: {differ} differ"):
            free_run(tmp_path / 'o', checkpoint=checkpoint, seeds=seeds, **played)
    assert free_run(tmp_path / 'c', checkpoint=later, seeds=(11, 22), **played)['played'] == 5
    assert json.loads((tmp_path / 'c' / 'manifest.json').read_text())['policy']['run']['checkpoint'] == \
        'checkpoints/ppo_16_steps.zip'
    with open(tmp_path / 'o' / 'lives-0.jsonl', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another process plays this part
        with pytest.raises(ValueError, match='being played by another process'):
            free_run(tmp_path / 'o', checkpoint=folder / 'latest.zip', seeds=(11, 22), **played)
    assert free_run(tmp_path / 'o', checkpoint=folder / 'latest.zip', seeds=(11, 22), **played)['kept'] == 5


def test_a_plan_frozen_from_a_real_runs_facts_reads_its_outputs_and_plan_refuses_early(tables, tiny, tmp_path,
                                                                                         monkeypatch):
    """A pilot plan written as Python literals from the tiny run's facts (a tuple life, an int half-life) gives a
    verdict on the outputs of sim.hour free under --plan, from one clean commit (source() stands in for it) and the
    run's own summary (its digests and tables as sim.train records them). --plan refuses, before anything is written, an
    output the plan would not read, with the verdict step's reasons."""
    from baseline import telemetry
    from baseline.telemetry import sha256
    from sim.build import DATA
    digest = lambda path: load_free_policy(path)[1]['policy_sha256']
    clean = dict(commit='0' * 40, changed=[], untracked=[], patch=None)
    run = run_copy(tiny, tmp_path / 'run', source=clean)  # trained from the outputs' commit, with no change
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['policy_sha256_at_end'] == digest(run / 'latest.zip')  # as sim.train wrote it
    facts = dict(decisions=32, worlds=2, n_steps=16, batch_size=32, n_epochs=1, learning_rate=0.0003, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lstm_hidden_size=128, half_life_ticks=1000,
                 value_norm=dict(method='popart', beta=0.1, min_std=0.01), muggers=False, life=('fixed', 200),
                 pool=summary['pool']['sha256'], tables=sha256(DATA), validate=True)
    assert summary['validate'] is True
    plan = dict(world='village-world-v1', split='development',
                first=4000, records=2, key=223, xp_rate=1, muggers=False, ticks=200, read=60,
                period=40, control_seed=5, tables=sha256(DATA), run=facts, start={TINY: summary['policy_sha256_at_start']},
                pass_seeds=dict(sampled1=11, sampled2=22), policy_seeds=(TINY,), claims=('L',))
    code = dict(source=clean)
    monkeypatch.setattr(telemetry, 'source', lambda directory=None, root=None: code['source'])
    monkeypatch.setitem(RULES['plans'], 'pilot', plan)
    common = dict(world='village-world-v1', split='development',
                  first=4000, records=2, key=223, muggers=False, xp_rate=1, ticks=200, read=60,
                  control_seed=5, plan='pilot', tables=tables)
    policy = dict(checkpoint=run / 'latest.zip', initial=run / 'initial.zip', policy_seed=TINY, seeds=(11, 22))
    free_run(tmp_path / 's1', **policy, **common)
    free_run(tmp_path / 'controls', controls=('wait', 'random', 'rotation'), period=40, **common)
    result = free_json([tmp_path / 's1', tmp_path / 'controls'], tmp_path / 'free.json')
    assert result['verdict']['kind'] == 'verdict' and result['verdict']['plan'] == 'pilot' and set(words(result)) == {'L'}
    assert result['seeds'][str(TINY)]['verdict']['incomplete_controls'] == []
    moved = run_copy(tiny, tmp_path / 'later', checkpoint=True)
    later = moved / 'checkpoints' / 'ppo_16_steps.zip'
    runs = {k: run_copy(tiny, tmp_path / k, **dict(dict(source=clean), **v)) for k, v in dict(
        envseed=dict(env_seed=TINY + 5), weights=dict(weights=dict(path='w', sha256='w')),
        dirtyrun=dict(source=dict(clean, untracked=['notes.txt'])), elsewhere=dict(source=dict(clean, commit='1' * 40))).items()}
    for name, changes, plans, why in (
            ('unfrozen', {}, dict(plan, key=None), 'plan pilot is not frozen'),
            ('key', dict(key=224), plan, 'key differ'),
            ('dirty', dict(source=dict(clean, untracked=['logs'])), plan, 'the source is not one commit with no change'),
            ('later', dict(checkpoint=later, initial=moved / 'initial.zip'), plan, "not its run's final"),
            ('facts', {}, dict(plan, run=dict(facts, decisions=64)), 'run (decisions) differ'),
            ('start', {}, dict(plan, start={TINY: 'another'}), f'start (seeds [{TINY}]) differ'),
            ('envseed', dict(initial=runs['envseed'] / 'initial.zip', checkpoint=runs['envseed'] / 'latest.zip'), plan,
             'environment seed 6'),
            ('weights', dict(initial=runs['weights'] / 'initial.zip', checkpoint=runs['weights'] / 'latest.zip'),
             dict(plan, start={TINY: 'another'}), f'start (seeds [{TINY}]) differ'),  # a file of other tensors
            ('dirtyrun', dict(initial=runs['dirtyrun'] / 'initial.zip', checkpoint=runs['dirtyrun'] / 'latest.zip'), plan,
             'a run trained from a source that is not one commit with no change'),
            ('elsewhere', dict(initial=runs['elsewhere'] / 'initial.zip', checkpoint=runs['elsewhere'] / 'latest.zip'), plan,
             "a run trained at another commit than the outputs' " + '0' * 40)):
        monkeypatch.setitem(RULES['plans'], 'pilot', plans)
        code['source'] = changes.pop('source', clean)
        with pytest.raises(ValueError, match=f'plan pilot would not read this output: .*{re.escape(why)}'):
            free_run(tmp_path / f'x-{name}', **{**policy, **common, **changes})
        assert not (tmp_path / f'x-{name}').exists()
    code['source'] = clean
    monkeypatch.setitem(RULES['plans'], 'pilot', plan)
    # the frozen initialization's file (--weights) at the seed's own environment seed, with the plan's start: read
    free_run(tmp_path / 'fromfile', **{**policy, **common, 'initial': runs['weights'] / 'initial.zip',
                                       'checkpoint': runs['weights'] / 'latest.zip'})
    with pytest.raises(ValueError, match='plan pilot would not read this output: period differ'):
        free_run(tmp_path / 'x-period', controls=('wait', 'rotation'), period=50, **common)  # the rotation at another period
    assert not (tmp_path / 'x-period').exists()
    with pytest.raises(ValueError, match='no plan nope'):
        free_run(tmp_path / 'x-nope', **policy, **dict(common, plan='nope'))
    monkeypatch.setattr(sys, 'argv', ['sim.hour', 'free', '--output', str(tmp_path / 'x-cli'), '--world', 'village-world-v1',
                                      '--first', '4000', '--records',
                                      '2', '--key', '224', '--muggers', '0', '--xp-rate', '1', '--ticks', '200',
                                      '--read-ticks', '60', '--controls', 'wait', '--control-seed', '5', '--plan', 'pilot'])
    with pytest.raises(SystemExit, match=r'\[free\] plan pilot would not read this output: key differ'):
        hour_main()
    assert not (tmp_path / 'x-cli').exists()


def test_a_run_of_the_pilots_seed_with_other_facts_is_refused_by_plan_pilot_before_any_life(tables, tmp_path,
                                                                                              name_table_path):
    """A tiny run of seed 77 at the pilot's evaluation settings: --plan pilot refuses it before the output folder exists,
    naming the run facts that differ from the frozen plan's and its start (the tests' name table is not the study's)."""
    run = train_tiny(tmp_path, tables, name_table_path, '--seed', '77')
    with pytest.raises(ValueError, match=re.escape('run (batch_size, decisions, life, n_epochs, n_steps, pool, worlds), '
                                                   'start (seeds [77]) differ')):
        free_run(tmp_path / 'out', world='village-world-v1', split='development',
                                   first=4000, records=64, key=223, muggers=False, xp_rate=1,
                 ticks=36000, read=6000, checkpoint=run / 'latest.zip', initial=run / 'initial.zip', policy_seed=77,
                 seeds=(229, 233), control_seed=4253, plan='pilot', tables=tables)
    assert not (tmp_path / 'out').exists()


def test_a_start_only_run_at_the_pilots_settings_holds_the_plans_facts_and_start(tables, tmp_path):
    """sim.train train --initial-only of seed 77 at the pilot's training settings (1,024 worlds, 8 workers, the 128
    training layouts, the study's name table): its run facts are the frozen plan's, its start digest is the plan's, and
    free_plan_differs finds nothing. What the plan still refuses is what a start-only run lacks: a completed run's final
    checkpoint (latest.zip) and the summary's digest of it."""
    from baseline.telemetry import sha256
    from pathlib import Path
    from sim.build import DATA
    from sim.hour import run_facts
    root = Path(__file__).resolve().parents[2]
    pool, names = root / 'logs/sim/world/train-128-1000-nc1.npz', root / 'logs/names/name-vectors.npz'
    if not (pool.exists() and names.exists()):
        pytest.skip('the 128 training layouts and the name table are private inputs')
    run = train_tiny(tmp_path, tables, names, '--worlds', '1024', '--workers', '8', '--pool', str(pool), '--n-steps', '64',
                     '--batch-size', '1024', '--epochs', '4', '--decisions', '4194304', '--seed', '77', '--batch-setting',
                     'world="village-world-v1"', 'muggers=false', 'life=["uniform",6000,36000]', '--initial-only')
    summary, pilot = json.loads((run / 'summary.json').read_text()), RULES['plans']['pilot']
    assert summary['status'] == 'initialized' and not (run / 'latest.zip').exists()
    assert run_facts(summary) == json.loads(json.dumps(pilot['run']))
    record = dict(facts=run_facts(summary), policy_sha256_at_start=summary['policy_sha256_at_start'], status=summary['status'],
                  checkpoint=None, policy_sha256_at_end=summary.get('policy_sha256_at_end'), env_seed=summary['env_seed'],
                  weights=summary.get('weights'), init=summary.get('init'), env=summary['env'], engine=summary['engine'],
                  source=summary['source'])
    policies = {77: dict(policy=dict(policy_sha256=None), initial=dict(policy_sha256=summary['policy_sha256_at_start']),
                         run=record, passes=dict(sampled1=229, sampled2=233, greedy=229))}
    settings = dict({k: pilot[k] for k in FREE_PLAN_SETTINGS if k != 'world'}, world=free_plan_world(pilot),
                    tables=sha256(DATA))
    assert free_plan_differs(pilot, json.loads(json.dumps(settings)), policies) == []
    changed = 'a run trained from a source that is not one commit with no change'  # this tree between commits
    assert [w for w in free_plan_run(policies, summary['source']) if not w.startswith(changed)] == [
        'a checkpoint of a run that has not completed (seeds [77])', "a checkpoint that is not its run's final (seed 77: None)",
        'a run summary that records no final checkpoint digest (policy_sha256_at_end; seeds [77])']


def test_a_start_only_run_of_each_paid_seed_holds_the_final_plans_facts_and_start_and_no_changed_fact(tables_v2, tmp_path):
    """sim.train train --initial-only of each of the five seeds at wider_final's training settings on this Mac's C
    engine (its pack, 8 workers, 1,024 worlds, village-world-v2's 128 training layouts and archive, the study's name
    table): its run facts, engine facts and start digest are the frozen plan's, and its pack records the plan's tables
    and pool as its inputs, so free_plan_differs finds nothing over the five. Each changed fact is named alone: each
    setting, the tables, a run fact, the pack, the pass seeds, one seed's start, a sixth seed and the run's world."""
    from baseline.telemetry import sha256
    from pathlib import Path
    from sim.hour import engine_facts, run_facts
    root, plan = Path(__file__).resolve().parents[2], RULES['plans']['wider_final']
    pool, names = root / 'logs/sim/world/train-128-20000-v2.npz', root / 'logs/names/name-vectors.npz'
    pack = root / 'logs/native/train-128-20000-v2-a6c325a-ids3-item3.pack'
    if not (pool.exists() and names.exists() and pack.exists()):
        pytest.skip('the 128 training layouts of village-world-v2, their pack and the name table are private inputs')
    plain = json.loads(json.dumps(plan))
    policies = {}
    for seed in plan['policy_seeds']:
        (tmp_path / str(seed)).mkdir()
        run = train_tiny(tmp_path / str(seed), tables_v2, names, '--worlds', '1024', '--workers', '8', '--pool', str(pool),
                         '--n-steps', '64', '--batch-size', '1024', '--epochs', '4', '--decisions', '4194304', '--seed',
                         str(seed), '--engine', 'native', '--pack', str(pack), '--batch-setting', 'world="village-world-v2"',
                         'muggers=false', 'life=["uniform",6000,36000]', '--initial-only', world='village-world-v2')
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['status'] == 'initialized' and summary['env_seed'] == seed and not (run / 'latest.zip').exists()
        assert run_facts(summary) == {k: v for k, v in plain['run'].items() if k in FREE['run_facts']}
        assert engine_facts(summary) == dict(engine='native', pack=plan['run']['pack'])
        assert summary['policy_sha256_at_start'] == plan['start'][seed]
        assert {k: summary['pack']['inputs'][k] for k in ('tables', 'pool')} == {k: plan['run'][k] for k in ('tables', 'pool')}
        record = dict(facts=run_facts(summary), policy_sha256_at_start=summary['policy_sha256_at_start'], status=summary['status'],
                      env_seed=summary['env_seed'], weights=summary.get('weights'), init=summary.get('init'),
                      env=summary['env'], engine=summary['engine'], pack=summary['pack'], source=summary['source'],
                      batch_in_effect=summary['free_play']['batch_in_effect'])
        policies[seed] = dict(policy=dict(policy_sha256=None), initial=dict(policy_sha256=summary['policy_sha256_at_start']),
                              run=record, passes=dict(plain['pass_seeds'], greedy=plain['pass_seeds']['sampled1']))
    settings = dict({k: plain[k] for k in FREE_PLAN_SETTINGS}, tables=sha256(free.world_tables('village-world-v2')))
    assert free_plan_differs(plan, settings, policies) == []
    changed = dict(world='village-world-v1', split='development', first=7000, records=64, key=239, xp_rate=25, muggers=True,
                   ticks=30000, read=5000, period=4000, control_seed=1)
    assert set(changed) == set(FREE_PLAN_SETTINGS)
    for key, value in [*changed.items(), ('tables', 'other')]:
        assert free_plan_differs(plan, dict(settings, **{key: value}), policies) == [key], key
    run = lambda seed, **change: dict(policies[seed], run=dict(policies[seed]['run'], **change))  # noqa: E731
    for name, other, differ in (
            ('decisions', {101: run(101, facts=dict(policies[101]['run']['facts'], decisions=8388608))}, ['run (decisions)']),
            ('pack', {202: run(202, pack=dict(policies[202]['run']['pack'], sha256='other'))}, ['run (pack)']),
            ('passes', {s: dict(p, passes=dict(sampled1=257, sampled2=251, greedy=257)) for s, p in policies.items()},
             ['pass_seeds']),
            ('start', {303: run(303, policy_sha256_at_start=plan['start'][404])}, ['start (seeds [303])']),
            ('sixth', {606: policies[505]}, ['policy_seeds', 'start (seeds [606])']),
            ('world', {404: run(404, batch_in_effect=dict(policies[404]['run']['batch_in_effect'], world='village-world-v1'))},
             ["the run's world (seeds [404])"])):
        assert free_plan_differs(plan, settings, {**policies, **other}) == differ, name


def test_one_part_in_two_real_processes_is_played_once(tables, tmp_path):
    """The same part started in a second process while the first plays: the second is refused, and every life is in the
    part's file once."""
    import subprocess
    import time
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    argv = [sys.executable, '-m', 'sim.hour', 'free', '--output', str(tmp_path / 'o'), '--world', 'village-world-v1',
            '--first', '4000', '--records', '2',
            '--key', '223', '--muggers', '0', '--xp-rate', '1', '--ticks', '12000', '--read-ticks', '100', '--controls',
            'wait', 'random', 'eat', '--control-seed', '1']
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(root / 'rl')]))
    first = subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lives, began = tmp_path / 'o' / 'lives-0.jsonl', time.monotonic()
    while not (lives.exists() and lives.read_text()) and first.poll() is None and time.monotonic() - began < 300:
        time.sleep(0.2)  # until the first process has written a life: it holds the part
    second = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=600)
    out, err = first.communicate(timeout=600)
    assert first.returncode == 0, err
    assert second.returncode != 0 and 'being played by another process' in second.stderr
    rows = rows_of(tmp_path / 'o')
    assert json.loads(out)['played'] == 6 == len(rows) == len({(r['actor'], r['record']) for r in rows})


def test_the_command_refuses_settings_it_must_be_given(tables, tiny, tmp_path, monkeypatch, name_table_path):
    base = ['sim.hour', 'free', '--output', str(tmp_path / 'x'), '--world', 'village-world-v1', '--first', '4000',
            '--records', '1', '--key', '223', '--xp-rate', '1', '--ticks', '50', '--read-ticks', '10']
    wait = ['--muggers', '0', '--controls', 'wait', '--control-seed', '1']
    for argv, message in (([], 2),  # no muggers: argparse refuses
                          (['--muggers', '0', '--controls', 'wait'], 2),  # no control seed
                          (['--muggers', '0', '--controls', 'rotation', '--control-seed', '1'], 'needs its period'),
                          (wait + ['--period', '40'], 'needs its period'),
                          (['--muggers', '0', '--controls', 'rotation', '--period', '0', '--control-seed', '1'], 'rotation period'),
                          (wait + ['--first', '-1'], 'a first index of 0 or more'),
                          (wait + ['--first', '9999', '--records', '2'], 'split has 10000 seeds'),
                          (['--muggers', '0', '--control-seed', '1', '--checkpoint', str(tiny / 'latest.zip'),
                            '--policy-seed', '1', '--seeds', '1', '2'], 'own saved initialization'),
                          (['--muggers', '0', '--control-seed', '1', '--checkpoint', str(tiny / 'latest.zip'), '--initial',
                            str(tiny / 'initial.zip'), '--policy-seed', '1', '--seeds', '1', '1'], 'two sampled passes'),
                          (wait + ['--split', 'final'], 'final=True'),
                          (['--muggers', '0', '--control-seed', '1'], 'a checkpoint or a control')):
        monkeypatch.setattr(sys, 'argv', base + argv)
        with pytest.raises(SystemExit) as stop:
            hour_main()
        assert stop.value.code == message if message == 2 else message in str(stop.value)
    assert not (tmp_path / 'x').exists()
    alone = tmp_path / 'alone'
    alone.mkdir()
    (alone / 'initial.zip').write_bytes((tiny / 'initial.zip').read_bytes())
    policy = dict(world='village-world-v1', split='development',
                  first=4000, records=1, key=223, muggers=False, xp_rate=1, ticks=50, read=10,
                  seeds=(1, 2), control_seed=1, tables=tables)
    for initial, seed, message in ((alone / 'initial.zip', TINY, 'has no run summary beside it'),
                                   (tiny / 'initial.zip', TINY + 1, 'is not shown to be the initialization'),
                                   (tiny / 'latest.zip', TINY, 'is not shown to be the initialization')):
        with pytest.raises(ValueError, match=message):  # no run summary, not its seed, or not its tensors at the start
            free_run(tmp_path / 'z', checkpoint=tiny / 'latest.zip', initial=initial, policy_seed=seed, **policy)
    runs = dict(other=dict(), failed=dict(status='failed'), started=dict(status='started'),
                muggers=dict(batch=dict(muggers=True)), cfg=dict(batch=dict(cfg=dict(speedMs=1))),
                x25=dict(batch=dict(xp_rate=25)), x3=dict(batch=dict(xp_rate=3)), obs=dict(batch=dict(observation='obs-v5')),
                v2=dict(batch=dict(world='village-world-v2')), blank=dict(batch=dict(world=None)),
                archive=dict(tables=dict(path='other.npz', sha256='0' * 64)),
                resume=dict(resume=dict(path='state')), nosummary=dict(),
                final=dict(policy_sha256_at_end='0' * 64), periodic=dict(checkpoint_digests={'ppo_16_steps.zip': '0' * 64}),
                unlisted=dict(checkpoint_digests={}))
    runs = {k: run_copy(tiny, tmp_path / k, checkpoint=k in ('started', 'periodic', 'unlisted'), **v) for k, v in runs.items()}
    (runs['nosummary'] / 'summary.json').unlink()  # a run killed before sim/train.py wrote its summary
    started = runs['started'] / 'checkpoints' / 'ppo_16_steps.zip'
    later = lambda name: runs[name] / 'checkpoints' / 'ppo_16_steps.zip'
    for name, checkpoint, extra, message in (
            ('other', tiny / 'latest.zip', {}, 'is not latest.zip or one of checkpoints/'),
            ('other', started, {}, 'is not latest.zip or one of checkpoints/'),  # another run's checkpoints/ file
            ('resume', None, {}, 'resumed from a recovery state: a resumed run is not evaluated'),
            ('nosummary', None, {}, 'has no run summary beside it'),
            ('nosummary', None, dict(unfinished=True), 'has no run summary beside it'),
            ('final', None, {}, 'holds other tensors than its run summary'),
            ('periodic', later('periodic'), {}, 'holds other tensors than its run summary'),
            ('unlisted', later('unlisted'), {}, 'holds other tensors than its run summary'),
            ('failed', None, {}, "status 'failed': latest.zip needs a completed run"),
            ('failed', None, dict(unfinished=True), "status 'failed': latest.zip needs a completed run"),
            ('started', started, {}, "status 'started'"),
            ('other', None, dict(unfinished=True), 'unfinished names a run that has not'),
            ('muggers', None, {}, 'another world than this evaluation: muggers differ'),
            ('cfg', None, {}, 'cfg differ'),
            ('x25', None, {}, 'xp_rate differ'),  # a policy trained at 25x read at 1x
            ('x3', None, dict(xp_rate=25), 'xp_rate differ'),  # the 25x read is of a 1x policy only
            ('obs', None, {}, 'observation differ'),
            ('v2', None, {}, 'another world than this evaluation: world differ'),  # a village-world-v2 run read in v1
            ('blank', None, {}, 'another world than this evaluation: world differ'),  # a recorded world is held to
            ('archive', None, {}, "trained over another archive \\(0000.*\\) than village-world-v1's")):
        with pytest.raises(ValueError, match=message):
            free_run(tmp_path / 'z', checkpoint=checkpoint or runs[name] / 'latest.zip',
                     initial=runs[name] / 'initial.zip', policy_seed=TINY, **dict(policy, **extra))
    assert not (tmp_path / 'z').exists()
    begun = train_tiny(tmp_path / 'begun', tables, name_table_path, '--initial-only')  # status initialized, no latest.zip
    assert json.loads((begun / 'summary.json').read_text())['status'] == 'initialized' and not (begun / 'latest.zip').exists()
    for checkpoint, message in ((begun / 'latest.zip', 'its run stopped at its start (status initialized'),
                                (begun / 'initial.zip', 'is not latest.zip or one of checkpoints/')):
        with pytest.raises(ValueError, match=re.escape(message)):
            free_run(tmp_path / 'z', checkpoint=checkpoint, initial=begun / 'initial.zip', policy_seed=TINY, **policy)
    unnamed = run_copy(tiny, tmp_path / 'unnamed')  # a summary written before its batch recorded the world: v1's
    record = json.loads((unnamed / 'summary.json').read_text())
    assert record['free_play']['batch_in_effect'].pop('world') == 'village-world-v1'
    (unnamed / 'summary.json').write_text(json.dumps(record))
    assert free_run(tmp_path / 'unnamed-out', checkpoint=unnamed / 'latest.zip', initial=unnamed / 'initial.zip',
                    policy_seed=TINY, **dict(policy, ticks=20, read=5))['played'] == 5
    kept = json.loads((tmp_path / 'unnamed-out' / 'manifest.json').read_text())
    assert kept['world'] == 'village-world-v1' and 'world' not in kept['policy']['run']['batch_in_effect']
    (unnamed / 'summary.json').write_text(json.dumps(dict(record, tables=None)))  # an older C-engine record: no archive
    assert free_run(tmp_path / 'older-out', checkpoint=unnamed / 'latest.zip', initial=unnamed / 'initial.zip',
                    policy_seed=TINY, **dict(policy, ticks=20, read=5))['played'] == 5
    with pytest.raises(ValueError, match="a world of \\['village-world-v1', 'village-world-v2'\\], named with no default"):
        free_run(tmp_path / 'z', **dict(policy, world='village-world-v9', controls=('wait',)))
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'world'"):
        free_run(tmp_path / 'z', **{k: v for k, v in policy.items() if k != 'world'}, controls=('wait',))
    monkeypatch.setattr(sys, 'argv', [a for a in base if a not in ('--world', 'village-world-v1')] + wait)
    with pytest.raises(SystemExit) as stop:  # the command without --world: argparse refuses
        hour_main()
    assert stop.value.code == 2
    old = run_copy(tiny, tmp_path / 'old')  # a summary written before sim.train recorded its digests
    record = json.loads((old / 'summary.json').read_text())
    older = ('policy_sha256_at_end', 'validate', 'source')
    (old / 'summary.json').write_text(json.dumps({k: v for k, v in record.items() if k not in older}))
    assert free_run(tmp_path / 'old-out', checkpoint=old / 'latest.zip', initial=old / 'initial.zip', policy_seed=TINY,
                    **dict(policy, ticks=20, read=5))['played'] == 5
    recorded = json.loads((tmp_path / 'old-out' / 'manifest.json').read_text())['policy']['run']
    assert recorded['source'] is None and recorded['facts']['validate'] is None and recorded['policy_sha256_at_end'] is None
    late = free_run(tmp_path / 'late', checkpoint=started, unfinished=True, initial=runs['started'] / 'initial.zip',
                    policy_seed=TINY, **dict(policy, ticks=20, read=5))
    assert late['played'] == 5
    assert json.loads((tmp_path / 'late' / 'manifest.json').read_text())['policy']['run']['unfinished'] is True
    read25 = free_run(tmp_path / 'x25', checkpoint=tiny / 'latest.zip', initial=tiny / 'initial.zip', policy_seed=TINY,
                      **dict(policy, xp_rate=25, ticks=20, read=5))  # the declared 25x read of a 1x policy
    assert read25['played'] == 5
    digest = lambda path: load_free_policy(path)[1]['policy_sha256']
    moved = run_copy(tiny, tmp_path / 'moved', checkpoint=True)
    summary = json.loads((moved / 'summary.json').read_text())
    assert summary['checkpoint_digests'] == {'ppo_16_steps.zip': digest(moved / 'checkpoints' / 'ppo_16_steps.zip')}
    assert summary['policy_sha256_at_end'] == digest(moved / 'latest.zip')  # the tiny run's own, by sim.train
    (moved / 'checkpoints').rename(tmp_path / 'elsewhere')
    (moved / 'checkpoints').symlink_to(tmp_path / 'elsewhere')  # checkpoints/ on another disk
    for checkpoint in (moved / 'latest.zip', moved / 'checkpoints' / 'ppo_16_steps.zip'):  # the recorded tensors
        assert free_run(tmp_path / f'ok-{checkpoint.name}', checkpoint=checkpoint, initial=moved / 'initial.zip',
                        policy_seed=TINY, **dict(policy, ticks=20, read=5))['played'] == 5
    lives = [trial_seed(11, s, r) for s in (101, 202) for r in range(64)]
    assert max(lives) >= 2 ** 32 and all(0 <= s < 2 ** 64 for s in lives) and FREE['rows'] == 2  # 64-bit seeds, revision 2
    with pytest.raises(ValueError, match='needs its control seed'):  # free_run itself, a policy-only run too
        free_run(tmp_path / 'z', checkpoint=tiny / 'latest.zip', initial=tiny / 'initial.zip', policy_seed=TINY,
                 **dict(policy, control_seed=None))
    with pytest.raises(ValueError, match='unfinished names the run of a checkpoint'):
        free_run(tmp_path / 'z', **dict(policy, seeds=None, controls=('wait',), unfinished=True))
    assert not (tmp_path / 'z').exists()
    played = dict(world='village-world-v1', split='development',
                  first=4000, records=1, key=223, muggers=False, xp_rate=1, ticks=50, read=10,
                  controls=('wait',), control_seed=1, tables=tables)
    free_run(tmp_path / 'y', **played)
    with pytest.raises(ValueError, match="another run's settings: key differ"):  # a resume under another record set
        free_run(tmp_path / 'y', **dict(played, key=224))


@pytest.fixture(scope='module')
def tables_v2():
    """village-world-v2's archive, with the pick content (sim/free.py world_tables)."""
    path = free.world_tables('village-world-v2')
    if not path.exists():
        pytest.skip(f'private simulator tables ({path.name}) absent; sim/build.py says how each archive was made')
    from sim.data import Tables
    return Tables.load(path)


def test_each_world_evaluates_only_its_own_runs_and_records_the_archive_it_plays(tables, tables_v2, tiny, tmp_path,
                                                                                  name_table_path):
    """A free-play run trained on village-world-v2's layouts over its archive records that world in its batch in effect
    and that archive in its summary. A village-world-v1 evaluation refuses it, and a village-world-v2 evaluation refuses
    the village-world-v1 run, before anything is written. Each world's evaluation plays its own run over its own archive,
    and its manifest and rows record that world and archive; a plan check compares the world. A village-world-v2
    evaluation takes no muggers and no other archive."""
    from baseline.telemetry import sha256
    v2 = train_tiny(tmp_path / 'v2', tables_v2, name_table_path, world='village-world-v2')
    summary, archive = json.loads((v2 / 'summary.json').read_text()), free.world_tables('village-world-v2')
    assert summary['status'] == 'completed' and summary['free_play']['batch_in_effect']['world'] == 'village-world-v2'
    assert summary['tables']['sha256'] == sha256(archive) != sha256(free.world_tables('village-world-v1'))
    common = dict(split='development', first=4000, records=1, key=223, muggers=False, xp_rate=1, ticks=20, read=5,
                  seeds=(1, 2), control_seed=1)
    policy = lambda run: dict(checkpoint=run / 'latest.zip', initial=run / 'initial.zip', policy_seed=TINY)  # noqa: E731
    for world, run, archive_of in (('village-world-v1', v2, tables), ('village-world-v2', tiny, tables_v2)):
        with pytest.raises(ValueError, match='another world than this evaluation: world differ'):
            free_run(tmp_path / 'z', world=world, tables=archive_of, **policy(run), **common)
    unnamed = run_copy(v2, tmp_path / 'v2-unnamed')  # village-world-v2 trained by code that recorded no world
    record = json.loads((unnamed / 'summary.json').read_text())
    del record['free_play']['batch_in_effect']['world']
    (unnamed / 'summary.json').write_text(json.dumps(record))
    for world, archive_of in (('village-world-v1', tables), ('village-world-v2', tables_v2)):
        with pytest.raises(ValueError, match=f"records no world and trained over another archive \\({summary['tables']['sha256']}"):
            free_run(tmp_path / 'z', world=world, tables=archive_of, **policy(unnamed), **common)
    with pytest.raises(ValueError, match='plan pilot would not read this output: .*world, records'):
        free_run(tmp_path / 'z', world='village-world-v2', tables=tables_v2, plan='pilot', **policy(v2), **common)
    with pytest.raises(ValueError, match='village-world-v2 holds no muggers'):
        free_run(tmp_path / 'z', world='village-world-v2', tables=tables_v2, controls=('wait',), **dict(common, muggers=True))
    with pytest.raises(ValueError, match='the tables given are not draynor-nc2.npz'):
        free_run(tmp_path / 'z', world='village-world-v2', tables=tables, controls=('wait',), **common)
    assert not (tmp_path / 'z').exists()
    for world, run, archive_of in (('village-world-v1', tiny, tables), ('village-world-v2', v2, tables_v2)):
        out, path = tmp_path / world, free.world_tables(world)
        assert free_run(out, world=world, tables=archive_of, controls=('wait',), **policy(run), **common)['played'] == 6
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['world'] == world and manifest['tables'] == dict(path=str(path), sha256=sha256(path))
        assert manifest['policy']['run']['batch_in_effect']['world'] == world and {r['world'] for r in rows_of(out)} == {world}


def test_an_unsupported_site_stops_the_part_and_its_row_keeps_the_evidence(tables, tmp_path, monkeypatch):
    real_start, real_step, lives = FreePlayEpisode.start, FreePlayEpisode.step, []

    def start(self, *args, **kwargs):
        lives.append(self)
        return real_start(self, *args, **kwargs)

    def step(self, action):
        if len(lives) > 1 and self is lives[1] and self.decisions == 3:
            raise Unsupported('engine script x')
        return real_step(self, action)
    monkeypatch.setattr(FreePlayEpisode, 'start', start)
    monkeypatch.setattr(FreePlayEpisode, 'step', step)
    played = dict(world='village-world-v1', split='development',
                  first=4000, records=2, key=223, muggers=False, xp_rate=1, ticks=100, read=50,
                  controls=('wait', 'random'), control_seed=1, tables=tables)
    with pytest.raises(FreeStop, match='engine script x'):
        free_run(tmp_path / 'u', **played)
    rows = rows_of(tmp_path / 'u')
    assert [(r['actor'], r['record']) for r in rows] == [('random', 0), ('wait', 0)]  # nothing after the stop
    u = next(r for r in rows if r['actor'] == 'random')
    assert u['reason'] == 'unsupported' and u['death'] is None
    assert u['read'] is None or u['read']['tick'] <= u['unsupported']['tick']
    assert u['unsupported']['message'] == 'engine script x' and u['unsupported']['function'] == 'step'
    assert u['unsupported']['file'] == 'sim/tests/test_free_play_eval.py' and len(u['unsupported']['actions']) == 4
    with pytest.raises(FreeStop, match='plays no further life'):  # a resume refuses to go on
        free_run(tmp_path / 'u', **played)
    result = free_json([tmp_path / 'u'], tmp_path / 'free.json')
    assert result['evidence']['unsupported'][0]['message'] == 'engine script x'
    assert result['controls']['random']['complete'] is False and result['controls']['random']['mean_gain'] is None


def test_the_controls_are_track_bs_and_the_rows_keep_the_soaks_form(tables, monkeypatch):
    """A control's evaluation row is its soak row (sim/free.py soak_life) of the same record and seed, plus the
    evaluation's own fields."""
    monkeypatch.setattr(free, '_TABLES', tables)
    for name, period in (('random', None), ('melee', None), ('rotation', 60)):
        soak = free.soak_life((False, 1, 300, 4000, 223, 1, name, period, 5))
        row = free_life((name, None, 1), settings_of(ticks=300, period=period), tables, {})
        assert {k: v for k, v in row.items() if k in soak and k != 'seconds'} == {k: v for k, v in soak.items() if k != 'seconds'}
        assert set(row) - set(soak) == {'rows', 'split', 'first', 'seed', 'read', 'pass', 'policy_seed', 'checkpoint',
                                        'deterministic', 'made', 'kills', 'eat_log', 'eats_full', 'activity'}
    assert free.CONTROLS == CONTROLS  # the controls this file plays are sim/free.py's


def test_a_changed_world_is_a_description_that_no_evaluation_of_a_plan_reads(tmp_path):
    """sim.hour free --change writes its change in the manifest and in each row; sim.gate free refuses either (a
    changed world is a description of the scorecard)."""
    rows = [life('wait', r, {}) for r in range(2)]
    marked = output(tmp_path / 'marked', rows, 2, change=dict(name='no-food', records=[0, 1], passes=[]))
    with pytest.raises(ValueError, match="holds the changed world 'no-food'"):
        free_json([marked], tmp_path / 'f.json')
    with pytest.raises(ValueError, match="a life of the changed world 'no-food'"):
        free_json([output(tmp_path / 'row', [rows[0], life('wait', 1, {}, change='no-food')], 2)], tmp_path / 'f.json')


def test_a_changed_world_plays_the_records_of_its_fact_with_the_base_rows_seeds_and_no_other_setting(tables_v2, tmp_path,
                                                                                                    monkeypatch):
    """The changed world no-highwayman on the plan's first three records: the records whose base holds a highwayman
    alone, each control with the base row's seed (the pair shares its random numbers until the lives part), every row
    and the manifest naming the change; the policy plays its sampled passes alone; refused before any life with a plan,
    in another world, with muggers, with a control its reads do not need or with a greedy pass."""
    from sim import changed
    common = dict(world='village-world-v2', split='development', first=7000, records=3, key=239, muggers=False,
                  xp_rate=1, ticks=200, read=60, control_seed=4253, tables=tables_v2)
    free_run(tmp_path / 'base', controls=('random', 'wait'), **common)
    built = []  # each changed life is built by sim/changed.py from its base record, never as the base life
    monkeypatch.setattr(changed, 'life', lambda change, *a, **k: built.append((change, a[2])) or changed.ChangedLife(change, *a, **k))
    free_run(tmp_path / 'changed', change='no-highwayman', controls=('random', 'wait'), **common)
    held = changed.records('no-highwayman', 7000, 3, 239, tables_v2, 'development')
    assert sorted(built) == sorted(('no-highwayman', r) for r in held for _ in ('random', 'wait'))
    base = {(r['actor'], r['record']): r for r in rows_of(tmp_path / 'base')}
    mine = rows_of(tmp_path / 'changed')
    assert 0 < len(held) < 3 and sorted({r['record'] for r in mine}) == held and len(mine) == 2 * len(held)
    assert all(r['change'] == 'no-highwayman' and r['seed'] == base[(r['actor'], r['record'])]['seed'] for r in mine)
    assert json.loads((tmp_path / 'changed/manifest.json').read_text())['change'] == \
        dict(name='no-highwayman', records=held, passes=[])
    assert free_jobs(None, True, ('random',), dict(records=[2, 5], passes=['sampled1'])) == \
        [('policy', 'sampled1', 2), ('random', None, 2), ('policy', 'sampled1', 5), ('random', None, 5)]
    refusals = [(dict(plan='wider_pilot'), 'no --plan'), (dict(world='village-world-v1'), "village-world-v2's"),
                (dict(muggers=True), 'without muggers'), (dict(controls=('once',)), "not \\['once'\\]"),
                (dict(change_passes=('greedy',)), 'sampled passes')]
    for changes, why in refusals:
        with pytest.raises(ValueError, match=why):
            free_run(tmp_path / 'refused', change='no-highwayman', **{**common, 'controls': ('random',), **changes})
        assert not (tmp_path / 'refused').exists()
