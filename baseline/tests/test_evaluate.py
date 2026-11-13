from itertools import groupby

from gymnasium import spaces
from hypothesis import given, strategies as st
import numpy as np
import pytest

from baseline.evaluate import Actor, run_trial, success, summarize


class Task:
    def __init__(self):self.steps=0;self.actions=[];self.raw={}
    def reset(self,seed):
        self.steps=0;self.seed=seed
        return dict(mask=np.asarray([0,1]+[0]*94))
    def step(self,action):
        assert action==1
        self.actions.append(action);self.steps+=1
        return dict(mask=np.asarray([0,1]+[0]*94)),0.,self.steps==2,dict(reward_bounds=[0,.1],resource_report=dict(survived=True))


def record(*args,**kwargs):pass


def test_uniform_random_evaluation_uses_only_legal_actions_and_keeps_bounds():
    task=Task()
    result=run_trial(task,dict(reset_seed=7),Actor('random'),19,record,3)
    assert task.seed==7 and task.actions==[1,1] and result['reward_bounds']==[0,.1]


def test_recurrent_evaluation_clears_state_at_each_new_trial():
    class Model:
        observation_space=spaces.Dict(mask=spaces.Box(0,1,shape=(96,),dtype=np.int8))
        def __init__(self):self.starts=[]
        def predict(self,obs,*,state,episode_start,deterministic):
            self.starts.append(bool(episode_start[0]))
            assert deterministic and (state is None)==bool(episode_start[0])
            return 1,('hidden',)
    model=Model();actor=Actor('checkpoint',model)
    for _ in range(2):run_trial(Task(),dict(reset_seed=0),actor,0,record,3)
    assert model.starts==[True,False,True,False]


def test_evaluation_never_converts_budget_exhaustion_to_a_scored_trial():
    with pytest.raises(RuntimeError,match='decision limit'):
        run_trial(Task(),dict(reset_seed=0),Actor('random'),0,record,1)


def report(**overrides):
    return dict(dict(survived=True,retained_lower=40,deposit_events=1,production_after_deposit=3),**overrides)


def test_success_requires_survival_and_the_fixture_specific_behavior():
    assert success('long',report(),bank_actions=5)
    assert not success('long',report(survived=False),bank_actions=5)
    assert not success('long',report(production_after_deposit=0),bank_actions=5)
    assert success('short',report(retained_lower=2,deposit_events=0),bank_actions=0)
    assert not success('short',report(retained_lower=2,deposit_events=0),bank_actions=1)


def test_summaries_score_a_death_zero_and_report_what_was_held_apart():
    trials = [dict(fixture='long', success=False, report=dict(survived=False, retained_lower=80, reward_eligible_lower=0, produced=20)),
              dict(fixture='long', success=True, report=dict(survived=True, retained_lower=40, reward_eligible_lower=40, produced=10))]
    assert summarize(trials)['long'] == dict(trials=2, successes=1, survived=1, mean_final_value=20.,
                                             mean_value_held_at_end=60., mean_items_produced=15.)


def production(tick, base):
    return dict(kind='native-op', tick=tick, productionSource=True, stackTail=[1, base, 1])


def test_goal_control_reports_the_first_completed_activity_and_what_came_before_it():
    from baseline.evaluate import goal_control
    trace = dict(startTick=100, events=[production(110, 436), production(120, 1511), production(125, 438),
                                        production(130, 1511), production(131, 317)])
    effects = [dict(tick=105, ended=dict(refused=True)), dict(tick=140, ended=dict(refused=True)), dict(tick=106)]
    report = dict(kills=dict(lower=1, upper=1, kill_ticks=[127]))
    mine = goal_control('mine', trace, report, effects, succeeded=True)  # two ores by 125: first
    assert mine['first_completed'] == 'mine' and mine['controlled_success'] and mine['ticks_to_outcome'] == 25
    assert mine['non_goal_before_outcome'] == dict(wood=1, fish=0, melee=0) and mine['refusals_before_outcome'] == 1
    wood = goal_control('wood', trace, report, effects, succeeded=True)  # the logs came second
    assert wood['first_completed'] == 'mine' and not wood['controlled_success'] and wood['ticks_to_outcome'] == 30
    assert wood['non_goal_before_outcome'] == dict(fish=0, mine=2, melee=1)
    melee = goal_control('melee', trace, report, effects, succeeded=False)
    assert melee['completion_ticks']['melee'] == 27 and not melee['controlled_success']
    fish = goal_control('fish', trace, dict(kills=None), effects, succeeded=False)  # one fish: never completed
    assert fish['ticks_to_outcome'] is None and fish['refusals_before_outcome'] == 2
    tie = goal_control('wood', dict(startTick=0, events=[production(5, 1511), production(9, 1511), production(7, 317),
                                                         production(9, 317)]), {}, [], succeeded=True)
    assert tie['first_completed'] == 'tie' and not tie['controlled_success']
    decisions = [(101, 3), (104, 3), (112, 3), (124, 7), (126, 3), (128, 3)]  # (tick, action); mine is done at 125
    mine = goal_control('mine', trace, report, effects, succeeded=True, decisions=decisions)
    assert (mine['decisions_before_outcome'], mine['repeats_before_outcome'], mine['longest_repeat_before_outcome']) == (4, 2, 3)
    assert goal_control('fish', trace, report, effects, succeeded=False, decisions=decisions)['repeats_before_outcome'] == 3


@given(st.lists(st.tuples(st.integers(0, 300), st.integers(0, 3))), st.one_of(st.none(), st.integers(0, 300)))
def test_repeats_count_each_decision_that_repeats_the_one_before_it(decisions, done):
    from baseline.evaluate import repeats
    decisions = sorted(decisions)
    got = repeats(decisions, done)
    before = [a for t, a in decisions if done is None or t < done]
    assert got['decisions_before_outcome'] == len(before)
    assert got['repeats_before_outcome'] == sum(a == b for a, b in zip(before, before[1:]))
    assert got['longest_repeat_before_outcome'] == max((len(list(run)) for _, run in groupby(before)), default=0)


def test_a_melee_goal_succeeds_with_one_certified_kill_and_survival():
    kills = lambda n: dict(survived=True, families={}, kills=dict(lower=n, upper=n))
    assert success('goal', kills(1), 0, 'melee') and not success('goal', kills(0), 0, 'melee')
    assert not success('goal', dict(kills(1), survived=False), 0, 'melee')


def test_each_start_set_serves_its_goals_with_one_record_list_drawn_from_the_simulator_spec():
    from baseline import config
    from baseline.scripted import KIT_NAMES
    from sim.spec import HORIZON_TICKS, START_KIT, START_LEVELS, START_WEAPONS
    assert {name: s['goals'] for name, s in config.START_SETS.items()} == dict(
        arena=('wood', 'fish', 'mine', 'melee'), entrana=('wood', 'fish', 'melee'), rimmington=('wood', 'mine'))
    for name, start_set in config.START_SETS.items():
        fixtures = [config.EVAL_FIXTURES[f'{name}_{goal}'] for goal in start_set['goals']]
        assert all(f['final'] and f['starts'] == fixtures[0]['starts'] and f['horizon'] == HORIZON_TICKS for f in fixtures)
        assert not any(f'{name}_{goal}' in config.EVAL_FIXTURES for goal in ('wood', 'fish', 'mine', 'melee')
                       if goal not in start_set['goals'])
    for start in config.start_records('arena'):
        items = [pair.split(':') for pair in start['seed_items'].split(',')]
        assert [(n, int(c)) for n, c in items[:-1]] == list(START_KIT) and items[-1] == [items[-1][0], '1']
        assert items[-1][0] in START_WEAPONS and all(n in KIT_NAMES for n, _ in items)
        stats = dict(pair.split(':') for pair in start['set_stats'].split(','))
        assert set(stats) == set(START_LEVELS) and all(int(stats[s]) in levels for s, levels in START_LEVELS.items())


def test_the_native_supply_demonstration_plays_training_draws_never_final_records():
    from baseline import config
    from baseline.evaluate import check_start_sets
    final = {(r['set_stats'], r['seed_items'], r['reset_seed']) for r in config.study_records('entrana', 'eval')}
    for name, fixture in config.SUPPLY_DEMO_FIXTURES.items():
        train = config.study_records(fixture['start_set'], 'train')[:config.SUPPLY_DEMO_RECORDS]
        assert [dict(r, start=r['start'].replace('-train-', '-demo-')) for r in train] == fixture['starts']
        assert not fixture['final'] and fixture['horizon'] == 300 and fixture['goal'] == 'wood'
        assert not {(r['set_stats'], r['seed_items'], r['reset_seed']) for r in fixture['starts']} & final
    assert check_start_sets(list(config.SUPPLY_DEMO_FIXTURES), False, lambda engine: config.ENGINE_TREES['pinned']) == 'pinned'


def test_the_native_check_records_are_a_new_frozen_draw_disjoint_from_every_earlier_native_record():
    """CHECK_RECORDS draws per site from CHECK_SEED, none equal to a study evaluation or training draw (the supply
    demonstration's among them) or a 17 October transfer start; frozen by their digests; final data at each skill's two
    sites, registered after every earlier fixture so their trial seeds stay as they were."""
    from baseline import config
    from baseline.evaluate import check_start_sets
    key = lambda r: (r['set_stats'], r['seed_items'], r['reset_seed'])
    assert config.verify_check_records() == config.CHECK_DIGESTS
    drawn = {key(r) for r in config.check_records('arena')}
    earlier = {key(r) for name in config.START_SETS for r in config.study_records(name, 'eval') + config.start_records(name)}
    earlier |= {key(r) for f in config.SUPPLY_DEMO_FIXTURES.values() for r in f['starts']}
    earlier |= {key(r) for r in config.study_records('entrana', 'train')}
    assert len(drawn) == config.CHECK_RECORDS == 16 and not drawn & earlier
    names = list(config.EVAL_FIXTURES)
    assert names[-12:] == list(config.CHECK_FIXTURES) + list(config.REHEARSAL_FIXTURES) and len(config.CHECK_FIXTURES) == 6
    final = {key(r) for name in config.START_SETS for r in config.study_records(name, 'eval') + config.start_records(name)}
    for name, f in config.REHEARSAL_FIXTURES.items():  # rehearsals play the study's training draws, never final data
        train = config.study_records(f['start_set'], 'train', config.CHECK_RECORDS)
        assert not f['final'] and [key(r) for r in f['starts']] == [key(r) for r in train]
        assert not {key(r) for r in f['starts']} & (final | drawn)
        assert (f['goal'], f['start_set'], f['horizon']) == (lambda c: (c['goal'], c['start_set'], c['horizon']))(
            config.CHECK_FIXTURES[name.replace('rehearse_', 'check_')])
    for name, f in config.CHECK_FIXTURES.items():
        assert f['final'] and f['horizon'] == config.CHECK_HORIZONS[f['goal']] and f['start_set'] in config.CHECK_SITES[f['goal']]
        assert f['goal'] in config.START_SETS[f['start_set']]['goals'] and len(f['starts']) == 16
        assert {key(r) for r in f['starts']} == drawn
    with pytest.raises(ValueError, match='pass --final'):
        check_start_sets(['check_entrana_wood'], False, lambda engine: config.ENGINE_TREES['pinned'])
    original = config.CHECK_DRAWS[3]
    try:
        config.CHECK_DRAWS[3] = (original[0], original[1], original[2] + 1)
        with pytest.raises(ValueError, match='frozen'):
            config.verify_check_records()
    finally:
        config.CHECK_DRAWS[3] = original


def test_evaluation_refuses_unsupported_goals_unflagged_final_data_and_the_wrong_engine(monkeypatch):
    from baseline import config
    from baseline.evaluate import check_start_sets
    pinned, arena = (lambda engine: config.ENGINE_TREES['pinned']), (lambda engine: config.ENGINE_TREES['arena'])
    check_start_sets(['entrana_wood', 'rimmington_mine'], True, pinned)
    check_start_sets(['arena_mine', 'arena_melee'], True, arena)
    check_start_sets(['goal_wood'], False, lambda engine: None)  # goal-v1 fixtures have no start set
    with pytest.raises(ValueError, match='pass --final'):
        check_start_sets(['entrana_wood'], False, pinned)
    with pytest.raises(ValueError, match='not the build'):
        check_start_sets(['arena_mine'], True, pinned)
    with pytest.raises(ValueError, match='not the build'):
        check_start_sets(['entrana_fish'], True, lambda engine: None)  # tracked changes in the engine checkout
    with pytest.raises(ValueError, match='different engine builds'):
        check_start_sets(['entrana_wood', 'arena_wood'], True, pinned)
    monkeypatch.setitem(config.EVAL_FIXTURES, 'entrana_mine', dict(config.EVAL_FIXTURES['entrana_wood'], goal='mine'))
    with pytest.raises(ValueError, match='does not support goal mine'):
        check_start_sets(['entrana_mine'], True, pinned)


@pytest.mark.parametrize('spectator,options', [
    (None, dict(demo=True)),  # --demo outside a demo world
    ('viewer', dict()),  # an ordinary evaluation in a demo world
    ('viewer', dict(demo=True, fixtures=['check_entrana_wood'], final=True)),
    ('viewer', dict(demo=True, fixtures=['entrana_wood'])),  # final data even without --final
    ('viewer', dict(demo=True, trials=2)),
    ('viewer', dict(demo=True, fixtures=['rehearse_entrana_wood', 'rehearse_entrana_fish'])),
    ('viewer', dict(demo=True, world_service=True)),
])
def test_a_demo_plays_one_development_trial_and_a_demo_world_serves_nothing_else(tmp_path, monkeypatch, spectator, options):
    from baseline.evaluate import evaluate
    if spectator:
        monkeypatch.setenv('RL_DEMO_SPECTATOR', spectator)
    else:
        monkeypatch.delenv('RL_DEMO_SPECTATOR', raising=False)
    with pytest.raises(ValueError, match='demo'):
        evaluate(tmp_path / 'out', policy='wait', **{'fixtures': ['rehearse_entrana_wood'], **options})
    assert not (tmp_path / 'out').exists()


def test_a_demo_start_is_known_only_to_a_demo_process(monkeypatch, tmp_path):
    """The start of a watched demo stays out of every other fixture list, and an ordinary evaluation refuses it."""
    from baseline import config, evaluate as module
    played = []
    monkeypatch.setattr(module, 'evaluate', lambda output, **kw: played.append(kw['fixtures']))
    monkeypatch.setattr(config, 'EVAL_FIXTURES', dict(config.EVAL_FIXTURES))
    assert 'demo_far_wood' not in config.EVAL_FIXTURES and config.DEMO_FIXTURES['demo_far_wood']['final'] is False
    argv = ['evaluate', '--output', str(tmp_path), '--policy', 'wait', '--fixtures', 'demo_far_wood']
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit):
        module.main()
    assert not played and 'demo_far_wood' not in config.EVAL_FIXTURES
    monkeypatch.setattr('sys.argv', [*argv, '--demo'])
    module.main()
    assert played == [['demo_far_wood']] and list(config.EVAL_FIXTURES)[-1] == 'demo_far_wood'
