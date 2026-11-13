"""Effects add up to the episode's change, a death is never scored as value held, setup is not
the policy's play, the trace rule keeps what it says, and a run's patch restores its exact
source (baseline/telemetry.py)."""
from collections import Counter
import subprocess

from hypothesis import given, strategies as st

from baseline.telemetry import EpisodeRecorder, TraceSink, effect, source

ITEMS = st.lists(st.tuples(st.sampled_from([(1511, 'Logs'), (317, 'Raw shrimps'), (1351, 'Bronze axe')]),
                           st.integers(1, 5)), max_size=6)
OBSERVATION = st.fixed_dictionaries(dict(
    inventory=ITEMS.map(lambda items: [dict(id=i, name=n, count=c) for (i, n), c in items]),
    skills=st.fixed_dictionaries(dict(Woodcutting=st.integers(0, 10**6), Attack=st.integers(0, 10**6)))
    .map(lambda xp: [dict(name=k, xp=v) for k, v in xp.items()]),
    player=st.fixed_dictionaries(dict(x=st.integers(2368, 3455), z=st.integers(2880, 3519), level=st.just(0),
                                      hp=st.integers(0, 20)))))


@given(st.lists(OBSERVATION, min_size=2, max_size=8))
def test_effects_add_up_to_the_change_over_the_episode(observations):
    total = dict(inventory=Counter(), xp=Counter(), hp=0, dx=0, dz=0)
    for before, after in zip(observations, observations[1:]):
        row = effect(before, after)
        total['inventory'].update(row['inventory']); total['xp'].update(row['xp'])
        total['hp'] += row['hp']; total['dx'] += row['dx']; total['dz'] += row['dz']
    first, last = observations[0], observations[-1]
    expected = Counter()
    for o, sign in ((last, 1), (first, -1)):
        for i in o['inventory']:
            expected[f"{i['name']} ({i['id']})"] += sign * i['count']
    assert {k: v for k, v in total['inventory'].items() if v} == {k: v for k, v in expected.items() if v}
    for skill in ('Woodcutting', 'Attack'):
        xp = lambda o: next(s['xp'] for s in o['skills'] if s['name'] == skill)
        assert total['xp'][skill] == xp(last) - xp(first)
    assert total['hp'] == last['player']['hp'] - first['player']['hp']
    assert (total['dx'], total['dz']) == (last['player']['x'] - first['player']['x'], last['player']['z'] - first['player']['z'])


def raw(tick, x, logs=0, woodcutting=0, feedback=None):
    return dict(tick=tick, player=dict(x=x, z=3200, level=0, hp=10), feedback=feedback or {},
                skills=[dict(name='Woodcutting', xp=woodcutting)], inventory=[dict(id=1511, name='Logs', count=logs)] if logs else [])


def episode(recorder, report, index=0):
    recorder('scheduled_episode', index=index, fixture=dict(id='test-long', kind='long'))
    recorder('requested', action=5)  # scripted setup, before the episode starts
    recorder('observed', action=5, observation=raw(1, 3000))
    recorder('episode_start', observation=raw(2, 3100))
    failed = dict(ended=dict(action=84, reason='failed', refused=False), action=dict(family='click', option='0,-9'))
    refused = dict(ended=dict(action=84, reason='done', refused=True), running=True)
    for tick, after in ((10, dict(raw(10, 3110, feedback=failed), actionNote='click 0,-9 to 3110,3191')),
                        (20, raw(20, 3110, logs=2, woodcutting=50, feedback=refused))):
        recorder('requested', action=84)
        recorder('observed', action=84, observation=after)
        recorder('task_transition', requested_action=84, executed_action=84, native_tick=tick, status='active', reward=.1)
    recorder('task_transition', requested_action=0, executed_action=None, native_tick=30, status='death', reward=-.2)
    recorder('task_terminal', report=report, reward=0.)  # .1 + .1 - .2: a death settles to zero


def test_a_death_scores_zero_even_with_items_held_and_setup_is_not_policy_play():
    rows = []
    recorder = EpisodeRecorder(lambda kind, **data: rows.append((kind, data)))
    episode(recorder, dict(survived=False, produced=2, retained_lower=8, reward_eligible_lower=0, deposit_events=0,
                           production_after_deposit=0, provenance_exact=True))
    assert all(data['episode'] == 0 for _, data in rows)
    effects = [data for kind, data in rows if kind == 'effect']
    assert [e['decision'] for e in effects] == [0, 1, 2] and effects[1]['inventory'] == {'Logs (1511)': 2}
    assert effects[0]['dx'] == 10 and effects[0]['ended'] == dict(action=84, reason='failed', refused=False, note='')
    assert effects[0]['action_note'] == 'click 0,-9 to 3110,3191' and effects[1]['running'] and 'inventory' not in effects[2]
    measures = recorder.last
    assert measures['final_value'] == 0 and measures['value_held_at_end'] == 8 and not measures['survived']
    assert measures['episode_return'] == 0 and measures['goal'] is None
    assert measures['gathering_xp_tick'] == 18 and measures['decision_cells'] == 2  # from the start at 3100, not setup
    assert measures['clicks_requested'] == 2 and measures['clicks_displaced'] == 0
    assert measures['option_ends'] == {'failed': 1, 'done': 1} and measures['options_refused'] == 1
    assert measures['option_problems'] == 2


def test_traces_keep_deaths_problem_heavy_and_sampled_episodes_up_to_the_cap(tmp_path):
    survived = lambda problems, decisions: dict(survived=True, option_problems=problems, decisions=decisions)
    assert TraceSink.reason(7, dict(survived=False, option_problems=0, decisions=3)) == 'death'
    assert TraceSink.reason(7, survived(80, 80)) == 'option_problems'  # e.g. 80 refusals that ended 'done'
    assert TraceSink.reason(7, survived(20, 81)) is None and TraceSink.reason(7, survived(19, 40)) is None
    assert TraceSink.reason(50, survived(0, 10)) == 'sample'
    sink = TraceSink(tmp_path / 'traces.jsonl.gz', cap=1)
    for index in (0, 50):
        sink.offer(index, survived(0, 10), [('effect', dict(agent=0, episode=index))])
    sink.close()
    assert sink.state() == dict(kept={'sample': 1}, dropped={'sample': 1}, cap=1)


def test_episodes_cut_off_before_their_terminal_row_are_kept_as_interrupted(tmp_path):
    sink = TraceSink(tmp_path / 'traces.jsonl.gz')
    recorder = EpisodeRecorder(lambda kind, **data: None, sink)
    recorder('scheduled_episode', index=3, fixture=dict(id='archive-1-2', kind='drop'), agent=0)
    recorder('restart_failed', fixture={}, error='no tile', agent=0)
    recorder('scheduled_episode', index=3, fixture=dict(id='train-long', kind='long'), agent=0)  # replaces it
    recorder('episode_start', observation=raw(2, 3100), agent=0)  # then the run ends mid-episode
    sink.close()
    assert sink.state()['kept'] == {'interrupted': 2}


def test_a_runs_patch_restores_its_exact_source(tmp_path):
    git = lambda *args, cwd: subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)
    repo, copy, run = tmp_path / 'repo', tmp_path / 'copy', tmp_path / 'run'
    for path in (repo, run):
        path.mkdir()
    git('init', '-q', cwd=repo)
    (repo / 'kept.py').write_text('a = 1\n'); (repo / '.gitignore').write_text('logs/\n')
    git('add', '.', cwd=repo); git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base', cwd=repo)
    (repo / 'kept.py').write_text('a = 2\n'); (repo / 'new.py').write_text('b = 3\n')
    (repo / 'logs').mkdir(); (repo / 'logs' / 'ignored.txt').write_text('not source\n')
    state = source(run, root=repo)
    assert state['changed'] == [' M kept.py'] and state['untracked'] == ['new.py'] and state['patch']['bytes'] > 0
    git('clone', '-q', str(repo), str(copy), cwd=tmp_path)
    git('apply', str(run / 'source.patch'), cwd=copy)
    assert (copy / 'kept.py').read_text() == 'a = 2\n' and (copy / 'new.py').read_text() == 'b = 3\n'
    assert not (copy / 'logs').exists() and source(run, root=copy)['commit'] == state['commit']
