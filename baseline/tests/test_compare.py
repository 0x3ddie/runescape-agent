"""Comparisons pair trials by start and trial and refuse runs that cannot be paired (baseline/compare.py)."""
from hypothesis import given, strategies as st
import pytest

from baseline.compare import compare


def trial(fixture, index, value, survived=True, seed=None):
    return dict(fixture=fixture, trial=index, seed=seed if seed is not None else index, success=survived,
                report=dict(reward_eligible_lower=value if survived else 0, produced=value // 4, survived=survived))


def test_paired_change_shows_each_start_and_the_pooled_change():
    baseline = [trial('a', 0, 100), trial('a', 1, 100), trial('b', 0, 40), trial('b', 1, 40)]
    candidate = [trial('b', 1, 0, survived=False), trial('b', 0, 0, survived=False), trial('a', 1, 300), trial('a', 0, 300)]
    result = compare(baseline, candidate)
    assert result['starts']['a']['final_value']['change'] == 200 and result['starts']['b']['final_value']['change'] == -40
    assert result['pooled']['final_value']['mean'] == 80 and result['pooled']['survived']['mean'] == -.5
    assert compare(baseline, baseline)['pooled']['final_value'] == dict(
        baseline=70., candidate=70., mean=0., p=1., low=0., high=0.)


def test_runs_on_different_starts_or_seeds_are_refused():
    baseline = [trial('a', 0, 100), trial('a', 1, 100)]
    with pytest.raises(ValueError, match='same starts'):
        compare(baseline, baseline[:1])
    with pytest.raises(ValueError, match='seeds'):
        compare(baseline, [trial('a', 0, 100), trial('a', 1, 100, seed=9)])


@given(st.lists(st.sampled_from((-1., 0., 1.)), min_size=1, max_size=40), st.integers(1, 4))
def test_the_paired_bootstrap_keeps_each_group_count_and_its_bounds(diffs, n_groups):
    from baseline.compare import paired_bootstrap
    groups = [k % n_groups for k in range(len(diffs))]
    got = paired_bootstrap(diffs, groups, draws=500)
    assert min(diffs) <= got['low'] <= got['mean'] + 1e-12 and got['mean'] - 1e-12 <= got['high'] <= max(diffs)
    assert paired_bootstrap(diffs, groups, draws=500) == got  # one seed, one result
    # a group whose pairs all agree contributes its exact share to every resample
    same = paired_bootstrap([1.] * 3 + [-1.] * 2, ['a'] * 3 + ['b'] * 2, draws=500)
    assert same['low'] == same['high'] == same['mean'] == pytest.approx(.2)


def goal_rows(policy, wins, n=4, profile=lambda k: f'p{k % 2}'):
    return [dict(policy=policy, site='arena', goal='wood', pair=k, group=profile(k), pairing=profile(k), success=w,
                 controlled_success=w, survived=True, first_completed='wood' if w else None,
                 ticks_to_outcome=40 if w else None, non_goal_before_outcome=dict(fish=0, mine=0, melee=0),
                 repeats_before_outcome=1, longest_repeat_before_outcome=2, decisions_before_outcome=5,
                 refusals_before_outcome=0) for k, w in enumerate(wins[:n])]


def test_a_trained_condition_passes_only_with_its_bar_and_both_improvements():
    from baseline.compare import goal_report
    final = goal_rows('final-1', [True] * 4)
    rows = final + goal_rows('initial-1', [False] * 4) + goal_rows('random', [False] * 4)
    report = goal_report(rows, expected=4, bar=3)
    assert report['passed'] and report['gates']['final-1|arena|wood']['comparisons']['random']['low'] == 1.
    assert report['conditions']['final-1|arena|wood']['by_profile'] == dict(p0=dict(trials=2, controlled_successes=2),
                                                                            p1=dict(trials=2, controlled_successes=2))
    tied = goal_report(final + goal_rows('initial-1', [True] * 4) + goal_rows('random', [False] * 4), expected=4, bar=3)
    assert tied['failed'] == ['final-1|arena|wood'] and not tied['gates']['final-1|arena|wood']['comparisons']['initial-1']['improved']
    short = goal_report(final[:3] + rows[4:], expected=4, bar=3)  # a missing trial: invalid, whatever the score
    assert not short['passed'] and short['conditions']['final-1|arena|wood']['missing'] == 1
    moved = goal_report(final + goal_rows('initial-1', [False] * 4) + goal_rows('random', [False] * 4, profile=lambda k: 'x'),
                        expected=4, bar=3)  # the control played other starts: no comparison
    assert not moved['gates']['final-1|arena|wood']['comparisons']['random']['improved']
