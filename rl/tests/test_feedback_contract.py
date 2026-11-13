"""feedback-v2 (text-v2) reaches the policy as fixed, finite arrays (rl/feedback_contract.py)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from feedback_contract import (END_REASONS, FAMILIES, FEEDBACK_FEATS, TEXT_DIM, TEXT_FEATS, episode_start, quiet,
                               text_vector, words)
from obs import project

CAPTURE = Path(__file__).resolve().parent / 'fixtures' / 'reset.json'
if not CAPTURE.exists():
    pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository', allow_module_level=True)
RESET = json.load(open(CAPTURE))
NET = dict(action=1, family='act', kind='npc', name='Fishing spot', option='Net', item='')

text = st.text(st.characters(codec='utf-8'), max_size=80)
action = st.one_of(st.none(), st.builds(lambda f, name, option, item: dict(action=0, family=f, kind='npc', name=name,
                                                                            option=option, item=item),
                                        st.sampled_from(FAMILIES), text, text, text))
ended = st.one_of(st.none(), st.builds(lambda r, refused: dict(action=1, reason=r, refused=refused),
                                       st.sampled_from(END_REASONS), st.booleans()))
count = st.integers(0, 10**6)
feedback = st.builds(
    lambda a, e, ticks, dx, dz, xp, hp, gained, lost, flags, chat, dialog, choices: dict(
        version='feedback-v2', action=a, running=flags[0], ended=e, ticks=ticks, dx=dx, dz=dz, levelChanged=flags[1],
        xp=xp, hp=hp, gained=gained, lost=lost, bankOpened=flags[2], dialogOpened=flags[3],
        chat=[dict(text=t, age=age) for t, age in chat], dialog=dict(text=dialog, choices=choices)),
    action, ended, count, st.integers(-10**4, 10**4), st.integers(-10**4, 10**4), count, st.integers(-99, 99), count,
    count, st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()),
    st.lists(st.tuples(text, st.integers(0, 10**4)), max_size=6), text, st.lists(text, max_size=6))


@settings(max_examples=200, deadline=None)
@given(feedback)
def test_any_valid_feedback_projects_to_fixed_finite_arrays(fb):
    out = project(dict(RESET, feedback=fb))
    assert out['fb_vec'].shape == (FEEDBACK_FEATS,) and out['text_vec'].shape == (TEXT_FEATS,)
    assert out['fb_ids'].shape == (2,) and out['fb_op'].shape == (1,) and out['choice_vec'].shape == (4, 16)
    assert all(np.isfinite(v).all() for v in out.values())


@pytest.mark.parametrize('mutation', [
    lambda f: f.update(version='feedback-v0'),
    lambda f: f.update(ticks=-1),
    lambda f: f.update(running=1),
    lambda f: f.update(ended=dict(action=1, reason='gave up', refused=False)),
    lambda f: f.update(action=dict(NET, family='chop')),
    lambda f: f.update(chat=[dict(text='hi', age=-1)]),
    lambda f: f.pop('dialog'),
])
def test_malformed_feedback_is_refused(mutation):
    fb = quiet()
    mutation(fb)
    with pytest.raises(ValueError):
        project(dict(RESET, feedback=fb))


def test_a_failed_net_and_a_catch_differ_and_setup_does_not_leak():
    failed = dict(quiet(), action=NET, ended=dict(action=1, reason='done', refused=True), ticks=6, dialogOpened=True,
                  dialog=dict(text='You need a net to catch these fish.', choices=[]))
    caught = dict(quiet(running=True), action=NET, ticks=6, xp=10, gained=1,
                  chat=[dict(text='You catch some shrimps.', age=2)])
    a, b = project(dict(RESET, feedback=failed)), project(dict(RESET, feedback=caught))
    assert a['fb_op'][0] == b['fb_op'][0] != 0
    assert not np.array_equal(a['fb_vec'], b['fb_vec'])
    assert np.any(a['text_vec'][TEXT_DIM:2 * TEXT_DIM]) and not np.any(b['text_vec'][TEXT_DIM:2 * TEXT_DIM])
    # Setup ran as ordinary steps: the episode's first observation keeps only what is on screen.
    start = episode_start(dict(RESET, feedback=dict(failed, chat=[dict(text='You deposit your logs.', age=1)])))['feedback']
    assert start == quiet(dialog=failed['dialog'])


def test_text_features_are_the_same_in_another_process():
    """A process-seeded hash would give a policy different inputs in every run."""
    line = 'You need a net to catch these fish.'
    other = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, "rl"); from feedback_contract import '
                            f'text_vector; import json; print(json.dumps(text_vector([{line!r}], 64).tolist()))'],
                           capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parents[2])
    np.testing.assert_allclose(json.loads(other.stdout), text_vector([line], 64), rtol=1e-6)


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 10**12), st.integers(0, 10**12))
def test_numbers_below_one_hundred_stay_distinct_and_larger_ones_fall_in_bounded_buckets(a, b):
    """text-v2: a level requirement of 15 and one of 30 are different text; the vocabulary stays bounded."""
    line = 'You need a Mining level of {} to mine this rock.'
    (ta,), (tb,) = ([t for t in words(line.format(n)) if t not in words(line.format(''))] for n in (a, b))
    assert (ta == tb) == ((a == b) if max(a, b) < 100 else len(str(a)) == len(str(b)) or min(a, b) >= 10**9)
    assert ta == str(a) if a < 100 else ta == f'<num{min(len(str(a)), 10)}>'
    if a != b and max(a, b) < 100:
        assert not np.array_equal(text_vector([line.format(a)], TEXT_DIM), text_vector([line.format(b)], TEXT_DIM))
