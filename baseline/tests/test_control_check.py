"""Control checks judge from what was recorded: a failed click counts whatever decision reported
it, a door passes only on that door's own state and response, and the click away's control verdict is
kept apart from its effect on the fight (baseline/control_check.py)."""
from baseline.control_check import CHECKS, verdict


def facts(**overrides):
    return dict(dict(arrived_tick=None, door_opens=0, entered=False, setup=None, retreat_decision=None), **overrides)


def seen(at, door=None, attacker=None, hp=20, in_combat=False):
    return dict(tick=0, at=at, hp=hp, in_combat=in_combat, attacker=attacker, door=door)


def row(decision, action, reason=None, refused=False, ended_action=None, **extra):
    e = dict(decision=decision, action=action, **extra)
    if reason:
        e['ended'] = dict(action=action if ended_action is None else ended_action, reason=reason, refused=refused, note='')
    return e


def try_door(decision, door, refused, chat=()):
    return row(decision, 1, 'done', refused, meaning=dict(name='Door', option='Open'),
               action_note=f'act_0 issued Open loc=1530@{door[0]},{door[1]},0', chat=list(chat))


def test_a_failed_click_counts_even_when_a_later_decision_reports_it():
    effects = [row(0, 85), row(1, 0, 'failed', ended_action=85)]
    found = verdict('open_route', CHECKS['open_route'], facts(arrived_tick=40), effects, [seen((0, 0))] * 3)
    assert found['clicks_failed'] == 1 and not found['passed']


def test_an_ordinary_door_passes_only_if_opening_that_door_changed_it():
    door = CHECKS['closed_door']['door']
    trace = [seen((3101, 3258), 'closed'), seen((3101, 3258), 'open'), seen((3108, 3258), 'open')]
    good = verdict('closed_door', CHECKS['closed_door'], facts(arrived_tick=9), [try_door(0, door, False), row(1, 86, 'done')], trace)
    assert good['passed'] and good['opened'] == 1 and good['door_states'] == ['closed', 'open']
    was_open = [seen((3101, 3258), 'open')] + trace[1:]
    assert not verdict('closed_door', CHECKS['closed_door'], facts(arrived_tick=9), [try_door(0, door, False)], was_open)['passed']
    assert not verdict('closed_door', CHECKS['closed_door'], facts(arrived_tick=9), [try_door(0, door, True)], trace)['passed']


def test_a_locked_door_needs_its_message_and_must_stay_closed():
    door, locked = CHECKS['locked_door']['door'], ['This door is locked.']
    trace = [seen((2609, 3316), 'closed')] * 3
    assert verdict('locked_door', CHECKS['locked_door'], facts(), [try_door(0, door, True, locked), try_door(1, door, True, locked)], trace)['passed']
    assert not verdict('locked_door', CHECKS['locked_door'], facts(), [try_door(0, door, True)], trace)['passed']  # no message
    assert not verdict('locked_door', CHECKS['locked_door'], facts(), [try_door(0, door, True, locked)],
                       [seen((2609, 3316), 'closed'), seen((2609, 3316), 'open')])['passed']


def test_the_click_away_is_judged_apart_from_its_effect_on_the_fight():
    # The attacker stood a tile away and followed: the click 9 west ended its route 6 tiles farther from where it
    # stood, but the actual attacker ended only 4 tiles off.
    away = 86  # click slot 14, (-9, 0)
    trace = [seen((100, 100), attacker=(101, 100), in_combat=True), seen((94, 100), attacker=(98, 100), in_combat=True, hp=18)]
    effects = [row(0, away, action_note='click -9,0 to 94,100'), row(1, 0, 'done', ended_action=away)]
    found = verdict('retreat', CHECKS['retreat'], facts(retreat_decision=0, retreat_action=away), effects, trace + [trace[-1]])
    assert found['passed'] and found['control'] == dict(retreat_end='done', destination=(94, 100), reached=True,
                                                        gain_from_attacker_start=6)
    assert found['effect'] == dict(attacker_distance=[1, 4], in_combat=[True, True], hp=[20, 18])
