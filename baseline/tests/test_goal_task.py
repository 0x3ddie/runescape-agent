"""goal-v2: the requested gathering activity's certified items, min(n, 2) / 2, or melee's certified
kills, min(n, 1), zero after a death (baseline/resource_task.py). The audit is the native one;
these check what the goal adds."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from baseline.adapter import GameVecEnv
from baseline.evidence import EvidenceError
from baseline.native_value import audit_native_value
from baseline.resource_task import GOAL_FAMILIES, GOAL_VEC, GOALS, ResourceBackend, ResourceEpisode, goal_return
from baseline.tests.test_native_flow import Trace, drop, item, pickup, trace
from baseline.tests.test_resource_task import CFG, World, raw, small_model

SHRIMP, OAK, COPPER = 317, 1521, 436
# Every goal family declared, as the engine's gathering-v1 scope does (46 families there).
DECLARED = [dict(base=base, note=base + 1, value=1) for families in GOAL_FAMILIES.values() for base in sorted(families)]


def fish(t, slot):
    t.op(inventory=t.current['inventory'] + [item(slot, id=SHRIMP)], production=True)
    t.events[-1].update(scriptName='[proc,fish_roll]', stackTail=[1, SHRIMP, 1])


def report(t, death=False):
    result = trace(t, death)
    result.update(targetScope='gathering-v1', targets=deepcopy(DECLARED))
    return audit_native_value(result)


@settings(max_examples=300, deadline=None)
@given(st.dictionaries(st.sampled_from([*(base for families in GOAL_FAMILIES.values() for base in sorted(families)), 973]),
                       st.integers(0, 5), max_size=6), st.booleans(), st.integers(0, 3), st.integers(0, 3))
def test_the_goal_return_counts_only_the_goal_capped_and_zero_after_death(retained, survived, kills, extra):
    families = {base: dict(retained_lower=n, retained_upper=n, produced=n) for base, n in retained.items()}
    report = dict(survived=survived, families=families, kills=dict(lower=kills, upper=kills + extra))
    for goal in GOALS:
        if goal == 'melee':
            assert goal_return(report, goal) == (min(kills, 1) if survived else 0.)
            assert goal_return(report, goal, 'retained_upper') == (min(kills + extra, 1) if survived else 0.)
            continue
        n = sum(k for base, k in retained.items() if base in GOAL_FAMILIES[goal])
        assert goal_return(report, goal) == (min(n, 2) / 2 if survived else 0.)


@pytest.mark.parametrize('kills', [None, dict(lower=1), dict(lower=2, upper=1), dict(lower=-1, upper=0),
                                   dict(lower=True, upper=1)])
def test_melee_without_a_valid_combat_audit_is_refused(kills):
    report = dict(survived=True, families={}, **({} if kills is None else dict(kills=kills)))
    with pytest.raises(EvidenceError):
        goal_return(report, 'melee')


def test_one_activity_never_scores_for_the_other():
    t = Trace(); t.produce(slot=0); t.produce(slot=1)
    assert (goal_return(report(t), 'wood'), goal_return(report(t), 'fish')) == (1., 0.)
    t = Trace(); fish(t, 0)
    assert (goal_return(report(t), 'wood'), goal_return(report(t), 'fish')) == (0., .5)


def test_moving_one_produced_log_around_never_adds_credit():
    t = Trace(); t.produce(slot=0)
    t.op(inventory=[], bank=[item(0)])        # deposit
    t.op(inventory=[item(3)], bank=[])        # withdraw into another slot
    for key in range(1, 6):
        pickup(t, drop(t, key, slot=3), slot=3)  # repeated drop and recovery
    assert goal_return(report(t), 'wood') == .5


def test_items_from_outside_the_episode_never_count():
    t = Trace(); t.external(slot=0, count=2)  # a stack of two logs picked up, none produced
    assert goal_return(report(t), 'wood') == 0.


def test_a_death_cancels_the_goal():
    t = Trace(); t.produce(slot=0); t.produce(slot=1)
    assert goal_return(report(t, death=True), 'wood') == 0.


class GoalWorld(World):
    """The fake world of test_resource_task, declaring every goal family at arm."""
    def reset(self, seed, cfg):
        observation = super().reset(seed, cfg)
        self.receipt['targets'] = deepcopy(DECLARED)
        return observation


def goal_episode(world, goal):
    return ResourceEpisode(world, world.control, CFG, 2, 10, world.record, goal=goal)


class MeleeWorld(GoalWorld):
    """GoalWorld whose capture also carries combat receipts (baseline/tests/test_native_combat.py)."""
    def __init__(self, fight, **kwargs):
        super().__init__(**kwargs)
        self.fight = fight

    def reset(self, seed, cfg):
        observation = super().reset(seed, cfg)
        if self.fight is not None:
            receipt = self.fight.trace(death=self.death)
            self.receipt['combat'] = dict(receipt['combat'], identity=deepcopy(self.receipt['identity']))
        return observation

    def control(self, operation, **kwargs):
        reply = super().control(operation, **kwargs)
        if operation == 'arm' and 'combat' in self.receipt:
            reply['audit']['combat'] = dict(deepcopy(self.receipt['combat']), events=[], open=True)
        return reply


def chicken_kill():
    from baseline.tests.test_native_combat import Fight
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1); f.hit('7:0', 'agent', 2)
    return f


@pytest.mark.parametrize('death,total', [(False, 1.), (True, 0.)])
def test_melee_settles_at_the_end_to_the_certified_kill(death, total):
    w = MeleeWorld(chicken_kill(), death=death); e = goal_episode(w, 'melee')
    assert e.reset(0)['goal_vec'].tolist() == [0., 0., 0., 1., 0., 0.]  # goal-v3: cook and fire last
    rewards = []
    for _ in range(2):
        _, r, done, info = e.step(0); rewards.append(r)
        if done: break
    assert rewards[:-1] == [0.] * (len(rewards) - 1) and sum(rewards) == pytest.approx(total)
    assert info['resource_report']['kills']['lower'] == 1 and info['reward_bounds'] == [total, total]


def test_melee_without_combat_evidence_is_refused_at_arm_and_other_goals_report_kills():
    with pytest.raises(EvidenceError, match='combat capture at arm'):
        goal_episode(MeleeWorld(None), 'melee').reset(0)
    e = goal_episode(MeleeWorld(chicken_kill()), 'wood'); e.reset(0); e.step(0)
    _, r, done, info = e.step(0)
    assert done and info['resource_report']['kills'] == dict(lower=1, upper=1, kill_ticks=[10], lives=1,
                                                             enemy_lives_hit=1, other_defeats=0)
    assert info['reward_bounds'][0] == 1.  # the fake capture's two logs; the kill adds nothing to wood


@pytest.mark.parametrize('goal,death,total', [('wood', False, 1.), ('wood', True, 0.), ('fish', False, 0.)])
def test_payments_follow_held_goal_items_and_settle_to_the_certified_goal_return(goal, death, total):
    w = GoalWorld(death=death); e = goal_episode(w, goal); first = e.reset(0); held = iter((3, 1))
    assert first['goal_vec'].tolist() == [goal == g for g in GOAL_VEC]
    def step(action):
        w.calls += 1
        if death: w.state = 'death' if w.calls == 2 else 'active'
        else:
            w.tick += 1
            if w.tick == 12: w.state = 'horizon'
        return dict(raw(), inventory=[dict(slot=k, id=1511, count=1, name='Logs') for k in range(next(held))])
    w.step = step
    _, a, done, _ = e.step(0)   # three logs carried: full progress for wood, nothing for fish
    _, b, done, info = e.step(0)  # two lost: half taken back; the terminal audit settles the rest
    assert a == (1. if goal == 'wood' else 0.) and done
    # The fake trace certifies two logs (one banked); death cancels them.
    assert info['reward_bounds'][0] == total and a + b == pytest.approx(total) and info['goal'] == goal


def test_a_new_life_starts_the_goal_at_zero():
    w = GoalWorld(); e = goal_episode(w, 'wood'); e.reset(0)
    step = w.step
    w.step = lambda action: dict(step(action), inventory=[dict(slot=k, id=1511, count=1, name='Logs') for k in range(2)])
    e.step(0); e.step(0)
    assert e.paid == 1. * e.scale  # two logs carried in the first life
    w.step = step
    e.reset(0)
    assert e.paid == 0 and e.banked == e.carried == 0
    _, r, _, _ = e.step(0)
    assert r == 0.


def test_a_goal_whose_families_are_not_all_certified_is_refused_at_arm():
    w = World()  # declares logs only
    with pytest.raises(EvidenceError):
        goal_episode(w, 'fish').reset(0)


def test_the_goal_reaches_actor_and_critic_and_leaves_the_legal_actions_alone():
    torch.set_num_threads(1)
    sample = goal_episode(GoalWorld(), 'wood').reset(0)
    model = small_model(GameVecEnv(ResourceBackend(goal_episode(GoalWorld(), 'wood')), sample), 7)
    policy = model.policy
    sample['mask'][:5] = 1
    outputs = []
    for goal in GOALS:
        obs, _ = policy.obs_to_tensor(dict(sample, goal_vec=np.asarray([goal == g for g in GOAL_VEC], np.float32)))
        state = torch.zeros(policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
        with torch.no_grad():
            dist, _ = policy.get_distribution(obs, (state, state), torch.ones(1))
            value = policy.predict_values(obs, (state, state.clone()), torch.ones(1))
        outputs.append((dist.distribution.probs[0], value))
    assert len(outputs) == 4
    for i, (p, v) in enumerate(outputs):
        assert torch.equal(p > 0, torch.as_tensor(sample['mask'] > 0))
        for q, w in outputs[i + 1:]:
            assert not torch.allclose(p, q) and not torch.allclose(v, w)
