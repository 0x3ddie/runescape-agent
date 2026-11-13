"""Certified melee kills (baseline/native_combat.py): one credit per enemy life, from full health at this
agent's first hit, only when this agent's damage alone defeats it; bad evidence raises."""
from copy import deepcopy

import pytest
from hypothesis import given, settings, strategies as st

from baseline.evidence import EvidenceError
from baseline.native_combat import SCHEMA, audit_native_combat
from baseline.resource_task import MELEE_ENEMIES, goal_return
from baseline.tests.test_native_flow import Trace, drop, pickup, trace

CHICKEN, GUARD = 41, 26


class Fight:
    """Combat receipts as env-ts/engine/combat-capture.ts writes them: a queued hit, then its damage."""
    def __init__(self, tick=10):
        self.events, self.tick, self.hp, self.base, self.heroes, self.types = [], tick, {}, {}, {}, {}

    def add(self, kind, **data):
        self.events.append(dict(sequence=len(self.events) + 1, tick=self.tick, kind=kind, **data))

    def life(self, key, hp=3, base=3, type=CHICKEN, previous=None):
        self.hp[key], self.base[key], self.heroes[key], self.types[key] = hp, base, dict(agent=0, other=0), type
        self.add('life', life=key, nid=int(key.split(':')[0]), type=type, hp=hp, baseHp=base, x=0, z=0, level=0,
                 previous=previous)

    def hit(self, key, hitter, damage):
        side = 'agent' if hitter == 'agent' else 'other'
        self.heroes[key][side] += min(damage, self.hp[key])  # npc_heropoints($damage_capped)
        self.add('hit', life=key, hitter=hitter, damage=damage, delay=2, hp=self.hp[key], baseHp=self.base[key],
                 heroes=dict(self.heroes[key]))
        before, after = self.hp[key], max(0, self.hp[key] - damage)
        extra = dict(heroes=dict(self.heroes[key])) if before > 0 and after == 0 else {}
        self.add('damage', life=key, hitter=hitter, hit=self.events[-1]['sequence'], amount=damage, hitType=1,
                 hpBefore=before, hpAfter=after, **extra)
        self.hp[key] = after

    def change(self, key, hp):
        self.add('hp', life=key, hpBefore=self.hp[key], hpAfter=hp)
        self.hp[key] = hp

    def respawn(self, key, new):
        self.add('removed', life=key)
        self.life(new, type=self.types[key], previous=key)

    def trace(self, t=None, death=False):
        result = trace(t or Trace(), death)
        result['combat'] = dict(schema=SCHEMA, identity=deepcopy(result['identity']), startTick=result['startTick'],
                                endTick=result['endTick'], enemyTypes=sorted(MELEE_ENEMIES), agentHp=10,
                                events=deepcopy(self.events), failure=None, open=False)
        return result


def kills(fight, **kwargs):
    report = audit_native_combat(fight.trace(**kwargs), MELEE_ENEMIES)
    return report['lower'], report['upper']


def test_this_agents_kill_from_full_health_counts_once():
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1); f.hit('7:0', 'agent', 0); f.hit('7:0', 'agent', 2)
    assert kills(f) == (1, 1)
    report = audit_native_combat(f.trace(), MELEE_ENEMIES)
    assert report['kill_ticks'] == [10] and report['enemy_lives_hit'] == 1


def test_damage_alone_a_wounded_start_and_another_killer_earn_nothing():
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 2)
    assert kills(f) == (0, 0)                                       # damaged, not defeated
    f = Fight(); f.life('7:0', hp=2); f.hit('7:0', 'agent', 2)
    assert kills(f) == (0, 1)                                       # wounded before the first hit
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1); f.hit('7:0', 'player', 2)
    assert kills(f) == (0, 1)                                       # another actor defeated it
    f = Fight(); f.life('7:0'); f.hit('7:0', 'player', 1); f.hit('7:0', 'agent', 2)
    assert kills(f) == (0, 1)                                       # hurt by another actor first
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1); f.change('7:0', 1); f.hit('7:0', 'agent', 1)
    assert kills(f) == (0, 1)                                       # a loss no hit explains
    f = Fight(); f.life('8:0', type=GUARD); f.hit('8:0', 'agent', 3)
    assert kills(f) == (0, 0)                                       # not a declared enemy


def test_regeneration_does_not_spoil_a_kill_and_each_respawn_is_a_new_life():
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1); f.change('7:0', 3); f.hit('7:0', 'agent', 3)
    f.respawn('7:0', '7:1'); f.hit('7:1', 'agent', 3)
    assert kills(f) == (2, 2)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 2), st.sampled_from((CHICKEN, 1017, GUARD)),
                          st.lists(st.tuples(st.sampled_from(('agent', 'player', 'npc', 'drain', 'regen')),
                                             st.integers(0, 3)), max_size=8)), min_size=1, max_size=3),
       st.booleans())
def test_credit_follows_the_rule_for_any_fight(lives, survived):
    """Each life: an initial wound, then operations until its defeat; a defeated life respawns once."""
    f, credited, defeated_by_agent = Fight(), 0, 0
    for nid, (wound, type_, operations) in enumerate(lives):
        key = f'{nid}:0'; f.life(key, hp=3 - wound, type=type_)
        full_at_first, foreign, agent_damage, respawned, last = None, False, False, False, None
        for who, amount in operations:
            if f.hp[key] == 0:
                if respawned: break
                enemy_credit = type_ in MELEE_ENEMIES
                credited += enemy_credit and bool(full_at_first) and not foreign and last == 'agent'
                defeated_by_agent += enemy_credit and agent_damage
                respawned = True; f.respawn(key, f'{nid}:1'); key = f'{nid}:1'
                full_at_first, foreign, agent_damage = None, False, False
            if who == 'drain':
                if f.hp[key] > 1: f.change(key, f.hp[key] - 1); foreign = True
            elif who == 'regen':
                if f.hp[key] < 3: f.change(key, f.hp[key] + 1)
            else:
                if who == 'agent' and full_at_first is None:
                    full_at_first = f.hp[key] == 3 and not foreign
                foreign |= who != 'agent'
                agent_damage |= who == 'agent' and amount > 0 and f.hp[key] > 0
                last = who; f.hit(key, who, amount)
        if f.hp[key] == 0:
            enemy_credit = type_ in MELEE_ENEMIES
            credited += enemy_credit and bool(full_at_first) and not foreign and last == 'agent'
            defeated_by_agent += enemy_credit and agent_damage
    report = audit_native_combat(f.trace(death=not survived), MELEE_ENEMIES)
    assert (report['lower'], report['upper']) == (credited, defeated_by_agent)
    assert report['lower'] <= report['upper']
    result = dict(survived=survived, families={}, kills=dict(lower=report['lower'], upper=report['upper']))
    assert goal_return(result, 'melee') == (min(credited, 1) if survived else 0.)  # death settles to zero


def test_drops_pickups_and_transfers_add_no_kill():
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 1)
    t = Trace(); t.produce(slot=0)
    for key in range(1, 4):
        pickup(t, drop(t, key, slot=0), slot=0)
    t.op(inventory=[], bank=[dict(slot=0, id=1511, count=1)])
    assert kills(f, t=t) == (0, 0)


def corrupt(change):
    f = Fight(); f.life('7:0'); f.hit('7:0', 'agent', 3)
    result = f.trace(); change(result['combat']); return result


@pytest.mark.parametrize('change,message', [
    (lambda c: c.pop('events'), 'missing combat receipts'),
    (lambda c: c.update(failure='queued damage entry not found'), 'invalid or still open'),
    (lambda c: c.update(open=True), 'invalid or still open'),
    (lambda c: c.update(endTick=99), 'windows differ'),
    (lambda c: c.update(enemyTypes=[41]), 'enemies differ'),
    (lambda c: c['events'].pop(1), 'gap'),
    (lambda c: c['events'][2].update(hpAfter=1), 'damage hitpoints disagree'),
    (lambda c: c['events'][2].update(hit=99), 'queued hit'),
    (lambda c: c['events'][2].update(hitter='player'), 'queued hit'),
    (lambda c: c['events'][2].update(tick=13), 'outside the window'),
    (lambda c: c['events'][2].pop('heroes'), 'hero points'),
    (lambda c: c['events'].append(dict(sequence=4, tick=12, kind='life', life='7:1', nid=7, type=41, hp=3, baseHp=3,
                                       x=0, z=0, level=0, previous='7:0')), 'respawn without'),
    (lambda c: c['events'].append(dict(sequence=4, tick=12, kind='loot', life='7:0')), 'unsupported'),
])
def test_missing_or_inconsistent_combat_evidence_raises(change, message):
    with pytest.raises(EvidenceError, match=message):
        audit_native_combat(corrupt(change), MELEE_ENEMIES)
