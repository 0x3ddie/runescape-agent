"""Certified produced items weighted by price (baseline/native_value.py)."""
import pytest

from baseline.native_value import audit_native_value
from baseline.evidence import EvidenceError
from baseline.resource_task import ResourceClock
from baseline.tests.test_native_flow import Trace, item, obj, trace

COPPER, COPPER_NOTE = 436, 437
# Oak logs never appear: a declared family with no items must change nothing.
TARGETS = [dict(base=1511, note=1512, value=4), dict(base=COPPER, note=COPPER_NOTE, value=3),
           dict(base=1521, note=1522, value=20)]


def mine(t, slot):
    t.op(inventory=t.current['inventory'] + [item(slot, id=COPPER)], production=True)
    t.events[-1].update(scriptName='[label,get_ore_normal]', stackTail=[1, COPPER, 1])


def gathering(t, death=False):
    result = trace(t, death)
    result.update(targetScope='gathering-v1', targets=[dict(target) for target in TARGETS])
    return result


def test_each_family_is_certified_alone_and_weighted_by_its_price():
    t = Trace(); t.produce(slot=0); mine(t, slot=1); mine(t, slot=2)
    t.op(inventory=[item(1, id=COPPER), item(2, id=COPPER)], bank=[item(0)])  # the log is banked
    report = audit_native_value(gathering(t))
    assert report['retained_lower'] == report['retained_upper'] == 4 * 1 + 3 * 2
    assert report['families'] == {1511: dict(value=4, produced=1, retained_lower=1, retained_upper=1,
                                             deposit_events=1, production_after_deposit=0),
                                  COPPER: dict(value=3, produced=2, retained_lower=2, retained_upper=2,
                                               deposit_events=0, production_after_deposit=0)}
    assert report['produced'] == 3 and report['reward_eligible_lower'] == 10


def test_an_undeclared_item_or_a_ground_call_mixing_families_fails_the_audit():
    t = Trace(); t.produce(slot=0)
    t.op(inventory=t.current['inventory'] + [item(1, id=438)], opcode=4302)  # tin ore: not declared
    with pytest.raises(EvidenceError, match='undeclared'):
        audit_native_value(gathering(t))
    # A log dropped onto a tile holding an ore is fine: each family's view sees its own items.
    t = Trace(); t.produce(slot=0)
    log, ore = obj(1), obj(2, id=COPPER)
    t.op(inventory=[], opcode=4312, ground=[t.ground('addObj', dict(log, isActive=False), log, [ore], [ore, log])])
    assert audit_native_value(gathering(t))['retained_lower'] == 0  # the dropped log is not retained
    # One call tree moving objects of two families is refused rather than split.
    t = Trace(); t.produce(slot=0)
    child = dict(method='addObj', before=dict(log, isActive=False), after=log, tileBefore=[], tileAfter=[log],
                 callId=2, endId=3, rootCallId=1, parentCallId=1)
    parent = dict(method='addObj', before=dict(ore, isActive=False), after=ore, tileBefore=[], tileAfter=[ore],
                  callId=1, endId=4, rootCallId=1, parentCallId=None)
    t.op(inventory=[], opcode=4312, ground=[child, parent])
    with pytest.raises(EvidenceError, match='mixes item families'):
        audit_native_value(gathering(t))


def test_the_task_clock_requires_stable_container_identity_and_retains_death_zero():
    t = Trace(); t.produce(); t.op(inventory=[], bank=[item()])
    result = gathering(t, death=True)
    clock = ResourceClock({**result, 'closed': False}, 2)
    clock.last_status = 'death'; clock.last_tick = 10
    reward, report = clock.score(result, 10)
    assert reward == 0 and report['retained_lower'] == 4 and not report['survived']
    result['ownedContainers']['bank']['capacity'] -= 1
    with pytest.raises(EvidenceError, match='containers changed'):
        clock.score(result, 10)
