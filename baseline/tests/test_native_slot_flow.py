from copy import deepcopy

import pytest

from baseline.native_slot_flow import audit_native_slot_flow
from baseline.evidence import EvidenceError
from baseline.tests.test_native_flow import CONTAINERS, Trace, item, drop, trace


def swap(t, first=('bank', 0), second=('bank', 1)):
    snapshot = deepcopy(t.current)
    values = {}
    for location, slot in (first, second):
        values[(location, slot)] = next((r for r in snapshot[location] if r['slot'] == slot), None)
    for location, slot in {first, second}:
        snapshot[location] = [r for r in snapshot[location] if r['slot'] != slot]
    for source, target in ((first, second), (second, first)):
        if values[source] and not any(r['slot'] == target[1] for r in snapshot[target[0]]):
            snapshot[target[0]].append({**values[source], 'slot': target[1]})
    for values_ in snapshot.values():
        values_.sort(key=lambda r: r['slot'])
    t.op(**snapshot, opcode=4322)
    t.events[-1].update(stackTail=[CONTAINERS[first[0]]['id'], CONTAINERS[second[0]]['id'], first[1], second[1]],
        ownedSlotSwap=dict(schema='owned-slot-swap-v1', **{
            key: dict(location=p[0], slot=p[1], inventoryId=CONTAINERS[p[0]]['id'])
            for key, p in (('from', first), ('to', second))}))


def banked():
    t = Trace();t.produce();t.op(inventory=[], bank=[item()])
    return t


def test_bank_reorganization_preserves_production():
    t = banked();swap(t)
    new = audit_native_slot_flow(trace(t))
    assert new['produced'] == new['retained_lower'] == new['retained_upper'] == 1
    assert new['owned_slot_swap_operations'] == new['native_bank_transfer_operations'] == 1
    assert new['native_ground_transfer_operations'] == new['uncertain_operations'] == 0


@pytest.mark.parametrize('lost,expected', [(0, 1), (1, 0)])
def test_equal_looking_slots_exchange_origins_not_just_quantities(lost, expected):
    t = Trace();t.produce();t.external(slot=1)
    swap(t, ('inventory', 0), ('inventory', 1))
    drop(t, slot=lost)
    report = audit_native_slot_flow(trace(t))
    assert (report['retained_lower'], report['retained_upper']) == (expected, expected)


def test_repeated_cross_container_and_same_slot_swaps_do_not_mint_production():
    t = banked()
    for _ in range(5):
        swap(t, ('bank', 0), ('inventory', 0))
        swap(t, ('inventory', 0), ('bank', 0))
        swap(t, ('bank', 0), ('bank', 0))
    report = audit_native_slot_flow(trace(t))
    assert report['owned_slot_swap_operations'] == 15
    assert report['produced'] == report['retained_lower'] == report['retained_upper'] == 1


def test_external_or_unauthenticated_slot_write_stays_conservative():
    t = banked();swap(t)
    t.events[-1]['ownedSlotSwap'] = None
    report = audit_native_slot_flow(trace(t))
    assert (report['retained_lower'], report['retained_upper']) == (0, 1)
    assert report['owned_slot_swap_operations'] == 0 and report['uncertain_operations'] == 1


def test_genuine_mixed_origin_loss_stays_ambiguous_after_reorganization():
    t = Trace();t.produce();t.external(slot=1)
    t.op(inventory=[], bank=[item(count=2)])
    swap(t)
    t.op(inventory=[item()], bank=[item(slot=1)])
    drop(t)
    report = audit_native_slot_flow(trace(t))
    assert (report['retained_lower'], report['retained_upper']) == (0, 1)


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop('slotSwapEvidenceVersion'),
    lambda r: r.update(slotSwapEvidenceVersion=True),
    lambda r: r['ownedContainers']['bank'].update(id=93),
    lambda r: r['ownedContainers']['bank'].update(capacity=0),
    lambda r: r['events'][0].pop('ownedSlotSwap'),
    lambda r: r['events'][2].update(opcode=4323),
    lambda r: r['events'][2]['stackTail'].__setitem__(0, 96),
    lambda r: r['events'][2]['stackTail'].__setitem__(2, False),
    lambda r: r['events'][2]['ownedSlotSwap']['from'].update(location='shop'),
    lambda r: r['events'][2]['ownedSlotSwap']['to'].update(slot=400),
    lambda r: r['events'][2]['ownedSlotSwap']['to'].update(inventoryId=96),
    lambda r: r['events'][2].update(resourceMutation=False),
    lambda r: r['events'][2].update(ground=[{}]),
])
def test_missing_or_forged_swap_evidence_is_rejected(mutation):
    t = banked();swap(t);result = trace(t);mutation(result)
    with pytest.raises(EvidenceError):
        audit_native_slot_flow(result)


def test_conserved_totals_alone_do_not_validate_a_claimed_exchange():
    t = banked();swap(t)
    t.events[-1]['ownedSlotSwap']['to']['slot'] = 2
    t.events[-1]['stackTail'][-1] = 2
    with pytest.raises(EvidenceError, match='exact exchange'):
        audit_native_slot_flow(trace(t))

