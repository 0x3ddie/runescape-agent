from copy import deepcopy
from itertools import product

import pytest

from baseline.native_slot_flow import audit_native_slot_flow
from baseline.evidence import EvidenceError


CONTAINERS = dict(inventory=dict(id=93, capacity=28), bank=dict(id=95, capacity=400))


def item(slot=0, count=1, id=1511):
    return dict(slot=slot, id=id, count=count)


def obj(key=1, count=1, id=1511, x=10, active=True):
    return dict(captureObjectId=key, id=id, count=count, level=0, x=x, z=20,
                lifecycle=1, isActive=active, receiver64='42')


class Trace:
    def __init__(self):
        self.initial = dict(inventory=[], bank=[])
        self.current = deepcopy(self.initial)
        self.events = []
        self.order = 0

    def ground(self, method, before, after, tiles_before, tiles_after):
        self.order += 2
        return dict(method=method, before=before, after=after,
                    tileBefore=tiles_before, tileAfter=tiles_after,
                    callId=self.order-1, endId=self.order, rootCallId=self.order-1, parentCallId=None)

    def op(self, inventory=None, bank=None, opcode=4320, ground=None, production=False):
        after = deepcopy(self.current)
        if inventory is not None:
            after['inventory'] = inventory
        if bank is not None:
            after['bank'] = bank
        self.events.append(dict(kind='native-op', tick=10, before=deepcopy(self.current), after=after,
                                opcode=4302 if production else opcode, productionSource=production,
                                scriptId=51, scriptName='[label,get_logs]' if production else '[proc,test]',
                                pc=5, stackTail=[1, 1511, 1], ground=ground or []))
        self.events[-1]['resourceMutation'] = True
        self.current = deepcopy(after)

    def produce(self, slot=0):
        self.op(inventory=self.current['inventory'] + [item(slot)], production=True)

    def external(self, slot=1, key=100, count=1):
        ground = obj(key, count, x=30)
        event = self.ground('removeObj', ground, dict(ground, isActive=False), [ground], [])
        self.op(inventory=self.current['inventory'] + [item(slot, count)], opcode=3510, ground=[event])

    def finish(self, death=False):
        rows = deepcopy(self.events)
        tick = 10 if death else 12
        if not death:
            for boundary in (11, 12):
                rows.append(dict(kind='tick-boundary', tick=boundary, holdings=self.current))
        terminal = dict(reason='death' if death else 'horizon', tick=tick, holdings=self.current,
                        toolAvailable=True, lifecycle=dict(phase='dying' if death else 'alive', failure=None,
                            deathSequence=int(death), respawnSequence=0, deathTick=10 if death else None))
        rows.append(dict(kind='terminal', **terminal))
        for sequence, row in enumerate(rows, 1):
            row['sequence'] = sequence
        return dict(schema='native-resource-capture-v2', flowEvidenceVersion=1, identity=dict(worldBootId='boot', accountId='a',
                    accountEpoch=1, incarnationId='p'), initial=self.initial, last=self.current,
                    startTick=10, endTick=12, observedTick=tick, events=rows, terminal=terminal,
                    failure=None, closed=True, completed=not death, evidenceComplete=True, lifecycleObserved=True)


def trace(t, death=False):
    """The capture with owned-slot evidence, as the slot-flow audit requires."""
    result = t.finish(death=death)
    result.update(slotSwapEvidenceVersion=1, ownedContainers=deepcopy(CONTAINERS))
    for event in result['events']:
        if event['kind'] == 'native-op':
            event.setdefault('ownedSlotSwap', None)
    return result


def drop(t, key=1, slot=0, tile=None, id=1511):
    g = obj(key, id=id)
    before = tile or []
    event = t.ground('addObj', dict(g, isActive=False), g, before, before + [g])
    t.op(inventory=[i for i in t.current['inventory'] if i['slot'] != slot], opcode=4312, ground=[event])
    return g


def pickup(t, g, slot=0, tile=None):
    before = tile or [g]
    event = t.ground('removeObj', g, dict(g, isActive=False), before,
                     [i for i in before if i['captureObjectId'] != g['captureObjectId']])
    t.op(inventory=t.current['inventory'] + [item(slot, g['count'], g['id'])], opcode=3510, ground=[event])


def bounds(t):
    report = audit_native_slot_flow(trace(t))
    return report['retained_lower'], report['retained_upper']


def test_repeated_drop_recovery_never_mints_production():
    t = Trace()
    t.produce()
    for key in range(1, 11):
        pickup(t, drop(t, key))
    report = audit_native_slot_flow(trace(t))
    assert report['produced'] == report['retained_lower'] == report['retained_upper'] == 1
    assert report['possible_external_inflows'] == 10  # the aggregate audit's diagnostic, not the flow attribution


def test_external_pickup_and_own_recovery_remain_distinct():
    t = Trace()
    t.produce()
    g = drop(t)
    t.external(slot=1)
    pickup(t, g)
    drop(t, key=2, slot=1)
    assert bounds(t) == (1, 1)


def test_leaving_own_log_on_ground_loses_credit_even_with_external_replacement():
    t = Trace()
    t.produce()
    drop(t)
    t.external(slot=0)
    assert bounds(t) == (0, 0)


def test_expiry_and_new_object_same_tile_do_not_restore_origin():
    t = Trace()
    t.produce()
    g = drop(t)
    event = t.ground('removeObj', g, dict(g, isActive=False), [g], [])
    t.events.append(dict(kind='native-ground', tick=10, **event))
    replacement = obj(2)
    event = t.ground('addObj', dict(replacement, isActive=False), replacement, [], [replacement])
    t.events.append(dict(kind='native-ground', tick=10, **event))
    pickup(t, replacement)
    assert bounds(t) == (0, 0)


@pytest.mark.parametrize('produced,external,removed', [
    (p, e, r) for p, e in product(range(1, 4), range(1, 4)) for r in range(p + e + 1)
])
def test_mixed_stack_loss_bounds_match_exhaustive_origin_allocations(produced, external, removed):
    t = Trace()
    for i in range(produced):
        t.produce(i)
    t.external(slot=produced, count=external)
    t.op(inventory=[], bank=[item(count=produced+external)])
    # A known bank withdrawal to distinct ordinary slots, then a known drop.
    remaining = produced + external - removed
    t.op(inventory=[item(i) for i in range(removed)], bank=[item(count=remaining)] if remaining else [])
    tile = []
    for slot in range(removed):
        tile.append(drop(t, key=slot+1, slot=slot, tile=list(tile)))
    possibilities = {produced - lost_own for lost_own in range(produced + 1)
                     if 0 <= removed - lost_own <= external}
    assert bounds(t) == (min(possibilities), max(possibilities))
    # Reuniting all ambiguous pieces restores an exact total. Independent
    # per-container intervals would incorrectly lose this correlation.
    for slot, g in enumerate(tile):
        pickup(t, g, slot=slot, tile=tile[slot:])
    t.op(inventory=[], bank=[item(count=produced+external)])
    assert bounds(t) == (produced, produced)


def test_nested_note_merge_is_counted_once_and_can_be_recovered():
    t = Trace()
    t.produce()
    t.op(inventory=[], bank=[item()])
    t.op(inventory=[item(id=1512)], bank=[], opcode=4319)
    existing, incoming = obj(100, count=2, id=1512), obj(1, id=1512, active=False)
    merged = dict(existing, count=3)
    child = dict(method='changeObj', before=existing, after=merged, tileBefore=[existing], tileAfter=[merged],
                 callId=2, endId=3, rootCallId=1, parentCallId=1)
    parent = dict(method='addObj', before=incoming, after=incoming, tileBefore=[existing], tileAfter=[merged],
                  callId=1, endId=4, rootCallId=1, parentCallId=None)
    t.order = 4
    t.op(inventory=[], opcode=4312, ground=[child, parent])
    pickup(t, merged)
    assert bounds(t) == (1, 1)


def test_nested_cross_tile_capacity_eviction_loses_the_evicted_origin():
    t = Trace()
    t.produce()
    old = drop(t)
    t.produce()
    new = obj(2, x=11)
    child = dict(method='removeObj', before=old, after=dict(old, isActive=False), tileBefore=[old], tileAfter=[],
                 callId=4, endId=5, rootCallId=3, parentCallId=3)
    parent = dict(method='addObj', before=dict(new, isActive=False), after=new, tileBefore=[], tileAfter=[new],
                  callId=3, endId=6, rootCallId=3, parentCallId=None)
    t.order = 6
    t.op(inventory=[], opcode=4312, ground=[child, parent])
    pickup(t, new)
    assert bounds(t) == (1, 1)


def test_unknown_equal_count_setter_does_not_certify_old_origin():
    t = Trace()
    t.produce()
    t.op(opcode=4323)
    assert bounds(t) == (0, 1)


def test_unknown_increase_can_replace_old_units_too():
    t = Trace()
    t.produce()
    t.op(inventory=[item(count=2)], opcode=4323)
    assert bounds(t) == (0, 1)


def test_death_after_recovery_still_zeroes_reward():
    t = Trace()
    t.produce()
    pickup(t, drop(t))
    r = audit_native_slot_flow(trace(t, death=True))
    assert r['retained_lower'] == 1 and r['reward_eligible_lower'] == 0


def test_external_native_add_preserves_existing_production_but_creates_no_credit():
    t = Trace()
    t.produce()
    t.op(inventory=[item(), item(1)], opcode=4302)
    assert bounds(t) == (1, 1)
    # Known deletion of the external slot leaves the produced slot intact.
    t.op(inventory=[item()], opcode=4308)
    assert bounds(t) == (1, 1)


def test_reactivated_object_identity_does_not_resurrect_old_production():
    t = Trace()
    t.produce()
    g = drop(t)
    expired = dict(g, isActive=False)
    event = t.ground('removeObj', g, expired, [g], [expired])
    t.events.append(dict(kind='native-ground', tick=10, **event))
    event = t.ground('addObj', expired, g, [expired], [g])
    t.events.append(dict(kind='native-ground', tick=10, **event))
    pickup(t, g)
    assert bounds(t) == (0, 0)


def test_a_trace_without_write_and_nested_call_coverage_is_refused():
    t = Trace()
    t.produce()
    result = trace(t)
    del result['flowEvidenceVersion']
    with pytest.raises(EvidenceError, match='coverage'):
        audit_native_slot_flow(result)


@pytest.mark.parametrize('fault', ['missing_order', 'duplicate', 'parent', 'root', 'state_gap',
                                  'moved', 'count', 'conservation', 'missing_call', 'incomplete'])
def test_reject_invalid_ground_evidence(fault):
    t = Trace()
    t.produce()
    pickup(t, drop(t))
    result = trace(t)
    ground = result['events'][2]['ground'][0]
    if fault == 'missing_order':
        del ground['callId']
    if fault == 'duplicate':
        ground['callId'] = ground['endId']
    if fault == 'parent':
        ground['parentCallId'] = 1
    if fault == 'root':
        ground['rootCallId'] = 1
    if fault == 'state_gap':
        ground['tileBefore'] = []
    if fault == 'moved':
        ground['before']['x'] = 11
    if fault == 'count':
        ground['tileBefore'][0]['count'] = -1
    if fault == 'conservation':
        ground['tileAfter'] = ground['tileBefore']
    if fault == 'missing_call':
        result['events'][2]['ground'] = []
    if fault == 'incomplete':
        result['evidenceComplete'] = False
    with pytest.raises(EvidenceError):
        audit_native_slot_flow(result)
