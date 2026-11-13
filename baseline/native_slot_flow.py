"""Provenance flow over the owned inventory and bank slots, with versioned slot swaps."""
from baseline.native_flow import Provenance, slots
from baseline.native_trace import LOGS, audit_native_trace, family_ids, require
from baseline.evidence import integer


def owned_containers(trace):
    require(type(trace.get('slotSwapEvidenceVersion')) is int and trace['slotSwapEvidenceVersion'] == 1,
            'missing owned-slot evidence version')
    containers = trace.get('ownedContainers')
    require(isinstance(containers, dict) and set(containers) == {'inventory', 'bank'},
            'missing owned-container identities')
    for row in containers.values():
        require(isinstance(row, dict) and set(row) == {'id', 'capacity'}, 'invalid owned container')
        integer(row['id'], 'container id')
        integer(row['capacity'], 'container capacity', 1)
    require(containers['inventory']['id'] != containers['bank']['id'], 'aliased owned containers')
    return containers


class SlotProvenance(Provenance):
    def __init__(self, containers, ids=family_ids(LOGS)):
        super().__init__(ids)
        self.containers = containers
        self.slot_swaps = self.bank_transfers = self.ground_transfers = 0

    def op(self, event):
        require('ownedSlotSwap' in event, 'missing slot operation coverage')
        swap = event['ownedSlotSwap']
        if swap is None:
            previous = self.transfers
            super().op(event)
            if self.transfers > previous:
                if event['opcode'] in (4319, 4320, 4321):
                    self.bank_transfers += 1
                else:
                    self.ground_transfers += 1
            return
        require(isinstance(swap, dict) and set(swap) == {'schema', 'from', 'to'} and
                swap['schema'] == 'owned-slot-swap-v1', 'invalid slot swap receipt')
        require(event['opcode'] == 4322 and event['productionSource'] is False and
                event['resourceMutation'] is True and not event['ground'], 'unsupported slot swap effects')
        endpoints = []
        for key in ('from', 'to'):
            row = swap[key]
            require(isinstance(row, dict) and set(row) == {'location', 'inventoryId', 'slot'},
                    'invalid slot endpoint')
            location = row['location']
            require(isinstance(location, str) and location in self.containers, 'external slot endpoint')
            inventory_id = integer(row['inventoryId'], 'slot inventory id')
            slot = integer(row['slot'], 'slot index')
            require(inventory_id == self.containers[location]['id'] and
                    slot < self.containers[location]['capacity'], 'slot identity or bounds mismatch')
            endpoints.append((location, slot, inventory_id))
        first, second = endpoints
        tail = event.get('stackTail')
        require(isinstance(tail, list) and len(tail) >= 4 and all(type(x) is int for x in tail[-4:]) and
                tail[-4:] == [first[2], second[2], first[1], second[1]], 'slot operands disagree')
        before, after = slots(event['before']), slots(event['after'])
        require({k: n for k, (_, n) in self.account.items()} == before, 'account flow state gap')
        def moved(key):
            location, slot, item = key
            if (location, slot) == first[:2]:
                return (*second[:2], item)
            if (location, slot) == second[:2]:
                return (*first[:2], item)
            return key
        require({moved(k): n for k, n in before.items()} == after, 'slot receipt is not an exact exchange')
        # Move provenance nodes themselves. Equal-count/equal-item swaps can still
        # exchange different origins; quantity-delta allocation would lose that fact.
        self.account = {moved(k): value for k, value in self.account.items()}
        self.slot_swaps += 1


def audit_native_slot_flow(trace, family=LOGS):
    base = audit_native_trace(trace, family)
    require(type(trace.get('flowEvidenceVersion')) is int and trace['flowEvidenceVersion'] == 1,
            'missing native flow evidence coverage')
    provenance = SlotProvenance(owned_containers(trace), family_ids(family))
    pending = []
    for event in trace['events']:
        if event['kind'] == 'native-ground':
            pending.append(event)
            if event.get('callId') == event.get('rootCallId'):
                removed, added = provenance.ground_changes(pending)
                provenance.allocate(removed, [], uncertain=True)
                provenance.allocate([], added)
                pending = []
        else:
            require(not pending, 'incomplete standalone ground call')
            if event['kind'] == 'native-op':
                provenance.op(event)
    require(not pending and provenance.flow.produced == base['produced'], 'incomplete production flow')
    lower, upper = provenance.finish()
    require(0 <= lower <= upper <= min(base['produced'], base['total_holdings']), 'invalid flow bounds')
    return {**base, 'schema': 'resource-native-slot-flow-v1', 'retained_lower': lower, 'retained_upper': upper,
            'provenance_exact': lower == upper, 'reward_eligible_lower': lower if base['survived'] else 0,
            'owned_slot_swap_operations': provenance.slot_swaps,
            'native_bank_transfer_operations': provenance.bank_transfers,
            'native_ground_transfer_operations': provenance.ground_transfers,
            'uncertain_operations': provenance.uncertain_ops,
            'limitation': 'Integral flow bounds with authenticated owned-slot exchanges; native source authentication remains required.'}
