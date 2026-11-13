"""Value of certified produced items (scope gathering-v1).

The engine declares the item families its scope certifies: each produced item, its noted
form and the game's price (env-ts/engine/resource-capture-native.ts). The slot-flow audit
checks each family alone, on a view of the trace that holds only that family's items, so
its conservative bounds and checks are unchanged; the reward is the price-weighted sum of
their certified retained counts. Production of one family is an ordinary event in another
family's view, with no effect there.
"""
from copy import deepcopy

from baseline.native_slot_flow import audit_native_slot_flow
from baseline.native_trace import family_ids, require
from baseline.evidence import integer

GATHERING = 'gathering-v1'
# The scripts that produce gathered items (GATHERING_SCRIPTS in the engine adapter).
GATHERING_SCRIPTS = ('[label,get_logs]', '[label,get_ore_normal]', '[label,get_ore_fast]', '[label,get_ore_essence]',
                     '[label,get_ore_gem_rock]', '[proc,fish_roll]', '[proc,fish_roll_loc]')


def declared_families(trace):
    require(trace.get('targetScope') == GATHERING, 'not a gathering-v1 trace')
    targets = trace.get('targets')
    require(isinstance(targets, list) and targets, 'missing target families')
    families, seen = [], set()
    for target in targets:
        require(isinstance(target, dict) and set(target) == {'base', 'note', 'value'}, 'invalid target family')
        note = None if target['note'] is None else integer(target['note'], 'noted item')
        family = dict(base=integer(target['base'], 'produced item'), note=note,
                      value=integer(target['value'], 'item price'), scripts=GATHERING_SCRIPTS)
        ids = set(family_ids(family))
        require(not ids & seen, 'item declared in two families')
        seen |= ids
        families.append(family)
    return families, seen


def family_view(trace, family, declared):
    """The trace as if only this family's items existed."""
    ids = set(family_ids(family))

    def holdings(snapshot):
        require(isinstance(snapshot, dict) and set(snapshot) == {'inventory', 'bank'}, 'invalid holdings fields')
        result = {}
        for location in ('inventory', 'bank'):
            rows = snapshot[location]
            require(isinstance(rows, list) and all(isinstance(r, dict) and r.get('id') in declared for r in rows),
                    'undeclared target item')
            result[location] = [r for r in rows if r['id'] in ids]
        return result

    def ground_calls(calls):
        """Keep a call tree wholly in this family, drop one wholly outside it, refuse a mix."""
        objects = {row.get('id') for call in calls for row in (call.get('before'), call.get('after')) if isinstance(row, dict)}
        require(objects <= declared, 'undeclared ground item')
        if not objects & ids:
            return []
        require(objects <= ids, 'ground call mixes item families')
        for call in calls:
            for key in ('tileBefore', 'tileAfter'):
                require(isinstance(call.get(key), list) and all(r.get('id') in declared for r in call[key]),
                        'undeclared ground item')
                call[key] = [r for r in call[key] if r['id'] in ids]
        return calls

    view = deepcopy(trace)
    for key in ('initial', 'last'):
        view[key] = holdings(view[key])
    require(isinstance(view.get('terminal'), dict), 'missing terminal')
    view['terminal']['holdings'] = holdings(view['terminal'].get('holdings'))
    events, pending = [], []
    for event in view['events']:
        kind = event.get('kind')
        if kind == 'native-ground':
            pending.append(event)
            if event.get('callId') == event.get('rootCallId'):
                events.extend(ground_calls(pending))
                pending = []
            continue
        require(not pending, 'incomplete standalone ground call')
        if kind == 'native-op':
            event['before'], event['after'] = holdings(event.get('before')), holdings(event.get('after'))
            require(isinstance(event.get('ground'), list), 'missing native source')
            event['ground'] = ground_calls(event['ground'])
            tail = event.get('stackTail')
            if event.get('productionSource') is True:
                event['productionSource'] = isinstance(tail, list) and len(tail) >= 2 and tail[-2] == family['base']
        elif kind in ('tick-boundary', 'terminal'):
            event['holdings'] = holdings(event.get('holdings'))
        events.append(event)
    require(not pending, 'incomplete standalone ground call')
    for sequence, event in enumerate(events, 1):
        event['sequence'] = sequence
    view['events'] = events
    return view


def item_ids(node, found=None):
    """Every integer `id` anywhere in the trace: a superset of the items it holds or moves."""
    found = set() if found is None else found
    if isinstance(node, dict):
        if type(node.get('id')) is int:
            found.add(node['id'])
        for value in node.values():
            item_ids(value, found)
    elif isinstance(node, list):
        for value in node:
            item_ids(value, found)
    return found


def audit_native_value(trace):
    families, declared = declared_families(trace)
    # A family none of whose items appear produced and kept nothing. Auditing all 46 took
    # about 5 s per episode end, long enough for the server to log an idle agent out.
    present = item_ids(trace)
    families = [f for f in families if present & set(family_ids(f))] or families[:1]
    reports = [audit_native_slot_flow(family_view(trace, family, declared), family) for family in families]
    survived = reports[0]['survived']
    lower = sum(f['value'] * r['retained_lower'] for f, r in zip(families, reports))
    upper = sum(f['value'] * r['retained_upper'] for f, r in zip(families, reports))
    return {'schema': 'resource-native-value-v1', 'identity': reports[0]['identity'], 'survived': survived,
            'tool_available': reports[0]['tool_available'], 'delivered_ticks': reports[0]['delivered_ticks'],
            'scheduled_ticks': reports[0]['scheduled_ticks'], 'retained_lower': lower, 'retained_upper': upper,
            'provenance_exact': lower == upper, 'reward_eligible_lower': lower if survived else 0,
            'produced': sum(r['produced'] for r in reports),
            'deposit_events': sum(r['deposit_events'] for r in reports),
            'production_after_deposit': sum(r['production_after_deposit'] for r in reports),
            'families': {f['base']: dict(value=f['value'], produced=r['produced'], retained_lower=r['retained_lower'],
                                         retained_upper=r['retained_upper'], deposit_events=r['deposit_events'],
                                         production_after_deposit=r['production_after_deposit'])
                         for f, r in zip(families, reports) if r['produced'] or r['total_holdings']},
            'limitation': 'Per-family integral flow bounds weighted by the game price; native source authentication '
                          'remains required.'}
