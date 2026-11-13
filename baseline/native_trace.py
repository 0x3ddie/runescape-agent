"""Independent v2 raw-trace audit with conservative aggregate provenance bounds.

No invented per-log identity. Non-production inflows are counted as potentially
external, including reacquisition until a ground-lineage solver establishes its
origin. Bounds are deliberately conservative; ambiguity is not an exact score.
"""
from collections import Counter

from baseline.evidence import EvidenceError, integer

# A certified item family: the produced item, its noted form, and the scripts that make it.
# Logs are the default; baseline/native_value.py audits other families one at a time.
LOGS = dict(base=1511, note=1512, scripts=('[label,get_logs]',))


def family_ids(family):
    return tuple(i for i in (family['base'], family['note']) if i is not None)


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def holdings(snapshot, ids=family_ids(LOGS)):
    require(isinstance(snapshot, dict) and set(snapshot) == {'inventory', 'bank'}, 'invalid holdings fields')
    result = Counter()
    for location in ('inventory', 'bank'):
        require(isinstance(snapshot[location], list), 'missing resource slots')
        seen = set()
        for item in snapshot[location]:
            require(isinstance(item, dict) and set(item) == {'slot', 'id', 'count'}, 'invalid resource slot')
            slot = integer(item['slot'], 'slot')
            integer(item['id'], 'item id')
            require(slot not in seen and item['id'] in ids, 'duplicate slot or unsupported resource')
            seen.add(slot)
            result[location] += integer(item['count'], 'resource count', 1)
    return result


def audit_native_trace(trace, family=LOGS):
    ids = family_ids(family)
    require(isinstance(trace, dict) and trace.get('schema') == 'native-resource-capture-v2', 'wrong capture schema')
    require(trace.get('failure') is None and trace.get('closed') is True and trace.get('evidenceComplete') is True,
            'invalid or incomplete capture')
    require(trace.get('lifecycleObserved') is True, 'missing native lifecycle observer')
    identity = trace.get('identity')
    require(isinstance(identity, dict) and set(identity) == {'worldBootId', 'accountId', 'accountEpoch', 'incarnationId'},
            'missing native generation identity')
    for key in ('worldBootId', 'accountId', 'incarnationId'):
        require(isinstance(identity[key], str) and bool(identity[key]), 'invalid native identity')
    integer(identity['accountEpoch'], 'account epoch', 1)
    start, end = integer(trace.get('startTick'), 'start tick'), integer(trace.get('endTick'), 'end tick')
    require(end > start, 'invalid horizon')
    previous = trace.get('initial')
    require(sum(holdings(previous, ids).values()) == 0, 'preloaded logs or notes')
    events = trace.get('events')
    require(isinstance(events, list) and events, 'missing raw receipts')
    last_tick, boundary, produced, possible_external = start, start, 0, 0
    deposits = post_deposit_production = 0
    for sequence, event in enumerate(events, 1):
        require(type(event.get('sequence')) is int and event['sequence'] == sequence, 'receipt gap/duplicate')
        tick = integer(event.get('tick'), 'event tick')
        require(last_tick <= tick <= end and tick >= start, 'clock outside ordered horizon')
        last_tick = tick
        kind = event.get('kind')
        if kind == 'native-op':
            require(tick == boundary and event.get('before') == previous, 'unobserved opcode interval or state gap')
            before, after = holdings(previous, ids), holdings(event.get('after'), ids)
            delta = sum(after.values()) - sum(before.values())
            require(type(event.get('productionSource')) is bool, 'missing source classification')
            integer(event.get('opcode'), 'opcode'); integer(event.get('scriptId'), 'script id'); integer(event.get('pc'), 'pc')
            require(isinstance(event.get('scriptName'), str) and isinstance(event.get('ground'), list), 'missing native source')
            if event['productionSource']:
                tail=event.get('stackTail')
                before_ids=Counter({})
                after_ids=Counter({})
                for item in previous['inventory']:before_ids[item['id']]+=item['count']
                for item in event['after']['inventory']:after_ids[item['id']]+=item['count']
                before_ids[family['base']]+=1
                require(event['opcode']==4302 and isinstance(tail,list) and tail[-2:]==[family['base'],1] and
                        event['scriptName'] in family['scripts'] and delta == 1 and
                        after['inventory'] - before['inventory'] == 1 and after['bank'] == before['bank'] and
                        event['after']['bank']==previous['bank'] and not event['ground'] and after_ids==before_ids,
                        'production source has unsupported effects')
                produced += 1
                post_deposit_production += deposits > 0
            elif delta > 0:
                possible_external += delta
            if after['bank'] > before['bank'] and before['inventory'] - after['inventory'] == after['bank'] - before['bank']:
                deposits += 1
            previous = event['after']
        elif kind == 'native-ground':
            require(tick == boundary and event.get('method') in ('addObj', 'removeObj', 'changeObj'), 'invalid ground interval')
            require(isinstance(event.get('tileBefore'), list) and isinstance(event.get('tileAfter'), list), 'ground snapshots missing')
        elif kind == 'tick-boundary':
            require(tick == boundary + 1 and event.get('holdings') == previous, 'missing/repeated native boundary')
            holdings(event['holdings'], ids)
            boundary = tick
        elif kind == 'terminal':
            require(sequence == len(events) and event.get('holdings') == previous, 'terminal tail/state mismatch')
        else:
            raise EvidenceError('unsupported raw receipt kind')
    terminal = trace.get('terminal')
    require(isinstance(terminal, dict) and events[-1]['kind'] == 'terminal' and
            all(events[-1].get(key) == value for key, value in terminal.items()) and
            trace.get('last') == previous, 'terminal summary mismatch')
    require(trace.get('observedTick')==last_tick,'final observed clock mismatch')
    require(type(terminal.get('toolAvailable')) is bool, 'missing tool observation')
    life = terminal.get('lifecycle')
    require(isinstance(life, dict) and life.get('failure') is None, 'invalid lifecycle evidence')
    reason = terminal.get('reason')
    if reason == 'horizon':
        require(last_tick == boundary == end and trace.get('completed') is True and life.get('phase') == 'alive'
                and life.get('deathSequence') == life.get('respawnSequence') == 0, 'incomplete surviving horizon')
    elif reason == 'death':
        require(last_tick == boundary < end and trace.get('completed') is False and life.get('phase') == 'dying'
                and life.get('deathSequence') == 1 and life.get('respawnSequence') == 0 and life.get('deathTick') == last_tick,
                'invalid first-life death closure')
    else:
        raise EvidenceError('unsupported terminal reason')
    total = sum(holdings(previous, ids).values())
    lower, upper = max(0, total - possible_external), min(total, produced)
    require(lower <= upper, 'inconsistent provenance totals')
    return {'schema': 'resource-native-bounds-v1', 'identity': identity, 'produced': produced,
            'total_holdings': total, 'retained_lower': lower, 'retained_upper': upper,
            'provenance_exact': lower == upper, 'possible_external_inflows': possible_external,
            'deposit_events': deposits, 'production_after_deposit': post_deposit_production,
            'survived': reason == 'horizon', 'tool_available': terminal['toolAvailable'],
            'delivered_ticks': last_tick-start, 'scheduled_ticks': end-start,
            'reward_eligible_lower': lower if reason == 'horizon' else 0,
            'limitation': 'Ground provenance is conservative, not reconstructed; source authentication requires the native code/cache manifest.'}
