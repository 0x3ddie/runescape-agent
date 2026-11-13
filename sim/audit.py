"""An independent account of a simulated episode's goal outcomes, from the world's events.

The payment in sim/task.py follows held items or kills as they happen; this replays the evidence
instead: the holdings at arm, every production, pickup, transfer and removal (eating, burying,
dropping) event, and the holdings at close. Each location's totals must be explained exactly by
the events, transfers must conserve items, and only the gathering scripts may produce
(baseline.native_value.GATHERING_SCRIPTS). The return counts goal items held at close that
production accounts for, and nothing after a death, as baseline.resource_task.goal_return does
for native reports.

Kills (goal-activity-v2 melee, sustained-combat-v1) are counted from the hits npcs take: one credit per enemy life of
an eligible type (baseline.resource_task.MELEE_ENEMIES, or the task's `enemies`), when this agent's hit was the first that
life took in the episode and found it at full hitpoints, every hit on it was this agent's, and this
agent's hit defeated it (hitpoints 0) before the close.
"""
from collections import Counter

from baseline.native_value import GATHERING_SCRIPTS
from baseline.resource_task import GOAL_FAMILIES, MELEE_ENEMIES


class AuditError(Exception):
    pass


def _totals(holdings):
    totals = {loc: Counter() for loc in ('inventory', 'bank', 'worn')}
    for loc, rows in holdings.items():
        for _, obj, count in rows:
            totals[loc][obj] += count
    return totals


def kill_ticks(events, agent, enemies=MELEE_ENEMIES):
    """The ticks of the certified kills by `agent` among (tick, kind, data) events (module docstring): each
    credited life's defeat, in order."""
    return sorted(kill_lives(events, agent, enemies).values())


def kill_lives(events, agent, enemies=MELEE_ENEMIES):
    """{(npc, life): defeat tick} of the certified kills by `agent` (module docstring)."""
    lives = {}
    for tick, kind, data in events:
        if kind not in ('npc_damage', 'npc_defeated'):
            continue
        key = (data['npc'], data['life'])
        if kind == 'npc_damage':
            life = lives.setdefault(key, dict(type=data['type'], fresh=data['source'] == agent and data['hp'] == data['full'],
                                              mine=True, defeated=None))
            life['mine'] &= data['source'] == agent
        else:
            if key not in lives:
                raise AuditError('an npc defeated without a hit')
            lives[key]['defeated'] = tick if data['source'] == agent else None
    return {key: life['defeated'] for key, life in lives.items()
            if life['type'] in enemies and life['fresh'] and life['mine'] and life['defeated'] is not None}


def kills(events, agent, enemies=MELEE_ENEMIES):
    """Certified kills by `agent` among (tick, kind, data) events (module docstring)."""
    return len(kill_ticks(events, agent, enemies))


def audit(events, initial, terminal, goal, tables, agent=None, enemies=MELEE_ENEMIES):
    family = set(GOAL_FAMILIES.get(goal, ()))
    for base in list(family):
        note = tables.obj_types.get(base, {}).get('certlink', -1)
        if note is not None and note >= 0:
            family.add(note)
    held, produced, external = _totals(initial), Counter(), Counter()
    names = {'inv': 'inventory', 'bank': 'bank', 'worn': 'worn'}
    window = []
    for tick, kind, data in events:
        if tick >= terminal['tick']:
            break
        window.append((tick, kind, data))
        if kind == 'produce':
            if data['script'] not in GATHERING_SCRIPTS:
                raise AuditError(f"production by {data['script']}")
            held['inventory'][data['obj']] += data['count']
            produced[data['obj']] += data['count']
        elif kind == 'pickup':  # a ground item: held, but not produced
            held['inventory'][data['obj']] += data['count']
            external[data['obj']] += data['count']
        elif kind == 'transfer':
            src, dst = names[data['src']], names[data['dst']]
            if held[src][data['obj']] < data['count']:
                raise AuditError('transfer of items not held')
            held[src][data['obj']] -= data['count']
            held[dst][data['to']] += data['count']
        elif kind == 'remove':  # eaten, buried or dropped from the inventory
            if held['inventory'][data['obj']] < data['count']:
                raise AuditError('removal of items not held')
            held['inventory'][data['obj']] -= data['count']
        elif kind not in ('npc_damage', 'npc_defeated', 'agent_hp', 'excluded'):
            raise AuditError(f'unaccounted event {kind}')
    final = _totals(terminal['holdings'])
    for loc in held:
        if +held[loc] != +final[loc]:
            raise AuditError(f'{loc} holdings not explained by the events: {dict(+held[loc])} != {dict(+final[loc])}')
    kept = sum(final[loc][obj] for loc in ('inventory', 'bank') for obj in family)
    made = sum(produced[obj] for obj in family)
    picked = sum(external[obj] for obj in family)
    # as native_trace's bounds: picked-up goal items may be among those kept
    lower, upper = max(0, kept - picked), min(kept, made)
    if lower > upper:
        raise AuditError('goal items held that production does not account for')
    killed = kills(window, agent, enemies)
    return dict(survived=terminal['reason'] != 'death', reason=terminal['reason'], produced=made, retained=lower,
                retained_upper=upper, picked=picked, produced_items=dict(produced), tick=terminal['tick'],
                kills=dict(lower=killed, upper=killed))
