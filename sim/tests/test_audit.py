from types import SimpleNamespace

from hypothesis import given, strategies as st
import pytest

from baseline.native_value import GATHERING_SCRIPTS
from sim.audit import AuditError, audit, kills

LOGS, OTHER = 1511, 1925
TABLES = SimpleNamespace(obj_types={})
SCRIPT = sorted(GATHERING_SCRIPTS)[0]
events = st.lists(st.tuples(st.sampled_from(['produce', 'pickup', 'deposit', 'withdraw', 'remove']),
                            st.sampled_from([LOGS, OTHER]), st.integers(1, 3)), max_size=30)


def play(steps):
    """Apply steps to empty holdings, keeping only the moves a player could make."""
    held = dict(inventory={}, bank={})
    log = []
    for tick, (kind, obj, count) in enumerate(steps):
        if kind in ('produce', 'pickup'):
            held['inventory'][obj] = held['inventory'].get(obj, 0) + count
            log.append((tick, kind, dict(obj=obj, count=count, script=SCRIPT)))
        elif kind == 'remove':  # eaten, buried or dropped
            count = min(count, held['inventory'].get(obj, 0))
            if count:
                held['inventory'][obj] -= count
                log.append((tick, 'remove', dict(obj=obj, count=count, why='drop')))
        else:
            src, dst = ('inventory', 'bank') if kind == 'deposit' else ('bank', 'inventory')
            count = min(count, held[src].get(obj, 0))
            if count:
                held[src][obj] -= count
                held[dst][obj] = held[dst].get(obj, 0) + count
                log.append((tick, 'transfer', dict(src='inv' if src == 'inventory' else 'bank',
                                                   dst='inv' if dst == 'inventory' else 'bank', obj=obj, to=obj, count=count)))
    rows = lambda counts: [(slot, obj, n) for slot, (obj, n) in enumerate(counts.items()) if n]
    return log, dict(inventory=rows(held['inventory']), bank=rows(held['bank']), worn=[])


@given(events)
def test_retained_is_bounded_by_production_and_pickups(steps):
    log, holdings = play(steps)
    report = audit(log, dict(inventory=[], bank=[], worn=[]), dict(reason='horizon', tick=len(steps), holdings=holdings),
                   'wood', TABLES)
    kept = sum(n for loc in ('inventory', 'bank') for _, obj, n in holdings[loc] if obj == LOGS)
    produced = sum(e[2]['count'] for e in log if e[1] == 'produce' and e[2]['obj'] == LOGS)
    picked = sum(e[2]['count'] for e in log if e[1] == 'pickup' and e[2]['obj'] == LOGS)
    assert report['produced'] == produced and report['retained'] <= produced
    assert report['retained'] == max(0, kept - picked)


@given(events)
def test_holdings_the_events_do_not_explain_are_refused(steps):
    log, holdings = play(steps)
    holdings['bank'] = holdings['bank'] + [(239, LOGS, 1)]
    with pytest.raises(AuditError):
        audit(log, dict(inventory=[], bank=[], worn=[]), dict(reason='horizon', tick=len(steps), holdings=holdings),
              'wood', TABLES)


def test_production_outside_gathering_scripts_is_refused():
    log = [(0, 'produce', dict(obj=LOGS, count=1, script='[proc,not_gathering]'))]
    holdings = dict(inventory=[(0, LOGS, 1)], bank=[], worn=[])
    with pytest.raises(AuditError):
        audit(log, dict(inventory=[], bank=[], worn=[]), dict(reason='horizon', tick=1, holdings=holdings), 'wood', TABLES)


def test_removing_items_not_held_is_refused():
    log = [(0, 'remove', dict(obj=LOGS, count=1, why='drop'))]
    with pytest.raises(AuditError):
        audit(log, dict(inventory=[], bank=[], worn=[]), dict(reason='horizon', tick=1, holdings=dict(inventory=[], bank=[],
              worn=[])), 'wood', TABLES)


lives = st.lists(st.tuples(st.sampled_from([41, 1017, 2]), st.sampled_from([3, 2]),
                           st.lists(st.tuples(st.sampled_from(['me', 'other']), st.integers(0, 3)), max_size=5)), max_size=6)


@given(lives, st.randoms())
def test_a_kill_counts_once_per_life_from_full_health_defeated_only_by_the_agent(rows, random):
    """Chicken lives (types 41 and 1017, 3 hitpoints; type 2 is not an eligible enemy) hit in turn by
    the agent and others, their events interleaved across lives."""
    streams, expected = [], 0
    for k, (kind, hp, hits) in enumerate(rows):
        stream, sources = [], []
        for source, damage in hits:
            if hp == 0:
                break
            stream.append(('npc_damage', dict(npc=k // 2, type=kind, life=k % 2, source=source, damage=damage, hp=hp, full=3)))
            sources.append(source)
            hp = max(0, hp - damage)
            if hp == 0:
                stream.append(('npc_defeated', dict(npc=k // 2, type=kind, life=k % 2, source=source)))
        expected += kind != 2 and rows[k][1] == 3 and hp == 0 and set(sources) == {'me'}
        streams.append(stream)
    merged = []
    while any(streams):
        stream = random.choice([s for s in streams if s])
        merged.append((len(merged), *stream.pop(0)))
    assert kills(merged, 'me') == expected
    assert kills(merged, 'nobody') == 0
