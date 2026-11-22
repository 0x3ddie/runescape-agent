"""The changed worlds of the evaluation scorecard (sim/changed.py) on village-world-v2's plan records (development
7,000 on, key 239): each change differs from its base in its fact alone and leaves the base as it was, the check refuses
a second or another fact, the base record built there plays as sim/free.py Life, open-pen's map holds what the engine's
own scripts open and close, each changed world plays WAIT, random legal play and each candidate script for 2,000 ticks,
and an only-X world gains levels in X alone."""
import copy
import json

import numpy as np
import pytest

from sim import changed as C, free
from sim.build import DATA_NC2
from sim.engine import AP_TO_OP, APLOC1, LOC_DURATION, SCRIPTS, SKILLS
from sim.world import ORIGIN, PEN_BARRIERS

FIRST, KEY, TICKS = 7000, 239, 2000
# plan records that hold each fact: door pens 0 and 3, gate pens 6 and 11, a highwayman in 0 and 11, none in 1 and 6
RECORDS = {'no-highwayman': (0, 11), 'no-chickens': (1, 6), 'open-pen': (0, 6), 'no-food': (1, 3),
           **{f'only-{kind}': (0, 6) for kind in C.KINDS}}
SKILLS_OF = dict(wood={'woodcutting'}, mine={'mining'}, fish={'fishing'},
                 fight={'attack', 'strength', 'defence', 'hitpoints'})


@pytest.fixture(scope='module')
def nc2():
    if not DATA_NC2.exists():
        pytest.skip('private simulator tables (draynor-nc2.npz) absent; sim/build.py says how it is made')
    from sim.data import Tables
    return Tables.load(DATA_NC2)


@pytest.fixture(scope='module')
def bases(nc2):
    return {r: C.base_record(r, FIRST, KEY, nc2) for r in sorted({r for rs in RECORDS.values() for r in rs})}


def contents(rec):
    return rec.layout.arrays(), copy.deepcopy(rec.start), copy.deepcopy(rec.spawns)


def equal(a, b):
    (la, sa, pa), (lb, sb, pb) = a, b
    return set(la) == set(lb) and all(np.array_equal(la[k], lb[k]) for k in la) and (sa, pa) == (sb, pb)


@pytest.mark.parametrize('change', sorted(C.CHANGES))
def test_a_change_differs_from_its_base_in_its_fact_alone(nc2, bases, change):
    c = C.CHANGES[change]
    npc, loc = (lambda s: nc2.npc_types[s['type']]['debugname']), (lambda row: nc2.loc_types[int(row[0])]['debugname'])
    for r in RECORDS[change]:
        base = bases[r]
        before = contents(base)
        changed = C.edit(change, base, nc2)
        assert C.applies(change, base, nc2) and C.check(change, base, changed, nc2)
        assert equal(before, contents(base)), 'the edit changed its base'
        assert C.diff(changed, C.edit(change, base, nc2), nc2) == {}, 'two edits differ'
        assert list(changed.spawns) == [s for s in base.spawns if npc(s) not in c.npcs]  # the rest keep their nids
        assert [row for row in changed.layout.locs.tolist() if loc(row) in c.locs] == []
        assert {npc(s) for s in base.spawns} - {npc(s) for s in changed.spawns} == {npc(s) for s in base.spawns} & set(c.npcs)
        if c.pen:
            assert (changed.layout.gate < 0).all() and not any(loc(row) in PEN_BARRIERS for row in changed.layout.locs)
        if c.food:  # the life's inventory at arm holds the tools and the weapon alone
            held = {nc2.obj_types[it[0]]['debugname'] for it in C.task(changed, nc2, 1, TICKS).agent.player.inv.items if it}
            assert held and not held & set(base.start['village']['food'])


def test_the_check_refuses_a_second_fact_another_fact_or_none(nc2, bases):
    base, plain = bases[0], bases[1]  # a highwayman in record 0, none in record 1
    without = C.edit('no-highwayman', base, nc2)
    with pytest.raises(ValueError, match='start_record'):
        C.check('no-highwayman', base, C.edit('no-food', without, nc2), nc2)
    with pytest.raises(ValueError, match='spawns'):
        C.check('no-chickens', base, without, nc2)
    with pytest.raises(ValueError, match='nothing changed'):
        C.check('no-highwayman', plain, C.edit('no-highwayman', plain, nc2), nc2)
    assert not C.applies('no-highwayman', plain, nc2) and not C.applies('open-pen', plain, nc2)


def test_the_base_record_plays_as_a_free_life(nc2):
    """The life built from a record (sim/changed.py task) is sim/free.py Life's: the same row, events and last
    observation for random legal play and the melee control on a record with a highwayman and a door pen."""
    for name in ('random', 'melee'):
        rows, events, last = [], [], []
        for life in (free.Life(FIRST, KEY, 0, False, 1, 6000, nc2, 'development', world=C.WORLD),
                     C.ChangedLife(None, FIRST, KEY, 0, False, 1, 6000, nc2)):
            actor = free.make(name)
            played = free.play(life.task, actor, [4253, 0])
            rows.append({k: v for k, v in life.row(name, None, actor, played, None, 0).items() if k != 'seconds'})
            events.append(life.task.world.events)
            last.append(life.task.projected)
        assert rows[0]['decisions'] > 50 and len(events[0]) > 50, rows[0]
        assert rows[0] == rows[1]
        assert events[0] == events[1]
        assert set(last[0]) == set(last[1]) and all(np.array_equal(last[0][k], last[1][k]) for k in last[0])


def active(world):
    added = [loc for at in world.loc_at.values() for loc in at if loc.index >= world.n_stored_locs]
    return sorted((loc.type, loc.x - ORIGIN[0], loc.z - ORIGIN[1], loc.shape, loc.angle)
                  for loc in [*world.locs, *added] if loc.active)


def op1(world, loc):
    """The loc's first op, run as its script runs it."""
    list(SCRIPTS[world.trigger(loc, APLOC1 + AP_TO_OP)](world, world.player, loc))


@pytest.mark.parametrize('record', RECORDS['open-pen'])
def test_an_open_pen_is_what_its_barrier_opens_to_and_it_closes_and_opens_again(nc2, bases, record):
    """In the base world the barrier's own open script leaves the locs of open-pen's map; there the open stage's own
    script closes it (the base's closed map) and it stands open again LOC_DURATION ticks later."""
    base = bases[record]
    leaf, opened = int(base.layout.gate[0]), C.edit('open-pen', base, nc2)
    rows = lambda rec: sorted(tuple(int(v) for v in row) for row in rec.layout.locs)
    world = C.task(base, nc2, 1, TICKS).world
    op1(world, world.locs[leaf])
    assert active(world) == rows(opened)
    own = C.task(opened, nc2, 1, TICKS).world
    assert active(own) == rows(opened)
    op1(own, own.locs[leaf])
    assert active(own) == rows(base)
    for _ in range(LOC_DURATION):
        own.cycle()
    assert active(own) == rows(opened)


@pytest.mark.parametrize('change', sorted(C.CHANGES))
def test_a_changed_world_plays_wait_random_and_each_candidate_script(nc2, change):
    """2,000 ticks each: no unsupported stop, and no failed check (an AuditError would stop the play)."""
    for r in RECORDS[change]:
        for name in ('wait', 'random', *C.CHANGES[change].scripts):
            life = C.life(change, FIRST, KEY, r, False, 1, TICKS, nc2)
            info, decisions, error, _, _ = free.play(life.task, free.make(name, 4500), [4253, r])
            assert error is None and info is not None and decisions > 0, (change, r, name, error)


@pytest.mark.parametrize('kind', sorted(C.KINDS))
def test_an_only_x_world_gains_levels_in_x_alone(nc2, kind):
    """X's script gains levels, in X's skills only; the scripts of the three kinds that are not there gain none."""
    scripts = dict(wood='wood', mine='mine', fish='fish', fight='melee')
    for r in RECORDS[f'only-{kind}']:
        for other, script in scripts.items():
            life = C.life(f'only-{kind}', FIRST, KEY, r, False, 1, TICKS, nc2)
            free.play(life.task, free.make(script), [4253, r])
            p = life.task.agent.player
            gained = {s for k, s in enumerate(SKILLS) if p.base_levels[k] > life.base[k]}
            assert (gained and gained <= SKILLS_OF[kind]) if other == kind else not gained, (kind, r, script, gained)


def test_the_diff_command_prints_a_records_checked_diff(nc2, capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['sim.changed', 'diff', '--change', 'no-highwayman', '--record', '0', '--first',
                                     str(FIRST), '--key', str(KEY)])
    C.main()
    out = json.loads(capsys.readouterr().out)
    assert set(out['diff']) == {'npcs', 'spawns'} and [s[1] for s in out['diff']['spawns']['removed']] == ['highwayman']
