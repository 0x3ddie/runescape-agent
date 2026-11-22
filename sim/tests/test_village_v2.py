"""village-world-v2 in the Python reference: its layouts and their start rule, the field plants' pick (item, depletion,
respawn, a full inventory, no XP), the cabbage inside v2 and still excluded in v1, the highwayman's attack on a player
in its range and its silence above its level bound, the door pen (open, the revert, the path through it), the pay
across a death by the highwayman, the batch and the soak under the world's setting, the archive with the pick rows,
village-world-v1 bit-equal to its earlier code, and XpCheck's rule for a hit in an npc life's respawn cycle on the
pilot's pool."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from sim import free, world as W
from sim.agent import A_ACT0, A_HELD0, A_STYLE0, A_USE0, A_WAIT
from sim.audit import AuditError, XpCheck, fractional_level
from sim.build import DATA, DATA_NC2
from sim.combat import VILLAGE, VILLAGE_V2, draw_village, outside, village_life, village_of
from sim.engine import (APNPC1, DOOR_OPEN, FIELD_USE, HELD, HITPOINTS, LOC_DURATION, LOOT_OUTSIDE, SKILLS, Unsupported,
                        archive_script, held_script, loot_outside)
from sim.vec import CpuBatch
from sim.world import ORIGIN, VILLAGE_WORLD, LayoutTables

KEY = 239
S = ('woodcutting', 'mining', 'fishing', 'attack', 'strength', 'defence', 'hitpoints')


@pytest.fixture(scope='module')
def nc2():
    if not DATA_NC2.exists():
        pytest.skip('private simulator tables (draynor-nc2.npz) absent; sim/build.py says how it is made')
    from sim.data import Tables
    return Tables.load(DATA_NC2)


@pytest.fixture(scope='module')
def layouts(nc2):
    """The first 40 village-world-v2 development layouts from index 7,000."""
    return [W.generate(seed, nc2, VILLAGE_WORLD) for seed in W.split_seeds('development', 40, 7000)]


def names_of(layout, tables):
    return [tables.loc_types[int(r[0])]['debugname'] for r in layout.locs], \
        [tables.npc_types[int(r[1])]['debugname'] for r in layout.npcs]


def find(layouts, tables, want):
    """The first layout whose (loc names, npc names) satisfy `want`."""
    return next(layout for layout in layouts if want(*names_of(layout, tables)))


def life(tables, layout, length=3000, record=0, levels=None, hp=None, food=None):
    start = draw_village(KEY, 0, record, [layout], tables, False, layout=0)
    if levels:
        start['levels'] = {**start['levels'], **levels}
        start['village']['hp'] = min(start['village']['hp'], start['levels']['hitpoints'])
    if hp is not None:
        start['village']['hp'] = hp
    if food is not None:
        start['village']['food'] = food
    lt = LayoutTables(tables, layout)
    lt.CACHE = 64
    return village_life(start, layout, lt, tables, 1, length)


def step_until(task, action, done, limit=400):
    for _ in range(limit):
        if done():
            return True
        task.step(action)
    return done()


def act_slot(task, option, x=None, z=None, name=None):
    for k, sel in enumerate(task.agent.last_observation['actSel']):
        if sel['option'] == option and (x is None or (sel['x'], sel['z']) == (x, z)) and (name is None or sel['name'] == name) \
                and task.projected['mask'][A_ACT0 + k]:
            return A_ACT0 + k
    return None


def cycle_until(world, done, limit=2000):
    """Native ticks, one World.cycle each, until done(); returns how many ran."""
    for k in range(limit):
        if done():
            return k
        world.cycle()
    raise AssertionError('not done within the limit')


# ---- the layouts and the start rule ---------------------------------------------------------------------------------
def test_v2_layouts_hold_their_spec_and_name_their_world(nc2, layouts, tmp_path):
    """Each layout is 64 tiles a side and carries its world's name through a pool file; each start tile has every goal's
    nearest target within 60 steps; the field is one kind of 6 to 16 plants within field_half of each other, each on a
    tile of the start component; a door pen has one door whose open stage stands on the tile in front, turned a quarter;
    a willow, a yew and the highwayman come with the seed's variant; the highwayman stands on the start component with a
    share of start tiles in its reach within hunter_share; the world's npcs stay within 16."""
    from baseline.resource_task import GOALS
    reach = W.hunter_reach(nc2)
    for layout in layouts:
        fenced = np.random.default_rng([layout.seed, W.PEN_STREAM]).random() < VILLAGE_WORLD['pen_fraction']
        variant = W._variant(layout.seed, VILLAGE_WORLD, fenced)
        locs, npcs = names_of(layout, nc2)
        assert layout.flags.shape == (64, 64) and layout.world == 'village-world-v2' and W.world_of(layout) is VILLAGE_WORLD
        assert village_of(layout) is VILLAGE_V2 and layout.start.sum() >= VILLAGE_WORLD['min_starts']
        assert np.all((layout.goal_dist[:, layout.start] >= 0) & (layout.goal_dist[:, layout.start] <= 60)) and len(GOALS) == 4
        region = outside(layout)
        plants = [(x, z) for name, (_, x, z, _, _) in zip(locs, layout.locs) if name in dict(VILLAGE_WORLD['field_kinds'])]
        assert {n for n in locs if n in dict(VILLAGE_WORLD['field_kinds'])} == {variant['kind']}
        lo, hi = VILLAGE_WORLD['field_plants']
        assert lo <= len(plants) <= hi and all(region[x, z] for x, z in plants)
        span = np.ptp(np.asarray(plants), axis=0)
        assert span.max() <= 2 * VILLAGE_WORLD['field_half']
        assert (W.WILLOW in locs) == variant['willow'] and (W.YEW in locs) == variant['yew']
        gate = [int(k) for k in layout.gate if k >= 0]
        assert bool(gate) == fenced and (W.PEN_DOOR in locs) == variant['door'] and layout.gate.shape == (2,)
        if variant['door']:
            assert len(gate) == 1 and locs[gate[0]] == W.PEN_DOOR and len(layout.open_locs) == 1
            _, x, z, shape, angle = (int(v) for v in layout.locs[gate[0]])
            stage, ox, oz, oshape, oangle = (int(v) for v in layout.open_locs[0])
            assert (ox - x, oz - z) == DOOR_OPEN[shape][angle] and oangle == (angle + 1) % 4 and oshape == shape
            assert nc2.loc_types[stage]['debugname'] == 'poordoor_cross'
        assert (npcs.count(W.HUNTER) == 1) == variant['hunter'] and npcs.count(W.HUNTER) <= 1
        if variant['hunter']:
            _, _, hx, hz = (int(v) for v in layout.npcs[npcs.index(W.HUNTER)])
            share = np.mean(np.max(np.abs(np.argwhere(layout.start) - (hx, hz)), axis=1) <= reach)
            assert region[hx, hz] and VILLAGE_WORLD['hunter_share'][0] <= share <= VILLAGE_WORLD['hunter_share'][1]
        hens = sum(n in W.CHICKENS for n in npcs)
        assert len(npcs) + hens <= 16  # the world adds a goblin at every chicken spawn (sim/combat.py village_npcs)
    W.save_pool(tmp_path / 'v2.npz', layouts[:3])
    loaded, _ = W.load_pool(tmp_path / 'v2.npz')
    assert [layout.world for layout in loaded] == ['village-world-v2'] * 3
    assert all(all(np.array_equal(a.arrays()[k], b.arrays()[k]) for k in a.arrays()) for a, b in zip(loaded, layouts))
    with pytest.raises(ValueError):  # a pool holds one declared world, named as it is
        W.save_pool(tmp_path / 'mixed.npz', layouts[:1] + [W.generate(W.split_seeds('train', 1)[0], nc2)])
    with pytest.raises(ValueError):
        W.save_pool(tmp_path / 'search.npz', layouts[:1], version='search-world-v1')


def test_regions_stand_20_apart_and_the_highwayman_4_to_10_tiles_from_its_drawn_centre(nc2, layouts):
    """Each layout's passing attempt drawn again from (seed, attempt) with the generator's own steps (its locs are the
    layout's): the region centres (pen, groves, mine, field) are at least region_separation (20) apart, Chebyshev, and the
    highwayman stands hunter_near (4 to 10) from the centre its placement drew next from the attempt's stream."""
    types, seen = W._Types(nc2), 0
    for layout in layouts:
        fenced = np.random.default_rng([layout.seed, W.PEN_STREAM]).random() < VILLAGE_WORLD['pen_fraction']
        variant = W._variant(layout.seed, VILLAGE_WORLD, fenced)
        d = W._Draw(np.random.default_rng([layout.seed, layout.attempt]), types, fenced, VILLAGE_WORLD, variant)
        d.draw_water()
        assert d.draw_fish() and d.draw_things()
        d.draw_fences()
        assert [tuple(int(v) for v in r) for r in d.locs] == [tuple(int(v) for v in r) for r in layout.locs]
        centres = [c for name in ('pen', 'grove', 'mine', 'field') for c in d.regions[name]]
        assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= VILLAGE_WORLD['region_separation'] == 20
                   for k, a in enumerate(centres) for b in centres[k + 1:])
        if variant['hunter']:
            cx, cz = centres[int(d.rng.integers(len(centres)))]
            _, _, hx, hz = (int(v) for v in layout.npcs[[nc2.npc_types[int(r[1])]['debugname'] for r in layout.npcs].index(W.HUNTER)])
            assert VILLAGE_WORLD['hunter_near'] == (4, 10) and 4 <= max(abs(hx - cx), abs(hz - cz)) <= 10
            seen += 1
    assert seen >= 10


def test_the_variant_shares_are_the_specs():
    """The per-seed draws (pen, door, field kind, willow, yew, highwayman) over 2,000 seeds lie near their shares."""
    rows = []
    for seed in W.split_seeds('train', 2000, 30000):
        fenced = np.random.default_rng([seed, W.PEN_STREAM]).random() < VILLAGE_WORLD['pen_fraction']
        rows.append(dict(W._variant(seed, VILLAGE_WORLD, fenced), fenced=fenced))
    share = lambda key, among=rows: np.mean([r[key] for r in among])
    assert abs(share('fenced') - .5) < .05 and abs(share('door', [r for r in rows if r['fenced']]) - .5) < .06
    assert not any(r['door'] for r in rows if not r['fenced'])
    for key in ('willow', 'yew', 'hunter'):
        assert abs(share(key) - .5) < .05
    kinds = [r['kind'] for r in rows]
    for kind, want in VILLAGE_WORLD['field_kinds']:
        assert abs(kinds.count(kind) / len(kinds) - want) < .05
    assert W._variant(1, W.WORLD, True) == {}  # the first world draws nothing of it


# ---- the pick, the cabbage and the field items ----------------------------------------------------------------------------
def at_plant(nc2, layouts, kind):
    layout = find(layouts, nc2, lambda locs, npcs: kind in locs and W.HUNTER not in npcs)
    task = life(nc2, layout)
    w, p = task.world, task.agent.player
    plant = next(loc for loc in w.locs if nc2.loc_types[loc.type]['debugname'] == kind)
    beside = next((plant.x + dx, plant.z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if w.t.source(0, plant.x + dx, plant.z + dz) >= 0 and not w.active_locs_at(0, plant.x + dx, plant.z + dz))
    w.teleport(p, *beside)
    task.step(A_WAIT)
    return task, plant


@pytest.mark.parametrize('kind,item,respawn', [('cabbage', 'cabbage', 70), ('potato', 'potato', 49), ('onion', 'onion', 49)])
def test_a_pick_adds_the_item_depletes_the_plant_and_respawns_it_with_no_xp(nc2, layouts, kind, item, respawn):
    """pickables.rs2: the player is put on the plant's tile, the plant goes and its item comes, with the picked message
    and no XP; it stands again after its respawn (70 ticks for a cabbage, 50 scaled to one player, 49, for the others)."""
    task, plant = at_plant(nc2, layouts, kind)
    w, p = task.world, task.agent.player
    before, stats, events = p.inv.total(nc2.obj_id(item)), list(p.stats), len(w.events)
    task.step(act_slot(task, 'Pick', plant.x, plant.z))
    assert step_until(task, A_WAIT, lambda: not plant.active, 20)
    picked = [e for e in w.events[events:] if e[1] == 'produce']
    assert [(e[2]['obj'], e[2]['count'], e[2]['script']) for e in picked] == [(nc2.obj_id(item), 1, free.PICK_SCRIPT)]
    assert p.inv.total(nc2.obj_id(item)) == before + 1 and (p.x, p.z) == (plant.x, plant.z) and p.stats == stats
    assert not any(e[1] == 'xp' for e in w.events[events:]) and (picked[0][0], w.text(f'pick_{kind}')) in w.chat_log
    assert act_slot(task, 'Pick', plant.x, plant.z) is None  # a picked plant is not offered
    gone = w.tick - picked[0][0]
    assert gone + cycle_until(w, lambda: plant.active) == respawn


def test_a_walked_pick_waits_for_the_arrival_then_a_tick_before_the_item(nc2, layouts):
    """pickables.rs2 after a walk: the op runs the tick after the last step (a loc op waits for a still player),
    p_arrivedelay holds it a tick, p_teleport comes the next (2 ticks after the last step) and p_delay(0) a tick more,
    then loc_del, the item and the message (3 ticks after the last step)."""
    task, plant = at_plant(nc2, layouts, 'cabbage')
    w, p, cabbage = task.world, task.agent.player, nc2.obj_id('cabbage')
    region = outside(task.world.t.layout)
    tile = next((plant.x + d * sx, plant.z + d * sz) for d in (3, 4) for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= plant.x + d * sx - ORIGIN[0] < 64 and 0 <= plant.z + d * sz - ORIGIN[1] < 64
                and region[plant.x + d * sx - ORIGIN[0], plant.z + d * sz - ORIGIN[1]])
    w.teleport(p, *tile)
    task.step(A_WAIT)
    seen, teleport, remove = [], w.teleport, w.remove_loc
    w.teleport = lambda pl, x, z: (seen.append(('teleport', w.tick)), teleport(pl, x, z))[1]
    w.remove_loc = lambda loc, duration: (seen.append(('loc_del', w.tick, loc is plant, duration)), remove(loc, duration))[1]
    issued, events = w.tick, len(w.events)
    task.step(act_slot(task, 'Pick', plant.x, plant.z))
    assert step_until(task, A_WAIT, lambda: p.inv.total(cabbage), 20)
    w.teleport, w.remove_loc = teleport, remove
    walked = p.last_movement - 1  # Player.lastMovement: the tick of the last step + 1
    assert walked >= issued and max(abs(tile[0] - plant.x), abs(tile[1] - plant.z)) >= 3
    item = [e[0] for e in w.events[events:] if e[1] == 'produce' and e[2]['obj'] == cabbage]
    assert seen == [('teleport', walked + 2), ('loc_del', walked + 3, True, 70)] and item == [walked + 3]
    assert (walked + 3, w.text('pick_cabbage')) in w.chat_log


def test_a_full_inventory_picks_nothing_but_moves_the_player_onto_the_plant(nc2, layouts):
    task, plant = at_plant(nc2, layouts, 'cabbage')
    w, p = task.world, task.agent.player
    while p.inv.freespace():
        p.inv.add(nc2, nc2.obj_id('logs'))
    task.step(A_WAIT)
    events, held = len(w.events), p.inv.listed()
    task.step(act_slot(task, 'Pick', plant.x, plant.z))
    step_until(task, A_WAIT, lambda: (p.x, p.z) == (plant.x, plant.z), 10)
    assert (p.x, p.z) == (plant.x, plant.z) and plant.active and p.inv.listed() == held
    assert w.text('pick_full_cabbage') in [text for _, text in w.chat_log] and not any(e[1] == 'produce' for e in w.events[events:])


def test_the_cabbage_is_eaten_used_and_dropped_in_v2(nc2, layouts):
    """The cabbage's Eat heals 1 ([opheld1,_category_131], consume_normal.dbrow) with its messages; using it on another item
    and a potato or onion likewise show the default message (use_default); it drops and is taken again."""
    task, plant = at_plant(nc2, layouts, 'cabbage')
    w, p, a = task.world, task.agent.player, task.agent
    task.step(act_slot(task, 'Pick', plant.x, plant.z))
    step_until(task, A_WAIT, lambda: p.inv.total(nc2.obj_id('cabbage')), 20)
    p.levels[HITPOINTS] = p.base_levels[HITPOINTS] - 3
    task.step(A_WAIT)
    eat = next(A_HELD0 + k for k, h in enumerate(a.last_observation['heldSel']) if h['option'] == 'Eat' and h['name'] == 'Cabbage')
    hp = p.hp
    task.step(eat)
    assert p.hp == hp + 1 and p.inv.total(nc2.obj_id('cabbage')) == 0
    assert [t for _, t in w.chat_log[-2:]] == ['You eat the cabbage. Yuck!', 'It heals some health anyway.']
    for name in FIELD_USE:  # ported over the pick archive, as the picks
        assert name not in HELD and held_script(task.world, name) is archive_script(task.world, name) is not None
    p.inv.add(nc2, nc2.obj_id('cabbage'))
    task.step(A_WAIT)
    use = next(A_USE0 + k for k, h in enumerate(a.last_observation['invSel']) if h['id'] == nc2.obj_id('cabbage'))
    task.step(use)
    axe = next(A_HELD0 + k for k, h in enumerate(a.last_observation['heldSel']) if h['name'] == 'Bronze axe')
    task.step(axe)
    assert w.chat_log[-1][1] == w.text('dm_default') and p.inv.total(nc2.obj_id('cabbage')) == 1


def test_the_field_use_scripts_take_their_scripts_cases():
    """Each ported OPHELDU switch (FIELD_USE) refuses exactly the cases its vendor script names besides the default."""
    import re
    from sim.build import CONTENT, label_body
    if not (CONTENT / 'scripts').exists():
        pytest.skip('needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md)')
    files = {'[opheldu,cabbage]': 'skill_cooking/scripts/gnome_cooking/gnome_cooking.rs2',
             '[opheldu,potato]': 'skill_cooking/scripts/cooking_inv/scripts/stew/stew.rs2',
             '[opheldu,onion]': 'skill_cooking/scripts/cooking_inv/scripts/ugthanki_kebab/ugthanki_kebab.rs2',
             '[opheldu,black_cape]': 'skill_crafting/scripts/dye_cape/dye_cape.rs2'}
    assert set(files) == set(FIELD_USE)
    for name, file in files.items():
        body = label_body(file, name)
        cases = {c.strip() for group in re.findall(r'case ([\w, ]+) :', body) for c in group.split(',')} - {'default'}
        assert set(FIELD_USE[name]) == cases and 'case default : ~displaymessage(^dm_default);' in body, name


def test_a_use_on_the_highwaymans_cape_shows_the_default_message(nc2, layouts):
    task, _ = at_plant(nc2, layouts, 'cabbage')
    w, p, a = task.world, task.agent.player, task.agent
    p.inv.add(nc2, nc2.obj_id('black_cape'))
    task.step(A_WAIT)
    use = next(A_USE0 + k for k, h in enumerate(a.last_observation['invSel']) if h['id'] == nc2.obj_id('bronze_axe'))
    task.step(use)
    cape = next(A_HELD0 + k for k, h in enumerate(a.last_observation['heldSel']) if h['name'] == 'Cape')
    task.step(cape)
    assert w.chat_log[-1][1] == w.text('dm_default')
    wear = next(A_HELD0 + k for k, h in enumerate(a.last_observation['heldSel']) if h['name'] == 'Cape' and h['option'] == 'Wear')
    task.step(wear)
    assert any(o == nc2.obj_id('black_cape') for _, o, _ in p.worn.listed())


def test_the_cabbage_is_inside_v2_and_still_excluded_in_v1(tables, nc2, layouts):
    """The lists and the ports: a v2 life's World holds VILLAGE_V2's loot list (LOOT_OUTSIDE without the cabbage) and a
    v1 life's LOOT_OUTSIDE; loot_outside answers by the list it is given; the picks and the field items' and the cape's
    use scripts have a port over draynor-nc2.npz and none over draynor-nc1.npz; v2 refuses draynor-nc1.npz and muggers.
    (The exclusion acting on a death's drop: test_a_cabbage_drop_is_excluded_in_v1_and_falls_in_v2.)"""
    cabbage = nc2.obj_id('cabbage')
    v2 = life(nc2, layouts[0])
    v1_layout = W.generate(W.split_seeds('development', 1, 4000)[0], tables)
    v1 = village_life(draw_village(KEY, 0, 0, [v1_layout], tables, False, layout=0), v1_layout, LayoutTables(tables, v1_layout),
                      tables, 1, 100)
    assert v2.world.loot_outside == VILLAGE_V2['loot_outside'] and 'cabbage' not in v2.world.loot_outside
    assert v1.world.loot_outside == LOOT_OUTSIDE and 'cabbage' in LOOT_OUTSIDE
    assert not loot_outside(nc2, cabbage, v2.world.loot_outside) and loot_outside(tables, cabbage, v1.world.loot_outside)
    assert set(VILLAGE_V2['loot_outside']) == set(LOOT_OUTSIDE) - {'cabbage'}
    for name in ('[oploc2,cabbage]', '[oploc2,potato]', '[oploc2,onion]', *FIELD_USE):  # over draynor-nc1.npz: no port
        assert archive_script(v1.world, name) is None and archive_script(v2.world, name) is not None
        assert name.startswith('[oploc2,') or held_script(v1.world, name) is None  # World.run_held: Unsupported, as before
    with pytest.raises(ValueError, match='pick'):  # v2 refuses an archive without the pick content
        village_life(draw_village(KEY, 0, 0, [layouts[0]], tables, False, layout=0), layouts[0],
                     LayoutTables(tables, layouts[0]), tables, 1, 100)
    with pytest.raises(ValueError, match='no muggers'):
        draw_village(KEY, 0, 0, [layouts[0]], nc2, True, layout=0)
    assert village_of(v1_layout) is VILLAGE


def mugger_death(tables, layout, lt, tick, loot=None):
    """A mugger's death script ([ai_queue3,mugger], the agent its hero) run whole at `tick` in a fresh v1 life with
    muggers, its World's loot list replaced by `loot` when given: (its events, the ground items it added)."""
    from sim.engine import npc_death_script
    start = draw_village(KEY, 0, 0, [layout], tables, True, layout=0)
    task = village_life(start, layout, lt, tables, 1, 100)
    w, p = task.world, task.agent.player
    if loot is not None:
        w.loot_outside = loot
    mugger = next(n for n in w.npcs if tables.npc_types[n.type]['debugname'] == 'mugger')
    mugger.heroes = {p.name: 1}
    w.tick, events, objs = tick, len(w.events), list(w.objs)
    for _ in npc_death_script(w, mugger, '[ai_queue3,mugger]'):
        pass
    return w.events[events:], [o for o in w.objs if o not in objs]


def test_a_cabbage_drop_is_excluded_in_v1_and_falls_in_v2(tables):
    """mugger.rs2 drops a cabbage on one roll in 128. In a v1 World (loot_drops, LOOT_OUTSIDE) that roll gives an
    'excluded' loot_drop event and no ground item; with VILLAGE_V2's list the same death at the same tick puts the cabbage
    on the ground, with no exclusion event."""
    layout = W.generate(W.split_seeds('development', 1, 4000)[0], tables)
    lt, cabbage = LayoutTables(tables, layout), tables.obj_id('cabbage')
    excluded = lambda events: [e[2] for e in events if e[1] == 'excluded' and e[2]['obj'] == cabbage]
    tick = next(t for t in range(20, 2000) if excluded(mugger_death(tables, layout, lt, t)[0]))
    events, objs = mugger_death(tables, layout, lt, tick)
    assert excluded(events) == [dict(what='loot_drop', obj=cabbage, npc=excluded(events)[0]['npc'])]
    assert cabbage not in [o.type for o in objs]
    events, objs = mugger_death(tables, layout, lt, tick, VILLAGE_V2['loot_outside'])
    assert not excluded(events) and [o.type for o in objs].count(cabbage) == 1


def test_a_pick_over_an_archive_without_its_rows_stays_unsupported(tables, nc2, layouts):
    """A v2 layout's plant in a World over draynor-nc1.npz: the Pick raises the same Unsupported as it did earlier."""
    from sim.engine import World
    from sim.task import WORLD_START_TICK, FreePlayEpisode
    from sim.combat import kit, village_npcs
    layout = find(layouts, nc2, lambda locs, npcs: 'cabbage' in locs)
    start = draw_village(KEY, 0, 0, [layout], nc2, False, layout=0)
    world = World(LayoutTables(tables, layout), seed=start['world_seed'], start_tick=WORLD_START_TICK,
                  npcs=village_npcs(layout, tables, []), exclusions=VILLAGE['exclusions'], xp_rate=1)
    task = FreePlayEpisode(world, 3000, 1)
    task.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], start['village']['food']),
               hp=start['village']['hp'], wield=start['village']['worn'])
    plant = next(loc for loc in world.locs if tables.loc_types[loc.type]['debugname'] == 'cabbage')
    world.teleport(task.agent.player, plant.x + 1 if world.t.source(0, plant.x + 1, plant.z) >= 0 else plant.x - 1, plant.z)
    task.step(A_WAIT)
    with pytest.raises(Unsupported, match=r'engine script \[oploc2,cabbage\]'):
        step_until(task, act_slot(task, 'Pick', plant.x, plant.z), lambda: False, 10)


# ---- the highwayman -------------------------------------------------------------------------------------------------------
def near_hunter(nc2, layouts, levels, hp=None, length=3000):
    layout = find(layouts, nc2, lambda locs, npcs: W.HUNTER in npcs)
    task = life(nc2, layout, length=length, levels=levels, hp=hp, food=[])
    w, p = task.world, task.agent.player
    hunter = next(n for n in w.npcs if nc2.npc_types[n.type]['debugname'] == W.HUNTER)
    tile = next((hunter.x + dx, hunter.z + dz) for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2))
                if w.t.source(0, hunter.x + dx, hunter.z + dz) >= 0 and w.line_of_sight(0, hunter.x + dx, hunter.z + dz, hunter.x, hunter.z)
                and not w.active_locs_at(0, hunter.x + dx, hunter.z + dz))
    w.teleport(p, *tile)
    return task, hunter


LOW = dict(attack=1, strength=1, defence=1, hitpoints=10)    # combat level 3: hunted (at most twice its level 5)
HIGH = dict(attack=10, strength=10, defence=10, hitpoints=12)  # combat level 12: above its bound
EDGE = {10: dict(attack=10, strength=10, defence=5, hitpoints=12), 11: dict(attack=10, strength=10, defence=6, hitpoints=12)}


@pytest.mark.parametrize('level', sorted(EDGE))
def test_the_highwaymans_bound_is_twice_its_level(nc2, layouts, level):
    """Npc.ts huntPlayers: a player whose combat level is above vislevel x 2 (5 x 2 = 10) is not hunted. Level 10 is
    attacked, level 11 never is."""
    task, hunter = near_hunter(nc2, layouts, EDGE[level])
    w, p = task.world, task.agent.player
    assert p.combat_level() == level and nc2.npc_types[hunter.type]['vislevel'] * 2 == 10
    events = len(w.events)
    hit = lambda: any(e[1] == 'agent_hp' and e[2]['cause'] == 'damage' and e[2]['source'] == hunter.nid for e in w.events[events:])
    assert step_until(task, A_WAIT, hit, 60 if level == 10 else 150) == (level == 10)


def test_the_highwayman_attacks_a_player_of_its_level_in_its_range(nc2, layouts):
    """Hunt mode ranged (all.hunt), [ai_applayer2,highwayman] then melee: a player of combat level 10 or less within its
    hunt range and sight is approached and hit, with no act of the player's; its largest hit is 1."""
    task, hunter = near_hunter(nc2, layouts, LOW)
    w, p = task.world, task.agent.player
    assert p.combat_level() <= 2 * nc2.npc_types[hunter.type]['vislevel']
    events = len(w.events)
    hit = lambda: any(e[1] == 'agent_hp' and e[2]['cause'] == 'damage' and e[2]['source'] == hunter.nid for e in w.events[events:])
    assert step_until(task, A_WAIT, hit, 60) and hunter.target is p
    damage = [e[2]['before'] - e[2]['after'] for e in w.events[events:] if e[1] == 'agent_hp' and e[2]['source'] == hunter.nid]
    assert max(damage) <= 1


def test_the_highwayman_is_silent_to_a_player_above_its_level_bound(nc2, layouts):
    task, hunter = near_hunter(nc2, layouts, HIGH)
    w, p = task.world, task.agent.player
    assert p.combat_level() > 2 * nc2.npc_types[hunter.type]['vislevel']
    events = len(w.events)
    step_until(task, A_WAIT, lambda: False, 150)
    assert not any(e[1] == 'agent_hp' and e[2]['source'] == hunter.nid for e in w.events[events:]) and hunter.target is not p


def test_a_death_by_the_highwayman_ends_the_life_and_its_pay_is_its_level_gain(nc2, layouts):
    """A life at 1 hitpoint beside the highwayman, WAIT only: auto-retaliation gains combat XP, the highwayman's hit ends
    the life (terminal death), and the paid sum is the fractional level gain over S, nothing taken back."""
    task, hunter = near_hunter(nc2, layouts, LOW, hp=2)
    w, p = task.world, task.agent.player
    start = list(p.stats)
    paid, done, info = 0.0, False, None
    for _ in range(2000):
        _, reward, done, info = task.step(A_WAIT)
        paid += reward
        if done:
            break
    assert done and info['report']['reason'] == 'death'
    last = [e for e in w.events[task.event_start:] if e[1] == 'agent_hp' and e[2]['cause'] == 'damage' and e[2]['after'] < e[2]['before']]
    assert last[-1][2]['source'] == hunter.nid and last[-1][2]['after'] <= 0
    gain = sum(fractional_level(p.stats[SKILLS.index(s)]) - fractional_level(start[SKILLS.index(s)]) for s in S)
    assert abs(paid - gain) < 1e-6 and abs(info['episode']['r'] - gain) < 1e-6 and info['report']['paid'] == pytest.approx(gain)
    assert info['outcomes'] == sum(p.base_levels[SKILLS.index(s)] - lv for s, (lv, _) in info['report']['levels'].items())


# ---- the door pen -----------------------------------------------------------------------------------------------------------
def test_a_door_pen_opens_reverts_after_500_ticks_and_lets_the_player_through(nc2, layouts):
    """doors.rs2 [oploc1,_door_closed]: the door goes and its next stage stands on the tile in front, turned a quarter, for
    500 ticks; then both revert. Closed, no step reaches the pen's inside from the start component; open, one does."""
    layout = find(layouts, nc2, lambda locs, npcs: W.PEN_DOOR in locs and W.HUNTER not in npcs)
    task = life(nc2, layout)
    w, p = task.world, task.agent.player
    door = next(loc for loc in w.locs if nc2.loc_types[loc.type]['debugname'] == W.PEN_DOOR)
    dx, dz = DOOR_OPEN[door.shape][door.angle]
    w.teleport(p, door.x + dx, door.z + dz)
    task.step(A_WAIT)
    inside = layout.walkable & ~outside(layout)
    target = tuple(int(v) for v in np.argwhere(inside)[0])
    reach = free.Reach()
    assert reach.steps(task)[target] < 0
    removed, remove = [], w.remove_loc
    w.remove_loc = lambda loc, duration: (removed.append((w.tick, loc)), remove(loc, duration))[1]
    task.step(act_slot(task, 'Open', door.x, door.z))
    assert step_until(task, A_WAIT, lambda: not door.active, 10)
    w.remove_loc = remove
    stage = [loc for at in w.loc_at.values() for loc in at if loc.index >= w.n_stored_locs and loc.active]
    assert [(nc2.loc_types[s.type]['debugname'], s.x, s.z, s.angle) for s in stage] == \
        [('poordoor_cross', door.x + dx, door.z + dz, (door.angle + 1) % 4)]
    assert free.Reach().steps(task)[target] >= 0
    opened = next(tick for tick, loc in removed if loc is door)
    cycle_until(w, lambda: door.active)
    assert w.tick - opened == LOC_DURATION and not stage[0].active and free.Reach().steps(task)[target] < 0


def test_the_controls_open_a_closed_pen_door(nc2, layouts):
    """The fighter, the chickens control and a rotation gatherer take a closed pen door as they take a closed gate."""
    layout = find(layouts, nc2, lambda locs, npcs: W.PEN_DOOR in locs and W.HUNTER not in npcs)
    task = life(nc2, layout)
    w, p = task.world, task.agent.player
    door = next(loc for loc in w.locs if nc2.loc_types[loc.type]['debugname'] == W.PEN_DOOR)
    dx, dz = DOOR_OPEN[door.shape][door.angle]
    w.teleport(p, door.x + dx, door.z + dz)
    task.step(A_WAIT)
    opening = act_slot(task, 'Open', door.x, door.z)
    assert opening is not None and free.open_gate(task) == opening
    assert free.ReachReference().targets(task, []) == {('loc', door.x, door.z, 'Open')}
    for name in ('melee', 'chickens'):  # each may wield, eat or set its style first: none moves the player
        control = free.make(name)
        control.reset(0)
        for _ in range(8):
            action = control.act(task.projected, task)
            if action == opening:
                break
            assert A_HELD0 <= action < A_HELD0 + 8 or A_STYLE0 <= action < A_STYLE0 + 4, (name, action)
            task.step(action)
        assert action == opening and door.active, name
    inside = layout.walkable & ~outside(layout)
    x, z = (int(v) for v in np.argwhere(inside)[0])
    w.teleport(p, ORIGIN[0] + x, ORIGIN[1] + z)
    task.step(A_WAIT)
    rotation = free.make('rotation', period=10 ** 6)
    rotation.reset(0)
    gatherer = rotation.members[rotation.order.index('wood')]
    assert free.in_pen(task, {}) and gatherer.act(task.projected, task) == free.open_gate(task) is not None


def test_a_reference_walks_toward_targets_past_the_client_scene_and_never_does_in_v1(nc2, layouts):
    """In a 64-tile layout the fishing spots can lie past the client's scene of the player's build area: the client
    search of click_toward reaches none and reference_action waits; the gatherer walks toward them. Every tile of a
    48-tile layout lies in the scene of a build area anywhere in it, so the walk never acts in compact-world-v1."""
    from sim.paths import SCENE, scene_base
    from sim.run import reference_action
    for ox in range(48):
        base = scene_base(ORIGIN[0] + ox, ORIGIN[1] + ox)
        assert base[0] <= ORIGIN[0] and base[0] + SCENE > ORIGIN[0] + 47 and base[1] <= ORIGIN[1] and base[1] + SCENE > ORIGIN[1] + 47
    for layout in layouts:
        task = life(nc2, layout)
        w, p = task.world, task.agent.player
        spots = [(n.x, n.z) for n in w.npcs if nc2.npc_types[n.type]['debugname'] == W.SPOT]
        tile = next(((ORIGIN[0] + int(x), ORIGIN[1] + int(z)) for x, z in np.argwhere(outside(layout))
                     if all(not (scene_base(ORIGIN[0] + int(x), ORIGIN[1] + int(z))[0] <= a < scene_base(ORIGIN[0] + int(x), ORIGIN[1] + int(z))[0] + SCENE
                                 and scene_base(ORIGIN[0] + int(x), ORIGIN[1] + int(z))[1] <= b < scene_base(ORIGIN[0] + int(x), ORIGIN[1] + int(z))[1] + SCENE)
                            for a, b in spots)), None)
        if tile is None:
            continue
        w.teleport(p, *tile)
        p.origin_x, p.origin_z = tile  # the build area as a login or a recentre there leaves it
        task.step(A_WAIT)
        assert not any(free.in_scene(task, a, b) for a, b in spots)
        assert reference_action(free.view(task, 'fish')) == A_WAIT
        gatherer = free.make('fish')
        gatherer.reset(0)
        action = gatherer.act(task.projected, task)
        assert action != A_WAIT and free.toward_scene(task, A_WAIT, {('npc', a, b, 'Net') for a, b in spots}) == action
        return
    raise AssertionError('no layout with a start-component tile whose scene leaves out every fishing spot')


def test_the_chickens_control_walks_toward_a_closed_pen_door_past_the_client_scene_and_opens_it(nc2, layouts):
    """The chickens control's door rule runs whenever a pen door is closed, wherever the player is: from a tile whose
    client scene leaves the door out, the click toward it (open_gate) is a WAIT, and nothing else moves the player. The
    rule goes through toward_scene, as the fighter's door target does: the control walks toward the door and opens it."""
    from act_contract import CLICK_ROWS
    from sim.paths import SCENE, scene_base
    past = lambda tile, door: not all(b <= v < b + SCENE for v, b in zip(door, scene_base(*tile)))
    for layout in layouts:
        locs, npcs = names_of(layout, nc2)
        if W.PEN_DOOR not in locs or W.HUNTER in npcs:
            continue
        _, dx, dz, _, _ = (int(v) for v in layout.locs[locs.index(W.PEN_DOOR)])
        door = (ORIGIN[0] + dx, ORIGIN[1] + dz)
        tile = next(((ORIGIN[0] + int(x), ORIGIN[1] + int(z)) for x, z in np.argwhere(outside(layout))
                     if past((ORIGIN[0] + int(x), ORIGIN[1] + int(z)), door)), None)
        if tile is None:
            continue
        task = life(nc2, layout)
        w, p = task.world, task.agent.player
        w.teleport(p, *tile)
        p.origin_x, p.origin_z = tile  # the build area as a login or a recentre there leaves it
        task.step(A_WAIT)
        control = free.make('chickens')
        control.reset(0)
        for _ in range(8):  # it may wield, eat or set its style first: none moves the player
            action = control.act(task.projected, task)
            if not (A_HELD0 <= action < A_HELD0 + 8 or A_STYLE0 <= action < A_STYLE0 + 4):
                break
            task.step(action)
        assert not free.in_scene(task, *door) and free.open_gate(task, (W.PEN_DOOR,)) == A_WAIT
        assert action in CLICK_ROWS and action == free.toward_scene(task, A_WAIT, free.closed(task, (W.PEN_DOOR,)))
        shut = next(loc for loc in w.locs if nc2.loc_types[loc.type]['debugname'] == W.PEN_DOOR)
        for _ in range(300):
            task.step(int(control.act(task.projected, task)))
            if not shut.active:
                return
        raise AssertionError('the control did not open the door')
    raise AssertionError('no door layout with a start-component tile whose scene leaves the door out')


def test_a_control_whose_pen_door_reverts_in_a_dialog_continues_it_opens_the_door_again_and_goes_on(nc2):
    """Training layout 20014 (a door pen), the chickens control, start (key 7, slot 0, episode 7): the door the control
    opened reverts by tick 1,118 while it fights in the pen, and a level-up's dialog opens. Its door rule ran before
    reference_action's dialog step, and with the dialog open the Open and every click are masked, so it waited to the
    end of the life. It now continues the dialog, opens the door again and gains XP after the revert."""
    from sim.agent import A_DIALOG_CONTINUE
    layout = W.generate(W.split_seeds('train', 1, 20014)[0], nc2, VILLAGE_WORLD)  # the training pool's layout 20014
    start = draw_village(7, 0, 7, [layout], nc2, False, layout=0)
    lt = LayoutTables(nc2, layout)
    lt.CACHE = 64
    task = village_life(start, layout, lt, nc2, 1, 3000)
    w, p = task.world, task.agent.player
    door = next(loc for loc in w.locs if nc2.loc_types[loc.type]['debugname'] == W.PEN_DOOR)
    opened, remove = [], w.remove_loc
    w.remove_loc = lambda loc, duration: (opened.append(w.tick - task.start_tick) if loc is door else None, remove(loc, duration))[1]
    actor = free.make('chickens')
    actor.reset([4253, 7])
    obs, waited, longest, xp_late = task.projected, 0, 0, 0
    while True:
        action = int(actor.act(obs, task))
        if action == A_WAIT and task.agent.last_observation['mask'][A_DIALOG_CONTINUE]:
            waited += 1
            longest = max(longest, waited)
        else:
            waited = 0
        before = sum(p.stats[SKILLS.index(s)] for s in S)
        obs, _, done, _ = task.step(action)
        if w.tick - task.start_tick > 1200:
            xp_late += sum(p.stats[SKILLS.index(s)] for s in S) - before
        if done:
            break
    assert longest < 3 and xp_late > 0
    assert any(t < 1118 for t in opened) and any(t > 1118 for t in opened)  # opened, reverted at 1,118, opened again


# ---- the batch, the soak and the archive -------------------------------------------------------------------------------------
def test_the_batch_takes_its_world_from_the_pool_and_refuses_what_it_cannot_play(tables, nc2, layouts):
    settings = dict(task='free-play-v1', muggers=False, life=('fixed', 300))
    env = CpuBatch(nc2, layouts[:4], 4, **settings)
    assert env.world == 'village-world-v2' and CpuBatch(nc2, layouts[:4], 4, world='village-world-v2', **settings).world == env.world
    v1_pool = [W.generate(s, tables) for s in W.split_seeds('development', 2)]
    assert CpuBatch(tables, v1_pool, 2, **settings).world == 'village-world-v1'
    for bad, why in ((dict(settings, muggers=True), 'no muggers'), (dict(settings, world='village-world-v1'), 'holds')):
        with pytest.raises(ValueError, match=why):
            CpuBatch(nc2, layouts[:4], 4, **bad)
    with pytest.raises(ValueError, match='one village world'):
        CpuBatch(nc2, layouts[:2] + v1_pool, 4, **settings)
    with pytest.raises(ValueError, match='pick'):
        CpuBatch(tables, layouts[:4], 4, **settings)
    with pytest.raises(ValueError, match='world are free-play-v1'):
        CpuBatch(tables, v1_pool, 4, world='village-world-v1')
    state, obs = env.reset(3)
    rng = np.random.default_rng(0)
    for _ in range(120):
        action = np.asarray([rng.choice(np.flatnonzero(row)) for row in obs['mask']], np.int32)
        state, obs, reward, done, info = env.step(state, action)
        assert not info['unsupported'].any()


def test_the_soak_names_its_world_and_plays_v2_without_muggers(tmp_path, nc2):
    with pytest.raises(TypeError):
        free.soak(tmp_path / 'x', forms=(False,))  # no default world
    with pytest.raises(ValueError, match='no muggers'):
        free.soak(tmp_path / 'y', world='village-world-v2', forms=(False, True))
    manifest = free.soak(tmp_path / 'soak', world='village-world-v2', forms=(False,), xp_rate=1, ticks=400, first=7000,
                         records=2, key=KEY, controls=('wait', 'random', 'melee'), random_decisions=50, seed=1, processes=1,
                         chunk=2)
    rows = [json.loads(line) for line in (tmp_path / 'soak' / 'lives.jsonl').read_text().splitlines()]
    assert manifest['world'] == 'village-world-v2' and manifest['tables']['path'].endswith('draynor-nc2.npz')
    assert {r['world'] for r in rows} == {'village-world-v2'} and all(r['unsupported'] is None for r in rows)
    assert {r['layout'] for r in rows} == set(W.split_seeds('development', 2, 7000))
    life = free.Life(7000, KEY, 0, False, 1, 100, free._tables('village-world-v2'), world='village-world-v2')
    assert life.layout.world == 'village-world-v2' and free.Life(4000, 223, 0, False, 1, 100, free._tables('village-world-v1')).world \
        == 'village-world-v1'


def test_the_pick_archive_is_nc1_with_the_pick_rows_alone(nc2):
    """draynor-nc2.npz holds draynor-nc1.npz's tables and content and only the pick rows and texts beside them, read from
    the scripts (sim/build.py picks); no firemaking or cooking rows; and add_picks never writes over an archive."""
    from sim.build import add_picks, picks
    from sim.data import Tables
    nc1 = Tables.load(DATA)
    rows, texts = picks()
    assert rows == {'cabbage': dict(item='cabbage', respawn=70, scaled=False), 'potato': dict(item='potato', respawn=50, scaled=True),
                    'onion': dict(item='onion', respawn=50, scaled=True)}
    assert nc2.meta['content']['picks'] == rows and not {'firemaking', 'cooking'} & set(nc2.meta['content'])
    same = {k: v for k, v in nc2.meta['content'].items() if k != 'picks'}
    same['texts'] = {k: v for k, v in same['texts'].items() if k not in texts}
    assert json.dumps(same, sort_keys=True) == json.dumps(nc1.meta['content'], sort_keys=True)
    assert {k: nc2.meta['content']['texts'][k] for k in texts} == texts and nc2.meta['pins'] == nc1.meta['pins']
    assert nc2.meta['derived']['source'] == 'draynor-nc1.npz' and all(np.array_equal(nc2.arrays[k], nc1.arrays[k]) for k in nc1.arrays)
    for output in (DATA_NC2, DATA):
        with pytest.raises(ValueError):
            add_picks(output, DATA)


# ---- village-world-v1 bit-equal to its earlier code ------------------------------------------------------------------
def v1_digests(tables):
    """Digests of village-world-v1's outputs, in code that runs at an earlier commit as in this tree: compact-world-v1
    layouts (and two search and supply layouts), draw_village starts in both forms, a CpuBatch free-play run in both
    forms under keyed random legal actions, and the soak rows of six controls (less the fields this tree adds)."""
    import hashlib
    import json
    import numpy as np
    from sim import free, world as W
    from sim.combat import draw_village
    from sim.engine import Rng
    from sim.vec import CpuBatch
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    out, h = {}, hashlib.sha256()
    for spec, split, first, count in ((W.WORLD, 'train', 1000, 16), (W.WORLD, 'development', 4000, 16),
                                      (W.SEARCH_WORLD, 'train', 0, 2), (W.SUPPLY_WORLD, 'train', 0, 2)):
        for seed in W.split_seeds(split, count, first):
            layout = W.generate(seed, tables, spec)
            for k, v in sorted(layout.arrays().items()):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v).tobytes())
            h.update(W.world_of(layout)['version'].encode())
    out['layouts'] = h.hexdigest()
    pool = [W.generate(seed, tables) for seed in W.split_seeds('development', 8, 4000)]
    starts = [draw_village(223, 0, r, [pool[r]], tables, m, layout=0) for r in range(8) for m in (False, True)]
    out['starts'] = digest(json.dumps(starts, sort_keys=True, default=str))
    for muggers in (False, True):
        env, h = CpuBatch(tables, pool, 8, task='free-play-v1', muggers=muggers, life=('uniform', 300, 1200), xp_rate=1), hashlib.sha256()
        state, obs = env.reset(31)
        rng = Rng(31)
        for step in range(300):
            legal = [np.flatnonzero(obs['mask'][b]) for b in range(env.n)]
            action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(env.n)], np.int32)
            state, obs, reward, done, info = env.step(state, action)
            for k in sorted(obs):
                h.update(np.ascontiguousarray(obs[k]).tobytes())
            h.update(np.ascontiguousarray(reward).tobytes())
            h.update(np.ascontiguousarray(done).tobytes())
            for k in sorted(info):
                h.update(k.encode())
                for part in ([info[k][j] for j in sorted(info[k])] if isinstance(info[k], dict) else [info[k]]):
                    h.update(np.ascontiguousarray(part).tobytes())
        out[f'batch_muggers_{muggers}'] = h.hexdigest()
    rows = []
    for name, period in (('wait', None), ('random', None), ('melee', None), ('chickens', None), ('wood', None), ('rotation', 300)):
        for r in range(2):
            life = free.Life(4000, 223, r, False, 1, 1500, tables, 'development')
            actor = free.make(name, period)
            marks = free.PhaseMarks(actor) if isinstance(actor, free.Rotation) else None
            row = life.row(name, period, actor, free.play(life.task, actor, [4253, r], watch=marks), marks, 0)
            rows.append({k: v for k, v in row.items() if k not in ('seconds', 'world', 'picks', 'hunter')})
    out['rows'] = digest(json.dumps(rows, sort_keys=True, default=str))
    return out


# v1_digests at an earlier commit, run in an export of that commit over draynor-nc1.npz under the tests' name table
# (conftest.py)
V1_PINNED_DIGESTS = {
    'batch_muggers_False': 'ff7965d718a5afb7a4ea9f4a08050f62f439368682086c6dcd10f832fc2bec35',
    'batch_muggers_True': 'e95f30e3701dd63bd9becf8da49547f59398a915064de4cba6e8e2038db51ca5',
    'layouts': '4e4b93777802f965ada39c0e5dc9b3eca8acc7a29ee425bf3a05f9c88bbba46b',
    'rows': '4054ea68b738d0a997936983e2871ea5575307af73e1d12a9574242919ad9a32',
    'starts': '671daa60ca319ed0dead3a045b435fec86d6fc1b0db4212566c8f8be5a6993cb',
}


def test_village_world_v1_is_bit_equal_to_its_pinned_digests(tables):
    """v1_digests' source alone, compiled in a namespace that holds nothing of this module (as it ran in an export of
    the earlier commit), gives the digests pinned from that commit."""
    tree = ast.parse(Path(__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'v1_digests')
    space = {'__builtins__': __builtins__}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'v1_digests', 'exec'), space)
    assert set(space) == {'__builtins__', 'v1_digests'} and space['v1_digests'](tables) == V1_PINNED_DIGESTS


# ---- a hit in the cycle its npc life respawned (sim/audit.py XpCheck, sim/engine.py World.npc_respawn) ---------------
POOL_V2 = DATA_NC2.parents[1] / 'world' / 'train-128-20000-v2.npz'


@pytest.fixture(scope='module')
def pilot_pool(nc2):
    """The wider world's pilot pool (ignored logs/), on which the respawn-cycle hit was found."""
    if not POOL_V2.exists():
        pytest.skip('the pilot pool (logs/sim/world/train-128-20000-v2.npz) absent')
    return W.load_pool(POOL_V2)


def pilot_batch(nc2, pool, first=0):
    """CpuBatch(n=1) at world slot `first` with the pilot's settings."""
    layouts, derived = pool
    return CpuBatch(nc2, layouts, 1, derived=None if derived[0] is None else derived, first=first, task='free-play-v1',
                    xp_rate=1, muggers=False, life=('uniform', 6000, 36000))


def test_a_hit_in_the_cycle_its_npc_respawned_owes_no_xp(nc2, pilot_pool):
    """The hit by construction: the player one tile from a chicken's spawn, set on it (op 2) before it is removed,
    held in a chat modal until the cycle before the respawn. In the respawn cycle it hits the new life, whose xp_mult
    is still 0 ([ai_spawn,_] runs in the next cycle), so the engine gives no XP. That tick's 'npc_respawn' event tells
    XpCheck the hit owes none; with the event left out, the hit is owed and the check raises."""
    env = pilot_batch(nc2, pilot_pool)
    env.reset(79)
    world = env.tasks[0].world
    p = world.player
    world.cycle()
    world.cycle()  # the cycle at which the hit below draws damage above 0
    npc = next(n for n in world.npcs if nc2.npc_types[n.type]['debugname'] == 'chicken' and n.active)
    world._occupy(p, False)
    p.x, p.z, p.level = npc.start_x + 1, npc.start_z, npc.level
    world._occupy(p, True)
    mark, check = len(world.events), XpCheck(world.t, 1, p.name, p.stats)
    world.remove_npc(npc)
    p.target, p.target_op, p.target_type = npc, APNPC1 + 1, npc.type
    p.ap_range, p.ap_range_called = 10, False
    p.modal_state = set(p.modal_state) | {'chat'}
    while npc.lifecycle_tick > 1:
        world.cycle()
    p.modal_state = set(p.modal_state) - {'chat'}
    tick = world.tick
    world.cycle()
    events = world.events[mark:]
    assert [kind for t, kind, _ in events if t == tick] == ['npc_respawn', 'hit']
    respawn, hit = events[-2][2], events[-1][2]
    assert respawn == dict(npc=npc.nid, type=npc.type, life=npc.life) == {k: hit[k] for k in ('npc', 'type', 'life')}
    assert hit['source'] == p.name and hit['damage'] > 0 and npc.vars['xp_mult'] == 0
    assert not any(check.observe(events))
    with pytest.raises(AuditError, match='has no xp event'):
        XpCheck(world.t, 1, p.name, p.stats).observe([e for e in events if e[1] != 'npc_respawn'])


# The life the respawn-cycle hit was found in (14 Nov): world slot 295, key 81, episode 7 of the pilot's batch, and the
# policy's 319 actions; its last step raised before the rule (a hit at tick 5782 on npc 11, a chicken, in the cycle its
# life 18 respawned)
RESPAWN_LIFE = dict(slot=295, key=81, episode=7, actions=[
    27, 94, 87, 79, 68, 1, 91, 8, 2, 28, 20, 6, 5, 1, 10, 14, 14, 8, 1, 8, 1, 7, 13, 3, 10, 0, 9, 10, 7, 1, 1, 7,
    9, 2, 15, 10, 2, 3, 1, 7, 16, 0, 9, 14, 1, 6, 1, 14, 10, 8, 0, 15, 6, 2, 2, 61, 3, 3, 95, 7, 2, 0, 61, 2, 5,
    1, 2, 2, 44, 3, 0, 0, 6, 12, 25, 68, 3, 40, 5, 61, 38, 37, 5, 1, 42, 39, 4, 44, 6, 40, 43, 4, 62, 8, 24, 42,
    68, 3, 0, 78, 7, 1, 11, 0, 9, 3, 13, 7, 40, 14, 7, 1, 8, 13, 5, 12, 4, 11, 1, 1, 15, 0, 61, 7, 12, 8, 6, 0, 0,
    37, 13, 13, 9, 15, 1, 4, 13, 8, 16, 3, 11, 0, 13, 13, 9, 1, 10, 4, 1, 1, 3, 0, 78, 0, 2, 15, 39, 38, 2, 5, 4,
    5, 4, 8, 61, 37, 40, 39, 5, 38, 4, 3, 38, 13, 61, 38, 40, 2, 61, 38, 6, 40, 39, 3, 37, 40, 4, 1, 75, 91, 9, 1,
    12, 73, 1, 9, 10, 1, 6, 10, 5, 37, 11, 90, 9, 0, 8, 2, 1, 72, 15, 13, 14, 2, 75, 0, 0, 77, 1, 0, 15, 16, 18,
    5, 5, 6, 0, 13, 1, 15, 13, 1, 1, 15, 12, 16, 14, 2, 14, 16, 9, 1, 16, 4, 1, 14, 0, 73, 1, 0, 10, 1, 5, 4, 6,
    0, 1, 38, 3, 0, 0, 1, 2, 3, 4, 0, 7, 0, 10, 2, 4, 0, 4, 12, 5, 16, 1, 2, 4, 5, 0, 7, 76, 9, 0, 1, 76, 3, 5,
    76, 5, 1, 3, 2, 8, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61
])


def test_the_respawn_cycle_life_plays_its_319_actions(nc2, pilot_pool):
    env = pilot_batch(nc2, pilot_pool, first=RESPAWN_LIFE['slot'])
    env.reset(RESPAWN_LIFE['key'])
    env.episodes[0] = RESPAWN_LIFE['episode']
    env._episode(0)  # world 0 rebuilt at that episode, as the harness's repro does (sim/native_harness.py start_episode)
    state, world = env._state(), env.tasks[0].world
    mark = len(world.events)
    for action in RESPAWN_LIFE['actions']:
        state, *_ = env.step(state, np.asarray([action], np.int32))
    assert [e for e in world.events[mark:] if e[0] == 5782] == [
        (5782, 'npc_respawn', dict(npc=11, type=41, life=18)),
        (5782, 'hit', dict(npc=11, type=41, life=18, source=world.player.name, damage=1, hp=3, style=1))]
