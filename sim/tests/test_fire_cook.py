"""Firemaking and cooking in the reference: the draynor-v3 archive, the
light and cook scripts as the native source runs them (vendor/rs-sdk server/content/scripts skill_firemaking/scripts/
firemaking.rs2 and skill_cooking/scripts/cooking.rs2, with the engine handlers they call), through the agent's USE,
HELD and ACT rows, and the declared exclusions, each of which stops the run as Unsupported."""
import numpy as np
import pytest

from baseline import config
from sim import build, engine, world as W
from sim.agent import A_ACT0, A_HELD0, A_USE0, Agent
from sim.data import Tables
from sim.audit import XpCheck
from sim.engine import COOKING, FIREMAKING, Obj, Player, Unsupported, World, xp_for
from sim.task import WORLD_START_TICK

FIRE, RANGE, CASTLE_RANGE = 2732, 2728, 114  # Fire, Range and Cooking range: found by name (their debugnames are loc_N)
CFG = dict(config.TASK_CFG, paceJitterTicks=0)  # a decision is 2 ticks


@pytest.fixture(scope='module')
def v3(tables):
    if not build.DATA_V3.exists():
        pytest.skip('simulator tables logs/sim/data/draynor-v3.npz absent (docs/RUNNING.md says how to build them)')
    v3 = Tables.load(build.DATA_V3)
    assert [v3.loc_types[k]['name'] for k in (FIRE, RANGE, CASTLE_RANGE)] == ['Fire', 'Range', 'Cooking range']
    return v3


@pytest.fixture(scope='module')
def site(v3):
    layout = W.generate(W.split_seeds('development', 1)[0], v3)
    return layout, W.LayoutTables(v3, layout)


@pytest.fixture(scope='module')
def draynor(v3):
    return World(v3)


def begin(site, kit, levels=None, locs=(), blocked=(), seed=1):
    """An agent on open ground: every tile within 2 is walkable and holds no loc, and no wall stands beside the tile or
    its neighbours. `locs` are (type, dx, dz) added first, relative to that tile, each for good; `blocked` are offsets a
    bush fills. With `locs` the agent starts on a tile that operates the first of them."""
    layout, lt = site
    world = World(lt, seed=seed, start_tick=WORLD_START_TICK)

    def open_ground(x, z):
        near = [(x + dx, z + dz) for dx in range(-2, 3) for dz in range(-2, 3)]
        return all(lt.source(0, a, b) >= 0 and not world.active_locs_at(0, a, b) for a, b in near) and \
            all(world.can_travel(0, a, b, sx, sz, 1, 0, 0) for a, b in near if max(abs(a - x), abs(b - z)) <= 1
                for sx, sz in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    x, z = next(xz for xz in ((W.ORIGIN[0] + int(x), W.ORIGIN[1] + int(z)) for x, z in np.argwhere(layout.start)) if open_ground(*xz))
    bush = next(k for k, t in lt.loc_types.items() if t['debugname'] == 'bush1')
    added = [world.loc_add(0, x + dx, z + dz, type_id, 0, engine.CENTREPIECE_STRAIGHT, 0)
             for type_id, dx, dz in [*locs, *((bush, dx, dz) for dx, dz in blocked)]]
    tile = next(xy for xy in world.loc_reaching(added[0]) if lt.source(0, *xy) >= 0) if locs else (x, z)
    agent = Agent(world)
    agent.start(0, CFG, tile, kit, levels or {})
    return world, agent, agent.player


def fix(world, **outcomes):
    """Fix the engine's draws of the named kinds (World.decide): one value, or a list taken in turn. Returns the list of
    (kind, tick, the draw the world made itself)."""
    seen = []

    def control(kind, default, obj, tick):
        if kind not in outcomes:
            return default
        seen.append((kind, tick, default))
        return outcomes[kind].pop(0) if isinstance(outcomes[kind], list) else outcomes[kind]
    world.control = control
    return seen


def use(agent, item, on):
    """A use's two decisions: USE on the inventory item `item` (a debugname), then the HELD row of the item `on` or the
    ACT row of the loc named `on`. Returns the tick the second one was sent at."""
    t = agent.world.t
    obs = agent.step(A_USE0 + next(k for k, s in enumerate(agent.last_observation['invSel']) if s['id'] == t.obj_id(item)))
    tick = agent.world.tick
    if on in t.obj_ids:
        agent.step(A_HELD0 + next(k for k, s in enumerate(obs['heldSel']) if s['id'] == t.obj_id(on)))
    else:
        agent.step(A_ACT0 + next(k for k, s in enumerate(obs['actSel']) if s['name'] == on))
    return tick


def to_tick(agent, tick):
    agent.wait_ticks(tick - agent.world.tick)


def said(world, since):
    return [(tick - since, text) for tick, text in world.chat_log if tick >= since]


def fires(world):
    return [(loc.x, loc.z) for loc in world.loc_events if loc.type == FIRE and loc.active]


def ground(world, name):
    return [(o.x, o.z) for o in world.objs if o.is_valid() and o.type == world.t.obj_id(name)]


# ---- step 1: the archive
def test_v3_holds_the_current_archives_tables_and_adds_the_firemaking_and_cooking_rows(tables, v3):
    """draynor-v3.npz is draynor-nc1.npz's extraction with full pins: every extracted table and array is equal, so the
    tables of every type the current archive holds are, and so is every content row it holds. It adds the texts the
    two skills can show, the bank zones and the cooking rows of the raw food the compact world produces."""
    assert v3.meta['pins'] == tables.meta['pins'] == build.pins()
    for kind in ('loc_types', 'npc_types', 'obj_types'):
        assert all(getattr(v3, kind).get(k) == row for k, row in getattr(tables, kind).items()), kind
    assert all(v3.meta[k] == value for k, value in tables.meta.items() if k not in ('content', 'pins'))
    assert set(v3.arrays) == set(tables.arrays) and all(np.array_equal(v3.arrays[k], a) for k, a in tables.arrays.items())
    old, new = tables.meta['content'], v3.meta['content']
    assert all(new[k] == value for k, value in old.items() if k != 'texts') and old['texts'].items() <= new['texts'].items()
    assert set(new) - set(old) == {'firemaking', 'cooking'}
    assert set(new['texts']) - set(old['texts']) == {'fm_level', 'fm_here', 'fm_tinderbox', 'fm_attempt', 'fm_catch', 'cook_cant'}
    assert [3088, 3240, 0, 3097, 3246, 0] in new['firemaking']['bank_zones']  # Draynor's bank (bank_zones.dbrow:11)
    rows = new['cooking']['rows']
    assert set(rows) == {'raw_shrimp', 'raw_anchovies', 'raw_chicken'}
    assert all((row['level'], row['exp'], row['chance']) == (1, 300, [128, 512]) for row in rows.values())
    assert (rows['raw_shrimp']['cooked'], rows['raw_anchovies']['burnt'], rows['raw_chicken']['cooked']) == \
        ('shrimp', 'burntfish1', 'cooked_chicken')
    assert {'raw_sardine', 'raw_beef', 'tbwt_jogre_bones'} <= set(new['cooking']['uncooked'])


def test_an_archive_without_the_rows_keeps_both_skills_unsupported(tables):
    """The ports need draynor-v3's rows: over the current archive their triggers name no port, as before this port, so
    no existing world or pack changes."""
    layout = W.generate(W.split_seeds('development', 1)[0], tables)
    site = layout, W.LayoutTables(tables, layout)
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1)])
    with pytest.raises(Unsupported, match=r'held script \[opheldu,tinderbox\]'):
        use(agent, 'tinderbox', 'logs')
    world, agent, p = begin(site, [('raw_shrimp', 1)], locs=[(FIRE, 1, 0)])
    with pytest.raises(Unsupported, match=r'engine script \[oplocu,_cooking_fire\]'):
        use(agent, 'raw_shrimp', 'Fire')


# ---- step 2: firemaking
@pytest.mark.parametrize('item, on', [('tinderbox', 'logs'), ('logs', 'tinderbox')])
def test_a_tinderbox_and_logs_light_a_fire_on_the_scripts_cycle(v3, site, item, on):
    """light_logs_inv (firemaking.rs2:79-117) in either order of use (OpHeldUHandler.ts:93-101): the logs drop under
    the player for 200 ticks; the attempt's message; a first roll 3 ticks on and, after a failure, the next 4 ticks
    later. On a success the player moves west and waits a tick (push_player, 160-163), then the logs go, 40 xp at the
    world's 25x, the message and a Fire on the logs' tile (firemaking_success)."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1)], dict(firemaking=1))
    rolls = fix(world, light=[False, True], burn=100)
    x, z = p.x, p.z
    at = use(agent, item, on)
    log = next(o for o in world.objs if o.type == v3.obj_id('logs'))
    assert not p.inv.total(v3.obj_id('logs')) and (log.x, log.z, log.receiver) == (x, z, p.name)
    assert log.ticks == 200 - 2 and p.target is log and said(world, at) == [(0, 'You attempt to light the logs.')]
    assert [e[1:] for e in world.events if e[0] == at] == [('remove', dict(obj=v3.obj_id('logs'), count=1, why='light'))]
    to_tick(agent, at + 8)  # the cycle of the second roll has run
    assert [(kind, tick - at) for kind, tick, _ in rolls] == [('light', 3), ('light', 7)]
    assert (p.x, p.z, p.delayed) == (x - 1, z, True) and not fires(world) and ground(world, 'logs') == [(x, z)]
    agent.wait_ticks(1)
    assert fires(world) == [(x, z)] and not ground(world, 'logs') and not p.delayed and p.target is None
    assert said(world, at)[1] == (8, 'The fire catches and the logs begin to burn.')
    assert (p.stats[FIREMAKING], p.base_levels[FIREMAKING]) == (400 * 25, 10)
    start = [xp_for(1)] * len(p.stats)
    start[engine.HITPOINTS] = xp_for(10)
    assert XpCheck(v3, 25, p.name, start).observe(world.events) == [400 * 25 * (k == FIREMAKING) for k in range(len(start))]


def test_a_light_begun_within_4_ticks_of_a_success_shows_no_message_and_rolls_on_that_tick(v3, site):
    """A success sets %action_delay to its tick + 4 (firemaking.rs2:62, in the op 4 loop, where both rolls here run;
    106 is the same for a roll on the use's own tick), so the next light finds it ahead of map_clock: no stop, no
    message, and its roll comes when the clock reaches it."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 2)], dict(firemaking=99))
    rolls = fix(world, light=True, burn=100)
    x, z = p.x, p.z
    at = use(agent, 'tinderbox', 'logs')  # rolls at +3, the fire at +4
    agent.step(0)
    assert use(agent, 'tinderbox', 'logs') == at + 6
    to_tick(agent, at + 9)
    assert [(kind, tick - at) for kind, tick, _ in rolls] == [('light', 3), ('burn', 4), ('light', 7), ('burn', 8)]
    assert said(world, at) == [(0, 'You attempt to light the logs.'), (4, 'The fire catches and the logs begin to burn.'),
                               (8, 'The fire catches and the logs begin to burn.')]
    assert fires(world) == [(x, z), (x - 1, z)] and (p.x, p.z) == (x - 2, z)


@pytest.mark.parametrize('blocked, moved, wait', [
    ((), (-1, 0), 1), (((-1, 0),), (1, 0), 1), (((-1, 0), (1, 0)), (0, -1), 1), (((-1, 0), (1, 0), (0, -1)), (0, 1), 1),
    (((-1, 0), (1, 0), (0, -1), (0, 1)), (0, 0), 0)])
def test_the_fire_moves_the_player_to_the_first_open_side_or_comes_at_once(site, blocked, moved, wait):
    """push_player (firemaking.rs2:146-175): west, east, south, north, each only with a line of walk; a move waits a
    tick (p_delay(0)), and a player with no open side stays on the fire, which then comes on the roll's own tick."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1)], dict(firemaking=99), blocked=blocked)
    fix(world, light=True, burn=100)
    x, z = p.x, p.z
    at = use(agent, 'tinderbox', 'logs')
    to_tick(agent, at + 5)
    assert (p.x - x, p.z - z) == moved and fires(world) == [(x, z)]
    assert said(world, at)[1] == (3 + wait, 'The fire catches and the logs begin to burn.')


def test_a_light_is_refused_below_the_logs_level_and_on_a_fire_and_other_items_get_the_default_message(v3, site):
    """The level comes from the log (oak_logs levelrequire 15, firemaking.obj:198) and is checked first; then
    area_allow_loc_add (firemaking.rs2:134-144) refuses a tile a loc of an active type stands on. The logs stay held.
    Bark and an item of no branch get ~displaymessage(^dm_default) (firemaking.rs2:20-21, 27)."""
    oak = v3.obj_id('oak_logs')
    world, agent, p = begin(site, [('tinderbox', 1), ('hollow_bark', 1), ('bronze_pickaxe', 1)])
    at = use(agent, 'hollow_bark', 'tinderbox')
    assert use(agent, 'tinderbox', 'bronze_pickaxe') == at + 4
    assert said(world, at) == [(0, 'Nothing interesting happens.'), (4, 'Nothing interesting happens.')] and not world.objs
    world, agent, p = begin(site, [('tinderbox', 1), ('oak_logs', 1)], dict(firemaking=14))
    at = use(agent, 'tinderbox', 'oak_logs')
    assert said(world, at) == [(0, 'You need a Firemaking level of 15 to burn Oak logs.')] and p.inv.total(oak) == 1
    world, agent, p = begin(site, [('tinderbox', 1), ('oak_logs', 1)], dict(firemaking=15))
    at = use(agent, 'tinderbox', 'oak_logs')
    assert said(world, at) == [(0, 'You attempt to light the logs.')] and not p.inv.total(oak)
    world, agent, p = begin(site, [('tinderbox', 1), ('oak_logs', 1)], dict(firemaking=15))
    world.loc_add(0, p.x, p.z, FIRE, 1, engine.CENTREPIECE_STRAIGHT, 0)
    at = use(agent, 'tinderbox', 'oak_logs')
    assert said(world, at) == [(0, "You can't light a fire here.")] and p.inv.total(oak) == 1 and p.target is None


def test_no_fire_is_lit_by_a_player_inside_a_bank_zone(v3, draynor):
    """area_allow_loc_add's zone check reads the player's coord, not the tile's (firemaking.rs2:141). Of two tiles that
    nothing covers or blocks, one in Draynor's bank and one east of it, each is refused to a player in the bank and
    allowed to a player outside it."""
    world, logs = draynor, v3.obj_id('logs')
    x0, z0, _, x1, z1, _ = next(zone for zone in v3.meta['content']['firemaking']['bank_zones'] if zone[:2] == [3088, 3240])
    clear = lambda x, z: not world.map_locaddunsafe(0, x, z) and not world.flags(0, x, z, npc_bits=False) & engine.WALK_BLOCKED
    bank = next((x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1) if clear(x, z))
    street = next((x, z) for x in range(x1 + 2, x1 + 10) for z in range(z0, z1 + 1) if clear(x, z))
    p = Player(world, 'tester')
    p.inv.add(v3, v3.obj_id('tinderbox'))
    p.x, p.z = bank
    assert engine.light_refused(world, p, logs, 0, *bank) and engine.light_refused(world, p, logs, 0, *street)
    assert [text for _, text in world.chat_log[-2:]] == ["You can't light a fire here."] * 2
    p.x, p.z = street
    assert not engine.light_refused(world, p, logs, 0, *bank) and not engine.light_refused(world, p, logs, 0, *street)


def test_an_eat_ends_a_light_and_leaves_the_logs_on_the_ground(v3, site):
    """held-v2 (OpHeldHandler.ts:54-56): the Eat clears the interaction, and the light goes on only by setting op 4
    again each cycle (firemaking.rs2:77), so no roll comes after it."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1), ('shrimp', 1)], dict(firemaking=1))
    rolls = fix(world, light=False)
    x, z = p.x, p.z
    at = use(agent, 'tinderbox', 'logs')
    agent.step(0)  # the roll at +3 has run
    agent.step(A_HELD0 + next(k for k, s in enumerate(agent.last_observation['heldSel']) if s['option'] == 'Eat'))
    assert p.target is None and not p.inv.total(v3.obj_id('shrimp'))
    agent.wait_ticks(12)
    assert [tick - at for _, tick, _ in rolls] == [3] and not fires(world) and ground(world, 'logs') == [(x, z)]


def test_a_fire_burns_for_its_count_and_leaves_ashes_a_cycle_after_it_goes(v3, site):
    """firemaking_success (firemaking.rs2:126-132): loc_add for 100 + random(100) ticks, world_delay(that - 2), then
    obj_addall(ashes, 100). The loc's count runs out in a zone phase (the fire is there after 99 cycles of a count of
    100, the lighting cycle among them); the world queue runs the script on in the next cycle's first phase
    (World.enqueueScript's delay + 1, World.processWorld), and the ashes last 99 cycles after theirs."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1)], dict(firemaking=99))
    fix(world, light=True, burn=100)
    at = use(agent, 'tinderbox', 'logs') + 4  # the fire's cycle
    to_tick(agent, at)
    seen = []
    for _ in range(202):
        agent.wait_ticks(1)
        seen.append((bool(fires(world)), bool(ground(world, 'ashes'))))
    assert seen == [(True, False)] * 99 + [(False, False)] + [(False, True)] * 99 + [(False, False)] * 3
    for seed in range(2, 8):  # the world's own draw of the count
        world, agent, p = begin(site, [('tinderbox', 1), ('logs', 1)], dict(firemaking=99), seed=seed)
        to_tick(agent, use(agent, 'tinderbox', 'logs') + 5)
        assert 100 <= next(loc for loc in world.loc_events if loc.type == FIRE).lifecycle_tick + 1 <= 199


# ---- step 2: cooking
@pytest.mark.parametrize('raw, source, cooked, product, lines', [
    ('raw_shrimp', FIRE, True, 'shrimp', ('You cook the shrimps on the fire...', 'The shrimps are now nicely cooked.')),
    ('raw_shrimp', RANGE, False, 'burntfish1', ('You put the shrimps on the stove.', 'You accidentally burn the shrimps.')),
    ('raw_anchovies', RANGE, True, 'anchovies', ('You put the anchovies on the stove.', 'The anchovies are now nicely cooked.')),
    ('raw_chicken', FIRE, False, 'burnt_chicken', ('You cook the chicken on the fire...', 'You accidentally burn the chicken.')),
    ('raw_chicken', RANGE, True, 'cooked_chicken', ('You cook the chicken on the stove...', 'The chicken is now nicely cooked.'))])
def test_raw_food_cooks_one_item_a_use_two_ticks_after_its_message(v3, site, raw, source, cooked, product, lines):
    """attempt_cook_item and cook_item (cooking.rs2:17-195) on a fire and on a range: the roll at the use, the source's
    cook message, p_delay(1) (PlayerOps.ts:376-380: 2 ticks), then one raw item goes and the cooked or burnt one comes
    with its message; 30 xp at 25x only for the cooked one, an 'xp' event whose cause is the cooked item's production."""
    world, agent, p = begin(site, [(raw, 2)], dict(cooking=1), locs=[(source, 1, 0)])
    fix(world, cook=cooked)
    at = use(agent, raw, v3.loc_types[source]['name'])
    assert said(world, at)[:2] == [(0, lines[0]), (2, lines[1])]
    assert (p.inv.total(v3.obj_id(raw)), p.inv.total(v3.obj_id(product))) == (1, 1)
    assert p.stats[COOKING] == (300 * 25 if cooked else 0)
    xp = [('xp', dict(stat=COOKING, xp=300, before=0, after=300 * 25, cause='produce', obj=v3.obj_id(product),
                      src=v3.obj_id(raw)))] if cooked else []
    assert [e[1:] for e in world.events if e[0] == at + 2] == [
        ('remove', dict(obj=v3.obj_id(raw), count=1, why='cook')),
        ('produce', dict(obj=v3.obj_id(product), count=1, script='[label,cook_item]')), *xp]
    start = [xp_for(1)] * len(p.stats)
    start[engine.HITPOINTS] = xp_for(10)
    assert XpCheck(v3, 25, p.name, start).observe(world.events)[COOKING] == p.stats[COOKING]  # the cook row's amount


def test_a_cook_waits_a_tick_after_the_walk_and_an_ordinary_item_gets_the_mesbox(v3, site):
    """p_arrivedelay (PlayerOps.ts:359-367): the op runs the cycle after the last step, when lastMovement still names
    that cycle, so the script goes on a tick later. An item with no cooking row gets the script's mesbox and stays."""
    world, agent, p = begin(site, [('raw_shrimp', 1), ('bronze_axe', 1)], dict(cooking=99))
    fix(world, cook=True)
    x = p.x
    world.loc_add(0, x - 2, p.z, FIRE, 1, engine.CENTREPIECE_STRAIGHT, 0)
    use(agent, 'raw_shrimp', 'Fire')
    step = p.last_movement - 1  # the cycle of the walk's last step
    assert p.x == x - 1 and said(world, step) == [(2, 'You cook the shrimps on the fire...'), (4, 'The shrimps are now nicely cooked.')]
    use(agent, 'bronze_axe', 'Fire')
    assert agent.last_observation['feedback']['dialog']['text'] == "You can't cook that."
    assert p.inv.total(v3.obj_id('bronze_axe')) == 1


def test_light_and_cook_rolls_follow_the_scripts_chances(v3, site):
    """stat_random(firemaking, 64, 512) (firemaking.rs2:61, 105) and the rows' 128, 512 (cooking_generic.dbrow:221) at
    level 1: 65 and 129 of 256 (PlayerOps.ts:582-590). The world's own draws, 200 light rolls and 190 cook rolls, lie
    inside the 99% binomial interval of each chance, which leaves out the other's."""
    world, agent, p = begin(site, [('tinderbox', 1), ('logs', 4)], dict(firemaking=1))
    rolls = fix(world, light=False)  # the draw is recorded, then every roll fails: the logs lie until they are gone
    for _ in range(4):
        use(agent, 'tinderbox', 'logs')
        agent.wait_ticks(200)
    lit = [draw for _, _, draw in rolls]
    assert len(lit) == 200 and abs(sum(lit) - 200 * 65 / 256) < 2.58 * (200 * 65 * 191) ** .5 / 256  # 50 a log
    world, agent, p = begin(site, [('raw_shrimp', 1)], dict(cooking=1), locs=[(FIRE, 1, 0)])
    rolls = fix(world, cook=False)
    raw, burnt = v3.obj_id('raw_shrimp'), v3.obj_id('burntfish1')
    fire = next(loc for locs in world.loc_at.values() for loc in locs if loc.type == FIRE)
    for _ in range(190):
        p.last_useitem = raw
        p.set_interaction(fire, engine.APLOCU)
        agent.wait_ticks(3)
        assert p.inv.remove(burnt, 1) == 1 and p.inv.add(v3, raw) == 1
    cooked = [draw for _, _, draw in rolls]
    assert len(cooked) == 190 and abs(sum(cooked) - 190 * 129 / 256) < 2.58 * (190 * 129 * 127) ** .5 / 256


# ---- step 3: the declared exclusions
@pytest.mark.parametrize('item', ['newbielogs', 'unlitarrow', 'tbwt_jogre_bones', 'viking_firecracker', 'unlit_candle'])
def test_the_tinderboxs_other_branches_stop_the_run(site, item):
    """firemaking.rs2:14-27: tutorial logs, an unlit arrow, jogre bones, the firecracker and a light source."""
    world, agent, p = begin(site, [('tinderbox', 1), (item, 1)])
    with pytest.raises(Unsupported, match='a tinderbox used with'):
        use(agent, item, 'tinderbox')


@pytest.mark.parametrize('item', ['unlitarrow', 'tbwt_jogre_bones_marinated_in_karambwanji', 'tbwt_raw_karambwan',
                                  'tbwt_jogre_bones', 'eadgar_troll_thistle', 'raw_sardine'])
def test_what_the_cooking_scripts_treat_apart_stops_the_run(site, item):
    """cooking.rs2:9-10, 21-51: an unlit arrow on a fire, the jogre bones and the karambwan; and a raw item whose row
    the archive does not hold (plain jogre bones, the troll thistle, a sardine) is never told "You can't cook that."."""
    world, agent, p = begin(site, [(item, 1)], locs=[(FIRE, 1, 0)])
    with pytest.raises(Unsupported, match='unlit arrow on a fire' if item == 'unlitarrow' else f'cooking {item}'):
        use(agent, item, 'Fire')


def test_a_random_event_the_castle_range_and_a_fremennik_fire_stop_the_run(v3, site, monkeypatch):
    """afk_event (cooking.rs2:85-88), [oplocu,cooksquestrange] (quest_cook.rs2:93-101) and the viking fires
    (cooking.rs2:3-8), whose types the archive does not hold: a Fire stands in for one by its type's name."""
    world, agent, p = begin(site, [('raw_shrimp', 1)], locs=[(FIRE, 1, 0)])
    p.afk_ready = True
    with pytest.raises(Unsupported, match='random event'):
        use(agent, 'raw_shrimp', 'Fire')
    world, agent, p = begin(site, [('raw_shrimp', 1)], locs=[(CASTLE_RANGE, 1, 0)])
    with pytest.raises(Unsupported, match=r'engine script \[oplocu,cooksquestrange\]'):
        use(agent, 'raw_shrimp', 'Cooking range')
    world, agent, p = begin(site, [('raw_shrimp', 1)], locs=[(FIRE, 1, 0)])
    monkeypatch.setitem(v3.loc_types[FIRE], 'debugname', 'viking_house_fire')
    with pytest.raises(Unsupported, match='Fremennik'):
        use(agent, 'raw_shrimp', 'Fire')


def test_tutorial_island_the_duel_arena_and_a_fire_outside_the_data_area_stop_the_run(v3, draynor):
    """in_tutorial_island (cooking.rs2:11-12; tutorial/scripts/util.rs2:1-12), push_player's duel-arena case
    (firemaking.rs2:148-156; duelarena.dbrow:1-11) and firemaking_success's Fire on a tile outside the data area
    (World.loc_add), each on the script itself: no player can stand in those places in a simulated world."""
    world, p = draynor, Player(draynor, 'tester')
    p.x, p.z, p.last_useitem = 3100, 3120, v3.obj_id('raw_shrimp')  # inside the data area and the island's zone
    fireplace = next(loc for loc in world.locs if world.t.loc_types[loc.type]['name'] == 'Fireplace')
    with pytest.raises(Unsupported, match='Tutorial Island'):
        next(engine.cook_on_fire(world, p, fireplace))
    with pytest.raises(Unsupported, match='duel arena'):
        next(engine.push_player(world, p, Obj(v3.obj_id('logs'), 1, 3340, 3250, static=False)))
    logs = Obj(v3.obj_id('logs'), 1, v3.area['x0'] - 1, v3.area['z0'], static=False)
    world.objs.append(logs)
    with pytest.raises(Unsupported, match='outside the data area'):
        next(engine.firemaking_success(world, p, logs))
