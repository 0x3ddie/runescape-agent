from hypothesis import given, strategies as st

from sim.engine import Rng, _links


def test_a_request_added_while_the_last_runs_waits_for_the_next_pass():
    items, seen = ['a'], []
    for item in _links(items):
        seen.append(item)
        if item == 'a':
            items.append('b')
    assert seen == ['a'] and items == ['a', 'b']


def test_a_request_added_before_the_last_runs_is_visited_this_pass():
    items, seen = ['a', 'b'], []
    for item in _links(items):
        seen.append(item)
        if item == 'a':
            items.append('c')
    assert seen == ['a', 'b', 'c']


def test_a_successor_unlinked_meanwhile_is_still_visited_and_ends_the_pass():
    items, seen = ['a', 'b', 'c'], []
    for item in _links(items):
        seen.append(item)
        if item == 'a':
            items.remove('b')
    assert seen == ['a', 'b']


@given(st.integers(0, 2 ** 32 - 1), st.lists(st.integers(0, 10 ** 6), min_size=1, max_size=5), st.integers(0, 5))
def test_a_shared_prefix_draws_what_the_full_key_draws(seed, keys, split):
    rng = Rng(seed)
    split = min(split, len(keys))
    assert rng.draw(rng.prefix(*keys[:split]), *keys[split:]) == rng(*keys)


def test_doors_and_gates_swing_and_come_back_to_the_boot_map(tables):
    """Each door and gate of the supported area, opened by its script and its next stage
    closed again, changes the collision; 500 ticks later the boot map is back, bit for bit."""
    from sim.engine import SCRIPTS, Player, World
    world = World(tables)
    player = Player(world, 'tester')
    support = tables.meta['support']
    opened = 0
    for loc in list(world.locs):
        script = tables.loc_types[loc.type]['triggers'].get('OPLOC1')
        inside = support['x0'] <= loc.x <= support['x1'] and support['z0'] <= loc.z <= support['z1']
        if not (inside and loc.active and script in ('[oploc1,_door_closed]', '[oploc1,_gate_main_closed]')):
            continue
        before = set(world.loc_events)
        list(SCRIPTS[script](world, player, loc))
        opened += 1
        added = [other for other in world.loc_events if other not in before and other.active]
        assert world.changed and not loc.active and added
        stage = added[0]  # the door's, or the main gate's, next stage
        list(SCRIPTS[tables.loc_types[stage.type]['triggers']['OPLOC1']](world, player, stage))
        assert not stage.active
    assert opened >= 5
    for _ in range(501):
        world.process_zones()
    assert not world.changed and world._flags == tables.collision.tolist()
    assert all(loc.active and not loc.is_changed() for loc in world.locs) and not world.loc_events


def test_stairs_climb_after_arriving_and_the_floor_left_behind_sleeps(tables):
    """[oploc1,loc_1722] and [oploc1,loc_1723]: a player who stepped this cycle waits a tick
    (p_arrivedelay), then p_telejump puts it on the case's tile of the other floor, where it
    takes no step that cycle; npcs of the floor it left turn dormant, those of its own floor run."""
    from sim.engine import APLOC1, Player, World
    world = World(tables)
    up = next(loc for loc in world.locs if (loc.level, loc.x, loc.z) == (0, 3090, 3251))
    assert world.t.stairs[('[oploc1,loc_1722]', 0, 3090, 3251)] == (3093, 3251, 1)
    p = Player(world, 'tester')
    p.x, p.z = 3088, 3251  # a step west of the staircase's reach tile
    world.pending_login = (p, lambda world, p: None)
    world.cycle()
    p.queue_waypoints([(3089, 3251)])
    p.set_interaction(up, APLOC1)
    world.cycle()  # the step
    world.cycle()  # the op, which waits a tick
    assert (p.level, p.x, p.z) == (0, 3089, 3251) and p.delayed
    p.queue_waypoints([(3089, 3255)])  # a walk issued meanwhile is kept, and waits a cycle
    world.cycle()
    assert (p.level, p.x, p.z) == (1, 3093, 3251) and p.path == [(3089, 3255)]
    assert world.occupied.get((1, 3093, 3251)) == 1 and not world.occupied.get((0, 3089, 3251))
    p.clear_waypoints()
    ground = next(npc for npc in world.npcs if npc.level == 0 and npc.nid == 4282)
    woman = next(npc for npc in world.npcs if npc.level == 1 and npc.nid == 4310)
    before = ground.regen_clock, woman.regen_clock
    for _ in range(5):
        world.cycle()
    assert ground.regen_clock == before[0] and woman.regen_clock == before[1] + 5
    # The arrival tile reaches the staircase down; standing still, no wait.
    down = next(loc for loc in world.locs if (loc.level, loc.x, loc.z) == (1, 3091, 3251))
    p.set_interaction(down, APLOC1)
    world.cycle()
    assert (p.level, p.x, p.z) == (0, 3089, 3251) and not p.delayed
