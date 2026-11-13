"""act-v3's client route (sim/paths.py client_search and client_route, a port of the 2004 client's tryMove) on the
shared cases in env-ts/core/clicks.fixture.json, which env-ts/core/clicks.test.ts plays through the TypeScript port and
the vendored client's own tryMove: scenes of random blocked tiles and walls, a short maze (its walls leave the rows
beside the scene's edge open, so its routes keep to that edge, at most 11 turn points), a long serpentine maze whose
routes pass the 25-turn cut, a corridor that sets the 3 x 3 fallback's 100-step limit, and a walled room with its door
closed and open; tile clicks with and without the 3 x 3 fallback, and the client's arrival tests for npcs,
centrepieces, walls and wall decoration. Then act-v3's dispatch in the Python reference:
a click refused while delayed or with no route, an ACT by the client's rule, an npc out of the player's list.

    PYTHONPATH=.:rl .venv-sim/bin/python sim/tests/test_clicks.py   # writes the fixture again from sim/paths.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from act_contract import CLICK_OFFSETS, CLICK_ROWS
from baseline import config
from sim import paths, world as W
from sim.agent import A_ACT0, Agent
from sim.engine import Loc, World
from sim.spec import START_KIT
from sim.task import WORLD_START_TICK

FIXTURE = Path(__file__).resolve().parents[2] / 'env-ts' / 'core' / 'clicks.fixture.json'
BASE = (3200, 3200)
W_N, W_E, W_S, W_W = 0x2, 0x8, 0x20, 0x80


def wall(grid, x, z, side):
    """A straight wall on the `side` edge of (x, z), as the client's addWall marks both tiles."""
    other = dict(N=((0, 1), W_N, W_S), E=((1, 0), W_E, W_W), S=((0, -1), W_S, W_N), W=((-1, 0), W_W, W_E))[side]
    (ox, oz), mine, theirs = other
    grid[x, z] |= mine
    if 0 <= x + ox < paths.SCENE and 0 <= z + oz < paths.SCENE:
        grid[x + ox, z + oz] |= theirs


def scenes():
    """name -> client collision of a 104 x 104 scene at BASE."""
    rng = np.random.default_rng(20261103)
    out = {}
    g = np.zeros((paths.SCENE, paths.SCENE), np.int64)
    g[rng.random(g.shape) < 0.12] = paths.C_GRND
    g[rng.random(g.shape) < 0.04] |= 0x100
    for x, z in zip(*np.nonzero(rng.random(g.shape) < 0.06)):
        wall(g, int(x), int(z), 'NESW'[int(rng.integers(4))])
    out['random'] = g
    m = np.zeros_like(g)
    for k, x in enumerate(range(4, 100, 3)):  # serpentine corridors: a wall of blocked tiles with a gap at alternate ends
        m[x, 2:101] = paths.C_GRND
        m[x, 97 if k % 2 == 0 else 6] = 0
    out['maze'] = m
    for name, closed in (('door-closed', True), ('door-open', False)):
        r = np.zeros_like(g)
        for x in range(40, 61):  # a gap in the north wall at x 41: with the east door closed, the way round
            wall(r, x, 40, 'S')
            if x != 41:
                wall(r, x, 60, 'N')
        for z in range(40, 61):
            wall(r, 40, z, 'W')
            if z != 50 or closed:
                wall(r, 60, z, 'E')
        out[name] = r
    # walls of blocked tiles across the whole inner scene, each with one gap at alternate ends, so a route from
    # the west edge to the east edge turns at every wall (far more than 25 turn points) ...
    s = np.zeros_like(g)
    for k, x in enumerate(range(4, 101, 3)):
        s[x, 1:paths.SCENE - 1] = paths.C_GRND
        s[x, paths.SCENE - 3 if k % 2 == 0 else 2] = 0
    out['maze-long'] = s
    # ... and a corridor along z 50 walled in on both sides, its tiles x - 1 steps from (1, 50): beside (101, 51) the
    # nearest is 99 steps away, beside (102, 51) 100, the client's tryNearest limit (fewer than 100)
    c = np.zeros_like(g)
    c[1:paths.SCENE - 1, 49] = c[1:paths.SCENE - 1, 51] = paths.C_GRND
    out['fallback-limit'] = c
    return out


def queries(name, grid, rng):
    """(source, dest, arrival as tryMove's (width, length, angle, locShape, forceapproach), nearest) of a scene."""
    walkable = [(int(x), int(z)) for x, z in np.argwhere((grid & 0x200100) == 0)
                if 1 <= x < paths.SCENE - 1 and 1 <= z < paths.SCENE - 1]
    pick = lambda: walkable[int(rng.integers(len(walkable)))]
    out = []
    if name == 'maze-long':   # fixed queries, no draw from rng: the earlier scenes' cases stay as recorded
        return [((2, 50), (102, 50), (0, 0, 0, 0, 0), False),   # to the far side: the 25 turn points nearest the source
                ((2, 50), (101, 60), (1, 1, 0, 0, 0), False),   # an npc there: reached by the arrival test, also cut
                ((2, 50), (100, 50), (0, 0, 0, 0, 0), True)]    # into the last wall: its 3 x 3 is reached, too far
    if name == 'fallback-limit':
        return [((1, 50), (101, 51), (0, 0, 0, 0, 0), True),    # the nearest of its 3 x 3 at 99 steps: the fallback
                ((1, 50), (102, 51), (0, 0, 0, 0, 0), True),    # at 100 steps: no route
                ((1, 50), (102, 51), (0, 0, 0, 0, 0), False)]   # and none without tryNearest
    if name.startswith('door'):
        for nearest in (True, False):
            out.append(((50, 50), (70, 50), (0, 0, 0, 0, 0), nearest))   # out through the east door, or round
            out.append(((70, 52), (50, 49), (1, 1, 0, 0, 0), nearest))   # in to an npc's tile
        out.append(((70, 50), (60, 50), (0, 0, 2, 1, 0), False))          # a wall (shape 0, angle 2) on the door's tile
    if name == 'random':  # clicks onto blocked tiles: the 3 x 3 fallback, and no route without it
        blocked = [(int(x), int(z)) for x, z in np.argwhere((grid & 0x200100) != 0) if 2 <= x < 102 and 2 <= z < 102]
        for k in range(8):
            sx, sz = pick()
            dx, dz = blocked[int(rng.integers(len(blocked)))]
            out.append(((sx, sz), (dx, dz), (0, 0, 0, 0, 0), k < 6))
    for _ in range(40 if name != 'maze' else 12):
        sx, sz = pick()
        if rng.random() < 0.5:
            dx, dz = sx + int(rng.integers(-16, 17)), sz + int(rng.integers(-16, 17))
        else:
            dx, dz = int(rng.integers(0, paths.SCENE)), int(rng.integers(0, paths.SCENE))
        kind = int(rng.integers(6))
        arrival = [(0, 0, 0, 0, 0), (1, 1, 0, 0, 0),
                   (int(rng.integers(1, 4)), int(rng.integers(1, 4)), 0, 0, int(rng.integers(16))),
                   (0, 0, int(rng.integers(4)), int(rng.choice([1, 3, 10])), 0),
                   (0, 0, int(rng.integers(4)), int(rng.choice([7, 8, 9])), 0),
                   (0, 0, int(rng.integers(4)), 13, 0)][kind]
        out.append(((sx, sz), (dx, dz), arrival, bool(kind == 0 and rng.random() < 0.6)))
    return out


def route_of(grid, source, dest, arrival, nearest):
    """sim/paths.py's answer in absolute coordinates: (waypoints, via) or None."""
    steps = paths.client_steps(grid)
    s = (BASE[0] + source[0], BASE[1] + source[1])
    tree = paths.client_search(steps, grid, BASE[0], BASE[1], *s, BASE)
    test = None if arrival == (0, 0, 0, 0, 0) else paths.client_arrival(*arrival)
    route = paths.client_route(tree, *s, (BASE[0] + dest[0], BASE[1] + dest[1]), test, nearest)
    return None if route is None else dict(waypoints=[list(p) for p in route], via=route.via)


def build():
    rng = np.random.default_rng(3)
    cases = dict(clicks=dict(offsets=[list(o) for o in CLICK_OFFSETS], rows=list(CLICK_ROWS)), base=list(BASE), scenes={},
                 queries=[])
    for name, grid in scenes().items():
        cases['scenes'][name] = [[int(x), int(z), int(grid[x, z])] for x, z in np.argwhere(grid != 0)]
        for source, dest, arrival, nearest in queries(name, grid, rng):
            cases['queries'].append(dict(scene=name, source=[BASE[0] + source[0], BASE[1] + source[1]],
                                         dest=[BASE[0] + dest[0], BASE[1] + dest[1]], arrival=list(arrival),
                                         nearest=nearest, route=route_of(grid, source, dest, arrival, nearest)))
    return cases


def grid_of(cells):
    grid = np.zeros((paths.SCENE, paths.SCENE), np.int64)
    for x, z, f in cells:
        grid[x, z] = f
    return grid


def test_the_shared_cases_cover_every_arrival_fallbacks_long_routes_and_the_door():
    cases = json.loads(FIXTURE.read_text())
    found = [q['route'] for q in cases['queries']]
    assert {r['via'] for r in found if r} == {'target', 'reach', 'fallback'} and any(r is None for r in found)
    assert any(r and len(r['waypoints']) == paths.MAX_WAYPOINTS for r in found)  # the 25-turn cut
    # the long maze's routes are cut at 25 turn points short of their end, and the fallback stops at 100 steps
    long = {(tuple(q['dest']), q['nearest']): q['route'] for q in cases['queries'] if q['scene'] == 'maze-long'}
    far, npc = long[((BASE[0] + 102, BASE[1] + 50), False)], long[((BASE[0] + 101, BASE[1] + 60), False)]
    assert len(far['waypoints']) == len(npc['waypoints']) == paths.MAX_WAYPOINTS and (far['via'], npc['via']) == ('target', 'reach')
    assert far['waypoints'][-1][0] < BASE[0] + 60 and npc['waypoints'][-1][0] < BASE[0] + 60
    assert long[((BASE[0] + 100, BASE[1] + 50), True)] is None
    limit = [q['route'] for q in cases['queries'] if q['scene'] == 'fallback-limit']
    assert limit[0] == dict(waypoints=[[BASE[0] + 100, BASE[1] + 50]], via='fallback') and limit[1] is None is limit[2]
    door = [q for q in cases['queries'] if q['scene'].startswith('door') and q['dest'] == [3270, 3250]]
    assert {q['scene']: len(q['route']['waypoints']) for q in door if q['nearest']}['door-closed'] > \
        {q['scene']: len(q['route']['waypoints']) for q in door if q['nearest']}['door-open']


def test_the_shared_click_table_is_act_contracts():
    clicks = json.loads(FIXTURE.read_text())['clicks']
    assert [tuple(o) for o in clicks['offsets']] == list(CLICK_OFFSETS) and clicks['rows'] == list(CLICK_ROWS)


def test_the_python_port_answers_every_shared_case_as_recorded():
    cases = json.loads(FIXTURE.read_text())
    grids = {name: grid_of(cells) for name, cells in cases['scenes'].items()}
    base = tuple(cases['base'])
    for q in cases['queries']:
        source, dest = (q['source'][0] - base[0], q['source'][1] - base[1]), (q['dest'][0] - base[0], q['dest'][1] - base[1])
        assert route_of(grids[q['scene']], source, dest, tuple(q['arrival']), q['nearest']) == q['route'], q



# ---- act-v3's dispatch in the Python reference ---------------------------------------
KIT = list(START_KIT) + [('bronze_sword', 1)]


@pytest.fixture(scope='module')
def site(tables):
    """A development layout with chickens and a tree, and its LayoutTables."""
    for seed in W.split_seeds('development', 200):
        layout = W.generate(seed, tables)
        names = {tables.loc_types[int(t)]['debugname'] for t in layout.locs[:, 0]}
        if 'tree' in names and len(layout.npcs):
            return layout, W.LayoutTables(tables, layout)
    raise AssertionError('no development layout with a tree and npcs')


def started(site, seed=1):
    layout, lt = site
    world = World(lt, seed=seed, start_tick=WORLD_START_TICK)
    agent = Agent(world)
    agent.start(0, config.TASK_CFG, tuple(int(v) + o for v, o in zip(np.argwhere(layout.start)[0], W.ORIGIN)), KIT, {})
    return world, agent


def routed_click(agent):
    """The first legal click slot whose target the client's route reaches, as its row."""
    p, obs = agent.player, agent.last_observation
    for k, c in enumerate(obs['clicks']):
        if obs['mask'][CLICK_ROWS[k]] and agent.route((p.x + c['dx'], p.z + c['dz']), nearest=True):
            return CLICK_ROWS[k]
    raise AssertionError('no routed click')


def test_a_click_refused_while_delayed_sends_nothing_and_ends_failed(tables, site):
    world, agent = started(site)
    p = agent.player
    p.delayed, p.delayed_until = True, world.tick + 50
    obs = agent.step(routed_click(agent))
    assert obs['route']['refused'] == 'delayed' and obs['route']['waypoints'] == [] and not p.path
    assert obs['feedback']['ended']['reason'] == 'failed' and obs['feedback']['ended']['refused']
    assert (obs['feedback']['dx'], obs['feedback']['dz']) == (0, 0)


def test_a_click_without_a_route_keeps_the_earlier_walk(tables, site, monkeypatch):
    world, agent = started(site)
    p = agent.player
    agent.click(routed_click(agent))
    walk = list(p.path)
    assert walk
    monkeypatch.setattr(agent, 'route', lambda dest, arrival=None, nearest=False: None)
    note = agent.click(CLICK_ROWS[0])
    assert 'no route' in note and p.path == walk and agent.decision_route['refused'] == 'no route'
    assert agent.option_end['reason'] == 'failed' and agent.option_end['refused']


def test_an_act_on_an_unreachable_loc_during_a_click_walk_sends_only_the_op(tables, site, monkeypatch):
    """The op alone goes out: the click walk continues, and the op resolves when it ends, with "I can't reach that!"
    from a tile the loc cannot be used from."""
    world, agent = started(site)
    p = agent.player
    agent.click(routed_click(agent))
    walk = list(p.path)
    tree = next(loc for loc in world.locs if world.t.loc_types[loc.type]['debugname'] == 'tree'
                and walk[-1] not in world.loc_reaching(loc))
    monkeypatch.setattr(agent, 'route', lambda dest, arrival=None, nearest=False: None)
    agent.acting = A_ACT0
    agent.interact_loc(tree, 'Chop down', 'act_0')
    assert p.path == walk and p.target is tree and agent.decision_route['refused'] == 'no route'
    said = len(world.chat_log)
    for _ in range(300):
        if not p.path and p.target is None:
            break
        agent.wait_ticks(1)
    assert (p.x, p.z) == walk[-1] and p.target is None
    assert "I can't reach that!" in [text for _, text in world.chat_log[said:]]


def test_an_act_walks_by_the_clients_route_with_no_server_path(tables, site, monkeypatch):
    """ACT without move-near: the walk an op sends is the client's route to the loc's arrival tiles, and the server's
    path finder is never asked."""
    world, agent = started(site)
    p = agent.player
    tree = min((loc for loc in world.locs if world.t.loc_types[loc.type]['debugname'] == 'tree'),
               key=lambda loc: max(abs(loc.x - p.x), abs(loc.z - p.z)))
    t = world.t.loc_types[tree.type]
    expected = paths.client_route(world.player_scene(p), p.x, p.z, (tree.x, tree.z),
                                  paths.loc_arrival(tree.shape, tree.angle, t['width'], t['length'], t['forceapproach']))
    monkeypatch.setattr(World, 'path_to_tile', lambda *args: pytest.fail('a server path was asked for'))
    agent.acting = A_ACT0
    agent.interact_loc(tree, 'Chop down', 'act_0')
    assert expected and p.target is tree
    assert p.path == ([] if list(expected) == [(p.x, p.z)] else list(expected))
    # the walk ends where the server's own reach rule (ReachStrategy, World.loc_reaching) accepts the op, so the
    # op is not refused when the walk ends
    assert expected[-1] in world.loc_reaching(tree)
    said = len(world.chat_log)
    for _ in range(300):
        if not p.path:
            break
        agent.wait_ticks(1)
    agent.wait_ticks(2)
    assert (p.x, p.z) == expected[-1] and "I can't reach that!" not in [text for _, text in world.chat_log[said:]]


def test_an_act_on_a_loc_whose_sides_are_all_blocked_sends_no_walk(tables, site):
    """No move-near: with every tile the client's arrival test accepts beside a tree blocked, tryMove finds no
    route although tiles near the tree are reached (the engine's move-near would walk to one), so the op goes out
    alone and the player, standing still, hears "I can't reach that!"."""
    world, agent = started(site)
    p = agent.player
    trees = [loc for loc in world.locs if world.t.loc_types[loc.type]['debugname'] == 'tree' and loc.shape == 10
             and max(abs(loc.x - p.x), abs(loc.z - p.z)) >= 3]
    tree = min(trees, key=lambda loc: max(abs(loc.x - p.x), abs(loc.z - p.z)))
    w, h = tree.width, tree.length  # 2 by 2 in the tables
    block = next(k for k, t in world.t.loc_types.items() if t['blockwalk'] and t['width'] == t['length'] == 1)
    sides = [(dx, dz) for dx in range(-1, w + 1) for dz in range(-1, h + 1) if (dx in (-1, w)) != (dz in (-1, h))]
    for k, (dx, dz) in enumerate(sides):
        world.loc_collision(Loc(len(world.locs) + k, block, block, tree.x + dx, tree.z + dz, 10, 0, 1, 1,
                                level=tree.level), True)
    t = world.t.loc_types[tree.type]
    arrival = paths.loc_arrival(tree.shape, tree.angle, t['width'], t['length'], t['forceapproach'])
    scene = world.player_scene(p)
    assert paths.client_route(scene, p.x, p.z, (tree.x, tree.z), arrival) is None
    assert min(scene.at(scene.dist, tree.x + dx, tree.z + dz, paths.SCENE_UNREACHED)
               for dx, dz in ((-1, -1), (-1, h), (w, -1), (w, h))) < paths.SCENE_UNREACHED
    here, said = (p.x, p.z), len(world.chat_log)
    agent.acting = A_ACT0
    agent.interact_loc(tree, 'Chop down', 'act_0')
    assert agent.decision_route['refused'] == 'no route' and not p.path and p.target is tree
    agent.wait_ticks(2)
    assert (p.x, p.z) == here and p.target is None
    assert "I can't reach that!" in [text for _, text in world.chat_log[said:]]


def test_an_npc_out_of_the_players_list_is_refused(tables, site):
    world, agent = started(site)
    p = agent.player
    npc = next(n for n in world.npcs if n.active and world.t.npc_types[n.type]['name'] == 'Chicken')
    npc.observers = 0
    agent.acting = A_ACT0
    note = agent.interact_npc(npc, 'Attack', 'act_0')
    assert 'npc not in view' in note and p.target is None and agent.decision_route['refused'] == 'npc not in view'


def test_an_act_on_an_adjacent_npc_sends_its_own_tile_and_does_not_follow(tables, site):
    world, agent = started(site)
    p = agent.player
    npc = next(n for n in world.npcs if n.active and world.t.npc_types[n.type]['name'] == 'Chicken')
    spot = next((p.x + dx, p.z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if not world.flags(p.level, p.x + dx, p.z + dz) & 0x240100)
    world.npc_teleport(npc, *spot)
    agent.observe_npcs()
    npc.tele, npc.observers = False, 1
    agent.acting = A_ACT0
    agent.interact_npc(npc, 'Attack', 'act_0')
    assert agent.decision_route['waypoints'] == [dict(x=p.x, z=p.z)] and not p.path and not p.allow_repath
    assert p.target is npc and agent.decision_route['refused'] is None   # the op went out
    here = (p.x, p.z)
    far = next((p.x + dx, p.z + dz) for dx, dz in ((3, 0), (-3, 0), (0, 3), (0, -3))
               if not world.flags(p.level, p.x + dx, p.z + dz) & 0x240100)
    world.npc_teleport(npc, *far)
    agent.wait_ticks(4)
    assert (p.x, p.z) == here


def test_npcs_are_listed_within_the_servers_15_tiles(tables, site):
    world, agent = started(site)
    p = agent.player
    a, b = [n for n in world.npcs if n.active][:2]
    away = 1 if world.inside(p.x + 16, p.z) else -1
    world.npc_teleport(a, p.x + 15 * away, p.z)
    world.npc_teleport(b, p.x + 16 * away, p.z)
    listed = {n['npc'].nid for n in agent.scan()['npcs']}
    assert a.nid in listed and b.nid not in listed


if __name__ == '__main__':
    FIXTURE.write_text(json.dumps(build(), separators=(',', ':')) + '\n')
    print(f'wrote {FIXTURE}', file=sys.stderr)
