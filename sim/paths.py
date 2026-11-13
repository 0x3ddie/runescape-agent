"""The native path finder: its breadth-first trees, reach strategies and line of sight.

PathFinder.findPath (vendor engine routefinder) searches a 128-tile window from the source
in a fixed queue order and stops at the first dequeued tile its reach strategy accepts; if
none, it takes the closest approach point within 10 tiles of the destination. The queue
order does not depend on the destination, so a query only picks the end tile in the
source's whole tree and walks back. sim/extract.ts stored the tree of every supported tile,
of every level, on the boot map; while a door or gate has changed the collision inside a
source's window, `search` builds that tree here from the level's live flags, as the engine would. ReachStrategy and
LineValidator are ported for the same reason. sim/tests check all of them against engine
answers sim/extract.ts recorded, with and without doors moved.

`client_search` and `client_route` are the 2004 client's tryMove (act-v3, native-click): one search over the scene
the client holds, ported from the vendored webclient.

The hot path also exists in C (sim/_paths.c, the end of this module): the same answers, checked against these
functions; `use` picks one and IMPL names it.
"""
import numpy as np

from sim.data import D_EAST, D_NORTH, D_SOUTH, D_WEST, HALF, SEARCH

MAX_WAYPOINTS = 25
# Collision flags (routefinder/flags.ts).
WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST = 0x2, 0x8, 0x20, 0x80
LOC, LOC_PROJ = 0x100, 0x20000
PROJ_NORTH, PROJ_EAST, PROJ_SOUTH, PROJ_WEST = 0x400, 0x1000, 0x4000, 0x10000
BLOCK_NORTH, BLOCK_EAST, BLOCK_SOUTH, BLOCK_WEST = 0x240120, 0x240180, 0x240102, 0x240108
# Search steps in the path finder's order, the bit each has in a step mask, and the way back.
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
BACK = np.array([D_EAST, D_WEST, D_NORTH, D_SOUTH, D_NORTH | D_EAST, D_NORTH | D_WEST, D_SOUTH | D_EAST,
                 D_SOUTH | D_WEST], np.uint8)
_BLOCK = dict(W=0x240108, E=0x240180, S=0x240102, N=0x240120, SW=0x24010e, SE=0x240183, NW=0x240138, NE=0x2401e0)


class OutsideTables(Exception):
    """A query the tables cannot answer exactly: the caller reports a coverage failure."""


class Tree:
    """One source's breadth-first tree: the queue position of each tile (`orders`, `none`
    where unreached), its distance (`dists`, 255 where unreached), both kept within `r` tiles
    of the source, and the step back toward the source of every tile of the window."""

    def __init__(self, parents, orders, order_r, dists, dist_r, none):
        self.parents, self.orders, self.order_r, self.dists, self.dist_r, self.none = parents, orders, order_r, dists, dist_r, none

    @classmethod
    def stored(cls, t, src):
        return cls(t.parents[src], t.orders[src], t.meta['order_r'], t.dists[src], t.meta['dist_r'], 65535)

    def order(self, dx, dz):
        r = self.order_r
        if max(abs(dx), abs(dz)) > r:
            if -HALF <= dx < HALF and -HALF <= dz < HALF:
                raise OutsideTables(f'tile {dx},{dz} from the source beyond the stored queue order')
            return None
        o = int(self.orders[dx + r, dz + r])
        return None if o == self.none else o

    def dist(self, dx, dz):
        r = self.dist_r
        if not (-HALF <= dx < HALF and -HALF <= dz < HALF):
            return 255
        if max(abs(dx), abs(dz)) > r:
            raise OutsideTables(f'tile {dx},{dz} from the source beyond the stored distances')
        return int(self.dists[dx + r, dz + r])


def shape_layer(shape):
    """locShapeLayer: 0 wall, 1 wall decoration, 2 ground (scenery, diagonal walls, roofs), 3 ground decoration."""
    return 0 if shape <= 3 else 1 if shape <= 8 else 3 if shape == 22 else 2


# CollisionEngine wall flags by angle (W, N, E, S): (dx, dz, flag) for each tile a wall marks.
_STRAIGHT = (((0, 0, 0x80), (-1, 0, 0x8)), ((0, 0, 0x2), (0, 1, 0x20)), ((0, 0, 0x8), (1, 0, 0x80)),
             ((0, 0, 0x20), (0, -1, 0x2)))
_CORNER = (((0, 0, 0x1), (-1, 1, 0x10)), ((0, 0, 0x4), (1, 1, 0x40)), ((0, 0, 0x10), (1, -1, 0x1)),
           ((0, 0, 0x40), (-1, -1, 0x4)))
_L = (((0, 0, 0x82), (-1, 0, 0x8), (0, 1, 0x20)), ((0, 0, 0xa), (0, 1, 0x20), (1, 0, 0x80)),
      ((0, 0, 0x28), (1, 0, 0x80), (0, -1, 0x2)), ((0, 0, 0xa0), (0, -1, 0x2), (-1, 0, 0x8)))


def collision_masks(x, z, shape, angle, loc_type):
    """GameMap.changeLocCollision: the (x, z, flags) a blocking loc adds when World.addLoc
    places it and removes when it goes. A wall that blocks range also sets its projectile
    flags (CollisionEngine.changeWall*); ground locs fill their rotated rectangle."""
    if not loc_type['blockwalk']:
        return []
    layer, blockrange = shape_layer(shape), bool(loc_type['blockrange'])
    if layer == 0:
        table = _STRAIGHT if shape == 0 else _L if shape == 2 else _CORNER
        return [(x + dx, z + dz, flag | (flag << 9 if blockrange else 0)) for dx, dz, flag in table[angle]]
    if layer == 2:
        w, l = (loc_type['length'], loc_type['width']) if angle & 1 else (loc_type['width'], loc_type['length'])
        mask = LOC | (LOC_PROJ if blockrange else 0)
        return [(x + i, z + j, mask) for j in range(l) for i in range(w)]
    if layer == 3 and loc_type['active'] == 1:
        return [(x, z, 0x200000)]  # changeFloor: CollisionFlag.FLOOR
    return []


def step_masks(flags):
    """The steps canTravel (size 1, NORMAL) allows from each tile of a flag grid, as STEPS bits;
    tiles beyond the grid are blocked (CollisionFlag.NULL)."""
    padded = np.pad(np.asarray(flags, np.int64), 1, constant_values=0x7fffffff)
    ok = lambda dx, dz, name: (padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dz:padded.shape[1] - 1 + dz] & _BLOCK[name]) == 0
    w, e, s, n = ok(-1, 0, 'W'), ok(1, 0, 'E'), ok(0, -1, 'S'), ok(0, 1, 'N')
    masks = (w, e, s, n, ok(-1, -1, 'SW') & w & s, ok(1, -1, 'SE') & e & s, ok(-1, 1, 'NW') & w & n,
             ok(1, 1, 'NE') & e & n)
    out = np.zeros(np.shape(flags), np.uint8)
    for k, m in enumerate(masks):
        out |= m.astype(np.uint8) << k
    return out


def _bfs(steps, height, start, lo, hi, stop=None):
    """Breadth-first search in PathFinder.findPath1's queue order over a step-mask grid stored
    flat (index x * height + z), within the window lo..hi (inclusive grid coordinates, (x, z)).
    Level by level: a level's tiles are taken in queue order, each appends its unvisited
    neighbours in step order, so the first discovery of a tile fixes its parent and position.
    Returns (tiles in queue order, their levels, parents by tile), stopping after the level in
    which `stop` is discovered."""
    offsets = np.array([dx * height + dz for dx, dz in STEPS], np.int64)
    sdx = np.array([dx for dx, _ in STEPS]); sdz = np.array([dz for _, dz in STEPS])
    parent = {start: 99}
    tiles, levels = [np.array([start], np.int64)], [0]
    frontier = tiles[0]
    level = 0
    seen_mask = np.zeros(steps.size, bool)
    seen_mask[start] = True
    while frontier.size and stop not in parent:
        fx, fz = frontier // height, frontier % height
        bits = steps[frontier]
        cand = frontier[:, None] + offsets[None, :]
        nx, nz = fx[:, None] + sdx[None, :], fz[:, None] + sdz[None, :]
        ok = ((bits[:, None] >> np.arange(8)) & 1).astype(bool)
        ok &= (nx >= lo[0]) & (nx <= hi[0]) & (nz >= lo[1]) & (nz <= hi[1])
        flat, dirs = cand[ok], np.broadcast_to(np.arange(8), cand.shape)[ok]
        fresh = ~seen_mask[flat]
        flat, dirs = flat[fresh], dirs[fresh]
        _, first = np.unique(flat, return_index=True)
        first.sort()
        new, back = flat[first], BACK[dirs[first]]
        seen_mask[new] = True
        level += 1
        parent.update(zip(new.tolist(), back.tolist()))
        tiles.append(new)
        levels.append(level)
        frontier = new
    return tiles, levels, parent


def _search_window(steps, x0, z0, sx, sz, bounded):
    """search's grid height, window origin (lx, lz) and window bounds lo..hi in grid coordinates."""
    height = steps.shape[1]
    lx, lz = sx - HALF - x0, sz - HALF - z0
    if bounded:
        return height, lx, lz, (0, 0), (steps.shape[0] - 1, height - 1)
    if lx < 0 or lz < 0 or lx + SEARCH > steps.shape[0] or lz + SEARCH > height:
        raise OutsideTables(f'search window of {sx},{sz} leaves the data area')
    return height, lx, lz, (lx, lz), (lx + SEARCH - 1, lz + SEARCH - 1)


def search(steps, x0, z0, sx, sz, bounded=False):
    """findPath1's whole tree from (sx, sz) over the live step masks of a grid whose (0, 0) is
    (x0, z0); the 128-tile window must lie inside the grid, unless the grid is `bounded`: smaller
    than the window and walled in (no step leaves it, as step_masks pads), so the window's search
    is the grid's."""
    height, lx, lz, lo, hi = _search_window(steps, x0, z0, sx, sz, bounded)
    tiles, levels, parent = _bfs(steps.ravel(), height, (sx - x0) * height + (sz - z0), lo, hi)
    size = 2 * HALF + 1
    orders = np.full((size, size), -1, np.int32)
    dists = np.full((size, size), 255, np.int32)
    parents = np.zeros((SEARCH, SEARCH), np.uint8)
    position = 0
    for group, level in zip(tiles, levels):
        gx, gz = group // height - lx, group % height - lz
        orders[gx, gz] = position + np.arange(group.size)
        dists[gx, gz] = min(level, 254)
        parents[gx, gz] = [parent[t] for t in group.tolist()]
        position += group.size
    return Tree(parents, orders, HALF, dists, HALF, -1)


def end_tile(tree, sx, sz, reaching, dest, rect=(1, 1)):
    """The tile the native search ends on, or None (no route and no approach point).
    `reaching` lists the tiles the reach strategy accepts; `dest` and `rect` are the
    destination's south-west corner and rotated size, for the approach-point search."""
    best, end = None, None
    for x, z in reaching:
        o = tree.order(x - sx, z - sz)
        if o is not None and (best is None or o < best):
            best, end = o, (x, z)
    if end is not None:
        return end
    # PathFinder.findClosestApproachPoint: x outer, z inner; a lower cost, or the same cost
    # with a shorter path, wins; only tiles fewer than 100 steps away count.
    (ddx, ddz), (w, h) = (dest[0] - sx, dest[1] - sz), rect
    lowest, shortest = 1000, 100
    for x in range(ddx - 10, ddx + 11):
        for z in range(ddz - 10, ddz + 11):
            d = tree.dist(x, z)
            if d >= 100:  # 255 marks unreached
                continue
            cx = ddx - x if x < ddx else x - (w + ddx - 1) if x > ddx + w - 1 else 0
            cz = ddz - z if z < ddz else z - (h + ddz - 1) if z > ddz + h - 1 else 0
            cost = cx * cx + cz * cz
            if cost < lowest or (cost == lowest and shortest > d):
                end, lowest, shortest = (sx + x, sz + z), cost, d
    return end


def backtrack(tree, sx, sz, end):
    """Waypoints (x, z) from the source to `end`, as PathFinder.findPath packs them:
    turning points, at most 25, the 25 nearest the source."""
    if end is None:
        return []
    parents = tree.parents
    cx, cz = end[0] - sx + HALF, end[1] - sz + HALF
    waypoints, curr = [], -1
    nxt = int(parents[cx, cz])
    for _ in range(SEARCH * SEARCH):
        if cx == HALF and cz == HALF:
            break
        if curr != nxt:
            curr = nxt
            if len(waypoints) >= MAX_WAYPOINTS:
                waypoints.pop()
            waypoints.insert(0, (sx + cx - HALF, sz + cz - HALF))
        if curr & D_EAST:
            cx += 1
        elif curr & D_WEST:
            cx -= 1
        if curr & D_NORTH:
            cz += 1
        elif curr & D_SOUTH:
            cz -= 1
        nxt = int(parents[cx, cz])
    return waypoints


def source_index(t, level, x, z):
    src = t.source(level, x, z)
    if src < 0:
        raise OutsideTables(f'no path table for {level},{x},{z}')
    return src


def path_to_tile(tree, sx, sz, x, z):
    """GameMap.findPath: to a tile (moveNear on)."""
    return backtrack(tree, sx, sz, end_tile(tree, sx, sz, [(x, z)], (x, z)))


NEIGHBOURS = tuple((dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1))


def entity_reaching(t, level, x, z):
    """Tiles that reach a size-1 entity on (x, z) (shape -2: beside it, not under it)."""
    i = int(t.tile(level, x, z))
    if i < 0:
        raise OutsideTables(f'entity at {level},{x},{z} outside the data area')
    mask = int(t.entity_reach[i])
    return [(x + dx, z + dz) for b, (dx, dz) in enumerate(NEIGHBOURS) if mask >> b & 1]


def packed(waypoints, level=0):
    return [((level & 3) << 28) | ((x & 0x3fff) << 14) | (z & 0x3fff) for x, z in waypoints]


# ---- ReachStrategy (routefinder/ReachStrategy.ts, RectangleBoundary.ts) for a 1x1 source ----
REACH_BOX = 4  # tiles around a destination that can reach it (sim/extract.ts REACH)


def reached(flags, sx, sz, dx, dz, width, length, angle, shape, block_access):
    """ReachStrategy.reached for srcSize 1; `flags(x, z)` reads the collision."""
    if shape == -2:
        rw, rh = (length, width) if angle & 1 else (width, length)
        inside = dx <= sx < dx + rw and dz <= sz < dz + rh
        return not inside and _reach_rectangle(flags, sx, sz, dx, dz, rw, rh, _rotate_access(angle, block_access))
    if sx == dx and sz == dz:
        return True
    if 0 <= shape <= 3 or shape == 9:
        return _reach_wall(flags(sx, sz), sx, sz, dx, dz, shape, angle)
    if shape < 0:
        return False
    if shape < 9:
        return _reach_wall_decor(flags(sx, sz), sx, sz, dx, dz, shape, angle)
    if shape in (10, 11, 22):
        rw, rh = (length, width) if angle & 1 else (width, length)
        return (dx <= sx < dx + rw and dz <= sz < dz + rh) or \
            _reach_rectangle(flags, sx, sz, dx, dz, rw, rh, _rotate_access(angle, block_access))
    return False


def _rotate_access(angle, flags):
    return flags if angle == 0 else ((flags << angle) & 0xf) | (flags >> (4 - angle))


def _reach_rectangle(flags, sx, sz, dx, dz, w, h, access):
    east, north = dx + w - 1, dz + h - 1
    if sx == dx - 1 and dz <= sz <= north and not flags(sx, sz) & WALL_EAST and not access & 0x8:
        return True
    if sx == east + 1 and dz <= sz <= north and not flags(sx, sz) & WALL_WEST and not access & 0x2:
        return True
    if sz + 1 == dz and dx <= sx <= east and not flags(sx, sz) & WALL_NORTH and not access & 0x4:
        return True
    return sz == north + 1 and dx <= sx <= east and not flags(sx, sz) & WALL_SOUTH and not access & 0x1


def _reach_wall(f, sx, sz, dx, dz, shape, angle):
    at = lambda ox, oz: sx == dx + ox and sz == dz + oz
    if shape == 0:
        if angle == 0:
            return at(-1, 0) or (at(0, 1) and not f & BLOCK_NORTH) or (at(0, -1) and not f & BLOCK_SOUTH)
        if angle == 1:
            return at(0, 1) or (at(-1, 0) and not f & BLOCK_WEST) or (at(1, 0) and not f & BLOCK_EAST)
        if angle == 2:
            return at(1, 0) or (at(0, 1) and not f & BLOCK_NORTH) or (at(0, -1) and not f & BLOCK_SOUTH)
        return at(0, -1) or (at(-1, 0) and not f & BLOCK_WEST) or (at(1, 0) and not f & BLOCK_EAST)
    if shape == 2:
        if angle == 0:
            return at(-1, 0) or at(0, 1) or (at(1, 0) and not f & BLOCK_EAST) or (at(0, -1) and not f & BLOCK_SOUTH)
        if angle == 1:
            return (at(-1, 0) and not f & BLOCK_WEST) or at(0, 1) or at(1, 0) or (at(0, -1) and not f & BLOCK_SOUTH)
        if angle == 2:
            return (at(-1, 0) and not f & BLOCK_WEST) or (at(0, 1) and not f & BLOCK_NORTH) or at(1, 0) or at(0, -1)
        return at(-1, 0) or (at(0, 1) and not f & BLOCK_NORTH) or (at(1, 0) and not f & BLOCK_EAST) or at(0, -1)
    if shape == 9:
        return ((at(0, 1) and not f & WALL_SOUTH) or (at(0, -1) and not f & WALL_NORTH)
                or (at(-1, 0) and not f & WALL_EAST) or (at(1, 0) and not f & WALL_WEST))
    return False


def _reach_wall_decor(f, sx, sz, dx, dz, shape, angle):
    at = lambda ox, oz: sx == dx + ox and sz == dz + oz
    if shape in (6, 7):
        rotation = (angle + 2) & 3 if shape == 7 else angle
        if rotation == 0:
            return (at(1, 0) and not f & WALL_WEST) or (at(0, -1) and not f & WALL_NORTH)
        if rotation == 1:
            return (at(-1, 0) and not f & WALL_EAST) or (at(0, -1) and not f & WALL_NORTH)
        if rotation == 2:
            return (at(-1, 0) and not f & WALL_EAST) or (at(0, 1) and not f & WALL_SOUTH)
        return (at(1, 0) and not f & WALL_WEST) or (at(0, 1) and not f & WALL_SOUTH)
    if shape == 8:
        return ((at(0, 1) and not f & WALL_SOUTH) or (at(0, -1) and not f & WALL_NORTH)
                or (at(-1, 0) and not f & WALL_EAST) or (at(1, 0) and not f & WALL_WEST))
    return False


def loc_reaching(flags, x, z, width, length, angle, shape, forceapproach):
    """The tiles around a loc that reach it, as sim/extract.ts lists them for the stored locs."""
    return [(sx, sz) for sx in range(x - REACH_BOX, x + REACH_BOX + width + 1)
            for sz in range(z - REACH_BOX, z + REACH_BOX + length + 1)
            if reached(flags, sx, sz, x, z, width, length, angle, shape, forceapproach)]


def entity_reaching_live(flags, x, z):
    return [(x + dx, z + dz) for dx, dz in NEIGHBOURS if reached(flags, x + dx, z + dz, x, z, 1, 1, 0, -2, 0)]


# ---- LineValidator.hasLineOfSight, 1x1 to 1x1, no extra flag --------------------------------------
def line_of_sight(flags, sx, sz, ex, ez):
    if sx == ex and sz == ez:
        return True
    if flags(sx, sz) & LOC:
        return False
    dx, dz = ex - sx, ez - sz
    east, north = dx >= 0, dz >= 0
    x_flags = (LOC_PROJ | PROJ_WEST) if east else (LOC_PROJ | PROJ_EAST)
    z_flags = (LOC_PROJ | PROJ_SOUTH) if north else (LOC_PROJ | PROJ_NORTH)
    half = 1 << 15
    if abs(dx) > abs(dz):
        ox, oz = (1 if east else -1), (0 if north else -1)
        scaled = (sz << 16) + half + oz
        tangent = int((dz << 16) / abs(dx))
        x = sx
        while x != ex:
            x += ox
            z = scaled >> 16
            if x == ex and z == ez:
                x_flags &= ~LOC_PROJ
            if flags(x, z) & x_flags:
                return False
            scaled += tangent
            nz = scaled >> 16
            if x == ex and nz == ez:
                z_flags &= ~LOC_PROJ
            if nz != z and flags(x, nz) & z_flags:
                return False
    else:
        ox, oz = (0 if east else -1), (1 if north else -1)
        scaled = (sx << 16) + half + ox
        tangent = int((dx << 16) / abs(dz))
        z = sz
        while z != ez:
            z += oz
            x = scaled >> 16
            if x == ex and z == ez:
                z_flags &= ~LOC_PROJ
            if flags(x, z) & z_flags:
                return False
            scaled += tangent
            nx = scaled >> 16
            if nx == ex and z == ez:
                x_flags &= ~LOC_PROJ
            if nx != x and flags(nx, z) & x_flags:
                return False
    return True


# ---- the SDK long route (sdk/pathfinding.ts findLongPath -> rsmod findLongPath) ---------------------
def walk_steps(t, src, sx, sz, x, z):
    """env-ts/reach.ts walkMap(...).steps on the SDK snapshot, -1 when unreached."""
    r = t.meta['walk_r']
    dx, dz = x - sx, z - sz
    if max(abs(dx), abs(dz)) > r:
        raise OutsideTables(f'walk steps {sx},{sz} -> {x},{z} beyond the stored walk map')
    return int(t.walks[src, dx + r, dz + r])


def walk_approach(t, src, sx, sz, x, z):
    best = -1
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            s = walk_steps(t, src, sx, sz, x + dx, z + dz)
            if s >= 0 and (best < 0 or s < best):
                best = s
    return best


# ---- the client's route finder (webclient Client.tryMove, dash3d/CollisionMap): native-click's clicks and ACT ----------
# act-v3 routes a click or an ACT as the 2004 client does: one breadth-first search over the
# collision the client holds of its loaded scene, in the queue order above (W, E, S, N, SW, SE, NW, NE), from the
# player's tile to the first dequeued tile that arrives, then the turn points back from it, the 25 nearest the player.
# The client's collision is the server's live flags without npc or player occupancy: its walls (W_*) and scenery
# (WALK_SCENERY) share the server's bits, and its WR_GRND holds the floor and blocking ground decoration, which the
# server keeps apart (0x200000, 0x40000; ClientBuild blockGround); BLOCK_NPCS_AND_PLAYERS (0x80000) is never set.
SCENE = 104                  # BuildArea.SIZE: 13 zones of 8 tiles a side around the zone of the build area's origin
NEAREST_STEPS = 100          # tryNearest: a tile of the 3 x 3 around the target reached in fewer steps
SCENE_UNREACHED = 99999999   # distMap's initial value
C_W_N, C_W_E, C_W_S, C_W_W, C_GRND, C_BOUNDS = 0x2, 0x8, 0x20, 0x80, 0x200000, 0xFFFFFF
PL_WALK = dict(N=0x280102, E=0x280108, NE=0x28010E, S=0x280120, SE=0x280138, W=0x280180, NW=0x280183, SW=0x2801E0)
C_EAST, C_WEST, C_SOUTH, C_NORTH = D_EAST, D_WEST, D_SOUTH, D_NORTH  # DirectionFlag, forceapproach's bits too


def scene_base(origin_x, origin_z):
    """The south-west tile of the client's scene: 6 zones below the zone of the build area's origin."""
    return ((origin_x >> 3) - 6) << 3, ((origin_z >> 3) - 6) << 3


def client_flags(flags):
    """The client's collision of a grid of the server's live flags (without npc or player bits)."""
    f = np.asarray(flags, np.int64)
    return (f & 0x1ff) | np.where(f & 0x240000, C_GRND, 0)


def client_steps(cflags):
    """The steps tryMove takes from each tile of a client collision grid, as STEPS bits; tiles beyond it are _BOUNDS."""
    padded = np.pad(np.asarray(cflags, np.int64), 1, constant_values=C_BOUNDS)
    free = lambda dx, dz, name: (padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dz:padded.shape[1] - 1 + dz]
                                 & PL_WALK[name]) == 0
    w, e, s, n = free(-1, 0, 'E'), free(1, 0, 'W'), free(0, -1, 'N'), free(0, 1, 'S')
    masks = (w, e, s, n, free(-1, -1, 'NE') & w & s, free(1, -1, 'NW') & e & s, free(-1, 1, 'SE') & w & n,
             free(1, 1, 'SW') & e & n)
    out = np.zeros(np.shape(cflags), np.uint8)
    for k, m in enumerate(masks):
        out |= m.astype(np.uint8) << k
    return out


class SceneTree:
    """tryMove's search from one source over the client's scene, by scene tile (x - bx, z - bz): its queue position
    (`order`, -1 unreached), its steps from the source (`dist`, SCENE_UNREACHED unreached) and its step back toward the
    source (`back`, DirectionFlag bits, 99 at the source); `flags(x, z)` reads the client collision of a tile."""

    def __init__(self, base, order, dist, back, flags):
        self.base, self.order, self.dist, self.back, self.flags = base, order, dist, back, flags

    def at(self, array, x, z, missing):
        i, j = x - self.base[0], z - self.base[1]
        return int(array[i, j]) if 0 <= i < SCENE and 0 <= j < SCENE else missing


def client_search(steps, cflags, x0, z0, sx, sz, base, bounded=False):
    """tryMove's search from (sx, sz) over `steps` (client_steps of `cflags`, a grid whose (0, 0) is (x0, z0)) in the scene
    at `base`. The scene's outer ring is _BOUNDS (CollisionMap.reset), so the search keeps to its inner tiles. A bounded
    grid is walled in, so the scene's tiles beyond it are walls; otherwise the scene must lie inside the grid."""
    W, H = steps.shape
    lo, hi = (base[0] + 1 - x0, base[1] + 1 - z0), (base[0] + SCENE - 2 - x0, base[1] + SCENE - 2 - z0)
    if bounded:
        lo, hi = (max(lo[0], 0), max(lo[1], 0)), (min(hi[0], W - 1), min(hi[1], H - 1))
    elif lo[0] < 0 or lo[1] < 0 or hi[0] >= W or hi[1] >= H:
        raise OutsideTables(f'the scene at {base[0]},{base[1]} leaves the data area')
    if not (lo[0] <= sx - x0 <= hi[0] and lo[1] <= sz - z0 <= hi[1]):
        raise OutsideTables(f'route source {sx},{sz} outside its scene')
    tiles, levels, parent = _bfs(steps.ravel(), H, (sx - x0) * H + (sz - z0), lo, hi)
    order = np.full((SCENE, SCENE), -1, np.int32)
    dist = np.full((SCENE, SCENE), SCENE_UNREACHED, np.int64)
    back = np.zeros((SCENE, SCENE), np.uint8)
    position = 0
    for group, level in zip(tiles, levels):
        gx, gz = group // H + x0 - base[0], group % H + z0 - base[1]
        order[gx, gz] = position + np.arange(group.size)
        dist[gx, gz] = level
        back[gx, gz] = [parent[t] for t in group.tolist()]
        position += group.size
    flags = lambda x, z: int(cflags[x - x0, z - z0]) if 0 <= x - x0 < W and 0 <= z - z0 < H else C_BOUNDS
    return SceneTree(base, order, dist, back, flags)


def client_test_wall(f, sx, sz, dx, dz, shape, angle):
    """CollisionMap.testWall: whether (sx, sz), whose client collision is `f`, reaches a wall of `shape` and `angle`."""
    if sx == dx and sz == dz:
        return True
    walk = lambda name: (f & PL_WALK[name]) == 0
    if shape == 0:
        if angle == 0:
            return (sx == dx - 1 and sz == dz) or (sx == dx and sz == dz + 1 and walk('S')) or \
                (sx == dx and sz == dz - 1 and walk('N'))
        if angle == 1:
            return (sx == dx and sz == dz + 1) or (sx == dx - 1 and sz == dz and walk('E')) or \
                (sx == dx + 1 and sz == dz and walk('W'))
        if angle == 2:
            return (sx == dx + 1 and sz == dz) or (sx == dx and sz == dz + 1 and walk('S')) or \
                (sx == dx and sz == dz - 1 and walk('N'))
        if angle == 3:
            return (sx == dx and sz == dz - 1) or (sx == dx - 1 and sz == dz and walk('E')) or \
                (sx == dx + 1 and sz == dz and walk('W'))
    elif shape == 2:
        if angle == 0:
            return (sx == dx - 1 and sz == dz) or (sx == dx and sz == dz + 1) or \
                (sx == dx + 1 and sz == dz and walk('W')) or (sx == dx and sz == dz - 1 and walk('N'))
        if angle == 1:
            return (sx == dx - 1 and sz == dz and walk('E')) or (sx == dx and sz == dz + 1) or \
                (sx == dx + 1 and sz == dz) or (sx == dx and sz == dz - 1 and walk('N'))
        if angle == 2:
            return (sx == dx - 1 and sz == dz and walk('E')) or (sx == dx and sz == dz + 1 and walk('S')) or \
                (sx == dx + 1 and sz == dz) or (sx == dx and sz == dz - 1)
        if angle == 3:
            return (sx == dx - 1 and sz == dz) or (sx == dx and sz == dz + 1 and walk('S')) or \
                (sx == dx + 1 and sz == dz and walk('W')) or (sx == dx and sz == dz - 1)
    elif shape == 9:
        return (sx == dx and sz == dz + 1 and not f & C_W_S) or (sx == dx and sz == dz - 1 and not f & C_W_N) or \
            (sx == dx - 1 and sz == dz and not f & C_W_E) or (sx == dx + 1 and sz == dz and not f & C_W_W)
    return False


def client_test_wdecor(f, sx, sz, dx, dz, shape, angle):
    """CollisionMap.testWDecor: whether (sx, sz), whose client collision is `f`, reaches a wall decoration."""
    if sx == dx and sz == dz:
        return True
    if shape in (6, 7):
        if shape == 7:
            angle = (angle + 2) & 3
        if angle == 0:
            return (sx == dx + 1 and sz == dz and not f & C_W_W) or (sx == dx and sz == dz - 1 and not f & C_W_N)
        if angle == 1:
            return (sx == dx - 1 and sz == dz and not f & C_W_E) or (sx == dx and sz == dz - 1 and not f & C_W_N)
        if angle == 2:
            return (sx == dx - 1 and sz == dz and not f & C_W_E) or (sx == dx and sz == dz + 1 and not f & C_W_S)
        if angle == 3:
            return (sx == dx + 1 and sz == dz and not f & C_W_W) or (sx == dx and sz == dz + 1 and not f & C_W_S)
    elif shape == 8:
        return (sx == dx and sz == dz + 1 and not f & C_W_S) or (sx == dx and sz == dz - 1 and not f & C_W_N) or \
            (sx == dx - 1 and sz == dz and not f & C_W_E) or (sx == dx + 1 and sz == dz and not f & C_W_W)
    return False


def client_test_loc(f, sx, sz, dx, dz, width, length, forceapproach):
    """CollisionMap.testLoc: whether (sx, sz), whose client collision is `f`, reaches a width x length rectangle at (dx,
    dz): inside it, or beside a side the loc's forceapproach leaves open with no wall between."""
    mx, mz = dx + width - 1, dz + length - 1
    if dx <= sx <= mx and dz <= sz <= mz:
        return True
    return (sx == dx - 1 and dz <= sz <= mz and not f & C_W_E and not forceapproach & C_WEST) or \
        (sx == mx + 1 and dz <= sz <= mz and not f & C_W_W and not forceapproach & C_EAST) or \
        (sz == dz - 1 and dx <= sx <= mx and not f & C_W_N and not forceapproach & C_SOUTH) or \
        (sz == mz + 1 and dx <= sx <= mx and not f & C_W_S and not forceapproach & C_NORTH)


def client_arrival(width, length, angle, loc_shape, forceapproach):
    """tryMove's arrival tests for its arguments (locWidth, locLength, locAngle, locShape, forceapproach): a wall test
    when locShape - 1 is a wall (below WALLDECOR_STRAIGHT_OFFSET, or WALL_DIAGONAL), a wall-decoration test when it is
    below CENTREPIECE_STRAIGHT, a rectangle test with a size; (rect the tests look beside, test)."""
    shape = loc_shape - 1
    wall, decor, rect = loc_shape != 0 and (loc_shape < 5 or loc_shape == 10), loc_shape != 0 and loc_shape < 10, \
        width != 0 and length != 0

    def test(f, sx, sz, dx, dz):
        return (wall and client_test_wall(f, sx, sz, dx, dz, shape, angle)) or \
            (decor and client_test_wdecor(f, sx, sz, dx, dz, shape, angle)) or \
            (rect and client_test_loc(f, sx, sz, dx, dz, width, length, forceapproach))
    return (max(width, 1), max(length, 1)), test


def loc_arrival(shape, angle, width, length, forceapproach):
    """The client's tryMove arguments for a loc op (Client.ts 7920-7939): centrepieces and ground decoration by their
    rotated size and rotated forceapproach, walls and wall decoration by their shape (locShape = shape + 1) and angle."""
    if shape in (10, 11, 22):
        w, h = (width, length) if angle in (0, 2) else (length, width)
        fa = forceapproach if angle == 0 else ((forceapproach << angle) & 0xf) + (forceapproach >> (4 - angle))
        return client_arrival(w, h, 0, 0, fa)
    return client_arrival(0, 0, angle, shape + 1, 0)


ENTITY_ARRIVAL = client_arrival(1, 1, 0, 0, 0)  # an npc op, or a ground item once its tile has no route


class Route(list):
    """client_route's turn points, and how tryMove ended (`via`: on the target tile, on a tile a reach test accepts,
    or on the 3 x 3 fallback tile)."""

    def __init__(self, points, via):
        super().__init__(points)
        self.via = via


def client_route(tree, sx, sz, dest, arrival=None, nearest=False):
    """tryMove's turn points from (sx, sz) as a Route: to the first tile in queue order that is `dest` or that
    `arrival` ((rect, test), client_arrival's) accepts; else, with `nearest`, to the tile of the 3 x 3 around `dest`
    with the fewest steps, fewer than NEAREST_STEPS (x outer, z inner, the first of equals); else None. The points are
    those nearest the source, at most MAX_WAYPOINTS, the source never one, except that a route ending on it is that one
    tile."""
    dx, dz = dest
    (w, h), test = arrival if arrival is not None else ((1, 1), None)
    end, best = None, None
    for x in range(dx - 1, dx + w + 1):
        for z in range(dz - 1, dz + h + 1):
            o = tree.at(tree.order, x, z, -1)
            if o >= 0 and (best is None or o < best) and \
                    ((x, z) == dest or (test is not None and test(tree.flags(x, z), x, z, dx, dz))):
                end, best = (x, z), o
    via = 'target' if end == dest else 'reach'
    if end is None and nearest:
        least, via = NEAREST_STEPS, 'fallback'
        for x in range(dx - 1, dx + 2):
            for z in range(dz - 1, dz + 2):
                d = tree.at(tree.dist, x, z, SCENE_UNREACHED)
                if d < least:
                    end, least = (x, z), d
    if end is None:
        return None
    x, z = end
    turns, curr = [end], tree.at(tree.back, x, z, 0)
    nxt = curr
    while (x, z) != (sx, sz):
        if nxt != curr:
            curr = nxt
            turns.append((x, z))
        x += 1 if nxt & D_EAST else -1 if nxt & D_WEST else 0
        z += 1 if nxt & D_NORTH else -1 if nxt & D_SOUTH else 0
        nxt = tree.at(tree.back, x, z, 0)
    return Route(turns[::-1][:MAX_WAYPOINTS], via)


# ---- the C port of the hot path (sim/_paths.c, built by sim/cpaths.py) ---------------------------
# The functions above stay the reference. use('c') puts the port's versions in their place and use('python') restores
# them; the port is used when it builds, unless SIM_PATHS=python. IMPL names the one in use. Callers reach these through
# the module (paths.walk_steps), so the choice holds everywhere. sim/tests/test_paths.py checks the two call by call
# and decision by decision.
from sim import cpaths  # noqa: E402

_C = cpaths.load()
if _C is not None:
    _C.configure(OutsideTables, HALF, SEARCH, MAX_WAYPOINTS, STEPS, BACK, D_NORTH, D_EAST, D_SOUTH, D_WEST)


def _c_walk_steps(t, src, sx, sz, x, z, _reference=walk_steps):
    m = t.walkmap
    return _reference(t, src, sx, sz, x, z) if m is None else m.steps(src, sx, sz, x, z)


def _c_walk_approach(t, src, sx, sz, x, z, _reference=walk_approach):
    m = t.walkmap
    return _reference(t, src, sx, sz, x, z) if m is None else m.approach(src, sx, sz, x, z)


def _c_search(steps, x0, z0, sx, sz, bounded=False):
    height, lx, lz, lo, hi = _search_window(steps, x0, z0, sx, sz, bounded)
    size = 2 * HALF + 1
    orders, dists = np.full((size, size), -1, np.int32), np.full((size, size), 255, np.int32)
    parents = np.zeros((SEARCH, SEARCH), np.uint8)
    _C.bfs_tree(np.ascontiguousarray(steps).ravel(), height, (sx - x0) * height + (sz - z0), *lo, *hi, lx, lz,
                orders, dists, parents)
    return Tree(parents, orders, HALF, dists, HALF, -1)


def _c_end_tile(tree, sx, sz, reaching, dest, rect=(1, 1)):
    return _C.end_tile(tree.orders, tree.order_r, tree.none, tree.dists, tree.dist_r, sx, sz, reaching, dest[0], dest[1],
                       rect[0], rect[1])


def _c_backtrack(tree, sx, sz, end):
    return [] if end is None else _C.backtrack(tree.parents, sx, sz, end[0], end[1])


IMPLS = dict(python=dict(walk_steps=walk_steps, walk_approach=walk_approach, search=search, end_tile=end_tile,
                         backtrack=backtrack),
             c=dict(walk_steps=_c_walk_steps, walk_approach=_c_walk_approach, search=_c_search, end_tile=_c_end_tile,
                    backtrack=_c_backtrack))


def use(impl):
    """Path finding by the Python reference ('python') or the C port ('c')."""
    global IMPL
    if impl == 'c' and _C is None:
        raise RuntimeError('sim/_paths.c is not built (sim/cpaths.py)')
    globals().update(IMPLS[impl])
    IMPL = impl


use('python' if _C is None else 'c')
