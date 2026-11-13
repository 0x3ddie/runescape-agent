// env-ts/core/clicks.ts: act-v3's clicks: 24 minimap clicks at fixed offsets from the player,
// routed by the 2004 client's own route finder.
//
// PURE. A player walks by clicking: a click on the minimap or a ground tile makes the client route the whole path
// with one breadth-first search over the collision it holds of its loaded scene (webclient Client.ts tryMove,
// 7954-8206) and send the turn points nearest the player; the server walks them (MoveClickHandler). An op on a loc,
// npc or ground item routes the same way first, with the client's arrival tests (dash3d/CollisionMap.ts testWall,
// testWDecor, testLoc) and no fallback. Mirrored by rl/act_contract.py (the table) and sim/paths.py (the route, which
// sim/tests and core/clicks.test.ts hold to the same cases).
//
// The client's collision is the server's live flags without npc or player occupancy: walls (W_*) and scenery
// (WALK_SCENERY) share the server's bits; WR_GRND holds the floor and blocking ground decoration, which the server
// keeps apart (FLOOR 0x200000, FLOOR_DECORATION 0x40000); BLOCK_NPCS_AND_PLAYERS (0x80000) is never set. The scene's
// outer ring is _BOUNDS (CollisionMap.reset).

import { A_CLICK_HIGH, A_CLICK_LOW, N_CLICK, N_CLICK_LOW } from './actions';

/** Slot k's offset (dx, dz; z grows north): radii 4, 9 and 16 in the eight directions, north first, clockwise. */
const DIRECTIONS: Array<[number, number]> = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];
export const CLICK_RADII = [4, 9, 16];
export const CLICK_DIAGONALS = [3, 6, 11];
export const CLICK_OFFSETS: Array<[number, number]> = CLICK_RADII.flatMap((r, i) => DIRECTIONS.map(
    ([ux, uz]): [number, number] => [ux * (ux && uz ? CLICK_DIAGONALS[i] : r), uz * (ux && uz ? CLICK_DIAGONALS[i] : r)]));
/** Slot k's action row: slots 0..11 on rows 17..28, slots 12..23 on rows 84..95. */
export const CLICK_ROWS: number[] = Array.from({ length: N_CLICK }, (_, k) => k < N_CLICK_LOW ? A_CLICK_LOW + k
    : A_CLICK_HIGH + k - N_CLICK_LOW);

/** The click slot of an action row, or null. */
export function clickSlot(action: number): number | null {
    if (action >= A_CLICK_LOW && action < A_CLICK_LOW + N_CLICK_LOW) return action - A_CLICK_LOW;
    if (action >= A_CLICK_HIGH && action < A_CLICK_HIGH + N_CLICK - N_CLICK_LOW) return action - A_CLICK_HIGH + N_CLICK_LOW;
    return null;
}

/** A click slot as the observation publishes it: its offset, whether its target lies in the declared world, and
 * whether the target tile's live collision has none of WALK_BLOCKED (walls and npcs aside). */
export type ClickSlot = { dx: number; dz: number; inside: boolean; open: boolean };

/** The server's walk-blocking bits of a tile: FLOOR | FLOOR_DECORATION | LOC. */
export const WALK_BLOCKED = 0x240100;

export function clickSlots(x: number, z: number, serverFlags: (x: number, z: number) => number,
                           inside: (x: number, z: number) => boolean = () => true): ClickSlot[] {
    return CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: inside(x + dx, z + dz),
        open: (serverFlags(x + dx, z + dz) & WALK_BLOCKED) === 0 }));
}

// ---- the client's collision and tryMove ---------------------------------------------------------------------------
export const SCENE = 104; // BuildArea.SIZE: 13 zones of 8 tiles around the zone of the build area's origin
export const NEAREST_STEPS = 100; // tryNearest: a tile of the 3 x 3 around the target reached in fewer steps
const UNREACHED = 99999999;
const C = { W_N: 0x2, W_E: 0x8, W_S: 0x20, W_W: 0x80, GRND: 0x200000, BOUNDS: 0xffffff,
    PL_WALK_N: 0x280102, PL_WALK_E: 0x280108, PL_WALK_NE: 0x28010e, PL_WALK_S: 0x280120, PL_WALK_SE: 0x280138,
    PL_WALK_W: 0x280180, PL_WALK_NW: 0x280183, PL_WALK_SW: 0x2801e0 };
const D = { NORTH: 0x1, EAST: 0x2, SOUTH: 0x4, WEST: 0x8 }; // DirectionFlag, forceapproach's bits too

/** The client's collision of one tile's server flags (CollisionFlag.NULL, an unloaded tile, stays blocked). */
export function clientFlags(server: number): number {
    return (server & 0x1ff) | ((server & 0x240000) !== 0 ? C.GRND : 0);
}

/** The south-west tile of the client's scene: 6 zones below the zone of the build area's origin. */
export function sceneBase(originX: number, originZ: number): { x: number; z: number } {
    return { x: ((originX >> 3) - 6) << 3, z: ((originZ >> 3) - 6) << 3 };
}

/** CollisionMap.testWall: whether (sx, sz), whose client collision is `f`, reaches a wall of `shape` and `angle`. */
export function testWall(f: number, sx: number, sz: number, dx: number, dz: number, shape: number, angle: number): boolean {
    if (sx === dx && sz === dz) return true;
    const walk = (mask: number) => (f & mask) === 0;
    if (shape === 0) {
        if (angle === 0) return (sx === dx - 1 && sz === dz) || (sx === dx && sz === dz + 1 && walk(C.PL_WALK_S))
            || (sx === dx && sz === dz - 1 && walk(C.PL_WALK_N));
        if (angle === 1) return (sx === dx && sz === dz + 1) || (sx === dx - 1 && sz === dz && walk(C.PL_WALK_E))
            || (sx === dx + 1 && sz === dz && walk(C.PL_WALK_W));
        if (angle === 2) return (sx === dx + 1 && sz === dz) || (sx === dx && sz === dz + 1 && walk(C.PL_WALK_S))
            || (sx === dx && sz === dz - 1 && walk(C.PL_WALK_N));
        if (angle === 3) return (sx === dx && sz === dz - 1) || (sx === dx - 1 && sz === dz && walk(C.PL_WALK_E))
            || (sx === dx + 1 && sz === dz && walk(C.PL_WALK_W));
    } else if (shape === 2) {
        if (angle === 0) return (sx === dx - 1 && sz === dz) || (sx === dx && sz === dz + 1)
            || (sx === dx + 1 && sz === dz && walk(C.PL_WALK_W)) || (sx === dx && sz === dz - 1 && walk(C.PL_WALK_N));
        if (angle === 1) return (sx === dx - 1 && sz === dz && walk(C.PL_WALK_E)) || (sx === dx && sz === dz + 1)
            || (sx === dx + 1 && sz === dz) || (sx === dx && sz === dz - 1 && walk(C.PL_WALK_N));
        if (angle === 2) return (sx === dx - 1 && sz === dz && walk(C.PL_WALK_E)) || (sx === dx && sz === dz + 1 && walk(C.PL_WALK_S))
            || (sx === dx + 1 && sz === dz) || (sx === dx && sz === dz - 1);
        if (angle === 3) return (sx === dx - 1 && sz === dz) || (sx === dx && sz === dz + 1 && walk(C.PL_WALK_S))
            || (sx === dx + 1 && sz === dz && walk(C.PL_WALK_W)) || (sx === dx && sz === dz - 1);
    } else if (shape === 9) {
        return (sx === dx && sz === dz + 1 && !(f & C.W_S)) || (sx === dx && sz === dz - 1 && !(f & C.W_N))
            || (sx === dx - 1 && sz === dz && !(f & C.W_E)) || (sx === dx + 1 && sz === dz && !(f & C.W_W));
    }
    return false;
}

/** CollisionMap.testWDecor: whether (sx, sz), whose client collision is `f`, reaches a wall decoration. */
export function testWDecor(f: number, sx: number, sz: number, dx: number, dz: number, shape: number, angle: number): boolean {
    if (sx === dx && sz === dz) return true;
    if (shape === 6 || shape === 7) {
        if (shape === 7) angle = (angle + 2) & 0x3;
        if (angle === 0) return (sx === dx + 1 && sz === dz && !(f & C.W_W)) || (sx === dx && sz === dz - 1 && !(f & C.W_N));
        if (angle === 1) return (sx === dx - 1 && sz === dz && !(f & C.W_E)) || (sx === dx && sz === dz - 1 && !(f & C.W_N));
        if (angle === 2) return (sx === dx - 1 && sz === dz && !(f & C.W_E)) || (sx === dx && sz === dz + 1 && !(f & C.W_S));
        if (angle === 3) return (sx === dx + 1 && sz === dz && !(f & C.W_W)) || (sx === dx && sz === dz + 1 && !(f & C.W_S));
    } else if (shape === 8) {
        return (sx === dx && sz === dz + 1 && !(f & C.W_S)) || (sx === dx && sz === dz - 1 && !(f & C.W_N))
            || (sx === dx - 1 && sz === dz && !(f & C.W_E)) || (sx === dx + 1 && sz === dz && !(f & C.W_W));
    }
    return false;
}

/** CollisionMap.testLoc: whether (sx, sz) reaches a width x length rectangle at (dx, dz): inside it, or beside a side
 * the loc's forceapproach leaves open with no wall between. */
export function testLoc(f: number, sx: number, sz: number, dx: number, dz: number, width: number, length: number,
                        forceapproach: number): boolean {
    const mx = dx + width - 1, mz = dz + length - 1;
    if (sx >= dx && sx <= mx && sz >= dz && sz <= mz) return true;
    return (sx === dx - 1 && sz >= dz && sz <= mz && !(f & C.W_E) && !(forceapproach & D.WEST))
        || (sx === mx + 1 && sz >= dz && sz <= mz && !(f & C.W_W) && !(forceapproach & D.EAST))
        || (sz === dz - 1 && sx >= dx && sx <= mx && !(f & C.W_N) && !(forceapproach & D.SOUTH))
        || (sz === mz + 1 && sx >= dx && sx <= mx && !(f & C.W_S) && !(forceapproach & D.NORTH));
}

/** What tryMove arrives at, beside its target tile: a loc op's rectangle and forceapproach, or its wall shape. */
export type Arrival = { width: number; length: number; angle: number; shape: number; forceapproach: number };
/** tryMove's arguments for a tile click (none of these). */
export const TILE: Arrival = { width: 0, length: 0, angle: 0, shape: 0, forceapproach: 0 };
/** tryMove's arguments for an npc op, and for a ground item once its tile has no route: a 1 x 1 rectangle. */
export const ENTITY: Arrival = { width: 1, length: 1, angle: 0, shape: 0, forceapproach: 0 };

/** The client's tryMove arguments for a loc op (Client.ts 7920-7939): centrepieces and ground decoration by their
 * rotated size and rotated forceapproach, walls and wall decoration by their shape (as locShape = shape + 1) and angle. */
export function locArrival(shape: number, angle: number, width: number, length: number, forceapproach: number): Arrival {
    if (shape === 10 || shape === 11 || shape === 22) {
        const [w, l] = angle === 0 || angle === 2 ? [width, length] : [length, width];
        const fa = angle === 0 ? forceapproach : ((forceapproach << angle) & 0xf) + (forceapproach >> (4 - angle));
        return { width: w, length: l, angle: 0, shape: 0, forceapproach: fa };
    }
    return { width: 0, length: 0, angle, shape: shape + 1, forceapproach: 0 };
}

export type Tile = { x: number; z: number };
/** A route tryMove found: its turn points nearest the source first (at most 25), and how it ended. */
export type Route = { waypoints: Tile[]; via: 'target' | 'reach' | 'fallback' };

/**
 * Client.tryMove from (srcX, srcZ) to (dx, dz), in absolute coordinates over the scene at `base`; `flags(x, z)` is
 * the client collision of an absolute tile inside the scene. Null when it finds no route (nothing is sent).
 */
export function tryMove(flags: (x: number, z: number) => number, base: Tile, srcX: number, srcZ: number, destX: number,
                        destZ: number, tryNearest: boolean, a: Arrival): Route | null {
    const size = SCENE;
    const at = (x: number, z: number) => x * size + z;
    // CollisionMap.reset: the ring is _BOUNDS; the rest is the live collision
    const cmap = new Int32Array(size * size);
    for (let x = 0; x < size; x++) for (let z = 0; z < size; z++)
        cmap[at(x, z)] = x === 0 || z === 0 || x === size - 1 || z === size - 1 ? C.BOUNDS : flags(base.x + x, base.z + z);
    const dirMap = new Int32Array(size * size), distMap = new Int32Array(size * size).fill(UNREACHED);
    const sx = srcX - base.x, sz = srcZ - base.z, dx = destX - base.x, dz = destZ - base.z;
    if (sx < 0 || sz < 0 || sx >= size || sz >= size) return null;
    dirMap[at(sx, sz)] = 99;
    distMap[at(sx, sz)] = 0;
    const routeX: number[] = [sx], routeZ: number[] = [sz];
    let length = 0, arrived = false, via: Route['via'] = 'target', x = sx, z = sz;
    const push = (nx: number, nz: number, dir: number, cost: number) => {
        routeX.push(nx); routeZ.push(nz); dirMap[at(nx, nz)] = dir; distMap[at(nx, nz)] = cost;
    };
    while (length !== routeX.length) {
        x = routeX[length]; z = routeZ[length]; length++;
        if (x === dx && z === dz) { arrived = true; break; }
        if (a.shape !== 0) {
            if ((a.shape < 5 || a.shape === 10) && testWall(cmap[at(x, z)], x, z, dx, dz, a.shape - 1, a.angle)) { arrived = true; via = 'reach'; break; }
            if (a.shape < 10 && testWDecor(cmap[at(x, z)], x, z, dx, dz, a.shape - 1, a.angle)) { arrived = true; via = 'reach'; break; }
        }
        if (a.width !== 0 && a.length !== 0 && testLoc(cmap[at(x, z)], x, z, dx, dz, a.width, a.length, a.forceapproach)) {
            arrived = true; via = 'reach'; break;
        }
        const cost = distMap[at(x, z)] + 1;
        const free = (nx: number, nz: number, mask: number) => (cmap[at(nx, nz)] & mask) === 0;
        if (x > 0 && dirMap[at(x - 1, z)] === 0 && free(x - 1, z, C.PL_WALK_E)) push(x - 1, z, 2, cost);
        if (x < size - 1 && dirMap[at(x + 1, z)] === 0 && free(x + 1, z, C.PL_WALK_W)) push(x + 1, z, 8, cost);
        if (z > 0 && dirMap[at(x, z - 1)] === 0 && free(x, z - 1, C.PL_WALK_N)) push(x, z - 1, 1, cost);
        if (z < size - 1 && dirMap[at(x, z + 1)] === 0 && free(x, z + 1, C.PL_WALK_S)) push(x, z + 1, 4, cost);
        if (x > 0 && z > 0 && dirMap[at(x - 1, z - 1)] === 0 && free(x - 1, z - 1, C.PL_WALK_NE)
            && free(x - 1, z, C.PL_WALK_E) && free(x, z - 1, C.PL_WALK_N)) push(x - 1, z - 1, 3, cost);
        if (x < size - 1 && z > 0 && dirMap[at(x + 1, z - 1)] === 0 && free(x + 1, z - 1, C.PL_WALK_NW)
            && free(x + 1, z, C.PL_WALK_W) && free(x, z - 1, C.PL_WALK_N)) push(x + 1, z - 1, 9, cost);
        if (x > 0 && z < size - 1 && dirMap[at(x - 1, z + 1)] === 0 && free(x - 1, z + 1, C.PL_WALK_SE)
            && free(x - 1, z, C.PL_WALK_E) && free(x, z + 1, C.PL_WALK_S)) push(x - 1, z + 1, 6, cost);
        if (x < size - 1 && z < size - 1 && dirMap[at(x + 1, z + 1)] === 0 && free(x + 1, z + 1, C.PL_WALK_SW)
            && free(x + 1, z, C.PL_WALK_W) && free(x, z + 1, C.PL_WALK_S)) push(x + 1, z + 1, 12, cost);
    }
    if (!arrived) {
        if (!tryNearest) return null;
        let min = NEAREST_STEPS;
        for (let px = dx - 1; px <= dx + 1; px++) for (let pz = dz - 1; pz <= dz + 1; pz++)
            if (px >= 0 && pz >= 0 && px < size && pz < size && distMap[at(px, pz)] < min) {
                min = distMap[at(px, pz)]; x = px; z = pz; arrived = true;
            }
        if (!arrived) return null;
        via = 'fallback';
    }
    // the turn points back from the end, then the 25 nearest the source (the client's bufferSize)
    const turns: Tile[] = [{ x, z }];
    let dir = dirMap[at(x, z)], next = dir;
    while (x !== sx || z !== sz) {
        if (next !== dir) { dir = next; turns.push({ x, z }); }
        if ((next & D.EAST) !== 0) x++; else if ((next & D.WEST) !== 0) x--;
        if ((next & D.NORTH) !== 0) z++; else if ((next & D.SOUTH) !== 0) z--;
        next = dirMap[at(x, z)];
    }
    return { waypoints: turns.reverse().slice(0, 25).map(t => ({ x: t.x + base.x, z: t.z + base.z })), via };
}

// ---- route-v1: what one decision's click or ACT asked for and what was sent (evidence, never a policy input) --------
export const ROUTE_RECORD = 'route-v1';
export type RouteRecord = {
    version: typeof ROUTE_RECORD;
    kind: 'click' | 'act';
    action: number;
    tick: number;
    source: { x: number; z: number; level: number };
    offset: { dx: number; dz: number } | null;
    target: Tile;
    thing: { kind: 'loc' | 'npc' | 'obj'; id: number; name: string; option: string } | null;
    via: Route['via'] | 'adjacent' | 'none';
    end: Tile | null;
    refused: null | 'delayed' | 'no route' | 'npc delayed' | 'npc not in view';
    waypoints: Tile[];
};
