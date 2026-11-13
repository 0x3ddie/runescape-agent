// Walking distance from the player, on the collision snapshot every walk plans on.
//
// Both transports plan NAV through the SDK's baked collision (sdk/collision-data.json,
// doors open), so a place this finds unreachable is one NAV cannot walk to either.
// A breadth-first fill over a window around the player costs under a millisecond
// at radius 52; one long-path query per place cost 8 ms each.
import { initPathfinding } from '../vendor/rs-sdk/sdk/pathfinding';
import * as rsmod from '../vendor/rs-sdk/server/vendor/rsmod-pathfinder';

const STEPS: ReadonlyArray<readonly [number, number]> =
    [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];

export interface WalkMap {
    /** Tiles walked from the player, or -1 when the window has no route. */
    steps(x: number, z: number): number;
    /** Fewest tiles to stand beside (x, z), or -1 when no side is reached. */
    approach(x: number, z: number): number;
    /**
     * Fewest tiles to stand on a side of the loc at origin (x, z) with no wall between,
     * or -1. A tile beside a tree across a fence let the Fishing Guild teacher re-issue
     * one chop 584 times. Only the four tiles beside the origin count: a larger loc's
     * north and east edges are unknown to the client scan, so this is conservative.
     */
    operable(x: number, z: number): number;
}

// Moving from a side tile into the loc: the side's wall facing it, the loc's wall facing back.
const SIDES: ReadonlyArray<readonly [number, number, number, number]> = [
    [-1, 0, rsmod.CollisionFlag.WALL_EAST, rsmod.CollisionFlag.WALL_WEST],
    [1, 0, rsmod.CollisionFlag.WALL_WEST, rsmod.CollisionFlag.WALL_EAST],
    [0, -1, rsmod.CollisionFlag.WALL_NORTH, rsmod.CollisionFlag.WALL_SOUTH],
    [0, 1, rsmod.CollisionFlag.WALL_SOUTH, rsmod.CollisionFlag.WALL_NORTH]];

export function walkMap(level: number, px: number, pz: number, radius: number): WalkMap {
    initPathfinding();
    const width = 2 * radius + 1;
    const dist = new Int16Array(width * width).fill(-1);
    const index = (x: number, z: number) =>
        Math.abs(x - px) > radius || Math.abs(z - pz) > radius ? -1 : (x - px + radius) * width + (z - pz + radius);
    const queue = new Int32Array(width * width);
    let head = 0, tail = 0;
    dist[index(px, pz)] = 0;
    queue[tail++] = index(px, pz);
    while (head < tail) {
        const i = queue[head++];
        const x = Math.floor(i / width) - radius + px, z = (i % width) - radius + pz;
        for (const [dx, dz] of STEPS) {
            const j = index(x + dx, z + dz);
            if (j < 0 || dist[j] >= 0) continue;
            if (!rsmod.canTravel(level, x, z, dx, dz, 1, 0, rsmod.CollisionType.NORMAL)) continue;
            dist[j] = dist[i] + 1;
            queue[tail++] = j;
        }
    }
    const steps = (x: number, z: number) => { const i = index(x, z); return i < 0 ? -1 : dist[i]; };
    return {
        steps,
        approach(x, z) {
            let best = -1;
            for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) {
                const s = steps(x + dx, z + dz);
                if (s >= 0 && (best < 0 || s < best)) best = s;
            }
            return best;
        },
        operable(x, z) {
            let best = -1;
            for (const [dx, dz, sideWall, locWall] of SIDES) {
                const s = steps(x + dx, z + dz);
                if (s < 0 || rsmod.isFlagged(x + dx, z + dz, level, sideWall) || rsmod.isFlagged(x, z, level, locWall)) continue;
                if (best < 0 || s < best) best = s;
            }
            return best;
        }
    };
}
