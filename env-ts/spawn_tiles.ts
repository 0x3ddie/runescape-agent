// Spawn-tile connectivity: which tiles may an episode legally start on?
//
// ::tele ignores collision, and the +/-3 jitter boxes around real POIs are full
// of tiles the pathfinder can never leave: 34/49 tiles around the Draynor fish
// spot are open water, and a sealed landing spends its whole episode returning
// "no waypoints" for every NAV while still earning fishing reward: return
// variance uncorrelated with the policy's choices. This, not teleport decay,
// is what starved the v20 overnight run (audit/notes/spawn-jitter-collision.md:
// 4/8 live fish-arm resets landed sealed, 66k failed walks in one night).
//
// A candidate is valid iff it is walkable AND the client BFS reaches the POI
// from it, so the episode's first walkTo is guaranteed a path. A POI that is
// itself unwalkable (the fish spot is a water-edge tile) is reached within 2
// tiles, since paths legitimately stop beside it.
//
// Everything here runs against the same baked collision snapshot walkTo uses
// (sdk/collision-data.json), so candidate selection and runtime pathfinding
// cannot disagree with each other. They can BOTH disagree with the live world
// (the snapshot can drift from the live map); that residual is what the
// runtime landing verification in doReset is for.
import { findLongPath, isTileWalkable } from '../vendor/rs-sdk/sdk/pathfinding';

export const JITTER_RADIUS = 3;

const candidateCache = new Map<string, Array<{ x: number; z: number }>>();

export function tileConnectedToPoi(x: number, z: number, poiX: number, poiZ: number): boolean {
    if (x === poiX && z === poiZ) return true;
    const wps = findLongPath(0, x, z, poiX, poiZ, 500, []);
    const last = wps[wps.length - 1];
    // A walkable POI must be reached exactly. "Within 2" also took the far side of a
    // one-tile bank counter: Draynor's spawn seeds 1-3 and 5-7 started shut in with the
    // bankers, and six other banks had such seeds too.
    const tolerance = isTileWalkable(0, poiX, poiZ) ? 0 : 2;
    return last !== undefined && Math.abs(last.x - poiX) + Math.abs(last.z - poiZ) <= tolerance;
}

export function connectedJitterTiles(baseX: number, baseZ: number): Array<{ x: number; z: number }> {
    const key = `${baseX},${baseZ}`;
    const hit = candidateCache.get(key);
    if (hit) return hit;
    const out: Array<{ x: number; z: number }> = [];
    for (let dx = -JITTER_RADIUS; dx <= JITTER_RADIUS; dx++) {
        for (let dz = -JITTER_RADIUS; dz <= JITTER_RADIUS; dz++) {
            const x = baseX + dx;
            const z = baseZ + dz;
            if (!isTileWalkable(0, x, z)) continue;
            if (tileConnectedToPoi(x, z, baseX, baseZ)) out.push({ x, z });
        }
    }
    candidateCache.set(key, out);
    return out;
}
