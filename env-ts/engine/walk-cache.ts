// env-ts/engine/walk-cache.ts: the walk fills of env-ts/reach.ts walkMap, kept and given again for a tile filled before.
//
// walkMap is a pure function of (level, x, z, radius): it reads the SDK's collision snapshot, which nothing changes
// after initPathfinding (findLongPath adds its blocked doors' walls only inside its own synchronous call and takes them
// away before it returns). So the engine agent asks this file, which fills through the unchanged walkMap the first
// time and keeps the fill (the oldest of WALK_FILLS dropped first): the same answers, without the fill (0.45 ms of a
// decision measured, 56% of tiles found again at 128 agents). reach.ts stays as it is: it is an input of the
// simulator's archive (sim/build.py pins), which this cache must not change.
// RL_WALK_CACHE_CHECK=1 fills again at every reuse and counts the fills that answer differently (walkCacheStats).
import { walkMap, type WalkMap } from '../reach';

const WALK_FILLS = 4096; // about 19 KB a fill at radius 48: about 80 MB
const fills = new Map<number, WalkMap>();
const CHECK = process.env.RL_WALK_CACHE_CHECK === '1';
const stats = { hits: 0, misses: 0, checked: 0, differ: 0 };

export function walkCacheStats(): typeof stats & { size: number } {
    return { ...stats, size: fills.size };
}

/** Two fills of the same tile answer alike at every tile of the window and its edge. */
export function sameWalk(a: WalkMap, b: WalkMap, px: number, pz: number, radius: number): boolean {
    for (let x = px - radius - 1; x <= px + radius + 1; x++)
        for (let z = pz - radius - 1; z <= pz + radius + 1; z++)
            if (a.steps(x, z) !== b.steps(x, z) || a.approach(x, z) !== b.approach(x, z) || a.operable(x, z) !== b.operable(x, z))
                return false;
    return true;
}

/** walkMap(level, px, pz, radius), kept. */
export function keptWalkMap(level: number, px: number, pz: number, radius: number): WalkMap {
    if (radius >= 256) return walkMap(level, px, pz, radius);
    const key = ((level * 16384 + px) * 16384 + pz) * 256 + radius; // x and z under 16384, radius under 256: exact
    const kept = fills.get(key);
    if (kept) {
        stats.hits++;
        fills.delete(key); // the most recent last
        fills.set(key, kept);
        if (CHECK) {
            stats.checked++;
            if (!sameWalk(kept, walkMap(level, px, pz, radius), px, pz, radius)) stats.differ++;
        }
        return kept;
    }
    stats.misses++;
    const made = walkMap(level, px, pz, radius);
    fills.set(key, made);
    if (fills.size > WALK_FILLS) fills.delete(fills.keys().next().value!);
    return made;
}
