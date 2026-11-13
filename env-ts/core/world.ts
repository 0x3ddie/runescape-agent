// env-ts/core/world.ts: coordinate conversion, spawn jitter, reset verification.
//
// SHARED. The engine-side environment sets position directly instead of typing
// `::tele`, but it must spawn on the SAME tiles from the SAME seed, or paired runs
// stop being comparable.

import { connectedJitterTiles } from '../spawn_tiles';

// determinism guard: derive a small spawn jitter from the per-episode seed so
// identical seeds replay, but different episodes cannot converge to a single
// bit-identical trajectory (zero-variance returns kill PPO gradients).
//
// The jitter samples CONNECTED tiles, not a blind box. ::tele ignores
// collision, so a blind +/-3 around the Draynor fish POI teleports the agent
// into open water on 34/49 tiles, where the client BFS returns no waypoints
// for every destination for the whole episode. This is what starved v20
// overnight (audit/notes/spawn-jitter-collision.md), not teleport decay.
// Candidate tiles are precomputed per POI from the same collision snapshot
// walkTo paths against, so a spawn accepted here always has a first walk.
export function jitterWorld(x: number, z: number, seed: number): { x: number; z: number } {
    const cands = connectedJitterTiles(x, z);
    if (cands.length === 0) {
        // bad POI in the cfg: no amount of retrying can spawn this arm.
        // preflight_cfg.py catches this before a run; at runtime it must be
        // loud, not a silent water spawn billed as a real episode.
        throw new Error(`spawn POI (${x},${z}) has no connected jitter tile; fix the cfg`);
    }
    if (seed === 0) {
        const baseValid = cands.find(c => c.x === x && c.z === z);
        return baseValid ?? cands[0];
    }
    const h = (seed * 1103515245 + 12345) >>> 16;
    return cands[h % cands.length];
}

/**
 * A start tile `lo`-`hi` walking steps from a village spawn, picked by seed: a start
 * state away from the bank. Imitation waited beside trees 30 tiles from a bank, and,
 * full, with a booth 40 steps away; the teacher rarely goes that far. `steps` is the
 * walking distance from the spawn on the snapshot (env-ts/reach.ts); `back` says
 * whether a walk on the live map returns from a tile, since the snapshot opens doors.
 */
export function farSpawn(base: { x: number; z: number }, band: readonly number[], seed: number,
                         steps: (x: number, z: number) => number,
                         back: (tile: { x: number; z: number }) => boolean): { x: number; z: number } {
    const [lo, hi] = band;
    if (band.length !== 2 || !Number.isInteger(lo) || !Number.isInteger(hi) || lo < 1 || hi < lo)
        throw new Error('spawnSteps must be [lo, hi] walking steps');
    const tiles: Array<{ x: number; z: number }> = [];
    for (let x = base.x - hi; x <= base.x + hi; x++) for (let z = base.z - hi; z <= base.z + hi; z++) {
        const s = steps(x, z);
        if (s >= lo && s <= hi) tiles.push({ x, z });
    }
    // Probe with a prime stride: neighbours of a tile sealed in a building are sealed too.
    const start = (seed * 1103515245 + 12345) >>> 16;
    for (let i = 0; i < Math.min(tiles.length, 64); i++) {
        const tile = tiles[(start + i * 7919) % tiles.length];
        if (back(tile)) return tile;
    }
    throw new Error(`no start tile ${lo}-${hi} steps from (${base.x},${base.z}) walks back to it`);
}

// LostCity ::tele level,mx,mz,lx,lz with x=(mx<<6)+lx (ClientCheatHandler).
export function teleFromWorld(x: number, z: number): string {
    return `0,${x >> 6},${z >> 6},${x & 63},${z & 63}`;
}

export function worldFromTele(tele: string): { x: number; z: number } {
    const c = tele.split(',').map(n => parseInt(n, 10));
    return { x: (c[1] << 6) + c[3], z: (c[2] << 6) + c[4] };
}

// xp-reset verification: ::minme sets every stat to level 1 with Hitpoints at
// 10 (ClientCheatHandler). Check LEVELS + non-HP xp (both unambiguous in obs);
// the raw HP xp constant is varp-scale-dependent and deliberately not used.
//
// Takes the SDK state shape (skills[].experience). The engine-side environment
// reads player.stats[] instead, so it must project to this shape first or use
// its own equivalent; the predicate is what must stay identical.
export function xpIsReset(s: any): boolean {
    const skills = s?.skills ?? [];
    if (!skills.length) return false;
    return skills.every((sk: any) => (sk.name === 'Hitpoints' ? sk.level === 10 : sk.level === 1 && sk.experience === 0));
}
