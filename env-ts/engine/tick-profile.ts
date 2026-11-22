// env-ts/engine/tick-profile.ts: RL_TICK_PROFILE=1: where a lockstep world's wall time goes, in
// performance.now() milliseconds (World.cycleStats holds whole milliseconds, too coarse for a 1 ms phase).
//
// Measurement only: it wraps the World's own phase methods and cycle and changes no game state. The
// engine's phases come from World.cycle (vendor World.ts); `cycle` is the vendor's tick, `captured` the
// same tick with the native captures' wrappers around it (installed after this one), so captured − cycle
// is the captures' own cost. Npc counts split the world's npcs by the dormant-npc gate (vendor patch
// 0009, RL_NPC_RADIUS): `awake` are inside a player's dilated chunks and take a full turn.
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import { walkCacheStats } from './walk-cache';

export const PHASES = ['processWorld', 'processClientsIn', 'processNpcEventQueue', 'processNpcs', 'processPlayers',
    'processLogouts', 'auditPlayerSlots', 'processLogins', 'processZones', 'processInfo', 'processClientsOut',
    'processCleanup'] as const;

type Totals = Record<string, { ms: number; count: number; bytes: number }>;

export interface TickProfile {
    /** Add `ms` (and `bytes`) to the named total (a protocol command, a reply's encoding, a control request). */
    add(name: string, ms: number, bytes?: number): void;
    /** Time `fn` under `name`; an async fn is timed to its settlement. */
    time<T>(name: string, fn: () => T): T;
    /** Wrap the cycle again, outside the captures' wrappers (call after they are installed). */
    wrapCaptured(): void;
    snapshot(): Record<string, unknown>;
    reset(): void;
}

/** The npcs of `world` inside a player's chunks dilated by `radius` tiles (vendor patch 0009's gate), and all. */
export function npcCounts(world: any, radius: number): { awake: number; total: number } {
    let total = 0, awake = 0;
    const active = new Set<number>();
    for (const player of world.playerLoop.all()) {
        for (let cx = (player.x - radius) >> 6; cx <= (player.x + radius) >> 6; cx++)
            for (let cz = (player.z - radius) >> 6; cz <= (player.z + radius) >> 6; cz++)
                active.add((player.level << 20) | (cx << 10) | cz);
    }
    for (const npc of world.npcs) {
        total++;
        if (radius <= 0 || active.has((npc.level << 20) | ((npc.x >> 6) << 10) | (npc.z >> 6))) awake++;
    }
    return { awake, total };
}

/** The profile of this process, once installTickProfile has run (RL_TICK_PROFILE=1); null otherwise. */
export let activeProfile: TickProfile | null = null;

export function installTickProfile(world: any = World, radius = parseInt(process.env.RL_NPC_RADIUS ?? '64', 10)): TickProfile {
    let totals: Totals = {};
    let ticks = 0, npcSamples = 0, awakeSum = 0, npcTotal = 0, since = performance.now();
    const add = (name: string, ms: number, bytes = 0) => {
        const t = totals[name] ?? (totals[name] = { ms: 0, count: 0, bytes: 0 });
        t.ms += ms;
        t.count++;
        t.bytes += bytes;
    };
    const wrap = (name: string, fn: (...args: any[]) => any) => function (this: any, ...args: any[]) {
        const t0 = performance.now();
        try {
            return fn.apply(this, args);
        } finally {
            add(name, performance.now() - t0);
        }
    };
    for (const name of PHASES) world[name] = wrap(name, world[name]);
    const vendorCycle = wrap('cycle', world.cycle);
    world.cycle = function (this: any, ...args: any[]) {
        ticks++;
        if (ticks % 50 === 0) { // a sample of the gate's split; counting every tick would cost a pass over 7,321 npcs
            const c = npcCounts(world, radius);
            npcSamples++;
            awakeSum += c.awake;
            npcTotal = c.total;
        }
        return vendorCycle.apply(this, args);
    };
    const time = <T>(name: string, fn: () => T): T => {
        const t0 = performance.now();
        const out = fn();
        if (out instanceof Promise) return out.finally(() => add(name, performance.now() - t0)) as T;
        add(name, performance.now() - t0);
        return out;
    };
    activeProfile = {
        add,
        time,
        wrapCaptured() {
            world.cycle = wrap('captured', world.cycle);
        },
        snapshot() {
            const round = (x: number) => Math.round(x * 1000) / 1000;
            return {
                schema: 'tick-profile-v1', ticks, wallMs: round(performance.now() - since), radius,
                players: world.getTotalPlayers(), walkCache: walkCacheStats(),
                npcs: npcTotal, awakeNpcs: npcSamples ? round(awakeSum / npcSamples) : null,
                memoryMiB: Object.fromEntries(Object.entries(process.memoryUsage()).map(([k, v]) => [k, Math.round(v / 1048576)])),
                totals: Object.fromEntries(Object.entries(totals).map(([k, v]) => [k, { ms: round(v.ms), count: v.count, bytes: v.bytes }]))
            };
        },
        reset() {
            totals = {};
            ticks = npcSamples = awakeSum = 0;
            since = performance.now();
        }
    };
    return activeProfile;
}
