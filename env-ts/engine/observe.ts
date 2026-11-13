// env-ts/engine/observe.ts: build the observation from a clientless Player.
//
// PARITY IS THE WHOLE JOB HERE. The policy has never seen a pixel: the observation
// is a projection of game state and the browser was only the courier. So a
// checkpoint trained through the engine must see the SAME projection it would see
// through a client, field for field.
//
// Engine code can trivially see too much: the whole world, exact npc hp, spawns
// far past the client's scan. Every clamp below exists to give the policy only
// what a client could have told it. The numbers are read from the browser path:
//
//   locs        ±15 tiles   webclient/src/bot/StateCollector.ts scanNearbyLocs
//   groundItems ±15 tiles   webclient/src/bot/StateCollector.ts scanGroundItems
//   npcs        ±15 tiles, the server's view distance (rsbuf/build.ts:57): a client knows no farther npc (act-v3)
//   distance    Chebyshev tiles, max(|dx|,|dz|), ascending sort  StateCollector:797
//   caps        8 npcs / 12 locs / 6 ground, applied AFTER sorting
//   xp          player.stats[] is xp*10 (Player.ts:84 and the note at 1843)
//
// Affordances (core/affordances.ts) read the FULL scan, not the capped lists: 12
// village fences once outranked every tree in the 12-loc cap. Legality must match
// what the effector can actually do.

import InvType from '../../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import LocType from '../../vendor/rs-sdk/server/engine/src/cache/config/LocType.js';
import NpcType from '../../vendor/rs-sdk/server/engine/src/cache/config/NpcType.js';
import ObjType from '../../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import Loc from '../../vendor/rs-sdk/server/engine/src/engine/entity/Loc.js';
import Npc from '../../vendor/rs-sdk/server/engine/src/engine/entity/Npc.js';
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { PlayerStat } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import ScriptState from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptState.js';

import { N_INV_SEL } from '../core/actions';
import { ACT_SCAN_TILES, TAKE, type ActCandidate } from '../core/affordances';

/** Client scan radii. See the header: these mirror the browser path exactly. */
export const LOC_SCAN_TILES = 15;
export const GROUND_SCAN_TILES = 15;
export const NPC_SCAN_TILES = 15;

/** Observation caps, applied after the nearest-first sort. */
export const N_NPC_OBS = 8;
export const N_LOC_OBS = 12;
export const N_GROUND_OBS = 6;

/** engine stats[] are xp*10; the client reports whole xp */
export const XP_SCALE = 10;

// Canonical skill order. MUST equal rl/obs.py SKILL_ORDER, because obs.py looks
// skills up BY NAME and builds the 21-slot progression vector from it. Engine
// PlayerStat indices 0..17 line up; 18/19 are dead in this content rev (the engine
// calls them STAT18/STAT19) and 20 is Runecraft.
import { SKILL_NAMES } from '../core/skills';
export { SKILL_NAMES } from '../core/skills';

const chebyshev = (ax: number, az: number, bx: number, bz: number) => Math.max(Math.abs(ax - bx), Math.abs(az - bz));

/**
 * The RAW option array, holes and all. The index IS the op number minus one, and
 * that mapping is what an interaction trigger is built from
 * (`ServerTriggerType.APNPC1 + (op - 1)`, OpNpcHandler.ts:46).
 *
 * Compacting this array is a bug that hides: "Attack" is op 2 on a goblin, so
 * after dropping the null at index 0 it lands at index 0 and the effector fires
 * APNPC1 instead of APNPC2. Chopping and mining still worked, because their option
 * happens to be op 1, so the error was invisible until combat produced 0 xp over
 * 250 attacks. Keep the holes; compact only for display.
 */
const rawOpsOf = (t: unknown): (string | null)[] => {
    const ops = (t as { op?: (string | null)[] })?.op;
    if (!ops) return [];
    return ops.map(o => (o && o !== 'hidden' ? o : null));
};

/** Display list for the observation. The browser publishes the same compaction. */
const opsOf = (t: unknown): string[] => rawOpsOf(t).filter((o): o is string => !!o);

/** True op index (0-based) of the first option matching a pattern, or -1. */
export function opIndexOf(rawOps: (string | null)[], re: RegExp): number {
    return rawOps.findIndex(o => !!o && re.test(o));
}

const nameOf = (t: unknown): string => {
    const v = t as { name?: string; debugname?: string };
    return (v?.name ?? v?.debugname ?? '').toString();
};

/** Locs in the zones covering a tile radius. Zones are 8x8, so step by 8. */
function* scanLocs(player: Player, radius: number): Generator<Loc> {
    const seen = new Set<string>();
    const px = (player as any).x as number;
    const pz = (player as any).z as number;
    const level = (player as any).level as number;
    for (let dx = -radius - 8; dx <= radius + 8; dx += 8) {
        for (let dz = -radius - 8; dz <= radius + 8; dz += 8) {
            for (const loc of World.gameMap.getZone(px + dx, pz + dz, level).getAllLocsSafe()) {
                if (chebyshev(px, pz, loc.x, loc.z) > radius) continue;
                const key = `${loc.x},${loc.z},${loc.type}`;
                if (seen.has(key)) continue;
                seen.add(key);
                yield loc;
            }
        }
    }
}

export function* scanNpcs(player: Player, radius: number): Generator<Npc> {
    const seen = new Set<number>();
    const px = (player as any).x as number;
    const pz = (player as any).z as number;
    const level = (player as any).level as number;
    for (let dx = -radius - 8; dx <= radius + 8; dx += 8) {
        for (let dz = -radius - 8; dz <= radius + 8; dz += 8) {
            for (const npc of World.gameMap.getZone(px + dx, pz + dz, level).getAllNpcsSafe()) {
                if (chebyshev(px, pz, npc.x, npc.z) > radius) continue;
                if (seen.has(npc.nid)) continue;
                seen.add(npc.nid);
                yield npc;
            }
        }
    }
}

function* scanObjs(player: Player, radius: number): Generator<{ x: number; z: number; level: number; id: number; count: number }> {
    const px = (player as any).x as number;
    const pz = (player as any).z as number;
    const level = (player as any).level as number;
    for (let dx = -radius - 8; dx <= radius + 8; dx += 8) {
        for (let dz = -radius - 8; dz <= radius + 8; dz += 8) {
            for (const obj of World.gameMap.getZone(px + dx, pz + dz, level).getAllObjsSafe()) {
                if (chebyshev(px, pz, obj.x, obj.z) > radius) continue;
                yield { x: obj.x, z: obj.z, level, id: obj.type, count: (obj as any).count ?? 1 };
            }
        }
    }
}

export interface ScanResult {
    // `options` is the compacted display list the observation publishes; `rawOps`
    // keeps the holes so the effector can recover the true op number.
    locs: Array<{ loc: Loc; name: string; options: string[]; rawOps: (string | null)[]; distance: number }>;
    npcs: Array<{ npc: Npc; name: string; options: string[]; rawOps: (string | null)[]; distance: number; combatLevel: number }>;
    objs: Array<{ x: number; z: number; level: number; id: number; count: number; name: string; distance: number }>;
}

/** One scan per step, shared by the observation and the effector's target search. */
export function scanWorld(player: Player): ScanResult {
    const px = (player as any).x as number;
    const pz = (player as any).z as number;

    const locs = [...scanLocs(player, LOC_SCAN_TILES)]
        .map(loc => {
            const t = LocType.get(loc.type);
            return { loc, name: nameOf(t), options: opsOf(t), rawOps: rawOpsOf(t), distance: chebyshev(px, pz, loc.x, loc.z) };
        })
        // Ties break on position then id, so the same scene is always listed (and
        // truncated) in the same order.
        .sort((a, b) => a.distance - b.distance || a.loc.x - b.loc.x || a.loc.z - b.loc.z || a.loc.type - b.loc.type);

    const npcs = [...scanNpcs(player, NPC_SCAN_TILES)]
        .map(npc => {
            const t = NpcType.get(npc.type) as unknown as { vislevel?: number };
            return {
                npc,
                name: nameOf(t),
                options: opsOf(t),
                rawOps: rawOpsOf(t),
                distance: chebyshev(px, pz, npc.x, npc.z),
                combatLevel: t?.vislevel ?? 0
            };
        })
        .sort((a, b) => a.distance - b.distance || a.npc.x - b.npc.x || a.npc.z - b.npc.z || a.npc.nid - b.npc.nid);

    const objs = [...scanObjs(player, GROUND_SCAN_TILES)]
        .map(o => ({ ...o, name: nameOf(ObjType.get(o.id)), distance: chebyshev(px, pz, o.x, o.z) }))
        .sort((a, b) => a.distance - b.distance || a.x - b.x || a.z - b.z || a.id - b.id);

    return { locs, npcs, objs };
}

/** An affordance candidate and the scanned thing behind it, which the effector acts on. */
export type EngineCandidate = ActCandidate & ({ kind: 'loc'; entry: ScanResult['locs'][number] }
    | { kind: 'npc'; entry: ScanResult['npcs'][number] } | { kind: 'obj'; entry: ScanResult['objs'][number] });

/** Affordance candidates (core/affordances.ts): everything in the client's scan with an option. */
export function actCandidates(scan: ScanResult): EngineCandidate[] {
    return [
        ...scan.locs.map(entry => ({ kind: 'loc' as const, id: entry.loc.type, name: entry.name, options: entry.options,
            x: entry.loc.x, z: entry.loc.z, level: entry.loc.level, entry })),
        ...scan.npcs.filter(n => n.distance <= ACT_SCAN_TILES).map(entry => ({ kind: 'npc' as const, id: entry.npc.type,
            name: entry.name, options: entry.options, x: entry.npc.x, z: entry.npc.z, level: entry.npc.level,
            combatLevel: entry.combatLevel, entry })),
        ...scan.objs.map(entry => ({ kind: 'obj' as const, id: entry.id, name: entry.name, options: [TAKE], x: entry.x,
            z: entry.z, level: entry.level, entry }))
    ];
}

export interface ObsMeta {
    decision: number;
    done: boolean;
    truncated: boolean;
    isDead?: boolean;
}

export function buildObs(player: Player, scan: ScanResult, meta: ObsMeta): Record<string, any> {
    const p = player as any;
    const px = p.x as number;
    const pz = p.z as number;

    const skills = SKILL_NAMES.map((name, i) => ({
        name,
        level: player.levels[i] ?? 1,
        baseLevel: player.baseLevels[i] ?? 1,
        xp: Math.floor((player.stats[i] ?? 0) / XP_SCALE)
    }));

    const invContainer = player.getInventory(InvType.INV);
    const inventory: Array<Record<string, any>> = [];
    if (invContainer) {
        for (let slot = 0; slot < invContainer.capacity; slot++) {
            const item = invContainer.get(slot);
            if (!item) continue;
            const t = ObjType.get(item.id);
            inventory.push({ slot, id: item.id, name: nameOf(t), count: item.count,
                options: (t.iop ?? []).filter((o): o is string => !!o && o !== 'hidden') });
        }
    }

    const wornContainer = player.getInventory(InvType.WORN);
    const equipment: Array<Record<string, any>> = [];
    if (wornContainer) {
        for (let slot = 0; slot < wornContainer.capacity; slot++) {
            const item = wornContainer.get(slot);
            if (!item) continue;
            equipment.push({ slot, id: item.id, name: nameOf(ObjType.get(item.id)) });
        }
    }

    const npcs = scan.npcs.slice(0, N_NPC_OBS).map(n => ({
        id: n.npc.nid,
        name: n.name,
        combatLevel: n.combatLevel,
        dx: n.npc.x - px,
        dz: n.npc.z - pz,
        distance: n.distance,
        inCombat: !!(n.npc as any).target,
        options: n.options
    }));

    const locs = scan.locs.slice(0, N_LOC_OBS).map(l => ({
        id: l.loc.type,
        name: l.name,
        dx: l.loc.x - px,
        dz: l.loc.z - pz,
        distance: l.distance,
        options: l.options
    }));

    const ground = scan.objs.slice(0, N_GROUND_OBS).map(o => ({
        id: o.id,
        name: o.name,
        count: o.count,
        dx: o.x - px,
        dz: o.z - pz,
        distance: o.distance,
        x: o.x,
        z: o.z
    }));

    const invSel = inventory.slice(0, N_INV_SEL).map(i => ({ slot: i.slot, id: i.id, name: i.name, count: i.count, options: i.options }));

    // A suspended script IS the dialog, engine-side. `resumeButtons` holds the
    // registered choice components; an empty list with a suspended script is a
    // continue-only page. The option TEXT lives in the client's rendered
    // interface and has no engine-side equivalent, so text is empty and the mask
    // treats these as real choices, which they are.
    const suspended = player.activeScript?.execution === ScriptState.PAUSEBUTTON;
    const dialogOptions = suspended ? player.resumeButtons.map(comId => ({ text: '', componentId: comId })) : [];
    const modalOpen = player.modalState !== 0;

    const maxHp = player.baseLevels[PlayerStat.HITPOINTS] ?? 10;
    const hp = player.levels[PlayerStat.HITPOINTS] ?? 0;

    // COMBAT, not "has an interaction target". `player.target` is set by ANY
    // interaction, so keying combat off it reported inCombat=true while chopping a
    // tree, which made RETREAT legal against a tree and put a wrong value in the
    // observation field obs.py feeds to player_vec[7]. The behavioural parity gate
    // caught it: the engine spent 7.7% of its chosen actions on RETREAT where the
    // browser spent 0%. Fishing spots are NPCs too, so `target instanceof Npc` is
    // equally wrong. What RETREAT and EAT actually care about is being attacked, so
    // that is what this reports: an NPC in range whose own target is this player.
    const attackers = scan.npcs.filter(n => (n.npc as unknown as { target?: unknown }).target === player);

    return {
        type: 'obs',
        tick: World.currentTick,
        decision: meta.decision,
        done: meta.done,
        truncated: meta.truncated,
        ground,
        invSel,
        dialogOptions,
        player: {
            x: px,
            z: pz,
            level: p.level ?? 0,
            hp,
            maxHp,
            combatLevel: p.combatLevel ?? 0,
            animId: p.animId ?? -1,
            isDead: meta.isDead ?? hp <= 0,
            inCombat: attackers.length > 0,
            targetIndex: attackers.length > 0 ? attackers[0].npc.nid : -1
        },
        skills,
        inventory,
        equipment,
        npcs,
        locs,
        bankOpen: player.modalMain !== -1 && scan.locs.some(l => /bank|booth/i.test(l.name) && l.distance <= 3),
        dialogOpen: suspended || player.modalChat !== -1,
        modalOpen,
        gameMessages: []
    };
}
