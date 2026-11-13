// sim/extract.ts: the static tables of the Draynor simulator, made by the engine's own code.
//
// Run only through sim/build.py, which places this file, env-ts/ and an APFS clone of
// vendor/rs-sdk side by side under logs/sim/work, so every import below resolves inside
// that private copy and nothing reads or writes the shared vendor checkout. The engine's
// configs and map are loaded as World.start() loads them, but no tick runs, no player is
// admitted and no port is opened: these are lookups, not a world.
//
// Everything the simulator would otherwise have to re-derive is computed here with the
// code that decides it in the native environment, on each level of LEVELS (the ground and
// the first floor, which Draynor's staircases reach):
//   - live collision (GameMap, doors as the map places them) with entity bits removed;
//   - the native path finder's breadth-first tree from every supported tile (an exact
//     replica of PathFinder.findPath1 that runs to exhaustion: the queue order does not
//     depend on the destination, so the path to any destination is this tree's);
//   - reach sets (ReachStrategy.reached) for every loc and for an entity on every tile;
//   - line of sight between tiles up to 10 apart;
//   - the SDK-snapshot walk map (env-ts/reach.ts) the affordances' steps come from, from every supported tile;
//   - the SDK snapshot's steps and its wall sides.
// A self-check replays random path queries through the engine's findPath* and through
// these tables and requires identical waypoints. Engine answers with doors and gates moved,
// and on the upper floor, are kept as samples for sim/tests to check the simulator's own search.
// act-v3 (native-click) dropped act-v2's compass moves and the SDK long routes its NAV walks planned.

import fs from 'node:fs';
import path from 'node:path';

import World from '../vendor/rs-sdk/server/engine/src/engine/World.js';
import FontType from '../vendor/rs-sdk/server/engine/src/cache/config/FontType.js';
import WordEnc from '../vendor/rs-sdk/server/engine/src/cache/wordenc/WordEnc.js';
import Midi from '../vendor/rs-sdk/server/engine/src/cache/midi/Midi.js';
import LocType from '../vendor/rs-sdk/server/engine/src/cache/config/LocType.js';
import NpcType from '../vendor/rs-sdk/server/engine/src/cache/config/NpcType.js';
import ObjType from '../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import EnumType from '../vendor/rs-sdk/server/engine/src/cache/config/EnumType.js';
import ParamType from '../vendor/rs-sdk/server/engine/src/cache/config/ParamType.js';
import HuntType from '../vendor/rs-sdk/server/engine/src/cache/config/HuntType.js';
import routefinder from '../vendor/rs-sdk/server/engine/src/engine/routefinder/index.js';
import { CollisionFlag, CollisionType } from '../vendor/rs-sdk/server/engine/src/engine/routefinder/flags.js';
import { canMove } from '../vendor/rs-sdk/server/engine/src/engine/routefinder/CollisionStrategy.js';
import { changeLocCollision, findPath, findPathToEntity, findPathToLoc, isApproached, isLineOfSight }
    from '../vendor/rs-sdk/server/engine/src/engine/GameMap.js';
import { initPathfinding } from '../vendor/rs-sdk/sdk/pathfinding';
import Environment from '../vendor/rs-sdk/server/engine/src/util/Environment.js';
import ScriptProvider from '../vendor/rs-sdk/server/engine/src/engine/script/ScriptProvider.js';
import ServerTriggerType from '../vendor/rs-sdk/server/engine/src/engine/script/ServerTriggerType.js';
import { walkMap } from '../env-ts/reach';
import * as sdkRsmod from '../vendor/rs-sdk/server/vendor/rsmod-pathfinder';
import { ACT_WALK_TILES } from '../env-ts/core/affordances';

// The supported area: player tiles the simulator models (98.9% of goal-v1 training episodes
// stayed inside it). DATA adds the 64-tile margin the native observation
// and the path finder's 128-tile window can reach.
const SUPPORT = { x0: 3040, x1: 3150, z0: 3180, z1: 3300 };
const MARGIN = 64;
const DATA = { x0: SUPPORT.x0 - MARGIN, x1: SUPPORT.x1 + MARGIN, z0: SUPPORT.z0 - MARGIN, z1: SUPPORT.z1 + MARGIN };
const SPAWN = { x: 3092, z: 3243 }; // config.VILLAGES.draynor spawnTele, reset seed 0
// Levels the tables cover. Supported tiles of a level are those reached from the spawn or
// from a staircase's destination on it (sim/build.py passes the stair scripts' cases).
const LEVELS = [0, 1];
// Path finder window (PathFinder.DEFAULT_SEARCH_MAP_SIZE), and the parts of it the tables keep.
const SEARCH = 128, HALF = 64;
const ORDER_R = 36;   // queue order kept within this many tiles: reach candidates of destinations <= 34 away
const DIST_R = 46;    // distances kept for the alternative-route search (10 around a destination)
const WALK_R = 34;    // SDK walk-map steps kept: affordances (15), places (32) and their approach tiles
const LOS_R = 10;
const ENTITY_BITS = CollisionFlag.NPC | CollisionFlag.PLAYER;

const [out, stairsFile] = process.argv.slice(2);
if (!out || !stairsFile) throw new Error('usage: bun run sim/extract.ts OUT_DIR STAIRS_JSON');
fs.mkdirSync(out, { recursive: true });
// script -> [[loc x, z, level, destination x, z, level]] (sim/build.py stairs)
const stairs: Record<string, number[][]> = JSON.parse(fs.readFileSync(stairsFile, 'utf8'));
const log = (m: string) => console.log(`[extract] ${m}`);

// ---- load configs and the map, as World.start() does, without starting the world
FontType.load('data/pack');
WordEnc.load('data/pack');
Midi.load();
World.reload();
World.gameMap.init();
log(`loaded; members=${Environment.node.members} production=${Environment.node.production} ` +
    `debug=${Environment.node.debug} clientRoutefinder=${Environment.node.clientRoutefinder} xpRate=${Environment.node.xpRate}`);

const flags = routefinder.collisionFlags;
const W = DATA.x1 - DATA.x0 + 1, H = DATA.z1 - DATA.z0 + 1;
const inData = (x: number, z: number) => x >= DATA.x0 && x <= DATA.x1 && z >= DATA.z0 && z <= DATA.z1;
// Grids over the data area, one after another by level, x-major within a level.
const tileIndex = (level: number, x: number, z: number) => level * W * H + (x - DATA.x0) * H + (z - DATA.z0);

// Npcs and players are moving collision; the tables use the map's own.
for (const level of LEVELS)
    for (let x = DATA.x0 - SEARCH; x <= DATA.x1 + SEARCH; x++)
        for (let z = DATA.z0 - SEARCH; z <= DATA.z1 + SEARCH; z++) {
            const f = flags.get(x, z, level);
            if (f !== CollisionFlag.NULL && (f & ENTITY_BITS)) flags.set(x, z, level, f & ~ENTITY_BITS);
        }

const collision = new Int32Array(LEVELS.length * W * H);
for (const level of LEVELS)
    for (let x = DATA.x0; x <= DATA.x1; x++) for (let z = DATA.z0; z <= DATA.z1; z++)
        collision[tileIndex(level, x, z)] = flags.get(x, z, level) | 0;
const unchanged = () => collision.every((f, i) => {
    const level = Math.floor(i / (W * H)), j = i % (W * H);
    return f === (flags.get(DATA.x0 + Math.floor(j / H), DATA.z0 + j % H, level) | 0);
});

// ---- locs, npcs, ground items
const locs: any[] = [];
const seenLoc = new Set<string>();
for (const level of LEVELS)
    for (let x = DATA.x0 - 8; x <= DATA.x1 + 8; x += 8)
        for (let z = DATA.z0 - 8; z <= DATA.z1 + 8; z += 8)
            for (const loc of World.gameMap.getZone(x, z, level).getAllLocsSafe()) {
                if (!inData(loc.x, loc.z)) continue;
                const key = `${level},${loc.x},${loc.z},${loc.type},${loc.shape},${loc.angle}`;
                if (seenLoc.has(key)) continue;
                seenLoc.add(key);
                locs.push({ type: loc.type, x: loc.x, z: loc.z, level, shape: loc.shape, angle: loc.angle, width: loc.width,
                            length: loc.length });
            }
const npcs: any[] = [];
for (const npc of World.npcs) {
    if (!LEVELS.includes(npc.level) || !inData(npc.startX, npc.startZ)) continue;
    npcs.push({ nid: npc.nid, type: npc.type, x: npc.startX, z: npc.startZ, level: npc.level, size: npc.width });
}
const objs: any[] = [];
for (const level of LEVELS)
    for (let x = DATA.x0 - 8; x <= DATA.x1 + 8; x += 8)
        for (let z = DATA.z0 - 8; z <= DATA.z1 + 8; z += 8)
            for (const obj of World.gameMap.getZone(x, z, level).getAllObjsSafe())
                if (inData(obj.x, obj.z)) objs.push({ type: obj.type, x: obj.x, z: obj.z, level, count: (obj as any).count ?? 1 });
for (const level of LEVELS)
    log(`level ${level}: ${locs.filter(l => l.level === level).length} locs, ${npcs.filter(n => n.level === level).length} npcs, ` +
        `${objs.filter(o => o.level === level).length} ground items in the data area`);

// ---- config records for every type the area or the slice can show
const params = (p: Map<number, any>) => Object.fromEntries([...p.entries()].map(([k, v]) => [ParamType.get(k)?.debugname ?? String(k), v]));
const locTypes: Record<number, any> = {};
const locType = (id: number) => {
    if (id < 0 || locTypes[id]) return;
    const t: any = LocType.get(id);
    locTypes[id] = { id, debugname: t.debugname, name: t.name, op: t.op ?? [], width: t.width, length: t.length,
        blockwalk: t.blockwalk, blockrange: t.blockrange, active: t.active, forceapproach: t.forceapproach,
        category: t.category, params: params(t.params) };
    // The stage a tree (next_loc_stage) or a mined rock (next_loc_stage_mining) changes into.
    for (const [k, v] of t.params ? [...t.params.entries()] : [])
        if (['next_loc_stage', 'next_loc_stage_mining'].includes(ParamType.get(k)?.debugname)) locType(v);
};
locs.forEach(l => locType(l.type));
const npcTypes: Record<number, any> = {};
const npcType = (id: number) => {
    if (npcTypes[id]) return;
    const t: any = NpcType.get(id);
    const hunt: any = t.huntmode >= 0 ? HuntType.get(t.huntmode) : null;
    npcTypes[id] = { id, debugname: t.debugname, name: t.name, op: t.op ?? [], vislevel: t.vislevel, size: t.size,
        wanderrange: t.wanderrange, maxrange: t.maxrange, huntrange: t.huntrange, huntmode: t.huntmode,
        hunt: hunt && { debugname: hunt.debugname, ...Object.fromEntries(Object.entries(hunt).filter(([k, v]) =>
            typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string')),
            checkVars: hunt.checkVars.filter((v: any) => v.varId !== -1) },
        givechase: t.givechase, attackrange: t.attackrange, moverestrict: t.moverestrict, blockwalk: t.blockwalk, timer: t.timer,
        respawnrate: t.respawnrate, defaultmode: t.defaultmode, regenrate: t.regenrate, stats: Array.from(t.stats ?? []),
        category: t.category, members: t.members, params: params(t.params) };
};
npcs.forEach(n => npcType(n.type));
// The compact world's enemies (baseline/resource_task.py MELEE_ENEMIES) that the Draynor map lacks.
const COMPACT_NPC_TYPES = [1017];  // chicken_brown
COMPACT_NPC_TYPES.forEach(npcType);
const objTypes: Record<number, any> = {};
const objType = (id: number) => {
    if (id < 0 || objTypes[id]) return;
    const t: any = ObjType.get(id);
    objTypes[id] = { id, debugname: t.debugname, name: t.name, op: t.op, iop: t.iop, cost: t.cost, stackable: t.stackable,
        certlink: t.certlink, certtemplate: t.certtemplate, wearpos: t.wearpos, wearpos2: t.wearpos2, wearpos3: t.wearpos3,
        weight: t.weight, members: t.members, category: t.category, tradeable: t.tradeable, dummyitem: t.dummyitem,
        respawnrate: t.respawnrate,
        params: params(t.params) };
    if (t.certlink >= 0) objType(t.certlink);
};
// Npc drops can be any obj, so every type is kept (names and stacking for ground items).
for (let id = 0; id < (ObjType as any).configs.length; id++) objType(id);

// Which script the engine runs for each trigger on each type (ScriptProvider.getByTrigger:
// the type's own, else its category's, else the default '_' script). The simulator runs
// only scripts it implements and reports the rest as coverage failures.
const triggerNames = Object.entries(ServerTriggerType).filter(([, v]) => typeof v === 'number') as [string, number][];
const triggersOf = (prefixes: string[], id: number, category: number) => Object.fromEntries(triggerNames
    .filter(([k]) => prefixes.some(p => k.startsWith(p)))
    .map(([k, v]) => [k, ScriptProvider.getByTrigger(v, id, category)?.name ?? null]).filter(([, s]) => s));
for (const t of Object.values(locTypes)) t.triggers = triggersOf(['APLOC', 'OPLOC'], t.id, t.category);
for (const t of Object.values(npcTypes)) t.triggers = triggersOf(['APNPC', 'OPNPC', 'AI_', 'OPPLAYER', 'APPLAYER'], t.id, t.category);
for (const t of Object.values(objTypes)) t.triggers = triggersOf(['APOBJ', 'OPOBJ', 'OPHELD'], t.id, t.category);
const triggerIds = Object.fromEntries(triggerNames);
// Defaults of every param a type here can read (ParamType.default): a type lists only those it sets.
const paramDefaults: Record<string, any> = {};
for (const t of [...Object.values(locTypes), ...Object.values(npcTypes), ...Object.values(objTypes)])
    for (const name of Object.keys(t.params)) paramDefaults[name] = null;
for (const name of ['attackrate', 'attackbonus', 'strengthbonus', 'stabdefence', 'slashdefence', 'crushdefence', 'magicdefence',
    'rangedefence', 'damagetype', 'death_drop', 'retreat', 'max_dealt', 'magicattack', 'attackrange', 'unbankable',
    'poison_severity', 'combat_xp_multiplier'])
    paramDefaults[name] = null;
for (const name of Object.keys(paramDefaults)) {
    const p: any = ParamType.getByName(name);
    paramDefaults[name] = p ? (p.isString() ? p.defaultString : p.defaultInt) : null;
}

// Draynor's fishing spots move between the coordinates of their enum (fishing_movement.rs2).
const enums: Record<string, any> = {};
for (const t of Object.values(npcTypes)) {
    const id = t.params.fishing_movement_enum;
    if (typeof id !== 'number' || enums[id]) continue;
    const e: any = EnumType.get(id);
    enums[id] = { id, debugname: e.debugname, values: [...e.values.entries()].map(([k, v]: [number, number]) =>
        [k, (v >> 14) & 0x3fff, v & 0x3fff, (v >> 28) & 0x3]) };
}
// Enums the npc death scripts read (clue scroll drops).
const namedEnums: Record<string, any> = {};
for (const name of ['trail_easy_enum', 'trail_medium_enum']) {
    const e: any = EnumType.getByName(name);
    if (e) namedEnums[name] = { inputtype: e.inputtype, outputtype: e.outputtype, defaultInt: e.defaultInt,
        values: [...e.values.entries()] };
}

// Multi-combat zones (GameMap.isMulti) over the data area, by zone origin and level.
const multiway: number[][] = [];
for (const level of LEVELS)
    for (let x = DATA.x0 & ~7; x <= DATA.x1; x += 8)
        for (let z = DATA.z0 & ~7; z <= DATA.z1; z += 8)
            if (World.gameMap.isMulti((level << 28) | (x << 14) | z)) multiway.push([x, z, level]);

// ---- the native path finder's tree, exhaustively (PathFinder.findPath1, srcSize 1, NORMAL)
const DIR = { North: 0x1, East: 0x2, South: 0x4, West: 0x8, SouthWest: 0xc, NorthWest: 0x9, SouthEast: 0x6, NorthEast: 0x3 };
const directions = new Int8Array(SEARCH * SEARCH);
const distances = new Int32Array(SEARCH * SEARCH);
const order = new Int32Array(SEARCH * SEARCH);
const qx = new Int32Array(4096), qz = new Int32Array(4096);
const can = (f: number, mask: number) => canMove(CollisionType.NORMAL, f, mask);

function tree(srcX: number, srcZ: number, level: number) {
    directions.fill(0); distances.fill(99_999_999); order.fill(-1);
    const baseX = srcX - HALF, baseZ = srcZ - HALF;
    const fl = (x: number, z: number) => flags.get(baseX + x, baseZ + z, level);
    let reader = 0, writer = 0, popped = 0;
    const append = (x: number, z: number, dir: number, dist: number) => {
        const i = x * SEARCH + z;
        directions[i] = dir; distances[i] = dist;
        qx[writer] = x; qz[writer] = z; writer = (writer + 1) & 4095;
    };
    append(HALF, HALF, 99, 0);
    const last = SEARCH - 1;
    while (writer !== reader) {
        const cx = qx[reader], cz = qz[reader];
        reader = (reader + 1) & 4095;
        order[cx * SEARCH + cz] = popped++;
        const next = distances[cx * SEARCH + cz] + 1;
        const empty = (x: number, z: number) => directions[x * SEARCH + z] === 0;
        let x = cx - 1, z = cz;
        if (cx > 0 && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_WEST)) append(x, z, DIR.East, next);
        x = cx + 1; z = cz;
        if (cx < last && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_EAST)) append(x, z, DIR.West, next);
        x = cx; z = cz - 1;
        if (cz > 0 && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_SOUTH)) append(x, z, DIR.North, next);
        x = cx; z = cz + 1;
        if (cz < last && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_NORTH)) append(x, z, DIR.South, next);
        x = cx - 1; z = cz - 1;
        if (cx > 0 && cz > 0 && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_SOUTH_WEST) &&
            can(fl(x, cz), CollisionFlag.BLOCK_WEST) && can(fl(cx, z), CollisionFlag.BLOCK_SOUTH)) append(x, z, DIR.NorthEast, next);
        x = cx + 1; z = cz - 1;
        if (cx < last && cz > 0 && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_SOUTH_EAST) &&
            can(fl(x, cz), CollisionFlag.BLOCK_EAST) && can(fl(cx, z), CollisionFlag.BLOCK_SOUTH)) append(x, z, DIR.NorthWest, next);
        x = cx - 1; z = cz + 1;
        if (cx > 0 && cz < last && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_NORTH_WEST) &&
            can(fl(x, cz), CollisionFlag.BLOCK_WEST) && can(fl(cx, z), CollisionFlag.BLOCK_NORTH)) append(x, z, DIR.SouthEast, next);
        x = cx + 1; z = cz + 1;
        if (cx < last && cz < last && empty(x, z) && can(fl(x, z), CollisionFlag.BLOCK_NORTH_EAST) &&
            can(fl(x, cz), CollisionFlag.BLOCK_EAST) && can(fl(cx, z), CollisionFlag.BLOCK_NORTH)) append(x, z, DIR.SouthWest, next);
    }
}

// Waypoints from the tree to the first dequeued tile `reached` accepts, else the closest
// approach point (PathFinder.findClosestApproachPoint), as PathFinder.findPath returns them.
function waypoints(srcX: number, srcZ: number, level: number, reached: (x: number, z: number) => boolean,
                   destX: number, destZ: number, rw: number, rh: number): number[] {
    const baseX = srcX - HALF, baseZ = srcZ - HALF;
    let best = -1, bx = -1, bz = -1;
    for (let x = 0; x < SEARCH; x++) for (let z = 0; z < SEARCH; z++) {
        const o = order[x * SEARCH + z];
        if (o >= 0 && (best < 0 || o < best) && reached(baseX + x, baseZ + z)) { best = o; bx = x; bz = z; }
    }
    if (best < 0) {
        let lowest = 1000, maxAlt = 100;
        const ldx = destX - baseX, ldz = destZ - baseZ;
        for (let x = ldx - 10; x <= ldx + 10; x++) for (let z = ldz - 10; z <= ldz + 10; z++) {
            if (x < 0 || x >= SEARCH || z < 0 || z >= SEARCH || distances[x * SEARCH + z] >= 100) continue;
            const dx = x < ldx ? ldx - x : x > ldx + rw - 1 ? x - (rw + ldx - 1) : 0;
            const dz = z < ldz ? ldz - z : z > ldz + rh - 1 ? z - (rh + ldz - 1) : 0;
            const cost = dx * dx + dz * dz;
            if (cost < lowest || (cost === lowest && maxAlt > distances[x * SEARCH + z])) {
                bx = x; bz = z; lowest = cost; maxAlt = distances[x * SEARCH + z];
            }
        }
        if (lowest === 1000) return [];
    }
    const wps: number[] = [];
    let cx = bx, cz = bz, next = directions[cx * SEARCH + cz], curr = -1;
    for (let i = 0; i < SEARCH * SEARCH; i++) {
        if (cx === HALF && cz === HALF) break;
        if (curr !== next) {
            curr = next;
            if (wps.length >= 25) wps.pop();
            wps.unshift(((level & 3) << 28) | (((baseX + cx) & 0x3fff) << 14) | ((baseZ + cz) & 0x3fff));
        }
        if (curr & DIR.East) cx += 1; else if (curr & DIR.West) cx -= 1;
        if (curr & DIR.North) cz += 1; else if (curr & DIR.South) cz -= 1;
        next = directions[cx * SEARCH + cz];
    }
    return wps;
}

// ---- doors and gates the simulator opens and closes (their scripts in sim/engine.py SCRIPTS)
const DOOR_SCRIPTS = new Set(['[oploc1,_door_closed]', '[oploc1,_door_opened]', '[oploc1,_gate_main_closed]',
    '[oploc1,_gate_outer_closed]', '[oploc1,_gate_main_open]', '[oploc1,_gate_outer_open]']);
const doors = locs.filter(l => DOOR_SCRIPTS.has(locTypes[l.type].triggers.OPLOC1));
// World.addLoc / removeLoc: a blocking loc's collision, as GameMap.changeLocCollision applies it.
const locCollision = (l: any, add: boolean) => {
    const t: any = LocType.get(l.type);
    if (t.blockwalk) changeLocCollision(l.shape, l.angle, t.blockrange, t.length, t.width, t.active, l.x, l.z, l.level, add);
};
// Saves the flags around some locs, so a temporary change can be undone exactly (flags are bit sets).
const saveFlags = (around: any[]) => {
    const saved: [number, number, number, number][] = [];
    for (const l of around) for (let x = l.x - 2; x <= l.x + 3; x++) for (let z = l.z - 2; z <= l.z + 3; z++)
        saved.push([x, z, l.level, flags.get(x, z, l.level)]);
    return () => { for (const [x, z, level, f] of saved.reverse()) flags.set(x, z, level, f); };
};

// ---- supported tiles: what a player walking from the spawn, or from where a staircase puts
// it, reaches without leaving SUPPORT on that level, with every door and gate above out of
// the way (the per-tile tables are of the boot map). Sorted by level, then x, then z.
const inSupport = (x: number, z: number) => x >= SUPPORT.x0 && x <= SUPPORT.x1 && z >= SUPPORT.z0 && z <= SUPPORT.z1;
const steps8 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]];
const arrivals = Object.values(stairs).flat().filter(c => inSupport(c[0], c[1]) && inSupport(c[3], c[4]) && LEVELS.includes(c[5]))
    .map(c => [c[3], c[4], c[5]]);
const sources: [number, number, number][] = [];
{
    const restore = saveFlags(doors);
    doors.forEach(l => locCollision(l, false));
    for (const level of LEVELS) {
        const starts = [...(level === 0 ? [[SPAWN.x, SPAWN.z]] : []), ...arrivals.filter(a => a[2] === level).map(a => [a[0], a[1]])];
        const seen = new Set<number>(starts.map(([x, z]) => x * 16384 + z));
        const queue = starts.map(([x, z]) => [x, z]);
        const found: [number, number, number][] = [];
        while (queue.length) {
            const [x, z] = queue.shift()!;
            found.push([x, z, level]);
            for (const [dx, dz] of steps8) {
                const nx = x + dx, nz = z + dz;
                if (!inSupport(nx, nz) || seen.has(nx * 16384 + nz)) continue;
                if (!routefinder.canTravel(level, x, z, dx, dz, 1, 0, CollisionType.NORMAL)) continue;
                seen.add(nx * 16384 + nz);
                queue.push([nx, nz]);
            }
        }
        found.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        sources.push(...found);
        log(`level ${level}: ${found.length} supported tiles from ${starts.length} starts ` +
            `(${doors.filter(d => d.level === level).length} doors and gates in the data area)`);
    }
    restore();
}

const N = sources.length;
const OW = 2 * ORDER_R + 1, DW = 2 * DIST_R + 1, WW = 2 * WALK_R + 1;
const parents = new Uint8Array(N * SEARCH * SEARCH);
const orders = new Uint16Array(N * OW * OW).fill(65535);
const dists = new Uint8Array(N * DW * DW).fill(255);
const walks = new Int16Array(N * WW * WW).fill(-1);
let maxDist = 0, maxOrder = 0;
const started = Date.now();
for (let s = 0; s < N; s++) {
    const [sx, sz, level] = sources[s];
    tree(sx, sz, level);
    for (let i = 0; i < SEARCH * SEARCH; i++) parents[s * SEARCH * SEARCH + i] = directions[i] & 0xff;
    for (let dx = -ORDER_R; dx <= ORDER_R; dx++) for (let dz = -ORDER_R; dz <= ORDER_R; dz++) {
        const o = order[(HALF + dx) * SEARCH + HALF + dz];
        if (o >= 0) { orders[s * OW * OW + (dx + ORDER_R) * OW + dz + ORDER_R] = o; maxOrder = Math.max(maxOrder, o); }
    }
    for (let dx = -DIST_R; dx <= DIST_R; dx++) for (let dz = -DIST_R; dz <= DIST_R; dz++) {
        const d = distances[(HALF + dx) * SEARCH + HALF + dz];
        if (d < 99_999_999) { dists[s * DW * DW + (dx + DIST_R) * DW + dz + DIST_R] = Math.min(d, 254); maxDist = Math.max(maxDist, d); }
    }
    const walk = walkMap(level, sx, sz, ACT_WALK_TILES);
    for (let dx = -WALK_R; dx <= WALK_R; dx++) for (let dz = -WALK_R; dz <= WALK_R; dz++)
        walks[s * WW * WW + (dx + WALK_R) * WW + dz + WALK_R] = walk.steps(sx + dx, sz + dz);
    if (s % 1000 === 0) log(`tables ${s}/${N} (${((Date.now() - started) / 1000).toFixed(0)} s)`);
}
log(`tables done in ${((Date.now() - started) / 1000).toFixed(0)} s; max distance ${maxDist}, max order ${maxOrder}`);
if (maxDist > 254 && maxDist < 99_999_999) log('warning: distances above 254 saturate (only the 100-step alternative route reads them)');

// ---- reach sets, as ReachStrategy decides them
const REACH = 4;  // tiles around a destination searched for reaching tiles
const locReach: number[][] = locs.map(l => {
    const t: any = LocType.get(l.type);
    const tiles: number[] = [];
    for (let x = l.x - REACH; x <= l.x + REACH + l.width; x++) for (let z = l.z - REACH; z <= l.z + REACH + l.length; z++)
        if (routefinder.reached(l.level, x, z, l.x, l.z, l.width, l.length, 1, l.angle, l.shape, t.forceapproach)) tiles.push(x, z);
    return tiles;
});
// An entity of size 1 on each tile (shape -2): which of the nine tiles around reach it.
const entityMask = (level: number, x: number, z: number) => {
    let mask = 0, bit = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++, bit++)
        if (routefinder.reached(level, x + dx, z + dz, x, z, 1, 1, 1, 0, -2, 0)) mask |= 1 << bit;
    return mask;
};
const entityReach = new Uint16Array(LEVELS.length * W * H);
for (const level of LEVELS)
    for (let x = DATA.x0; x <= DATA.x1; x++) for (let z = DATA.z0; z <= DATA.z1; z++)
        entityReach[tileIndex(level, x, z)] = entityMask(level, x, z);
// Line of sight (1x1 to 1x1) and isApproached (the same with the player-blocking bit, absent here).
const LW = 2 * LOS_R + 1;
const los = new Uint8Array(LEVELS.length * W * H * LW * LW);
let losDiff = 0;
for (const level of LEVELS)
    for (let x = DATA.x0; x <= DATA.x1; x++) for (let z = DATA.z0; z <= DATA.z1; z++)
        for (let dx = -LOS_R; dx <= LOS_R; dx++) for (let dz = -LOS_R; dz <= LOS_R; dz++) {
            const ok = isLineOfSight(level, x, z, x + dx, z + dz);
            los[tileIndex(level, x, z) * LW * LW + (dx + LOS_R) * LW + dz + LOS_R] = ok ? 1 : 0;
            if ((dx * 7 + dz) % 97 === 0 && ok !== isApproached(level, x, z, x + dx, z + dz, 1, 1, 1, 1)) losDiff++;
        }
log(`reach and line-of-sight tables done (isApproached differs from line of sight on ${losDiff} sampled pairs)`);

// ---- the SDK snapshot every NAV plans on (sdk/pathfinding.ts: the exported collision, doors open).
// The steps rsmod.canTravel allows from each tile of the long route's 2048-tile window around
// every source, as bits in its search order (W, E, S, N, SW, SE, NW, NE); unallocated zones are
// blocked. And env-ts/reach.ts walk.operable: which of a tile's four sides (W, E, S, N) has no
// wall between, over the data area. Both by level, as the other grids.
initPathfinding();
const SNAP = { x0: SUPPORT.x0 - 1024, x1: SUPPORT.x1 + 1023, z0: SUPPORT.z0 - 1024, z1: SUPPORT.z1 + 1023 };
const SNAP_W = SNAP.x1 - SNAP.x0 + 1, SNAP_H = SNAP.z1 - SNAP.z0 + 1;
const snapshot = new Uint8Array(LEVELS.length * SNAP_W * SNAP_H);
const LONG_STEPS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]];
for (const level of LEVELS)
    for (let zx = SNAP.x0 & ~7; zx <= SNAP.x1; zx += 8) for (let zz = SNAP.z0 & ~7; zz <= SNAP.z1; zz += 8) {
        if (!sdkRsmod.isZoneAllocated(zx, zz, level)) continue;
        for (let x = Math.max(zx, SNAP.x0); x < Math.min(zx + 8, SNAP.x1 + 1); x++)
            for (let z = Math.max(zz, SNAP.z0); z < Math.min(zz + 8, SNAP.z1 + 1); z++) {
                let bits = 0;
                LONG_STEPS.forEach(([dx, dz], k) => {
                    if (sdkRsmod.canTravel(level, x, z, dx, dz, 1, 0, sdkRsmod.CollisionType.NORMAL)) bits |= 1 << k;
                });
                snapshot[level * SNAP_W * SNAP_H + (x - SNAP.x0) * SNAP_H + (z - SNAP.z0)] = bits;
            }
    }
const SIDES: [number, number, number, number][] = [
    [-1, 0, sdkRsmod.CollisionFlag.WALL_EAST, sdkRsmod.CollisionFlag.WALL_WEST],
    [1, 0, sdkRsmod.CollisionFlag.WALL_WEST, sdkRsmod.CollisionFlag.WALL_EAST],
    [0, -1, sdkRsmod.CollisionFlag.WALL_NORTH, sdkRsmod.CollisionFlag.WALL_SOUTH],
    [0, 1, sdkRsmod.CollisionFlag.WALL_SOUTH, sdkRsmod.CollisionFlag.WALL_NORTH]];
const sides = new Uint8Array(LEVELS.length * W * H);
for (const level of LEVELS)
    for (let x = DATA.x0; x <= DATA.x1; x++) for (let z = DATA.z0; z <= DATA.z1; z++)
        SIDES.forEach(([dx, dz, side, own], k) => {
            if (!sdkRsmod.isFlagged(x + dx, z + dz, level, side) && !sdkRsmod.isFlagged(x, z, level, own))
                sides[tileIndex(level, x, z)] |= 1 << k;
        });
log('snapshot steps and sides done');

// ---- engine answers: the self-check of the tables against findPath*, and samples for sim/tests
const unpackWps = (a: ArrayLike<number>) => Array.from(a, n => n >>> 0);
let checked = 0, mismatched = 0;
const samples: any[] = [];
let seed = 20261014;
const rand = (n: number) => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed % n; };
const onLevel = (level: number) => sources.filter(s => s[2] === level);
const operable = (l: any) => (LocType.get(l.type) as any).op?.some((o: any) => o);

// One query from (sx, sz) on `level` through the engine and the tables: a tile, an entity, a loc
// near the source, or (on the upper floor) an entity's reach mask and a line of sight, which the
// stored grids answer.
function pathQuery(kind: number, level: number, sx: number, sz: number) {
    tree(sx, sz, level);
    let engine: any, ours: any, query: any;
    if (kind === 0) {
        const dx = sx + rand(69) - 34, dz = sz + rand(69) - 34;
        query = { kind: 'tile', level, sx, sz, x: dx, z: dz };
        engine = unpackWps(findPath(level, sx, sz, dx, dz));
        ours = waypoints(sx, sz, level, (x, z) => x === dx && z === dz, dx, dz, 1, 1);
    } else if (kind === 1) {
        const dx = sx + rand(41) - 20, dz = sz + rand(41) - 20;
        query = { kind: 'entity', level, sx, sz, x: dx, z: dz };
        engine = unpackWps(findPathToEntity(level, sx, sz, dx, dz, 1, 1, 1));
        ours = waypoints(sx, sz, level, (x, z) => routefinder.reached(level, x, z, dx, dz, 1, 1, 1, 0, -2, 0), dx, dz, 1, 1);
    } else if (kind === 2) {
        const near = locs.filter(l => l.level === level && Math.max(Math.abs(l.x - sx), Math.abs(l.z - sz)) <= 20 && operable(l));
        if (!near.length) return null;
        const l = near[rand(near.length)];
        const t: any = LocType.get(l.type);
        const rw = (l.angle & 1) ? l.length : l.width, rh = (l.angle & 1) ? l.width : l.length;
        query = { kind: 'loc', level, sx, sz, x: l.x, z: l.z, loc: locs.indexOf(l) };
        engine = unpackWps(findPathToLoc(level, sx, sz, l.x, l.z, 1, l.width, l.length, l.angle, l.shape, t.forceapproach));
        ours = waypoints(sx, sz, level, (x, z) => routefinder.reached(level, x, z, l.x, l.z, l.width, l.length, 1, l.angle, l.shape,
            t.forceapproach), l.x, l.z, rw, rh);
    } else if (kind === 3) {
        const x = sx + rand(7) - 3, z = sz + rand(7) - 3;
        query = { kind: 'reach', level, x, z };
        engine = entityMask(level, x, z);
        ours = entityReach[tileIndex(level, x, z)];
    } else {
        const tx = sx + rand(21) - 10, tz = sz + rand(21) - 10;
        query = { kind: 'los', level, x: sx, z: sz, tx, tz };
        engine = isLineOfSight(level, sx, sz, tx, tz);
        ours = los[tileIndex(level, sx, sz) * LW * LW + (tx - sx + LOS_R) * LW + tz - sz + LOS_R] === 1;
    }
    checked++;
    if (String(engine) !== String(ours)) mismatched++;
    return { ...query, [{ reach: 'mask', los: 'los' }[query.kind as string] ?? 'waypoints']: engine };
}

// ---- engine answers under changed collision, for sim/tests to replay through the simulator's own
// collision changes and on-the-fly search: each door and gate in SUPPORT on a level replaced by its
// next stage on the tile in front, turned a quarter (as [proc,door_open] moves a straight door), then
// all of that level's at once.
const DOOR_FRONT = [[-1, 0], [0, 1], [1, 0], [0, -1]];
function doorScenarios(level: number) {
    const scenarios: any[] = [];
    const supportDoors = doors.filter(l => l.level === level && inSupport(l.x, l.z) && locTypes[l.type].params.next_loc_stage !== undefined);
    const near = (x: number, z: number, set: any[], r: number) => set.some(l => Math.max(Math.abs(l.x - x), Math.abs(l.z - z)) <= r);
    for (const opened of supportDoors.length ? [...supportDoors.map(l => [l]), supportDoors] : []) {
        const from = onLevel(level).filter(([x, z]) => near(x, z, opened, 24));
        if (!from.length) continue;
        const added = opened.map(l => {
            const next = locTypes[l.type].params.next_loc_stage, t: any = LocType.get(next);
            return { type: next, x: l.x + DOOR_FRONT[l.angle][0], z: l.z + DOOR_FRONT[l.angle][1], level, shape: l.shape,
                     angle: (l.angle + 1) & 3, width: t.width, length: t.length };
        });
        const restore = saveFlags([...opened, ...added]);
        opened.forEach(l => locCollision(l, false));
        added.forEach(l => locCollision(l, true));
        const targets = [...locs.filter(l => l.level === level && operable(l) && near(l.x, l.z, opened, 24)), ...added];
        const queries: any[] = [];
        for (let i = 0; i < (opened.length > 1 ? 400 : 60); i++) {
            const [sx, sz] = from[rand(from.length)];
            const door = opened[rand(opened.length)];
            const kind = i % 5;
            if (kind === 0) {
                const x = sx + rand(49) - 24, z = sz + rand(49) - 24;
                queries.push({ kind: 'tile', sx, sz, x, z, waypoints: unpackWps(findPath(level, sx, sz, x, z)) });
            } else if (kind === 1) {
                const x = door.x + rand(13) - 6, z = door.z + rand(13) - 6;
                queries.push({ kind: 'entity', sx, sz, x, z, waypoints: unpackWps(findPathToEntity(level, sx, sz, x, z, 1, 1, 1)) });
            } else if (kind === 2) {
                const l = targets[rand(targets.length)], t: any = LocType.get(l.type);
                queries.push({ kind: 'loc', sx, sz, loc: { ...l, forceapproach: t.forceapproach },
                    waypoints: unpackWps(findPathToLoc(level, sx, sz, l.x, l.z, 1, l.width, l.length, l.angle, l.shape, t.forceapproach)) });
            } else if (kind === 3) {
                const x = door.x + rand(5) - 2, z = door.z + rand(5) - 2;
                queries.push({ kind: 'reach', x, z, mask: entityMask(level, x, z) });
            } else {
                const ax = door.x + rand(13) - 6, az = door.z + rand(13) - 6, bx = ax + rand(21) - 10, bz = az + rand(21) - 10;
                queries.push({ kind: 'los', x: ax, z: az, tx: bx, tz: bz, los: isLineOfSight(level, ax, az, bx, bz) });
            }
        }
        scenarios.push({ level, removed: opened.map(l => locs.indexOf(l)), added, queries });
        restore();
    }
    if (!unchanged()) throw new Error('door samples did not restore the collision map');
    return scenarios;
}

// ---- self-check and samples: the ground first, then the upper floors
const ground = onLevel(0);
for (let i = 0; i < 6000; i++) {
    const [sx, sz] = ground[rand(ground.length)];
    const sample = pathQuery(i % 3, 0, sx, sz);
    if (sample && samples.length < 3000) samples.push(sample);
}
const doorSamples = doorScenarios(0);
for (const level of LEVELS.filter(l => l > 0)) {
    const upper = onLevel(level);
    for (let i = 0; i < 1000; i++) {
        const [sx, sz] = upper[rand(upper.length)];
        const sample = pathQuery(i % 5, level, sx, sz);
        if (sample) samples.push(sample);
    }
    doorSamples.push(...doorScenarios(level));
}
log(`self-check: ${mismatched} of ${checked} path, reach and sight queries differ from the engine`);
log(`door samples: ${doorSamples.length} scenarios, ${doorSamples.reduce((n, s) => n + s.queries.length, 0)} queries`);

// ---- write
const bin = (name: string, a: ArrayBufferView) => fs.writeFileSync(path.join(out, name), new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
bin('collision.i32', collision);
bin('parents.u8', parents);
bin('orders.u16', orders);
bin('dists.u8', dists);
bin('walks.i16', walks);
bin('entity_reach.u16', entityReach);
bin('los.u8', los);
bin('snapshot.u8', snapshot);
bin('sides.u8', sides);
fs.writeFileSync(path.join(out, 'tables.json'), JSON.stringify({
    schema: 'draynor-sim-tables-v3', support: SUPPORT, data: DATA, snapshot_area: SNAP, spawn: SPAWN, levels: LEVELS,
    search: SEARCH, order_r: ORDER_R, dist_r: DIST_R, walk_r: WALK_R, los_r: LOS_R,
    environment: { members: Environment.node.members, production: Environment.node.production, debug: Environment.node.debug,
        clientRoutefinder: Environment.node.clientRoutefinder, xpRate: Environment.node.xpRate,
        randomEvents: (Environment as any).NODE_RANDOM_EVENTS, infiniteRun: (Environment as any).NODE_INFINITE_RUN,
        npcRadius: process.env.RL_NPC_RADIUS ?? '64' },
    trigger_ids: triggerIds, param_defaults: paramDefaults, sources, locs, loc_reach: locReach, npcs, objs, loc_types: locTypes, npc_types: npcTypes, obj_types: objTypes, enums,
    named_enums: namedEnums, multiway,
    self_check: { checked, mismatched }, path_samples: samples, door_samples: doorSamples,
    max_distance: maxDist, max_order: maxOrder
}));
log(`wrote ${out}`);
process.exit(mismatched ? 1 : 0);
