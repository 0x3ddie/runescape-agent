/** Evaluator-only snapshots for the registered recovery gate. Never policy input. */
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { PlayerStat } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import InvType from '../../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import Component from '../../vendor/rs-sdk/server/engine/src/cache/config/Component.js';
import { assertFixtureIdle } from '../core/fixture-audit';
import { randomBytes } from 'node:crypto';
import { createRecoveryPlacement, confirmPlacementSnapshot, requireRecoveryPlacementProfile, RecoveryPlacementRejected, placementTargets } from '../core/recovery-placement';

const placeOnce = createRecoveryPlacement();
const armed = new WeakSet<Player>();
const confirmations = new WeakMap<Player, { nonce:string; issuedTick:number; after:any; read:()=>any }>();
export function markRecoveryArmed(player: Player) { armed.add(player); }
export function recoveryPlace(player: Player, audit: () => any, counters: () => object, browser = false) {
    requireRecoveryDiagnostics();
    const host = {
        admission: () => ({ profile: process.env.RL_RECOVERY_FIXTURE_PROFILE,
            flags: process.env.RL_TEST_HOOKS === '1' && process.env.RL_CONTINUITY_EXPERIMENT === '1' && process.env.RL_DROP_LINEAGE_EXPERIMENT === '1',
            owned: !!process.env.RL_DEMO_BOOT_ID && audit().worldBootId === process.env.RL_DEMO_BOOT_ID,
            players: World.getTotalPlayers(), production: Environment.node.production }),
        snapshot: () => {
            const s = recoverySnapshot(player), a = audit();
            const { observedTick, currentPrivateItems, challenges, ...lifecycle } = a;
            const npcTargets=placementTargets(World.npcs,player);
            const queueCounts={normal:[...player.queue.all()].length,weak:[...player.weakQueue.all()].length,
                engine:[...player.engineQueue.all()].length};
            return { transport:browser ? 'browser' : 'engine', ...s.player, tick:s.tick, hp:s.player.hp, maxHp:player.baseLevels[PlayerStat.HITPOINTS],
                idle:s.idle, aggressors:npcTargets.count,npcTargets,queueCounts,
                queuedWork:Object.values(queueCounts).some(count=>count>0),
                armed:armed.has(player), inventory:s.inventory,worn:s.worn,bank:s.bank,rawStats:s.rawStats,
                levels:Array.from(player.levels),baseLevels:Array.from(player.baseLevels),lifecycle,...counters() };
        },
        place: (tile: {x:number;z:number;level:number}) => player.teleJump(tile.x,tile.z,tile.level)
    };
    let receipt;
    try { receipt = placeOnce(player, host); }
    catch (error) {
        if (error instanceof RecoveryPlacementRejected) return error.receipt;
        throw error; // Authentication, identity, spent and mutation failures remain errors.
    }
    if (!browser) return receipt;
    const nonce = randomBytes(16).toString('hex');
    confirmations.set(player, {nonce,issuedTick:World.currentTick,after:receipt.after,read:host.snapshot});
    player.messageGame('__RL_CONTINUITY_V1__:' + nonce);
    return { ...receipt, nonce, issuedTick:World.currentTick };
}

export function requireRecoveryDiagnostics() {
    if (process.env.RL_TEST_HOOKS !== '1' || process.env.RL_CONTINUITY_EXPERIMENT !== '1' ||
        process.env.RL_DROP_LINEAGE_EXPERIMENT !== '1' || !process.env.RL_DEMO_BOOT_ID ||
        Environment.node.production || World.getTotalPlayers() !== 1)
        throw new Error('recovery diagnostics require one actor in an owned nonproduction experiment');
}

export function recoverySnapshot(player: Player) {
    requireRecoveryDiagnostics();
    if (!(player instanceof Player) || World.getPlayerByUsername(player.username) !== player)
        throw new Error('recovery player identity mismatch');
    const state = { hp: player.levels[PlayerStat.HITPOINTS], isDead: player.levels[PlayerStat.HITPOINTS] <= 0,
        loggingOut: player.loggingOut, activeScript: player.activeScript, target: player.target,
        hasWaypoints: player.hasWaypoints(), bankOpen: player.modalMain === Component.getId('bank_main'),
        delayed: player.delayed, protect: player.protect };
    let idle = true;
    try { assertFixtureIdle(state); } catch { idle = false; }
    const inventory = (id: number) => {
        const inv = player.getInventory(id);
        if (!inv) throw new Error('missing recovery inventory');
        return Array.from({ length: inv.capacity }, (_, slot) => {
            const item = inv.get(slot); return item ? { slot, id: item.id, count: item.count } : null;
        }).filter(x => x !== null);
    };
    return { type: 'recovery_snapshot', tick: World.currentTick, idle,
        player: { x: player.x, z: player.z, level: player.level, hp: state.hp },
        busy: { activeScript: state.activeScript !== null, target: state.target !== null,
            waypoints: state.hasWaypoints, bankOpen: state.bankOpen, delayed: state.delayed,
            protect: state.protect, loggingOut: state.loggingOut },
        rawStats: Array.from(player.stats), woodcuttingXp: Math.floor(player.stats[PlayerStat.WOODCUTTING] / 10),
        inventory: inventory(InvType.INV), worn: inventory(InvType.WORN), bank: inventory(InvType.getId('bank')) };
}

/** Read-only post-decode confirmation of the same one-shot native receipt. */
export function confirmRecoveryPlace(player: Player) {
    requireRecoveryDiagnostics(); requireRecoveryPlacementProfile(process.env);
    const previous = confirmations.get(player);
    if (!previous) throw new Error('missing browser placement challenge');
    const snapshot = previous.read();
    confirmPlacementSnapshot(previous.after,snapshot);
    return {type:'recovery_place_confirm',nonce:previous.nonce,issuedTick:previous.issuedTick,snapshot};
}
