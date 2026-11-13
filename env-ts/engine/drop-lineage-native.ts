/** Native adapter for evaluator-only axe lineage. No fixture or policy mutations. */
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { NetworkPlayer } from '../../vendor/rs-sdk/server/engine/src/engine/entity/NetworkPlayer.js';
import Obj from '../../vendor/rs-sdk/server/engine/src/engine/entity/Obj.js';
import { CoordGrid } from '../../vendor/rs-sdk/server/engine/src/engine/CoordGrid.js';
import { EntityLifeCycle } from '../../vendor/rs-sdk/server/engine/src/engine/entity/EntityLifeCycle.js';
import { PlayerStat } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import ObjType from '../../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import ScriptProvider from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptProvider.js';
import ScriptRunner from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptRunner.js';
import { ScriptOpcode } from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptOpcode.js';
import { installDropLineage } from './drop-lineage';
import { monitorLifecycle } from './lifecycle';
import type { LifecycleIdentity } from './lifecycle-bridge';
import { recoverySnapshot, requireRecoveryDiagnostics } from './recovery-diagnostics';

let tracker: ReturnType<typeof installDropLineage> | null = null;
let installedBoot: string | null = null;
const owners = new WeakMap<Player, { identity: LifecycleIdentity; socket: unknown; session: unknown }>();
const enabled = () => process.env.RL_TEST_HOOKS === '1' && process.env.RL_CONTINUITY_EXPERIMENT === '1' &&
    process.env.RL_DROP_LINEAGE_EXPERIMENT === '1' && !!process.env.RL_DEMO_BOOT_ID &&
    (installedBoot === null || process.env.RL_DEMO_BOOT_ID === installedBoot);

export function installNativeDropLineage() {
    if (!enabled()) return;
    if (tracker) throw new Error('native drop lineage already installed');
    installedBoot = process.env.RL_DEMO_BOOT_ID!;
    const loss = ScriptProvider.getByName('[proc,player_death_lose_items]');
    if (!loss) throw new Error('exact death loss procedure unavailable');
    tracker = installDropLineage({
        world: World, handlers: ScriptRunner.HANDLERS, invDropAllOpcode: ScriptOpcode.INV_DROPALL,
        objTakeItemOpcode: ScriptOpcode.OBJ_TAKEITEM, deathLossProc: loss, despawnLifecycle: EntityLifeCycle.DESPAWN,
        tick: () => World.currentTick, enabled, production: () => Environment.node.production,
        playerCount: () => World.getTotalPlayers(), isPlayer: p => p instanceof Player, isObj: o => o instanceof Obj,
        validatePlayer(player: Player, identity) {
            const owner = owners.get(player);
            if (!owner || identity.worldBootId !== installedBoot || JSON.stringify(owner.identity) !== JSON.stringify(identity) ||
                World.getPlayerByUsername(player.username) !== player)
                throw new Error('native lineage account changed');
            if (player instanceof NetworkPlayer && (player.client !== owner.socket || player.session !== owner.session ||
                player.client.state === -1 || player.client.player !== player || player.loggingOut))
                throw new Error('native lineage network session changed');
        },
        verifyArm(player) { if (!recoverySnapshot(player).idle) throw new Error('arm requires idle native player'); },
        privateItems(player) { const s = recoverySnapshot(player); return { inventory: s.inventory, worn: s.worn, bank: s.bank }; },
        groundAt(level, x, z) { return World.gameMap.getZone(x, z, level).getObjsUnsafe(CoordGrid.packZoneCoord(x, z)); },
        targetIsNonstackable: () => !ObjType.get(1351).stackable,
        settings: () => ({ objDespawnScale: Environment.node.objDespawnScale, objRevealTicks: Environment.node.objRevealTicks })
    });
}

export function armNativeDropLineage(player: Player, identity: LifecycleIdentity) {
    requireRecoveryDiagnostics();
    if (!tracker || owners.has(player)) throw new Error('lineage missing or already bound');
    const root = ScriptProvider.getByName('[queue,player_death]');
    if (!root) throw new Error('death queue unavailable');
    const captured = { worldBootId: identity.worldBootId, accountId: identity.accountId,
        accountEpoch: identity.accountEpoch, incarnationId: identity.incarnationId };
    owners.set(player, { identity: captured, socket: player instanceof NetworkPlayer ? player.client : null,
        session: player instanceof NetworkPlayer ? player.session : null });
    tracker.bind(player, captured, monitorLifecycle(player, PlayerStat.HITPOINTS, () => World.currentTick, root));
    tracker.arm(player);
    return readNativeDropLineage(player);
}

export function readNativeDropLineage(player: Player) {
    requireRecoveryDiagnostics();
    if (!tracker) throw new Error('drop lineage unavailable');
    return { type: 'recovery_audit', audit: tracker.audit(player), snapshot: recoverySnapshot(player) };
}
