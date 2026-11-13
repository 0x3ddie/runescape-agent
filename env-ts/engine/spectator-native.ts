// The demo spectator (env-ts/core/spectator.ts) on the engine's own objects. Installed only by
// demo-services in a world that declares a spectator; no vendor file changes.
import { installSpectator, type Tile } from '../core/spectator';
import { clickMarker, routeRequest, type RouteMark } from './click-marker';
import type Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { NetworkPlayer } from '../../vendor/rs-sdk/server/engine/src/engine/entity/NetworkPlayer.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import ClientGameProtRepository from '../../vendor/rs-sdk/server/engine/src/network/game/client/ClientGameProtRepository.js';
import VarPlayerType from '../../vendor/rs-sdk/server/engine/src/cache/config/VarPlayerType.js';
import InvType from '../../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import UpdateStat from '../../vendor/rs-sdk/server/engine/src/network/game/server/model/UpdateStat.js';
import * as rsbuf from '../../vendor/rs-sdk/server/engine/src/network/rsbuf/index.js';
import { Visibility } from '../../vendor/rs-sdk/server/engine/src/network/rsbuf/visibility.js';
import { PlayerInfoProt } from '../../vendor/rs-sdk/server/engine/src/network/rsbuf/prot.js';
import { CollisionFlag } from '../../vendor/rs-sdk/server/engine/src/engine/routefinder/index.js';
import { CoordGrid } from '../../vendor/rs-sdk/server/engine/src/engine/CoordGrid.js';
import { changeNpcCollision, changePlayerCollision, isFlagged } from '../../vendor/rs-sdk/server/engine/src/engine/GameMap.js';

const BLOCKWALK_NONE = 0; // BlockWalk.NONE, a const enum
const REFUSED = 3; // login response "Invalid username or password"
const HINT_CENTRE = 2; // HintArrow type 2: the arrow over the tile's centre

let installed: ReturnType<typeof installSpectator<Player>> | null = null;

/**
 * population-v2: the players in the world that are not the declared spectator. The certified-outcome
 * guards count these; in a world without a declared spectator it is every player, as in v1.
 */
export function actorCount() {
    return World.getTotalPlayers() - (installed?.count() ?? 0);
}

/** The actor's agent's route-v1 record; undefined when the agent publishes none (act-v2). */
export type RouteOf = (player: Player) => RouteMark | null | undefined;

export function installNativeSpectator(name: string, routeOf: RouteOf = () => undefined) {
    if (installed) throw new Error('spectator: already installed');
    const tutorial = VarPlayerType.getByName('tutorial');
    if (!tutorial) throw new Error('spectator: tutorial varp unavailable');
    const sent = new WeakMap<Player, { stats: number[]; levels: number[] }>(); // stats in each spectator's client
    const spectator = installed = installSpectator<Player>({
        world: World as any,
        handlers: (ClientGameProtRepository as any).handlers,
        players: () => World.players.filter(Boolean) as Player[],
        isNetwork: player => player instanceof NetworkPlayer,
        name: player => player.username,
        tile: player => ({ x: player.x, z: player.z, level: player.level }),
        refuse: player => World.forceLogout(player, REFUSED),
        restrain(player) {
            // Fields only: setVisibility clears the collision of the tile it stands on.
            player.visibility = Visibility.HARD;
            player.blockWalk = BLOCKWALK_NONE as any;
            // Post-tutorial, so no character-design modal covers the view.
            if (player.getVar(tutorial.id) !== 1000) player.setVar(tutorial.id, 1000);
            if (player.body.some(part => part !== -1)) {
                player.body.fill(-1); // no body parts: the client draws no model over the agent
                player.masks |= PlayerInfoProt.APPEARANCE;
            }
            player.activeScript = null;
            player.queue.clear();
            player.weakQueue.clear();
            player.engineQueue.clear();
            player.timers.clear();
        },
        moveTo: (player, tile: Tile) => player.teleport(tile.x, tile.z, tile.level),
        mirror(spectator, actor) {
            // Its inventory panel listens to the actor's inventory, as a trade screen does; updateInvs reads
            // it with Inventory.get, which resource capture does not wrap.
            const listener = spectator.invListeners.find(l => l.type === InvType.INV);
            if (listener && listener.source !== actor.uid) {
                listener.source = actor.uid;
                listener.firstSeen = true;
            }
            let shown = sent.get(spectator);
            if (!shown) sent.set(spectator, shown = { stats: [], levels: [] });
            for (let i = 0; i < actor.stats.length; i++) {
                if (actor.stats[i] !== shown.stats[i] || actor.levels[i] !== shown.levels[i]) {
                    spectator.write(new UpdateStat(i, actor.stats[i], actor.levels[i]));
                    shown.stats[i] = actor.stats[i];
                    shown.levels[i] = actor.levels[i];
                }
                // marked sent, so updateStats never writes its own stats over the actor's
                spectator.lastStats[i] = spectator.stats[i];
                spectator.lastLevels[i] = spectator.levels[i];
            }
        },
        spectatorOnlyNpcs(spectator, actor) {
            let count = 0;
            for (const npc of World.npcs) {
                if (rsbuf.hasNpc(spectator.slot, npc.nid) && !rsbuf.hasNpc(actor.slot, npc.nid)) count++;
            }
            return count;
        },
        collision: tile => ({ npc: isFlagged(tile.x, tile.z, tile.level, CollisionFlag.NPC),
            player: isFlagged(tile.x, tile.z, tile.level, CollisionFlag.PLAYER) }),
        setCollision: (tile, kind) => (kind === 'npc' ? changeNpcCollision : changePlayerCollision)(1, tile.x, tile.z, tile.level, true)
    }, name);
    const marker = installNativeMarker(name, routeOf);
    return { ...spectator, status: () => ({ ...spectator.status(), marker: marker.status() }) };
}

/**
 * The click marker (engine/click-marker.ts) at the start of each tick: the agent's route-v1 record when it
 * publishes one (act-v3). act-v2 routes every walk and act in the engine and keeps no record, so its request is
 * read from the engine: the tile of its interaction target (a loc, npc or ground item), else the end of its walk.
 */
function installNativeMarker(name: string, routeOf: RouteOf) {
    const marker = clickMarker<Player>({
        request(actor) {
            const route = routeOf(actor);
            if (route !== undefined) return routeRequest(route);
            const target = actor.target;
            const tile = target ? { x: target.x, z: target.z, level: target.level }
                : actor.hasWaypoints() ? { ...CoordGrid.unpackCoord(actor.waypoints[0]), level: actor.level } : null;
            return tile && { key: `${tile.level}:${tile.x}:${tile.z}`, tile, source: 'engine' as const };
        },
        show: (spectator, tile) => spectator.hintTile(HINT_CENTRE, tile.x, tile.z, 0),
        clear: spectator => spectator.stopHint()
    });
    const world = World as any;
    const processWorld = world.processWorld;
    if (typeof processWorld !== 'function') throw new Error('spectator marker: world has no processWorld');
    world.processWorld = function (this: unknown, ...args: unknown[]) {
        const players = World.players.filter(Boolean) as Player[];
        const watching = players.filter(player => player instanceof NetworkPlayer && player.username === name);
        const actors = players.filter(player => !watching.includes(player));
        marker.update(watching, actors.length === 1 ? actors[0]! : null);
        return processWorld.apply(this, args);
    };
    return marker;
}
