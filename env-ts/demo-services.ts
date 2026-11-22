// Opt-in local demo services for the SAME world as engine-env. Never imported
// during ordinary GPU collection. Telemetry is out-of-band, not policy input.
import { Worker } from 'worker_threads';
import { nativeResourceHttp } from './engine/resource-capture-native';
import { demoSettings, viewRequests } from './core/spectator';
import { activeProfile } from './engine/tick-profile';
import { actorCount, installNativeSpectator, type RouteOf } from './engine/spectator-native';
import World from '../vendor/rs-sdk/server/engine/src/engine/World.js';
import Environment from '../vendor/rs-sdk/server/engine/src/util/Environment.js';
import TcpServer from '../vendor/rs-sdk/server/engine/src/server/tcp/TcpServer.js';
import { startWeb, startManagementWeb } from '../vendor/rs-sdk/server/engine/src/web.js';
import OpLocHandler from '../vendor/rs-sdk/server/engine/src/network/game/client/handler/OpLocHandler.js';
import MoveClickHandler from '../vendor/rs-sdk/server/engine/src/network/game/client/handler/MoveClickHandler.js';
import { isZoneAllocated } from '../vendor/rs-sdk/server/engine/src/engine/GameMap.js';

/**
 * shared: a shared world's id and capacity (engine-env RL_SHARED_AGENTS), whose demo spectator takes a view
 * (core/spectator.ts) by POST /demo-view; null in a one-agent world, which refuses one.
 */
export async function startDemoServices(routeOf?: RouteOf, shared: { id: string; capacity: number } | null = null): Promise<void> {
    if (Environment.node.production) throw new Error('demo services require a local non-production world');
    const port = Number(process.env.RL_TELEMETRY_PORT ?? 43610);
    if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('invalid telemetry port');
    const resourceCapture = nativeResourceHttp();
    // A demo world's watching login (scripts/stack.py --demo-spectator), hooked before any client connects.
    const demo = demoSettings(process.env);
    const spectator = demo ? installNativeSpectator(demo.spectator, routeOf) : null;
    // A shared demo world's view, from its command or a request (core/spectator.ts viewRequests): a tile with no map
    // (isZoneAllocated, which Player.teleport checks) is refused.
    const setView = viewRequests(spectator, shared, tile => isZoneAllocated(tile.level, tile.x, tile.z), () => World.currentTick,
        line => console.log(`[demo-services] ${line}`));
    // Diagnostics never enter observations sent to the policy. Preserve original
    // handler arguments/result and retain only a bounded ring of dispatch facts.
    const interactionEvents: object[] = [];
    const state = (p: any) => { try { return ({name:p.username, x:p.x, z:p.z, level:p.level,
        originX:p.originX, originZ:p.originZ,
        delayed:p.delayed, delayedUntil:p.delayedUntil, hasWaypoints:p.hasWaypoints(),
        target:p.target ? {x:p.target.x, z:p.target.z, type:p.target.type,
            id:p.target.type ?? p.target.id, constructor:p.target.constructor.name} : null,
        activeScript:!!p.activeScript, animId:p.animId});
        } catch (error) { return {telemetryError:String(error)}; } };
    const original = OpLocHandler.prototype.handle;
    OpLocHandler.prototype.handle = function(message, player) {
        const before = state(player);
        const accepted = original.call(this, message, player);
        interactionEvents.push({kind:'oploc', tick:World.currentTick, request:{op:message.op,x:message.x,z:message.z,loc:message.loc},
            accepted, before, after:state(player)});
        if (interactionEvents.length > 64) interactionEvents.shift();
        return accepted;
    };
    const originalMove = MoveClickHandler.prototype.handle;
    MoveClickHandler.prototype.handle = function(message, player) {
        const before = state(player);
        const accepted = originalMove.call(this, message, player);
        interactionEvents.push({kind:'move', tick:World.currentTick,
            request:{path:message.path, ctrlHeld:message.ctrlHeld, opClick:message.opClick},
            accepted, before, after:state(player)});
        if (interactionEvents.length > 64) interactionEvents.shift();
        return accepted;
    };
    if (Environment.easyStartup) {
        for (const service of ['login', 'friend', 'logger']) {
            new Worker(new URL(`../vendor/rs-sdk/server/engine/src/${service}.ts`, import.meta.url));
        }
    }
    new TcpServer().start();
    await startWeb();
    await startManagementWeb();
    const bootId = process.env.RL_DEMO_BOOT_ID ?? crypto.randomUUID();
    Bun.serve({
        hostname: '127.0.0.1', port,
        fetch(req) {
            if (new URL(req.url).pathname === '/resource-capture')
                return activeProfile ? activeProfile.time('http:resource-capture', () => resourceCapture(req)) : resourceCapture(req);
            if (new URL(req.url).pathname === '/demo-view' && req.method === 'POST') return setView(req);
            if (req.method !== 'GET' || new URL(req.url).pathname !== '/status') {
                return new Response('not found', { status: 404 });
            }
            return Response.json({
                schema: 1, bootId, tick: World.currentTick, tickRateMs: World.tickRate, lockstep: World.lockstep,
                monotonicMs: performance.now(), xpRate: Environment.node.xpRate,
                population: 'population-v2', players: World.getTotalPlayers(), actors: actorCount(),
                playerStates: World.players.filter(Boolean).map(state), interactionEvents,
                ...(spectator && { demo: { paceMs: demo!.paceMs, spectator: spectator.status() } }),
                provenance: 'same-process World.currentTick and resolved Environment.node.xpRate'
            }, { headers: { 'Cache-Control': 'no-store' } });
        }
    });
    console.log(`[demo-services] telemetry=http://127.0.0.1:${port}/status xpRate=${Environment.node.xpRate}`);
}
