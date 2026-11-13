// env-ts/engine-env.ts: the environment server, inside the game engine.
//
// Boots ONE world in this process and serves reinforcement-learning agents over
// TCP, each a clientless `Player`. No Chromium, no gateway, no game protocol.
//
//   cd vendor/rs-sdk/server/engine
//   bun run ../../../../env-ts/engine-env.ts
//
// Protocol: `PROTO|`-prefixed NDJSON with reset / step / close, read by
// rl/vec_adapter.py. ONE CONNECTION IS ONE AGENT, which keeps the Python side
// synchronous; a clientless player costs a game object, not a browser.
//
// SCALING IS HORIZONTAL, NOT VERTICAL. Agents in one world share one tick stream,
// so past ~16 the world stops delivering the tick rate it promises (measured: 32
// agents get 112 of 125 ticks/s, 64 get 78). A world that under-delivers makes
// every wall-clock budget mean a different amount of game time, which is the same
// trap that ruled out 5ms ticks. Run MORE WORLDS instead, one process each, and
// keep every world honest. It also dilutes resource contention between agents.
//
// Env vars:
//   RL_ENV_PORT   listen port                    (default 43595)
//   RL_SPEED_MS   ms per world tick              (default 8)
//   RL_WORLD_ID   world label, MUST be unique per process when running several.
//                 Agent usernames are derived from it, and the engine writes
//                 `data/players/<profile>/<username>.sav` on logout, identical
//                 names across worlds would race on the same save files.

import './engine-cwd-guard';

import World from '../vendor/rs-sdk/server/engine/src/engine/World.js';
import { EngineAgent } from './engine/agent';
import { K_TICKS } from './core/budgets';
import { PlayerStat } from '../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import ScriptProvider from '../vendor/rs-sdk/server/engine/src/engine/script/ScriptProvider.js';
import ScriptRunner from '../vendor/rs-sdk/server/engine/src/engine/script/ScriptRunner.js';
import Environment from '../vendor/rs-sdk/server/engine/src/util/Environment.js';
import VarPlayerType from '../vendor/rs-sdk/server/engine/src/cache/config/VarPlayerType.js';
import InvType from '../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import ObjType from '../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import { installNativeDropLineage, armNativeDropLineage, readNativeDropLineage } from './engine/drop-lineage-native';
import { installNativeResourceCapture, nativeResourceControl } from './engine/resource-capture-native';
import { recoverySnapshot, recoveryPlace, markRecoveryArmed } from './engine/recovery-diagnostics';
import { demoSettings } from './core/spectator';

const PORT = parseInt(process.env.RL_ENV_PORT ?? '43595', 10);
const SPEED_MS = parseInt(process.env.RL_SPEED_MS ?? '8', 10);
const WORLD_ID = (process.env.RL_WORLD_ID ?? '0').replace(/[^a-z0-9]/gi, '').slice(0, 2);
// RL_LOCKSTEP=1: the world ticks only when its one agent steps (World.advance), as fast
// as the work allows and frozen while the controller thinks. Ticks per decision are the
// same as in real time; only wall-clock time changes.
const LOCKSTEP = process.env.RL_LOCKSTEP === '1';
// A demo world (scripts/stack.py --demo-spectator, env-ts/core/spectator.ts): lockstep, paced, watched.
const DEMO = demoSettings(process.env);

let agentSeq = 0;
let admitted = false; // an agent has been admitted; a demo world's idle ticks have stopped
const agents = new Set<EngineAgent>();

interface Session {
    agent: EngineAgent | null;
    buf: string;
    /** serialises request handling per connection */
    chain: Promise<void>;
}

const log = (msg: string) => console.log(`[engine-env] ${msg}`);

function send(socket: { write(data: string): number }, msg: object): void {
    socket.write('PROTO|' + JSON.stringify(msg) + '\n');
}

async function handle(socket: any, session: Session, line: string): Promise<void> {
    let req: any;
    try {
        req = JSON.parse(line);
    } catch (e) {
        send(socket, { type: 'error', message: `bad json: ${(e as Error).message}` });
        return;
    }

    const agent = session.agent;
    if (!agent) {
        send(socket, { type: 'error', message: 'no agent on this connection' });
        return;
    }

    try {
        switch (req.cmd) {
            case 'runtime_info': {
                const tutorial = VarPlayerType.getByName('tutorial');
                send(socket, { type: 'runtime_info', tick: World.currentTick, worldId: WORLD_ID,
                    tickRateMs: World.tickRate, xpRate: Environment.node.xpRate,
                    testHooksEnabled: process.env.RL_TEST_HOOKS === '1',
                    continuityExperimentEnabled: process.env.RL_CONTINUITY_EXPERIMENT === '1',
                    production: Environment.node.production, staffModLevel: agent.player.staffModLevel,
                    tutorial: tutorial ? agent.player.getVar(tutorial.id) : null });
                break;
            }
            case 'test_hook': {
                // Fault injection for the registered mechanics gate only. This
                // is not a policy action and is disabled in ordinary training.
                if (process.env.RL_TEST_HOOKS !== '1') throw new Error('test hooks disabled');
                if (req.operation === 'resource_capture') {
                    if(Object.keys(req).length!==3 || !req.request || req.request.accountName!==agent.player.username)
                        throw new Error('resource request must target this connection actor');
                    send(socket,nativeResourceControl()(req.request));
                    break;
                }
                if (req.operation === 'combat_varps') {
                    // Read-only mechanics evidence: the varps [proc,player_combat_stat] writes, and the levels behind them.
                    const p = agent.player;
                    const names = ['com_mode', 'com_maxhit', 'com_stabattack', 'com_slashattack', 'com_crushattack', 'com_stabdef',
                        'damagestyle', 'damagetype'];
                    const varps = Object.fromEntries(names.map(name => {
                        const varp = VarPlayerType.getByName(name);
                        return [name, varp ? p.getVar(varp.id) : null];
                    }));
                    const stats = { attack: PlayerStat.ATTACK, strength: PlayerStat.STRENGTH, defence: PlayerStat.DEFENCE,
                        hitpoints: PlayerStat.HITPOINTS };
                    const levels = Object.fromEntries(Object.entries(stats).map(([name, stat]) => [name, [p.levels[stat], p.baseLevels[stat]]]));
                    send(socket, { type: 'combat_varps', tick: World.currentTick, varps, levels });
                    break;
                }
                if (req.operation === 'lifecycle_snapshot') {
                    send(socket, { type: 'lifecycle_snapshot', audit: agent.continuityAudit(req.afterSequence ?? 0) });
                    break;
                }
                if (['recovery_snapshot','recovery_arm','recovery_audit','recovery_place'].includes(req.operation)) {
                    if (Object.keys(req).length !== 2) throw new Error('unexpected recovery fields');
                    const identity = agent.continuityAudit();
                    if (req.operation === 'recovery_place') {
                        send(socket, recoveryPlace(agent.player, () => agent.continuityAudit(), () => agent.recoveryPlacementState()));
                        break;
                    }
                    if (req.operation === 'recovery_arm') markRecoveryArmed(agent.player);
                    send(socket, req.operation === 'recovery_snapshot' ? recoverySnapshot(agent.player) :
                        req.operation === 'recovery_arm' ? armNativeDropLineage(agent.player, identity) : readNativeDropLineage(agent.player));
                    break;
                }
                const p = agent.player;
                const stat = PlayerStat.HITPOINTS;
                const before = p.levels[stat];
                switch (req.operation) {
                    case 'food_fixture': {
                        // Explicit mechanics-only inventory, never a policy action
                        // or ordinary reset curriculum. Preserve request order.
                        const fixtures: Record<string, string[]> = {
                            raw_before_bread: ['raw_shrimp', 'burnt_meat', 'bread'],
                            raw_only: ['raw_shrimp', 'burnt_meat'],
                            unsupported_cooked: ['anchovies']
                        };
                        const names = fixtures[req.fixture];
                        if (!Array.isArray(names)) throw new Error('unknown food fixture');
                        const ids = names.map(name => ObjType.getId(name));
                        if (ids.some(id => id < 0)) throw new Error('food fixture item unavailable');
                        p.invClear(InvType.INV);
                        for (const id of ids) p.invAdd(InvType.INV, id, 1);
                        break;
                    }
                    case 'damage_three': p.applyDamage(3, 0); break;
                    case 'lethal': p.applyDamage(before, 0); break;
                    case 'scripted_lethal': {
                        // The real game damage procedure queues player_death.
                        const script = ScriptProvider.getByName('[proc,damage_self]');
                        if (!script) throw new Error('damage_self script unavailable');
                        p.executeScript(ScriptRunner.init(script, p, null, [before]), true);
                        break;
                    }
                    case 'lethal_then_heal':
                        p.applyDamage(before, 0);
                        p.levels[stat] = p.baseLevels[stat];
                        p.changeStat(stat);
                        break;
                    case 'drain_then_heal':
                        p.levels[stat] = 0;
                        p.changeStat(stat);
                        p.levels[stat] = p.baseLevels[stat];
                        p.changeStat(stat);
                        break;
                    default: throw new Error('unknown test operation');
                }
                send(socket, { type: 'ok', scope: 'mechanics fault injection; not learned behavior',
                    operation: req.operation, tick: World.currentTick, beforeHp: before, afterHp: p.levels[stat],
                    queuedScripts: [...p.queue.all()].map(request => request.script.name) });
                break;
            }
            case 'reset':
                send(socket, await agent.reset(req.seed ?? 0, req.cfg ?? {}));
                break;
            case 'step':
                send(socket, await agent.step(req.action ?? 0));
                break;
            case 'cheat': {
                // The only world-level cheat the harness needs is ::speed, which
                // vec_env issues once from env 0 to set the tick rate. Inside the
                // engine that is a field, not a packet typed into a chat box.
                const m = /^::speed\s+(\d+)/.exec(String(req.text ?? ''));
                if (m) {
                    World.tickRate = parseInt(m[1], 10);
                    log(`tickRate set to ${World.tickRate}ms`);
                }
                send(socket, { type: 'ok', cheat: req.text ?? '' });
                break;
            }
            case 'hud':
                send(socket, { type: 'ok' }); // no rendered client to overlay
                break;
            case 'close':
                send(socket, { type: 'closing' });
                agent.close();
                agents.delete(agent);
                session.agent = null;
                socket.end();
                break;
            default:
                send(socket, { type: 'error', message: `unknown cmd ${req.cmd}` });
        }
    } catch (e) {
        send(socket, { type: 'error', message: (e as Error)?.message ?? String(e) });
    }
}

async function main(): Promise<void> {
    if (DEMO && (!LOCKSTEP || process.env.RL_DEMO_SERVICES !== '1'))
        throw new Error('a demo world is a lockstep world with its demo services');
    log('booting world...');
    await World.start();
    installNativeDropLineage();
    installNativeResourceCapture();
    World.tickRate = SPEED_MS;
    World.lockstep = LOCKSTEP;
    log(`world up, tickRate=${World.tickRate}ms${LOCKSTEP ? ', lockstep' : ''}`);

    // A visible, local client can join this same world for transport checks.
    // Default GPU training never imports or starts these services.
    if (process.env.RL_DEMO_SERVICES === '1') {
        const { startDemoServices } = await import('./demo-services');
        // A demo spectator's click marker reads the actor's agent's route-v1 record. A player that no agent holds
        // yet (its admission's ticks, a reset's replacement) has no request.
        await startDemoServices(player => {
            const agent = [...agents].find(a => a.player === player);
            return agent ? agent.lastRoute : null;
        });
    }

    Bun.listen<Session>({
        hostname: '127.0.0.1',
        port: PORT,
        socket: {
            open(socket) {
                // 12-char RS username limit; w<world><seq> stays well inside it.
                const name = `w${WORLD_ID}a${agentSeq++}`;
                if (LOCKSTEP && agents.size > 0) {
                    socket.data = { agent: null, buf: '', chain: Promise.resolve() };
                    send(socket, { type: 'error', message: 'a lockstep world serves one agent' });
                    socket.end();
                    return;
                }
                admitted = true;
                const agent = new EngineAgent(name);
                if (LOCKSTEP) World.advance(4); // process the login, as real time would before reset
                agents.add(agent);
                socket.data = { agent, buf: '', chain: Promise.resolve() };
                log(`agent ${name} admitted (${agents.size} live) at tick ${World.currentTick}`);
                send(socket, { type: 'ready', bot: name, kTicks: K_TICKS });
            },
            data(socket, chunk) {
                const s = socket.data;
                if (!s.agent) return; // refused at open
                s.buf += chunk.toString();
                let nl: number;
                while ((nl = s.buf.indexOf('\n')) !== -1) {
                    const line = s.buf.slice(0, nl).trim();
                    s.buf = s.buf.slice(nl + 1);
                    if (!line) continue;
                    // Requests on one connection are strictly sequential: the
                    // Python side is synchronous and expects one reply per line.
                    s.chain = s.chain.then(() => handle(socket, s, line));
                }
            },
            close(socket) {
                const s = socket.data;
                if (s?.agent) {
                    s.agent.close();
                    agents.delete(s.agent);
                    log(`agent ${s.agent.name} released (${agents.size} live)`);
                }
            },
            error(socket, err) {
                log(`socket error: ${err?.message ?? err}`);
                const s = socket.data;
                if (s?.agent) {
                    s.agent.close();
                    agents.delete(s.agent);
                }
            }
        }
    });

    log(`world ${WORLD_ID} listening on 127.0.0.1:${PORT}`);

    if (DEMO) {
        // Until its agent is admitted, a demo world ticks at the demo pace so that the
        // spectator can log in; from then on only the agent's steps advance it.
        log(`demo world: spectator ${DEMO.spectator}, ${DEMO.paceMs} ms a tick`);
        void (async () => {
            while (!admitted) {
                World.advance(1);
                await new Promise(r => setTimeout(r, DEMO.paceMs));
            }
        })();
    }

    // RL_CYCLE_STATS=<n>: every n ticks, report where the tick budget goes.
    // cycleStats is a Uint16Array of whole milliseconds, so this is a coarse
    // share, not a profile: enough to size the NPC lever, not to tune it.
    const statEvery = parseInt(process.env.RL_CYCLE_STATS ?? '0', 10);
    if (statEvery > 0) {
        let n = 0, cyc = 0, npc = 0, plr = 0, zone = 0, world = 0;
        setInterval(() => {
            n++;
            cyc += World.lastCycleStats[0];
            npc += World.lastCycleStats[3];
            plr += World.lastCycleStats[4];
            zone += World.lastCycleStats[7];
            world += World.lastCycleStats[1];
            if (n >= statEvery) {
                log(`cyclestats n=${n} cycle=${(cyc / n).toFixed(2)}ms npc=${(npc / n).toFixed(2)}ms `
                    + `player=${(plr / n).toFixed(2)}ms zone=${(zone / n).toFixed(2)}ms world=${(world / n).toFixed(2)}ms `
                    + `npcs=${World.getTotalNpcs()} players=${World.getTotalPlayers()} agents=${agents.size} `
                    + `tick=${World.currentTick} unix_ms=${Date.now()}`);
                n = cyc = npc = plr = zone = world = 0;
            }
        }, SPEED_MS);
    }
}

process.on('SIGINT', () => {
    for (const a of agents) a.close();
    process.exit(0);
});
process.on('SIGTERM', () => {
    for (const a of agents) a.close();
    process.exit(0);
});

main().catch(e => {
    console.error('[engine-env] fatal:', e);
    process.exit(1);
});
