import { expect, test } from 'bun:test';
import { TUTORIAL_ISLAND, beside, demoSettings, idleTicks, inIslandChunks, installSpectator, parseView, sharedPace,
    viewRequests, wakesIslandNpcs, type Tile } from './spectator';

interface Fake { name: string; network: boolean; tile: Tile; restrained: number }

function world() {
    const at = (x: number, z: number, level = 0): Tile => ({ x, z, level });
    const flags = new Map<string, { npc: boolean; player: boolean }>();
    const key = (t: Tile) => `${t.level}:${t.x}:${t.z}`;
    const flagsAt = (t: Tile) => flags.get(key(t)) ?? { npc: false, player: false };
    const log: string[] = [];
    const players: Fake[] = [];
    const handled: string[] = [];
    const engine = {
        newPlayers: new Set<Fake>(),
        processLogins() { log.push('logins'); for (const p of this.newPlayers) players.push(p); this.newPlayers.clear(); },
        processInfo() { log.push('info'); },
        // The engine clears the npc flag of a leaving player's tile, whoever else stands there.
        removePlayer(p: Fake) { players.splice(players.indexOf(p), 1); flags.set(key(p.tile), { ...flagsAt(p.tile), npc: false }); },
        removePlayerWithoutSave(p: Fake) { this.removePlayer(p); }
    };
    const handlers = new Map<number, { handle(message: any, player: any): boolean }>([
        [215, { handle(message, player) { handled.push(`${player.name}:${message}`); return true; } }]
    ]);
    const refused: string[] = [];
    const seenOnlyBySpectator = { count: 0, actors: [] as string[][] };
    const spectator = installSpectator<Fake>({
        world: engine, handlers, players: () => players,
        isNetwork: p => p.network, name: p => p.name, tile: p => ({ ...p.tile }),
        refuse: p => refused.push(p.name),
        restrain: p => { p.restrained++; log.push(`restrain ${p.name}`); },
        moveTo: (p, tile) => { p.tile = { ...tile }; log.push(`move ${p.name}`); },
        mirror: (spectator, actor) => log.push(`mirror ${spectator.name} ${actor.name}`),
        unmirror: spectator => log.push(`unmirror ${spectator.name}`),
        spectatorOnlyNpcs: (_, actors) => { seenOnlyBySpectator.actors.push(actors.map(a => a.name)); return seenOnlyBySpectator.count; },
        collision: tile => flagsAt(tile),
        setCollision: (tile, kind) => flags.set(key(tile), { ...flagsAt(tile), [kind]: true })
    }, 'viewer');
    const player = (name: string, network: boolean, tile = at(3094, 3107)): Fake => ({ name, network, tile, restrained: 0 });
    return { engine, handlers, players, flags, key, log, handled, refused, spectator, player, at, seenOnlyBySpectator };
}

test('a demo declares a valid spectator and pace together, or nothing', () => {
    expect(demoSettings({})).toBeNull();
    expect(demoSettings({ RL_DEMO_SPECTATOR: 'viewer', RL_DEMO_PACE_MS: '450' })).toEqual({ spectator: 'viewer', paceMs: 450 });
    for (const env of [{ RL_DEMO_SPECTATOR: 'viewer' }, { RL_DEMO_PACE_MS: '450' },
        { RL_DEMO_SPECTATOR: 'wd1a0', RL_DEMO_PACE_MS: '450' }, { RL_DEMO_SPECTATOR: 'wd1s12', RL_DEMO_PACE_MS: '450' },
        { RL_DEMO_SPECTATOR: 'Viewer', RL_DEMO_PACE_MS: '450' },
        { RL_DEMO_SPECTATOR: 'viewer', RL_DEMO_PACE_MS: '0' }, { RL_DEMO_SPECTATOR: 'viewer', RL_DEMO_PACE_MS: '501' },
        { RL_DEMO_SPECTATOR: 'viewer', RL_DEMO_PACE_MS: '4.5' }])
        expect(() => demoSettings(env)).toThrow();
});

test('only the spectator logs in over the network, restrained before the world admits it', () => {
    const w = world();
    const agent = w.player('wd1a0', false), viewer = w.player('viewer', true);
    for (const p of [agent, viewer, w.player('wd1a0', true), w.player('guest', true)]) w.engine.newPlayers.add(p);
    w.engine.processLogins();
    expect(w.refused).toEqual(['wd1a0', 'guest']); // a network login with the agent's name would take its session
    expect(w.players).toEqual([agent, viewer]);
    expect(w.log.slice(0, 2)).toEqual(['restrain viewer', 'logins']);
    expect(agent.restrained).toBe(0);
    expect(w.spectator.status()).toMatchObject({ refusedLogins: 2, present: [{ name: 'viewer' }] });
    expect(w.spectator.count()).toBe(1); // population-v2: the agent is the one actor
});

test('the spectator stands beside the one actor, in its chunk, and shows its panels at every processInfo', () => {
    const w = world();
    const viewer = w.player('viewer', true), agent = w.player('wd1a0', false, w.at(2859, 3342));
    w.players.push(viewer);
    w.engine.processInfo(); // no actor yet: it stays where it logged in
    expect(viewer.tile).toEqual(w.at(3094, 3107));
    w.players.push(agent);
    w.seenOnlyBySpectator.count = 2;
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(2859, 3341)); // the client draws one standing player a tile, its own first
    expect(w.log.slice(-4)).toEqual(['restrain viewer', 'move viewer', 'mirror viewer wd1a0', 'info']);
    w.engine.processInfo(); // already there
    agent.tile = w.at(2860, 3328); // the southern edge of its chunk (3328 >> 6 = 52)
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(2860, 3329));
    expect(beside(agent.tile).z >> 6).toBe(agent.tile.z >> 6);
    expect(w.spectator.status()).toMatchObject({ moves: 2, spectatorOnlyNpcTicks: 6 });
    w.players.push(w.player('wd1a1', false, w.at(1, 1))); // two actors: none to follow or show
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(2860, 3329));
    expect(w.log.filter(line => line.startsWith('mirror'))).toHaveLength(3);
    expect(viewer.restrained).toBe(5);
});

test('removing the spectator leaves the collision map as it was', () => {
    const w = world();
    const viewer = w.player('viewer', true, w.at(2859, 3342)), agent = w.player('wd1a0', false, w.at(2859, 3342));
    w.players.push(viewer, agent);
    w.flags.set(w.key(agent.tile), { npc: true, player: false }); // the agent's own flag
    w.engine.removePlayer(viewer);
    expect(w.flags.get(w.key(agent.tile))).toEqual({ npc: true, player: false });
    expect(w.spectator.status()).toMatchObject({ collisionRestores: 1, present: [] });
    const alone = w.player('viewer', true, w.at(10, 10)); // a free tile stays free
    w.players.push(alone);
    w.engine.removePlayerWithoutSave(alone);
    expect(w.flags.get(w.key(alone.tile))).toEqual({ npc: false, player: false });
    w.engine.removePlayer(agent); // an actor leaves as the engine makes it
    expect(w.flags.get(w.key(agent.tile))).toEqual({ npc: false, player: false });
    expect(w.spectator.status().collisionRestores).toBe(1);
});

test('every game packet of the spectator is dropped and counted; others are handled', () => {
    const w = world();
    const viewer = w.player('viewer', true), agent = w.player('wd1a0', false), other = w.player('viewer', false);
    const handler = w.handlers.get(215)!;
    expect(handler.handle('chop', viewer)).toBe(false);
    expect(handler.handle('chop', agent)).toBe(true);
    expect(handler.handle('chop', other)).toBe(true); // only a network player is the spectator
    expect(w.handled).toEqual(['wd1a0:chop', 'viewer:chop']);
    expect(w.spectator.status().droppedPackets).toEqual({ 215: 1 });
});

test('a view is a tile, an agent of a slot to follow, or both; anything else is refused', () => {
    const world = { id: 'd1', capacity: 48 };
    const tile = parseView({ tile: [3002, 3279] }, null, world);
    expect(tile).toEqual({ tile: { x: 3002, z: 3279, level: 0 }, follow: null });
    expect(parseView({ follow: 7 }, tile, world)).toEqual({ tile: tile.tile, follow: 'wd1s7' }); // keeps the tile
    expect(parseView({ tile: [3050, 3270] }, { tile: tile.tile, follow: 'wd1s7' }, world)).toEqual(
        { tile: { x: 3050, z: 3270, level: 0 }, follow: null }); // a tile alone follows no one
    expect(parseView({ tile: [1, 2], follow: 0 }, null, world)).toEqual({ tile: { x: 1, z: 2, level: 0 }, follow: 'wd1s0' });
    for (const body of [null, [], {}, { tile: [1] }, { tile: [1, 2, 0] }, { tile: [1.5, 2] }, { tile: [-1, 2] },
        { tile: [16384, 2] }, { tile: '1,2' }, { follow: 48 }, { follow: -1 }, { follow: '3' }, { tile: [1, 2], zoom: 3 }])
        expect(() => parseView(body, null, world)).toThrow();
});

test('a shared demo world\'s spectator stands at the view\'s tile among many actors, and counts npcs no actor sees', () => {
    const w = world();
    const viewer = w.player('viewer', true); // logged in on Tutorial Island
    const agents = [0, 1, 2].map(k => w.player(`wd1s${k}`, false, w.at(3000 + k, 3280)));
    w.players.push(viewer, ...agents);
    w.spectator.setView(parseView({ tile: [3002, 3279] }, null, { id: 'd1', capacity: 3 }));
    w.seenOnlyBySpectator.count = 4;
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3002, 3279)); // off Tutorial Island at its first processInfo
    w.engine.processInfo();
    expect(w.log.filter(line => line.startsWith('move') || line.includes('mirror'))).toEqual(['move viewer']);
    expect(w.seenOnlyBySpectator.actors).toEqual([['wd1s0', 'wd1s1', 'wd1s2'], ['wd1s0', 'wd1s1', 'wd1s2']]);
    expect(w.spectator.status()).toMatchObject({ moves: 1, spectatorOnlyNpcTicks: 8, views: 1,
        view: { tile: { x: 3002, z: 3279, level: 0 }, follow: null } });
});

test('a followed agent is watched beside it with its panels, through its new life, until the view goes back to a tile', () => {
    const w = world();
    const at = { id: 'd1', capacity: 3 };
    const viewer = w.player('viewer', true);
    const agents = [0, 1, 2].map(k => w.player(`wd1s${k}`, false, w.at(3000 + 4 * k, 3280)));
    w.players.push(viewer, ...agents);
    let view = parseView({ tile: [3002, 3279] }, null, at);
    w.spectator.setView(view = parseView({ follow: 1 }, view, at));
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3004, 3279));
    expect(w.log.slice(-2)).toEqual(['mirror viewer wd1s1', 'info']);
    // a new life: the agent's player leaves for a tick and comes back at another tile; the spectator waits where it is
    const player = w.players.splice(w.players.indexOf(agents[1]!), 1)[0]!;
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3004, 3279));
    player.tile = w.at(3020, 3290);
    w.players.push(player);
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3020, 3289));
    expect(w.log.filter(line => line.includes('mirror'))).toEqual(['mirror viewer wd1s1', 'mirror viewer wd1s1']);
    w.spectator.setView(view = parseView({ tile: [3002, 3279] }, view, at));
    w.engine.processInfo();
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3002, 3279));
    expect(w.log.filter(line => line.includes('mirror')).slice(-1)).toEqual(['unmirror viewer']); // once
    expect(w.spectator.followed(agents)).toBeNull();
});

test('a follow view whose agent has not joined places a new spectator at its tile, not where it logged in', () => {
    const w = world();
    const viewer = w.player('viewer', true);
    w.players.push(viewer);
    w.spectator.setView(parseView({ tile: [3092, 3326], follow: 5 }, null, { id: 'd1', capacity: 8 }));
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3092, 3326));
    const agent = w.player('wd1s5', false, w.at(3100, 3330));
    w.players.push(w.player('wd1s4', false, w.at(3090, 3320)), agent);
    expect(w.spectator.followed(w.players.filter(p => !p.network))).toBe(agent);
    w.engine.processInfo();
    expect(viewer.tile).toEqual(w.at(3100, 3329));
});

// sim/watch.py SITES: never demo at the two Rimmington sites
const SITES: Record<string, [number, number]> = { port_sarim_north: [3050, 3270], draynor_west: [3070, 3277],
    falador_farm_west: [3002, 3279], rimmington_coast: [2978, 3197], rimmington_mine_west: [2985, 3231],
    draynor_manor_south: [3092, 3326] };

test('the Rimmington sites wake npcs a Tutorial Island login wakes, and no view stands in that login\'s own chunks', () => {
    const warned = Object.entries(SITES).filter(([, [x, z]]) => wakesIslandNpcs({ x, z })).map(([name]) => name);
    expect(warned).toEqual(['rimmington_coast', 'rimmington_mine_west']);
    expect(Object.values(SITES).some(([x, z]) => inIslandChunks({ x, z }))).toBe(false);
    expect([inIslandChunks(TUTORIAL_ISLAND), inIslandChunks({ x: 3008, z: 3199 }), inIslandChunks({ x: 3007, z: 3199 }),
        inIslandChunks({ x: 3045, z: 3260 })]).toEqual([true, true, false, false]); // chunks 47-49 on both axes only
    const world = { id: 'd1', capacity: 4 };
    expect(() => parseView({ tile: [3100, 3150] }, null, world)).toThrow('Tutorial Island');
    expect(parseView({ tile: [2978, 3197] }, null, world).tile).toEqual({ x: 2978, z: 3197, level: 0 }); // recorded only
    expect(() => parseView({ tile: [3002, 3279] }, null, world, () => false)).toThrow('no map');
    // following an agent there is allowed: the agent wakes those npcs itself
    expect(parseView({ follow: 2 }, { tile: { x: 3002, z: 3279, level: 0 }, follow: null }, world).follow).toBe('wd1s2');
});

test('POST /demo-view: 409 outside a shared demo world, 400 with the reason for a refused view, else the view and counts', async () => {
    const set: any[] = [];
    const spectator = { setView: (v: any) => set.push(v),
        status: () => ({ declared: 'viewer', present: [{ name: 'viewer', x: 1, z: 1, level: 0 }], view: null, moves: 3, spectatorOnlyNpcTicks: 7 }) };
    const post = (body: string) => new Request('http://127.0.0.1/demo-view', { method: 'POST', body });
    const shared = { id: 'd1', capacity: 48 }, mapped = (t: Tile) => t.x < 3300, said: string[] = [];
    for (const handler of [viewRequests(null, shared, mapped, () => 5), viewRequests(spectator, null, mapped, () => 5)]) {
        const reply = await handler(post('{"tile":[3002,3279]}'));
        expect(reply.status).toBe(409);
    }
    const handler = viewRequests(spectator, shared, mapped, () => 9, line => said.push(line));
    for (const [body, reason] of [['{"tile":[3002', 'JSON'], ['{"zoom":2}', 'names a tile'], ['{"follow":48}', 'slot in 0..47'],
        ['{"tile":[3400,3279]}', 'no map'], ['{"tile":[3094,3107]}', 'Tutorial Island']] as const) {
        const reply = await handler(post(body));
        expect([reply.status, (await reply.text()).includes(reason)]).toEqual([400, true]);
    }
    expect(set).toEqual([]); // a refused view changes nothing
    const first = await handler(post('{"tile":[3002,3279]}'));
    expect(await first.json()).toEqual({ view: { tile: { x: 3002, z: 3279, level: 0 }, follow: null }, tick: 9,
        wakesIslandNpcs: false, audit: { moves: 3, spectatorOnlyNpcTicks: 7 } });
    const second = await handler(post('{"follow":3}'));
    expect((await second.json()).view).toEqual({ tile: { x: 3002, z: 3279, level: 0 }, follow: 'wd1s3' }); // keeps the tile
    const rimmington = await handler(post('{"tile":[2978,3197]}'));
    expect((await rimmington.json()).wakesIslandNpcs).toBe(true);
    expect(set).toHaveLength(3);
    expect(said).toEqual(['view: tile 3002,3279, follow none at tick 9; {"moves":3,"spectatorOnlyNpcTicks":7}',
        'view: tile 3002,3279, follow wd1s3 at tick 9; {"moves":3,"spectatorOnlyNpcTicks":7}',
        'view: tile 2978,3197, follow none at tick 9, its tile wakes npcs a Tutorial Island login wakes; ' +
        '{"moves":3,"spectatorOnlyNpcTicks":7}']);
});

test('a switch to another agent that is not in the world yet takes the last agent\'s panels off the spectator\'s client', () => {
    const w = world();
    const at = { id: 'd1', capacity: 4 };
    const viewer = w.player('viewer', true), a = w.player('wd1s0', false, w.at(3000, 3280));
    w.players.push(viewer, a);
    let view = parseView({ tile: [3002, 3279], follow: 0 }, null, at);
    w.spectator.setView(view);
    w.engine.processInfo();
    w.spectator.setView(view = parseView({ follow: 1 }, view, at));
    w.engine.processInfo();
    w.engine.processInfo();
    expect(w.log.filter(line => line.includes('mirror'))).toEqual(['mirror viewer wd1s0', 'unmirror viewer']);
    expect(viewer.tile).toEqual(w.at(3002, 3279)); // a new view: at its tile until its agent joins
    w.players.push(w.player('wd1s1', false, w.at(3010, 3290)));
    w.engine.processInfo();
    expect(w.log.filter(line => line.includes('mirror')).slice(-1)).toEqual(['mirror viewer wd1s1']);
});

test('a shared world\'s pace comes from its demo settings alone, and idle ticks stop once the world is admitted', async () => {
    expect(sharedPace({})).toBe(0);
    expect(sharedPace({ RL_SHARED_AGENTS: '48', RL_LOCKSTEP: '1' })).toBe(0);
    expect(sharedPace({ RL_SHARED_AGENTS: '48', RL_DEMO_SPECTATOR: 'viewer', RL_DEMO_PACE_MS: '450' })).toBe(450);
    expect(() => sharedPace({ RL_SHARED_AGENTS: '48', RL_DEMO_PACE_MS: '450' })).toThrow('together');
    let ticks = 0, marks = 0, admitted = false;
    const slept: number[] = [];
    const made = await idleTicks({ advance: () => ticks++, admitted: () => admitted, paceMs: 450, ticked: () => marks++ },
        async ms => { slept.push(ms); if (ticks === 3) admitted = true; });  // the controller connects during the third pace
    expect([made, ticks, marks, slept]).toEqual([3, 3, 3, [450, 450, 450]]);
    expect(await idleTicks({ advance: () => ticks++, admitted: () => true, paceMs: 450 }, async () => {})).toBe(0);
    expect(ticks).toBe(3);
});
