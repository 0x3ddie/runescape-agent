import { expect, test } from 'bun:test';
import { beside, demoSettings, installSpectator, type Tile } from './spectator';

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
    const seenOnlyBySpectator = { count: 0 };
    const spectator = installSpectator<Fake>({
        world: engine, handlers, players: () => players,
        isNetwork: p => p.network, name: p => p.name, tile: p => ({ ...p.tile }),
        refuse: p => refused.push(p.name),
        restrain: p => { p.restrained++; log.push(`restrain ${p.name}`); },
        moveTo: (p, tile) => { p.tile = { ...tile }; log.push(`move ${p.name}`); },
        mirror: (spectator, actor) => log.push(`mirror ${spectator.name} ${actor.name}`),
        spectatorOnlyNpcs: () => seenOnlyBySpectator.count,
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
        { RL_DEMO_SPECTATOR: 'wd1a0', RL_DEMO_PACE_MS: '450' }, { RL_DEMO_SPECTATOR: 'Viewer', RL_DEMO_PACE_MS: '450' },
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
