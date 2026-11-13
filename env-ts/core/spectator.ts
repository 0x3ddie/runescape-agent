/**
 * Demo spectator: one declared network login that watches the agent in a lockstep demo world
 * (scripts/stack.py --demo-spectator). Never a policy input, never an actor.
 *
 * The spectator is hidden from other players and npcs and blocks nothing from before it enters a
 * zone, its queues and timers are cleared every tick, every game packet it sends is dropped and
 * counted, and once the world has exactly one actor it stands beside the actor at every
 * processInfo, and its client shows the actor's inventory and stats. Beside, because the client
 * draws one standing player per tile and its own player first (webclient Client.ts pushPlayers): on
 * the actor's tile the spectator, which has no body, would hide the actor. The tile is in the actor's
 * 64-tile chunk, so the npcs awake (RL_NPC_RADIUS 64, by chunk) are the actor's own; the npcs only the
 * spectator sees, whose hunts its presence can run (rsbuf observers), are counted. Its removal leaves
 * the collision map as it was (the engine clears the npc flag of a leaving player's tile). No other
 * network login enters a demo world. A world without a declared spectator installs none of this.
 */

const SPECTATOR_NAME = /^[a-z0-9]{1,12}$/;
const AGENT_NAME = /^w[a-z0-9]{1,2}a\d+$/; // engine-env's clientless agents: w<world>a<seq>
// A step waits at most 105 ticks (K_TICKS plus jitter, then DECISION_WAIT_TICKS plus a last chunk),
// which must stay under rl/vec_adapter.py's 60 s step timeout.
export const MAX_DEMO_PACE_MS = 500;

export interface DemoSettings { spectator: string; paceMs: number }

/** RL_DEMO_SPECTATOR and RL_DEMO_PACE_MS, which come together; null when neither is set. */
export function demoSettings(env: Record<string, string | undefined>): DemoSettings | null {
    const spectator = env.RL_DEMO_SPECTATOR, pace = env.RL_DEMO_PACE_MS;
    if (spectator === undefined && pace === undefined) return null;
    if (spectator === undefined || pace === undefined) throw new Error('a demo declares its spectator and pace together');
    if (!SPECTATOR_NAME.test(spectator) || AGENT_NAME.test(spectator)) throw new Error(`invalid demo spectator name ${spectator}`);
    const paceMs = Number(pace);
    if (!Number.isInteger(paceMs) || paceMs < 1 || paceMs > MAX_DEMO_PACE_MS)
        throw new Error(`demo pace must be 1..${MAX_DEMO_PACE_MS} ms a tick`);
    return { spectator, paceMs };
}

export interface Tile { x: number; z: number; level: number }

/** One tile south of the actor, or north at its chunk's southern edge: the same level and 64-tile chunk. */
export function beside(at: Tile): Tile {
    return { ...at, z: (at.z - 1) >> 6 === at.z >> 6 ? at.z - 1 : at.z + 1 };
}

/** The engine objects the spectator hooks wrap, and what it may do to a player. */
export interface SpectatorHost<P> {
    /** the world, whose processLogins, processInfo, removePlayer and removePlayerWithoutSave are wrapped */
    world: { newPlayers: Set<P> } & Record<string, any>;
    /** the game packet handlers by opcode */
    handlers: Map<number, { handle(message: any, player: any): boolean }>;
    players(): Iterable<P>;
    isNetwork(player: P): boolean;
    name(player: P): string;
    tile(player: P): Tile;
    /** refuse a queued login before the world admits it */
    refuse(player: P): void;
    /** hidden, non-blocking and inert; run on every spectator before its login and at every processInfo */
    restrain(player: P): void;
    /** a same-level move within 8 tiles renders as a walk in the 274 client */
    moveTo(player: P, tile: Tile): void;
    /** the actor's inventory and stats in the spectator's client; only the spectator's client bookkeeping changes */
    mirror(spectator: P, actor: P): void;
    /** npcs in the spectator's npc info that are not in the actor's */
    spectatorOnlyNpcs(spectator: P, actor: P): number;
    /** the npc and player collision flags of a tile */
    collision(tile: Tile): { npc: boolean; player: boolean };
    setCollision(tile: Tile, kind: 'npc' | 'player'): void;
}

export function installSpectator<P>(host: SpectatorHost<P>, spectator: string) {
    const audit = { refusedLogins: 0, droppedPackets: {} as Record<string, number>, moves: 0, collisionRestores: 0,
        spectatorOnlyNpcTicks: 0 };
    const isSpectator = (player: P) => host.isNetwork(player) && host.name(player) === spectator;
    const present = () => [...host.players()].filter(isSpectator);
    const wrap = (method: string, around: (original: (...args: any[]) => any) => (...args: any[]) => any) => {
        const original = host.world[method];
        if (typeof original !== 'function') throw new Error(`spectator hook: world has no ${method}`);
        host.world[method] = around(original);
    };

    wrap('processLogins', original => function (this: unknown, ...args) {
        for (const player of [...host.world.newPlayers]) {
            if (!host.isNetwork(player)) continue; // the clientless agent
            if (isSpectator(player)) host.restrain(player);
            else {
                host.world.newPlayers.delete(player);
                host.refuse(player);
                audit.refusedLogins++;
            }
        }
        return original.apply(this, args);
    });
    wrap('processInfo', original => function (this: unknown, ...args) {
        const watching = present();
        for (const player of watching) host.restrain(player);
        const actors = [...host.players()].filter(player => !isSpectator(player));
        if (actors.length === 1) {
            const actor = actors[0]!, at = beside(host.tile(actor));
            for (const player of watching) {
                // npc info from the last tick, which this tick's hunts used
                audit.spectatorOnlyNpcTicks += host.spectatorOnlyNpcs(player, actor);
                const from = host.tile(player);
                if (from.x !== at.x || from.z !== at.z || from.level !== at.level) {
                    host.moveTo(player, at);
                    audit.moves++;
                }
                host.mirror(player, actor);
            }
        }
        return original.apply(this, args);
    });
    for (const method of ['removePlayer', 'removePlayerWithoutSave']) {
        wrap(method, original => function (this: unknown, player: P, ...rest) {
            if (!isSpectator(player)) return original.call(this, player, ...rest);
            const at = host.tile(player), before = host.collision(at);
            try {
                return original.call(this, player, ...rest);
            } finally {
                const after = host.collision(at);
                for (const kind of ['npc', 'player'] as const) {
                    if (before[kind] && !after[kind]) {
                        host.setCollision(at, kind);
                        audit.collisionRestores++;
                    }
                }
            }
        });
    }
    for (const [opcode, handler] of host.handlers) {
        host.handlers.set(opcode, {
            handle(message, player) {
                if (!isSpectator(player)) return handler.handle(message, player);
                audit.droppedPackets[opcode] = (audit.droppedPackets[opcode] ?? 0) + 1;
                return false;
            }
        });
    }

    return {
        /** spectators in the world now */
        count: () => present().length,
        status: () => ({ declared: spectator, present: present().map(player => ({ name: host.name(player), ...host.tile(player) })),
            refusedLogins: audit.refusedLogins, droppedPackets: { ...audit.droppedPackets }, moves: audit.moves,
            collisionRestores: audit.collisionRestores, spectatorOnlyNpcTicks: audit.spectatorOnlyNpcTicks })
    };
}
