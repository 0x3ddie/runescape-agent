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
 *
 * A shared demo world (RL_SHARED_AGENTS: many agents of one controller, engine/batch.ts) has a view instead, which the
 * demo's command sets and a request may change during the run (demo-services POST /demo-view): a tile to stand on,
 * an agent to follow (beside it, its panels shown, the click marker on its requests), or both. While the followed
 * agent is not in the world (a new life replaces its player for a tick or two) the spectator stays where it is, or
 * goes to the view's tile when this view has not placed it yet (a login on Tutorial Island). Without a view the
 * one-actor rule above holds. The npcs only the spectator sees are those no actor's npc info holds.
 */

const SPECTATOR_NAME = /^[a-z0-9]{1,12}$/;
const AGENT_NAME = /^w[a-z0-9]{1,2}[as]\d+$/; // engine-env's clientless agents: w<world>a<seq>, w<world>s<slot> shared
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

/** A shared world's clock pace (core/lockstep-clock.ts): a demo world's, from its demo settings alone; else none. */
export function sharedPace(env: Record<string, string | undefined>): number {
    return demoSettings(env)?.paceMs ?? 0;
}

/**
 * A demo world's ticks before its agents join (engine-env): one tick, then the pace, until `admitted` (an agent was
 * admitted, or a shared world's controller connected), so that the spectator can log in. `ticked` follows each tick
 * (a shared world's clock counts its pace from it). Returns the ticks made.
 */
export async function idleTicks(world: { advance(): void; admitted(): boolean; paceMs: number; ticked?: () => void },
                                sleep: (ms: number) => Promise<unknown> = ms => new Promise(r => setTimeout(r, ms))): Promise<number> {
    let ticks = 0;
    while (!world.admitted()) {
        world.advance();
        world.ticked?.();
        ticks++;
        await sleep(world.paceMs);
    }
    return ticks;
}

export interface Tile { x: number; z: number; level: number }

/** A shared demo world's view: the tile to stand on (level 0) and the agent to follow (its engine name), or null. */
export interface View { tile: Tile | null; follow: string | null }

const COORD_LIMIT = 1 << 14; // the engine's coordinates are 14-bit (CoordGrid)
// Where a new login stands (a demo caveat: a spectator waiting there wakes npc chunks 47-49 on both axes, which
// the Rimmington sites' npcs share). A player wakes the npcs of its 64-tile chunk and the chunks beside it (RL_NPC_RADIUS
// 64, engine-env's npc rule), so two players wake a chunk in common when their chunks are within 2 on both axes.
export const TUTORIAL_ISLAND = { x: 3094, z: 3107 };
const islandGap = (tile: { x: number; z: number }) =>
    Math.max(Math.abs((tile.x >> 6) - (TUTORIAL_ISLAND.x >> 6)), Math.abs((tile.z >> 6) - (TUTORIAL_ISLAND.z >> 6)));

/** `tile` is in the npc chunks a Tutorial Island login wakes (47-49 on both axes): no view stands there. */
export function inIslandChunks(tile: { x: number; z: number }): boolean {
    return islandGap(tile) <= 1;
}

/** A player at `tile` wakes a chunk a Tutorial Island login wakes too (as the Rimmington sites do): recorded. */
export function wakesIslandNpcs(tile: { x: number; z: number }): boolean {
    return islandGap(tile) <= 2;
}

/**
 * The view a POST /demo-view body asks for, over the view it replaces: {tile: [x, z]} stands there and follows no one;
 * {follow: slot} follows the agent of that slot (engine-env's name w<world>s<slot>) and keeps the tile to stand on
 * while it is not in the world; both set both. Anything else is refused, and so is a tile the engine has no map for
 * (`mapped`: its teleport would be ignored every tick) or one in the chunks a Tutorial Island login wakes
 * (inIslandChunks); following an agent there is allowed, since that agent wakes them itself.
 */
export function parseView(body: any, before: View | null, world: { id: string; capacity: number },
                          mapped: (tile: Tile) => boolean = () => true): View {
    if (typeof body !== 'object' || body === null || Array.isArray(body)) throw new Error('a view is a JSON object');
    const extra = Object.keys(body).filter(k => k !== 'tile' && k !== 'follow');
    if (extra.length || !('tile' in body || 'follow' in body)) throw new Error('a view names a tile, an agent to follow or both');
    let tile = before?.tile ?? null, follow: string | null = null;
    if ('tile' in body) {
        const t = body.tile;
        if (!Array.isArray(t) || t.length !== 2 || !t.every((v: unknown) => Number.isInteger(v) && (v as number) >= 0 && (v as number) < COORD_LIMIT))
            throw new Error(`a view's tile is [x, z], whole numbers in 0..${COORD_LIMIT - 1}`);
        tile = { x: t[0], z: t[1], level: 0 };
        if (!mapped(tile)) throw new Error(`the engine has no map for ${t[0]},${t[1]}: a spectator there would not land`);
        if (inIslandChunks(tile))
            throw new Error(`${t[0]},${t[1]} is in the npc chunks a Tutorial Island login wakes: no view stands there`);
    }
    if ('follow' in body) {
        const slot = body.follow;
        if (!Number.isInteger(slot) || slot < 0 || slot >= world.capacity)
            throw new Error(`a view follows a slot in 0..${world.capacity - 1}`);
        follow = `w${world.id}s${slot}`;
    }
    return { tile, follow };
}

/**
 * POST /demo-view of a demo world (demo-services): 409 unless the world is a shared world with a declared spectator;
 * 400 with the reason for a body parseView refuses (the view stays as it was); else the new view, set on the spectator,
 * with the tick, whether its tile wakes a chunk a Tutorial Island login wakes (wakesIslandNpcs: the demo caveat above,
 * recorded, not refused) and the spectator's counts at the change (`audit`); the same in the engine's log (`say`).
 */
export function viewRequests(spectator: { setView(view: View): void; status(): Record<string, any> } | null,
                             shared: { id: string; capacity: number } | null, mapped: (tile: Tile) => boolean,
                             tick: () => number, say: (line: string) => void = () => {}) {
    let view: View | null = null;
    return async (req: Request): Promise<Response> => {
        if (!spectator || !shared)
            return new Response('a view needs a shared demo world (RL_SHARED_AGENTS and --demo-spectator)', { status: 409 });
        let next: View;
        try {
            next = parseView(await req.json(), view, shared, mapped);
        } catch (error) {
            return new Response(String((error as Error)?.message ?? error), { status: 400 });
        }
        spectator.setView(view = next);
        const { present: _present, view: _view, declared: _declared, ...audit } = spectator.status();
        const island = !!view.tile && wakesIslandNpcs(view.tile);
        say(`view: tile ${view.tile ? `${view.tile.x},${view.tile.z}` : 'none'}, follow ${view.follow ?? 'none'} at tick ${tick()}` +
            `${island ? ', its tile wakes npcs a Tutorial Island login wakes' : ''}; ${JSON.stringify(audit)}`);
        return Response.json({ view, tick: tick(), wakesIslandNpcs: island, audit });
    };
}

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
    /** the spectator's own inventory and stats in its client again, after a mirror */
    unmirror(spectator: P): void;
    /** npcs in the spectator's npc info that are in no actor's */
    spectatorOnlyNpcs(spectator: P, actors: P[]): number;
    /** the npc and player collision flags of a tile */
    collision(tile: Tile): { npc: boolean; player: boolean };
    setCollision(tile: Tile, kind: 'npc' | 'player'): void;
}

export function installSpectator<P>(host: SpectatorHost<P>, spectator: string) {
    const audit = { refusedLogins: 0, droppedPackets: {} as Record<string, number>, moves: 0, collisionRestores: 0,
        spectatorOnlyNpcTicks: 0, views: 0 };
    const isSpectator = (player: P) => host.isNetwork(player) && host.name(player) === spectator;
    const present = () => [...host.players()].filter(isSpectator);
    let view: View | null = null; // a shared demo world's (setView); null: the one-actor rule
    const mirrored = new Map<P, string>(); // spectators whose client shows an actor's panels, with that actor's name
    const placed = new Set<P>(); // spectators this view has placed
    /** The actor the spectator follows now: the view's agent when it is in the world, else the one actor (no view). */
    const followed = (actors: P[]): P | null => view === null ? (actors.length === 1 ? actors[0]! : null)
        : view.follow === null ? null : actors.find(actor => host.name(actor) === view!.follow) ?? null;
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
        const actor = followed(actors);
        for (const player of watching) {
            // npc info from the last tick, which this tick's hunts used
            if (actors.length) audit.spectatorOnlyNpcTicks += host.spectatorOnlyNpcs(player, actors);
            const at = actor ? beside(host.tile(actor))
                : view?.tile && (view.follow === null || !placed.has(player)) ? view.tile : null;
            const from = host.tile(player);
            if (at) {
                placed.add(player);
                if (from.x !== at.x || from.z !== at.z || from.level !== at.level) {
                    host.moveTo(player, at);
                    audit.moves++;
                }
            }
            if (actor) {
                host.mirror(player, actor);
                mirrored.set(player, host.name(actor));
            } else if (view !== null && mirrored.has(player) && mirrored.get(player) !== view.follow) {
                // a tile view, or another agent to follow that is not in the world yet: not the last agent's panels
                mirrored.delete(player);
                host.unmirror(player);
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
        /** the actor followed now among `actors` (the click marker's; followed above) */
        followed,
        /** a shared demo world's view from now on (parseView) */
        setView(next: View) {
            view = next;
            placed.clear();
            audit.views++;
        },
        status: () => ({ declared: spectator, present: present().map(player => ({ name: host.name(player), ...host.tile(player) })),
            refusedLogins: audit.refusedLogins, droppedPackets: { ...audit.droppedPackets }, moves: audit.moves,
            collisionRestores: audit.collisionRestores, spectatorOnlyNpcTicks: audit.spectatorOnlyNpcTicks,
            view: view && { tile: view.tile && { ...view.tile }, follow: view.follow }, views: audit.views })
    };
}
