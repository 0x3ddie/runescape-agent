/**
 * The demo spectator's click marker (scripts/stack.py --demo-spectator; core/spectator.ts). The tile the actor last
 * asked to reach or act on is drawn in the watching client as a type-2 hint arrow over the tile and on its minimap
 * (webclient Client.ts coordArrow and the minimap's hint, 7126-7136 and 13754-13757). The arrow goes to the
 * spectator's client only: no game state, observation or action changes, and a world without a declared spectator
 * installs none of this.
 *
 * It reads the actor's request at the start of each tick, before the world moves anyone, so a one-tile walk queued
 * by the last decision is seen before its step. A new request moves the arrow; when the actor has none (it stands
 * idle) the arrow stays on its last request. A new actor (a reset replaces the agent's player) or no single actor
 * clears it. Under act-v3 the request is the agent's route-v1 record (EngineAgent.lastRoute, env-ts/core/clicks.ts,
 * agreed with the native-click port); under act-v2, which keeps no record, the host reads the engine.
 */

export interface MarkTile { x: number; z: number; level: number }
/** One request of the actor's; `key` changes when the request does. `source` says where it was read. */
export interface MarkRequest { key: unknown; tile: MarkTile; source: 'route-v1' | 'engine' }

export interface MarkerHost<P> {
    /** the actor's request now, or null when it has none */
    request(actor: P): MarkRequest | null;
    show(spectator: P, tile: MarkTile): void;
    clear(spectator: P): void;
}

/** The route-v1 fields the marker reads (env-ts/core/clicks.ts RouteRecord). */
export interface RouteMark { version: string; source: { level: number }; target: { x: number; z: number } }
export const ROUTE_VERSION = 'route-v1';

/**
 * A route-v1 record as a request: the clicked tile (a click's source + offset, or the thing an ACT chose), refused
 * or not. Each dispatch is a new record, so a second click on the same tile marks again. An unknown version stops
 * the world rather than draw a wrong arrow.
 */
export function routeRequest(route: RouteMark | null): MarkRequest | null {
    if (route === null) return null;
    if (route.version !== ROUTE_VERSION) throw new Error(`spectator marker: unknown route record ${route.version}`);
    return { key: route, tile: { x: route.target.x, z: route.target.z, level: route.source.level }, source: 'route-v1' };
}

export function clickMarker<P>(host: MarkerHost<P>) {
    const audit = { shown: 0, cleared: 0, 'route-v1': 0, engine: 0, last: null as MarkTile | null };
    const shown = new Map<P, { actor: P; key: unknown }>(); // per spectator: the actor and request its arrow marks

    return {
        /** at the start of each tick: the spectators in the world and the single actor, or null */
        update(spectators: P[], actor: P | null) {
            for (const spectator of shown.keys()) {
                if (!spectators.includes(spectator)) shown.delete(spectator); // it left the world
            }
            const request = actor === null ? null : host.request(actor);
            for (const spectator of spectators) {
                const marked = shown.get(spectator);
                if (marked && marked.actor !== actor) {
                    host.clear(spectator);
                    audit.cleared++;
                    shown.delete(spectator);
                }
                if (actor === null || request === null || shown.get(spectator)?.key === request.key) continue;
                host.show(spectator, request.tile);
                audit.shown++;
                audit[request.source]++;
                audit.last = request.tile;
                shown.set(spectator, { actor, key: request.key });
            }
        },
        status: () => ({ ...audit })
    };
}
