import { expect, test } from 'bun:test';
import { ROUTE_RECORD, type RouteRecord } from '../core/clicks';
import { clickMarker, routeRequest, ROUTE_VERSION, type MarkRequest, type RouteMark } from './click-marker';

interface Fake { name: string; request: MarkRequest | null }

function marker() {
    const sent: string[] = [];
    const m = clickMarker<Fake>({
        request: actor => actor.request,
        show: (s, t) => sent.push(`${s.name} show ${t.x},${t.z},${t.level}`),
        clear: s => sent.push(`${s.name} clear`)
    });
    return { m, sent };
}
const at = (x: number, z: number): MarkRequest => ({ key: `${x},${z}`, tile: { x, z, level: 0 }, source: 'engine' });

test('a new request moves the arrow; the same request or none leaves it', () => {
    const { m, sent } = marker();
    const viewer: Fake = { name: 'viewer', request: null }, agent: Fake = { name: 'w1a1', request: null };
    m.update([viewer], agent);
    expect(sent).toEqual([]);
    agent.request = at(3200, 3201);
    m.update([viewer], agent);
    m.update([viewer], agent);
    agent.request = null; // the walk ended: the arrow stays on it
    m.update([viewer], agent);
    agent.request = at(3205, 3201);
    m.update([viewer], agent);
    expect(sent).toEqual(['viewer show 3200,3201,0', 'viewer show 3205,3201,0']);
    expect(m.status()).toEqual({ shown: 2, cleared: 0, 'route-v1': 0, engine: 2, last: { x: 3205, z: 3201, level: 0 } });
});

test('a new actor or no single actor clears the arrow, and only spectators receive it', () => {
    const { m, sent } = marker();
    const viewer: Fake = { name: 'viewer', request: null };
    const before: Fake = { name: 'w1a1', request: at(3200, 3201) };
    m.update([viewer], before);
    const after: Fake = { name: 'w1a1', request: at(3200, 3201) }; // a reset replaced the player; same tile
    m.update([viewer], after);
    m.update([viewer], null);
    m.update([viewer], null);
    expect(sent).toEqual(['viewer show 3200,3201,0', 'viewer clear', 'viewer show 3200,3201,0', 'viewer clear']);
    expect(sent.every(line => line.startsWith('viewer '))).toBe(true);
});

test('a spectator that logs in again gets the current request', () => {
    const { m, sent } = marker();
    const agent: Fake = { name: 'w1a1', request: at(3200, 3201) };
    const first: Fake = { name: 'viewer', request: null };
    m.update([first], agent);
    m.update([], agent);
    const again: Fake = { name: 'viewer', request: null };
    m.update([again], agent);
    expect(sent).toEqual(['viewer show 3200,3201,0', 'viewer show 3200,3201,0']);
});

test('route-v1: each dispatch marks its clicked tile, even a second click on the same tile', () => {
    const { m, sent } = marker();
    const viewer: Fake = { name: 'viewer', request: null }, agent: Fake = { name: 'w1a1', request: null };
    const click = (): RouteMark => ({ version: 'route-v1', source: { level: 1 }, target: { x: 3204, z: 3205 } });
    let route: RouteMark | null = null;
    const step = () => { agent.request = routeRequest(route); m.update([viewer], agent); };
    step(); // reset: no record yet
    route = click();
    step();
    step(); // the same record: no new click
    route = click(); // a new dispatch on the same tile
    step();
    expect(sent).toEqual(['viewer show 3204,3205,1', 'viewer show 3204,3205,1']);
    expect(() => routeRequest({ ...click(), version: 'route-v2' })).toThrow('unknown route record route-v2');
});

test("the marker reads the record the port publishes: route-v1's version and a refused click's tile", () => {
    expect(ROUTE_VERSION).toBe(ROUTE_RECORD);
    const refused: RouteRecord = { version: ROUTE_RECORD, kind: 'click', action: 90, tick: 7,
        source: { x: 3200, z: 3200, level: 0 }, offset: { dx: 16, dz: 0 }, target: { x: 3216, z: 3200 }, thing: null,
        via: 'none', end: null, refused: 'no route', waypoints: [] };
    expect(routeRequest(refused)).toEqual({ key: refused, tile: { x: 3216, z: 3200, level: 0 }, source: 'route-v1' });
});
