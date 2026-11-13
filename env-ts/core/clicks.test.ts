import { expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { A_CLICK_HIGH, A_CLICK_LOW, N_ACTIONS } from './actions';
import { CLICK_OFFSETS, CLICK_ROWS, ENTITY, TILE, clickSlot, clickSlots, clientFlags, locArrival, sceneBase, tryMove,
    type Arrival } from './clicks';

// The shared cases (sim/tests/test_clicks.py writes them from sim/paths.py, the Python port): scenes of the client's
// collision, tile clicks and op arrivals, and each route the Python reference found.
const cases = JSON.parse(readFileSync(new URL('./clicks.fixture.json', import.meta.url), 'utf8'));
const base = { x: cases.base[0], z: cases.base[1] };
const grids: Record<string, Int32Array> = Object.fromEntries(Object.entries(cases.scenes).map(([name, cells]) => {
    const grid = new Int32Array(104 * 104);
    for (const [x, z, f] of cells as number[][]) grid[x * 104 + z] = f;
    return [name, grid];
}));
const arrivalOf = ([width, length, angle, shape, forceapproach]: number[]): Arrival => ({ width, length, angle, shape, forceapproach });

test('the click table is rl/act_contract.py\'s: 24 offsets on rows 17..28 and 84..95', () => {
    expect(CLICK_OFFSETS).toEqual(cases.clicks.offsets);
    expect(CLICK_ROWS).toEqual(cases.clicks.rows);
    expect(CLICK_ROWS.map(clickSlot)).toEqual([...Array(24).keys()]);
    for (let a = 0; a < N_ACTIONS; a++) if (!CLICK_ROWS.includes(a)) expect(clickSlot(a)).toBeNull();
    expect([CLICK_ROWS[0], CLICK_ROWS[11], CLICK_ROWS[12], CLICK_ROWS[23]]).toEqual([A_CLICK_LOW, 28, A_CLICK_HIGH, 95]);
});

test('the port routes every shared case as the Python reference does, the 25-turn cut and fallback included', () => {
    for (const q of cases.queries) {
        const grid = grids[q.scene];
        const route = tryMove((x, z) => grid[(x - base.x) * 104 + (z - base.z)], base, q.source[0], q.source[1],
            q.dest[0], q.dest[1], q.nearest, arrivalOf(q.arrival));
        expect(route && { waypoints: route.waypoints.map(p => [p.x, p.z]), via: route.via }).toEqual(q.route);
    }
});

// The vendored client's own Client.tryMove, extracted from its source and run over a CollisionMap of each scene: the
// fidelity check of both ports, a closed door between player and target among the cases.
const CLIENT = new URL('../../vendor/rs-sdk/server/webclient/src/client/Client.ts', import.meta.url);
test.skipIf(!existsSync(CLIENT))('the vendored client\'s tryMove sends the turn points of every shared case', async () => {
    const { default: CollisionMap } = await import('../../vendor/rs-sdk/server/webclient/src/dash3d/CollisionMap.ts');
    const enums = await Promise.all(['dash3d/CollisionFlag', 'dash3d/LocShape', 'dash3d/DirectionFlag']
        .map(m => import(`../../vendor/rs-sdk/server/webclient/src/${m}.ts`)));
    const source = readFileSync(CLIENT, 'utf8');
    const start = source.indexOf('private tryMove(');
    const open = source.indexOf('{', source.indexOf('): boolean', start));
    let depth = 0, end = open;
    do { if (source[end] === '{') depth++; else if (source[end] === '}') depth--; end++; } while (depth > 0);
    const method = new Bun.Transpiler({ loader: 'ts' }).transformSync(
        `export function tryMove${source.slice(start + 'private tryMove'.length, end)}`).replace('export function', 'return function');
    const clientTryMove = new Function('CollisionMap', 'BuildArea', 'CollisionFlag', 'LocShape', 'DirectionFlag', 'ClientProt', method)(
        CollisionMap, { SIZE: 104, LEVELS: 4 }, enums[0].CollisionFlag, enums[1].LocShape, enums[2].DirectionFlag,
        new Proxy({}, { get: () => 0 }));
    for (const q of cases.queries) {
        const map = new CollisionMap();
        const grid = grids[q.scene];
        for (let x = 1; x < 103; x++) for (let z = 1; z < 103; z++) map.flags[x * 104 + z] = grid[x * 104 + z];
        const sent: number[] = [];
        const client: any = { collision: [map], minusedlevel: 0, dirMap: new Int32Array(104 * 104),
            distMap: new Int32Array(104 * 104), routeX: new Int32Array(4000), routeZ: new Int32Array(4000), tryMoveNearest: 0,
            mapBuildBaseX: base.x, mapBuildBaseZ: base.z, minimapFlagX: 0, minimapFlagZ: 0,
            out: { p1Enc: () => {}, p1: (v: number) => sent.push(v), p2: (v: number) => sent.push(v) } };
        const [width, length, angle, shape, forceapproach] = q.arrival;
        const ok = clientTryMove.call(client, q.source[0] - base.x, q.source[1] - base.z, q.dest[0] - base.x,
            q.dest[1] - base.z, q.nearest, width, length, angle, shape, forceapproach, 0);
        if (!q.route) { expect(ok).toBe(false); continue; }
        // p1 length, p1 run, p2 start x, p2 start z, then each further turn point as p1 offsets from the start
        const [startX, startZ] = [sent[2], sent[3]], points = [[startX, startZ]];
        for (let i = 4; i < sent.length; i += 2) points.push([startX + sent[i], startZ + sent[i + 1]]);
        expect(ok).toBe(true);
        expect(points).toEqual(q.route.waypoints);
        expect(client.tryMoveNearest === 1).toBe(q.route.via === 'fallback');
    }
});

test('the client\'s collision keeps walls, scenery and the floor (ground decoration in it), and no npc or player', () => {
    expect(clientFlags(0x8 | 0x100)).toBe(0x108);
    expect(clientFlags(0x40000)).toBe(0x200000); // FLOOR_DECORATION is the client's WR_GRND
    expect(clientFlags(0x80000 | 0x100000)).toBe(0); // NPC and PLAYER occupancy
    expect(clientFlags(0x7fffffff) & 0x200100).not.toBe(0); // an unloaded tile stays blocked
    expect(sceneBase(3222, 3218)).toEqual({ x: 3168, z: 3168 }); // zone 402 on both axes -> (402 - 6) x 8
});

test('a loc op arrives by its rotated size and forceapproach, or by its wall shape; an npc or item by 1 x 1', () => {
    // a 2 x 3 centrepiece turned a quarter (angle 1): 3 x 2, and forceapproach WEST (8) becomes NORTH (1)
    expect(locArrival(10, 1, 2, 3, 8)).toEqual({ width: 3, length: 2, angle: 0, shape: 0, forceapproach: 1 });
    expect(locArrival(22, 2, 2, 3, 1)).toEqual({ width: 2, length: 3, angle: 0, shape: 0, forceapproach: 4 });
    expect(locArrival(0, 3, 1, 1, 0)).toEqual({ width: 0, length: 0, angle: 3, shape: 1, forceapproach: 0 });
    expect([ENTITY.width, ENTITY.length, TILE.width, TILE.shape]).toEqual([1, 1, 0, 0]);
});

test('a click slot\'s open bit is the target tile\'s WALK_BLOCKED, walls and npcs aside; natively always inside', () => {
    const blocked = new Set(['100,104', '104,100']);
    const flags = (x: number, z: number) => blocked.has(`${x},${z}`) ? 0x100 : x === 96 ? 0x8 | 0x80000 : 0;
    const slots = clickSlots(100, 100, flags);
    expect(slots.length).toBe(24);
    expect(slots[0]).toEqual({ dx: 0, dz: 4, inside: true, open: false }); // the loc 4 north
    expect(slots[2]).toEqual({ dx: 4, dz: 0, inside: true, open: false }); // the loc 4 east
    expect(slots[6]).toEqual({ dx: -4, dz: 0, inside: true, open: true }); // a wall and an npc do not close it
});
