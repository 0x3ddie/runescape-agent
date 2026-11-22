import { test, expect } from 'bun:test';
import { keepNpcsNear, parseSites } from './near-site';

function fakeWorld(tiles: Array<[number, number]>) {
    const npcs = tiles.map(([x, z], nid) => ({ nid, x, z, level: 0, lifecycle: 1 }));
    const removed: Array<{ nid: number; lifecycle: number; duration: number }> = [];
    return {
        npcs, removed,
        removeNpc(npc: any, duration: number) {
            removed.push({ nid: npc.nid, lifecycle: npc.lifecycle, duration });
            npcs.splice(npcs.indexOf(npc), 1);
        },
        getTotalNpcs: () => npcs.length,
    };
}

test('only the npcs farther than the radius from every site leave, despawned for good', () => {
    // sites at (0,0) and (1000,0); radius 10 in the larger of |dx| and |dz|
    const world = fakeWorld([[10, -10], [11, 0], [0, 11], [995, 9], [500, 0], [1010, 10]]);
    const counts = keepNpcsNear(world, [[0, 0], [1000, 0]], 10);
    expect(counts).toEqual({ kept: 3, removed: 3 });
    expect(world.npcs.map(n => n.nid)).toEqual([0, 3, 5]);
    // DESPAWN (2) with no duration: World.removeNpc drops it from the npc list instead of scheduling a respawn
    expect(world.removed).toEqual([1, 2, 4].map(nid => ({ nid, lifecycle: 2, duration: -1 })));
});

test('sites parse as x,z pairs and a bad list or radius is refused', () => {
    expect(parseSites('3050,3270;2978,3197')).toEqual([[3050, 3270], [2978, 3197]]);
    expect(() => parseSites('3050')).toThrow();
    expect(() => parseSites('')).toThrow();
    expect(() => keepNpcsNear(fakeWorld([]), [[0, 0]], NaN)).toThrow();
});
