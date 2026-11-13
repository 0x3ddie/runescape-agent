import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// world.ts (through spawn_tiles.ts) needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));
const { farSpawn } = engine ? await import('./world') : ({} as typeof import('./world'));

// Walking distance: straight-line tiles from (100, 100), except that x > 130 is sealed off.
const steps = (x: number, z: number) => x > 130 ? -1 : Math.max(Math.abs(x - 100), Math.abs(z - 100));
const base = { x: 100, z: 100 };

test.skipIf(!engine)('a far start is in the walking band, repeatable by seed, and can walk back', () => {
    const tile = farSpawn(base, [25, 45], 3, steps, () => true);
    expect(steps(tile.x, tile.z)).toBeGreaterThanOrEqual(25);
    expect(steps(tile.x, tile.z)).toBeLessThanOrEqual(45);
    expect(farSpawn(base, [25, 45], 3, steps, () => true)).toEqual(tile);
    // Tiles the live map cannot leave (west of x=80 here) are skipped, not spawned on.
    for (const seed of [1, 2, 3, 4, 5]) expect(farSpawn(base, [25, 45], seed, steps, t => t.x >= 80).x).toBeGreaterThanOrEqual(80);
    expect(() => farSpawn(base, [25, 45], 1, steps, () => false)).toThrow();
    expect(() => farSpawn(base, [45, 25], 1, steps, () => true)).toThrow();
});
