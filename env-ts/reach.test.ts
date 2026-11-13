import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// reach.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../vendor/rs-sdk', import.meta.url));
const { walkMap } = engine ? await import('./reach') : ({} as typeof import('./reach'));

test.skipIf(!engine)('a tree across a fence is beside the player but must be walked round to', () => {
    // Draynor: a fence runs between (3110,3247) and the tree at (3111,3247); the engine
    // chop stood on the wrong side and was re-issued 532 times.
    const walk = walkMap(0, 3110, 3247, 48);
    expect(walk.approach(3111, 3247)).toBe(0);
    expect(walk.operable(3111, 3247)).toBeGreaterThan(0);
});

test.skipIf(!engine)('a route may leave the listing radius', () => {
    // Ardougne south's bank is 24 tiles from its trees but 50+ steps round a building.
    expect(walkMap(0, 2680, 3296, 33).approach(2656, 3283)).toBe(-1);
    expect(walkMap(0, 2680, 3296, 48).approach(2656, 3283)).toBeGreaterThan(24);
});
