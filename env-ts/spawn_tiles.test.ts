import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// spawn_tiles.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../vendor/rs-sdk', import.meta.url));
const { connectedJitterTiles } = engine ? await import('./spawn_tiles') : ({} as typeof import('./spawn_tiles'));

test.skipIf(!engine)('no spawn jitter tile is behind a bank counter', () => {
    // Draynor's bank spawn (3092,3243): the bankers' side of the booths (x 3089-3091)
    // is within 2 tiles across the counter, and seeds 1-3 and 5-7 started there.
    const tiles = connectedJitterTiles(3092, 3243);
    expect(tiles.length).toBeGreaterThan(0);
    expect(tiles.every(t => t.x >= 3092)).toBe(true);
});
