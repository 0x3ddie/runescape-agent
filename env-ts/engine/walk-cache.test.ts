import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// reach.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));
const { walkMap } = engine ? await import('../reach') : ({} as typeof import('../reach'));
const { keptWalkMap, sameWalk, walkCacheStats } = engine ? await import('./walk-cache')
    : ({} as typeof import('./walk-cache'));

test.skipIf(!engine)('a kept fill answers exactly as a new fill, at every tile of its window', () => {
    const tiles: Array<[number, number]> = [[3110, 3247], [3110, 3262], [2680, 3296], [3222, 3218], [3050, 3270]]; // two of one x
    for (const [x, z] of tiles) keptWalkMap(0, x, z, 48);
    const before = walkCacheStats().hits;
    for (const [x, z] of tiles) expect(sameWalk(keptWalkMap(0, x, z, 48), walkMap(0, x, z, 48), x, z, 48)).toBe(true);
    expect(walkCacheStats().hits - before).toBe(tiles.length);
    // and the check itself tells two different fills apart
    expect(sameWalk(keptWalkMap(0, 3110, 3247, 48), walkMap(0, 3111, 3247, 48), 3110, 3247, 48)).toBe(false);
});
