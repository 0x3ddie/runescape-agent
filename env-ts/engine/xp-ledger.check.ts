// env-ts/engine/xp-ledger.check.ts: the XP ledger over the real Player.addXp, at the world's XP rate and under the cap.
// It loads the engine's modules (no world boots), whose World starts its login, friend and logger threads and reads
// data/ from the working directory: so it runs in its own `bun test` process with the vendor engine as its working
// directory, started by xp-ledger.test.ts, and leaves nothing in env-ts.
import { test, expect } from 'bun:test';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js'; // first: Player's import cycle needs it
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import { installXpLedger } from './xp-ledger';

test('the ledger counts what the real Player.addXp applies, at the XP rate and under the cap', () => {
    expect(typeof World.currentTick).toBe('number');
    let tick = 40;
    const ledger = installXpLedger({ playerPrototype: Player.prototype as any, tick: () => tick, stats: 21 });
    const player = (levels: number): any => Object.assign(Object.create(Player.prototype), {
        stats: new Int32Array(21), levels: new Uint8Array(21).fill(levels), baseLevels: new Uint8Array(21).fill(levels) });
    const rate = Environment.node.xpRate;
    const low = player(1), high = player(99), other = player(1);
    high.stats[8] = 1_999_999_990;
    ledger.follow(low);
    ledger.follow(high);
    low.addXp(8, 2); // woodcutting, below level 2 at any rate up to 40
    tick = 41;
    low.addXp(10, 1, false); // fishing, no multiplier
    high.addXp(8, 25); // 10 tenths left under the 200M cap
    other.addXp(8, 2); // not followed
    expect(ledger.running(low)).toEqual(Object.assign(new Array(21).fill(0), { 8: 2 * rate, 10: 1 }));
    expect(ledger.close(low)).toMatchObject({ start: 40, end: 41, grants: [[8, 2 * rate, 40], [10, 1, 41]] });
    expect(ledger.close(high)!.grants).toEqual([[8, 10, 41]]);
    expect([low.stats[8], high.stats[8]]).toEqual([2 * rate, 2_000_000_000]); // the engine's own XP agrees
    expect(ledger.running(other)).toBeNull();
});
