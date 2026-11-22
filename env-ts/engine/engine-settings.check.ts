// env-ts/engine/engine-settings.check.ts: hello's engine settings read from the engine's own Environment, which parses
// the process's settings itself. Run by engine-settings.test.ts in its own process from the vendor engine (loading
// the engine reads data/ from the working directory), once with random events on at XP rate 1, once with defaults.
import { expect, test } from 'bun:test';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import { engineSettings } from './batch';

test('hello reports the engine\'s own XP rate and random-event setting', () => {
    const want = JSON.parse(process.env.EXPECT_SETTINGS!);
    expect(engineSettings(Environment as any)).toEqual(want);
    expect(engineSettings(Environment as any).randomEvents).toBe(Environment.NODE_RANDOM_EVENTS);
});
