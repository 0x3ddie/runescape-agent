// hello's engine settings (engine-settings.check.ts), each case in its own process from the vendor engine folder.
import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// engine-settings.check.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));

const settings = (set: Record<string, string>, unset: string[], want: object) => {
    const env = Object.fromEntries(Object.entries({ ...process.env, ...set }).filter(([k]) => !unset.includes(k)));
    const out = Bun.spawnSync([process.execPath, 'test', `${import.meta.dir}/engine-settings.check.ts`], {
        cwd: new URL('../../vendor/rs-sdk/server/engine/', import.meta.url).pathname, stdout: 'pipe', stderr: 'pipe',
        env: { ...env, EXPECT_SETTINGS: JSON.stringify(want) } });
    const text = `${out.stdout.toString()}${out.stderr.toString()}`;
    if (out.exitCode !== 0) console.log(text);
    expect(out.exitCode).toBe(0);
    expect(text).toContain('1 pass');
};

test.skipIf(!engine)('hello reports the engine\'s own XP rate and random events, as the engine parses its settings', () => {
    settings({ NODE_RANDOM_EVENTS: 'true', NODE_XPRATE: '1' }, [], { xpRate: 1, randomEvents: true });
    settings({ NODE_XPRATE: '25' }, ['NODE_RANDOM_EVENTS'], { xpRate: 25, randomEvents: false }); // unset: off
}, 60_000);
