// The XP ledger over the real Player.addXp (xp-ledger.check.ts), run in its own `bun test` process with the vendor
// engine as its working directory: loading the engine starts its threads and reads data/ from there.
import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// xp-ledger.check.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));

test.skipIf(!engine)('the ledger counts what the real Player.addXp applies (xp-ledger.check.ts)', () => {
    const run = Bun.spawnSync([process.execPath, 'test', `${import.meta.dir}/xp-ledger.check.ts`], {
        cwd: new URL('../../vendor/rs-sdk/server/engine/', import.meta.url).pathname, stdout: 'pipe', stderr: 'pipe' });
    const out = `${run.stdout.toString()}${run.stderr.toString()}`;
    if (run.exitCode !== 0) console.log(out);
    expect(run.exitCode).toBe(0);
    expect(out).toContain('1 pass');
}, 60_000);
