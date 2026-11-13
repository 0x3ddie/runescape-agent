// The native engine agent's click, ACT and inventory option dispatch (agent-dispatch.check.ts), run in
// its own `bun test` process: its module mocks of the game engine last for the whole process and must not reach the
// other engine tests.
import { expect, test } from 'bun:test';

test('the engine agent dispatches clicks and ACT by the client\'s rule (agent-dispatch.check.ts)', () => {
    const run = Bun.spawnSync([process.execPath, 'test', './agent-dispatch.check.ts'], { cwd: import.meta.dir,
        stdout: 'pipe', stderr: 'pipe' });
    const out = run.stdout.toString() + run.stderr.toString();
    expect(run.exitCode, out).toBe(0);
    expect(out).toMatch(/\b12 pass\b/);
    expect(out).toMatch(/\b0 fail\b/);
});
