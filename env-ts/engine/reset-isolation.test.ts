import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { replacePlayerForReset } from './reset-isolation';

test.each(['alive-truncated', 'dead', 'pending-admission'])('explicit reset isolates %s object and queued work', state => {
    const old = { state, inventory: ['old axe'], x: 3100, queued: [] as Array<() => void> };
    old.queued.push(() => { old.inventory = []; old.x = 0; });
    const next = { state: 'fresh', inventory: ['new axe'], x: 3092, queued: [] as Array<() => void> };
    const pending = new Set([old]);
    const admitted = new Set(state === 'pending-admission' ? [] : [old]);
    const events: string[] = [];
    const result = replacePlayerForReset(old, pending, player => {
        expect(pending.has(player)).toBe(false);
        admitted.delete(player); events.push('unregister');
    }, () => {
        expect(admitted.has(old)).toBe(false);
        events.push('admit'); pending.add(next); return next;
    });
    expect(events).toEqual(['unregister', 'admit']);
    expect(result).toBe(next);
    // Even an external reference to old queued work cannot alias the new state.
    old.queued[0]();
    expect(result.inventory).toEqual(['new axe']); expect(result.x).toBe(3092);
    expect(pending.size).toBe(1); expect(pending.has(next)).toBe(true);
});

test('failed retirement prevents admission and same-object admission fails closed', () => {
    const old = {}, pending = new Set([old]); let admissions = 0;
    expect(() => replacePlayerForReset(old, pending, () => { throw new Error('retirement failed'); },
        () => { admissions++; return {}; })).toThrow('retirement failed');
    expect(admissions).toBe(0);
    expect(() => replacePlayerForReset(old, pending, () => {}, () => old)).toThrow('reused');
});

test('engine binds replacement only to explicit reset and retains requested stat copy', () => {
    const source = readFileSync(new URL('./agent.ts', import.meta.url), 'utf8');
    const reset = source.split('async reset(')[1].split('// ---- option tracking')[0];
    expect(reset).toContain('replacePlayerForReset(previous, World.newPlayers,');
    expect(reset).toContain('player => World.removePlayerWithoutSave(player), () => this.admit()');
    expect(reset).not.toContain('replaceForContinuity');
    expect(reset).not.toContain('if (this.death.dead || !this.isAdmitted()) {\n            const previous');
    expect(reset).toContain('cfg.persistLevels');
    expect(reset).toContain('stats: previous.stats.slice(), baseLevels: previous.baseLevels.slice()');
    expect(reset).toContain('this.player.stats.set(retained.stats)');
    expect(source.split('async step(')[1]).not.toContain('replacePlayerForReset');
});
