import { expect, test } from 'bun:test';
import { HurtTicks } from './hurt';

/** The full-deficit ticks of a count: the sum over its bases of n / base. */
const full = (facts: number[][]) => facts.reduce((sum, [base, n]) => sum + n / base, 0);

test('each tick of an armed life adds its missing hitpoints to its base, and a full tick adds 0', () => {
    const hurt = new HurtTicks();
    hurt.arm();
    for (const hp of [7, 7, 8, 10, 10]) hurt.tick(hp, 10, true);
    expect(hurt.facts()).toEqual([[10, 3 + 3 + 2]]);
    expect(full(hurt.facts())).toBeCloseTo(0.8, 12);
});

test('a level-up of hitpoints starts a count of the new base, and hitpoints above the base count 0', () => {
    const hurt = new HurtTicks();
    hurt.arm();
    hurt.tick(9, 10, true);   // 1 of 10
    hurt.tick(9, 11, true);   // the base rose: 2 of 11
    hurt.tick(13, 11, true);  // boosted above the base: 0
    hurt.tick(11, 11, true);
    expect(hurt.facts()).toEqual([[10, 1], [11, 2]]);
});

test('nothing counts before the arm, after the end, or while the player is dead; an end and an arm give []', () => {
    const hurt = new HurtTicks();
    hurt.tick(1, 10, true);   // before any arm: the reset's own ticks
    expect(hurt.facts()).toEqual([]);
    hurt.arm();
    hurt.tick(5, 10, true);
    hurt.tick(0, 10, false);  // the death's tick: the life is over, its count kept for its last result
    expect(hurt.facts()).toEqual([[10, 5]]);
    hurt.end();               // a reset begins (a failed reset's result then carries []) or the agent leaves
    expect(hurt.facts()).toEqual([]);
    hurt.tick(5, 10, true);   // the next reset's setup
    expect(hurt.facts()).toEqual([]);
    hurt.arm();
    hurt.tick(9, 10, true);
    hurt.arm();
    expect(hurt.facts()).toEqual([]);
    expect(() => hurt.tick(5, 0, true)).toThrow();
});
