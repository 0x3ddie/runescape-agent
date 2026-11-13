// env-ts/core/budgets.test.ts: episode budget and tick-budget semantics.
//
// Small but load-bearing: `charged` accounting is what made v21f readable as an
// environment defect rather than an RL result, and tickBudgetMs is what keeps
// option patience meaning the same thing at 20ms and 8ms per tick.

import { describe, expect, test } from 'bun:test';

import { MAX_DECISIONS, MAX_EPISODE_TICKS, NAV_STALL_TICKS, isTruncated, tickBudgetMs } from './budgets';

describe('tickBudgetMs converts game ticks at the current world speed', () => {
    test('defaults to 20ms per tick when cfg.speedMs is absent', () => {
        expect(tickBudgetMs(NAV_STALL_TICKS)).toBe(1200); // the original wall-clock constant
        expect(tickBudgetMs(NAV_STALL_TICKS, undefined)).toBe(1200);
    });

    test('a faster world shrinks the wall-clock budget, keeping game-time meaning fixed', () => {
        // At 8ms/tick the old hardcoded 1200ms silently became 150 ticks of patience.
        expect(tickBudgetMs(NAV_STALL_TICKS, 8)).toBe(480);
        expect(tickBudgetMs(NAV_STALL_TICKS, 5)).toBe(300);
    });
});

describe('isTruncated charges only decisions that offered a choice', () => {
    test('forced steps do not end an episode', () => {
        // An interface option holds the mask at WAIT-only for ~8-20 steps. Counting
        // those against the budget left ~26 real choices per episode in v21f.
        expect(isTruncated({ charged: 0, episodeTicks: 0 }, {})).toBe(false);
        expect(isTruncated({ charged: MAX_DECISIONS - 1, episodeTicks: 0 }, {})).toBe(false);
        expect(isTruncated({ charged: MAX_DECISIONS, episodeTicks: 0 }, {})).toBe(true);
    });

    test('game time is an independent budget, so a few very long options still stop', () => {
        expect(isTruncated({ charged: 0, episodeTicks: MAX_EPISODE_TICKS }, {})).toBe(true);
    });

    test('cfg overrides both budgets', () => {
        expect(isTruncated({ charged: 400, episodeTicks: 0 }, { maxDecisions: 400 })).toBe(true);
        expect(isTruncated({ charged: 399, episodeTicks: 0 }, { maxDecisions: 400 })).toBe(false);
        expect(isTruncated({ charged: 0, episodeTicks: 5000 }, { maxEpisodeTicks: 5000 })).toBe(true);
    });
});
