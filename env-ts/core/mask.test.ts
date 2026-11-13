// env-ts/core/mask.test.ts: the mask contract, run with `bun test env-ts/core`.
//
// Every assertion below pins a rule that was bought with a real bug, named in the
// comment. These are pure unit tests: no engine, no world.

import { describe, expect, test } from 'bun:test';

import {
    A_ACT0, A_CLOSE_IF, A_DEPOSIT0, A_DIALOG_CHOICE0, A_DIALOG_CONTINUE, A_DROP0, A_HELD0, A_INV_NEXT, A_INV_PREV,
    A_RETIRED, A_STYLE0, A_USE0, A_WITHDRAW0, N_ACTIONS
} from './actions';
import { CLICK_OFFSETS, CLICK_ROWS } from './clicks';
import { buildMask, type MaskContext } from './mask';

const baseObs = (over: Record<string, any> = {}): Record<string, any> => ({
    contractVersion: 'item-v2', invPage: 0, bankPage: 0, bankPages: 1, bankSel: [],
    actTargetVersion: 'act-v3', actSel: [], heldSel: [],
    // the clicks of a player whose every target lies outside the declared world (none legal) unless a test says so
    clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: false, open: true })),
    player: { x: 3222, z: 3222, hp: 10, maxHp: 10, inCombat: false },
    inventory: [], npcs: [], invSel: [], dialogOptions: [],
    bankOpen: false, dialogOpen: false, modalOpen: false,
    style: { mode: 0, count: 3, stale: false }, // a new player, unarmed: three styles, the first selected
    ...over
});
const idle: MaskContext = {};
const legalSet = (mask: number[]): number[] => mask.flatMap((v, i) => (v ? [i] : []));
const stacks = (n: number, name = 'logs') => Array.from({ length: n }, (_, i) => ({ slot: i, id: 1511, name, count: 1 }));
const act = (option: string) => ({ kind: 'loc', id: 1276, name: 'Tree', option, x: 3223, z: 3222, level: 0 });

describe('shape and the floor', () => {
    test('mask is 96 wide, WAIT is always legal, RETREAT never is (act-v3)', () => {
        const all = Array.from({ length: N_ACTIONS }, (_, i) => i);
        const fight = { player: { x: 3222, z: 3222, hp: 10, maxHp: 10, inCombat: true }, npcs: [{ name: 'Goblin', inCombat: true }],
            retreat: true };
        const mask = buildMask(baseObs({ actSel: Array(16).fill(act('Chop down')), ...fight }), { allowedActions: all });
        expect(mask).toHaveLength(N_ACTIONS);
        expect(mask[0]).toBe(1);
        expect(mask[A_RETIRED]).toBe(0);
    });

    test('an idle agent in an empty world can wait, page and choose another combat style', () => {
        expect(legalSet(buildMask(baseObs(), idle))).toEqual([0, A_INV_NEXT, A_STYLE0 + 1, A_STYLE0 + 2, A_STYLE0 + 3]);
    });

    test('the cfg allowlist restricts the mask but restores WAIT', () => {
        expect(legalSet(buildMask(baseObs(), { allowedActions: [A_STYLE0 + 1] }))).toEqual([0, A_STYLE0 + 1]);
    });

    test('an observation without the act-v3 or item-v2 contract, or without its 24 clicks, is refused', () => {
        expect(() => buildMask(baseObs({ actTargetVersion: 'act-v2' }), idle)).toThrow();
        expect(() => buildMask(baseObs({ clicks: undefined }), idle)).toThrow();
        expect(() => buildMask(baseObs({ clicks: [] }), idle)).toThrow();
        expect(() => buildMask(baseObs({ actTargetVersion: undefined }), idle)).toThrow();
        expect(() => buildMask(baseObs({ contractVersion: 'legacy-v1' }), idle)).toThrow();
        expect(() => buildMask(baseObs({ actSel: Array(17).fill(act('Chop down')) }), idle)).toThrow();
        expect(() => buildMask(baseObs({ style: undefined }), idle)).toThrow();
        expect(() => buildMask(baseObs({ style: { mode: 0, count: 0, stale: false } }), idle)).toThrow();
    });
});

describe('published slots', () => {
    test('every published affordance and held option is a legal action, and nothing else', () => {
        const actSel = [act('Chop down'), act('Look at')];
        const heldSel = [{ slot: 0, id: 1351, name: 'Bronze axe', count: 1, option: 'Wield' }];
        const mask = buildMask(baseObs({ actSel, heldSel }), idle);
        expect(mask.slice(A_ACT0, A_ACT0 + 3)).toEqual([1, 1, 0]);
        expect(mask.slice(A_HELD0, A_HELD0 + 2)).toEqual([1, 0]);
    });

    test('any held item can be picked for use; then only using it, turning the page or putting it back is offered', () => {
        const invSel = [{ slot: 0, id: 590, name: 'Tinderbox', count: 1 }, { slot: -1, id: -1, name: '', count: 0 }];
        expect(buildMask(baseObs({ invSel }), idle).slice(A_USE0, A_USE0 + 2)).toEqual([1, 0]);
        const using = baseObs({ invSel, using: { slot: 0, id: 590, name: 'Tinderbox' }, actSel: [act('Use')],
            heldSel: [{ slot: 1, id: 1511, name: 'Logs', count: 1, option: 'Use' }] });
        // mask-v5: the page turns stay legal with an item selected (mask-v3 left only the use and the way back)
        expect(legalSet(buildMask(using, idle))).toEqual([0, A_ACT0, A_HELD0, A_CLOSE_IF, A_INV_NEXT]);
        expect(legalSet(buildMask({ ...using, invPage: 3 }, idle))).toEqual([0, A_ACT0, A_HELD0, A_CLOSE_IF, A_INV_PREV]);
    });

    test('a click is legal where its target lies in the declared world, blocked or not, and in no interface mode', () => {
        // mask-v3 (and v5): no reachability or walkability judgement; the game decides where a click goes.
        const clicks = CLICK_OFFSETS.map(([dx, dz], k) => ({ dx, dz, inside: k !== 5, open: k % 2 === 0 }));
        const mask = buildMask(baseObs({ clicks }), idle);
        expect(CLICK_ROWS.map(row => mask[row])).toEqual(CLICK_OFFSETS.map((_, k) => k === 5 ? 0 : 1));
        for (const mode of [{ bankOpen: true, modalOpen: true }, { dialogOpen: true }, { using: { slot: 0, id: 590, name: 'Tinderbox' } }])
            expect(CLICK_ROWS.some(row => buildMask(baseObs({ clicks, ...mode }), idle)[row])).toBe(false);
    });
});

describe('interfaces always have an exit', () => {
    test('a choice dialog offers its choices and CLOSE_IF, not DIALOG_CONTINUE', () => {
        // sendClickDialog(0) clears a mesbox but does nothing to a choice dialog.
        // Offering both meant most of the legal set were no-ops and the policy sat
        // in the guard conversation indefinitely.
        const mask = buildMask(baseObs({ dialogOpen: true, actSel: [act('Chop down')],
            dialogOptions: [{ text: 'Yes', componentId: 1 }, { text: 'No', componentId: 2 }] }), idle);
        expect(legalSet(mask)).toEqual([0, A_CLOSE_IF, A_DIALOG_CHOICE0, A_DIALOG_CHOICE0 + 1]);
    });

    test('a continue-only mesbox offers DIALOG_CONTINUE', () => {
        const mask = buildMask(baseObs({ dialogOpen: true, dialogOptions: [{ text: 'Click here to continue' }] }), idle);
        expect(legalSet(mask)).toEqual([0, A_CLOSE_IF, A_DIALOG_CONTINUE]);
    });

    test('an open bank deposits any item, tools included, withdraws, and can be closed', () => {
        // v21f banked its axe with no way to withdraw it and waited out the episode;
        // item-v2 can withdraw, so a deposit is reversible and the bank takes any item.
        const invSel = [{ slot: 0, id: 1351, name: 'Bronze axe', count: 1 }, { slot: 1, id: 1511, name: 'Logs', count: 1 }];
        const bankSel = [{ slot: 0, id: 995, name: 'Coins', count: 25 }];
        const mask = buildMask(baseObs({ bankOpen: true, modalOpen: true, invSel, bankSel, actSel: [act('Chop down')] }), idle);
        expect(legalSet(mask)).toEqual([0, A_DEPOSIT0, A_DEPOSIT0 + 1, A_WITHDRAW0, A_CLOSE_IF, A_INV_NEXT]);
    });
});

describe('the rules bought with bugs', () => {
    test('DROP waits for inventory pressure and never takes a tool', () => {
        // Ungated, DROP_0 targets slot 0, the axe, so an exploring policy destroyed its
        // own axe within the first few decisions of nearly every episode.
        const invSel = [{ slot: 0, id: 1351, name: 'Bronze axe', count: 1 }, { slot: 1, id: 1511, name: 'Logs', count: 1 }];
        expect(buildMask(baseObs({ inventory: stacks(23), invSel }), idle).slice(A_DROP0, A_DROP0 + 2)).toEqual([0, 0]);
        expect(buildMask(baseObs({ inventory: stacks(24), invSel }), idle).slice(A_DROP0, A_DROP0 + 2)).toEqual([0, 1]);
        for (const name of ['Bronze pickaxe', 'Small fishing net', 'Tinderbox', 'Fishing rod', 'Harpoon', 'Knife'])
            expect(buildMask(baseObs({ inventory: stacks(24), invSel: [{ slot: 0, id: 1, name, count: 1 }] }), idle)[A_DROP0]).toBe(0);
    });

    test('RETREAT is never offered, even in a fight: act-v3 flees by clicks', () => {
        expect(buildMask(baseObs({ player: { hp: 10, maxHp: 10, inCombat: true }, retreat: true }), idle)[A_RETIRED]).toBe(0);
        expect(buildMask(baseObs({ npcs: [{ name: 'Goblin', inCombat: true }], retreat: true }), idle)[A_RETIRED]).toBe(0);
    });

    test('a combat-style row is offered only where it changes something (mask-v2)', () => {
        // all four rows were always legal, and greedy decoding repeated a request for the
        // current style, which changes nothing, until the episode ended.
        const rows = (mode: number, count: number, stale = false) => buildMask(baseObs({ style: { mode, count, stale } }), idle)
            .slice(A_STYLE0, A_STYLE0 + 4);
        expect(rows(1, 4)).toEqual([1, 0, 1, 1]); // the current style is no request
        expect(rows(2, 3)).toEqual([1, 1, 0, 0]); // three styles: row 3 selects the last one too
        expect(rows(2, 4)).toEqual([1, 1, 0, 1]); // a Wield of a four-style weapon makes row 3 a style of its own
        expect(rows(0, 3, true)).toEqual([1, 1, 1, 1]); // the reset's levels wait for the recompute row 0 runs
    });
});
