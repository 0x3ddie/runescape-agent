import { expect, test } from 'bun:test';
import { ItemPages, itemPage, checkItemContract, matchItem } from './items';
import { buildMask } from './mask';
import { A_DEPOSIT0, A_WITHDRAW0, A_INV_NEXT, A_BANK_NEXT, A_RETIRED, N_ACTIONS } from './actions';
import { CLICK_OFFSETS, CLICK_ROWS } from './clicks';

const it = (slot: number, id = slot + 100, count = 1, name = 'Bronze axe') => ({ slot, id, count, name });
const obs = (inventory = [it(0)]) => ({ inventory, bankOpen: false, modalOpen: false,
    player: { hp: 10, maxHp: 10 }, skills: [], equipment: [], npcs: [], locs: [], ground: [],
    invSel: inventory, actTargetVersion: 'act-v3', actSel: [], heldSel: [],
    clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: true, open: true })),
    style: { mode: 0, count: 3, stale: false } } as Record<string, any>);
const mask = (o: any, allowedActions?: number[]) => buildMask(o, { allowedActions });

test('physical inventory pages reach every slot without item ranking', () => {
    const pages = new ItemPages(); const o = obs(Array.from({ length: 28 }, (_, k) => it(k)));
    const seen = [];
    for (let p = 0; p < 4; p++) {
        pages.project(o, []);
        seen.push(...o.invSel.filter((i: any) => i.count > 0).map((i: any) => i.slot));
        pages.turn(A_INV_NEXT);
    }
    expect(seen).toEqual(Array.from({ length: 28 }, (_, k) => k));
    expect(o.invSel.slice(4).every((i: any) => i.id === -1 && i.count === 0)).toBe(true);
});
test('sparse slots keep position, raw item zero is valid, and snapshots are copied', () => {
    const source = [it(7, 0)]; const page = itemPage(source, 0, 28);
    expect(page[0].count).toBe(0); expect(page[7].id).toBe(0);
    source[0].count = 9; expect(page[7].count).toBe(1);
    expect(() => itemPage([it(0), it(0)], 0, 28)).toThrow();
    expect(() => itemPage([it(28)], 0, 28)).toThrow();
});
test('closed bank reveals no bank inventory or remembered extent; reopening starts page zero', () => {
    const pages = new ItemPages(); const o = obs(); o.bankOpen = true;
    pages.project(o, [it(239)]); expect(o.bankPages).toBe(30);
    for (let p = 0; p < 29; p++) pages.turn(A_BANK_NEXT);
    pages.project(o, [it(239)]); expect(o.bankSel[7].slot).toBe(239);
    o.bankOpen = false; pages.project(o, [it(239)]);
    expect(o.bankPages).toBe(1); expect(o.bankPage).toBe(0);
    expect(o.bankSel.every((i: any) => i.count === 0)).toBe(true);
    pages.reset(); expect(pages.invPage).toBe(0);
});
test('selection rejects changed slot, identity, count and empty slots', () => {
    expect(matchItem(it(2), it(2))).toBe(true);
    expect(matchItem(it(1), it(2))).toBe(false);
    expect(matchItem(it(2, 7), it(2))).toBe(false);
    expect(matchItem(it(2, 102, 3), it(2))).toBe(false);
    expect(matchItem(undefined, it(2))).toBe(false);
});
test('a tool is deposited only in the open bank, where it can be withdrawn again', () => {
    expect(() => mask(obs())).toThrow(); // an observation without item pages is refused
    const o = obs(); const pages = new ItemPages(); pages.project(o, []);
    expect(mask(o)[A_DEPOSIT0]).toBe(0);
    o.bankOpen = true; o.modalOpen = true; pages.project(o, [it(0)]);
    expect(mask(o)[A_DEPOSIT0]).toBe(1); expect(mask(o)[A_WITHDRAW0]).toBe(1);
    expect(mask(o)[A_DEPOSIT0 + 1]).toBe(0);
    expect(mask(o)[A_RETIRED]).toBe(0); expect(mask(o)).toHaveLength(N_ACTIONS);
    expect(CLICK_ROWS.some(row => mask(o)[row])).toBe(false); // an open bank offers no click
    expect(mask(o, [0, A_DEPOSIT0])[A_DEPOSIT0]).toBe(0); // no way back without the withdrawals
});
test('every config but item-v2 is rejected, as are deposits without the way back', () => {
    for (const cfg of [{}, { contract: 'legacy-v1' }, { contract: 'future' }, { contract: 'item-v2', allowedActions: [0, A_DEPOSIT0] }])
        expect(() => checkItemContract(cfg)).toThrow();
    expect(() => checkItemContract({ contract: 'item-v2', allowedActions: [0] })).not.toThrow();
});
