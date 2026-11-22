import { expect, test } from 'bun:test';
import { ItemPages, itemPage, kindPage, checkItemContract, keptPick, matchItem } from './items';
import { selectHeld } from './affordances';
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

// ---- item-v3, the page of kinds ----------------------------------------------------------------------------------
const kind = (slot: number, id: number, count = 1, name = `item ${id}`) => ({ slot, id, count, name, options: ['Use'] });
const PACK = [kind(0, 1351, 1, 'Bronze axe'), kind(1, 1511, 1, 'Logs'), kind(2, 2309, 1, 'Bread'), kind(3, 1511),
    kind(4, 1511), kind(5, 995, 250, 'Coins'), kind(6, 2138), kind(7, 2138), kind(9, 590, 1, 'Tinderbox'), kind(12, 1511)];

test('item-v3 lists each kind once, in the order of its first slot, with that slot\'s count and the pack\'s total', () => {
    const page = kindPage(PACK, 0, 28);
    expect(page.map(e => [e.slot, e.id, e.count, e.total])).toEqual([[0, 1351, 1, 1], [1, 1511, 1, 4], [2, 2309, 1, 1],
        [5, 995, 250, 250], [6, 2138, 1, 2], [9, 590, 1, 1], [-1, -1, 0, 0], [-1, -1, 0, 0]]);
    expect(page[1]).toEqual({ ...PACK[1], total: 4 });  // the first slot's own item: name, options, its count
    expect(page[6]).toEqual({ slot: -1, id: -1, count: 0, name: '', total: 0 });
    PACK[1].count = 9; expect(page[1].count).toBe(1); PACK[1].count = 1;  // a copy
    expect(matchItem(PACK[1], page[1])).toBe(true);   // the stale-selection check reads the entry's own slot
    expect(matchItem({ ...PACK[1], count: 2 }, page[1])).toBe(false);
    expect(() => kindPage([kind(0, 5), kind(0, 6)], 0, 28)).toThrow();  // a slot held twice
    expect(() => kindPage(PACK, 4, 28)).toThrow();  // four pages of kinds, as of slots
});

test('item-v3 pages hold 8 kinds each, and a kind that leaves and returns is listed by its new first slot', () => {
    const many = Array.from({ length: 28 }, (_, k) => kind(k, 2000 + (k % 11)));  // 11 kinds over 28 slots
    expect(kindPage(many, 0, 28).map(e => e.id)).toEqual([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]);
    expect(kindPage(many, 1, 28).map(e => [e.id, e.total])).toEqual([[2008, 2], [2009, 2], [2010, 2],
        [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]]);
    expect(kindPage(many, 3, 28).every(e => e.slot === -1)).toBe(true);
    const gone = PACK.filter(i => i.id !== 1511);  // every log burned: the kinds after it move up
    expect(kindPage(gone, 0, 28).map(e => e.id).slice(0, 5)).toEqual([1351, 2309, 995, 2138, 590]);
    const back = [...gone, kind(20, 1511, 1, 'Logs')];  // a log taken again lands in slot 20: listed by that slot
    expect(kindPage(back, 0, 28).map(e => [e.id, e.slot]).slice(0, 6)).toEqual(
        [[1351, 0], [2309, 2], [995, 5], [2138, 6], [590, 9], [1511, 20]]);
});

test('item-v3 is a contract of its own: the reset names it, every observation stamps it, item-v2 stays the default', () => {
    expect(() => checkItemContract({ contract: 'item-v3' })).not.toThrow();
    expect(() => checkItemContract({ contract: 'item-v5' })).toThrow('unsupported item contract');
    const pages = new ItemPages();
    expect(pages.contract).toBe('item-v2');
    expect(() => pages.reset('item-v5')).toThrow('unsupported item contract');
    pages.reset('item-v3');
    const o = obs(PACK);
    pages.project(o, []);
    expect([o.contractVersion, o.invSel.map((e: any) => e.slot)]).toEqual(['item-v3', [0, 1, 2, 5, 6, 9, -1, -1]]);
    expect(mask(o)).toHaveLength(N_ACTIONS);  // mask-v5 takes the page of kinds
    pages.turn(A_INV_NEXT); pages.project(o, []);
    expect([o.invPage, o.invSel.every((e: any) => e.slot === -1)]).toEqual([1, true]);  // a turn to an empty page
    pages.reset(); pages.project(o, []);
    expect([o.contractVersion, o.invSel.map((e: any) => e.slot)]).toEqual(['item-v2', [0, 1, 2, 3, 4, 5, 6, 7]]);
    expect('total' in o.invSel[0]).toBe(false);  // item-v2 entries are as before
});

test('item-v3 pack targets of a picked item are the page\'s other kinds, each once; its own kind is not one', () => {
    const page = kindPage(PACK, 0, 28);
    const using = { slot: 1 };  // USE_1 picked the logs of slot 1, the entry's first slot
    const targets = selectHeld(page, using);
    expect(targets.map(h => [h.slot, h.id, h.option])).toEqual([[0, 1351, 'Use'], [2, 2309, 'Use'], [5, 995, 'Use'],
        [6, 2138, 'Use'], [9, 590, 'Use']]);
    expect(targets.every(h => h.count === page.find(e => e.slot === h.slot)!.count)).toBe(true);  // the slot's count
});

test('a pick is of the item under item-v2 and of the kind under item-v3, which is never its own target', () => {
    // logs in slots 3 and 5, slot 1 free; USE picked the logs of slot 3 (the kind's first slot then)
    const log = (slot: number) => kind(slot, 1511, 1, 'Logs');
    const pack = [kind(0, 1351, 1, 'Bronze axe'), log(3), log(5), kind(6, 2309, 1, 'Bread')];
    const using = { slot: 3, id: 1511, name: 'Logs' };
    const landed = [...pack, log(1)];   // a chop still running lands a log in slot 1: the kind's first slot is now 1
    const v3 = keptPick(using, landed, 'item-v3', false);
    expect(v3).toEqual({ slot: 1, id: 1511, name: 'Logs' });
    const targets = selectHeld(kindPage(landed, 0, 28), v3);
    expect(targets.map(h => [h.slot, h.id])).toEqual([[0, 1351], [6, 2309]]);   // no log among the targets
    expect(selectHeld(kindPage(landed, 0, 28), using).some(h => h.id === 1511)).toBe(true);   // as it was before
    expect(keptPick(using, pack.filter(i => i.id !== 1511), 'item-v3', false)).toBeNull();   // the kind left the pack
    expect(keptPick(using, landed, 'item-v3', true)).toBeNull();   // the bank opened
    // item-v2: the pick stays the item of slot 3 while slot 3 holds logs, the same object; none once it does not
    expect(keptPick(using, landed, 'item-v2', false)).toBe(using);
    expect(keptPick(using, [...pack.filter(i => i.slot !== 3), kind(3, 2309)], 'item-v2', false)).toBeNull();
    expect(keptPick(using, pack.filter(i => i.slot !== 3), 'item-v2', false)).toBeNull();
    expect(keptPick(using, landed, 'item-v2', true)).toBeNull();
    expect(keptPick(null, landed, 'item-v3', false)).toBeNull();
});
