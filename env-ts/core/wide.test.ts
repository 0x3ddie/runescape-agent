// item-v4, act-v4 and mask-v6 (core/items.ts, core/actions.ts and core/mask.ts), and mask-v5 unchanged by them.
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    A_ACT0, A_CLOSE_IF, A_DEPOSIT0, A_DROP0, A_HELD0, A_INV_NEXT, A_INV_PREV, A_USE0, A_WITHDRAW0, N_ACTIONS, N_ACTIONS_V4,
    pageEntry, pageRow
} from './actions';
import { selectHeld } from './affordances';
import { CLICK_OFFSETS } from './clicks';
import { describeAction, familyOf } from './feedback';
import { ItemPages, kindPage } from './items';
import { MASK_WIDE, buildMask } from './mask';
import { maskCases } from './mask-cases';
import { selectHeldWide } from './wide';

const kind = (slot: number, id: number, count = 1, name = `item ${id}`, options = ['Use']) => ({ slot, id, count, name, options });

test('mask-v5 is byte for byte the mask recorded before item-v4 and mask-v6 (240 cases, fixtures/mask-v5.json)', () => {
    const recorded = JSON.parse(readFileSync(new URL('./fixtures/mask-v5.json', import.meta.url), 'utf8'));
    expect(recorded.commit).toBe('063dfbb');
    expect(maskCases().map(c => buildMask(c.obs, { allowedActions: c.allowedActions }).join(''))).toEqual(recorded.masks);
});

test('act-v4 keeps rows 0..95 and puts entries 8..15 of the page families at 96..127, each row once', () => {
    const rows = new Map<number, string>();
    for (const [name, base] of [['held', A_HELD0], ['drop', A_DROP0], ['deposit', A_DEPOSIT0], ['use', A_USE0]] as const)
        for (let k = 0; k < 16; k++) {
            const row = pageRow(base, k);
            expect(rows.has(row)).toBe(false);
            rows.set(row, `${name} ${k}`);
            expect(pageEntry(row)).toEqual({ base, k });
            if (k < 8) expect(row).toBe(base + k);  // act-v3's own row
            else expect(row).toBeGreaterThanOrEqual(N_ACTIONS);
        }
    expect([...rows.keys()].filter(r => r >= N_ACTIONS).sort((a, b) => a - b)).toEqual(
        Array.from({ length: 32 }, (_, k) => N_ACTIONS + k));
    expect([pageRow(A_HELD0, 8), pageRow(A_DROP0, 8), pageRow(A_DEPOSIT0, 8), pageRow(A_USE0, 15)]).toEqual([96, 104, 112, 127]);
    expect([pageEntry(0), pageEntry(A_ACT0), pageEntry(A_WITHDRAW0), pageEntry(A_CLOSE_IF), pageEntry(84)]).toEqual(
        [null, null, null, null, null]);
    expect(() => pageRow(A_WITHDRAW0, 0)).toThrow();
    expect(() => pageRow(A_DROP0, 16)).toThrow();
    expect(N_ACTIONS_V4).toBe(128);
});

test('item-v4 lists 16 kinds a page by first slot with count and total; 17 kinds take two pages; a kind returns by its slot', () => {
    const pack = Array.from({ length: 28 }, (_, s) => kind(s, 2000 + (s % 17)));  // 17 kinds, 11 of them twice
    const first = kindPage(pack, 0, 28, 16), second = kindPage(pack, 1, 28, 16);
    expect(first.map(e => [e.slot, e.id, e.count, e.total])).toEqual(Array.from({ length: 16 }, (_, k) =>
        [k, 2000 + k, 1, k < 11 ? 2 : 1]));
    expect(second.map(e => [e.id, e.total])).toEqual([[2016, 1], ...Array.from({ length: 15 }, () => [-1, 0])]);
    expect(() => kindPage(pack, 2, 28, 16)).toThrow();  // 28 slots are at most 2 pages of 16
    const gone = pack.filter(i => i.id !== 2003);  // kind 2003 leaves: the kinds after it move up
    expect(kindPage(gone, 0, 28, 16).map(e => e.id).slice(2, 5)).toEqual([2002, 2004, 2005]);
    const back = [...gone.filter(i => i.slot !== 27), kind(27, 2003)];  // it returns at slot 27: listed last
    expect(kindPage(back, 1, 28, 16)[0].id).toBe(2003);
    const pages = new ItemPages();
    pages.reset('item-v4');
    const obs: Record<string, any> = { inventory: pack, bankOpen: false };
    pages.project(obs, []);
    expect([obs.contractVersion, obs.invPage, obs.invSel.length]).toEqual(['item-v4', 0, 16]);
    pages.turn(A_INV_NEXT); pages.turn(A_INV_NEXT); pages.project(obs, []);
    expect([obs.invPage, obs.invSel[0].id]).toEqual([1, 2016]);  // the last page is 1
    pages.turn(A_INV_PREV); pages.project(obs, []);
    expect(obs.invPage).toBe(0);
});

test('item-v4 pack options are selectHeld\'s entry by entry up to 16: equal to it while 8 fit, the picked kind skipped', () => {
    const options = ['Eat', 'Wield', 'Bury', 'Drop'];
    let seed = 7;
    const rand = (n: number) => { seed = (seed * 1103515245 + 12345) >>> 0; return seed % n; };
    for (let trial = 0; trial < 200; trial++) {
        const page = Array.from({ length: 16 }, (_, k) => rand(4) ? kind(k, 3000 + k, 1, `k${k}`, options.slice(0, rand(5))) :
            { slot: -1, id: -1, count: 0, name: '', options: [] });
        const using = rand(3) === 0 ? { slot: rand(16) } : null;
        const wide = selectHeldWide(page, using), narrow = selectHeld(page, using);
        expect(wide.slice(0, 8)).toEqual(narrow);  // the first 8 are act-v3's selection
        expect(wide.length).toBeLessThanOrEqual(16);
        expect(wide.every(h => h.slot !== using?.slot && h.option !== 'Drop')).toBe(true);
    }
    const many = Array.from({ length: 16 }, (_, k) => kind(k, 4000 + k, 1, `k${k}`, ['Eat']));
    expect(selectHeldWide(many).map(h => h.slot)).toEqual(Array.from({ length: 16 }, (_, k) => k));
    expect(selectHeld(many).length).toBe(8);
});

const wideObs = (inventory: any[], extra: Record<string, any> = {}) => {
    const pages = new ItemPages();
    pages.reset(extra.contract ?? 'item-v4');
    const obs: Record<string, any> = { inventory, bankOpen: false, modalOpen: false, dialogOpen: false, using: null,
        actTargetVersion: 'act-v3', actSel: [], clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: true, open: true })),
        style: { mode: 0, count: 3, stale: false }, ...extra };
    pages.project(obs, extra.bank ?? []);
    obs.heldSel = obs.contractVersion === 'item-v4' ? selectHeldWide(obs.invSel, obs.using) : selectHeld(obs.invSel, obs.using);
    return obs;
};

test('mask-v6 on 128 rows: a drop of any held entry, a tool and a pack under 24 stacks included; mask-v5 keeps both rules', () => {
    // 10 kinds in 10 slots: under 24 stacks; the axe and the tinderbox are tools mask-v5 protects
    const names = ['Bronze axe', 'Logs', 'Bread', 'Shrimps', 'Tinderbox', 'Feather', 'Bones', 'Coins', 'Raw chicken', 'Bronze sword'];
    const pack = names.map((n, s) => kind(s, 5000 + s, 1, n, s === 2 || s === 3 ? ['Eat'] : s === 9 ? ['Wield'] : []));
    const obs = wideObs(pack);
    const mask = buildMask(obs, {});
    expect(mask).toHaveLength(128);
    for (let k = 0; k < 10; k++) {
        expect(mask[pageRow(A_DROP0, k)]).toBe(1);  // the axe (entry 0) and the tinderbox (4) too
        expect(mask[pageRow(A_USE0, k)]).toBe(1);
    }
    expect([mask[pageRow(A_DROP0, 9)], mask[pageRow(A_DROP0, 10)], mask[pageRow(A_USE0, 10)]]).toEqual([1, 0, 0]);
    expect(obs.heldSel.map((h: any) => h.slot)).toEqual([2, 3, 9]);  // Eat, Eat, Wield
    expect([mask[A_HELD0], mask[A_HELD0 + 2], mask[A_HELD0 + 3]]).toEqual([1, 1, 0]);
    expect([mask[A_INV_NEXT], mask[A_INV_PREV]]).toEqual([1, 0]);  // page 0 of 2
    const v5 = buildMask(wideObs(pack, { contract: 'item-v3' }), {});
    expect(v5).toHaveLength(96);
    expect(Array.from({ length: 8 }, (_, k) => v5[A_DROP0 + k])).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);  // under 24 stacks
    const full = [...pack, ...Array.from({ length: 16 }, (_, s) => kind(10 + s, 6000 + s))];  // 26 stacks: pressure
    const v5full = buildMask(wideObs(full, { contract: 'item-v3' }), {});
    expect([v5full[A_DROP0], v5full[A_DROP0 + 1], v5full[A_DROP0 + 4]]).toEqual([0, 1, 0]);  // the tools stay protected
    const wfull = buildMask(wideObs(full), {});
    expect([wfull[A_DROP0], wfull[pageRow(A_DROP0, 15)], wfull[pageRow(A_USE0, 15)]]).toEqual([1, 1, 1]);  // 16 entries
});

test('mask-v6 keeps mask-v5\'s other states: the bank (deposits at both groups), a picked item, a dialog, the last page', () => {
    const pack = Array.from({ length: 12 }, (_, s) => kind(s, 7000 + s, 1, `k${s}`, ['Eat']));
    const bank = buildMask(wideObs(pack, { bankOpen: true, bank: [{ slot: 0, id: 1, count: 2, name: 'Logs' }] }), {});
    expect([bank[pageRow(A_DEPOSIT0, 0)], bank[pageRow(A_DEPOSIT0, 11)], bank[pageRow(A_DEPOSIT0, 12)]]).toEqual([1, 1, 0]);
    expect([bank[A_WITHDRAW0], bank[pageRow(A_DROP0, 0)], bank[A_CLOSE_IF]]).toEqual([1, 0, 1]);
    const picked = wideObs(pack, { using: { slot: 3, id: 7003, name: 'k3' } });
    const use = buildMask(picked, {});
    expect(picked.heldSel.map((h: any) => h.slot)).toEqual([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]);  // all but the picked
    expect([use[pageRow(A_HELD0, 10)], use[pageRow(A_HELD0, 11)], use[pageRow(A_DROP0, 0)], use[A_CLOSE_IF]]).toEqual([1, 0, 0, 1]);
    const dialog = buildMask(wideObs(pack, { dialogOpen: true, dialogOptions: [] }), {});
    expect(dialog.slice(A_DROP0, A_DROP0 + 8).concat(dialog.slice(96))).toEqual(new Array(40).fill(0));
    const last = wideObs([...pack, ...Array.from({ length: 6 }, (_, s) => kind(12 + s, 8000 + s))]);
    const pages = new ItemPages(); pages.reset('item-v4'); pages.turn(A_INV_NEXT);
    last.invPage = 1;
    expect([buildMask(last, {})[A_INV_NEXT], buildMask(last, {})[A_INV_PREV]]).toEqual([0, 1]);  // next only on page 0
    expect(MASK_WIDE).toBe('mask-v6');
});

test('the feedback names the family and entry of a row of either group', () => {
    const page = Array.from({ length: 16 }, (_, k) => kind(k, 9000 + k, 1, `k${k}`, ['Eat']));
    const obs = { invSel: page, heldSel: selectHeldWide(page), using: null };
    expect([familyOf(96), familyOf(104), familyOf(112), familyOf(127)]).toEqual(['held', 'drop', 'deposit', 'use']);
    expect(describeAction(obs, pageRow(A_HELD0, 9))).toMatchObject({ family: 'held', name: 'k9', option: 'Eat' });
    expect(describeAction(obs, pageRow(A_DROP0, 12))).toMatchObject({ family: 'drop', name: 'k12', option: 'Drop' });
    expect(describeAction(obs, pageRow(A_DEPOSIT0, 8))).toMatchObject({ family: 'deposit', name: 'k8', option: 'Deposit' });
    expect(describeAction(obs, pageRow(A_USE0, 15))).toMatchObject({ family: 'use', name: 'k15', option: 'Use' });
    expect(describeAction(obs, A_DROP0 + 3)).toMatchObject({ family: 'drop', name: 'k3' });  // act-v3's row as before
    expect(() => familyOf(128)).toThrow();
});
