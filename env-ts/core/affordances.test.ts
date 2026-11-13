import { expect, test } from 'bun:test';
import { checkActContract, publishAffordances, selectAffordances, selectHeld, type ActCandidate } from './affordances';
import { buildMask } from './mask';
import { A_ACT0, N_ACT, optionLabelFor } from './actions';
import { CLICK_OFFSETS } from './clicks';

const p = { x: 100, z: 100, level: 0 };
const thing = (id: number, x: number, name = 'Tree', options = ['Chop down'], kind: ActCandidate['kind'] = 'loc', z = 100):
    ActCandidate => ({ kind, id, x, z, level: 0, name, options });
const near = (c: ActCandidate) => Math.max(Math.abs(c.x - p.x), Math.abs(c.z - p.z));
const pick = (cs: ActCandidate[], steps = near) => selectAffordances(cs, p, steps).map(c => c.affordance);

test('order is spatial and deterministic across scan orders, never by species or level', () => {
    const input = [thing(1308, 103, 'Willow'), thing(1276, 101), thing(1281, 102, 'Oak')];
    expect(pick(input).map(a => a.id)).toEqual([1276, 1281, 1308]);
    expect(pick(input)).toEqual(pick([...input].reverse()));
});

test('one affordance per option, so a thing with two options fills two slots', () => {
    const fish = thing(316, 101, 'Fishing spot', ['Net', 'Bait'], 'npc');
    expect(pick([fish]).map(a => a.option)).toEqual(['Bait', 'Net']);
});

test('every kind of affordance is listed before a second of one already listed', () => {
    // Near a bank, twelve men and doors once pushed the only tree out of a nearest-first cap.
    const crowd = Array.from({ length: 20 }, (_, k) => thing(3000 + k, 101 + (k % 3), 'Man', ['Talk-to'], 'npc', 100 + k % 5));
    const listed = pick([...crowd, thing(1276, 112)]);
    expect(listed).toHaveLength(N_ACT);
    expect(listed.some(a => a.name === 'Tree')).toBe(true);
});

test('out of reach, on another plane, beyond the scan or without an option: not listed', () => {
    const input = [thing(1, 101), thing(2, 116), { ...thing(3, 101), level: 1 }, thing(4, 102, 'Rock', []), thing(5, 99)];
    expect(pick(input, c => c.x < 100 ? -1 : near(c)).map(a => a.id)).toEqual([1]);
});

test('affordances copy only public fields, and the candidate behind each comes back', () => {
    const raw: any = { ...thing(1276, 101), entry: { secret: true } }; raw.entry.self = raw;
    const [chosen] = selectAffordances([raw, raw], p, near);
    expect(chosen.affordance).not.toHaveProperty('entry');
    expect(() => JSON.stringify(chosen.affordance)).not.toThrow();
    expect(chosen.candidate).toBe(raw);
});

test('a refused affordance is left out until it is allowed again', () => {
    expect(selectAffordances([thing(1276, 101)], p, near, a => a.id === 1276)).toEqual([]);
});

test('held options come from the visible page, in slot order, and leave Drop to its own rows', () => {
    const page = [{ slot: 0, id: 1351, name: 'Bronze axe', count: 1, options: ['Wield', 'Drop'] },
        { slot: 1, id: -1, name: '', count: 0 }, { slot: 2, id: 2309, name: 'Bread', count: 1, options: ['Eat'] }];
    expect(selectHeld(page).map(h => `${h.name}:${h.option}`)).toEqual(['Bronze axe:Wield', 'Bread:Eat']);
});

test('published slots are legal actions, and act-v3 needs item-v2', () => {
    const obs: Record<string, any> = { player: { ...p, hp: 10, maxHp: 10 }, contractVersion: 'item-v2', invPage: 0,
        bankPage: 0, bankPages: 1, bankSel: [], invSel: [], inventory: [], npcs: [], dialogOptions: [], bankOpen: false,
        style: { mode: 0, count: 3, stale: false }, clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: true, open: true })) };
    publishAffordances(obs, [thing(1276, 101), thing(1278, 104)], near);
    expect(buildMask(obs, {}).slice(A_ACT0, A_ACT0 + 3)).toEqual([1, 1, 0]);
    expect(() => checkActContract({ contract: 'item-v2', actTargets: 'act-v3' })).not.toThrow();
    for (const bad of [{}, { actTargets: 'act-v3' }, { contract: 'item-v2', actTargets: 'act-v2' }, { contract: 'item-v2', actTargets: 'gather-v1' }])
        expect(() => checkActContract(bad)).toThrow();
});

test('different slots supersede; the same slot continues its option', () => {
    expect(optionLabelFor(A_ACT0)).toBe('act_0');
    expect(optionLabelFor(A_ACT0)).not.toBe(optionLabelFor(A_ACT0 + 1));
    // a click's walk is a 'nav' option (the tracker's stall budget); each slot its own
    expect([optionLabelFor(17), optionLabelFor(28), optionLabelFor(84), optionLabelFor(95)])
        .toEqual(['nav_click_0', 'nav_click_11', 'nav_click_12', 'nav_click_23']);
    expect(optionLabelFor(71)).toBeNull();
});

test('with an item picked for use, every loc and npc offers Use, and every other held item is a target', () => {
    const obs: Record<string, any> = { player: p, using: { slot: 1, id: 590, name: 'Tinderbox' },
        invSel: [{ slot: 0, id: 1511, name: 'Logs', count: 1, options: [] }, { slot: 1, id: 590, name: 'Tinderbox', count: 1 }] };
    const fire = thing(2732, 101, 'Fire', [], 'loc'), cook = thing(278, 102, 'Cook', ['Talk-to'], 'npc');
    const sources = publishAffordances(obs, [fire, cook, thing(1511, 103, 'Logs', ['Take'], 'obj')], near);
    expect(obs.actSel.map((a: any) => `${a.name}:${a.option}`)).toEqual(['Fire:Use', 'Cook:Use']);
    expect(sources).toEqual([fire, cook]); // the candidates themselves, for the effector
    expect(obs.heldSel.map((h: any) => `${h.name}:${h.option}`)).toEqual(['Logs:Use']);
});
