// env-ts/core/mask-cases.ts: fixed observations for mask-v5's recorded masks (core/fixtures/mask-v5.json, recorded
// before item-v4 and mask-v6): item-v2 and item-v3 pages in every state the mask reads (the plain state with
// and without pack pressure and tools, an open bank with bank pages and allow-lists, an item picked, a dialog with and
// without choices, a modal, each inventory page, combat styles), drawn from a fixed seed. Tests only.
import { CLICK_OFFSETS } from './clicks';

const NAMES = ['Bronze axe', 'Logs', 'Bread', 'Shrimps', 'Tinderbox', 'Coins', 'Raw chicken', 'Bronze sword', 'Small fishing net',
    'Bronze pickaxe', 'Feather', 'Bones'];

export function maskCases(): { obs: Record<string, any>; allowedActions?: number[] }[] {
    let seed = 20261120;
    const rand = (n: number) => { seed = (seed * 1103515245 + 12345) >>> 0; return seed % n; };
    const cases: { obs: Record<string, any>; allowedActions?: number[] }[] = [];
    for (let c = 0; c < 240; c++) {
        const contract = c % 2 ? 'item-v3' : 'item-v2';
        const inventory = [];
        const filled = [5, 12, 23, 24, 26, 28][rand(6)];
        for (let slot = 0; slot < 28 && inventory.length < filled; slot++) {
            if (rand(4) === 0 && filled < 28) continue;
            const k = rand(NAMES.length);
            inventory.push({ slot, id: 1000 + k, count: k === 5 ? 1 + rand(300) : 1, name: NAMES[k], options: ['Eat', 'Wield'].slice(0, rand(3)) });
        }
        const invPage = rand(4), page = Array.from({ length: 8 }, (_, k) =>
            inventory.find(i => i.slot === invPage * 8 + k) ?? { slot: -1, id: -1, count: 0, name: '' });
        const state = rand(6);  // 0, 1 plain; 2 bank; 3 picked; 4 dialog; 5 modal
        const held = page.filter(i => i.slot >= 0).slice(0, rand(9)).map(i => ({ slot: i.slot, id: i.id, name: i.name,
            count: i.count, option: state === 3 ? 'Use' : 'Eat' }));
        const bankPages = 1 + rand(5);
        cases.push({
            obs: { contractVersion: contract, actTargetVersion: 'act-v3', inventory, invSel: page, invPage, heldSel: held,
                actSel: Array.from({ length: rand(17) }, () => ({})),
                clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: rand(5) > 0, open: true })),
                style: { mode: rand(4), count: 1 + rand(4), stale: rand(5) === 0 },
                bankOpen: state === 2, bankPage: state === 2 ? rand(bankPages) : 0, bankPages: state === 2 ? bankPages : 1,
                bankSel: Array.from({ length: 8 }, () => state === 2 && rand(2) ? { slot: 0, id: 1, count: 1 + rand(5), name: 'Logs' }
                    : { slot: -1, id: -1, count: 0, name: '' }),
                using: state === 3 ? { slot: page.find(i => i.slot >= 0)?.slot ?? 0, id: 1, name: 'Logs' } : null,
                dialogOpen: state === 4, modalOpen: state === 5,
                dialogOptions: state === 4 && rand(2) ? Array.from({ length: 1 + rand(5) }, (_, k) => ({ text: `choice ${k}` })) : [] },
            allowedActions: rand(6) === 0 ? Array.from({ length: 96 }, (_, a) => a).filter(() => rand(3) > 0) : undefined,
        });
    }
    return cases;
}
