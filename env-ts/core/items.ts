/**
 * The item contracts. item-v2 (the default): physical-slot pages, which never rank items by task or value. item-v3, the
 * page of kinds: the pack's page lists one entry for each distinct kind of item (its item id) in the order of the
 * kinds' first slots, 8 kinds a page; an entry is the kind's first slot (its slot, id, name, options and that slot's
 * own count, so the stale-selection check and the eat and food-drop facts keep their rule) with `total`, the kind's
 * count over the whole pack. item-v4: item-v3's page with 16 entries a page, so 28 slots are at most 2 pages (index 0
 * and 1); its rows are act-v4's (core/actions.ts pageRow). Bank pages are physical in all three.
 */
import { A_BANK_NEXT, A_BANK_PREV, A_DEPOSIT0, A_INV_NEXT, A_INV_PREV, A_WITHDRAW0, N_INV_SEL, N_PAGE_V4 } from './actions';

export const ITEM_CONTRACT = 'item-v2';
export const ITEM_KINDS = 'item-v3';
export const ITEM_WIDE = 'item-v4';
export const ITEM_CONTRACTS: readonly string[] = [ITEM_CONTRACT, ITEM_KINDS, ITEM_WIDE];
/** The entries of a contract's inventory page: 8, or item-v4's 16. */
export const pageSize = (contract: string): number => contract === ITEM_WIDE ? N_PAGE_V4 : N_INV_SEL;
export const BANK_CAPACITY = 240;
export type Item = { slot: number; id: number; count: number; name: string; total?: number; [key: string]: any };
const empty = (): Item => ({ slot: -1, id: -1, count: 0, name: '' });

export function checkItemContract(cfg: { contract?: string; allowedActions?: number[] }): void {
    if (!ITEM_CONTRACTS.includes(cfg.contract ?? '')) throw new Error(`unsupported item contract: ${cfg.contract}`);
    if (cfg.allowedActions?.some(a => a >= A_DEPOSIT0 && a < A_DEPOSIT0 + N_INV_SEL) && !hasBankAccess(cfg.allowedActions))
        throw new Error(`${cfg.contract} deposits require every withdrawal and page action`);
}

/** Deposits are reversible only when every withdrawal and page action is allowed too. */
export function hasBankAccess(allowed?: number[]): boolean {
    const needed = [...Array.from({ length: N_INV_SEL }, (_, k) => A_WITHDRAW0 + k), A_INV_PREV, A_INV_NEXT, A_BANK_PREV, A_BANK_NEXT];
    return !allowed || needed.every(a => allowed.includes(a));
}

/** The items of a container by slot, refused when a slot is invalid or held twice; a page index within capacity (pages
 * of `per` entries). */
function slotsOf(items: Item[], page: number, capacity: number, per = N_INV_SEL): Map<number, Item> {
    if (!Number.isInteger(page) || page < 0 || page >= Math.ceil(capacity / per))
        throw new Error('invalid item page');
    const slots = new Map<number, Item>();
    for (const item of items) {
        if (!Number.isInteger(item.slot) || item.slot < 0 || item.slot >= capacity ||
            !Number.isInteger(item.id) || item.id < 0 || !Number.isInteger(item.count) || item.count <= 0 || slots.has(item.slot))
            throw new Error('invalid or duplicate item slot');
        slots.set(item.slot, item);
    }
    return slots;
}

export function itemPage(items: Item[], page: number, capacity: number): Item[] {
    const slots = slotsOf(items, page, capacity);
    return Array.from({ length: N_INV_SEL }, (_, k) => {
        const found = slots.get(page * N_INV_SEL + k);
        return found ? { ...found } : empty();
    });
}

/** item-v3's page of kinds: page `page` of the pack's distinct kinds in the order of their first slots, each entry its
 * first slot's item with `total` (the kind's count over the pack); empty entries pad it to `per` entries (total 0): 8
 * (item-v3), or item-v4's 16. */
export function kindPage(items: Item[], page: number, capacity: number, per = N_INV_SEL): Item[] {
    const kinds = new Map<number, Item>();  // id -> its entry, in the order of the kinds' first slots
    for (const item of [...slotsOf(items, page, capacity, per).values()].sort((a, b) => a.slot - b.slot)) {
        const kind = kinds.get(item.id);
        if (kind) kind.total! += item.count;
        else kinds.set(item.id, { ...item, total: item.count });
    }
    const list = [...kinds.values()];
    return Array.from({ length: per }, (_, k) => list[page * per + k] ?? { ...empty(), total: 0 });
}

/**
 * The pick (an item picked to use) an observation keeps, from the pack's items. item-v2: the picked slot's item, while
 * that slot still holds the picked id. item-v3: the pick is of the kind, so while the kind is in the pack it moves to the
 * kind's first slot (the page entry of that kind, which selectHeld then skips: no item is used on its own kind); none
 * once the kind has left the pack. Neither keeps a pick while the bank is open.
 */
export function keptPick<P extends { slot: number; id: number }>(using: P | null, items: Item[], contract: string,
                                                                  bankOpen: boolean): P | null {
    if (!using || bankOpen) return null;
    if (contract === ITEM_KINDS || contract === ITEM_WIDE) {
        const first = items.filter(i => i.id === using.id).sort((a, b) => a.slot - b.slot)[0];
        return first ? { ...using, slot: first.slot } : null;
    }
    return items.find(i => i.slot === using.slot)?.id === using.id ? using : null;
}

/** Refuse stale selection instead of silently selecting another item. */
export function matchItem(current: Item | undefined, selected: Item | undefined): boolean {
    return !!current && !!selected && selected.count > 0 && current.slot === selected.slot &&
        current.id === selected.id && current.count === selected.count;
}

export class ItemPages {
    invPage = 0;
    bankPage = 0;
    bankPages = 1;
    contract: string = ITEM_CONTRACT;  // the reset cfg's, which every observation stamps (contractVersion)
    reset(contract: string = ITEM_CONTRACT): void {
        if (!ITEM_CONTRACTS.includes(contract)) throw new Error(`unsupported item contract: ${contract}`);
        this.contract = contract;
        this.invPage = this.bankPage = 0; this.bankPages = 1;
    }
    /** The last inventory page index: 3 (pages of 8), or item-v4's 1 (pages of 16). */
    get lastInvPage(): number { return Math.ceil(28 / pageSize(this.contract)) - 1; }
    turn(action: number): boolean {
        if (action === A_INV_PREV) this.invPage = Math.max(0, this.invPage - 1);
        else if (action === A_INV_NEXT) this.invPage = Math.min(this.lastInvPage, this.invPage + 1);
        else if (action === A_BANK_PREV) this.bankPage = Math.max(0, this.bankPage - 1);
        else if (action === A_BANK_NEXT) this.bankPage = Math.min(this.bankPages - 1, this.bankPage + 1);
        else return false;
        return true;
    }
    project(obs: Record<string, any>, bankItems: Item[]): void {
        const visible = obs.bankOpen ? bankItems : [];
        // Capacity is a schema bound, while the page count reveals only visible occupied slots.
        if (visible.some(i => i.slot >= BANK_CAPACITY)) throw new Error(`bank exceeds ${this.contract} capacity`);
        this.bankPages = Math.max(1, Math.ceil((Math.max(-1, ...visible.map(i => i.slot)) + 1) / N_INV_SEL));
        this.bankPage = obs.bankOpen ? Math.min(this.bankPage, this.bankPages - 1) : 0;
        obs.contractVersion = this.contract;
        obs.invPage = this.invPage;
        obs.bankPage = this.bankPage;
        obs.bankPages = this.bankPages;
        obs.invSel = this.contract === ITEM_WIDE ? kindPage(obs.inventory, this.invPage, 28, N_PAGE_V4)
            : (this.contract === ITEM_KINDS ? kindPage : itemPage)(obs.inventory, this.invPage, 28);
        obs.bankSel = itemPage(visible, this.bankPage, BANK_CAPACITY);
    }
}
