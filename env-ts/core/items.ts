/** The item-v2 contract. Physical-slot pages never rank items by task or value. */
import { A_BANK_NEXT, A_BANK_PREV, A_DEPOSIT0, A_INV_NEXT, A_INV_PREV, A_WITHDRAW0, N_INV_SEL } from './actions';

export const ITEM_CONTRACT = 'item-v2';
export const BANK_CAPACITY = 240;
export type Item = { slot: number; id: number; count: number; name: string; [key: string]: any };
const empty = (): Item => ({ slot: -1, id: -1, count: 0, name: '' });

export function checkItemContract(cfg: { contract?: string; allowedActions?: number[] }): void {
    if (cfg.contract !== ITEM_CONTRACT) throw new Error(`unsupported item contract: ${cfg.contract}`);
    if (cfg.allowedActions?.some(a => a >= A_DEPOSIT0 && a < A_DEPOSIT0 + N_INV_SEL) && !hasBankAccess(cfg.allowedActions))
        throw new Error('item-v2 deposits require every withdrawal and page action');
}

/** Deposits are reversible only when every withdrawal and page action is allowed too. */
export function hasBankAccess(allowed?: number[]): boolean {
    const needed = [...Array.from({ length: N_INV_SEL }, (_, k) => A_WITHDRAW0 + k), A_INV_PREV, A_INV_NEXT, A_BANK_PREV, A_BANK_NEXT];
    return !allowed || needed.every(a => allowed.includes(a));
}

export function itemPage(items: Item[], page: number, capacity: number): Item[] {
    if (!Number.isInteger(page) || page < 0 || page >= Math.ceil(capacity / N_INV_SEL))
        throw new Error('invalid item page');
    const slots = new Map<number, Item>();
    for (const item of items) {
        if (!Number.isInteger(item.slot) || item.slot < 0 || item.slot >= capacity ||
            !Number.isInteger(item.id) || item.id < 0 || !Number.isInteger(item.count) || item.count <= 0 || slots.has(item.slot))
            throw new Error('invalid or duplicate item slot');
        slots.set(item.slot, item);
    }
    return Array.from({ length: N_INV_SEL }, (_, k) => {
        const found = slots.get(page * N_INV_SEL + k);
        return found ? { ...found } : empty();
    });
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
    reset(): void { this.invPage = this.bankPage = 0; this.bankPages = 1; }
    turn(action: number): boolean {
        if (action === A_INV_PREV) this.invPage = Math.max(0, this.invPage - 1);
        else if (action === A_INV_NEXT) this.invPage = Math.min(3, this.invPage + 1);
        else if (action === A_BANK_PREV) this.bankPage = Math.max(0, this.bankPage - 1);
        else if (action === A_BANK_NEXT) this.bankPage = Math.min(this.bankPages - 1, this.bankPage + 1);
        else return false;
        return true;
    }
    project(obs: Record<string, any>, bankItems: Item[]): void {
        const visible = obs.bankOpen ? bankItems : [];
        // Capacity is a schema bound, while the page count reveals only visible occupied slots.
        if (visible.some(i => i.slot >= BANK_CAPACITY)) throw new Error('bank exceeds item-v2 capacity');
        this.bankPages = Math.max(1, Math.ceil((Math.max(-1, ...visible.map(i => i.slot)) + 1) / N_INV_SEL));
        this.bankPage = obs.bankOpen ? Math.min(this.bankPage, this.bankPages - 1) : 0;
        obs.contractVersion = ITEM_CONTRACT;
        obs.invPage = this.invPage;
        obs.bankPage = this.bankPage;
        obs.bankPages = this.bankPages;
        obs.invSel = itemPage(obs.inventory, this.invPage, 28);
        obs.bankSel = itemPage(visible, this.bankPage, BANK_CAPACITY);
    }
}
