// env-ts/core/mask.ts: legal-action mask, shared by every environment.
//
// PURE. No globals, no SDK, no engine. Everything it needs arrives in `obs` and
// `ctx`. The mask says what a player could do now; it never says what is wise, so
// there is no rule here for any skill. The few judgements left were each bought with
// a bug: an open interface always has a way out, a tool is never dropped, and DROP
// waits for inventory pressure (ungated, an exploring policy dropped its axe within
// a few decisions of nearly every episode).

import { ITEM_CONTRACTS, ITEM_WIDE, hasBankAccess } from './items';
import { ACT_CONTRACT } from './affordances';
import { CLICK_ROWS } from './clicks';
import {
    A_ACT0, A_BANK_NEXT, A_BANK_PREV, A_CLOSE_IF, A_DEPOSIT0, A_DIALOG_CHOICE0, A_DIALOG_CONTINUE, A_DROP0, A_HELD0,
    A_INV_NEXT, A_INV_PREV, A_STYLE0, A_USE0, A_WITHDRAW0, DROP_MIN_STACKS, N_ACT,
    N_ACTIONS, N_ACTIONS_V4, N_CLICK, N_DIALOG_CHOICE, N_HELD, N_PAGE_V4, TOOL_PROTECT_RE, pageRow
} from './actions';

// mask-v2: a combat-style row is legal only where selecting it changes something. Under
// mask-v1 all four were always legal, and greedy decoding repeated a style request that changed
// nothing. mask-v3 (native-click): a click is legal where its target lies in the declared
// world (obs.clicks[k].inside; natively always) and no bank, use or dialog mode is open, with no
// reachability judgement; RETREAT (71) is never legal. mask-v5: INV_PREV and INV_NEXT
// stay legal while an item is selected for use; the selection stays and HELD targets follow the new page (the native
// client shows all 28 slots). Each observation names its rule (obs.maskVersion, engine/agent.ts; rl/obs.py MASK).
export const MASK_CONTRACT = 'mask-v5';
// mask-v6 (item-v4 and act-v4): mask-v5's rules on 128 rows, entries 8..15 of the page families at their second row
// group (pageRow), the next-page row legal only on page 0 (2 pages of 16), and no drop rule: a DROP row is legal
// whenever its entry holds an item in the plain state (no rule of DROP_MIN_STACKS and no TOOL_PROTECT_RE). mask-v5 is
// unchanged.
export const MASK_WIDE = 'mask-v6';

export interface MaskContext {
    /** cfg allowlist; WAIT is always restored after it is applied */
    allowedActions?: number[];
}

/** obs.style (engine/agent.ts): the com_mode varp, the wielded weapon's style count, and whether
 * this reset wrote levels that no style request has recomputed the combat stats from yet. */
function validStyle(style: any): boolean {
    return !!style && Number.isInteger(style.mode) && style.mode >= 0 && Number.isInteger(style.count)
        && style.count >= 1 && typeof style.stale === 'boolean';
}

export function buildMask(obs: Record<string, any>, ctx: MaskContext): number[] {
    if (!ITEM_CONTRACTS.includes(obs.contractVersion)) throw new Error('invalid item observation contract');
    const wide = obs.contractVersion === ITEM_WIDE;  // mask-v6 on 128 rows; mask-v5 otherwise
    const row = (base: number, k: number) => wide ? pageRow(base, k) : base + k;
    const lastPage = wide ? 1 : 3;
    if (!validStyle(obs.style)) throw new Error('invalid style observation');
    if (obs.actTargetVersion !== ACT_CONTRACT || !Array.isArray(obs.actSel) || obs.actSel.length > N_ACT
        || !Array.isArray(obs.heldSel) || obs.heldSel.length > (wide ? N_PAGE_V4 : N_HELD))
        throw new Error('invalid act observation contract');
    if (!Array.isArray(obs.clicks) || obs.clicks.length !== N_CLICK) throw new Error('invalid click slots');
    const mask = new Array(wide ? N_ACTIONS_V4 : N_ACTIONS).fill(0);
    const finish = () => applyAllowList(mask, ctx.allowedActions);
    mask[0] = 1; // WAIT always

    if (obs.bankOpen) {
        mask[A_CLOSE_IF] = 1;
        if (obs.invPage > 0) mask[A_INV_PREV] = 1;
        if (obs.invPage < lastPage) mask[A_INV_NEXT] = 1;
        if (obs.bankPage > 0) mask[A_BANK_PREV] = 1;
        if (obs.bankPage + 1 < obs.bankPages) mask[A_BANK_NEXT] = 1;
        (obs.invSel ?? []).forEach((it: any, k: number) => {
            if (it.count > 0 && hasBankAccess(ctx.allowedActions)) mask[row(A_DEPOSIT0, k)] = 1;
        });
        (obs.bankSel ?? []).forEach((it: any, k: number) => { if (it.count > 0) mask[A_WITHDRAW0 + k] = 1; });
        // Full-capacity requests can be refused by the game without moving items.
        return finish();
    }
    if (obs.using) {
        // An item picked for use: use it on a published thing or held item, turn the page to another held item
        // (mask-v5), or put it back.
        if (obs.invPage > 0) mask[A_INV_PREV] = 1;
        if (obs.invPage < lastPage) mask[A_INV_NEXT] = 1;
        obs.actSel.forEach((_: any, k: number) => mask[A_ACT0 + k] = 1);
        obs.heldSel.forEach((_: any, k: number) => mask[row(A_HELD0, k)] = 1);
        mask[A_CLOSE_IF] = 1;
        return finish();
    }
    if (obs.dialogOpen || obs.modalOpen) {
        // Offer only what can move THIS dialog: a mesbox clears on continue, a choice
        // dialog only on one of its buttons. Listing both everywhere made most of the
        // legal set no-ops, and a policy sat in a guard's conversation indefinitely.
        const real = (obs.dialogOptions ?? []).filter((o: any) => !/click here to continue/i.test(o.text ?? ''));
        if (real.length) {
            for (let k = 0; k < Math.min(real.length, N_DIALOG_CHOICE); k++) mask[A_DIALOG_CHOICE0 + k] = 1;
        } else if (obs.dialogOpen) {
            mask[A_DIALOG_CONTINUE] = 1;
        }
        mask[A_CLOSE_IF] = 1; // always a way out of an interface
        return finish();
    }

    if (obs.invPage > 0) mask[A_INV_PREV] = 1;
    if (obs.invPage < lastPage) mask[A_INV_NEXT] = 1;
    obs.actSel.forEach((_: any, k: number) => mask[A_ACT0 + k] = 1);
    // A click into the declared world, routed or not: the game decides what it reaches.
    obs.clicks.forEach((c: any, k: number) => { if (c.inside) mask[CLICK_ROWS[k]] = 1; });
    obs.heldSel.forEach((_: any, k: number) => mask[row(A_HELD0, k)] = 1);
    const pressure = obs.inventory.length >= DROP_MIN_STACKS;
    (obs.invSel ?? []).forEach((it: any, k: number) => {
        // mask-v5's two drop rules (24 stacks, protected tools); mask-v6 has neither
        if (it.count > 0 && (wide || (pressure && !TOOL_PROTECT_RE.test(String(it.name ?? ''))))) mask[row(A_DROP0, k)] = 1;
        if (it.count > 0) mask[row(A_USE0, k)] = 1;
    });
    // Combat style row k selects style min(k, count - 1): [proc,player_combat_stat] clamps com_mode
    // to the weapon's styles. A row that selects the current style is no request, except while the
    // reset's levels wait for the recompute a style request runs (the declared start's STYLE_0 step).
    const { mode, count, stale } = obs.style;
    const current = Math.min(mode, count - 1);
    for (let k = 0; k < 4; k++) if (stale || Math.min(k, count - 1) !== current) mask[A_STYLE0 + k] = 1;
    return finish();
}

export function applyAllowList(mask: number[], allowedActions?: number[]): number[] {
    if (allowedActions) {
        const allow = new Set(allowedActions);
        for (let i = 0; i < mask.length; i++) if (!allow.has(i)) mask[i] = 0;
        mask[0] = 1; // WAIT is always legal
    }
    return mask;
}
