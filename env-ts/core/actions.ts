// env-ts/core/actions.ts: the action table (act-v3), shared by every environment.
//
// Mirrored by rl/obs.py: the policy and the environment must agree on the table
// exactly. A policy trained against one numbering and run against another fails
// silently and looks like an RL result.
//
// One row family per kind of thing a player can click, each scored by the policy one
// published slot at a time, so no row is tied to a skill:
//   0       WAIT              – let the running option continue, or stand still
//   1..16   ACT_k             – use actSel[k]: one option of a nearby loc, npc or ground item
//   17..28  CLICK_k           – a minimap click at click slot k's offset (core/clicks.ts), slots 0..11
//   29..36  HELD_k            – use heldSel[k]: one option of a held item (Wield, Eat, ...)
//   37..44  DROP_k            – drop invSel[k]
//   45..52  DEPOSIT_k         – bank invSel[k], with the bank open
//   53..60  WITHDRAW_k        – withdraw one of bankSel[k], with the bank open
//   61      CLOSE_IF          – close the open interface
//   62      DIALOG_CONTINUE
//   63..66  DIALOG_CHOICE_0..3
//   67..70  inventory previous/next page, bank previous/next page (item-v2)
//   71      RETREAT           – act-v2's retreat, kept as a row and never legal (act-v3 flees by clicks)
//   72..75  STYLE_0..3        – combat style
//   76..83  USE_k             – pick invSel[k] to use on something, as the client's "Use";
//                               then ACT_k uses it on a nearby loc or npc, HELD_k on another
//                               held item, and CLOSE_IF puts it back
//   84..95  CLICK_k           – a minimap click at click slot k's offset, slots 12..23
//
// The width stays 96 so a new row family fits without reshaping every head. act-v3 (native-click)
// replaced act-v2's NAV_PLACE (17..28), MOVE (84..91) and RETREAT with the clicks, so act-v2
// checkpoints are refused (baseline/policy.py check_names), never converted.

export const N_ACTIONS = 96;
export const A_WAIT = 0;
export const A_ACT0 = 1;
export const N_ACT = 16;
export const A_CLICK_LOW = 17;
export const N_CLICK_LOW = 12;
export const A_HELD0 = 29;
export const N_HELD = 8;
export const N_INV_SEL = 8; // inventory and bank page slots (item-v2)
export const A_DROP0 = 37;
export const A_DEPOSIT0 = 45;
export const A_WITHDRAW0 = 53;
// Opening a shop or bank collapses the mask to interface actions; without a way out
// the agent once sat in a shop screen for the rest of the episode.
export const A_CLOSE_IF = 61;
export const A_DIALOG_CONTINUE = 62;
export const A_DIALOG_CHOICE0 = 63;
export const N_DIALOG_CHOICE = 4;
export const A_INV_PREV = 67;
export const A_INV_NEXT = 68;
export const A_BANK_PREV = 69;
export const A_BANK_NEXT = 70;
export const A_RETIRED = 71;  // act-v2's RETREAT: a row kept, never legal
export const A_STYLE0 = 72;
export const A_USE0 = 76;
export const A_CLICK_HIGH = 84;
export const N_CLICK = 24;

// act-v4 (item-v4's page): rows 0..95 keep act-v3's meaning exactly, and 32 rows follow for entries 8..15 of the four
// families that point at a page entry, so each of those families lies in two groups of rows, as the clicks do (17..28
// and 84..95):
//   96..103   HELD_8..15     pack options of heldSel[8..15]
//   104..111  DROP_8..15     drop invSel[8..15]
//   112..119  DEPOSIT_8..15  bank invSel[8..15], with the bank open
//   120..127  USE_8..15      pick invSel[8..15] to use
// Withdrawals (the bank's page of 8), the 16 nearby targets and the 24 clicks do not change.
export const ACT_V4 = 'act-v4';
export const N_ACTIONS_V4 = 128;
export const N_PAGE_V4 = 16; // entries of item-v4's page and of its pack options
const PAGE_FAMILIES: ReadonlyArray<[number, number]> = [[A_HELD0, 96], [A_DROP0, 104], [A_DEPOSIT0, 112], [A_USE0, 120]];

/** act-v4's row of entry k (0..15) of the page family whose first act-v3 row is `base` (A_HELD0, A_DROP0, A_DEPOSIT0
 * or A_USE0): base + k for entries 0..7, the family's second group for entries 8..15. */
export function pageRow(base: number, k: number): number {
    const group = PAGE_FAMILIES.find(([first]) => first === base);
    if (!group || !Number.isInteger(k) || k < 0 || k >= N_PAGE_V4) throw new Error(`no act-v4 row for ${base} entry ${k}`);
    return k < N_INV_SEL ? base + k : group[1] + k - N_INV_SEL;
}

/** The page family (its first act-v3 row) and entry of an act-v4 row of one of the four page families, or null. */
export function pageEntry(action: number): { base: number; k: number } | null {
    for (const [base, second] of PAGE_FAMILIES) {
        if (action >= base && action < base + N_INV_SEL) return { base, k: action - base };
        if (action >= second && action < second + N_INV_SEL) return { base, k: N_INV_SEL + action - second };
    }
    return null;
}

// DROP becomes legal only under inventory pressure (see buildMask; mask-v5 and before).
export const DROP_MIN_STACKS = 24;

// Tools the agent needs to earn anything are never dropped: a dropped tool ends every
// income stream for the rest of the episode. Measured on v21f: a policy that lost its
// axe WAITed out 359 of 400 decisions. Deposits stay legal; a withdrawal undoes them.
export const TOOL_PROTECT_RE = /axe|pickaxe|fishing net|tinderbox|fishing rod|harpoon|lobster pot|hammer|knife|needle/i;

// Which option label an action would start. Re-selecting the SAME option while it is
// still running is a continuation (no restart); selecting a different one supersedes it.
export function optionLabelFor(action: number): string | null {
    if (action >= A_ACT0 && action < A_ACT0 + N_ACT) return `act_${action - A_ACT0}`;
    // 'nav' prefix: the option tracker clears a walk that stops making progress.
    if (action >= A_CLICK_LOW && action < A_CLICK_LOW + N_CLICK_LOW) return `nav_click_${action - A_CLICK_LOW}`;
    if (action >= A_CLICK_HIGH && action < A_CLICK_HIGH + N_CLICK - N_CLICK_LOW)
        return `nav_click_${action - A_CLICK_HIGH + N_CLICK_LOW}`;
    return null; // instantaneous actions supersede nothing
}
