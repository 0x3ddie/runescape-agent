// env-ts/core/wide.ts: item-v4's pack options.
//
// core/affordances.ts is pinned by the simulator's archives (sim/build.py pins), so its selectHeld, which lists the pack
// options of a page up to act-v3's 8 rows, stays as it is. item-v4's page holds 16 entries and its pack options take up
// to 16 rows (act-v4: HELD_0..7 at 29..36, HELD_8..15 at 96..103). The selection is selectHeld's own, entry by entry:
// each entry's options in their order, the picked item's own entry skipped, Drop left to its rows.
import { selectHeld, type HeldAffordance } from './affordances';
import { N_PAGE_V4 } from './actions';
import type { Item } from './items';

/** item-v4's heldSel: selectHeld of each page entry in turn, up to 16 options; on a page whose options fit in 8 it equals
 * selectHeld of the whole page. */
export function selectHeldWide(page: Item[], using?: { slot: number } | null): HeldAffordance[] {
    return page.flatMap(entry => selectHeld([entry], using)).slice(0, N_PAGE_V4);
}
