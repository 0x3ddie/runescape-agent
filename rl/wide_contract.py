"""The wide contract, of the real engine alone (free-play-v11):

- item-v4: item-v3's page of kinds (rl/item_contract.py kind_page) with 16 entries a page, so 2 pages of the 28 slots
  (env-ts/core/items.ts kindPage with N_PAGE_V4). Bank pages stay physical pages of 8.
- act-v4: 128 rows. Rows 0..95 keep act-v3's meanings; entries 8..15 of the four page families take rows 96..103 (pack
  options), 104..111 (drop), 112..119 (deposit) and 120..127 (pick to use) (env-ts/core/actions.ts pageRow, pageEntry).
- mask-v6 (env-ts/core/mask.ts): mask-v5's rules on 128 rows, the next page legal on page 0 alone, and no drop rule.
- obs-v7: obs-v6 with the page's and the pack options' inputs at 16 entries (sel_ids, sel_feats, held_ids, held_op_ids,
  held_feats) and the mask at 128 rows; the page index input stays invPage / 3.

The pinned modules (rl/obs.py, rl/act_contract.py, rl/item_contract.py) hold everything the contracts share; this one
holds only what differs and calls them for the rest."""
from __future__ import annotations

import numpy as np

from act_contract import A_HELD0, HELD_FEATS, N_HELD, project_acts
from item_contract import ITEM_V3, check_kind_page, kind_page, observation_contract
from names import identities
from obs import A_DEPOSIT0, A_DROP0, A_USE0, MASK, N_ACTIONS, N_INV_SEL, project_v6

ITEM_V4, ACT_V4, MASK_V6, OBSERVATION_V7 = 'item-v4', 'act-v4', 'mask-v6', 'obs-v7'
N_PAGE, N_ACTIONS_V4 = 16, 128
# Each page family's first row (act-v3's group, entries 0..7) and the first row of its second group (entries 8..15).
PAGE_FAMILIES = ((A_HELD0, 96), (A_DROP0, 104), (A_DEPOSIT0, 112), (A_USE0, 120))
# obs-v7's shapes where they differ from obs-v6's (sim/spec.py OBSERVATION_SHAPES_V6).
WIDE_SHAPES = {'sel_ids': (N_PAGE,), 'sel_feats': (N_PAGE, 1), 'held_ids': (N_PAGE,), 'held_op_ids': (N_PAGE,),
               'held_feats': (N_PAGE, HELD_FEATS), 'mask': (N_ACTIONS_V4,)}


def page_row(base, k):
    """act-v4's row of entry `k` (0..15) of the page family whose first row is `base`."""
    second = dict(PAGE_FAMILIES).get(base)
    if second is None or type(k) is not int or not 0 <= k < N_PAGE:
        raise ValueError(f'no act-v4 row for family {base!r} entry {k!r}')
    return base + k if k < N_INV_SEL else second + k - N_INV_SEL


def page_entry(action):
    """(first row of the family, entry) of an act-v4 row of a page family, or None for any other row."""
    for base, second in PAGE_FAMILIES:
        if base <= action < base + N_INV_SEL:
            return base, action - base
        if second <= action < second + N_INV_SEL:
            return base, action - second + N_INV_SEL
    return None


def item_v3_view(obs):
    """The item-v3 observation that item-v4 page `p` contains: item-v3's page 2p (the first 8 entries), no pack options,
    and mask-v5's 96 rows of the mask with the pack options' rows cleared. Its inputs are obs-v7's for everything but
    the page's and pack options' entries, the mask and the page index."""
    mask = obs.get('mask')
    if not isinstance(mask, list) or len(mask) != N_ACTIONS_V4:
        raise ValueError(f'an item-v4 observation carries a mask of {N_ACTIONS_V4} rows')
    page = obs.get('invSel')
    if not isinstance(page, list) or len(page) != N_PAGE:
        raise ValueError(f'item-v4 invSel requires {N_PAGE} entries')
    held_rows = range(A_HELD0, A_HELD0 + N_HELD)
    return dict(obs, contractVersion=ITEM_V3, invPage=2 * obs['invPage'], invSel=page[:N_INV_SEL], heldSel=[],
                maskVersion=MASK, mask=[0 if a in held_rows else m for a, m in enumerate(mask[:N_ACTIONS])])


def observation_contract_v4(obs):
    """item-v4's validator: the stamp, a page of 0 or 1, and its 16 entries the pack's kinds 16p..16p+15 (kind_page's
    pages 2p and 2p+1) with each kind's first slot, count and total, then empty entries; the bank as item-v3's."""
    if obs.get('contractVersion') != ITEM_V4:
        raise ValueError(f"not an item-v4 observation: {obs.get('contractVersion')!r}")
    page = obs.get('invPage')
    if type(page) is not int or not 0 <= page <= 1:
        raise ValueError(f'Invalid item-v4 invPage: {page!r}')
    view = item_v3_view(obs)
    observation_contract(view)
    check_kind_page(obs['invSel'], kind_page(obs.get('inventory'), 2 * page) + kind_page(obs.get('inventory'), 2 * page + 1))


def check_items(obs):
    """An observation's item pages checked under its own stamp: the pinned validator (rl/item_contract.py
    observation_contract) for item-v2 and item-v3, observation_contract_v4 for item-v4; returns the action rows its
    contract has (act-v3's 96, act-v4's 128), which its mask must hold."""
    if obs.get('contractVersion') == ITEM_V4:
        observation_contract_v4(obs)
        return N_ACTIONS_V4
    observation_contract(obs)
    return N_ACTIONS


def project_v7(obs):
    """obs-v7 (module docstring): project_v6 of item_v3_view, with the page's entries 8..15 added by project()'s own rule
    (the name's row, log1p of the kind's total over 5), the pack options' 16 entries by project_acts's own rule (each
    group of 8 checked against its own rows), the mask's 128 rows and the page index invPage / 3."""
    observation_contract_v4(obs)
    if obs.get('maskVersion') != MASK_V6:
        raise ValueError(f"the observation's mask follows {obs.get('maskVersion') or 'mask-v1'}, not {MASK_V6}")
    mask = np.asarray(obs['mask'], dtype=np.int8)
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('Invalid action mask')
    result = project_v6(item_v3_view(obs))
    ident = identities()
    sel_ids = np.zeros(N_PAGE, dtype=np.int32)
    sel_feats = np.zeros((N_PAGE, 1), dtype=np.float32)
    sel_ids[:N_INV_SEL], sel_feats[:N_INV_SEL] = result['sel_ids'], result['sel_feats']
    for i, item in enumerate(obs['invSel'][N_INV_SEL:], N_INV_SEL):
        if item['slot'] != -1:
            sel_ids[i] = ident['obj'](item.get('name', ''))
            sel_feats[i, 0] = float(np.log1p(item.get('total', 0)) / 5.0)
    held = obs.get('heldSel')
    if not isinstance(held, list) or len(held) > N_PAGE:
        raise ValueError('Invalid act slots')
    groups = []
    for g, (first, second) in enumerate(((A_HELD0, A_HELD0), (A_HELD0, PAGE_FAMILIES[0][1]))):
        rows = list(obs['mask'][:N_ACTIONS])
        rows[first:first + N_HELD] = obs['mask'][second:second + N_HELD]
        groups.append(project_acts(dict(obs, contractVersion=ITEM_V3, heldSel=held[g * N_HELD:(g + 1) * N_HELD],
                                        mask=rows), ident['place']))
    result.update({key: np.concatenate([group[key] for group in groups]) for key in ('held_ids', 'held_op_ids', 'held_feats')})
    result.update(sel_ids=sel_ids, sel_feats=sel_feats, mask=mask)
    result['item_vec'][0] = obs['invPage'] / 3
    return result


def project_free(obs):
    """A free-play observation's projection by its item contract's stamp: obs-v7 under item-v4, obs-v6 otherwise."""
    return project_v7(obs) if obs.get('contractVersion') == ITEM_V4 else project_v6(obs)
