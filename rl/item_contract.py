"""The item observation contracts: item-v2, physical inventory and bank pages padded to eight slots; item-v3, the page of
kinds (env-ts/core/items.ts kindPage), whose inventory page lists one entry for each distinct kind of item in the order of
the kinds' first slots, each the first slot's own item with the kind's `total` over the pack. Bank pages are physical in
both."""
from __future__ import annotations

ITEM_V2 = 'item-v2'
ITEM_V3 = 'item-v3'
ITEM_CONTRACTS = (ITEM_V2, ITEM_V3)
N_BANK_SEL = 8
BANK_CAPACITY = 240
MAX_BANK_PAGES = BANK_CAPACITY // N_BANK_SEL


def kind_page(inventory, page):
    """item-v3's inventory page `page`: (slot, id, count, total) of the pack's distinct kinds (item ids) in the order of
    their first slots, 8 a page; `inventory` the observation's items (slot, id, count), refused when malformed."""
    slots = {}
    for item in inventory if isinstance(inventory, list) else [None]:
        slot, ident, count = (item.get(x) if isinstance(item, dict) else None for x in ('slot', 'id', 'count'))
        if any(type(x) is not int for x in (slot, ident, count)) or not 0 <= slot < 28 or ident < 0 or count <= 0 \
                or slot in slots:
            raise ValueError('Invalid item-v3 inventory item')
        slots[slot] = (ident, count)
    kinds = {}
    for slot in sorted(slots):
        ident, count = slots[slot]
        first = kinds.setdefault(ident, [slot, ident, count, 0])
        first[3] += count
    return [tuple(k) for k in kinds.values()][page * 8:page * 8 + 8]


def observation_contract(obs):
    """Check an observation's item pages; raise ValueError on anything malformed."""
    version = obs.get('contractVersion')
    if version not in ITEM_CONTRACTS:
        raise ValueError(f"Unsupported item contract: {obs.get('contractVersion')!r}")
    def integer(key, lo, hi):
        value = obs.get(key)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'Invalid {version} {key}: {value!r}')
        return value
    inv_page = integer('invPage', 0, 3)
    bank_pages = integer('bankPages', 1, MAX_BANK_PAGES)
    bank_page = integer('bankPage', 0, bank_pages - 1)
    if type(obs.get('bankOpen')) is not bool:
        raise ValueError(f'{version} bankOpen must be boolean')
    for key, page, capacity in [('invSel', inv_page, 28), ('bankSel', bank_page, BANK_CAPACITY)]:
        items = obs.get(key)
        if not isinstance(items, list) or len(items) != 8:
            raise ValueError(f'{version} {key} requires eight padded slots')
        if key == 'invSel' and version == ITEM_V3:
            check_kind_page(items, kind_page(obs.get("inventory"), page))
            continue
        for k, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError('Invalid item slot')
            slot, ident, count = (item.get(x) for x in ('slot', 'id', 'count'))
            if any(type(x) is not int for x in (slot, ident, count)):
                raise ValueError('Item identity and count must be integers')
            if slot == -1:
                if ident != -1 or count != 0 or item.get('name') != '':
                    raise ValueError('Invalid empty item sentinel')
            elif slot != page * 8 + k or slot >= capacity or ident < 0 or count <= 0:
                raise ValueError('Item slot does not match physical page')
    if not obs['bankOpen'] and (bank_page != 0 or bank_pages != 1 or any(x['slot'] != -1 for x in obs['bankSel'])):
        raise ValueError('Closed bank must not disclose bank contents or pages')


def check_kind_page(items, want):
    """item-v3's inventory page `items` against `want` (kind_page of the observation's own inventory): its entries are
    those kinds, in that order, each with its slot, id, count and total, then empty entries (slot -1, id -1, count 0,
    total 0, no name)."""
    for k, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError('Invalid item slot')
        found = tuple(item.get(x) for x in ('slot', 'id', 'count', 'total'))
        if any(type(x) is not int for x in found):
            raise ValueError('item-v3 entry identity, count and total must be integers')
        if k < len(want):
            if found != want[k]:
                raise ValueError(f'item-v3 entry {k} {found} is not the kind of the pack {want[k]} (slot, id, count, total)')
        elif found != (-1, -1, 0, 0) or item.get('name') != '':
            raise ValueError('Invalid empty item-v3 entry')
