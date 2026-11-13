"""The item-v2 observation contract: physical inventory and bank pages, padded to eight slots."""
from __future__ import annotations

ITEM_V2 = 'item-v2'
N_BANK_SEL = 8
BANK_CAPACITY = 240
MAX_BANK_PAGES = BANK_CAPACITY // N_BANK_SEL


def observation_contract(obs):
    """Check an observation's item pages; raise ValueError on anything malformed."""
    if obs.get('contractVersion') != ITEM_V2:
        raise ValueError(f"Unsupported item contract: {obs.get('contractVersion')!r}")
    def integer(key, lo, hi):
        value = obs.get(key)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'Invalid item-v2 {key}: {value!r}')
        return value
    inv_page = integer('invPage', 0, 3)
    bank_pages = integer('bankPages', 1, MAX_BANK_PAGES)
    bank_page = integer('bankPage', 0, bank_pages - 1)
    if type(obs.get('bankOpen')) is not bool:
        raise ValueError('item-v2 bankOpen must be boolean')
    for key, page, capacity in [('invSel', inv_page, 28), ('bankSel', bank_page, BANK_CAPACITY)]:
        items = obs.get(key)
        if not isinstance(items, list) or len(items) != 8:
            raise ValueError(f'item-v2 {key} requires eight padded slots')
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
