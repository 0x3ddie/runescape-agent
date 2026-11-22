"""act-v3 (env-ts/core/affordances.ts, env-ts/core/clicks.ts): what a player could click, as slots the policy points at.

act-v3 (native-click) is act-v2 with its walks replaced by 24 minimap clicks at fixed offsets from the player:
radii 4, 9 and 16 tiles in the eight compass directions, north first and clockwise, diagonals (3, 3), (6, 6) and
(11, 11). Click slot k is action row CLICK_ROWS[k] (rows 17..28 and 84..95); NAV_PLACE, MOVE and RETREAT are gone, and
row 71 is never legal. act-v2 checkpoints are refused (baseline/policy.py check_names), never converted.

An affordance is one option of one nearby loc, npc or ground item; a held affordance is
one option of an item on the visible inventory page. Slots are ordered by walking
steps, then position, never by level or value. A click's features are its offset and whether
its target tile's live collision blocks no step onto it (`open`); no route, step count or visited share.
"""
import re
import zlib

import numpy as np

from item_contract import ITEM_CONTRACTS, ITEM_V3

ACT_V3 = 'act-v3'
A_ACT0, N_ACT = 1, 16
A_HELD0, N_HELD = 29, 8
ACT_SCAN_TILES = 15
KINDS = ('loc', 'npc', 'obj')
ACT_FEATS = 10
HELD_FEATS = 2
# The clicks: slot k at CLICK_OFFSETS[k] tiles (dx, dz; z grows north) on row CLICK_ROWS[k].
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
CLICK_RADII, CLICK_DIAGONALS = (4, 9, 16), (3, 6, 11)
CLICK_OFFSETS = tuple((ux * (d if ux and uz else r), uz * (d if ux and uz else r))
                      for r, d in zip(CLICK_RADII, CLICK_DIAGONALS) for ux, uz in DIRECTIONS)
N_CLICK = len(CLICK_OFFSETS)
A_CLICK_LOW, A_CLICK_HIGH, N_CLICK_LOW = 17, 84, 12  # slots 0..11 on rows 17..28, slots 12..23 on rows 84..95
CLICK_ROWS = tuple(range(A_CLICK_LOW, A_CLICK_LOW + N_CLICK_LOW)) + \
    tuple(range(A_CLICK_HIGH, A_CLICK_HIGH + N_CLICK - N_CLICK_LOW))
CLICK_FEATS = 3
CLICK_SCALE = 16
# A bank: a booth or banker that offers banking, in the 15-tile loc and npc lists.
BANK_NAME, BANK_OPTION = re.compile(r'bank|booth', re.I), re.compile(r'use-quickly|bank', re.I)
BANK_SCALE = 32
NO_BANK = 2.0  # the bank column when no bank is in sight


def click_slot(action):
    """The click slot of an action row, or None."""
    return action - A_CLICK_LOW if A_CLICK_LOW <= action < A_CLICK_LOW + N_CLICK_LOW else \
        action - A_CLICK_HIGH + N_CLICK_LOW if A_CLICK_HIGH <= action < A_CLICK_HIGH + N_CLICK - N_CLICK_LOW else None


def name_id(text):
    """A hashed identity for option text, which the name table does not hold."""
    return zlib.crc32(str(text).lower().encode()) % 4000 + 1


def acts_from_config(cfg):
    if cfg.get('actTargets') != ACT_V3 or cfg.get('contract') not in ITEM_CONTRACTS:
        raise ValueError('the task requires item-v2 or item-v3 and act-v3 affordances')
    return ACT_V3


def banks(obs):
    """(x, z) of the banks the observation lists: booths in its locs and bankers in its npcs (15 tiles)."""
    p = obs['player']
    return [(p['x'] + t['dx'], p['z'] + t['dz']) for t in [*obs.get('locs', []), *obs.get('npcs', [])]
            if BANK_NAME.search(t.get('name') or '') and any(BANK_OPTION.search(o) for o in t.get('options') or [])]


def bank_distance(target, near):
    """Chebyshev tiles from a target to the nearest listed bank, over BANK_SCALE; NO_BANK if none."""
    tiles = [max(abs(target['x'] - x), abs(target['z'] - z)) for x, z in near]
    return min(min(tiles) / BANK_SCALE, NO_BANK) if tiles else NO_BANK


def project_acts(obs, ident):
    """`ident(kind, name)` gives a thing's row in the name table (rl/names.py)."""
    if obs.get('actTargetVersion') != ACT_V3 or obs.get('contractVersion') not in ITEM_CONTRACTS:
        raise ValueError('Invalid act observation version')
    acts, held, mask = obs.get('actSel'), obs.get('heldSel'), obs.get('mask', [0] * 96)
    if not isinstance(acts, list) or len(acts) > N_ACT or not isinstance(held, list) or len(held) > N_HELD:
        raise ValueError('Invalid act slots')
    p = obs.get('player', {})
    if any(type(p.get(k)) is not int for k in ('x', 'z', 'level')):
        raise ValueError('Missing act observer position')
    combat = p.get('combatLevel') or 0
    ids = np.zeros(N_ACT, dtype=np.int32)
    ops = np.zeros(N_ACT, dtype=np.int32)
    feats = np.zeros((N_ACT, ACT_FEATS), dtype=np.float32)
    seen, ordering = set(), []
    for i, a in enumerate(acts):
        if (not isinstance(a, dict) or a.get('kind') not in KINDS or not isinstance(a.get('name'), str) or not a['name']
                or not isinstance(a.get('option'), str) or not a['option']
                or any(type(a.get(k)) is not int for k in ('id', 'x', 'z', 'level', 'dx', 'dz', 'distance', 'steps',
                                                            'count', 'combatLevel'))):
            raise ValueError('Invalid affordance')
        distance = max(abs(a['x'] - p['x']), abs(a['z'] - p['z']))
        key = (a['kind'], a['id'], a['x'], a['z'], a['level'], a['option'].lower())
        if (key in seen or a['id'] < 0 or a['level'] != p['level'] or distance > ACT_SCAN_TILES or a['distance'] != distance
                or a['dx'] != a['x'] - p['x'] or a['dz'] != a['z'] - p['z'] or a['steps'] < 0 or a['count'] < 1):
            raise ValueError('Inconsistent/duplicate affordance')
        seen.add(key)
        ordering.append((a['steps'], a['x'], a['z'], a['kind'], a['id'], a['option']))
        ids[i], ops[i] = ident(a['kind'], a['name']), name_id(a['option'])
        feats[i] = [a['dx'] / ACT_SCAN_TILES, a['dz'] / ACT_SCAN_TILES, distance / ACT_SCAN_TILES, min(a['steps'] / 32, 2.0),
                    min(a['count'], 10) / 10, a['kind'] == 'loc', a['kind'] == 'npc', a['kind'] == 'obj',
                    min(a['combatLevel'] / 40, 4.0), a['combatLevel'] > combat]
    if ordering != sorted(ordering) or any(mask[A_ACT0 + len(acts):A_ACT0 + N_ACT]):
        raise ValueError('Affordance slot order/action mapping mismatch')
    page = {item['slot']: item for item in obs.get('invSel', []) if item.get('slot', -1) >= 0}
    held_ids = np.zeros(N_HELD, dtype=np.int32)
    held_ops = np.zeros(N_HELD, dtype=np.int32)
    held_feats = np.zeros((N_HELD, HELD_FEATS), dtype=np.float32)
    for i, h in enumerate(held):
        item = page.get(h.get('slot')) if isinstance(h, dict) else None
        if (item is None or h.get('id') != item['id'] or h.get('count') != item['count'] or not isinstance(h.get('option'), str)
                or not h['option'] or h['option'].lower() == 'drop'):
            raise ValueError('Held affordance does not match the visible page')
        held_ids[i], held_ops[i] = ident('obj', item['name']), name_id(h['option'])
        # the count input: the slot's own count (item-v2), or the kind's total over the pack (item-v3)
        held_feats[i] = [np.log1p(item['total'] if obs['contractVersion'] == ITEM_V3 else item['count']) / 5.0, 1.0]
    if any(mask[A_HELD0 + len(held):A_HELD0 + N_HELD]):
        raise ValueError('Held slot/action mapping mismatch')
    return {'act_ids': ids, 'act_op_ids': ops, 'act_feats': feats,
            'held_ids': held_ids, 'held_op_ids': held_ops, 'held_feats': held_feats}


def project_clicks(obs):
    """One row per click slot: its offset over CLICK_SCALE and its `open` bit. A legal click targets a tile in the
    declared world; the offsets are act-v3's."""
    clicks, mask = obs.get('clicks'), obs.get('mask', [0] * 96)
    if not isinstance(clicks, list) or len(clicks) != N_CLICK:
        raise ValueError('Invalid click slots')
    feats = np.zeros((N_CLICK, CLICK_FEATS), dtype=np.float32)
    for k, c in enumerate(clicks):
        if (not isinstance(c, dict) or (c.get('dx'), c.get('dz')) != CLICK_OFFSETS[k] or type(c.get('open')) is not bool
                or type(c.get('inside')) is not bool or (mask[CLICK_ROWS[k]] and not c['inside'])):
            raise ValueError('Click slot/action mapping mismatch')
        feats[k] = [c['dx'] / CLICK_SCALE, c['dz'] / CLICK_SCALE, c['open']]
    return {'click_feats': feats}
