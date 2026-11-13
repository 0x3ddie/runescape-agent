"""act-v3's clicks and bank column (rl/act_contract.py), converted from place-v1's tests (rl/place_contract.py, removed
with act-v3): what reaches the policy of a click, the observations it refuses, and the affordances' distance to a bank
in the 15-tile loc and npc lists; and no source still reads a name act-v3 retired."""
from pathlib import Path
import re

import numpy as np
import pytest

from act_contract import (A_CLICK_HIGH, A_CLICK_LOW, BANK_SCALE, CLICK_OFFSETS, CLICK_ROWS, NO_BANK, bank_distance,
                          banks, project_clicks)


def raw():
    mask = [0] * 96
    mask[0] = 1
    for row in CLICK_ROWS[:20]:
        mask[row] = 1
    clicks = [dict(dx=dx, dz=dz, inside=k < 20, open=k % 3 != 0) for k, (dx, dz) in enumerate(CLICK_OFFSETS)]
    return dict(mask=mask, clicks=clicks, player=dict(x=100, z=100, level=0),
                locs=[dict(name='Bank booth', options=['Use-quickly', 'Use'], dx=-10, dz=0)],
                npcs=[dict(name='Banker', options=['Talk-to', 'Bank'], dx=0, dz=12), dict(name='Man', options=['Talk-to'], dx=1, dz=1)])


def test_a_click_reaches_the_policy_as_its_offset_and_open_bit():
    feats = project_clicks(raw())['click_feats']
    assert feats.shape == (24, 3)
    np.testing.assert_allclose(feats[2], [4 / 16, 0, 1])           # 4 east, open
    np.testing.assert_allclose(feats[3], [3 / 16, -3 / 16, 0])     # the radius-4 diagonal, blocked
    np.testing.assert_allclose(feats[16], [0, 1, 1])               # 16 north
    assert (CLICK_ROWS[0], CLICK_ROWS[11], CLICK_ROWS[12], CLICK_ROWS[23]) == (A_CLICK_LOW, 28, A_CLICK_HIGH, 95)


@pytest.mark.parametrize('change', [
    lambda o: o['clicks'].pop(),                                      # not the 24 slots
    lambda o: o['clicks'][4].update(dx=1),                            # another table than act-v3's offsets
    lambda o: o['clicks'][0].update(open=1),                          # open not a bool
    lambda o: o['mask'].__setitem__(CLICK_ROWS[22], 1),               # a click offered beyond the declared world
])
def test_inconsistent_clicks_rejected(change):
    obs = raw()
    change(obs)
    with pytest.raises(ValueError):
        project_clicks(obs)


def test_the_bank_column_reads_the_15_tile_lists_and_says_no_bank_without_one():
    obs = raw()
    assert banks(obs) == [(90, 100), (100, 112)]
    assert bank_distance(dict(x=104, z=100), banks(obs)) == 12 / BANK_SCALE   # the banker 12 north is nearer than 14 west
    obs['locs'][0]['options'], obs['npcs'][0]['options'] = ['Look-at'], ['Talk-to']
    assert banks(obs) == [] and bank_distance(dict(x=104, z=100), banks(obs)) == NO_BANK


# The act-v2 names act-v3 retired: no source reads them any more.
RETIRED = ('A_NAV_PLACE0', 'N_PLACES', 'A_MOVE0', 'N_MOVE', 'A_RETREAT', 'RETREAT_GAIN', 'PLACE_SCAN_TILES', 'MOVE_STEPS')


def test_no_source_reads_a_name_act_v3_retired():
    root = Path(__file__).resolve().parents[2]
    sources = [*root.glob('rl/*.py'), *root.glob('baseline/*.py'), *root.glob('sim/*.py'), *root.glob('scripts/*.py'),
               *root.glob('sim/native/*.[ch]'), *root.glob('env-ts/core/*.ts'), *root.glob('env-ts/engine/*.ts'),
               *root.glob('*/tests/*.py')]
    word = re.compile(r'\b(' + '|'.join(RETIRED) + r')\b')
    found = [f'{path.relative_to(root)}:{n}' for path in sources if path.resolve() != Path(__file__).resolve()
             for n, line in enumerate(path.read_text().splitlines(), 1) if word.search(line)]
    assert not found, found
