import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from item_contract import observation_contract
from obs import project, N_ACTIONS

CAPTURE = Path(__file__).resolve().parent / "fixtures" / "reset.json"
if not CAPTURE.exists():
    pytest.skip("needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository", allow_module_level=True)
RESET = json.load(open(CAPTURE))


def raw(inv_page=0, bank_page=0, bank_pages=1, opened=False):
    """The reset with other item pages; no held options, since those come from the page."""
    empty = dict(slot=-1, id=-1, count=0, name='')
    mask = [0 if 29 <= a < 37 else m for a, m in enumerate(RESET['mask'])]
    return dict(RESET, invPage=inv_page, bankPage=bank_page, bankPages=bank_pages, bankOpen=opened, heldSel=[], mask=mask,
                invSel=[dict(empty) for _ in range(8)], bankSel=[dict(empty) for _ in range(8)])


def test_physical_pages_include_last_inventory_and_bank_slots(name_table):
    obs = raw(3, 29, 30, True)
    obs['invSel'][3] = dict(slot=27, id=1511, count=1, name='Logs')
    obs['bankSel'][7] = dict(slot=239, id=1351, count=5, name='Bronze axe')
    out = project(obs)
    assert out['sel_ids'][3] == name_table.row('obj', 'Logs') and np.count_nonzero(out['sel_ids']) == 1
    assert out['bank_ids'][7] == name_table.row('obj', 'Bronze axe') and np.count_nonzero(out['bank_ids']) == 1
    np.testing.assert_allclose(out['bank_feats'][7], [np.log1p(5)/5, 1])
    np.testing.assert_array_equal(out['item_vec'], [1, 1, 1, 1])
    assert out['mask'].shape == (96,) == (N_ACTIONS,)


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(invPage=4),
    lambda x: x.update(bankPages=31),
    lambda x: x.update(bankPage=1),
    lambda x: x.update(bankOpen=1),
    lambda x: x['invSel'].pop(),
    lambda x: x['invSel'][0].update(slot=8,id=1,count=1),
    lambda x: x['bankSel'][0].update(slot=0,id=1,count=1),
    lambda x: x['bankSel'][0].update(id=0),
    lambda x: x.update(bankPages=2),
    lambda x: x.update(contractVersion='legacy-v1'),
    lambda x: x.pop('contractVersion'),
])
def test_malformed_pages_closed_bank_leaks_and_other_contracts_fail(mutation):
    obs = raw()
    mutation(obs)
    with pytest.raises(ValueError):
        observation_contract(obs)


@st.composite
def open_pages(draw):
    """Any legal open-bank view: occupied inventory slots on the shown page, bank slots on theirs."""
    inv_page, bank_pages = draw(st.integers(0, 3)), draw(st.integers(1, 30))
    bank_page = draw(st.integers(0, bank_pages - 1))
    obs = raw(inv_page, bank_page, bank_pages, True)
    for key, page, capacity in (('invSel', inv_page, 28), ('bankSel', bank_page, 240)):
        for k in range(8):
            slot = page * 8 + k
            if slot < capacity and draw(st.booleans()):
                obs[key][k] = dict(slot=slot, id=draw(st.integers(0, 20000)), count=draw(st.integers(1, 10**6)),
                                   name=draw(st.text(max_size=12)))
    return obs


@settings(max_examples=300, deadline=None)
@given(open_pages())
def test_every_legal_page_view_projects_to_fixed_shapes(obs):
    out = project(obs)
    assert out['sel_ids'].shape == out['bank_ids'].shape == (8,) and out['mask'].shape == (N_ACTIONS,)
    assert set(np.unique(out['mask'])) <= {0, 1} and all(np.isfinite(v).all() for v in out.values())
    np.testing.assert_equal(project(obs), out)


@settings(max_examples=300, deadline=None)
@given(open_pages(), st.integers(0, 7), st.sampled_from(('invSel', 'bankSel')), st.integers(1, 30))
def test_an_item_off_its_page_is_rejected(obs, k, key, pages_away):
    page = obs['invPage' if key == 'invSel' else 'bankPage']
    obs[key][k] = dict(slot=(page + pages_away) * 8 + k, id=1, count=1, name='x')
    with pytest.raises(ValueError):
        observation_contract(obs)
