import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from item_contract import ITEM_V3, kind_page, observation_contract
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


# ---- item-v2 unchanged by item-v3: projections recorded before item-v3 ----------------------------------------------
V2_NAMES = [('Logs', 1511), ('Bronze axe', 1351), ('Bread', 2309), ('Shrimps', 315), ('Tinderbox', 590), ('Coins', 995),
            ('Raw chicken', 2138), ('Bronze sword', 1277)]


def item_v2_observations():
    """21 item-v2 observations: the reset fixture and 20 drawn from a fixed seed (every inventory page, open and closed
    banks with bank pages, stacks, Eat options on the page), as the engine shows them."""
    import random
    rng, out = random.Random(20261120), [dict(RESET)]
    empty = dict(slot=-1, id=-1, count=0, name='')
    for k in range(20):
        inv_page, opened = k % 4, k % 3 == 0
        bank_pages = rng.randint(1, 5) if opened else 1
        bank_page = rng.randint(0, bank_pages - 1) if opened else 0
        inventory = []
        for slot in range(28):
            if rng.random() < 0.6:
                name, ident = rng.choice(V2_NAMES)
                inventory.append(dict(slot=slot, id=ident, count=rng.choice([1, 1, 1, 2, 250]), name=name))
        by_slot = {i['slot']: i for i in inventory}
        inv_sel = [dict(by_slot[inv_page * 8 + j]) if inv_page * 8 + j in by_slot else dict(empty) for j in range(8)]
        bank_sel = [dict(empty) for _ in range(8)]
        if opened:
            for j in range(8):
                if rng.random() < 0.5:
                    name, ident = rng.choice(V2_NAMES)
                    bank_sel[j] = dict(slot=bank_page * 8 + j, id=ident, count=rng.randint(1, 500), name=name)
        held = [dict(slot=e['slot'], id=e['id'], count=e['count'], name=e['name'], option='Eat')
                for e in inv_sel if e['slot'] >= 0 and e['name'] in ('Bread', 'Shrimps')][:8]
        mask = [1 if 29 <= a < 37 and a - 29 < len(held) else 0 if 29 <= a < 37 else m for a, m in enumerate(RESET['mask'])]
        out.append(dict(RESET, invPage=inv_page, bankPage=bank_page, bankPages=bank_pages, bankOpen=opened,
                        inventory=inventory, invSel=inv_sel, bankSel=bank_sel, heldSel=held, mask=mask))
    return out


def projection_digests(observations):
    """sha256 (its first 16 hex digits) of each array's dtype, shape and bytes, for project() and project_v6() of each
    observation."""
    import hashlib
    from obs import project_v6
    out = []
    for o in observations:
        out.append({f'{name}/{key}': hashlib.sha256(f'{v.dtype}{v.shape}'.encode() + v.tobytes()).hexdigest()[:16]
                    for name, fn in (('v5', project), ('v6', project_v6)) for key, v in sorted(fn(o).items())})
    return out


def test_item_v2_projections_are_byte_for_byte_those_recorded_before_item_v3():
    """Every array of project() and project_v6() of 21 item-v2 observations has the bytes, dtype and shape recorded
    before item-v3 (fixtures/item_v2_projections.json: projection_digests of item_v2_observations, the synthetic name
    table of conftest.py)."""
    recorded = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'item_v2_projections.json').read_text())
    assert recorded['commit'] == '35d0c6c' and projection_digests(item_v2_observations()) == recorded['digests']



# ---- item-v3, the page of kinds -----------------------------------------------------------------------------------------
KIND_NAMES = {1511: 'Logs', 1351: 'Bronze axe', 2309: 'Bread', 315: 'Shrimps', 590: 'Tinderbox', 995: 'Coins',
              2138: 'Raw chicken', 1277: 'Bronze sword', 303: 'Small fishing net', 1265: 'Bronze pickaxe'}


def kinds_obs(pack, page=0):
    """An item-v3 observation of the reset fixture: `pack` {slot: (id, count)} and its page of kinds as the engine shows
    it, built here from its definition: each kind at its lowest slot with that slot's count and the kind's total over
    the pack, kinds ordered by that slot, 8 a page, then empty entries."""
    first, total = {}, {}
    for slot, (ident, count) in pack.items():
        first[ident] = min(first.get(ident, slot), slot)
        total[ident] = total.get(ident, 0) + count
    kinds = sorted(first, key=first.get)[page * 8:page * 8 + 8]
    entries = [dict(slot=first[i], id=i, count=pack[first[i]][1], name=KIND_NAMES[i], total=total[i]) for i in kinds]
    empty = dict(slot=-1, id=-1, count=0, name='', total=0)
    return dict(raw(page), contractVersion=ITEM_V3, invSel=entries + [dict(empty) for _ in range(8 - len(entries))],
                inventory=[dict(slot=s, id=i, count=c, name=KIND_NAMES[i]) for s, (i, c) in sorted(pack.items())])


packs = st.dictionaries(st.integers(0, 27), st.tuples(st.sampled_from(sorted(KIND_NAMES)), st.integers(1, 300)),
                        max_size=28)


@settings(max_examples=300, deadline=None)
@given(packs, st.integers(0, 3), st.data())
def test_item_v3_takes_the_page_of_kinds_of_any_pack_and_refuses_any_other_page(pack, page, data):
    """item-v3's validator (observation_contract) takes the page of kinds of any pack on any page, kind_page holds the
    definition's kinds, and a page with two kinds swapped, a total or a count off, a kind left out, a kind of another
    page or the item-v2 stamp over it is refused."""
    obs = kinds_obs(pack, page)
    observation_contract(obs)
    entries = [e for e in obs['invSel'] if e['slot'] >= 0]
    assert kind_page(obs['inventory'], page) == [(e['slot'], e['id'], e['count'], e['total']) for e in entries]
    mutations = [lambda o: o.update(contractVersion='item-v2')] if entries and any(
        e['slot'] != page * 8 + k for k, e in enumerate(entries)) else []
    if entries:
        k = data.draw(st.integers(0, len(entries) - 1))
        mutations += [lambda o: o['invSel'][k].update(total=o['invSel'][k]['total'] + 1),
                      lambda o: o['invSel'][k].update(count=o['invSel'][k]['count'] + 1),
                      lambda o: o['invSel'].__setitem__(k, dict(slot=-1, id=-1, count=0, name='', total=0)),
                      lambda o: o['invSel'][k].pop('total')]
    if len(entries) > 1:
        mutations.append(lambda o: o['invSel'].__setitem__(slice(0, 2), o['invSel'][1::-1]))
    others = kinds_obs(pack, (page + 1) % 4)['invSel']
    if others[0]['slot'] >= 0:  # a kind of another page in an empty entry, or in place of this page's first
        mutations.append(lambda o: o['invSel'].__setitem__(len(entries) % 8, others[0]))
    for mutation in mutations:
        bad = kinds_obs(pack, page)
        mutation(bad)
        with pytest.raises(ValueError):
            observation_contract(bad)


def test_item_v3_count_inputs_read_the_kinds_total_and_the_rest_stays_item_v2s(name_table):
    """Under item-v3 the page's count input (sel_feats) and a held row's (held_feats) read the kind's total over the
    pack; the 28 slots' own inputs, the page's names and every other input are those of the item-v2 view of the same
    pack and page whose slots hold the same kinds in the same places."""
    pack = {0: (1351, 1), 1: (1511, 1), 2: (2309, 1), 3: (1511, 1), 4: (1511, 1), 5: (2309, 1), 9: (1511, 1)}
    kinds = kinds_obs(pack)
    kinds['heldSel'] = [dict(slot=2, id=2309, count=1, name='Bread', option='Eat')]
    kinds['mask'] = [1 if a == 29 else m for a, m in enumerate(kinds['mask'])]
    observation_contract(kinds)
    out = project(kinds)
    assert [e['slot'] for e in kinds['invSel'][:4]] == [0, 1, 2, -1]
    assert out['sel_ids'][1] == name_table.row('obj', 'Logs') and np.count_nonzero(out['sel_ids']) == 2  # axe and logs
    np.testing.assert_array_equal(out['sel_feats'][:3, 0], np.float32([np.log1p(1), np.log1p(4), np.log1p(2)]) / np.float32(5))
    np.testing.assert_array_equal(out['held_feats'][0], np.float32([np.log1p(2) / 5, 1]))  # bread: 2 in the pack
    slots = dict(kinds, contractVersion='item-v2', invSel=[dict(e, slot=k) if e['slot'] >= 0 else e
                                                           for k, e in enumerate(kinds['invSel'])])
    for e in slots['invSel']:
        e.pop('total', None)
    slots['invSel'] = [dict(slot=k, id=e['id'], count=e['count'], name=e['name']) if e['slot'] >= 0 else
                       dict(slot=-1, id=-1, count=0, name='') for k, e in enumerate(slots['invSel'])]
    slots['heldSel'] = [dict(kinds['heldSel'][0], slot=2)]
    plain = project(slots)  # the same kinds at slots 0 to 2 under item-v2: the slots' own counts
    np.testing.assert_array_equal(plain['sel_feats'][:3, 0], np.float32([np.log1p(1)] * 3) / np.float32(5))
    for key in out:
        if key not in ('sel_feats', 'held_feats'):
            np.testing.assert_array_equal(out[key], plain[key], err_msg=key)
