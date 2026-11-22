"""The wide contract's Python side (rl/wide_contract.py): its row map and versions are the engine's, item-v4's validator
takes the page of 16 kinds of any pack and refuses any other page, obs-v7 reads entries 8..15 and their pack options on
their own rows, and on a pack of 8 kinds or fewer obs-v7 holds obs-v6's inputs of the same pack's item-v3 page."""
import json
from pathlib import Path
import re

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from item_contract import ITEM_V3
from obs import project_v6
from wide_contract import (ITEM_V4, MASK_V6, N_ACTIONS_V4, N_PAGE, PAGE_FAMILIES, observation_contract_v4, page_entry,
                           page_row, project_free, project_v7)

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'rl/tests/fixtures/reset.json'
if not CAPTURE.exists():
    pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository',
                allow_module_level=True)
RESET = json.loads(CAPTURE.read_text())
EMPTY = dict(slot=-1, id=-1, count=0, name='', total=0)
NAMES = {1511: 'Logs', 1351: 'Bronze axe', 2309: 'Bread', 315: 'Shrimps', 590: 'Tinderbox', 995: 'Coins',
         2138: 'Raw chicken', 1277: 'Bronze sword', 303: 'Small fishing net', 1265: 'Bronze pickaxe', 1931: 'Pot',
         1925: 'Bucket', 1205: 'Bronze dagger', 1171: 'Wooden shield', 841: 'Shortbow', 882: 'Bronze arrow',
         556: 'Air rune', 558: 'Mind rune', 555: 'Water rune', 557: 'Earth rune', 559: 'Body rune'}


def kinds(pack, page, per):
    """The page of kinds as the engine shows it, from its definition: each kind at its lowest slot with that slot's count
    and the kind's total over the pack, kinds ordered by that slot, `per` a page, then empty entries."""
    first, total = {}, {}
    for slot, (ident, count) in pack.items():
        first[ident] = min(first.get(ident, slot), slot)
        total[ident] = total.get(ident, 0) + count
    shown = sorted(first, key=first.get)[page * per:page * per + per]
    entries = [dict(slot=first[i], id=i, count=pack[first[i]][1], name=NAMES[i], total=total[i]) for i in shown]
    return entries + [dict(EMPTY) for _ in range(per - len(entries))]


def wide_obs(pack, page=0, held=()):
    """An item-v4, mask-v6 observation of the recorded reset with `pack` ({slot: (id, count)}) and page `page`; `held`
    the entries whose first option is offered (pack options in entry order, as core/wide.ts selectHeldWide gives them),
    each legal on its act-v4 row."""
    page_ = kinds(pack, page, N_PAGE)
    options = [dict(slot=e['slot'], id=e['id'], count=e['count'], name=e['name'], option='Eat') for k, e in enumerate(page_)
               if k in held and e['slot'] >= 0]
    mask = [0 if 29 <= a < 37 else m for a, m in enumerate(RESET['mask'])] + [0] * (N_ACTIONS_V4 - 96)
    for i in range(len(options)):
        mask[page_row(29, i)] = 1
    return dict(RESET, contractVersion=ITEM_V4, maskVersion=MASK_V6, invPage=page, invSel=page_, heldSel=options, mask=mask,
                inventory=[dict(slot=s, id=i, count=c, name=NAMES[i]) for s, (i, c) in sorted(pack.items())])


def narrow_obs(wide):
    """The item-v3, mask-v5 observation of the same pack and page (a pack of 8 kinds or fewer, page 0 or 1): the page's
    first 8 entries, the same pack options on act-v3's rows and the mask's 96 rows."""
    return dict(wide, contractVersion=ITEM_V3, maskVersion='mask-v5', invSel=wide['invSel'][:8], mask=wide['mask'][:96])


def test_the_row_map_and_versions_are_the_engines():
    """rl/wide_contract.py's row map, sizes and names are env-ts/core/actions.ts's, items.ts's and mask.ts's, and the
    map is one to one: the four families' 16 entries each on rows 29..36, 37..44, 45..52, 76..83 and 96..127, every
    other row none."""
    source = (ROOT / 'env-ts/core/actions.ts').read_text()
    families = re.search(r'const PAGE_FAMILIES[^=]*= \[(.*)\];', source).group(1)
    assert re.findall(r'\[(A_\w+), (\d+)\]', families) == [('A_HELD0', '96'), ('A_DROP0', '104'), ('A_DEPOSIT0', '112'),
                                                           ('A_USE0', '120')]
    assert [second for _, second in PAGE_FAMILIES] == [96, 104, 112, 120] and [b for b, _ in PAGE_FAMILIES] == [29, 37, 45, 76]
    assert f"export const N_ACTIONS_V4 = {N_ACTIONS_V4};" in source and f"export const N_PAGE_V4 = {N_PAGE};" in source
    assert "export const ITEM_WIDE = 'item-v4';" in (ROOT / 'env-ts/core/items.ts').read_text()
    assert "export const MASK_WIDE = 'mask-v6';" in (ROOT / 'env-ts/core/mask.ts').read_text()
    rows = {page_row(base, k): (base, k) for base, _ in PAGE_FAMILIES for k in range(N_PAGE)}
    assert len(rows) == 64 and all(page_entry(row) == entry for row, entry in rows.items())
    assert sorted(rows) == [*range(29, 53), *range(76, 84), *range(96, 128)]
    assert all(page_entry(a) is None for a in range(N_ACTIONS_V4) if a not in rows)
    for base, k in ((29, 16), (29, -1), (0, 0), (53, 1)):
        with pytest.raises(ValueError):
            page_row(base, k)


packs = st.dictionaries(st.integers(0, 27), st.tuples(st.sampled_from(sorted(NAMES)), st.integers(1, 300)), max_size=28)


@settings(max_examples=300, deadline=None)
@given(packs, st.integers(0, 1), st.data())
def test_item_v4_takes_the_page_of_16_kinds_of_any_pack_and_refuses_any_other_page(pack, page, data):
    """item-v4's validator takes the page of 16 kinds of any pack on either page, and refuses a page with two kinds
    swapped, a total or a count off, a kind left out, a kind of the other page, 8 entries, page 2, or another stamp."""
    obs = wide_obs(pack, page)
    observation_contract_v4(obs)
    entries = [e for e in obs['invSel'] if e['slot'] >= 0]
    mutations = [lambda o: o.update(contractVersion=ITEM_V3), lambda o: o.update(invPage=2),
                 lambda o: o.update(invSel=o['invSel'][:8])]
    if entries:
        k = data.draw(st.integers(0, len(entries) - 1))
        mutations += [lambda o: o['invSel'][k].update(total=o['invSel'][k]['total'] + 1),
                      lambda o: o['invSel'][k].update(count=o['invSel'][k]['count'] + 1),
                      lambda o: o['invSel'].__setitem__(k, dict(EMPTY))]
    if len(entries) > 1:
        mutations.append(lambda o: o['invSel'].__setitem__(slice(0, 2), o['invSel'][1::-1]))
    other = kinds(pack, 1 - page, N_PAGE)[0]
    if other['slot'] >= 0:
        mutations.append(lambda o: o['invSel'].__setitem__(len(entries) % N_PAGE, other))
    for mutation in mutations:
        bad = wide_obs(pack, page)
        mutation(bad)
        with pytest.raises(ValueError):
            observation_contract_v4(bad)


def test_obs_v7_reads_the_recorded_resets_18_kinds_on_two_pages_and_the_pack_options_on_their_own_rows(name_table):
    """The recorded reset's pack holds 18 kinds: page 0 shows 16, page 1 the last 2. obs-v7 gives entries 8..15 their
    names and totals, the pack options of entries 8 and 9 (rows 96 and 97) their own inputs, the page
    input invPage / 3, the 128-row mask; a pack option legal on a row past the options offered is refused, as on act-v3's
    rows."""
    pack = {i['slot']: (i['id'], i['count']) for i in RESET['inventory']}
    obs = wide_obs(pack, 0, held=range(10))
    out = project_v7(obs)
    assert [e['name'] for e in obs['invSel'][8:]] == [i['name'] for i in RESET['inventory'][8:16]]
    assert out['sel_ids'].shape == (16,) and out['held_ids'].shape == (16,) and out['mask'].shape == (128,)
    assert out['sel_ids'][12] == name_table.row('obj', 'Bronze arrow') and out['sel_feats'][12, 0] == np.float32(np.log1p(25) / 5)
    assert out['held_ids'][9] == name_table.row('obj', 'Bronze sword') and out['mask'][97] == 1 and out['mask'][98] == 0
    np.testing.assert_array_equal(out['held_feats'][:10], np.float32([[np.log1p(1) / 5, 1]] * 10))
    assert not out['held_ids'][10:].any() and out['item_vec'][0] == 0
    last = wide_obs(pack, 1)
    second = project_v7(last)
    assert [e['name'] for e in last['invSel'] if e['slot'] >= 0] == ['Earth rune', 'Body rune']
    assert np.count_nonzero(second['sel_feats']) == 2 and second['item_vec'][0] == np.float32(1 / 3)
    bad = wide_obs(pack, 0, held=range(10))
    bad['mask'][98] = 1  # an eleventh option that is not offered
    with pytest.raises(ValueError, match='Held slot'):
        project_v7(bad)
    for change in (dict(maskVersion='mask-v5'), dict(mask=obs['mask'][:96]), dict(contractVersion=ITEM_V3)):
        with pytest.raises(ValueError):
            project_v7(dict(obs, **change))
    assert all(np.array_equal(project_free(obs)[k], out[k]) for k in out)


@settings(max_examples=60, deadline=None)
@given(st.dictionaries(st.integers(0, 27), st.tuples(st.sampled_from(sorted(NAMES)[:8]), st.integers(1, 300)), max_size=28),
       st.integers(0, 1), st.sets(st.integers(0, 7)))
def test_obs_v7_of_8_kinds_or_fewer_holds_obs_v6_of_the_same_item_v3_page(pack, page, held):
    """On a pack of 8 kinds or fewer, obs-v7 of the item-v4 page holds obs-v6 of the same pack's item-v3 page and pack
    options: every input equal, the page's and pack options' first 8 entries equal and the rest zero, and the mask's
    first 96 rows equal (the conversion's premise, baseline/tests/test_integration.py)."""
    wide = wide_obs(pack, page, held=held)
    out, narrow = project_v7(wide), project_v6(narrow_obs(wide))
    assert out.keys() == narrow.keys()
    for key, value in out.items():
        if key in ('sel_ids', 'sel_feats', 'held_ids', 'held_op_ids', 'held_feats', 'mask'):
            np.testing.assert_array_equal(value[:len(narrow[key])], narrow[key], err_msg=key)
            assert not value[len(narrow[key]):].any(), key
        else:
            np.testing.assert_array_equal(value, narrow[key], err_msg=key)


def free_play_policies():
    """A free-play policy of obs-v6, item-v3 and act-v3's 96 rows under the learner's own settings (config.LEARNER's,
    smaller, with PopArt), its weights moved off their initialization as training would leave them, and the obs-v7,
    item-v4, act-v4 policy that widen_state makes of it."""
    from copy import deepcopy
    import torch
    from gymnasium import spaces
    from baseline import config
    from baseline.adapter import observation_space
    from baseline.policy import POPART, MaskedRecurrentPolicy, run_contract, widen_state
    from names import name_table
    from sim.spec import OBSERVATION_SHAPES_V6, dtype
    from wide_contract import OBSERVATION_V7, WIDE_SHAPES
    kwargs = dict(deepcopy(config.LEARNER['policy_kwargs']), lstm_hidden_size=16, features_extractor_kwargs=dict(features_dim=16),
                  net_arch=dict(pi=[16], vf=[16]), names=name_table().spec, value_norm=POPART)
    built = []
    for observation, items, shapes in (('obs-v6', ITEM_V3, OBSERVATION_SHAPES_V6),
                                       (OBSERVATION_V7, ITEM_V4, {**OBSERVATION_SHAPES_V6, **WIDE_SHAPES})):
        space = observation_space({k: np.zeros(v, dtype(k)) for k, v in shapes.items()})
        built.append(MaskedRecurrentPolicy(space, spaces.Discrete(len(space['mask'].low)), lambda _: 3e-4,
                                           observation=observation, contract=run_contract(items), **kwargs))
    source, wide = built
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(parameter.numel())) * .1)
    wide.load_state_dict(widen_state(source.state_dict(), wide), strict=True)
    return source, wide


def test_a_converted_policy_scores_rows_0_to_95_values_and_chooses_as_its_source_on_8_kinds_or_fewer():
    """The tested crossing (baseline/policy.py widen_state). Over a recurrent sequence of the recorded reset with packs
    of 8 kinds or fewer on both pages, pack options offered and an open bank: (i) the converted policy's scores before
    the mask for rows 0..95 and its value equal the source's, and (ii) with the source's mask applied to both, its
    choice probabilities equal the source's, rows 96..127 none; the new rows' own weights and the new input columns are
    zero and the scorers are the source's. Nothing crosses without the conversion: the source's tensors do not load into
    the wide policy, and check_names refuses each under the other's contract."""
    import torch
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    from baseline.policy import check_names, widen_state
    source, wide = free_play_policies()
    reset = {i['slot']: (i['id'], i['count']) for i in RESET['inventory'][:8]}
    steps = [wide_obs(reset, 0, held={0, 3, 6, 7}), wide_obs(reset, 1),
             wide_obs({0: (1511, 1), 1: (1511, 1), 5: (2309, 1), 9: (1511, 1), 27: (995, 250)}, 0, held={1}),
             wide_obs({}, 0), wide_obs({**reset, **{k: (2309, 1) for k in range(8, 28)}}, 0, held=set(range(8)))]
    steps[2].update(bankOpen=True, bankSel=[dict(slot=0, id=1511, count=40, name='Logs')] +
                    [dict(slot=-1, id=-1, count=0, name='') for _ in range(7)])
    assert all(len({e['id'] for e in s['invSel'] if e['slot'] >= 0}) <= 8 for s in steps)
    pairs = [(project_v6(narrow_obs(s)), project_v7(s)) for s in steps]
    worst = dict(scores=0.0, values=0.0, probabilities=0.0)
    for policy in (source, wide):
        policy.set_training_mode(False)
    states = {id(p): RNNStates(*[tuple(torch.zeros(1, 1, 16) for _ in range(2))] * 2) for p in (source, wide)}
    for k, (narrow, broad) in enumerate(pairs):
        start = torch.as_tensor([float(k == 0)])
        seen = {}
        for policy, row in ((source, narrow), (wide, broad)):
            scores = []
            original = policy._masked_logits
            policy._masked_logits = lambda latent, original=original: scores.append(original(latent)) or scores[-1]
            mask = np.concatenate([narrow['mask'], np.zeros(len(row['mask']) - 96, np.int8)])
            open_ = {key: v[None] for key, v in dict(row, mask=np.ones_like(row['mask'])).items()}
            masked = {key: v[None] for key, v in dict(row, mask=mask).items()}
            with torch.no_grad():
                # the distribution object is the policy's own, which the next call refills: its probabilities first
                probs = policy.get_distribution(policy.obs_to_tensor(masked)[0], states[id(policy)].pi,
                                                start)[0].distribution.probs[0].clone()
                _, value, _, states[id(policy)] = policy.forward(policy.obs_to_tensor(open_)[0], states[id(policy)], start,
                                                                 deterministic=True)
            del policy._masked_logits
            seen[id(policy)] = (scores[-1][0], value[0], probs)
        (s1, v1, p1), (s2, v2, p2) = seen[id(source)], seen[id(wide)]
        assert not p2[96:].any()
        worst = dict(scores=max(worst['scores'], float((s1 - s2[:96]).abs().max())),
                     values=max(worst['values'], float((v1 - v2).abs().max())),
                     probabilities=max(worst['probabilities'], float((p1 - p2[:96]).abs().max())))
    assert max(worst.values()) <= 1e-6, worst
    state, widened = wide.state_dict(), source.state_dict()
    assert not state['action_net.weight'][96:].any() and not state['action_net.bias'][96:].any()
    assert all(torch.equal(state[k], widened[k]) for k in widened if 'scorer' in k or 'lstm' in k or 'value_net' in k)
    with pytest.raises(RuntimeError):
        wide.load_state_dict(source.state_dict(), strict=True)
    with pytest.raises(ValueError):
        widen_state(source.state_dict(), source)
    with pytest.raises(ValueError):
        widen_state(wide.state_dict(), wide)
    check_names(wide, observation='obs-v7', contract=wide.contract)
    for policy, other in ((source, wide), (wide, source)):
        with pytest.raises(ValueError):
            check_names(policy, observation=other.observation, contract=other.contract)
