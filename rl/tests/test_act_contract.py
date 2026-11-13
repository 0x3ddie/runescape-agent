from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from act_contract import A_ACT0, A_HELD0, ACT_SCAN_TILES, acts_from_config, name_id, project_acts
from names import identities


def affordance(id=1276, x=101, z=100, name='Tree', option='Chop down', kind='loc', combat=0):
    d = max(abs(x - 100), abs(z - 100))
    return dict(kind=kind, id=id, name=name, option=option, x=x, z=z, level=0, dx=x - 100, dz=z - 100, distance=d,
                steps=d, count=1, combatLevel=combat)


def raw(acts=None, held=()):
    acts = [affordance()] if acts is None else acts
    empty = dict(slot=-1, id=-1, count=0, name='')
    page = [dict(slot=0, id=1351, count=1, name='Bronze axe', options=['Wield'])] + [dict(empty) for _ in range(7)]
    mask = [0] * 96; mask[0] = 1
    for k in range(len(acts)): mask[A_ACT0 + k] = 1
    for k in range(len(held)): mask[A_HELD0 + k] = 1
    return dict(contractVersion='item-v2', actTargetVersion='act-v3', actSel=acts, heldSel=list(held), invSel=page,
                player=dict(x=100, z=100, level=0, combatLevel=23), mask=mask)


def acts(obs):
    return project_acts(obs, identities()['place'])


def test_an_affordance_reaches_the_policy_as_name_option_and_geometry(name_table):
    out = acts(raw([affordance(), affordance(3000, 103, 101, 'Goblin', 'Attack', 'npc', combat=30)]))
    assert out['act_ids'][:3].tolist() == [name_table.row('loc', 'Tree'), 0, 0]  # the test table has no goblin
    assert out['act_op_ids'][:2].tolist() == [name_id('Chop down'), name_id('Attack')]
    np.testing.assert_allclose(out['act_feats'][0], [1 / 15, 0, 1 / 15, 1 / 32, .1, 1, 0, 0, 0, 0])
    assert out['act_feats'][1, 6] == 1 and out['act_feats'][1, 9] == 1  # an npc that outranks the player
    assert not out['act_feats'][2:].any()


def test_held_options_match_the_visible_page(name_table):
    held = [dict(slot=0, id=1351, count=1, name='Bronze axe', option='Wield')]
    out = acts(raw(held=held))
    assert out['held_ids'][0] == name_table.row('obj', 'Bronze axe') and out['held_op_ids'][0] == name_id('Wield')
    for wrong in (dict(held[0], slot=3), dict(held[0], count=2), dict(held[0], option='Drop')):
        with pytest.raises(ValueError):
            acts(raw(held=[wrong]))


@pytest.mark.parametrize('field,value', [('id', True), ('id', -1), ('level', 1), ('dx', 2), ('distance', 2), ('steps', -1),
                                         ('count', 0), ('option', ''), ('kind', 'item'), ('name', '')])
def test_invalid_affordances_are_rejected(field, value):
    obs = raw(); obs['actSel'][0][field] = value
    with pytest.raises(ValueError):
        acts(obs)


def test_duplicate_distant_reordered_or_unmapped_slots_are_rejected():
    obs = raw([affordance(), affordance()])
    with pytest.raises(ValueError): acts(obs)
    with pytest.raises(ValueError): acts(raw([affordance(x=101 + ACT_SCAN_TILES)]))
    with pytest.raises(ValueError): acts(raw([affordance(x=104), affordance(x=101)]))
    obs = raw(); obs['mask'][A_ACT0 + 1] = 1
    with pytest.raises(ValueError): acts(obs)
    for version in (dict(actTargetVersion='gather-v1'), dict(contractVersion='legacy-v1')):
        with pytest.raises(ValueError): acts(dict(raw(), **version))


def test_the_task_requires_item_v2_and_act_v3():
    assert acts_from_config({'contract': 'item-v2', 'actTargets': 'act-v3'}) == 'act-v3'
    for bad in ({}, {'actTargets': 'act-v3'}, {'contract': 'item-v2', 'actTargets': 'act-v2'},
                {'contract': 'item-v2', 'actTargets': 'act-v1'}, {'contract': 'item-v2', 'actTargets': 'gather-v1'}):
        with pytest.raises(ValueError): acts_from_config(bad)


@st.composite
def published(draw):
    """Any ordered, distinct list of affordances the engine could publish."""
    things = draw(st.lists(st.tuples(st.integers(85, 115), st.integers(85, 115), st.sampled_from(('loc', 'npc', 'obj')),
                                     st.sampled_from(('Chop down', 'Net', 'Take', 'Attack'))), max_size=16, unique=True))
    out = [affordance(id=7, x=x, z=z, kind=kind, option=option) for x, z, kind, option in things]
    return sorted(out, key=lambda a: (a['steps'], a['x'], a['z'], a['kind'], a['id'], a['option']))


@settings(max_examples=200, deadline=None)
@given(published())
def test_every_ordered_list_projects_to_fixed_shapes_and_a_shuffle_is_refused(listed):
    out = acts(raw(listed))
    assert out['act_ids'].shape == (16,) and out['act_feats'].shape == (16, 10)
    assert all(np.isfinite(v).all() for v in out.values())
    if len(listed) > 1 and listed != list(reversed(listed)):
        shuffled = deepcopy(listed)[::-1]
        if shuffled != sorted(shuffled, key=lambda a: (a['steps'], a['x'], a['z'], a['kind'], a['id'], a['option'])):
            with pytest.raises(ValueError):
                acts(raw(shuffled))
