"""names-v1: a thing is identified by its row in the name table (rl/names.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

import names
from obs import project

CAPTURE = Path(__file__).resolve().parent / "fixtures" / "reset.json"
if not CAPTURE.exists():
    pytest.skip("needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository", allow_module_level=True)
RESET = json.load(open(CAPTURE))


@pytest.fixture
def table(tmp_path, monkeypatch):
    path = tmp_path / 'names.npz'
    np.savez(path, kind=np.array(['loc', 'npc', 'obj', 'npc']), name=np.array(['tree', 'banker', 'logs', 'guard']),
             vector=np.eye(4, dtype=np.float16))
    monkeypatch.setenv('RL_NAME_TABLE', str(path))
    names.name_table.cache_clear()
    return names.name_table()


def test_known_names_get_their_table_row_and_unknown_ones_zero(table):
    # Hashed into 4,000 buckets, 520 of the cache's 1,625 loc and npc names shared one.
    obs = dict(RESET, npcs=[dict(name='Guard', dx=1, dz=0, distance=1, combatLevel=21, options=['Attack']),
                            dict(name='Hans', dx=2, dz=0, distance=2, options=['Talk-to'])],
               inventory=[dict(slot=0, id=1511, name='Logs', count=1)], invSel=[dict(slot=0, id=1511, name='Logs', count=1)]
               + RESET['invSel'][1:], heldSel=[], mask=[0 if 29 <= a < 37 else m for a, m in enumerate(RESET['mask'])])
    out = project(obs)
    assert out['npc_ids'][:2].tolist() == [table.row('npc', 'Guard'), 0] == [4, 0]
    assert out['inv_ids'][0] == out['sel_ids'][0] == table.row('obj', 'Logs') == 3
    # An affordance is known by its kind and name: the banker (npc) has a row, the booth (loc) none.
    assert not out['act_ids'][[a['name'] == 'Bank booth' for a in RESET['actSel']]].any()
    assert out['act_ids'][[a['name'] == 'Tree' for a in RESET['actSel']]].tolist() == [1]
    assert out['act_ids'][[a['name'] == 'Banker' for a in RESET['actSel']]].tolist() == [2, 2]  # Bank and Talk-to
