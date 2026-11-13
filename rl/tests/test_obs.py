import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from obs import project, N_NPC, N_INV, N_ACTIONS, PLAYER_VEC, TASK_VEC

# A fresh Draynor reset from the engine (item-v2, act-v3, mask-v3, feedback-v2): the first reset of the native-click
# fidelity recording (logs/native-click/fidelity/random, 3 November 2026). Its maskVersion was relabeled mask-v5 on
# 9 November 2026: with no item selected, mask-v5 gives the same mask. It predates held-v2 (no heldVersion).
CAPTURE = Path(__file__).resolve().parent / "fixtures" / "reset.json"
if not CAPTURE.exists():
    pytest.skip("needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository", allow_module_level=True)
FIXTURE = json.load(open(CAPTURE))
assert FIXTURE["using"] is None


def test_projection_shapes_and_dtypes():
    out = project(FIXTURE)
    assert out["player_vec"].shape == (PLAYER_VEC,) and out["player_vec"].dtype == np.float32
    assert out["npc_ids"].shape == (N_NPC,) and out["npc_ids"].dtype == np.int32
    assert out["npc_feats"].shape == (N_NPC, 6)
    assert out["inv_ids"].shape == (N_INV,) and out["inv_feats"].shape == (N_INV, 1)
    assert out["act_feats"].shape == (16, 11) and out["click_feats"].shape == (24, 3)
    assert out["held_feats"].shape == (8, 2) and out["use_ids"].shape == (1,)
    assert out["task_vec"].shape == (TASK_VEC,)
    assert out["mask"].shape == (N_ACTIONS,) and out["mask"].dtype == np.int8
    assert set(np.unique(out["mask"]).tolist()) <= {0, 1} and out["mask"][0] == 1  # WAIT is always legal


def test_projection_determinism_and_padding():
    a, b = project(FIXTURE), project(FIXTURE)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    # 18 inventory items: the slots beyond are zero-padded.
    assert not a["inv_ids"][18:].any() and not a["inv_feats"][18:].any()


def test_dialog_state_reaches_the_policy():
    A_DIALOG_CONTINUE, A_DIALOG_CHOICE0 = 62, 63
    obs = deepcopy(FIXTURE)
    obs["dialogOpen"] = True
    obs["dialogOptions"] = [{"text": "Yes, ok."}, {"text": "No thank you."}]
    obs["mask"] = [0] * N_ACTIONS
    for action in (0, A_DIALOG_CONTINUE, A_DIALOG_CHOICE0, A_DIALOG_CHOICE0 + 1):
        obs["mask"][action] = 1
    obs["actSel"], obs["heldSel"] = [], []
    out = project(obs)
    assert out["mask"][A_DIALOG_CONTINUE] == out["mask"][A_DIALOG_CHOICE0] == 1
    assert out["task_vec"][2] == 1.0


def test_items_and_affordances_are_known_by_name(name_table):
    out = project(FIXTURE)
    assert out["sel_ids"][0] == name_table.row("obj", "Bronze axe") != 0
    trees = [k for k, a in enumerate(FIXTURE["actSel"]) if a["name"] == "Tree"]
    assert trees and out["act_ids"][trees[0]] == name_table.row("loc", "Tree") != 0
    assert not out["use_ids"].any() and out["task_vec"][6] == 0  # nothing picked for use


def test_running_option_is_visible():
    # The teacher waits exactly while an option runs; the policy must see that too.
    assert not project(FIXTURE)["option_vec"].any()
    walking = project({**FIXTURE, "optionBusy": True, "optionAction": 1})["option_vec"]
    assert walking[1] == 1 and walking.sum() == 1
    with pytest.raises(ValueError):
        project({**FIXTURE, "optionBusy": True})


def test_projection_does_not_see_absolute_position():
    # With x, z and the map square as inputs, imitation learned Draynor, not the task.
    moved = deepcopy(FIXTURE)
    for thing in [moved["player"], *moved["actSel"]]:
        thing["x"] += 500
        thing["z"] -= 300
    here, there = project(FIXTURE), project(moved)
    assert here.keys() == there.keys()
    for key in here:
        np.testing.assert_array_equal(here[key], there[key])
