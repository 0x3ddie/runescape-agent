"""rl/obs.py: an environment observation (item-v2, act-v3, feedback-v2) -> fixed-size arrays.

obs-v5 is what a policy reads: these arrays with text-v2 (rl/feedback_contract.py), plus the task's budget_vec,
critic_vec and goal-v3's six-entry goal_vec (baseline/resource_task.py GOAL_VEC). obs-v6 (free-play-v1 only) is
project_v6: these arrays without task_vec's decision count, with each skill's progress to its next level after
prog_vec's 25 entries, and no task input: no goal and no time. It is obs-v3
with two more goal entries (cook, fire); obs-v4 is a name reserved by another draft. obs-v3 was obs-v2 with the click
features (act-v3) in place of the place slots and compass moves, npcs within 15 tiles (the server's view distance) and
each affordance's bank column read from the 15-tile loc and npc lists. A checkpoint records the observation, action,
mask and feedback versions it was trained on (baseline/policy.py check_names); any other is refused. The one conversion
is baseline/policy.py convert_checkpoint: an obs-v3, mask-v3 checkpoint to obs-v5, mask-v5, recorded in the result.

The mask follows MASK (env-ts/core/mask.ts), named by each observation. mask-v3 is mask-v2 with a click legal when its
target lies in the declared world and no bank, use or dialog mode is open, and RETREAT (row 71) never legal. mask-v5 is
mask-v3 with INV_PREV and INV_NEXT legal while an item is selected for use.

Things are identified by their row in the names-v1 table (rl/names.py; 0 is padding or a
name the table lacks). Entity lists are nearest first with zero padding and presence
features; coordinates are egocentric. No feature names a skill: what a thing is comes
from its name, what it allows from its options.
"""
from __future__ import annotations

import numpy as np
from act_contract import bank_distance, banks, project_acts, project_clicks
from feedback_contract import project_feedback
from names import identities
from item_contract import ITEM_V3, N_BANK_SEL, BANK_CAPACITY, MAX_BANK_PAGES, observation_contract

N_NPC = 8
N_INV = 28
N_WORN = 14
N_INV_SEL = 8   # inventory page slots (DROP/DEPOSIT_k, item-v2)
# must match N_ACTIONS and the act-v3 table in env-ts/core/actions.ts: 1..16 act on
# affordances, 17..28 and 84..95 minimap clicks, 29..36 use held items, 37..60 drop, deposit and
# withdraw page slots, 61..75 interfaces, dialogs, pages, retreat (never legal) and combat style,
# 76..83 pick a page item to use on something.
N_ACTIONS = 96
A_DROP0, A_DEPOSIT0, A_WITHDRAW0, A_USE0 = 37, 45, 53, 76
PLAYER_VEC = 7
TASK_VEC = 7
MAX_DECISIONS = 1500
OBSERVATION = 'obs-v5'
OBSERVATION_V6 = 'obs-v6'
MASK = 'mask-v5'

# canonical skill order for the progression vector (RS 2004 has 21 skills)
SKILL_ORDER = (
    "Attack", "Defence", "Strength", "Hitpoints", "Ranged", "Prayer", "Magic",
    "Cooking", "Woodcutting", "Fletching", "Fishing", "Firemaking", "Crafting",
    "Smithing", "Mining", "Herblore", "Agility", "Thieving", "Slayer",
    "Farming", "Runecrafting",
)
PROG_VEC = len(SKILL_ORDER) + 4
PROG_VEC_V6, TASK_VEC_V6 = PROG_VEC + len(SKILL_ORDER), TASK_VEC - 1  # obs-v6: progress added, the decision count gone


def _level_xp():
    """Player.ts levelExperience in tenths of XP, as the server stores it: the client's copy of the table its skills tab
    reads (sim/engine.py LEVEL_XP is the server's; a test holds them equal). Entry k is the XP of level k + 2."""
    table, acc = [], 0
    for level in range(1, 100):
        acc += int(level + 2.0 ** (level / 10.0) * 300.0)
        table.append(acc // 4 * 10)
    return table


LEVEL_XP = _level_xp()


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def project(obs: dict) -> dict[str, np.ndarray]:
    observation_contract(obs)
    ident = identities()
    npc_id, item_id = ident['npc'], lambda item: ident['obj'](item.get("name", ""))
    p = obs.get("player", {}) or {}

    # No absolute position: a policy given x, z and the map square learned Draynor
    # rather than the task (0 logs in Varrock and Falador). Where things are comes
    # from the slots, relative to here.
    pv = np.zeros(PLAYER_VEC, dtype=np.float32)
    pv[0] = float(p.get("level", 0))
    pv[1] = _clip(p.get("hp", 0) / max(1, p.get("maxHp", 1)), 0.0, 1.0)
    pv[2] = _clip(p.get("combatLevel", 0) / 126.0, 0.0, 1.0)
    pv[3] = 1.0 if p.get("animId", -1) != -1 else 0.0
    pv[4] = 1.0 if p.get("isDead") else 0.0
    pv[5] = 1.0 if p.get("inCombat") else 0.0
    pv[6] = 1.0 if p.get("targetIndex", -1) != -1 else 0.0

    npc_ids = np.zeros(N_NPC, dtype=np.int32)
    npc_feats = np.zeros((N_NPC, 6), dtype=np.float32)
    for i, n in enumerate((obs.get("npcs", []) or [])[:N_NPC]):
        npc_ids[i] = npc_id(n.get("name", ""))
        npc_feats[i, 0] = _clip(n.get("dx", 0) / 64.0, -1.0, 1.0)
        npc_feats[i, 1] = _clip(n.get("dz", 0) / 64.0, -1.0, 1.0)
        npc_feats[i, 2] = _clip(n.get("distance", 0) / 32.0, 0.0, 1.0)
        npc_feats[i, 3] = _clip((n.get("combatLevel") or 0) / 40.0, 0.0, 4.0)
        npc_feats[i, 4] = 1.0 if n.get("inCombat") else 0.0
        npc_feats[i, 5] = 1.0 if any("attack" in str(o).lower() for o in (n.get("options") or [])) else 0.0

    inv_ids = np.zeros(N_INV, dtype=np.int32)
    inv_feats = np.zeros((N_INV, 1), dtype=np.float32)
    for item in (obs.get("inventory", []) or [])[:N_INV]:
        slot = int(item.get("slot", -1))
        if 0 <= slot < N_INV:
            inv_ids[slot] = item_id(item)
            inv_feats[slot, 0] = float(np.log1p(item.get("count", 0)) / 5.0)

    worn_ids = np.zeros(N_WORN, dtype=np.int32)
    for item in (obs.get("equipment", []) or [])[:N_WORN]:
        slot = int(item.get("slot", -1))
        if 0 <= slot < N_WORN:
            worn_ids[slot] = item_id(item)

    # The inventory page: what DROP_k and DEPOSIT_k would act on. Its count input is the slot's own count (item-v2), or
    # under item-v3 (the page of kinds) the kind's total over the pack.
    sel_ids = np.zeros(N_INV_SEL, dtype=np.int32)
    sel_feats = np.zeros((N_INV_SEL, 1), dtype=np.float32)
    counted = "total" if obs["contractVersion"] == ITEM_V3 else "count"
    for i, item in enumerate((obs.get("invSel", []) or [])[:N_INV_SEL]):
        if item["slot"] == -1:
            continue
        sel_ids[i] = item_id(item)
        sel_feats[i, 0] = float(np.log1p(item.get(counted, 0)) / 5.0)

    # Progression: every skill's permanent level, total level, wealth, hitpoints, fullness.
    pg = np.zeros(PROG_VEC, dtype=np.float32)
    levels = {s.get("name"): int(s.get("baseLevel") or s.get("level", 1)) for s in obs.get("skills", []) or []}
    for i, name in enumerate(SKILL_ORDER):
        pg[i] = _clip(levels.get(name, 1) / 99.0, 0.0, 1.0)
    inv = obs.get("inventory", []) or []
    k = len(SKILL_ORDER)
    pg[k] = _clip(sum(levels.get(name, 1) for name in SKILL_ORDER) / 2079.0, 0.0, 1.0)  # 21 skills x 99
    gp = sum(int(i.get("count", 0)) for i in inv if str(i.get("name", "")).lower() == "coins")
    pg[k + 1] = float(np.log1p(gp) / 12.0)  # log wealth: 1gp..~160k maps into [0,1]
    pg[k + 2] = _clip(p.get("hp", 0) / max(1, p.get("maxHp", 1)), 0.0, 1.0)
    pg[k + 3] = _clip(p.get("combatLevel", 0) / 126.0, 0.0, 1.0)

    tv = np.zeros(TASK_VEC, dtype=np.float32)
    tv[0] = _clip(float(obs.get("decision", 0)) / MAX_DECISIONS, 0.0, 1.0)
    tv[1] = 1.0 if obs.get("bankOpen") else 0.0
    tv[2] = 1.0 if obs.get("dialogOpen") else 0.0
    tv[3] = 1.0 if obs.get("modalOpen") else 0.0
    tv[4] = _clip(len(inv) / 28.0, 0.0, 1.0)  # fullness
    tv[5] = _clip(len(obs.get("dialogOptions", []) or []) / 5.0, 0.0, 1.0)
    # An item picked for use (USE_k): the affordances then offer only "Use".
    using = obs.get("using")
    tv[6] = 1.0 if using else 0.0
    use_ids = np.asarray([item_id(using) if using else 0], dtype=np.int32)

    # The option still running, by the action that started it (all zero when idle).
    # WAIT continues it, so without this walking and standing idle look alike.
    option_vec = np.zeros(N_ACTIONS, dtype=np.float32)
    if obs.get("optionBusy"):
        running = obs.get("optionAction")
        if type(running) is not int or not 0 < running < N_ACTIONS:
            raise ValueError(f"busy observation without a running optionAction: {running!r}")
        option_vec[running] = 1.0

    if "mask" in obs and obs.get("maskVersion") != MASK:
        raise ValueError(f"the observation's mask follows {obs.get('maskVersion') or 'mask-v1'}, not {MASK}")
    mask = np.asarray(obs.get("mask", [1] + [0] * (N_ACTIONS - 1)), dtype=np.int8)
    assert mask.shape == (N_ACTIONS,), f"mask shape {mask.shape}"

    bank_ids = np.zeros(N_BANK_SEL, dtype=np.int32)
    bank_feats = np.zeros((N_BANK_SEL, 2), dtype=np.float32)
    for i, item in enumerate(obs['bankSel']):
        if item['slot'] == -1:
            continue
        bank_ids[i] = item_id(item)
        bank_feats[i] = (np.log1p(item['count']) / 5.0, item['slot'] / (BANK_CAPACITY - 1))

    result = {
        "player_vec": pv, "npc_ids": npc_ids, "npc_feats": npc_feats, "inv_ids": inv_ids, "inv_feats": inv_feats,
        "worn_ids": worn_ids, "use_ids": use_ids, "sel_ids": sel_ids, "sel_feats": sel_feats, "bank_ids": bank_ids, "bank_feats": bank_feats,
        "prog_vec": pg, "task_vec": tv, "option_vec": option_vec, "mask": mask,
        "item_vec": np.asarray([obs['invPage'] / 3, obs['bankPage'] / max(1, obs['bankPages'] - 1),
                                float(obs['bankOpen']), obs['bankPages'] / MAX_BANK_PAGES], dtype=np.float32),
    }
    result.update(project_acts(obs, ident['place']))
    result.update(project_clicks(obs))
    result.update(project_feedback(obs, ident['place']))
    # Each affordance also knows how far it stands from a bank in the 15-tile lists.
    column, near = np.zeros((len(result['act_feats']), 1), dtype=np.float32), banks(obs)
    for i, a in enumerate(obs['actSel']):
        column[i] = bank_distance(a, near)
    result['act_feats'] = np.concatenate([result['act_feats'], column], axis=1)
    return result


def level_progress(xp, level):
    """A skill's progress to its next level, in [0, 1): whole XP `xp` (as observed) at base level `level`; 0 at 99."""
    if level >= 99:
        return 0.0
    low = LEVEL_XP[level - 2] if level >= 2 else 0
    return _clip((10 * xp - low) / (LEVEL_XP[level - 1] - low), 0.0, 1.0)


def project_v6(obs: dict) -> dict[str, np.ndarray]:
    """obs-v6 (module docstring): project() without task_vec's decision count, the time input of a native observation,
    and with prog_vec's 25 entries followed by each SKILL_ORDER skill's level_progress. Every skill observed must carry
    its XP."""
    result = project(obs)
    result['task_vec'] = result['task_vec'][1:]
    skills = {s.get('name'): s for s in obs.get('skills', []) or []}
    progress = np.zeros(len(SKILL_ORDER), dtype=np.float32)
    for i, name in enumerate(SKILL_ORDER):
        s = skills.get(name)
        if s is None:
            continue
        if type(s.get('xp')) is not int:
            raise ValueError(f'skill {name} observed without its XP: obs-v6 shows the progress to the next level')
        progress[i] = level_progress(s['xp'], int(s.get('baseLevel') or s.get('level', 1)))
    result['prog_vec'] = np.concatenate([result['prog_vec'], progress])
    return result
