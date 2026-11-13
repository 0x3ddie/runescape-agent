"""The batched environment contract between the simulator and the device learner.

The CPU reference (sim/vec.py, sim/workers.py) and sim/learn.py FakeEnv implement it; the learner drives it. Shapes
are those of obs-v5 (rl/obs.py), so a model built from OBSERVATION_SHAPES is the model the native evaluator loads.

    env = make(...)                              # B worlds, a pool of layouts, the declared world spec
    state, obs = env.reset(key)                  # obs: {name: [B, *shape]} as OBSERVATION_SHAPES
    state, obs, reward, done, info = env.step(state, action)   # action: int32 [B], legal under obs['mask']

- reward: float32 [B], paid as the requested goal's certified progress changes and settled at the end
  to the terminal goal return (baseline/resource_task.py goal_return): the rewards of an episode sum to
  its return. Death settles to 0.
- done: bool [B]. The task horizon or death ended the episode: zero successor value, no time-limit
  bootstrap. The returned obs of a done world is the first observation of its next episode (auto-reset)
  and the recurrent state must restart there. A rollout boundary is not done.
- info: {name: [B]} arrays, INFO_FIELDS. `unsupported` marks a world whose step reached something the
  simulator does not implement: its obs, reward and done are meaningless and the collector must stop
  before any update uses the rollout. It is never a done, a failure score or a truncation.

Goals are assigned per world slot (world b asks for GOALS[b % 4] in every episode), so every update
holds the same number of transitions per goal; the physical start is sampled independently of it.
The declared exception: one goal in every world slot (goal_set).
"""
from __future__ import annotations

import numpy as np

from baseline.resource_task import GOAL_VEC, GOALS

N_ACTIONS = 96
# obs-v5: rl/obs.py project() plus the task's inputs (baseline/resource_task.py ResourceEpisode._budget), goal_vec in
# goal-v3's six entries (GOAL_VEC).
OBSERVATION_SHAPES = {
    'act_feats': (16, 11), 'act_ids': (16,), 'act_op_ids': (16,),
    'bank_feats': (8, 2), 'bank_ids': (8,),
    'choice_vec': (4, 16), 'click_feats': (24, 3), 'fb_ids': (2,), 'fb_op': (1,), 'fb_vec': (31,),
    'held_feats': (8, 2), 'held_ids': (8,), 'held_op_ids': (8,),
    'inv_feats': (28, 1), 'inv_ids': (28,), 'item_vec': (4,), 'mask': (N_ACTIONS,),
    'npc_feats': (8, 6), 'npc_ids': (8,), 'option_vec': (N_ACTIONS,),
    'player_vec': (7,), 'prog_vec': (25,), 'sel_feats': (8, 1), 'sel_ids': (8,), 'task_vec': (7,),
    'text_vec': (133,), 'use_ids': (1,), 'worn_ids': (14,),
    'budget_vec': (2,), 'critic_vec': (1,), 'goal_vec': (len(GOAL_VEC),),
}


# The first activity task's physical start, identical for every goal
# and sampled before the goal is known. Items by the game's debug names (sim and native alike); the
# weapon is one of START_WEAPONS, carried, not worn: wielding it is a learned choice.
START_KIT = (('bronze_axe', 1), ('net', 1), ('bronze_pickaxe', 1), ('shrimp', 2), ('bread', 1))
START_WEAPONS = ('bronze_dagger', 'bronze_sword')
# Base levels, each drawn uniformly and independently from its list; others stay 1. Gathering levels
# lie on both sides of the oak and iron unlocks (15); combat stats are those of an early character.
START_LEVELS = dict(woodcutting=(1, 5, 10, 14, 15, 16, 20), fishing=(1, 5, 10, 14, 15, 16, 20),
                    mining=(1, 5, 10, 14, 15, 16, 20), attack=tuple(range(1, 11)), strength=tuple(range(1, 11)),
                    defence=tuple(range(1, 11)), hitpoints=(10, 11, 12))
# Native ticks from arm, frozen 15 October after the reference checks.
HORIZON_TICKS = 240


def goal_set(goals=GOALS):
    """A batch's declared goals as a tuple: all four GOALS in order, or one of them in every world slot (the
    specialists' declared exception to equal goal counts). Any other set is refused."""
    goals = tuple(goals)
    if goals != GOALS and not (len(goals) == 1 and goals[0] in GOALS):
        raise ValueError(f'a goal set is all of {GOALS} in that order or one of them, not {goals}')
    return goals


def slot_goal(goals, slot):
    """The goal world slot `slot` asks for in every episode under the declared goal set."""
    return goals[slot % len(goals)]

def dtype(name):
    """The dtype a simulator emits: name-table rows and hashed options int32, the mask int8, else float32.
    (baseline/adapter.observation_space keeps ids as int64 Boxes; the learner casts.)"""
    if name == 'mask':
        return np.int8
    return np.int32 if name.endswith(('_ids', '_op')) else np.float32


# Per-world step information. Episode fields are those of the episode that ended at this step
# (zero elsewhere); `goal` is the goal of the episode the step belongs to.
INFO_FIELDS = {
    'goal': np.int32,              # index into GOALS
    'unsupported': np.bool_,       # see above: stop before the update
    'terminal': np.int32,          # 0 running, 1 horizon, 2 death (this step ended the episode)
    'episode_return': np.float32,  # the audited goal return of the ended episode
    'episode_decisions': np.int32,
    'episode_ticks': np.int32,     # native ticks from arm to the end
    'outcomes': np.int32,          # certified goal outcomes of the ended episode (items or kills)
    'decision_ticks': np.int32,    # native ticks this step advanced
}
TERMINAL_RUNNING, TERMINAL_HORIZON, TERMINAL_DEATH = 0, 1, 2


def check_batch(obs, n):
    """Shapes, dtypes and legality of one batch; raises on any departure (used by tests and adapters)."""
    if set(obs) != set(OBSERVATION_SHAPES):
        raise ValueError(f'observation keys differ: {sorted(set(obs) ^ set(OBSERVATION_SHAPES))}')
    for name, shape in OBSERVATION_SHAPES.items():
        value = np.asarray(obs[name])
        if value.shape != (n, *shape) or value.dtype != dtype(name):
            raise ValueError(f'{name}: {value.dtype}{value.shape}, expected {np.dtype(dtype(name))}{(n, *shape)}')
        if value.dtype.kind == 'f' and not np.isfinite(value).all():
            raise ValueError(f'{name}: non-finite values')
    mask = np.asarray(obs['mask'])
    if not np.isin(mask, (0, 1)).all() or not mask.any(axis=1).all():
        raise ValueError('invalid or all-empty action mask')
