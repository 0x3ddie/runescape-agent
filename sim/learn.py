"""The device learner: SB3-Contrib RecurrentPPO over a sim/spec.py environment with every per-decision
tensor on the policy's device.

    model = build(DeviceEnv(env, seed=101, device='cuda'), seed=101, device='cuda', n_steps=..., ...)
    model.learn(total_timesteps)   # SB3's own learn() and RecurrentPPO.train(), unchanged

The policy is baseline/policy.py MaskedRecurrentPolicy with config.LEARNER's settings, so a saved model is a
standard RecurrentPPO zip that baseline/evaluate.py loads. Only the rollout path differs from SB3's:

- HostDeviceEnv steps the CPU workers (sim/workers.py) and copies each step to the learner's device; DeviceEnv
  steps a JAX environment (FakeEnv) and hands its arrays to torch by DLPack (same memory, no copy); ids are cast to
  the model's int64 on the device and actions go back as int32.
- DeviceRolloutBuffer is RecurrentDictRolloutBuffer as torch operations: the same stored LSTM states,
  episode starts, GAE arithmetic, split-index shuffle, sequence splitting, zero padding and masks.
- DeviceRecurrentPPO.collect_rollouts is RecurrentPPO.collect_rollouts without host copies. Episode
  statistics and `unsupported` are read once per rollout. An unsupported world voids the rollout: it
  raises Unsupported, with the evidence written down, before the bootstrap value and before train().

sim/tests/test_learn.py proves the equivalence with SB3's own collector and buffer. FakeEnv implements the
spec with random legal batches for tests and development. Tracker and parameter_change record read-only
per-update telemetry; attach adds them. The self-distillation phase (sim/distill.py) and self-imitation
(self-imitation-v1) were removed.

The goal-free study adds three learner settings, each off for the goal
tasks, whose arithmetic then stays SB3's: free_play (a life cut by its drawn length, TERMINAL_HORIZON, bootstraps from the
value of its last observation under that life's recurrent state; a death does not; info['goal'] is never read),
half_life_ticks (the advantage's per-tick discount, 0.5 ** (1 / half_life_ticks) a game tick, applied as its power tau
for a decision of tau ticks, info['decision_ticks']), and the policy's value_norm (baseline/policy.py PopArt). The model is
built from the observation version and table of shapes the environment declares (sim/spec.py), so an obs-v6 policy
reads no goal_vec and no budget_vec. sim/tests/test_free_play_learner.py tests them.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RecurrentDictRolloutBufferSamples, RNNStates
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv
from torch.utils._python_dispatch import TorchDispatchMode

from baseline import config
from baseline.adapter import observation_space
from baseline.policy import OPTION_VOCAB
from baseline.resource_task import GOAL_VEC, GOALS, GOAL_TARGETS
from obs import OBSERVATION
from sim.spec import (HORIZON_TICKS, INFO_FIELDS, N_ACTIONS, OBSERVATION_SHAPES, TERMINAL_DEATH, TERMINAL_HORIZON,
                      TERMINAL_RUNNING, dtype, goal_set, slot_goal)

# What the policy reads (adapter.observation_space): ids as int64, the mask int8, everything else float32.
MODEL_DTYPES = {np.dtype(np.int64): torch.int64, np.dtype(np.int8): torch.int8, np.dtype(np.float32): torch.float32}
# What the buffer stores: the simulator's own dtypes (ids fit int32), cast back to the model's in get().
STORE_DTYPES = {np.dtype(np.int64): torch.int32, np.dtype(np.int8): torch.int8, np.dtype(np.float32): torch.float32}


def option_ids(name):
    """Hashed option text (act_contract.name_id, OPTION_VOCAB rows); every other *_ids is a name-table row."""
    return name.endswith('_op_ids') or name == 'fb_op'


class Ended(NamedTuple):
    """The lives that ended at one step (a free-play step's info['last_obs'] on the learner's device): their world rows
    [k] and the observation each life ended on, {name: [k, ...]} in the model's dtypes."""
    rows: torch.Tensor
    obs: dict


class Unsupported(RuntimeError):
    """A world reached something the simulator does not implement (sim/spec.py): the rollout is void.
    `evidence` lists every unsupported world-step with its action and info; `path` is where it was saved."""
    def __init__(self, message, evidence, path=None):
        super().__init__(message)
        self.evidence, self.path = evidence, path


# ---- a spec environment for tests and development ---------------------------------------------------------
class FakeState(NamedTuple):
    key: jax.Array
    step: jax.Array       # vector steps since reset
    illegal: jax.Array    # actions the mask did not offer, over all worlds and steps
    mask: jax.Array       # [B, 96] the mask the next action answers
    target: jax.Array     # [B] the action that produces an outcome this episode
    decisions: jax.Array  # [B]
    ticks: jax.Array      # [B]
    outcomes: jax.Array   # [B] goal outcomes so far this episode


# FakeEnv's stand-in for obs-v6 (the real table is sim/spec.py's): obs-v5 without
# goal_vec and budget_vec, and prog_vec wider by one progress entry for each of the seven skills of S. Its version name
# says it is a stand-in, so no checkpoint of it passes for an obs-v6 one.
FAKE_FREE_PLAY = 'fake-obs-v6'
FAKE_FREE_PLAY_SHAPES = {**{k: v for k, v in OBSERVATION_SHAPES.items() if k not in ('goal_vec', 'budget_vec')},
                         'prog_vec': (OBSERVATION_SHAPES['prog_vec'][0] + 7,)}


class FreeState(NamedTuple):
    """FakeEnv's free-play state: FakeState's fields, each life's drawn length in ticks and its pay so far."""
    key: jax.Array
    step: jax.Array
    illegal: jax.Array
    mask: jax.Array
    target: jax.Array
    decisions: jax.Array
    ticks: jax.Array
    outcomes: jax.Array
    length: jax.Array   # [B] the life's drawn length in ticks, never observed
    paid: jax.Array     # [B] float32, the life's rewards summed as paid


class FakeEnv:
    """Random legal obs-v2 batches with simple dynamics, implementing sim/spec.py. World b asks for
    GOALS[b % 4], or for the one goal of a single-goal set (sim/spec.py goal_set). Each episode has one target action, always legal and marked in option_vec; taking it
    produces one goal outcome (up to GOAL_TARGETS), paid when produced, so an episode's rewards sum to
    its return. Each decision advances 1-3 native ticks; the episode ends at horizon_ticks, or by death
    (probability death_rate per decision), which pays back everything paid. unsupported: (step, world)
    pairs whose step reports `unsupported`; with garbage=True those worlds' next mask is empty, as a
    simulator's meaningless observation can be. Ids are drawn below id_rows (the name table's rows).
    free_play: the goal-free task's stand-in instead, as the seam with the free-play batch has it: FAKE_FREE_PLAY_SHAPES,
    no goal (the info's `goal` is -1, which no reader may take for one), each life's length drawn uniformly from
    life_ticks and never shown, the target action paying `gain` whenever taken (a death takes nothing back),
    TERMINAL_HORIZON when the life's ticks reach its length, and info['last_obs']: each ended life's observation after its
    last action, zero elsewhere, while obs is the next life's first; episode_return is the life's pay, summed as paid, and
    outcomes its whole part (one paid skill's integer gain). settings_in_effect: these settings, as sim/workers.py
    CpuWorkers reports a free-play batch's (None for the goal task)."""

    def __init__(self, n, *, id_rows, horizon_ticks=HORIZON_TICKS, death_rate=.002, unsupported=(), garbage=False,
                 goals=GOALS, free_play=False, life_ticks=(6, 30), gain=.25):
        self.free_play = free_play
        if free_play:
            if goals is not GOALS or not 1 <= life_ticks[0] <= life_ticks[1]:
                raise ValueError('the free-play stand-in has no goals and draws life lengths from 1 tick up')
            goals = ()
            self.observation, self.shapes, self.last_obs = FAKE_FREE_PLAY, FAKE_FREE_PLAY_SHAPES, True
            self.life_ticks, self.gain = tuple(life_ticks), gain
            self.settings_in_effect = dict(observation=FAKE_FREE_PLAY, life=['uniform', *map(int, life_ticks)], gain=gain,
                                           death_rate=death_rate)
        else:
            goals = goal_set(goals)
            if n % len(goals):
                raise ValueError(f'worlds ({n}) must be a multiple of {len(goals)}: goals are per world slot')
            self.observation, self.shapes, self.last_obs = OBSERVATION, OBSERVATION_SHAPES, False
            self.settings_in_effect = None
        self.n, self.id_rows, self.horizon_ticks, self.death_rate = n, id_rows, horizon_ticks, death_rate
        self.unsupported, self.garbage = tuple(unsupported), garbage
        self.goals = goals
        if not free_play:
            self.goal = jnp.asarray([GOALS.index(slot_goal(goals, b)) for b in range(n)], jnp.int32)
            self.cap = jnp.asarray([GOAL_TARGETS[goal] for goal in GOALS], jnp.int32)[self.goal]
        # Compiled here: callers use reset and step as they are.
        self.reset, self.step = (jax.jit(self._reset_free), jax.jit(self._step_free)) if free_play else \
            (jax.jit(self._reset), jax.jit(self._step))

    def _observe(self, key, target, decisions, ticks):
        obs = {}
        for (name, shape), k in zip(sorted(self.shapes.items()), jax.random.split(key, len(self.shapes))):
            size = (self.n, *shape)
            if name == 'mask':
                value = jax.random.bernoulli(k, .3, size) | jax.nn.one_hot(target, N_ACTIONS, dtype=bool)
            elif name == 'option_vec':
                value = jax.nn.one_hot(target, N_ACTIONS)
            elif name == 'goal_vec':
                value = jax.nn.one_hot(self.goal, len(GOAL_VEC))
            elif name == 'budget_vec':
                value = jnp.stack([decisions / self.horizon_ticks, ticks / self.horizon_ticks], 1)
            elif dtype(name) == np.int32:
                value = jax.random.randint(k, size, 0, OPTION_VOCAB if option_ids(name) else self.id_rows)
            else:
                value = jax.random.normal(k, size)
            obs[name] = value.astype(dtype(name))
        return obs

    def _reset(self, key):
        key, k_target, k_obs = jax.random.split(key, 3)
        zero = jnp.zeros(self.n, jnp.int32)
        target = jax.random.randint(k_target, (self.n,), 0, N_ACTIONS)
        obs = self._observe(k_obs, target, zero, zero)
        return FakeState(key, jnp.int32(0), jnp.int32(0), obs['mask'], target, zero, zero, zero), obs

    def _step(self, state, action):
        key, k_ticks, k_death, k_target, k_obs = jax.random.split(state.key, 5)
        offered = jnp.take_along_axis(state.mask, action[:, None], 1)[:, 0] > 0
        death = jax.random.bernoulli(k_death, self.death_rate, (self.n,))
        produced = (action == state.target) & (state.outcomes < self.cap) & ~death
        outcomes = state.outcomes + produced
        decisions, ticks = state.decisions + 1, state.ticks + jax.random.randint(k_ticks, (self.n,), 1, 4)
        horizon = ticks >= self.horizon_ticks
        done = death | horizon
        terminal = jnp.where(death, TERMINAL_DEATH, jnp.where(horizon, TERMINAL_HORIZON, TERMINAL_RUNNING))
        reward = jnp.where(death, -state.outcomes / self.cap, produced / self.cap).astype(jnp.float32)
        unsupported = jnp.zeros(self.n, bool)
        for step, world in self.unsupported:
            unsupported = unsupported.at[world].set(unsupported[world] | (state.step == step))
        info = dict(goal=self.goal, unsupported=unsupported, terminal=terminal.astype(jnp.int32),
                    episode_return=jnp.where(done & ~death, outcomes / self.cap, 0.).astype(jnp.float32),
                    episode_decisions=jnp.where(done, decisions, 0), episode_ticks=jnp.where(done, ticks, 0),
                    outcomes=jnp.where(done & ~death, outcomes, 0), decision_ticks=ticks - state.ticks)
        info = {name: info[name].astype(kind) for name, kind in INFO_FIELDS.items()}
        target = jnp.where(done, jax.random.randint(k_target, (self.n,), 0, N_ACTIONS), state.target)
        decisions, ticks, outcomes = (jnp.where(done, 0, value) for value in (decisions, ticks, outcomes))
        obs = self._observe(k_obs, target, decisions, ticks)
        if self.garbage:
            obs['mask'] = jnp.where(unsupported[:, None], jnp.int8(0), obs['mask'])
        state = FakeState(key, state.step + 1, state.illegal + jnp.sum(~offered), obs['mask'], target, decisions, ticks,
                          outcomes)
        return state, obs, reward, done, info

    def _unsupported(self, state):
        unsupported = jnp.zeros(self.n, bool)
        for step, world in self.unsupported:
            unsupported = unsupported.at[world].set(unsupported[world] | (state.step == step))
        return unsupported

    def _length(self, key):
        return jax.random.randint(key, (self.n,), self.life_ticks[0], self.life_ticks[1] + 1)

    def _reset_free(self, key):
        key, k_target, k_length, k_obs = jax.random.split(key, 4)
        zero = jnp.zeros(self.n, jnp.int32)
        target = jax.random.randint(k_target, (self.n,), 0, N_ACTIONS)
        obs = self._observe(k_obs, target, zero, zero)
        return FreeState(key, jnp.int32(0), jnp.int32(0), obs['mask'], target, zero, zero, zero, self._length(k_length),
                         jnp.zeros(self.n, jnp.float32)), obs

    def _step_free(self, state, action):
        key, k_ticks, k_death, k_target, k_length, k_last, k_obs = jax.random.split(state.key, 7)
        offered = jnp.take_along_axis(state.mask, action[:, None], 1)[:, 0] > 0
        death = jax.random.bernoulli(k_death, self.death_rate, (self.n,))
        produced = (action == state.target) & ~death
        reward = jnp.where(produced, self.gain, 0.).astype(jnp.float32)
        outcomes, paid = state.outcomes + produced, state.paid + reward
        decisions, ticks = state.decisions + 1, state.ticks + jax.random.randint(k_ticks, (self.n,), 1, 4)
        cut = (ticks >= state.length) & ~death
        done = death | cut
        terminal = jnp.where(death, TERMINAL_DEATH, jnp.where(cut, TERMINAL_HORIZON, TERMINAL_RUNNING))
        unsupported = self._unsupported(state)
        info = dict(goal=jnp.full(self.n, -1), unsupported=unsupported, terminal=terminal,
                    episode_return=jnp.where(done, paid, 0.), episode_decisions=jnp.where(done, decisions, 0),
                    episode_ticks=jnp.where(done, ticks, 0), outcomes=jnp.where(done, jnp.floor(paid), 0),
                    decision_ticks=ticks - state.ticks)
        info = {name: info[name].astype(kind) for name, kind in INFO_FIELDS.items()}
        ended = lambda value: jnp.where(done.reshape(-1, *[1] * (value.ndim - 1)), value, jnp.zeros_like(value))
        info['last_obs'] = {k: ended(v) for k, v in self._observe(k_last, state.target, decisions, ticks).items()}
        target = jnp.where(done, jax.random.randint(k_target, (self.n,), 0, N_ACTIONS), state.target)
        length = jnp.where(done, self._length(k_length), state.length)
        decisions, ticks, outcomes = (jnp.where(done, 0, value) for value in (decisions, ticks, outcomes))
        paid = jnp.where(done, 0., paid).astype(jnp.float32)
        obs = self._observe(k_obs, target, decisions, ticks)
        if self.garbage:
            obs['mask'] = jnp.where(unsupported[:, None], jnp.int8(0), obs['mask'])
        state = FreeState(key, state.step + 1, state.illegal + jnp.sum(~offered), obs['mask'], target, decisions, ticks,
                          outcomes, length, paid)
        return state, obs, reward, done, info


# ---- the environment as SB3 sees it ------------------------------------------------------------------------
class SpecVecEnv(VecEnv):
    """A sim/spec.py environment behind SB3's VecEnv interface: B worlds. env.reset and env.step are called
    as they are: the environment compiles them, and wrapping them in another jit would make its tables
    constants of that program. One warm-up reset and step, from a key the run never uses again and
    a discarded state, compile both and check the observation and info layout against the spec. Keys and
    actions are committed to the JAX device, as DLPack actions are: jit compiles again for an uncommitted
    argument (a 20-second compile per signature on Draynor).
    The environment declares the observation version and table of shapes its observations follow (`observation`,
    `shapes`; default obs-v5, OBSERVATION_SHAPES) and whether its steps give info['last_obs'] (`last_obs`, the free-play
    batch); the observation space, and so the model, is built from that table, and the warm-up holds the environment to
    it. A free-play environment also reports its settings in effect (`settings_in_effect`) for the run's record."""
    render_mode = None

    def __init__(self, env, seed):
        self.env, self.jax_device = env, jax.devices()[0]
        self.observation = getattr(env, 'observation', OBSERVATION)
        self.shapes = getattr(env, 'shapes', OBSERVATION_SHAPES)
        self.last_obs = getattr(env, 'last_obs', False)
        self.settings_in_effect = getattr(env, 'settings_in_effect', None)
        self._key = jax.device_put(jax.random.key(seed), self.jax_device)
        state, obs = env.reset(self._key)
        n = obs['mask'].shape[0]
        layout = lambda values: {k: (v.shape, np.dtype(v.dtype)) for k, v in values.items()}
        observations = {name: ((n, *shape), np.dtype(dtype(name))) for name, shape in self.shapes.items()}
        if layout(obs) != observations:
            raise ValueError(f'the environment departs from its declared {self.observation} observations: {layout(obs)}')
        _, _, reward, done, info = env.step(state, jax.device_put(jnp.zeros(n, jnp.int32), self.jax_device))
        info = dict(info)
        ended = info.pop('last_obs', None)
        expected = {name: ((n,), np.dtype(kind)) for name, kind in INFO_FIELDS.items()}
        if layout(dict(reward=reward, done=done)) != dict(reward=((n,), np.dtype(np.float32)), done=((n,), np.dtype(bool))) \
                or {k: v for k, v in layout(info).items() if k in expected} != expected:
            raise ValueError(f'the environment departs from sim/spec.py steps: {layout(info)}')
        if (ended is not None) != self.last_obs or (ended is not None and layout(ended) != observations):
            raise ValueError(f'the environment\'s last_obs departs from its declaration (last_obs={self.last_obs}): '
                             f'{None if ended is None else layout(ended)}')
        self._reset, self._step = env.reset, env.step
        self.state = None
        sample = {name: np.zeros(shape, dtype(name)) for name, shape in self.shapes.items()}
        super().__init__(n, observation_space(sample), spaces.Discrete(N_ACTIONS))

    def _reset_state(self):
        self._key, key = jax.random.split(self._key)
        self.state, obs = self._reset(key)
        return obs

    def get_attr(self, attr_name, indices=None):
        if attr_name != 'render_mode':
            raise AttributeError(attr_name)
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError('the simulator configuration is fixed when the environment is made')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('use the spec interface')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.state = None


class HostVecEnv(SpecVecEnv):
    """The reference path: the environment through numpy, as SB3's own collect_rollouts consumes a VecEnv
    (sim/tests/test_learn.py, and the host-copy comparison of `sim.train transfers`). A done is terminal,
    never a time-limit truncation, so SB3 does not bootstrap at an episode end (sim/spec.py). The exception is an
    environment that gives last_obs (the free-play stand-in): there each ended life's info holds its
    terminal_observation and a cut life (TERMINAL_HORIZON) is a time-limit truncation, so SB3's own collector bootstraps
    it from that observation's value with the life's recurrent state, the independent reference of DeviceRecurrentPPO's
    free_play bootstrap (at a constant gamma)."""

    def _numpy(self, obs):
        return {k: np.array(v, dtype=self.observation_space[k].dtype) for k, v in obs.items()}

    def reset(self):
        return self._numpy(self._reset_state())

    def step_async(self, actions):
        self._actions = jax.device_put(np.asarray(actions, np.int32), self.jax_device)

    def step_wait(self):
        self.state, obs, reward, done, info = self._step(self.state, self._actions)
        info = dict(info)
        ended = info.pop('last_obs', None)
        info = {k: np.asarray(v) for k, v in info.items()}
        infos = [{**{k: v[i] for k, v in info.items()}, 'TimeLimit.truncated': False} for i in range(self.num_envs)]
        if ended is not None:
            ended = self._numpy(ended)
            for i in np.flatnonzero(info['terminal'] != TERMINAL_RUNNING):
                infos[i].update(terminal_observation={k: v[i] for k, v in ended.items()},
                                **{'TimeLimit.truncated': bool(info['terminal'][i] == TERMINAL_HORIZON)})
        return self._numpy(obs), np.array(reward), np.array(done), infos


class DeviceEnv(SpecVecEnv):
    """The environment on the learner's device: observations reach torch by DLPack (the same memory),
    ids are cast to the model's int64 there, and actions return to JAX as int32 the same way. JAX and
    torch must use the same device; nothing here copies through the host."""

    def __init__(self, env, seed, device='cpu'):
        super().__init__(env, seed)
        self.device = torch.device(device)
        self._dtypes = {k: MODEL_DTYPES[space.dtype] for k, space in self.observation_space.spaces.items()}

    def _torch(self, value):
        value = torch.from_dlpack(value)
        if value.device.type != self.device.type:
            raise ValueError(f'JAX produced {value.device} arrays for a {self.device} learner; run both on one device')
        return value

    def _observation(self, obs):
        return {k: self._torch(v).to(self._dtypes[k]) for k, v in obs.items()}

    def reset(self):
        return self._observation(self._reset_state())

    def step_device(self, actions):
        """actions: int64 [B] on the device. Returns observations (model dtypes), reward, done, info: all on it. An
        environment's info['last_obs'] becomes Ended: the rows of the lives that ended and their last observations."""
        actions = jax.dlpack.from_dlpack(actions.to(torch.int32))
        self.state, obs, reward, done, info = self._step(self.state, actions)
        info = dict(info)
        ended = info.pop('last_obs', None)
        info = {k: self._torch(v) for k, v in info.items()}
        if ended is not None:
            rows = (info['terminal'] != TERMINAL_RUNNING).nonzero().squeeze(1)
            info['last_obs'] = Ended(rows, {k: v.index_select(0, rows) for k, v in self._observation(ended).items()})
        return self._observation(obs), self._torch(reward), self._torch(done), info

    def step_async(self, actions):
        raise NotImplementedError('DeviceRecurrentPPO steps a DeviceEnv with step_device')

    def step_wait(self):
        raise NotImplementedError('DeviceRecurrentPPO steps a DeviceEnv with step_device')


def integer_key(key):
    """The 31-bit integer key of a reset's keyed start draws (sim/task.py draw_start) for a JAX key."""
    return jax.random.bits(jax.random.split(key, 3)[2], (), jnp.uint32) >> 1


class _IntegerKeyed:
    """A host environment (integer reset keys, numpy actions) behind SpecVecEnv's JAX keys and actions."""

    def __init__(self, env, integer_key):
        self.env, self._integer_key, self.horizon = env, integer_key, env.horizon
        self.horizons, self.targets = env.horizons, env.targets  # per goal (sim/workers.py CpuWorkers)
        self.goals = env.goals
        # the observation version and table the batch declares, and whether its steps give last_obs (SpecVecEnv)
        self.observation = getattr(env, 'observation', OBSERVATION)
        self.shapes = getattr(env, 'shapes', OBSERVATION_SHAPES)
        self.last_obs = getattr(env, 'last_obs', False)
        self.settings_in_effect = getattr(env, 'settings_in_effect', None)  # CpuWorkers' report of a free-play batch

    def reset(self, key):
        return self.env.reset(int(self._integer_key(key)))

    def step(self, state, action):
        return self.env.step(state, np.asarray(action, np.int32))


class HostDeviceEnv(DeviceEnv):
    """A host environment (sim/workers.py CpuWorkers, whose numpy arrays its next step reuses) for DeviceRecurrentPPO:
    each step's observations, reward, done flags and info are copied to the learner's device and the actions come back
    to the host. Its reset key is integer_key of the JAX key, as the removed JAX simulator drew its starts. worker_timing(): the host environment's timing() since the last call, its waiting and
    communication included."""

    def __init__(self, env, seed, device='cpu'):
        self.host = env
        super().__init__(_IntegerKeyed(env, integer_key), seed, device)
        env.timing()  # SpecVecEnv's warm-up step is startup, not a rollout's

    def _torch(self, value):
        return torch.from_numpy(np.array(value)).to(self.device)  # np.array copies: the host arrays are reused

    def step_device(self, actions):
        """DeviceEnv's step_device. A free-play batch's last_obs is read on the host: only the rows of the lives that
        ended at this step are copied to the device (Ended), not the whole table every step."""
        self.state, obs, reward, done, info = self._step(self.state, actions.to(torch.int32).cpu().numpy())
        info = dict(info)
        ended = info.pop('last_obs', None)
        out = {k: self._torch(v) for k, v in info.items()}
        if ended is not None:
            rows = np.flatnonzero(np.asarray(info['terminal']) != TERMINAL_RUNNING)
            out['last_obs'] = Ended(torch.from_numpy(rows).to(self.device),
                                    self._observation({k: np.asarray(v)[rows] for k, v in ended.items()}))
        return self._observation(obs), self._torch(reward), self._torch(done), out

    def worker_timing(self):
        return self.host.timing()

    def close(self):
        super().close()
        self.host.close()


# ---- rollout storage -----------------------------------------------------------------------------------------
# DeviceRolloutBuffer's storage tensors besides the observations (allocated once; a captured graph reads them)
BUFFER_STORAGE = ('actions', 'rewards', 'episode_starts', '_values', 'log_probs', 'advantages', '_returns', 'hidden_states_pi',
                  'cell_states_pi', 'hidden_states_vf', 'cell_states_vf')


class CompactSamples(NamedTuple):
    """A minibatch's real rows (DeviceRolloutBuffer.compact): [B] or [B, ...] tensors in SB3's order, the LSTM states at
    each sequence's start, keep (1 - its episode start) and pos (each row's place in the padded n_seq x max_length grid)."""
    observations: dict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    keep: torch.Tensor
    pos: torch.Tensor
    n_seq: int
    max_length: int


class DeviceRolloutBuffer:
    """sb3_contrib RecurrentDictRolloutBuffer as torch operations on one device, with its arithmetic and its
    sampling: get() draws SB3's split index from numpy's global generator and yields the same
    RecurrentDictRolloutBufferSamples (sequences, zero padding, masks, observation dtypes). Storage keeps
    the [n_steps, n_envs] layout; SB3's env-major flattening becomes an index map, not a copy. `values`
    and `returns` are host copies for RecurrentPPO.train()'s explained variance (once per update).
    Goal-free play (module docstring), each off by default: half_life (ticks) discounts each step by
    0.5 ** (decision_ticks / half_life) in place of gamma; free_play stores each step's bootstrap value of the lives cut at
    it (`cut_values`, zero elsewhere), which their next value is in place of the next life's, and counts episodes without
    goals."""

    def __init__(self, buffer_size, observation_space, action_space, hidden_state_shape, device='cpu', gae_lambda=1.,
                 gamma=.99, n_envs=1, *, half_life=None, free_play=False):
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError('the game policy has one Discrete(96) action')
        if half_life is not None and not (half_life > 0 and gamma == 1):
            raise ValueError(f'a per-tick discount has a positive half-life and replaces gamma (1), not {half_life}, {gamma}')
        self.buffer_size, self.n_envs, self.device = buffer_size, n_envs, torch.device(device)
        self.gae_lambda, self.gamma = gae_lambda, gamma
        self.half_life, self.free_play = half_life, free_play
        self._dtypes = {k: MODEL_DTYPES[space.dtype] for k, space in observation_space.spaces.items()}
        steps, layers, envs, hidden = hidden_state_shape
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype, device=self.device)
        self.observations = {k: zeros(steps, envs, *space.shape, dtype=STORE_DTYPES[space.dtype])
                             for k, space in observation_space.spaces.items()}
        self.actions = zeros(steps, envs, 1, dtype=torch.int64)
        self.rewards, self.episode_starts, self._values, self.log_probs, self.advantages, self._returns = (
            zeros(steps, envs) for _ in range(6))
        # [n_steps, n_envs, layers, hidden]: SB3's [n_steps, layers, n_envs, hidden] with its later swap applied.
        self.hidden_states_pi, self.cell_states_pi, self.hidden_states_vf, self.cell_states_vf = (
            zeros(steps, envs, layers, hidden) for _ in range(4))
        self.cut_values = zeros(steps, envs) if free_play else None
        self.infos = {}
        self.reset()

    def reset(self):
        # Every row is rewritten before get() can read it, so nothing is cleared.
        self.pos, self.full, self.generator_ready = 0, False, False

    def add(self, obs, action, reward, episode_start, value, log_prob, *, lstm_states, infos=(), cut_values=None):
        """cut_values: free play's bootstrap values of the step's cut lives [n_envs] (zero elsewhere); none otherwise."""
        if (cut_values is None) != (self.cut_values is None):
            raise ValueError('a free-play buffer stores every step\'s cut values, any other none')
        if cut_values is not None:
            self.cut_values[self.pos].copy_(cut_values)
        for k, v in obs.items():
            self.observations[k][self.pos].copy_(v)
        self.actions[self.pos].copy_(action.reshape(self.n_envs, 1))
        self.rewards[self.pos].copy_(reward)
        self.episode_starts[self.pos].copy_(episode_start)
        self._values[self.pos].copy_(value.flatten())
        self.log_probs[self.pos].copy_(log_prob)
        for store, state in zip((self.hidden_states_pi, self.cell_states_pi, self.hidden_states_vf, self.cell_states_vf),
                                (*lstm_states.pi, *lstm_states.vf)):
            store[self.pos].copy_(state.transpose(0, 1))
        for k, v in dict(infos).items():
            if k not in self.infos:
                self.infos[k] = torch.zeros((self.buffer_size, *v.shape), dtype=v.dtype, device=self.device)
            self.infos[k][self.pos].copy_(v)
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def compute_returns_and_advantage(self, last_values, dones):
        """SB3's GAE loop, operation for operation in float32: a done is terminal, last_values bootstrap the
        rollout boundary. With a per-tick discount or free play, the loop of goal-free play (cut_returns_and_advantage);
        without either it is this one, unchanged."""
        if self.half_life is not None or self.free_play:
            return self.cut_returns_and_advantage(last_values, dones)
        last_values = last_values.clone().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self._values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self._values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self._returns = self.advantages + self._values

    def discounts(self):
        """Each stored step's discount [n_steps, n_envs]: gamma, or 0.5 ** (decision_ticks / half_life), the per-tick
        discount 0.5 ** (1 / half_life) to the power of the step's ticks (float64, then float32)."""
        if self.half_life is None:
            return None
        ticks = self.infos.get('decision_ticks')
        if ticks is None:
            raise ValueError('a per-tick discount reads each step\'s decision_ticks (sim/spec.py INFO_FIELDS)')
        return torch.exp2(-ticks.to(torch.float64) / self.half_life).to(torch.float32)

    def cut_returns_and_advantage(self, last_values, dones):
        """The GAE loop of goal-free play: step t's discount g_t (discounts, else gamma), and its next value the next
        step's where the life goes on, the cut value where it was cut (cut_values) and zero at a death:
        delta_t = r_t + g_t (nnt_t V_t+1 + cut_t) - V_t,  A_t = delta_t + g_t lambda nnt_t A_t+1. A cut or a death still
        ends the recursion (nnt_t = 0). With g_t = gamma and no cut it is SB3's loop's arithmetic, bit for bit."""
        last_values = last_values.clone().flatten()
        discounts = self.discounts()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self._values[step + 1]
            gamma = self.gamma if discounts is None else discounts[step]
            next_values = next_values * next_non_terminal
            if self.cut_values is not None:
                next_values = next_values + self.cut_values[step]
            delta = self.rewards[step] + gamma * next_values - self._values[step]
            last_gae_lam = delta + gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self._returns = self.advantages + self._values

    def _host(self, value):
        # SB3's layout: [n_steps, n_envs] until get() flattens it env-major to [n_steps * n_envs, 1].
        value = value.detach().cpu().numpy()
        return value.swapaxes(0, 1).reshape(-1, 1) if self.generator_ready else value

    @property
    def values(self):
        return self._host(self._values)

    @property
    def returns(self):
        return self._host(self._returns)

    def _rows(self, flat):
        """Storage rows of SB3's env-major flat indices (env * n_steps + step)."""
        return (flat % self.buffer_size) * self.n_envs + flat // self.buffer_size

    def get(self, batch_size=None):
        """SB3's minibatches."""
        for rows, first, max_length in self.layout(batch_size):
            yield self._get_samples(rows, first, max_length)

    def layout(self, batch_size=None):
        """SB3's minibatch layout, one pass: per minibatch its storage rows, its sequences' first positions in it and
        the longest sequence. Draws the split index from numpy's global generator as RecurrentDictRolloutBuffer.get."""
        assert self.full, 'Rollout buffer must be full before sampling from it'
        self.generator_ready = True
        total = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = total
        split_index = np.random.randint(total)
        indices = torch.arange(total, device=self.device).roll(-split_index)
        rows = self._rows(indices)
        # create_sequencers for every minibatch at once: a sequence starts at an episode start, at each env's
        # first step and at a minibatch's first row. Two host reads a pass, not two a minibatch.
        starts = (self.episode_starts.reshape(-1)[rows] != 0) | (indices % self.buffer_size == 0)
        starts[::batch_size] = True
        first = starts.nonzero().squeeze(1)
        on_host = first.cpu().numpy()
        edges = [*range(0, total, batch_size), total]
        cuts = np.searchsorted(on_host, edges)
        for start, end, a, b in zip(edges, edges[1:], cuts, cuts[1:]):
            max_length = int(np.diff(on_host[a:b], append=end).max())
            yield rows[start:end], first[a:b] - start, max_length

    def compact(self, rows, first, max_length):
        """One minibatch of layout() without padding (DeviceRecurrentPPO.FAST_UPDATE): its real rows in SB3's order,
        which is the order of `x[mask]` over the padded samples, and where each lands in the padded [sequence, step]
        grid. Every episode start begins a sequence (layout), so a start is only ever a sequence's first step."""
        n = len(rows)
        seq = torch.searchsorted(first, torch.arange(n, device=self.device), right=True) - 1
        pos = seq * max_length + torch.arange(n, device=self.device) - first[seq]
        flat = lambda store: store.reshape(-1, *store.shape[2:])
        state = lambda store: flat(store)[rows[first]].swapaxes(0, 1).contiguous()
        return CompactSamples(
            observations={k: flat(v)[rows].to(self._dtypes[k]) for k, v in self.observations.items()},
            actions=flat(self.actions)[rows, 0], old_values=flat(self._values)[rows], old_log_prob=flat(self.log_probs)[rows],
            advantages=flat(self.advantages)[rows], returns=flat(self._returns)[rows],
            lstm_states=RNNStates((state(self.hidden_states_pi), state(self.cell_states_pi)),
                                  (state(self.hidden_states_vf), state(self.cell_states_vf))),
            keep=1.0 - flat(self.episode_starts)[rows[first]], pos=pos, n_seq=len(first), max_length=max_length)

    def _get_samples(self, rows, first, max_length):
        """One minibatch: its storage rows, its sequences' first positions in it, the longest sequence."""
        lengths = torch.diff(first, append=first.new_full((1,), len(rows)))
        offsets = torch.arange(max_length, device=self.device)
        valid = (offsets < lengths[:, None]).reshape(-1)
        padded = rows[(first[:, None] + offsets).clamp(max=len(rows) - 1).reshape(-1)]

        def pad(store):
            value = store.reshape(-1, *store.shape[2:])[padded]
            return value.masked_fill(~valid.reshape(-1, *[1] * (value.dim() - 1)), 0)

        def state(store):
            return store.reshape(-1, *store.shape[2:])[rows[first]].swapaxes(0, 1).contiguous()
        return RecurrentDictRolloutBufferSamples(
            observations={k: pad(v).to(self._dtypes[k]) for k, v in self.observations.items()},
            actions=pad(self.actions), old_values=pad(self._values), old_log_prob=pad(self.log_probs),
            advantages=pad(self.advantages), returns=pad(self._returns),
            lstm_states=RNNStates((state(self.hidden_states_pi), state(self.cell_states_pi)),
                                  (state(self.hidden_states_vf), state(self.cell_states_vf))),
            episode_starts=pad(self.episode_starts), mask=valid.to(torch.float32))

    def unsupported(self):
        """Every unsupported world-step stored so far, with its action and info; empty when there is none.
        One host read decides; the evidence is gathered only on failure."""
        flags = self.infos.get('unsupported')
        if flags is None or not bool(flags[:self.pos].any()):
            return []
        steps, worlds = flags[:self.pos].nonzero(as_tuple=True)
        return [dict(step=int(s), world=int(w), action=int(self.actions[s, w, 0]),
                     **{k: v[s, w].tolist() for k, v in self.infos.items()}) for s, w in zip(steps.tolist(), worlds.tolist())]

    def episodes(self):
        """The episodes that ended in the steps stored so far ({info field: list}), and their counts per goal; in free
        play the lives (no goal), all transitions together and the cut lives' bootstrap values."""
        infos = {k: v[:self.pos] for k, v in self.infos.items()}
        ended = infos['terminal'] != TERMINAL_RUNNING
        keys = ('terminal', 'episode_return', 'episode_decisions', 'episode_ticks', 'outcomes')
        if self.free_play:
            episodes = {k: infos[k][ended].tolist() for k in keys}
            episodes['cut_value'] = self.cut_values[:self.pos][ended].tolist()
            return episodes, dict(transitions=[int(ended.numel())], native_ticks=int(infos['decision_ticks'].sum()))
        episodes = {k: infos[k][ended].tolist() for k in ('goal', *keys)}
        return episodes, dict(transitions=torch.bincount(infos['goal'].reshape(-1), minlength=len(GOALS)).tolist(),
                              native_ticks=int(infos['decision_ticks'].sum()))


class HostReads(TorchDispatchMode):
    """Counts torch operations that read device data on the host (a transfer on a GPU) or whose output size
    depends on data (a device synchronization on a GPU), apart for code inside the policy (`inside`).
    counted(name, method) wraps a Tensor method (numpy, tolist, item, cpu) to count its calls too."""
    SYNCS = ('aten._local_scalar_dense', 'aten._unique2', 'aten.nonzero', 'aten.masked_select', 'aten.bincount')

    def __init__(self):
        super().__init__()
        self.inside = False
        self.reset()

    def reset(self):
        self.counts = {True: Counter(), False: Counter()}

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        name = func.overloadpacket._qualified_op_name.replace('::', '.')
        if name in ('aten.index', 'aten.index_put', 'aten.index_put_') and any(
                isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in args[1]):
            name += '(mask)'  # x[mask]: the kernel's own nonzero runs below this mode
        if name in self.SYNCS or name.endswith('(mask)'):
            self.counts[self.inside][name] += 1
        return func(*args, **(kwargs or {}))

    def counted(self, name, method):
        def wrapped(tensor, *args, **kwargs):
            self.counts[self.inside][name] += 1
            return method(tensor, *args, **kwargs)
        return wrapped


# ---- read-only telemetry ----------------------------------------------------------------------------------------
# Parameters of the critic by module (share_features_extractor=False, enable_critic_lstm=True); every other trainable
# parameter is the actor's.
CRITIC_MODULES = ('vf_features_extractor.', 'lstm_critic.', 'mlp_extractor.value_net.', 'value_net.')


def parameter_groups(policy):
    """{'actor': [names], 'critic': [names]} of the policy's trainable parameters, by module (CRITIC_MODULES)."""
    names = [n for n, p in policy.named_parameters() if p.requires_grad]
    critic = [n for n in names if n.startswith(CRITIC_MODULES)]
    return dict(actor=[n for n in names if n not in critic], critic=critic)


def parameter_change(policy, before):
    """L2 norm of each parameter group's change since `before` ({name: tensor}), as floats (one host read)."""
    params = dict(policy.named_parameters())
    groups = parameter_groups(policy)
    sums = torch.stack([sum(((params[n].detach() - before[n]) ** 2).sum() for n in names) for names in groups.values()])
    return dict(zip(groups, (float(v) for v in sums.sqrt().cpu())))


def _tracked_layers(policy):
    """(name, activation module) of the tracked layers: both encoder layers of the actor's and the critic's feature
    extractors and the first layer of each MLP head."""
    out = []
    for side, ex in (('actor', policy.features_extractor), ('critic', policy.vf_features_extractor)):
        out += [(f'{side}.encoder0', ex.encoder[1]), (f'{side}.encoder1', ex.encoder[3])]
    for side, net in (('actor', policy.mlp_extractor.policy_net), ('critic', policy.mlp_extractor.value_net)):
        if len(net):
            out.append((f'{side}.mlp0', net[1]))
    return out


class Tracker:
    """Forward hooks that keep the last training forward's activations of each tracked layer and the goal of each row
    (read from the extractors' inputs), and the per-update activation telemetry: per layer and goal, the share of units
    never active (ReLU) or with mean |h| below 0.01 (tanh) on that goal's training rows, and the mean |h|. Hooks only
    read: a run with a tracker updates exactly as one without. A dormant share alone is not proof of lost plasticity."""

    def __init__(self, policy):
        self.policy, self.layers = policy, _tracked_layers(policy)
        self.last, self.goals, self.handles, self.acc = {}, {}, [], {}
        for side, ex in (('actor', policy.features_extractor), ('critic', policy.vf_features_extractor)):
            self.handles.append(ex.register_forward_pre_hook(self._goal(side)))
        for name, act in self.layers:
            self.handles.append(act.register_forward_hook(self._keep(name)))

    def _goal(self, side):
        def hook(module, inputs):  # a padded row (all zero) has no goal: -1
            if module.training:
                g = inputs[0]['goal_vec']
                self.goals[side] = torch.where(g.sum(-1) > 0, g.argmax(-1), -1)
        return hook

    def _keep(self, name):
        def hook(module, inputs, output):
            if module.training:
                self.last[name] = output.detach()
        return hook

    def accumulate(self):
        """After an optimizer step: add the step's activations to the update's per-goal sums. A padded row (goal -1)
        adds zeros to goal 0 rather than being selected out: a selection's or a bincount's size depends on the data, so
        it waits for the device, and a graphed update runs this after every replay (12 selections and 6 bincounts a
        step). Adding +0.0 changes no sum, so the sums are the selection's bit for bit wherever index_add_ is ordered
        (the CPU)."""
        for name, _ in self.layers:
            h, goal = self.last.get(name), self.goals.get(name.split('.')[0])
            if h is None or goal is None or h.shape[0] != goal.shape[0]:
                continue
            keep = (goal >= 0).float()
            goal = goal.clamp(min=0)
            a = h.abs() * keep.unsqueeze(1)
            sums = torch.zeros(len(GOALS), a.shape[1], device=a.device).index_add_(0, goal, a)
            active = torch.zeros(len(GOALS), a.shape[1], device=a.device).index_add_(0, goal, (a > 0).float())
            rows = torch.zeros(len(GOALS), device=a.device).index_add_(0, goal, keep)
            if name not in self.acc:
                self.acc[name] = [sums, active, rows]
            else:
                for k, v in enumerate((sums, active, rows)):
                    self.acc[name][k] += v

    def report(self):
        """The update's telemetry; clears the sums."""
        out = {}
        for name, act in self.layers:
            if name not in self.acc:
                continue
            sums, active, rows = self.acc[name]
            tanh = isinstance(act, torch.nn.Tanh)
            mean = sums / rows.clamp(min=1).unsqueeze(1)
            dormant = (mean < 0.01) if tanh else (active == 0)
            out[name] = dict(units=int(sums.shape[1]), rows=rows.long().tolist(),
                             dormant_share=[round(float(d.float().mean()), 4) if r > 0 else None for d, r in zip(dormant, rows)],
                             mean_abs=[round(float(m.mean()), 5) if r > 0 else None for m, r in zip(mean, rows)])
        self.acc = {}
        return out


# ---- the update as captured CUDA graphs ------------------------------------------------------------------------
class GraphedSteps:
    """fast_train's minibatch step (DeviceRecurrentPPO._minibatch_step) as CUDA graphs, one per padded shape (a bucket
    of sequences x steps; GRAPHED_UPDATE). A graph allows no synchronization and no data-dependent shape, so its step
    reads the minibatch from the buffer's storage through static row indices, pads the LSTM grid to the bucket (extra
    zero steps after each sequence and zero sequences with zero states, whose rows no loss reads), projects the whole
    name table (NameEmbedding.FULL_TABLE; with NameEmbedding.MERGED's default each embedding then looks up its inputs at
    once) and steps a capturable Adam. Each change is equal in exact arithmetic, none bitwise. Before a capture the step
    runs twice on a side stream (cuDNN, Adam's state); the parameters and Adam's state are then put back exactly. A graph
    holds its constants (learning rate, clip range, coefficients), the addresses of the parameters, Adam's state and the
    buffer, and its lookups: prepare() drops every graph when one changed (a loaded optimizer state, a new schedule
    value, NameEmbedding.MERGED). Telemetry (Tracker) accumulates after each replay, outside the graph, from the
    activations that graph's capture left the hooks pointing at, with the optimizer's other step post-hooks. Nothing
    here moves a parameter (LSTM.flatten_parameters on CUDA does, which left the collector's graph reading old
    weights)."""
    SEQUENCES = (16, 32, 64, 128, 256)
    STEPS = (16, 32, 48, 64)

    def __init__(self, model):
        self.model = model
        b, device = model.rollout_buffer, model.device
        self.batch = model.batch_size
        if (b.buffer_size * b.n_envs) % self.batch:
            raise ValueError('GRAPHED_UPDATE needs minibatches of equal size')
        self.steps = tuple(t for t in self.STEPS if t < b.buffer_size) + (b.buffer_size,)
        self.sequences = tuple(n for n in self.SEQUENCES if n < self.batch) + (self.batch,)
        self.rows = torch.zeros(self.batch, dtype=torch.long, device=device)
        self.first = torch.full((self.batch,), self.batch, dtype=torch.long, device=device)
        self.stats = torch.zeros(6, device=device)
        self.graphs, self.hooked, self.signature = {}, {}, None

    def prepare(self, clip_range):
        """Adam capturable with its step counters on the device; every graph dropped if its constants or addresses
        changed since the captures."""
        model, device = self.model, self.model.device
        if device.type != 'cuda':
            raise ValueError('GRAPHED_UPDATE captures CUDA graphs')
        optimizer = model.policy.optimizer
        for group in optimizer.param_groups:
            group['capturable'] = True
        for state in optimizer.state.values():
            if 'step' in state and state['step'].device != device:
                state['step'] = state['step'].to(device)
        b = model.rollout_buffer
        tensors = [p for group in optimizer.param_groups for p in group['params']]
        tensors += [v for state in optimizer.state.values() for v in state.values()]
        tensors += [*b.observations.values(), *(getattr(b, k) for k in BUFFER_STORAGE)]
        from baseline.policy import NameEmbedding
        signature = (clip_range, optimizer.param_groups[0]['lr'], model.ent_coef, model.vf_coef, model.max_grad_norm,
                     NameEmbedding.MERGED, tuple(t.data_ptr() for t in tensors))
        if signature != self.signature:
            self.graphs, self.hooked, self.signature = {}, {}, signature

    def step(self, rows, first, max_length):
        """One minibatch of DeviceRolloutBuffer.layout: copy its indices into the static inputs and replay its bucket's
        graph (captured on first use). Returns a copy of the logged values."""
        shape = (next(n for n in self.sequences if n >= len(first)), next(t for t in self.steps if t >= max_length))
        self.rows.copy_(rows)
        self.first[:len(first)].copy_(first)
        self.first[len(first):] = self.batch
        graph = self.graphs.get(shape)
        if graph is None:
            graph = self.graphs[shape] = self._capture(*shape)
        graph.replay()
        tracker = self.model.tracker
        if tracker is not None:
            tracker.last, tracker.goals = (dict(x) for x in self.hooked[shape])
        run_step_hooks(self.model.policy.optimizer)
        return self.stats.clone()

    def _body(self, n_seq, max_length):
        model, b = self.model, self.model.rollout_buffer
        rows, first, n = self.rows, self.first[:n_seq], self.batch
        flat = lambda store: store.reshape(-1, *store.shape[2:])
        at = torch.arange(n, device=rows.device)
        seq = torch.searchsorted(first, at, right=True) - 1
        starts = rows[first.clamp(max=n - 1)]
        state = lambda store: flat(store)[starts].swapaxes(0, 1).contiguous()
        data = CompactSamples(
            observations={k: flat(v)[rows].to(b._dtypes[k]) for k, v in b.observations.items()},
            actions=flat(b.actions)[rows, 0], old_values=flat(b._values)[rows], old_log_prob=flat(b.log_probs)[rows],
            advantages=flat(b.advantages)[rows], returns=flat(b._returns)[rows],
            lstm_states=RNNStates((state(b.hidden_states_pi), state(b.cell_states_pi)),
                                  (state(b.hidden_states_vf), state(b.cell_states_vf))),
            keep=(1.0 - flat(b.episode_starts)[starts]) * (first < n), pos=seq * max_length + at - first[seq],
            n_seq=n_seq, max_length=max_length)
        self.stats.copy_(model._minibatch_step(data, self.signature[0]))

    def _capture(self, n_seq, max_length):
        from baseline.policy import NameEmbedding
        model, policy = self.model, self.model.policy
        optimizer = policy.optimizer
        params = [p for group in optimizer.param_groups for p in group['params']]
        saved = [p.detach().clone() for p in params]
        state = {p: {k: v.clone() for k, v in optimizer.state[p].items()} for p in params if optimizer.state.get(p)}
        full_table, NameEmbedding.FULL_TABLE = NameEmbedding.FULL_TABLE, True
        try:
            with step_hooks_suspended(optimizer):  # the warm-up and the capture count no step and add no telemetry
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._body(n_seq, max_length)
                torch.cuda.current_stream().wait_stream(side)
                with torch.no_grad():
                    for p, value in zip(params, saved):
                        p.copy_(value)
                    for p in params:
                        for k, v in optimizer.state[p].items():
                            v.copy_(state[p][k]) if p in state else v.zero_()
                optimizer.zero_grad(set_to_none=True)
                graph = torch.cuda.CUDAGraph()
                # an eager backward before (a stock update, a replay) leaves the parameters' gradient nodes on the default
                # stream, which a capture refuses; this redirects them to the capturing stream, for the capture only
                torch.autograd.graph.set_override_stale_capture_stream(True)
                try:
                    with torch.cuda.graph(graph):
                        self._body(n_seq, max_length)
                finally:
                    torch.autograd.graph.set_override_stale_capture_stream(False)
        finally:
            NameEmbedding.FULL_TABLE = full_table
        if model.tracker is not None:  # the forward hooks' tensors of this capture: the graph rewrites them at every replay
            self.hooked[(n_seq, max_length)] = (dict(model.tracker.last), dict(model.tracker.goals))
        return graph


class GraphedForward:
    """The collector's policy call (MaskedRecurrentPolicy.forward: features, both LSTMs, heads, the masked distribution,
    a sampled action and its log-probability) as one CUDA graph over static inputs (GRAPHED_COLLECTION). A graph allows no
    synchronization: the name table is projected whole (NameEmbedding.FULL_TABLE), the mask checks of _masked run before
    each replay on the step's mask (check_rows), and the distribution is torch Categorical's arithmetic with ATen's
    one-sample multinomial (an exponential draw per logit, the argmax of probability over draw) without its host-read
    argument checks; that draw equals Categorical.sample bitwise on the CPU for the same generator state, and a graph
    advances the CUDA generator's offset at every replay as the eager calls do. Equal in exact arithmetic to the eager
    call, not bitwise (the projection). The graph holds the parameters' addresses: it is captured again when one moved
    (LSTM.flatten_parameters on CUDA moves the weights)."""

    def __init__(self, policy):
        self.policy, self.graph, self.addresses = policy, None, None

    def __call__(self, obs, lstm_states, episode_starts):
        addresses = tuple(p.data_ptr() for p in self.policy.parameters())
        if self.graph is None or addresses != self.addresses:  # a graph reads the memory it was captured with
            self._capture(obs, lstm_states, episode_starts)
            self.addresses = addresses
        self.policy.check_rows(obs['mask'])
        for k, v in obs.items():
            self.obs[k].copy_(v)
        for static, value in zip(self.states, (*lstm_states.pi, *lstm_states.vf)):
            static.copy_(value)
        self.starts.copy_(episode_starts)
        self.graph.replay()
        actions, values, log_prob = (t.clone() for t in self.out[:3])
        h_pi, c_pi, h_vf, c_vf = (t.clone() for t in self.out[3])
        return actions, values, log_prob, RNNStates((h_pi, c_pi), (h_vf, c_vf))

    def _forward(self):
        """RecurrentActorCriticPolicy.forward with MaskedRecurrentPolicy's heads, operation for operation."""
        policy, obs = self.policy, self.obs
        policy._active_mask, policy._active_slots, policy._late_starts = obs['mask'].bool(), obs, False
        shared = {}
        for embedding in policy._name_embeddings:
            embedding.shared = shared
        try:
            pi_features, vf_features = policy.extract_features(obs)
            latent_pi, (h_pi, c_pi) = policy._process_sequence(pi_features, tuple(self.states[:2]), self.starts, policy.lstm_actor)
            latent_vf, (h_vf, c_vf) = policy._process_sequence(vf_features, tuple(self.states[2:]), self.starts, policy.lstm_critic)
            latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
            latent_vf = policy.mlp_extractor.forward_critic(latent_vf)
            values = policy.value_net(latent_vf)
            logits = policy._masked_logits(latent_pi)
        finally:
            policy._active_mask = policy._active_slots = policy._late_starts = None
            for embedding in policy._name_embeddings:
                embedding.shared = None
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)  # Categorical(logits=...)
        probs = logits.softmax(-1)
        actions = torch.argmax(probs / torch.empty_like(probs).exponential_(1), dim=-1)  # multinomial(probs, 1)
        log_prob = logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, values, log_prob, (h_pi, c_pi, h_vf, c_vf)

    def _capture(self, obs, lstm_states, episode_starts):
        from baseline.policy import NameEmbedding
        self.obs = {k: v.clone() for k, v in obs.items()}
        self.states = [t.clone() for t in (*lstm_states.pi, *lstm_states.vf)]
        self.starts = episode_starts.clone()
        full_table, NameEmbedding.FULL_TABLE = NameEmbedding.FULL_TABLE, True
        try:
            with torch.no_grad():
                generator = torch.cuda.get_rng_state()  # the warm-up's draws are not the run's: put the stream back
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._forward()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.set_rng_state(generator)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = self._forward()
        finally:
            NameEmbedding.FULL_TABLE = full_table


# ---- the algorithm --------------------------------------------------------------------------------------------
class DeviceRecurrentPPO(RecurrentPPO):
    """RecurrentPPO with DeviceRolloutBuffer and a collector for DeviceEnv; learn() and train() are SB3's.
    Saved zips are ordinary RecurrentPPO checkpoints: the device state below is not pickled with them.
    evidence_dir: where Unsupported writes its evidence (None keeps it on the exception only).
    rollouts: one record per collected rollout (ended episodes, per-goal transitions, native ticks).
    step_timing: None, or a dict to which each rollout adds the seconds of its parts, each part ended by a
    device synchronization (CUDA; JAX's work on the GPU included): policy (inference), environment (the
    JAX step and the DLPack handover), callbacks, buffer (storage and bookkeeping) and rollout_end (the
    unsupported check, the last values, returns and advantages, the rollout record); in free play also bootstrap (the
    values of the lives cut at each step).
    Goal-free play's settings (module docstring), keyword arguments that are off by default and, off, are not saved
    (class attributes): free_play and half_life_ticks; value_norm is the policy's (baseline/policy.py PopArt), whose
    statistics each update moves towards its rollout's value targets before the gradient steps (fast_train only)."""
    evidence_dir = None
    free_play = False      # the goal-free task: a cut life bootstraps from its last observation, a death does not
    half_life_ticks = None  # the per-tick discount's half-life in game ticks, in place of gamma
    # fast_train in place of RecurrentPPO.train(): the same update without host reads inside it (a class switch like
    # baseline.policy's speed paths, never saved with a checkpoint)
    FAST_UPDATE = True
    # with FAST_UPDATE on CUDA: each minibatch step a captured CUDA graph (GraphedSteps; equal in exact arithmetic)
    GRAPHED_UPDATE = False
    # on CUDA: the collector's policy call a captured CUDA graph (GraphedForward; equal in exact arithmetic)
    GRAPHED_COLLECTION = False
    step_timing = None
    tracker = None  # the read-only activation telemetry (Tracker, attach)
    update_sizes = None  # with telemetry: per update, each parameter group's L2 change (parameter_change)

    def __init__(self, *args, free_play=False, half_life_ticks=None, **kwargs):
        if free_play:
            self.free_play = True
        if half_life_ticks is not None:
            self.half_life_ticks = half_life_ticks
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        lstm = self.policy.lstm_actor
        self.rollout_buffer = DeviceRolloutBuffer(
            self.n_steps, self.observation_space, self.action_space,
            (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size), self.device, gamma=self.gamma,
            gae_lambda=self.gae_lambda, n_envs=self.n_envs, half_life=self.half_life_ticks, free_play=self.free_play)
        self.rollouts = []

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), '_last_obs', '_last_episode_starts', 'evidence_dir', 'rollouts',
                'step_timing', 'tracker', 'update_sizes', '_graphs', '_forward_graph']

    def get_parameters(self):
        """SB3's, with a capturable Adam (GraphedSteps) saved as the stock one: capturable off and its step counters on
        the host, so a checkpoint loads into any run as before."""
        params = super().get_parameters()
        optimizer = params.get('policy.optimizer')
        if optimizer and any(g.get('capturable') for g in optimizer['param_groups']):
            optimizer['param_groups'] = [dict(g, capturable=False) for g in optimizer['param_groups']]
            optimizer['state'] = {i: {k: v.cpu() if k == 'step' else v for k, v in state.items()}
                                  for i, state in optimizer['state'].items()}
        return params

    def train(self):
        """SB3's RecurrentPPO.train() (with FAST_UPDATE, fast_train: the same update). With update_sizes, the update's
        parameter change is recorded (read-only)."""
        popart = self._popart()
        if popart is not None:  # the value targets' statistics, the outputs kept: before the update that fits them
            if not self.FAST_UPDATE:
                raise ValueError('a normalized value target trains with fast_train (FAST_UPDATE)')
            popart.update(self.rollout_buffer._returns)
        before = None if self.update_sizes is None else \
            {n: p.detach().clone() for n, p in self.policy.named_parameters() if p.requires_grad}
        result = self.fast_train() if self.FAST_UPDATE else super().train()
        if before is not None:
            self.update_sizes.append(parameter_change(self.policy, before))
        return result

    def fast_train(self):
        """RecurrentPPO.train() without a host read inside the minibatch loop: the same minibatches (DeviceRolloutBuffer
        layout, the same split-index draws), the same losses, gradient clipping and optimizer steps, evaluated on each
        minibatch's real rows (MaskedRecurrentPolicy.evaluate_compact) instead of the padded rows the loss then masks.
        `x[mask]` over the padded samples is the real rows in this order, so every mean, the advantage normalization and
        every gradient is the same arithmetic. Each minibatch's logged values stay on the device and are read once, as
        the policy's per-call checks are made once (check_rows). Settings it does not cover raise: target_kl's early
        stop needs a read each minibatch, and the KL penalty and a value clip are not used by simulator training.
        With GRAPHED_UPDATE (CUDA) each minibatch replays a captured step (GraphedSteps)."""
        if self.target_kl is not None or self.clip_range_vf is not None or self.policy.kl_weight \
                or not self.normalize_advantage or not isinstance(self.action_space, spaces.Discrete):
            raise ValueError('FAST_UPDATE covers no target_kl, value clipping or KL penalty, and normalized advantages')
        policy, buffer = self.policy, self.rollout_buffer
        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        policy.check_rows(buffer.observations['mask'])
        graphed = None
        if self.GRAPHED_UPDATE:
            graphed = self._graphs = getattr(self, '_graphs', None) or GraphedSteps(self)
            graphed.prepare(clip_range)
        logged = []
        for _ in range(self.n_epochs):
            for layout in buffer.layout(self.batch_size):
                if graphed is not None:
                    logged.append(graphed.step(*layout))
                    continue
                stats = self._minibatch_step(buffer.compact(*layout), clip_range)
                logged.append(stats)
            self._n_updates += 1
        # The one read: SB3 takes each value with .item() (a float32 widened to a Python float) and averages them as
        # float64; its approximate KLs are float32 numpy scalars of the last epoch, averaged as float32.
        logged = torch.stack(logged).cpu().numpy()
        pg, clip, value, entropy = logged[:, :4].astype(np.float64).T
        kl = np.ascontiguousarray(logged[-len(logged) // self.n_epochs:, 4])
        if graphed is not None and not np.isfinite(logged).all():
            raise ValueError('a non-finite loss in the update (the graphed steps skip torch.distributions\' checks)')
        explained_var = explained_variance(buffer.values.flatten(), buffer.returns.flatten())
        self.logger.record('train/entropy_loss', np.mean(entropy))
        self.logger.record('train/policy_gradient_loss', np.mean(pg))
        self.logger.record('train/value_loss', np.mean(value))
        self.logger.record('train/approx_kl', np.mean(kl))
        self.logger.record('train/clip_fraction', np.mean(clip))
        self.logger.record('train/loss', float(logged[-1, 5]))
        self.logger.record('train/explained_variance', explained_var)
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/clip_range', clip_range)
        popart = self._popart()
        if popart is not None:
            self.logger.record('train/value_target_mean', float(popart.mean))
            self.logger.record('train/value_target_std', float(popart.std))

    def _popart(self):
        from baseline.policy import PopArt
        head = self.policy.value_net
        return head if isinstance(head, PopArt) else None

    def _minibatch_step(self, data, clip_range):
        """RecurrentPPO.train()'s loss, backward, clip and optimizer step on one compact minibatch (CompactSamples).
        Returns its logged values on the device: policy, clip fraction, value, entropy, approximate KL, loss."""
        policy = self.policy
        values, log_prob, entropy = policy.evaluate_compact(
            data.observations, data.actions, data.lstm_states, data.keep, data.pos, data.n_seq, data.max_length)
        advantages = data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(log_prob - data.old_log_prob)
        policy_loss = -torch.mean(torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)))
        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float())
        error = data.returns - values
        popart = self._popart()
        if popart is not None:  # the normalized target's error, (G - mean) / std - n(h), is the error over std
            error = error / popart.std
        value_loss = torch.mean(error ** 2)
        entropy_loss = -torch.mean(entropy)
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        with torch.no_grad():
            log_ratio = log_prob - data.old_log_prob
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
        policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        policy.optimizer.step()
        return torch.stack([policy_loss, clip_fraction, value_loss, entropy_loss, approx_kl, loss]).detach()

    def set_parameters(self, load_path_or_dict, exact_match=True, device='auto'):
        """SB3's, with Adam's step counters back on the host, where torch keeps them unless Adam is capturable
        or fused. SB3 loads a checkpoint's tensors onto the model's device and torch keeps a loaded counter
        there; foreach Adam then read its 70 counters twice from the GPU every optimizer step. Each
        counter is a small whole number in float32, the same on either device, so no update changes."""
        super().set_parameters(load_path_or_dict, exact_match=exact_match, device=device)
        optimizer = self.policy.optimizer
        for group in optimizer.param_groups:
            if group.get('capturable') or group.get('fused'):
                continue
            for p in group['params']:
                state = optimizer.state.get(p, {})
                if 'step' in state:
                    state['step'] = state['step'].cpu()

    def _mark(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """RecurrentPPO.collect_rollouts with the environment, the policy and the buffer on one device. An asynchronous
        environment (sim/real.py RealEnv: agents that decide when they are due) collects the rollout itself."""
        if getattr(env, 'asynchronous', False):
            return env.collect(self, callback, rollout_buffer, n_rollout_steps)
        assert isinstance(env, DeviceEnv) and isinstance(rollout_buffer, DeviceRolloutBuffer) and not self.use_sde
        if self.free_play != env.last_obs:  # the free-play batch and the free-play learner come together
            raise ValueError(f'a free-play learner (here {self.free_play}) bootstraps a cut life from its last observation, '
                             f'which only the free-play batch gives (here {env.last_obs})')
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        if not torch.is_tensor(self._last_episode_starts):  # SB3's _setup_learn starts every world anew
            self._last_episode_starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
        lstm_states = deepcopy(self._last_lstm_states)
        start = self.num_timesteps
        timing = self.step_timing
        if timing is not None:
            for part in ('policy', 'environment', 'callbacks', 'buffer', 'rollout_end') + (('bootstrap',) if self.free_play else ()):
                timing.setdefault(part, 0.)
            t = self._mark()

        def part(name):  # with step_timing: the synchronized seconds since the last part
            nonlocal t
            if timing is not None:
                now = self._mark()
                timing[name] += now - t
                t = now
        try:
            forward = self.policy
            if self.GRAPHED_COLLECTION:
                if self.device.type != 'cuda':
                    raise ValueError('GRAPHED_COLLECTION captures a CUDA graph')
                forward = self._forward_graph = getattr(self, '_forward_graph', None) or GraphedForward(self.policy)
            while n_steps < n_rollout_steps:
                with torch.no_grad():
                    actions, values, log_probs, lstm_states = forward(self._last_obs, lstm_states, self._last_episode_starts)
                part('policy')
                new_obs, rewards, dones, infos = env.step_device(actions)
                part('environment')
                ended = infos.pop('last_obs', None)
                cut_values = None
                if self.free_play:
                    cut_values = self._cut_values(ended, infos['terminal'], lstm_states)
                    part('bootstrap')
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                part('callbacks')
                n_steps += 1
                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                                   lstm_states=self._last_lstm_states, infos=infos, cut_values=cut_values)
                self._last_obs = new_obs
                self._last_episode_starts = dones.to(torch.float32)
                self._last_lstm_states = lstm_states
                part('buffer')
        except Exception as error:
            # A meaningless observation can fail the policy's own checks before the rollout ends.
            self._check_supported(rollout_buffer, start, error)
            raise
        self._check_supported(rollout_buffer, start)
        with torch.no_grad():
            values = self.policy.predict_values(new_obs, lstm_states.vf, self._last_episode_starts)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._record_rollout(rollout_buffer, start)
        part('rollout_end')
        callback.on_rollout_end()
        return True

    def _cut_values(self, ended, terminal, lstm_states):
        """Free play: [n_envs] the value of each cut life's last observation (TERMINAL_HORIZON), zero for every other
        world, a death included. The value is the critic's from the life's own recurrent state, the one its last
        action's step produced (lstm_states, before the next step's reset), with no episode start: SB3 RecurrentPPO's
        time-limit bootstrap. One policy call on the ended rows; a step where no life ended makes none."""
        values = torch.zeros(self.n_envs, device=self.device)
        rows, obs = ended
        if len(rows):
            h, c = lstm_states.vf
            state = (h.index_select(1, rows).contiguous(), c.index_select(1, rows).contiguous())
            with torch.no_grad():
                value = self.policy.predict_values(obs, state, torch.zeros(len(rows), device=self.device)).flatten()
            values[rows] = torch.where(terminal.index_select(0, rows) == TERMINAL_HORIZON, value, torch.zeros_like(value))
        return values

    def _check_supported(self, buffer, start, error=None):
        evidence = buffer.unsupported()
        if not evidence:
            return
        for row in evidence:
            row['decision'] = start // self.n_envs + row['step']  # this world's decision index in the run
        record = dict(num_timesteps_at_rollout_start=start, n_envs=self.n_envs, rows=evidence,
                      error=None if error is None else f'{type(error).__name__}: {error}',
                      worker_evidence=list(getattr(getattr(self.env, 'host', None), 'evidence', None) or []))
        path = None
        if self.evidence_dir is not None:
            path = Path(self.evidence_dir) / f'unsupported-{start}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
        worlds = sorted({row['world'] for row in evidence})
        raise Unsupported(f'{len(evidence)} unsupported world-steps (worlds {worlds}) in the rollout from decision '
                          f'{start}: no update uses it; evidence {path or "on this exception"}', record, path) from error

    def _record_rollout(self, buffer, start):
        episodes, counts = buffer.episodes()
        self.rollouts.append(dict(num_timesteps=start, **counts, episodes=episodes))
        for ret, length in zip(episodes['episode_return'], episodes['episode_decisions']):
            self.ep_info_buffer.append(dict(r=ret, l=length))
        self.logger.record('rollout/native_ticks', counts['native_ticks'])
        if self.free_play:  # lives, no goals: their ends by reason and their pay
            ends = Counter(episodes['terminal'])
            self.logger.record('rollout/lives', len(episodes['terminal']))
            self.logger.record('rollout/deaths', ends[TERMINAL_DEATH])
            self.logger.record('rollout/cut_lives', ends[TERMINAL_HORIZON])
            if episodes['episode_return']:
                self.logger.record('rollout/life_return', float(np.mean(episodes['episode_return'])))
            return
        for index, goal in enumerate(GOALS):
            mine = [i for i, g in enumerate(episodes['goal']) if g == index]
            self.logger.record(f'rollout/{goal}_transitions', counts['transitions'][index])
            self.logger.record(f'rollout/{goal}_episodes', len(mine))
            if mine:
                self.logger.record(f'rollout/{goal}_return', float(np.mean([episodes['episode_return'][i] for i in mine])))
                self.logger.record(f'rollout/{goal}_deaths',
                                   Counter(episodes['terminal'][i] for i in mine)[TERMINAL_DEATH])


def attach(model, *, telemetry=True):
    """Read-only telemetry (Tracker, with each update's parameter change). An optimizer-step hook accumulates the
    telemetry after each of the policy optimizer's steps."""
    model.tracker = Tracker(model.policy) if telemetry else None
    model.update_sizes = [] if telemetry else None

    def stepped(optimizer, args, kwargs):
        if model.tracker is not None:
            model.tracker.accumulate()
    model.policy.optimizer.register_step_post_hook(stepped)
    return model


@contextmanager
def step_hooks_suspended(optimizer):
    """The optimizer's step post-hooks (the run's step count, telemetry, a replay's snapshots: register_step_post_hook)
    off for the steps inside a CUDA graph's capture (GraphedSteps, which runs them after each replay instead,
    run_step_hooks)."""
    hooks, optimizer._optimizer_step_post_hooks = optimizer._optimizer_step_post_hooks, OrderedDict()
    try:
        yield
    finally:
        optimizer._optimizer_step_post_hooks = hooks


def run_step_hooks(optimizer):
    for hook in list(optimizer._optimizer_step_post_hooks.values()):
        hook(optimizer, (), {})


def build(env, *, seed, device, free_play=False, half_life_ticks=None, value_norm=None, **changes):
    """baseline/train.build_model's fresh model (config.LEARNER, names-v1) as a DeviceRecurrentPPO, reading the
    observation version and table that `env` declares (obs-v5 for the goal tasks).
    changes: explicit departures from config.LEARNER (the GPU batching), reported by the caller. free_play, half_life_ticks
    and value_norm (baseline/policy.py POPART): goal-free play's settings (DeviceRecurrentPPO), off by default."""
    from baseline.policy import MaskedRecurrentPolicy, load_name_vectors, run_contract
    from item_contract import ITEM_V2
    from names import name_table
    unknown = set(changes) - set(config.LEARNER)
    if unknown:
        raise ValueError(f'not a config.LEARNER setting: {sorted(unknown)}')
    learner = deepcopy(config.LEARNER)
    learner.update(changes)
    table = name_table()
    # the contract the policy records: item-v2's CONTRACT, or with the item contract of an env that declares another
    learner['policy_kwargs'].update(names=table.spec, observation=env.observation,
                                    contract=run_contract(getattr(env, 'item_contract', ITEM_V2)))
    if value_norm is not None:
        learner['policy_kwargs']['value_norm'] = dict(value_norm)
    model = DeviceRecurrentPPO(MaskedRecurrentPolicy, env, seed=seed, device=device, free_play=free_play,
                               half_life_ticks=half_life_ticks, **learner)
    load_name_vectors(model.policy, table.vectors)
    return model
