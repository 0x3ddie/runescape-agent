"""Bridge the existing vector transport contract to SB3 without another reset.

The injected backend must already implement reset/step/close as RSVecEnv does.
This bridge does not certify its task reward, clock or live-world configuration.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


def observation_space(sample):
    result = {}
    for key, value in sample.items():
        value = np.asarray(value)
        if key == 'mask':  # one entry for each action row: act-v3's 96, act-v4's 128
            result[key] = spaces.Box(0, 1, shape=value.shape, dtype=np.int8)
        elif key.endswith(('_ids', '_op')):  # rows of the name table, hashed options
            result[key] = spaces.Box(0, 4000, shape=value.shape, dtype=np.int64)
        else:
            result[key] = spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)
    return spaces.Dict(result)


class GameVecEnv(VecEnv):
    render_mode = None

    def __init__(self, backend, sample):
        self.backend = backend
        self.pending = None
        self._obs = None
        self._failed = False
        super().__init__(backend.n_envs, observation_space(sample), spaces.Discrete(96))

    def _validate(self, obs, *, batched=True):
        if set(obs) != set(self.observation_space.spaces):
            raise ValueError('Observation keys differ from the schema')
        for key, space in self.observation_space.spaces.items():
            value = np.asarray(obs[key])
            shape = (self.num_envs, *space.shape) if batched else space.shape
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid observation shape/value: ' + key)
            if key.endswith('_ids') and (np.any(value != value.astype(np.int64)) or
                                         np.any(value < 0) or np.any(value > 4000)):
                raise ValueError('Invalid entity ID')
        mask = np.asarray(obs['mask'])
        if not np.isin(mask, [0, 1]).all() or not mask.any(axis=-1).all():
            raise ValueError('Invalid/all-empty action mask')
        return {key: np.array(value, dtype=self.observation_space[key].dtype, copy=True)
                for key, value in obs.items()}

    def reset(self):
        if self._failed or self.pending is not None:
            raise RuntimeError('Cannot reset failed or pending environment')
        if any(self._options):
            raise ValueError('Reset options unsupported; pin backend configuration')
        seeds = self._seeds
        if seeds[0] is not None and seeds != [seeds[0] + i for i in range(self.num_envs)]:
            raise ValueError('Backend requires consecutive initial seeds')
        try:
            self._obs = self._validate(self.backend.reset(seed=seeds[0]))
        except Exception:
            self._failed = True
            self.backend.close()
            raise
        self._reset_seeds()
        self._reset_options()
        return self._obs

    def step_async(self, actions):
        if self._failed or self.pending is not None or self._obs is None:
            raise RuntimeError('Invalid step lifecycle')
        actions = np.asarray(actions)
        if (actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer)
                or np.any(actions < 0) or np.any(actions >= 96)):
            raise ValueError('Invalid actions')
        if not self._obs['mask'][np.arange(self.num_envs), actions].all():
            raise ValueError('Masked action requested')
        self.pending = actions.copy()

    def step_wait(self):
        if self.pending is None or self._failed:
            raise RuntimeError('No pending valid step')
        actions, self.pending = self.pending, None
        try:
            obs, rewards, terminated, truncated, infos = self.backend.step(actions)
            rewards = np.asarray(rewards)
            terminated, truncated = np.asarray(terminated), np.asarray(truncated)
            if (rewards.shape != (self.num_envs,) or not np.isfinite(rewards).all()
                    or terminated.shape != rewards.shape or truncated.shape != rewards.shape
                    or terminated.dtype != np.bool_ or truncated.dtype != np.bool_
                    or len(infos) != self.num_envs):
                raise ValueError('Malformed vector transition')
            dones = terminated | truncated
            output = []
            for i, original in enumerate(infos):
                info = dict(original)
                info['TimeLimit.truncated'] = bool(truncated[i] and not terminated[i])
                if dones[i]:
                    if 'final_observation' not in info:
                        raise ValueError('Auto-reset lost final observation')
                    info['terminal_observation'] = self._validate(info.pop('final_observation'), batched=False)
                    if 'episode' in info.get('final_info', {}):
                        info['episode'] = dict(info['final_info']['episode'])
                output.append(info)
            self._obs = self._validate(obs)
            return self._obs, rewards.astype(np.float32), dones, output
        except Exception:
            self._failed = True
            self.backend.close()
            raise

    def close(self):
        self.backend.close()

    def get_attr(self, attr_name, indices=None):
        if attr_name != 'render_mode':
            raise AttributeError(attr_name)
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError('Mutating game configuration is unsupported')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('Use the explicit backend protocol')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
