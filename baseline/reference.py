"""Attach a frozen reference policy's action probabilities to every observation.

The KL penalty in baseline/policy.py compares the learner against them, as VPT
kept RL fine-tuning close to its imitation policy. The reference keeps its own
recurrent state, reset whenever an episode starts. Until a reference is set the
probabilities are zero and the penalty has nothing to act on.
"""
import copy

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

KEY = 'ref_probs'


class ReferenceProbs(VecEnvWrapper):
    def __init__(self, venv):
        space = spaces.Dict({**venv.observation_space.spaces,
                             KEY: spaces.Box(0., 1., shape=(venv.action_space.n,), dtype=np.float32)})
        super().__init__(venv, observation_space=space)
        self.policy = None
        self.state = None

    def set_reference(self, policy):
        self.policy = copy.deepcopy(policy)
        self.policy.set_training_mode(False)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        for module in self.policy.modules():
            if isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()  # a deep copy leaves cuDNN weights fragmented
        self.state = None

    def _attach(self, obs, starts):
        probs = np.zeros((self.num_envs, self.action_space.n), np.float32)
        if self.policy is not None:
            device = self.policy.device
            if self.state is None:
                lstm = self.policy.lstm_actor
                shape = (lstm.num_layers, self.num_envs, lstm.hidden_size)
                self.state = (torch.zeros(shape, device=device), torch.zeros(shape, device=device))
            tensors = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
            with torch.no_grad():
                distribution, self.state = self.policy.get_distribution(
                    tensors, self.state, torch.as_tensor(starts, dtype=torch.float32, device=device))
            probs = distribution.distribution.probs.cpu().numpy().astype(np.float32)
        return {**obs, KEY: probs}

    def reset(self):
        obs = self.venv.reset()
        self.state = None
        return self._attach(obs, np.ones(self.num_envs, np.float32))

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._attach(obs, dones.astype(np.float32)), rewards, dones, infos
