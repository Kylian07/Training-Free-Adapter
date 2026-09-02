"""
Rollout buffer for storing sequential saccade trajectories and calculating GAE.
"""

from typing import List, Dict, Tuple
import torch


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_outputs = []
        self.spatial_actions = []
        self.spectral_actions = []
        self.log_probs = []
        self.values = []
        self.hidden_states = []
        self.rewards = None
        self.advantages = None
        self.returns = None
        self.images = None
        self.targets = None

    def store_step(
        self,
        hidden: torch.Tensor,
        spatial_act: torch.Tensor,
        spectral_act: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        step_dict: Dict[str, any]
    ):
        self.hidden_states.append(hidden)
        self.spatial_actions.append(spatial_act)
        self.spectral_actions.append(spectral_act)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.step_outputs.append(step_dict)

    def finalize_rollout(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ):
        self.images = images
        self.targets = targets
        self.rewards = rewards
        self.advantages = advantages
        self.returns = returns
