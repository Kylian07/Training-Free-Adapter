"""
Actor-Critic Policy Network for Multimodal Saccade Decisions:
1. Continuous Spatial Action: Fixation displacement (Delta x, Delta y) and zoom scale.
2. Discrete Spectral Action: Wavelet sub-band selection (LL, LH, HL, HH).
3. Halting Action: Early stopping probability.
4. Value Function: State value estimation V(h_t).
"""

from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli


class MultiActionActorCritic(nn.Module):
    """
    Joint Actor-Critic network conditioned on recurrent belief state h_t.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        init_action_std: float = 0.5,
        num_spectral_actions: int = 4  # LL, LH, HL, HH
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_spectral_actions = num_spectral_actions

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # 1. Spatial Continuous Actor: Mean (Delta x, Delta y, scale)
        self.spatial_mean = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # (x, y)
            nn.Tanh()  # Squashed to [-1, 1]
        )
        # Learnable log standard deviation for spatial exploration
        self.spatial_log_std = nn.Parameter(torch.ones(1, 2) * math.log(init_action_std))

        # 2. Spectral Discrete Actor: Logits over sub-bands
        self.spectral_actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_spectral_actions)
        )

        # 3. Halting Action Actor: Stopping probability
        self.stopping_actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 4. Critic: State value function V(h_t)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate policy without sampling.
        """
        feat = self.trunk(h)
        loc_mean = self.spatial_mean(feat)
        spec_logits = self.spectral_actor(feat)
        stop_prob = self.stopping_actor(feat)
        value = self.critic(feat)
        return {
            "spatial_mean": loc_mean,
            "spatial_std": torch.exp(self.spatial_log_std).expand_as(loc_mean),
            "spectral_logits": spec_logits,
            "stopping_prob": stop_prob,
            "value": value
        }

    def sample_actions(self, h: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Sample actions from policy distributions.
        """
        feat = self.trunk(h)
        loc_mean = self.spatial_mean(feat)
        spatial_std = torch.exp(self.spatial_log_std).expand_as(loc_mean)
        spec_logits = self.spectral_actor(feat)
        stop_prob = self.stopping_actor(feat)
        value = self.critic(feat)

        # Spatial Distribution
        spatial_dist = Normal(loc_mean, spatial_std)
        if deterministic:
            spatial_action = loc_mean
        else:
            spatial_action = spatial_dist.sample()
            spatial_action = torch.clamp(spatial_action, -1.0, 1.0)
        spatial_log_prob = spatial_dist.log_prob(spatial_action).sum(dim=-1, keepdim=True)

        # Spectral Distribution
        spectral_dist = Categorical(logits=spec_logits)
        if deterministic:
            spectral_action = torch.argmax(spec_logits, dim=-1)
        else:
            spectral_action = spectral_dist.sample()
        spectral_log_prob = spectral_dist.log_prob(spectral_action).unsqueeze(-1)

        # Total Action Log Prob
        total_log_prob = spatial_log_prob + spectral_log_prob

        # Entropy for exploration bonus
        spatial_entropy = spatial_dist.entropy().sum(dim=-1, keepdim=True)
        spectral_entropy = spectral_dist.entropy().unsqueeze(-1)
        total_entropy = spatial_entropy + spectral_entropy

        return {
            "spatial_action": spatial_action,
            "spectral_action": spectral_action,
            "stopping_prob": stop_prob,
            "total_log_prob": total_log_prob,
            "spatial_log_prob": spatial_log_prob,
            "spectral_log_prob": spectral_log_prob,
            "entropy": total_entropy,
            "value": value
        }

    def evaluate_actions(
        self,
        h: torch.Tensor,
        spatial_action: torch.Tensor,
        spectral_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities and entropy of taken actions for PPO updates.
        """
        feat = self.trunk(h)
        loc_mean = self.spatial_mean(feat)
        spatial_std = torch.exp(self.spatial_log_std).expand_as(loc_mean)
        spec_logits = self.spectral_actor(feat)
        value = self.critic(feat)

        spatial_dist = Normal(loc_mean, spatial_std)
        spatial_log_prob = spatial_dist.log_prob(spatial_action).sum(dim=-1, keepdim=True)
        spatial_entropy = spatial_dist.entropy().sum(dim=-1, keepdim=True)

        spectral_dist = Categorical(logits=spec_logits)
        spectral_log_prob = spectral_dist.log_prob(spectral_action).unsqueeze(-1)
        spectral_entropy = spectral_dist.entropy().unsqueeze(-1)

        total_log_prob = spatial_log_prob + spectral_log_prob
        total_entropy = spatial_entropy + spectral_entropy

        return total_log_prob, total_entropy, value
