"""
InfoSaccade-RL Complete End-to-End Model.
Integrates:
- Differentiable Spatial-Spectral Glimpse Sensor
- Recurrent Belief State (GRU)
- Multi-Action Actor-Critic Policy Network
- Subjective Logic Evidential Dirichlet Head
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_glimpse import SpatialGlimpseSensor
from models.policy_network import MultiActionActorCritic
from models.evidential_head import EvidentialClassificationHead


class InfoSaccadeModel(nn.Module):
    """
    End-to-end framework formulating medical diagnostic classification as a
    sequential multiscale spectral-spatial POMDP.
    """
    def __init__(
        self,
        img_channels: int = 3,
        num_classes: int = 9,
        glimpse_size: int = 14,
        num_scales: int = 2,
        glimpse_emb_dim: int = 128,
        rnn_hidden_dim: int = 256,
        max_steps: int = 6,
        use_wavelet: bool = True,
        init_action_std: float = 0.5,
        enable_early_stopping: bool = True,
        stopping_threshold: float = 0.20  # Vacuity uncertainty threshold
    ):
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.rnn_hidden_dim = rnn_hidden_dim
        self.enable_early_stopping = enable_early_stopping
        self.stopping_threshold = stopping_threshold

        # 1. Sensor
        self.sensor = SpatialGlimpseSensor(
            img_channels=img_channels,
            glimpse_size=glimpse_size,
            num_scales=num_scales,
            emb_dim=glimpse_emb_dim,
            use_wavelet=use_wavelet
        )

        # 2. Recurrent Core (Belief State Accumulator)
        self.rnn_cell = nn.GRUCell(glimpse_emb_dim, rnn_hidden_dim)

        # 3. Policy & Value Network (Actor-Critic)
        self.actor_critic = MultiActionActorCritic(
            hidden_dim=rnn_hidden_dim,
            init_action_std=init_action_std,
            num_spectral_actions=4
        )

        # 4. Evidential Dirichlet Classifier
        self.classifier = EvidentialClassificationHead(
            in_features=rnn_hidden_dim,
            num_classes=num_classes
        )

        # Learnable initial location prior (default: centered)
        self.init_loc = nn.Parameter(torch.zeros(1, 2))

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)

    def forward_step(
        self,
        images: torch.Tensor,
        current_loc: torch.Tensor,
        hidden_state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Executes a single saccadic decision step.
        """
        # Glimpse feature extraction
        glimpse_emb, patch = self.sensor(images, current_loc)

        # Update belief state
        new_hidden = self.rnn_cell(glimpse_emb, hidden_state)

        # Evidential prediction
        evidential_out = self.classifier(new_hidden)

        # Sample actions from policy
        policy_out = self.actor_critic.sample_actions(new_hidden, deterministic=deterministic)

        # Calculate next location displacement
        next_loc = torch.clamp(current_loc + policy_out["spatial_action"], -1.0, 1.0)

        return {
            "hidden": new_hidden,
            "next_loc": next_loc,
            "current_loc": current_loc,
            "patch": patch,
            "glimpse_emb": glimpse_emb,
            "evidential": evidential_out,
            "policy": policy_out
        }

    def forward(
        self,
        images: torch.Tensor,
        deterministic: bool = False,
        force_max_steps: bool = False
    ) -> Dict[str, any]:
        """
        Executes full sequential saccade scanpath on a batch of images.
        """
        B = images.shape[0]
        device = images.device

        h_t = self.init_hidden(B, device)
        loc_t = self.init_loc.expand(B, 2)

        # Tracking trajectory
        step_outputs = []
        loc_history = [loc_t]
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(self.max_steps):
            step_res = self.forward_step(images, loc_t, h_t, deterministic=deterministic)
            h_t = step_res["hidden"]
            loc_t = step_res["next_loc"]
            loc_history.append(loc_t)
            step_outputs.append(step_res)

            # Check early stopping condition:
            # Active if uncertainty is sufficiently low or stop prob is high
            u_t = step_res["evidential"]["uncertainty"].squeeze(-1)  # (B,)
            stop_prob = step_res["policy"]["stopping_prob"].squeeze(-1)  # (B,)
            
            # If early stopping is active (in inference or optionally train)
            if self.enable_early_stopping and not force_max_steps:
                should_stop = (u_t < self.stopping_threshold) | (stop_prob > 0.7)
                just_stopped = should_stop & (~stopped)
                steps_taken = torch.where(just_stopped, torch.full_like(steps_taken, step + 1), steps_taken)
                stopped = stopped | should_stop

        # Assign final step for any samples that didn't stop early
        steps_taken = torch.where(steps_taken == 0, torch.full_like(steps_taken, self.max_steps), steps_taken)

        # Final predictions from the last executed step
        final_evidential = step_outputs[-1]["evidential"]

        return {
            "final_prob": final_evidential["prob"],
            "final_alpha": final_evidential["alpha"],
            "final_evidence": final_evidential["evidence"],
            "final_uncertainty": final_evidential["uncertainty"],
            "step_outputs": step_outputs,
            "loc_history": loc_history,
            "steps_taken": steps_taken
        }
