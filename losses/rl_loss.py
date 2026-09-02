"""
Information-Theoretic Reward Calculation and PPO / Actor-Critic Objective.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_information_theoretic_rewards(
    step_outputs: List[Dict[str, any]],
    targets: torch.Tensor,
    info_gain_weight: float = 1.0,
    uncertainty_weight: float = 0.5,
    step_cost: float = 0.05,
    terminal_acc_reward: float = 2.0,
    terminal_err_penalty: float = -1.0
) -> torch.Tensor:
    """
    Computes per-step information-theoretic rewards:
    R_t = w_1 * Delta I_t + w_2 * Delta U_t - step_cost + I(t=T) * R_task
    
    Returns:
        rewards: Tensor of shape (B, T)
    """
    T = len(step_outputs)
    B = targets.shape[0]
    device = targets.device

    if targets.dim() > 1 and targets.shape[-1] == 1:
        y_true = targets.squeeze(-1).long()
    else:
        y_true = targets.long()

    rewards = torch.zeros(B, T, device=device)

    prev_entropy = None
    prev_uncertainty = None

    for t in range(T):
        evidential = step_outputs[t]["evidential"]
        entropy = evidential["entropy"].squeeze(-1)       # (B,)
        uncertainty = evidential["uncertainty"].squeeze(-1) # (B,)
        probs = evidential["prob"]                        # (B, K)

        if t == 0:
            # Baseline uniform entropy log(K) and uncertainty 1.0
            K = probs.shape[-1]
            prev_entropy = torch.full_like(entropy, math_log_k := torch.log(torch.tensor(float(K), device=device)))
            prev_uncertainty = torch.ones_like(uncertainty)

        # 1. Mutual Information Gain: H(Y | h_{t-1}) - H(Y | h_t)
        delta_info = prev_entropy - entropy
        
        # 2. Vacuity Uncertainty Reduction: u_{t-1} - u_t
        delta_unc = prev_uncertainty - uncertainty

        # Step reward
        r_step = (info_gain_weight * delta_info) + (uncertainty_weight * delta_unc) - step_cost

        # Update trackers
        prev_entropy = entropy.detach()
        prev_uncertainty = uncertainty.detach()

        # Terminal reward at t = T - 1
        if t == T - 1:
            pred_class = torch.argmax(probs, dim=-1)
            is_correct = (pred_class == y_true).float()
            
            # Scaled by confidence: rewarded more for confident correct calls, penalized for overconfident mistakes
            conf = 1.0 - uncertainty
            r_term = torch.where(
                is_correct > 0.5,
                terminal_acc_reward * (0.5 + 0.5 * conf),
                terminal_err_penalty * (0.5 + 0.5 * conf)
            )
            rewards[:, t] = r_step + r_term
        else:
            rewards[:, t] = r_step

    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.95,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and Returns.
    rewards: (B, T)
    values: (B, T)
    Returns:
        advantages: (B, T)
        returns: (B, T)
    """
    B, T = rewards.shape
    device = rewards.device
    advantages = torch.zeros(B, T, device=device)
    last_gae = torch.zeros(B, device=device)

    # Backward recursion for GAE
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(B, device=device)
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = delta + gamma * gae_lambda * last_gae
        last_gae = advantages[:, t]

    returns = advantages + values
    return advantages, returns


def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PPO Clipped Surrogate Loss + Value Function MSE + Entropy Bonus.
    """
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std

    # Probability ratio r_t(theta)
    ratios = torch.exp(new_log_probs - old_log_probs.detach())

    # Clipped surrogate objective
    surr1 = ratios * norm_advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * norm_advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns.detach())

    # Entropy loss (maximize entropy -> negative loss)
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    return total_loss, policy_loss, value_loss, entropy_loss
