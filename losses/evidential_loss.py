"""
Loss functions for Subjective Logic Dirichlet Evidential Learning.
Implements Bayesian Risk with Digamma function and KL-divergence regularizer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes KL[ Dir(alpha) || Dir(<1, 1, ..., 1>) ] (Uniform Dirichlet prior).
    KL(p || q) = ln Gamma(S) - sum ln Gamma(alpha_k) - ln Gamma(K) + sum ln Gamma(1)
                 + sum (alpha_k - 1) * (psi(alpha_k) - psi(S))
    """
    beta = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_gamma_S_alpha = torch.lgamma(S_alpha)
    ln_gamma_S_beta = torch.lgamma(S_beta)
    sum_ln_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    sum_ln_gamma_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)

    psi_alpha = torch.digamma(alpha)
    psi_S_alpha = torch.digamma(S_alpha)

    kl = (
        ln_gamma_S_alpha
        - ln_gamma_S_beta
        - sum_ln_gamma_alpha
        + sum_ln_gamma_beta
        + torch.sum((alpha - beta) * (psi_alpha - psi_S_alpha), dim=-1, keepdim=True)
    )
    return kl.squeeze(-1)


class EvidentialLoss(nn.Module):
    """
    Computes Type-II Bayesian Dirichlet loss with Digamma expectation:
    L_bayes = sum_k y_k * (psi(S) - psi(alpha_k))
    + Annealed KL divergence to prevent ungrounded evidence on misleading classes.
    """
    def __init__(self, num_classes: int, kl_weight: float = 0.2, annealing_epochs: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs

    def forward(self, alpha: torch.Tensor, targets: torch.Tensor, current_epoch: int = 0) -> torch.Tensor:
        """
        Args:
            alpha: Dirichlet parameters (B, K)
            targets: Class labels (B,) or one-hot (B, K)
            current_epoch: int, for annealing schedule
        Returns:
            loss: scalar loss
        """
        device = alpha.device
        B, K = alpha.shape

        if targets.dim() == 1:
            y_onehot = F.one_hot(targets.long(), num_classes=K).float()
        elif targets.shape[-1] == 1:
            y_onehot = F.one_hot(targets.squeeze(-1).long(), num_classes=K).float()
        else:
            y_onehot = targets.float()

        S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)

        # Expected cross-entropy under Dirichlet distribution:
        # E_p[-log p_y] = psi(S) - psi(alpha_y)
        bayes_loss = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)  # (B,)

        # Regularization: remove ground-truth evidence to form misleading Dirichlet distribution
        tilde_alpha = y_onehot + (1.0 - y_onehot) * alpha
        kl_reg = kl_divergence_dirichlet(tilde_alpha, self.num_classes)  # (B,)

        # Annealing coefficient for KL term (gradually ramps up to avoid early collapse)
        annealing_coef = min(1.0, float(current_epoch) / max(1, self.annealing_epochs))
        
        total_loss = torch.mean(bayes_loss + self.kl_weight * annealing_coef * kl_reg)
        return total_loss
