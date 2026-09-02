"""
Evidential Deep Learning (Subjective Logic / Dirichlet Prior) Classification Head.
Parameterizes a Dirichlet distribution Dir(alpha) over class multinomials,
explicitly decoupling belief mass per class from epistemic/vacuity uncertainty.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialClassificationHead(nn.Module):
    """
    Predicts non-negative evidence vectors e >= 0 using softplus activation.
    Derives Dirichlet parameters alpha_k = e_k + 1, expected probabilities p_k = alpha_k / S,
    belief masses b_k = e_k / S, and vacuity uncertainty u = K / S.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: Belief state / feature representation (B, in_features)
        Returns:
            dict containing:
              - 'logits': raw logits (B, K)
              - 'evidence': non-negative evidence e (B, K)
              - 'alpha': Dirichlet parameters alpha = e + 1 (B, K)
              - 'prob': Expected class probabilities p = alpha / S (B, K)
              - 'belief': Belief mass vector b = e / S (B, K)
              - 'uncertainty': Vacuity epistemic uncertainty u = K / S (B, 1)
              - 'entropy': Dirichlet differential entropy (B, 1)
        """
        logits = self.fc(h)
        # Non-negative evidence via smooth softplus
        evidence = F.softplus(logits)
        alpha = evidence + 1.0  # (B, K)
        S = torch.sum(alpha, dim=-1, keepdim=True)  # Total Dirichlet strength (B, 1)
        
        prob = alpha / S
        belief = evidence / S
        uncertainty = self.num_classes / S  # u in (0, 1]
        
        # Shannon entropy of expected probabilities
        prob_clamped = torch.clamp(prob, min=1e-8)
        entropy = -torch.sum(prob_clamped * torch.log(prob_clamped), dim=-1, keepdim=True)

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "belief": belief,
            "uncertainty": uncertainty,
            "entropy": entropy,
            "total_strength": S
        }
