"""
InfoSaccade-RL: Complete Standalone Kaggle T4 GPU Script.
Can be executed directly on Kaggle with GPU Accelerator (T4 x1 or x2).

To run on Kaggle:
1. Create a new notebook on Kaggle.
2. Under "Notebook Options" on the right sidebar, set "Accelerator" -> "GPU T4 x1".
3. Copy and paste this script into a cell and execute.
"""

import os
import math
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchvision.models import resnet18

# --------------------------------------------------------------------------
# 0. Setup & Device Configuration
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Using compute device: {DEVICE}")
if torch.cuda.is_available():
    print(f"--> GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"--> Initial VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

# --------------------------------------------------------------------------
# 1. Differentiable 2D Discrete Wavelet Transform (Haar Sub-bands)
# --------------------------------------------------------------------------
class DifferentiableDWT2D(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        h = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        g = torch.tensor([-inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        ll = torch.outer(h, h)
        lh = torch.outer(h, g)
        hl = torch.outer(g, h)
        hh = torch.outer(g, g)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        weight = filters.repeat(in_channels, 1, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = F.conv2d(x, self.weight, stride=2, groups=C)
        out_reshaped = out.view(B, C, 4, out.shape[2], out.shape[3])
        subbands = {
            "LL": out_reshaped[:, :, 0],
            "LH": out_reshaped[:, :, 1],
            "HL": out_reshaped[:, :, 2],
            "HH": out_reshaped[:, :, 3]
        }
        return subbands, out_reshaped


class SpectralSpatialFusion(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.dwt = DifferentiableDWT2D(in_channels=in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        _, out_reshaped = self.dwt(patch)
        B, C, _, H2, W2 = out_reshaped.shape
        return self.conv(out_reshaped.view(B, C * 4, H2, W2))

# --------------------------------------------------------------------------
# 2. Differentiable Spatial Glimpse Sensor
# --------------------------------------------------------------------------
class SpatialGlimpseSensor(nn.Module):
    def __init__(self, img_channels=3, glimpse_size=14, num_scales=2, scale_factor=1.5, emb_dim=128, use_wavelet=True):
        super().__init__()
        self.img_channels = img_channels
        self.glimpse_size = glimpse_size
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.use_wavelet = use_wavelet

        self.patch_conv = nn.Sequential(
            nn.Conv2d(img_channels * num_scales, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True)
        )
        self.loc_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True)
        )
        if use_wavelet:
            self.spectral = SpectralSpatialFusion(in_channels=img_channels, out_dim=emb_dim // 2)
            fusion_in = emb_dim + (emb_dim // 2) + 32
        else:
            fusion_in = emb_dim + 32

        self.fc_fuse = nn.Sequential(
            nn.Linear(fusion_in, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True)
        )

    def extract_patch(self, images: torch.Tensor, locs: torch.Tensor, scale: float) -> torch.Tensor:
        B = images.shape[0]
        grid_size = self.glimpse_size
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size, device=images.device),
            torch.linspace(-1, 1, grid_size, device=images.device),
            indexing="ij"
        ), dim=-1).flip(-1).unsqueeze(0).expand(B, grid_size, grid_size, 2)
        grid = base_grid * scale + locs.unsqueeze(1).unsqueeze(1)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, images: torch.Tensor, locs: torch.Tensor):
        patches = [self.extract_patch(images, locs, 0.5 * (self.scale_factor ** s)) for s in range(self.num_scales)]
        stacked_patches = torch.cat(patches, dim=1)
        v_feat = self.patch_conv(stacked_patches)
        l_feat = self.loc_mlp(locs)

        if self.use_wavelet:
            s_feat = self.spectral(patches[0])
            fused = torch.cat([v_feat, s_feat, l_feat], dim=-1)
        else:
            fused = torch.cat([v_feat, l_feat], dim=-1)

        return self.fc_fuse(fused), stacked_patches

# --------------------------------------------------------------------------
# 3. Subjective Logic Evidential Head & Bayesian Loss
# --------------------------------------------------------------------------
class EvidentialHead(nn.Module):
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

    def forward(self, h: torch.Tensor):
        logits = self.fc(h)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S
        prob_clamped = torch.clamp(prob, min=1e-8)
        entropy = -torch.sum(prob_clamped * torch.log(prob_clamped), dim=-1, keepdim=True)
        return {"prob": prob, "alpha": alpha, "evidence": evidence, "uncertainty": uncertainty, "entropy": entropy}


def edl_loss_fn(alpha: torch.Tensor, targets: torch.Tensor, num_classes: int, kl_weight: float = 0.2):
    K = num_classes
    if targets.dim() == 1:
        y = F.one_hot(targets.long(), num_classes=K).float()
    else:
        y = F.one_hot(targets.squeeze(-1).long(), num_classes=K).float()

    S = torch.sum(alpha, dim=-1, keepdim=True)
    bayes_loss = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)

    tilde_alpha = y + (1.0 - y) * alpha
    beta = torch.ones((1, K), dtype=alpha.dtype, device=alpha.device)
    S_tilde = torch.sum(tilde_alpha, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)

    kl = (
        torch.lgamma(S_tilde) - torch.lgamma(S_beta)
        - torch.sum(torch.lgamma(tilde_alpha), dim=-1, keepdim=True)
        + torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)
        + torch.sum((tilde_alpha - beta) * (torch.digamma(tilde_alpha) - torch.digamma(S_tilde)), dim=-1, keepdim=True)
    ).squeeze(-1)

    return torch.mean(bayes_loss + kl_weight * kl)

# --------------------------------------------------------------------------
# 4. Multi-Action Actor-Critic Policy
# --------------------------------------------------------------------------
class MultiActionActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 256, init_action_std: float = 0.5):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.spatial_mean = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh())
        self.spatial_log_std = nn.Parameter(torch.ones(1, 2) * math.log(init_action_std))
        self.stopping_actor = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())
        self.critic = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def sample(self, h: torch.Tensor, deterministic: bool = False):
        feat = self.trunk(h)
        loc_mean = self.spatial_mean(feat)
        spatial_std = torch.exp(self.spatial_log_std).expand_as(loc_mean)
        stop_prob = self.stopping_actor(feat)
        value = self.critic(feat)

        dist = Normal(loc_mean, spatial_std)
        act = loc_mean if deterministic else torch.clamp(dist.sample(), -1.0, 1.0)
        logp = dist.log_prob(act).sum(dim=-1, keepdim=True)
        ent = dist.entropy().sum(dim=-1, keepdim=True)
        return {"act": act, "logp": logp, "ent": ent, "value": value, "stop_prob": stop_prob}

    def evaluate(self, h: torch.Tensor, act: torch.Tensor):
        feat = self.trunk(h)
        loc_mean = self.spatial_mean(feat)
        spatial_std = torch.exp(self.spatial_log_std).expand_as(loc_mean)
        value = self.critic(feat)
        dist = Normal(loc_mean, spatial_std)
        logp = dist.log_prob(act).sum(dim=-1, keepdim=True)
        ent = dist.entropy().sum(dim=-1, keepdim=True)
        return logp, ent, value

# --------------------------------------------------------------------------
# 5. Full InfoSaccade Model
# --------------------------------------------------------------------------
class InfoSaccadeModel(nn.Module):
    def __init__(self, img_channels=3, num_classes=9, glimpse_size=14, max_steps=6, use_wavelet=True):
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.rnn_hidden_dim = 256

        self.sensor = SpatialGlimpseSensor(img_channels=img_channels, glimpse_size=glimpse_size, emb_dim=128, use_wavelet=use_wavelet)
        self.rnn = nn.GRUCell(128, 256)
        self.actor_critic = MultiActionActorCritic(hidden_dim=256)
        self.classifier = EvidentialHead(in_features=256, num_classes=num_classes)
        self.init_loc = nn.Parameter(torch.zeros(1, 2))

    def forward(self, images: torch.Tensor, deterministic: bool = False):
        B = images.shape[0]
        h_t = torch.zeros(B, self.rnn_hidden_dim, device=images.device)
        loc_t = self.init_loc.expand(B, 2)

        step_outputs = []
        loc_history = [loc_t]
        stopped = torch.zeros(B, dtype=torch.bool, device=images.device)
        steps_taken = torch.zeros(B, dtype=torch.long, device=images.device)

        for step in range(self.max_steps):
            g_emb, _ = self.sensor(images, loc_t)
            h_t = self.rnn(g_emb, h_t)
            ev_out = self.classifier(h_t)
            pol_out = self.actor_critic.sample(h_t, deterministic=deterministic)

            step_outputs.append({"evidential": ev_out, "policy": pol_out, "loc": loc_t})
            loc_t = torch.clamp(loc_t + pol_out["act"], -1.0, 1.0)
            loc_history.append(loc_t)

            u_t = ev_out["uncertainty"].squeeze(-1)
            stop_prob = pol_out["stop_prob"].squeeze(-1)
            should_stop = (u_t < 0.25) | (stop_prob > 0.7)
            just_stopped = should_stop & (~stopped)
            steps_taken = torch.where(just_stopped, torch.full_like(steps_taken, step + 1), steps_taken)
            stopped = stopped | should_stop

        steps_taken = torch.where(steps_taken == 0, torch.full_like(steps_taken, self.max_steps), steps_taken)
        return {
            "final_prob": step_outputs[-1]["evidential"]["prob"],
            "final_uncertainty": step_outputs[-1]["evidential"]["uncertainty"],
            "step_outputs": step_outputs,
            "loc_history": loc_history,
            "steps_taken": steps_taken
        }

# --------------------------------------------------------------------------
# 6. Training Pipeline with Info-Gain Rewards & PPO
# --------------------------------------------------------------------------
def compute_rewards(step_outputs, targets, num_classes):
    T = len(step_outputs)
    B = targets.shape[0]
    rewards = torch.zeros(B, T, device=targets.device)
    y_true = targets.squeeze(-1).long() if targets.dim() > 1 else targets.long()

    prev_entropy = torch.full((B,), math.log(num_classes), device=targets.device)
    prev_unc = torch.ones((B,), device=targets.device)

    for t in range(T):
        ev = step_outputs[t]["evidential"]
        ent = ev["entropy"].squeeze(-1)
        unc = ev["uncertainty"].squeeze(-1)

        delta_info = prev_entropy - ent
        delta_unc = prev_unc - unc
        r_step = (1.0 * delta_info) + (0.5 * delta_unc) - 0.03

        prev_entropy, prev_unc = ent.detach(), unc.detach()

        if t == T - 1:
            pred = torch.argmax(ev["prob"], dim=-1)
            is_corr = (pred == y_true).float()
            r_term = torch.where(is_corr > 0.5, 2.0 * (0.5 + 0.5 * (1.0 - unc)), -1.0 * (0.5 + 0.5 * (1.0 - unc)))
            rewards[:, t] = r_step + r_term
        else:
            rewards[:, t] = r_step

    return rewards


def run_kaggle_experiment(dataset_name: str = "pathmnist", epochs: int = 20, batch_size: int = 128):
    print(f"\n=======================================================")
    print(f" Starting InfoSaccade-RL Benchmark on {dataset_name.upper()}")
    print(f"=======================================================\n")

    # Load MedMNIST dataset
    import medmnist
    from medmnist import INFO
    info = INFO[dataset_name.lower()]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])
    img_channels = info["n_channels"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = DataClass(split="train", transform=transform, download=True, as_rgb=True)
    val_data = DataClass(split="val", transform=transform, download=True, as_rgb=True)
    test_data = DataClass(split="test", transform=transform, download=True, as_rgb=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Loaded {dataset_name}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")

    # 1. Train Proposed InfoSaccade-RL
    model = InfoSaccadeModel(img_channels=3, num_classes=num_classes, glimpse_size=14, max_steps=6, use_wavelet=True).to(DEVICE)
    opt_backbone = torch.optim.AdamW(list(model.sensor.parameters()) + list(model.rnn.parameters()) + list(model.classifier.parameters()) + [model.init_loc], lr=1e-3, weight_decay=1e-4)
    opt_rl = torch.optim.AdamW(model.actor_critic.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            B = x.shape[0]

            h_t = torch.zeros(B, 256, device=DEVICE)
            loc_t = model.init_loc.expand(B, 2)
            step_outputs, actions, old_logps, values, hiddens = [], [], [], [], []

            for step in range(6):
                g_emb, _ = model.sensor(x, loc_t)
                h_t = model.rnn(g_emb, h_t)
                ev_out = model.classifier(h_t)
                pol_out = model.actor_critic.sample(h_t)

                step_outputs.append({"evidential": ev_out, "policy": pol_out, "loc": loc_t})
                actions.append(pol_out["act"])
                old_logps.append(pol_out["logp"])
                values.append(pol_out["value"])
                hiddens.append(h_t)
                loc_t = torch.clamp(loc_t + pol_out["act"].detach(), -1.0, 1.0)

            # Evidential supervised loss
            edl_loss = sum(((s_idx + 1) / 6.0) * edl_loss_fn(s["evidential"]["alpha"], y, num_classes) for s_idx, s in enumerate(step_outputs)) / 6.0
            opt_backbone.zero_grad()
            edl_loss.backward(retain_graph=True)
            opt_backbone.step()

            # PPO Policy Loss
            rewards = compute_rewards(step_outputs, y, num_classes)
            vals = torch.cat(values, dim=-1)
            advs = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(6)):
                next_v = 0.0 if t == 5 else vals[:, t + 1]
                delta = rewards[:, t] + 0.95 * next_v - vals[:, t]
                advs[:, t] = delta + 0.95 * 0.95 * last_gae
                last_gae = advs[:, t]
            rets = advs + vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            old_logps_t = torch.cat(old_logps, dim=-1).detach()
            actions_t = torch.stack(actions, dim=1).detach()

            for _ in range(3):
                opt_rl.zero_grad()
                new_lps, new_ents, new_vs = [], [], []
                for t in range(6):
                    lp, ent, v = model.actor_critic.evaluate(hiddens[t].detach(), actions_t[:, t])
                    new_lps.append(lp)
                    new_ents.append(ent)
                    new_vs.append(v)
                ratios = torch.exp(torch.cat(new_lps, dim=-1) - old_logps_t)
                surr1 = ratios * advs
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advs
                pol_l = -torch.min(surr1, surr2).mean()
                val_l = F.mse_loss(torch.cat(new_vs, dim=-1), rets.detach())
                ent_l = -torch.cat(new_ents, dim=-1).mean()
                (pol_l + 0.5 * val_l + 0.02 * ent_l).backward()
                opt_rl.step()

        # Validation
        model.eval()
        val_probs, val_targets, val_steps = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x, deterministic=True)
                val_probs.append(out["final_prob"].cpu().numpy())
                val_targets.append(y.numpy())
                val_steps.append(out["steps_taken"].cpu().numpy())

        val_probs = np.concatenate(val_probs, axis=0)
        val_targets = np.concatenate(val_targets, axis=0).squeeze()
        val_steps = np.concatenate(val_steps, axis=0)
        val_acc = accuracy_score(val_targets, np.argmax(val_probs, axis=1))
        val_auc = roc_auc_score(val_targets, val_probs, multi_class="ovr", average="macro")
        avg_s = np.mean(val_steps)

        print(f"Epoch {epoch:02d}/{epochs:02d} ({time.time()-t0:.1f}s) -> Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f} | Avg Steps: {avg_s:.2f}/6")

    # 2. Test Evaluation
    model.eval()
    test_probs, test_targets, test_steps = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x, deterministic=True)
            test_probs.append(out["final_prob"].cpu().numpy())
            test_targets.append(y.numpy())
            test_steps.append(out["steps_taken"].cpu().numpy())

    test_probs = np.concatenate(test_probs, axis=0)
    test_targets = np.concatenate(test_targets, axis=0).squeeze()
    test_steps = np.concatenate(test_steps, axis=0)
    test_acc = accuracy_score(test_targets, np.argmax(test_probs, axis=1))
    test_auc = roc_auc_score(test_targets, test_probs, multi_class="ovr", average="macro")
    test_f1 = f1_score(test_targets, np.argmax(test_probs, axis=1), average="macro")
    test_steps_avg = np.mean(test_steps)

    print(f"\n=======================================================")
    print(f" InfoSaccade-RL Final Results on {dataset_name.upper()}:")
    print(f" - Test Accuracy : {test_acc*100:.2f}%")
    print(f" - Test AUC-ROC  : {test_auc:.4f}")
    print(f" - Test F1-Macro : {test_f1:.4f}")
    print(f" - Avg Steps     : {test_steps_avg:.2f} / 6 ({(1.0 - test_steps_avg/6)*100:.1f}% Savings!)")
    print(f"=======================================================\n")


if __name__ == "__main__":
    run_kaggle_experiment("pathmnist", epochs=15, batch_size=128)
