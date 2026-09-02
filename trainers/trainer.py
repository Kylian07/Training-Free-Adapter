"""
Trainer module for InfoSaccade-RL.
Implements joint optimization:
- Differentiable feature extractor and evidential classifier via Dirichlet Evidential Loss.
- Active visual policy via Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE).
"""

import os
import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.infosaccade_model import InfoSaccadeModel
from losses.evidential_loss import EvidentialLoss
from losses.rl_loss import compute_information_theoretic_rewards, compute_gae, compute_ppo_loss
from utils.metrics import calculate_all_metrics


class InfoSaccadeTrainer:
    def __init__(
        self,
        model: InfoSaccadeModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        lr_backbone: float = 1e-3,
        lr_actor_critic: float = 5e-4,
        weight_decay: float = 1e-4,
        ppo_clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        use_amp: bool = True,
        save_dir: str = "./checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        self.ppo_clip_eps = ppo_clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Loss functions
        self.edl_criterion = EvidentialLoss(num_classes=num_classes, kl_weight=0.2, annealing_epochs=10)

        # Separate optimizers for fine-grained learning rate tuning
        # Vision backbone + GRU + Evidential Head
        backbone_params = (
            list(self.model.sensor.parameters())
            + list(self.model.rnn_cell.parameters())
            + list(self.model.classifier.parameters())
            + [self.model.init_loc]
        )
        self.optimizer_backbone = torch.optim.AdamW(
            backbone_params,
            lr=lr_backbone,
            weight_decay=weight_decay
        )

        # Actor-Critic Policy
        self.optimizer_rl = torch.optim.AdamW(
            self.model.actor_critic.parameters(),
            lr=lr_actor_critic,
            weight_decay=weight_decay
        )

        # Learning rate schedulers
        self.scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_backbone, T_max=30, eta_min=1e-5)
        self.scheduler_rl = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_rl, T_max=30, eta_min=1e-5)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_edl_loss = 0.0
        total_ppo_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            B = images.shape[0]

            # ----------------------------------------------------
            # 1. Trajectory Rollout Collection (Forward Pass)
            # ----------------------------------------------------
            h_t = self.model.init_hidden(B, self.device)
            loc_t = self.model.init_loc.expand(B, 2)

            step_outputs = []
            spatial_actions = []
            spectral_actions = []
            old_log_probs = []
            values = []
            hidden_list = []

            for step in range(self.model.max_steps):
                glimpse_emb, patch = self.model.sensor(images, loc_t)
                h_t = self.model.rnn_cell(glimpse_emb, h_t)
                evidential_out = self.model.classifier(h_t)
                policy_out = self.model.actor_critic.sample_actions(h_t, deterministic=False)

                step_outputs.append({
                    "evidential": evidential_out,
                    "policy": policy_out,
                    "loc": loc_t
                })
                spatial_actions.append(policy_out["spatial_action"])
                spectral_actions.append(policy_out["spectral_action"])
                old_log_probs.append(policy_out["total_log_prob"])
                values.append(policy_out["value"])
                hidden_list.append(h_t)

                loc_t = torch.clamp(loc_t + policy_out["spatial_action"].detach(), -1.0, 1.0)

            # ----------------------------------------------------
            # 2. Information-Theoretic Reward & GAE Computation
            # ----------------------------------------------------
            values_tensor = torch.cat(values, dim=-1).detach()  # (B, T)
            rewards_tensor = compute_information_theoretic_rewards(
                step_outputs=step_outputs,
                targets=targets,
                info_gain_weight=1.0,
                uncertainty_weight=0.5,
                step_cost=0.03
            ).detach()
            advantages, returns = compute_gae(
                rewards=rewards_tensor,
                values=values_tensor,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
            advantages = advantages.detach()
            returns = returns.detach()

            # ----------------------------------------------------
            # 3. Supervised / Evidential Loss Update
            # ----------------------------------------------------
            self.optimizer_backbone.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Evidential loss across all steps with increasing weight towards the end
                edl_loss = 0.0
                for step_idx, s_out in enumerate(step_outputs):
                    step_alpha = s_out["evidential"]["alpha"]
                    step_weight = (step_idx + 1) / self.model.max_steps
                    edl_loss = edl_loss + step_weight * self.edl_criterion(step_alpha, targets, current_epoch=epoch)
                edl_loss = edl_loss / self.model.max_steps

            if self.use_amp:
                self.scaler.scale(edl_loss).backward()
                self.scaler.step(self.optimizer_backbone)
                self.scaler.update()
            else:
                edl_loss.backward()
                self.optimizer_backbone.step()

            # ----------------------------------------------------
            # 4. PPO Actor-Critic Policy Updates
            # ----------------------------------------------------
            old_log_probs_tensor = torch.cat(old_log_probs, dim=-1).detach()  # (B, T)
            spatial_actions_tensor = torch.stack(spatial_actions, dim=1).detach()  # (B, T, 2)
            spectral_actions_tensor = torch.stack(spectral_actions, dim=1).detach()  # (B, T)
            hidden_tensor = [h.detach() for h in hidden_list]

            for _ in range(self.ppo_epochs):
                self.optimizer_rl.zero_grad()
                new_log_probs_list = []
                new_entropies_list = []
                new_values_list = []

                for t_idx in range(self.model.max_steps):
                    h_curr = hidden_tensor[t_idx]
                    logp, ent, val = self.model.actor_critic.evaluate_actions(
                        h_curr,
                        spatial_actions_tensor[:, t_idx],
                        spectral_actions_tensor[:, t_idx]
                    )
                    new_log_probs_list.append(logp)
                    new_entropies_list.append(ent)
                    new_values_list.append(val)

                new_log_probs_t = torch.cat(new_log_probs_list, dim=-1)
                new_entropies_t = torch.cat(new_entropies_list, dim=-1)
                new_values_t = torch.cat(new_values_list, dim=-1)

                ppo_total_loss, pol_loss, val_loss, ent_loss = compute_ppo_loss(
                    old_log_probs=old_log_probs_tensor,
                    new_log_probs=new_log_probs_t,
                    advantages=advantages,
                    values=new_values_t,
                    returns=returns,
                    entropy=new_entropies_t,
                    clip_eps=self.ppo_clip_eps,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef
                )

                ppo_total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor_critic.parameters(), max_norm=1.0)
                self.optimizer_rl.step()

            # Track metrics
            total_edl_loss += edl_loss.item()
            total_ppo_loss += pol_loss.item()
            total_value_loss += val_loss.item()
            total_entropy += ent_loss.item()
            total_reward += rewards_tensor.mean().item()
            n_batches += 1

            pbar.set_postfix({
                "EDL": f"{edl_loss.item():.3f}",
                "PPO": f"{pol_loss.item():.3f}",
                "Rew": f"{rewards_tensor.mean().item():.2f}"
            })

        self.scheduler_backbone.step()
        self.scheduler_rl.step()

        return {
            "loss_edl": total_edl_loss / n_batches,
            "loss_ppo": total_ppo_loss / n_batches,
            "loss_value": total_value_loss / n_batches,
            "mean_reward": total_reward / n_batches
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        self.model.eval()
        all_probs = []
        all_targets = []
        all_steps = []
        all_uncertainties = []

        for images, targets in tqdm(data_loader, desc=f"Evaluating [{desc}]", leave=False):
            images = images.to(self.device)
            out = self.model(images, deterministic=True)
            
            probs = out["final_prob"].cpu().numpy()
            uncertainties = out["final_uncertainty"].cpu().numpy()
            steps = out["steps_taken"].cpu().numpy()
            
            if isinstance(targets, torch.Tensor):
                targets_np = targets.numpy()
            else:
                targets_np = np.array(targets)

            all_probs.append(probs)
            all_targets.append(targets_np)
            all_steps.append(steps)
            all_uncertainties.append(uncertainties)

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_steps = np.concatenate(all_steps, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)

        metrics = calculate_all_metrics(all_probs, all_targets)
        metrics["avg_steps"] = float(np.mean(all_steps))
        metrics["avg_uncertainty"] = float(np.mean(all_uncertainties))
        metrics["efficiency_gain_pct"] = float((1.0 - (np.mean(all_steps) / self.model.max_steps)) * 100.0)

        return metrics

    def fit(self, num_epochs: int) -> Dict[str, any]:
        best_val_auc = 0.0
        best_val_acc = 0.0
        history = []

        print(f"\n=======================================================")
        print(f" Starting InfoSaccade-RL Training for {num_epochs} Epochs")
        print(f" Target Device: {self.device} | AMP Enabled: {self.use_amp}")
        print(f"=======================================================\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, desc="Validation")
            elapsed = time.time() - t0

            print(
                f"[Epoch {epoch:02d}/{num_epochs:02d} - {elapsed:.1f}s] "
                f"Train EDL: {train_metrics['loss_edl']:.4f} | Rew: {train_metrics['mean_reward']:.2f} | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | Val AUC: {val_metrics['auc']:.4f} | "
                f"Avg Steps: {val_metrics['avg_steps']:.2f}/{self.model.max_steps} | ECE: {val_metrics['ece']:.4f}"
            )

            # Save best checkpoint by AUC
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_val_acc = val_metrics["accuracy"]
                best_path = os.path.join(self.save_dir, "best_infosaccade_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc": best_val_auc,
                    "val_acc": best_val_acc,
                    "val_metrics": val_metrics
                }, best_path)
                print(f"  --> Saved new best checkpoint to {best_path} (Val AUC: {best_val_auc:.4f})")

            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics
            })

        # Final Test Evaluation with best model
        best_path = os.path.join(self.save_dir, "best_infosaccade_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\nLoaded Best Checkpoint from Epoch {checkpoint['epoch']} for Final Testing.")

        test_metrics = self.evaluate(self.test_loader, desc="Test")
        print(f"\n=======================================================")
        print(f" Final Test Results:")
        print(f" - Test Accuracy : {test_metrics['accuracy'] * 100:.2f}%")
        print(f" - Test AUC-ROC  : {test_metrics['auc']:.4f}")
        print(f" - Test F1-Macro : {test_metrics['f1_macro']:.4f}")
        print(f" - Test ECE      : {test_metrics['ece']:.4f}")
        print(f" - Avg Steps     : {test_metrics['avg_steps']:.2f} / {self.model.max_steps}")
        print(f" - FLOPs Savings : {test_metrics['efficiency_gain_pct']:.1f}%")
        print(f"=======================================================\n")

        return {
            "history": history,
            "best_val_auc": best_val_auc,
            "test_metrics": test_metrics
        }
