"""
Comprehensive Evaluator and Benchmark Suite for ICASSP paper tables and visualizations.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.infosaccade_model import InfoSaccadeModel
from utils.metrics import calculate_all_metrics
from utils.visualizer import plot_saccade_scanpath, plot_uncertainty_entropy_dynamics, plot_wavelet_subbands


class BenchmarkEvaluator:
    def __init__(self, model: InfoSaccadeModel, test_loader: DataLoader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.test_loader = test_loader

    @torch.no_grad()
    def run_full_evaluation(self, output_dir: str = "./results") -> Dict[str, any]:
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        all_probs = []
        all_targets = []
        all_steps = []
        all_uncertainties = []
        sample_scanpaths = []

        for b_idx, (images, targets) in enumerate(tqdm(self.test_loader, desc="Running Benchmark")):
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

            # Collect visual scanpaths for first batch
            if b_idx == 0:
                step_outputs = out["step_outputs"]
                loc_history_tensors = out["loc_history"]  # list of (B, 2)
                
                for sample_i in range(min(5, images.shape[0])):
                    img_np = images[sample_i].detach().cpu().numpy()
                    if img_np.shape[0] in [1, 3]:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                    
                    # Un-normalize from [-1, 1] to [0, 1]
                    img_np = np.clip((img_np * 0.5) + 0.5, 0.0, 1.0)

                    locs = [loc[sample_i].detach().cpu().numpy() for loc in loc_history_tensors]
                    true_cls = str(targets_np[sample_i])
                    pred_cls = str(np.argmax(probs[sample_i]))

                    save_p = os.path.join(output_dir, f"scanpath_sample_{sample_i}.png")
                    plot_saccade_scanpath(
                        image=img_np,
                        loc_history=locs,
                        glimpse_size=self.model.sensor.glimpse_size,
                        true_label=f"Class {true_cls}",
                        pred_label=f"Class {pred_cls}",
                        save_path=save_p
                    )

                    # Dynamic uncertainty plot
                    ent_steps = [s["evidential"]["entropy"][sample_i].item() for s in step_outputs]
                    unc_steps = [s["evidential"]["uncertainty"][sample_i].item() for s in step_outputs]
                    plot_uncertainty_entropy_dynamics(
                        entropies_per_step=ent_steps,
                        uncertainties_per_step=unc_steps,
                        save_path=os.path.join(output_dir, f"uncertainty_sample_{sample_i}.png")
                    )

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_steps = np.concatenate(all_steps, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)

        metrics = calculate_all_metrics(all_probs, all_targets)
        metrics["avg_steps"] = float(np.mean(all_steps))
        metrics["avg_uncertainty"] = float(np.mean(all_uncertainties))
        metrics["efficiency_gain_pct"] = float((1.0 - (np.mean(all_steps) / self.model.max_steps)) * 100.0)

        return metrics
