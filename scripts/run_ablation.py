"""
Ablation Study and Baseline Benchmarking Script for ICASSP Paper.
Generates comprehensive comparative performance table:
- Accuracy, AUC-ROC, Macro-F1, ECE, Mean Steps, and Efficiency Gain.
"""

import os
import sys
import argparse

# Add repository root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset_loader import get_medmnist_loaders
from models.infosaccade_model import InfoSaccadeModel
from models.baselines import BaselineResNet, BaselineViTTiny, StandardRAMBaseline
from trainers.trainer import InfoSaccadeTrainer
from utils.metrics import calculate_all_metrics


def train_baseline_supervised(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, epochs: int = 15, lr: float = 1e-3, device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.long())
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Test evaluation
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            all_probs.append(probs)
            all_targets.append(y)

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_all_metrics(all_probs, all_targets)
    metrics["avg_steps"] = "Full Image (1.0x)"
    metrics["efficiency_gain_pct"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pathmnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_table", type=str, default="./results/ablation_table.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_table)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, ds_info = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        download=True
    )
    num_classes = ds_info.get("n_classes", 9)
    img_channels = ds_info.get("n_channels", 3)

    results_table = []

    print(f"\n[1/5] Training Proposed Method: InfoSaccade-RL (Full Framework)...")
    model_full = InfoSaccadeModel(img_channels=img_channels, num_classes=num_classes, use_wavelet=True, enable_early_stopping=True)
    trainer_full = InfoSaccadeTrainer(model_full, train_loader, val_loader, test_loader, num_classes=num_classes, device=device)
    res_full = trainer_full.fit(num_epochs=args.epochs)["test_metrics"]
    res_full["Method"] = "InfoSaccade-RL (Ours - Full)"
    results_table.append(res_full)

    print(f"\n[2/5] Training Ablation: InfoSaccade-RL w/o Wavelet Spectral Sub-bands...")
    model_no_wave = InfoSaccadeModel(img_channels=img_channels, num_classes=num_classes, use_wavelet=False, enable_early_stopping=True)
    trainer_no_wave = InfoSaccadeTrainer(model_no_wave, train_loader, val_loader, test_loader, num_classes=num_classes, device=device)
    res_no_wave = trainer_no_wave.fit(num_epochs=args.epochs)["test_metrics"]
    res_no_wave["Method"] = "InfoSaccade w/o Wavelet"
    results_table.append(res_no_wave)

    print(f"\n[3/5] Training Ablation: InfoSaccade-RL w/o Early Stopping (Fixed Steps)...")
    model_fixed = InfoSaccadeModel(img_channels=img_channels, num_classes=num_classes, use_wavelet=True, enable_early_stopping=False)
    trainer_fixed = InfoSaccadeTrainer(model_fixed, train_loader, val_loader, test_loader, num_classes=num_classes, device=device)
    res_fixed = trainer_fixed.fit(num_epochs=args.epochs)["test_metrics"]
    res_fixed["Method"] = "InfoSaccade w/o Early Stop"
    results_table.append(res_fixed)

    print(f"\n[4/5] Training Baseline: Standard ResNet-18 (Dense Supervised)...")
    resnet_model = BaselineResNet(in_channels=img_channels, num_classes=num_classes)
    res_resnet = train_baseline_supervised(resnet_model, train_loader, val_loader, test_loader, epochs=args.epochs, device=device)
    res_resnet["Method"] = "Baseline ResNet-18"
    results_table.append(res_resnet)

    print(f"\n[5/5] Training Baseline: Standard RAM (Recurrent Models of Visual Attention)...")
    ram_model = StandardRAMBaseline(in_channels=img_channels, num_classes=num_classes)
    res_ram = train_baseline_supervised(ram_model, train_loader, val_loader, test_loader, epochs=args.epochs, device=device)
    res_ram["Method"] = "Baseline RAM (Mnih et al.)"
    results_table.append(res_ram)

    df = pd.DataFrame(results_table)
    # Reorder columns
    cols = ["Method", "accuracy", "auc", "f1_macro", "ece", "avg_steps", "efficiency_gain_pct"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.save_table, index=False)

    print(f"\n=======================================================")
    print(f" FINAL ABLATION & BENCHMARK SUMMARY TABLE:")
    print(df.to_string(index=False))
    print(f"=======================================================\n")
    print(f"Saved complete comparative table to {args.save_table}")


if __name__ == "__main__":
    main()
