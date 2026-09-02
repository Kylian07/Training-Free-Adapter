"""
CLI Training Script for InfoSaccade-RL.
Usage:
    python scripts/train.py --dataset pathmnist --epochs 30 --batch_size 128
"""

import os
import sys
import argparse

# Add repository root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from configs.config import build_config
from data.dataset_loader import get_medmnist_loaders
from models.infosaccade_model import InfoSaccadeModel
from trainers.trainer import InfoSaccadeTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train InfoSaccade-RL on Medical Benchmarks")
    parser.add_argument("--dataset", type=str, default="pathmnist", help="MedMNIST dataset name (e.g. pathmnist, dermamnist, bloodmnist)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (128 recommended for Kaggle T4)")
    parser.add_argument("--max_steps", type=int, default=6, help="Maximum visual saccade fixations per image")
    parser.add_argument("--glimpse_size", type=int, default=14, help="Spatial glimpse patch dimension")
    parser.add_argument("--lr_backbone", type=float, default=1e-3, help="Learning rate for visual encoder and classifier")
    parser.add_argument("--lr_rl", type=float, default=5e-4, help="Learning rate for PPO Actor-Critic")
    parser.add_argument("--no_wavelet", action="store_true", help="Disable wavelet spectral decomposition (for ablation)")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable dynamic early stopping")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--data_dir", type=str, default="./data_cache", help="Directory to download/cache datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"--> Initializing Experiment for dataset: {args.dataset}")

    # Build config
    cfg = build_config(dataset_name=args.dataset, max_steps=args.max_steps)
    cfg.model.use_wavelet = not args.no_wavelet
    cfg.model.enable_early_stopping = not args.no_early_stop
    cfg.model.glimpse_size = args.glimpse_size
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.save_dir = os.path.join(args.save_dir, args.dataset)

    # Load data
    train_loader, val_loader, test_loader, ds_info = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=cfg.train.batch_size,
        data_dir=args.data_dir,
        download=True
    )
    num_classes = ds_info.get("n_classes", cfg.model.num_classes)
    img_channels = ds_info.get("n_channels", cfg.model.img_channels)
    print(f"--> Dataset Loaded: {args.dataset} | Classes: {num_classes} | Channels: {img_channels}")

    # Instantiate Model
    model = InfoSaccadeModel(
        img_channels=img_channels,
        num_classes=num_classes,
        glimpse_size=cfg.model.glimpse_size,
        num_scales=cfg.model.num_scales,
        glimpse_emb_dim=cfg.model.glimpse_emb_dim,
        rnn_hidden_dim=cfg.model.rnn_hidden_dim,
        max_steps=cfg.model.max_steps,
        use_wavelet=cfg.model.use_wavelet,
        enable_early_stopping=cfg.model.enable_early_stopping
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> InfoSaccade-RL Parameter Count: {total_params:,} (Extremely lightweight & fast on Kaggle T4!)")

    # Trainer
    trainer = InfoSaccadeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        lr_backbone=args.lr_backbone,
        lr_actor_critic=args.lr_rl,
        save_dir=cfg.train.save_dir,
        use_amp=True
    )

    results = trainer.fit(num_epochs=cfg.train.epochs)
    print(f"\nExperiment finished successfully. Metrics saved to {cfg.train.save_dir}")


if __name__ == "__main__":
    main()
