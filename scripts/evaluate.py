"""
Evaluation and Visualization Script for trained InfoSaccade-RL models.
Loads a checkpoint and produces publication figures (Scanpaths, Wavelets, Calibration).
"""

import os
import sys
import argparse

# Add repository root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from data.dataset_loader import get_medmnist_loaders
from models.infosaccade_model import InfoSaccadeModel
from trainers.evaluator import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pathmnist/best_infosaccade_model.pth")
    parser.add_argument("--dataset", type=str, default="pathmnist")
    parser.add_argument("--output_dir", type=str, default="./results/eval_plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, ds_info = get_medmnist_loaders(dataset_name=args.dataset, batch_size=64, download=True)

    num_classes = ds_info.get("n_classes", 9)
    img_channels = ds_info.get("n_channels", 3)

    model = InfoSaccadeModel(img_channels=img_channels, num_classes=num_classes)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} (Epoch {ckpt.get('epoch', 'N/A')})")
    else:
        print(f"[Warning] Checkpoint {args.checkpoint} not found. Running with initialized weights for visualization.")

    evaluator = BenchmarkEvaluator(model, test_loader, device=device)
    metrics = evaluator.run_full_evaluation(output_dir=args.output_dir)

    print("\n--- Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
