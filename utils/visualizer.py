"""
Visualization utilities for publication-quality figures:
1. Saccadic Scanpath overlaid on Medical Images.
2. Multiresolution Wavelet Sub-band Decomposition.
3. Evidential Uncertainty & Entropy Collapse across Fixation Steps.
4. Reliability Calibration Diagrams.
"""

import os
from typing import List, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def plot_saccade_scanpath(
    image: np.ndarray,
    loc_history: List[np.ndarray],
    glimpse_size: int = 14,
    true_label: Optional[str] = None,
    pred_label: Optional[str] = None,
    uncertainty_history: Optional[List[float]] = None,
    save_path: str = "scanpath.png"
):
    """
    Plots the sequential visual fixation scanpath overlaid on a medical image.
    image: (H, W, C) or (H, W) in [0, 1]
    loc_history: list of (x, y) coordinates in [-1, 1]
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    H, W = image.shape[:2]
    colors = plt.cm.plasma(np.linspace(0.2, 1.0, len(loc_history)))

    # Convert [-1, 1] coordinates to pixel coordinates
    pixel_coords = []
    for (lx, ly) in loc_history:
        px = (lx + 1.0) * 0.5 * (W - 1)
        py = (ly + 1.0) * 0.5 * (H - 1)
        pixel_coords.append((px, py))

    # Draw scanpath lines
    for i in range(len(pixel_coords) - 1):
        x1, y1 = pixel_coords[i]
        x2, y2 = pixel_coords[i + 1]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.5, alpha=0.85)
        )

    # Draw fixation bounding boxes & markers
    patch_w = glimpse_size
    patch_h = glimpse_size
    for i, (px, py) in enumerate(pixel_coords):
        # Draw bounding box
        rect = patches.Rectangle(
            (px - patch_w / 2, py - patch_h / 2),
            patch_w,
            patch_h,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            linestyle="--" if i < len(pixel_coords) - 1 else "-"
        )
        ax.add_patch(rect)
        ax.plot(px, py, "o", color=colors[i], markersize=8)
        ax.text(px + 1, py - 1, f"t={i}", color="white", fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.8))

    title_str = "InfoSaccade Saccadic Scanpath"
    if true_label is not None and pred_label is not None:
        title_str += f"\nTrue: {true_label} | Pred: {pred_label}"
    ax.set_title(title_str, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_uncertainty_entropy_dynamics(
    entropies_per_step: List[float],
    uncertainties_per_step: List[float],
    save_path: str = "uncertainty_dynamics.png"
):
    """
    Plots the step-wise monotonic reduction of predictive entropy and evidential vacuity.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    steps = list(range(len(entropies_per_step)))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_ent = "#1f77b4"
    ax1.set_xlabel("Fixation Step $t$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Shannon Predictive Entropy $H(Y|h_t)$", color=color_ent, fontsize=11, fontweight="bold")
    line1 = ax1.plot(steps, entropies_per_step, color=color_ent, marker="o", linewidth=2.5, label="Entropy $H(Y)$")
    ax1.tick_params(axis="y", labelcolor=color_ent)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_unc = "#d62728"
    ax2.set_ylabel("Evidential Vacuity $u_t = K/S$", color=color_unc, fontsize=11, fontweight="bold")
    line2 = ax2.plot(steps, uncertainties_per_step, color=color_unc, marker="s", linestyle="--", linewidth=2.5, label="Vacuity $u_t$")
    ax2.tick_params(axis="y", labelcolor=color_unc)

    plt.title("Information-Theoretic Ambiguity Collapse during Saccades", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_wavelet_subbands(
    image: np.ndarray,
    subbands: Dict[str, np.ndarray],
    save_path: str = "wavelet_decomposition.png"
):
    """
    Visualizes the 2D Wavelet Multiresolution Sub-bands: LL, LH, HL, HH.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

    titles = ["Original Image", "LL (Approximation)", "LH (Horizontal Detail)", "HL (Vertical Detail)", "HH (Diagonal Detail)"]
    images_to_show = [image, subbands["LL"], subbands["LH"], subbands["HL"], subbands["HH"]]

    for ax, title, img in zip(axes, titles, images_to_show):
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            # Channel first to channel last
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        # Normalize for visualization
        norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if norm_img.ndim == 2:
            ax.imshow(norm_img, cmap="magma")
        else:
            ax.imshow(norm_img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Multiresolution 2D Wavelet Sub-Band Decomposition in InfoSaccade", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
