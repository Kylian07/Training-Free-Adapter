"""
Differentiable Spatial Glimpse Sensor and Multiscale Patch Extractor.
Extracts dynamic spatial patches at continuous coordinates (x, y) with variable scale (zoom)
using bilinear grid sampling (Spatial Transformer / STN mechanism).
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavelet_transform import SpectralSpatialFusion


class SpatialGlimpseSensor(nn.Module):
    """
    Extracts multi-scale glimpse patches centered at continuous coordinates (l_x, l_y) in [-1, 1]^2.
    For each location, extracts:
      - Scale 1: High-resolution fine patch of size (glimpse_size x glimpse_size)
      - Scale 2: Coarser context patch (e.g. 1.5x larger field-of-view, downscaled)
    Then passes patches through a visual convolutional encoder + location coordinate encoder.
    """
    def __init__(
        self,
        img_channels: int = 3,
        glimpse_size: int = 14,
        num_scales: int = 2,
        scale_factor: float = 1.5,
        emb_dim: int = 128,
        use_wavelet: bool = True
    ):
        super().__init__()
        self.img_channels = img_channels
        self.glimpse_size = glimpse_size
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.emb_dim = emb_dim
        self.use_wavelet = use_wavelet

        # Visual patch convolutional encoder
        # Total input channels = img_channels * num_scales
        in_ch = img_channels * num_scales
        self.patch_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
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

        # Wavelet spectral encoder (optional, complementary stream)
        if use_wavelet:
            self.spectral_fusion = SpectralSpatialFusion(in_channels=img_channels, out_dim=emb_dim // 2)
            fusion_in = emb_dim + (emb_dim // 2) + 32
        else:
            fusion_in = emb_dim + 32

        # Location coordinate MLP (encodes (x, y, scale))
        self.loc_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True)
        )

        # Joint fusion layer
        self.fc_fuse = nn.Sequential(
            nn.Linear(fusion_in, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True)
        )

    def extract_patch(self, images: torch.Tensor, locs: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Differentiably extracts rectangular patches from images centered at locs with given scale.
        images: (B, C, H, W)
        locs: (B, 2) with coordinates in [-1, 1]
        scale: float or (B, 1) scale factor relative to full image
        """
        B, C, H, W = images.shape
        grid_size = self.glimpse_size
        
        # Base normalized grid in [-1, 1]
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size, device=images.device),
            torch.linspace(-1, 1, grid_size, device=images.device),
            indexing="ij"
        ), dim=-1).flip(-1)  # (grid_size, grid_size, 2), (x, y) order
        
        base_grid = base_grid.unsqueeze(0).expand(B, grid_size, grid_size, 2)  # (B, H_g, W_g, 2)
        
        # Scale and shift grid
        # locs: (B, 2) -> (B, 1, 1, 2)
        locs_expanded = locs.unsqueeze(1).unsqueeze(1)
        
        if isinstance(scale, torch.Tensor):
            scale_expanded = scale.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 1)
            transformed_grid = base_grid * scale_expanded + locs_expanded
        else:
            transformed_grid = base_grid * scale + locs_expanded
            
        # Bilinear sampling from full image
        patches = F.grid_sample(images, transformed_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return patches

    def forward(self, images: torch.Tensor, locs: torch.Tensor, scale_action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W)
            locs: (B, 2) coordinates in [-1, 1]
            scale_action: (B, 1) optional dynamic zoom in [0.2, 1.0]
        Returns:
            glimpse_emb: (B, emb_dim) fused feature embedding
            patches_stacked: (B, C * num_scales, glimpse_size, glimpse_size)
        """
        B = images.shape[0]
        if scale_action is None:
            base_scale = 0.5
        else:
            base_scale = scale_action  # e.g., (B, 1)

        scale_list = []
        for s_idx in range(self.num_scales):
            current_scale = base_scale * (self.scale_factor ** s_idx)
            # Clamped so it doesn't exceed image boundaries
            if isinstance(current_scale, torch.Tensor):
                current_scale = torch.clamp(current_scale, 0.1, 1.2)
            else:
                current_scale = min(max(current_scale, 0.1), 1.2)
            
            p = self.extract_patch(images, locs, current_scale)
            scale_list.append(p)

        patches_stacked = torch.cat(scale_list, dim=1)  # (B, C * num_scales, glimpse_size, glimpse_size)
        visual_feat = self.patch_conv(patches_stacked)  # (B, emb_dim)

        # Coordinate features: (x, y, scale)
        if scale_action is None:
            coord_input = torch.cat([locs, torch.ones(B, 1, device=images.device) * 0.5], dim=-1)
        else:
            coord_input = torch.cat([locs, scale_action], dim=-1)
        loc_feat = self.loc_mlp(coord_input)  # (B, 32)

        if self.use_wavelet:
            fine_patch = scale_list[0]  # Take fine patch for wavelet analysis
            spec_feat = self.spectral_fusion(fine_patch)  # (B, emb_dim // 2)
            fused_feat = torch.cat([visual_feat, spec_feat, loc_feat], dim=-1)
        else:
            fused_feat = torch.cat([visual_feat, loc_feat], dim=-1)

        glimpse_emb = self.fc_fuse(fused_feat)  # (B, emb_dim)
        return glimpse_emb, patches_stacked
