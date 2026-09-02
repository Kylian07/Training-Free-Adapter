"""
Differentiable 2D Discrete Wavelet Transform (DWT) Module.
Decomposes medical images/patches into multiresolution frequency sub-bands:
- LL: Approximation (low-frequency anatomical baseline)
- LH: Horizontal detail (horizontal edge/vessel structures)
- HL: Vertical detail (vertical tissue boundaries)
- HH: Diagonal detail (high-frequency micro-calcifications / cellular textures)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableDWT2D(nn.Module):
    """
    Differentiable 2D Discrete Wavelet Transform (Haar / Orthonormal Filter).
    Operates on (B, C, H, W) and outputs 4 sub-bands:
    - (B, C, H/2, W/2) for each of LL, LH, HL, HH
    Can also stack sub-bands into (B, 4*C, H/2, W/2) or return discrete sub-band tensor.
    """
    def __init__(self, in_channels: int = 3, wavelet: str = "haar", learnable: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = wavelet.lower()

        # Haar 1D low-pass and high-pass filters (scaled by 1/sqrt(2) for energy conservation)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        h = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)  # Low-pass
        g = torch.tensor([-inv_sqrt2, inv_sqrt2], dtype=torch.float32) # High-pass

        # 2D Separable Filter Kernels: LL, LH, HL, HH
        ll = torch.outer(h, h)  # (2, 2)
        lh = torch.outer(h, g)  # (2, 2)
        hl = torch.outer(g, h)  # (2, 2)
        hh = torch.outer(g, g)  # (2, 2)

        # Stack into 4 filters of shape (4, 1, 2, 2)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        
        # Repeat across in_channels for depthwise separable conv: (4 * in_channels, 1, 2, 2)
        # Using groups=in_channels, weight shape: (4 * in_channels, 1, 2, 2)
        weight = filters.repeat(in_channels, 1, 1, 1)

        if learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            subbands: Dict with keys 'LL', 'LH', 'HL', 'HH', each (B, C, H/2, W/2)
            stacked: Tensor of shape (B, C, 4, H/2, W/2) or (B, 4*C, H/2, W/2)
        """
        B, C, H, W = x.shape
        # Ensure even dimensions via reflection padding if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Depthwise 2D conv with stride 2
        out = F.conv2d(x, self.weight, stride=2, groups=C)  # (B, 4*C, H/2, W/2)
        
        # Reshape to (B, C, 4, H/2, W/2) -> dim 2: [0:LL, 1:LH, 2:HL, 3:HH]
        out_reshaped = out.view(B, C, 4, out.shape[2], out.shape[3])
        
        subbands = {
            "LL": out_reshaped[:, :, 0],
            "LH": out_reshaped[:, :, 1],
            "HL": out_reshaped[:, :, 2],
            "HH": out_reshaped[:, :, 3]
        }
        return subbands, out_reshaped

    def select_subband(self, x: torch.Tensor, subband_idx: torch.Tensor) -> torch.Tensor:
        """
        Differentiable / indexed selection of sub-band per batch item.
        Args:
            x: Input image (B, C, H, W)
            subband_idx: LongTensor of shape (B,) with indices in {0, 1, 2, 3} (0:LL, 1:LH, 2:HL, 3:HH)
        Returns:
            selected: Tensor of shape (B, C, H/2, W/2)
        """
        _, out_reshaped = self.forward(x)  # (B, C, 4, H/2, W/2)
        B, C, _, H_half, W_half = out_reshaped.shape
        
        # Gather indexed sub-band: shape (B, C, 1, H/2, W/2)
        idx_expanded = subband_idx.view(B, 1, 1, 1, 1).expand(B, C, 1, H_half, W_half)
        selected = torch.gather(out_reshaped, dim=2, index=idx_expanded).squeeze(2)
        return selected


class SpectralSpatialFusion(nn.Module):
    """
    Fuses spatial patch features with wavelet sub-band coefficients
    using cross-spectral channel attention.
    """
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.dwt = DifferentiableDWT2D(in_channels=in_channels)
        
        # Conv block processing all 4 wavelet sub-bands stacked (4 * in_channels)
        self.spec_conv = nn.Sequential(
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
        """
        Args:
            patch: (B, C, H, W)
        Returns:
            spectral_emb: (B, out_dim)
        """
        _, out_reshaped = self.dwt(patch) # (B, C, 4, H/2, W/2)
        B, C, _, H2, W2 = out_reshaped.shape
        stacked = out_reshaped.view(B, C * 4, H2, W2)
        return self.spec_conv(stacked)
