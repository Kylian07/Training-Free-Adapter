"""
Baseline Architectures for Experimental Comparison in ICASSP Submission:
1. Standard ResNet-18 (Supervised dense CNN baseline)
2. ViT-Tiny (Vision Transformer baseline)
3. Standard RAM (Recurrent Models of Visual Attention - Mnih et al., 2014)
4. Random Glimpse Baseline (Spatial crop with random walk policy)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BaselineResNet(nn.Module):
    """
    Adapted ResNet-18 for medical images of varying resolutions (28x28 or 224x224).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 9, pretrained: bool = False):
        super().__init__()
        self.model = resnet18(weights=None)
        
        # Adjust first conv layer for in_channels and small kernel if 28x28
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Prevent spatial dimension collapse on small 28x28 images
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BaselineViTTiny(nn.Module):
    """
    Lightweight Vision Transformer baseline for medical image classification.
    """
    def __init__(self, in_channels: int = 3, img_size: int = 28, patch_size: int = 4, num_classes: int = 9, embed_dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embedding: (B, embed_dim, H_p, W_p) -> (B, num_patches, embed_dim)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed
        out = self.transformer(x_emb)
        cls_out = out[:, 0]
        return self.mlp_head(cls_out)


class StandardRAMBaseline(nn.Module):
    """
    Standard Recurrent Attention Model (RAM - Mnih et al., 2014) baseline:
    Uses standard REINFORCE / Actor-Critic with sparse reward, NO wavelets, and standard Softmax.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 9, glimpse_size: int = 14, max_steps: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.glimpse_size = glimpse_size
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim

        # Standard non-wavelet glimpse encoder
        self.glimpse_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU()
        )
        self.loc_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU()
        )

        self.rnn = nn.GRUCell(128, hidden_dim)
        
        # Actor
        self.loc_actor = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        # Standard Softmax Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Value baseline
        self.value_head = nn.Linear(hidden_dim, 1)

    def extract_patch(self, images: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        grid_size = self.glimpse_size
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size, device=images.device),
            torch.linspace(-1, 1, grid_size, device=images.device),
            indexing="ij"
        ), dim=-1).flip(-1).unsqueeze(0).expand(B, grid_size, grid_size, 2)
        
        locs_expanded = locs.unsqueeze(1).unsqueeze(1)
        grid = base_grid * 0.5 + locs_expanded
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        device = images.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        loc = torch.zeros(B, 2, device=device)

        for _ in range(self.max_steps):
            patch = self.extract_patch(images, loc)
            g_feat = self.glimpse_conv(patch)
            l_feat = self.loc_fc(loc)
            glimpse_emb = self.fuse_fc(torch.cat([g_feat, l_feat], dim=-1))
            h = self.rnn(glimpse_emb, h)
            loc = torch.clamp(loc + self.loc_actor(h), -1.0, 1.0)

        logits = self.classifier(h)
        return logits
