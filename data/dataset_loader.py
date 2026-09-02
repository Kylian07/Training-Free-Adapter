"""
Dataset loader module for MedMNIST v2 benchmarks and custom medical image sets.
Supports: PathMNIST, DermaMNIST, BloodMNIST, ChestMNIST, OrganAMNIST, etc.
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SyntheticMedicalDataset(Dataset):
    """
    Synthetic multi-class medical lesion dataset for instant unit testing and offline execution.
    Generates realistic textured patches with class-dependent spatial lesion features.
    """
    def __init__(self, num_samples: int = 500, img_channels: int = 3, img_size: int = 28, num_classes: int = 9):
        self.num_samples = num_samples
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes

        # Deterministic generation
        np.random.seed(42)
        self.data = np.random.randn(num_samples, img_channels, img_size, img_size).astype(np.float32) * 0.2 + 0.5
        self.labels = np.random.randint(0, num_classes, size=(num_samples, 1)).astype(np.int64)

        # Inject diagnostic patterns correlated with class
        for i in range(num_samples):
            cls = self.labels[i, 0]
            # Lesion center
            cx = (cls % 3) * (img_size // 3) + img_size // 6
            cy = (cls // 3) * (img_size // 3) + img_size // 6
            r = max(2, img_size // 8)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            self.data[i, :, mask] += 0.5

        self.data = np.clip(self.data, 0.0, 1.0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).squeeze(-1).long()
        return img, label


def get_medmnist_loaders(
    dataset_name: str = "pathmnist",
    batch_size: int = 128,
    data_dir: str = "./data_cache",
    download: bool = True,
    num_workers: int = 2,
    as_rgb: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Loads Train, Validation, and Test loaders for the requested MedMNIST dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_name_lower = dataset_name.lower()

    # Pre-calculated mean and standard deviation for MedMNIST
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    info = {}
    try:
        import medmnist
        from medmnist import INFO

        if dataset_name_lower not in INFO:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(INFO.keys())}")

        info = INFO[dataset_name_lower]
        DataClass = getattr(medmnist, info["python_class"])

        train_dataset = DataClass(split="train", transform=transform_train, download=download, root=data_dir, as_rgb=as_rgb)
        val_dataset = DataClass(split="val", transform=transform_eval, download=download, root=data_dir, as_rgb=as_rgb)
        test_dataset = DataClass(split="test", transform=transform_eval, download=download, root=data_dir, as_rgb=as_rgb)

    except Exception as e:
        print(f"[Warning] Could not load MedMNIST directly ({e}). Falling back to SyntheticMedicalDataset.")
        train_dataset = SyntheticMedicalDataset(num_samples=1000, img_channels=3 if as_rgb else 1, img_size=28, num_classes=9)
        val_dataset = SyntheticMedicalDataset(num_samples=200, img_channels=3 if as_rgb else 1, img_size=28, num_classes=9)
        test_dataset = SyntheticMedicalDataset(num_samples=300, img_channels=3 if as_rgb else 1, img_size=28, num_classes=9)
        info = {
            "n_channels": 3 if as_rgb else 1,
            "n_classes": 9,
            "label": {str(i): f"Class {i}" for i in range(9)},
            "task": "multi-class",
            "description": "Synthetic Medical Benchmark"
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, info
