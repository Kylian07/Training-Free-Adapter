"""
Configuration module for InfoSaccade-RL.
Includes presets for Kaggle T4 GPU, local CPU/GPU execution, and dataset configurations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class ModelConfig:
    # Input Image Dimensions
    img_channels: int = 3
    img_size: int = 28  # Can be 28 (MedMNIST standard) or 224 (MedMNIST+ / ISIC)
    num_classes: int = 9  # e.g., PathMNIST (9), DermaMNIST (7), BloodMNIST (8)

    # Glimpse / Saccade Receptive Field
    glimpse_size: int = 14  # Size of cropped patch
    num_scales: int = 2     # Multiscale glimpse (coarse context + fine detail)
    scale_factor: float = 1.5

    # Wavelet / Spectral Parameters
    use_wavelet: bool = True
    wavelet_type: str = "haar"  # 'haar', 'db1'
    spectral_channels: int = 4  # LL, LH, HL, HH sub-bands

    # Feature Encoders
    glimpse_emb_dim: int = 128
    rnn_hidden_dim: int = 256
    backbone_type: str = "convnet"  # 'convnet' or 'resnet_small'

    # Saccade Trajectory
    max_steps: int = 6  # Maximum fixation steps per image
    enable_early_stopping: bool = True
    stopping_threshold: float = 0.85  # Vacuity confidence threshold to trigger early stop


@dataclass
class RLConfig:
    # PPO Hyperparameters
    gamma: float = 0.95              # Discount factor
    gae_lambda: float = 0.95         # Generalized Advantage Estimation parameter
    ppo_clip_eps: float = 0.2        # PPO clipping epsilon
    ppo_epochs: int = 4              # PPO update epochs per rollout
    entropy_coef: float = 0.02       # Action exploration bonus
    value_loss_coef: float = 0.5     # Value function loss coefficient
    
    # Continuous Action (Location) Variance
    init_action_std: float = 0.5
    min_action_std: float = 0.1
    action_std_decay: float = 0.9995

    # Reward Formulation Hyperparameters
    reward_mode: str = "info_gain"   # 'info_gain', 'evidential_delta', 'sparse_acc'
    info_gain_weight: float = 1.0    # Mutual information reduction weight
    uncertainty_weight: float = 0.5  # Subjective logic vacuity reduction weight
    step_cost: float = 0.05          # Inspection cost penalty per saccade
    terminal_acc_reward: float = 2.0 # Reward on correct terminal classification
    terminal_err_penalty: float = -1.0 # Penalty on incorrect terminal classification


@dataclass
class TrainingConfig:
    # Optimization
    seed: int = 42
    dataset_name: str = "pathmnist"  # 'pathmnist', 'dermamnist', 'bloodmnist', 'chestmnist', 'organamnist'
    data_dir: str = "./data_cache"
    download: bool = True

    # Batch and Epochs (Optimized for Kaggle T4 GPU)
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 30
    lr_backbone: float = 1e-3
    lr_actor_critic: float = 5e-4
    weight_decay: float = 1e-4
    use_amp: bool = True             # Mixed precision (torch.cuda.amp) for Kaggle T4

    # Checkpoints & Logs
    save_dir: str = "./checkpoints"
    eval_freq: int = 1
    log_freq: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FullExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)


def get_dataset_config(dataset_name: str) -> dict:
    """Pre-configured specs for popular MedMNIST datasets."""
    specs = {
        "pathmnist": {"channels": 3, "classes": 9, "task": "multi-class", "description": "Colon Pathology"},
        "dermamnist": {"channels": 3, "classes": 7, "task": "multi-class", "description": "Dermatology Lesions"},
        "bloodmnist": {"channels": 3, "classes": 8, "task": "multi-class", "description": "Blood Cell Microscope"},
        "chestmnist": {"channels": 1, "classes": 14, "task": "multi-label", "description": "Chest X-Ray"},
        "organamnist": {"channels": 1, "classes": 11, "task": "multi-class", "description": "Abdominal CT Slices"},
        "retinamnist": {"channels": 3, "classes": 5, "task": "ordinal", "description": "Retina Fundus Photography"},
        "octmnist": {"channels": 1, "classes": 4, "task": "multi-class", "description": "Retinal OCT"},
        "pneumoniamnist": {"channels": 1, "classes": 2, "task": "binary", "description": "Pediatric Chest X-Ray"}
    }
    return specs.get(dataset_name.lower(), {"channels": 3, "classes": 9, "task": "multi-class", "description": "Custom"})


def build_config(dataset_name: str = "pathmnist", max_steps: int = 6, img_size: int = 28) -> FullExperimentConfig:
    cfg = FullExperimentConfig()
    ds_meta = get_dataset_config(dataset_name)
    cfg.train.dataset_name = dataset_name
    cfg.model.img_channels = ds_meta["channels"]
    cfg.model.num_classes = ds_meta["classes"]
    cfg.model.img_size = img_size
    cfg.model.glimpse_size = max(8, img_size // 2)
    cfg.model.max_steps = max_steps
    return cfg
