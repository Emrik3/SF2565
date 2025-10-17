"""
Optimized training script for handling class imbalance.
Combines WeightedRandomSampler with pos_weight for maximum F1 score improvement.

This script is specifically designed for the 6:1 class imbalance
(Sick=0: 86%, Healthy=1: 14%)
"""

import torch
from config import TrainingConfig
from trainer import FeatureNNTrainer

torch.multiprocessing.set_sharing_strategy("file_system")


def train_weighted_sampler_only():
    """
    Using only WeightedRandomSampler (no pos_weight).
    """
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="balanced_sampler_only",
        hidden_dims=[512, 256],
        num_epochs=40,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.5,
        use_weighted_sampler=True,  # Only this enabled
        pos_weight=None,  # Not using pos_weight
        verbose=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print("\n" + metrics.summary())

    return metrics


def train_pos_weight_only():
    """
    Using only pos_weight (no WeightedRandomSampler).
    """
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="balanced_5050",
        hidden_dims=[512, 256],
        num_epochs=40,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.5,
        use_weighted_sampler=False,  # Not using sampler
        pos_weight=6.0,  # Only this enabled
        verbose=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print("\n" + metrics.summary())

    return metrics


def train_aggressive_balance():
    """
    Agressive for high recall.
    """
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="balanced_aggressive",
        hidden_dims=[1024, 512, 256],
        dropout_rate=0.7,  # Very high dropout
        num_epochs=60,  # More epochs
        batch_size=64,
        learning_rate=0.0003,  # Even lower LR
        weight_decay=0.0001,
        use_weighted_sampler=True,
        # pos_weight=8.0,  # Even higher weight for minority class
        use_scheduler=True,
        scheduler_type="cosine",
        early_stopping_patience=20,
        verbose=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print("\n" + metrics.summary())

    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "sampler":
            train_weighted_sampler_only()
        elif mode == "posweight":
            train_pos_weight_only()
        elif mode == "aggressive":
            train_aggressive_balance()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: best, sampler, posweight, aggressive, compare")
            sys.exit(1)
