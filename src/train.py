"""
Main training script for feature-based neural network.
Easy-to-use interface for training with different configurations.
"""

import argparse
import sys
import torch

from config import TrainingConfig
from trainer import FeatureNNTrainer

torch.multiprocessing.set_sharing_strategy("file_system")


def train_default():
    """Train with default configuration"""
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="default_run_using_pos_weight",
        hidden_dims=[512, 256],
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.5,
        verbose=True,
        use_weighted_sampler=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Experiment directory: {config.get_experiment_dir()}")
    print(metrics.summary())


def train_simple():
    """Train a simple 2-layer network"""
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="simple_2layer",
        hidden_dims=[1024],
        num_epochs=50,
        batch_size=128,
        learning_rate=0.001,
        dropout_rate=0.3,
        activation="relu",
        batch_norm=True,
        verbose=True,
        use_weighted_sampler=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print(metrics.summary())


def train_deep():
    """Train a deeper network with multiple layers"""
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="deep_4layer",
        hidden_dims=[1024, 512, 256, 128],
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.4,
        activation="relu",
        batch_norm=True,
        use_scheduler=True,
        scheduler_type="cosine",
        verbose=True,
        use_weighted_sampler=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print(metrics.summary())


def train_high_dropout():
    """Train with high dropout for regularization"""
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="high_dropout",
        hidden_dims=[512, 256],
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.7,
        activation="relu",
        weight_decay=0.001,
        verbose=True,
        use_weighted_sampler=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print(metrics.summary())


def train_custom(args):
    """Train with custom parameters from command line"""
    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    config = TrainingConfig(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        hidden_dims=hidden_dims,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        activation=args.activation,
        batch_norm=args.batch_norm,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        scheduler_type=args.scheduler,
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        pos_weight=args.pos_weight,
        use_weighted_sampler=args.use_weighted_sampler,
        verbose=True,
    )

    trainer = FeatureNNTrainer(config)
    metrics = trainer.train()
    print(metrics.summary())


def main():
    parser = argparse.ArgumentParser(
        description="Train feature-based neural network for medical image classification"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "simple", "deep", "high_dropout", "custom"],
        help="Training mode (default: default)",
    )

    # Custom training parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/xray_features_frontal_only.pt",
        help="Path to pre-extracted features",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="custom_exp", help="Experiment name"
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512,256",
        help="Hidden layer dimensions (comma-separated, e.g., '512,256,128')",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "elu", "gelu"],
        help="Activation function",
    )
    parser.add_argument(
        "--batch_norm",
        type=bool,
        default=True,
        help="Use batch normalization",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument(
        "--use_scheduler", type=bool, default=True, help="Use learning rate scheduler"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "plateau"],
        help="Scheduler type",
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=True, help="Use early stopping"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive class weight for BCEWithLogitsLoss (for class imbalance)",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use WeightedRandomSampler to balance classes during training",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Feature-Based Neural Network Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # Run training based on mode
    if args.mode == "default":
        train_default()
    elif args.mode == "simple":
        train_simple()
    elif args.mode == "deep":
        train_deep()
    elif args.mode == "high_dropout":
        train_high_dropout()
    elif args.mode == "custom":
        train_custom(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
