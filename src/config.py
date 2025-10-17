"""
Configuration and metrics tracking for feature-based neural network training.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for training the feature-based neural network.

    All hyperparameters can be easily adjusted here.
    """

    # Data paths
    data_path: str = "data/xray_features_frontal_only.pt"
    output_dir: str = "experiments"
    experiment_name: str = "feature_nn_exp"

    # Model architecture
    input_dim: int = 2048  # ResNet50 feature dimension
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_classes: int = 1  # Binary classification
    dropout_rate: float = 0.5
    activation: str = "relu"  # 'relu', 'leaky_relu', 'elu', 'gelu'
    batch_norm: bool = True

    # Training parameters
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Optimizer settings
    optimizer: str = "adam"  # 'adam', 'sgd', 'adamw'
    momentum: float = 0.9  # Only for SGD

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "step"  # 'step', 'cosine', 'plateau'
    scheduler_step_size: int = 10  # For StepLR
    scheduler_gamma: float = 0.5  # For StepLR
    scheduler_patience: int = 5  # For ReduceLROnPlateau

    # Data split
    train_split: float = 0.8
    val_split: float = 0.2
    random_seed: int = 42

    # Loss function
    pos_weight: Optional[float] = (
        None  # For handling class imbalance (BCEWithLogitsLoss)
    )

    # Class imbalance handling
    use_weighted_sampler: bool = False  # Use WeightedRandomSampler for training

    # Training options
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    save_best_only: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Logging
    print_every: int = 10  # Print training stats every N batches
    save_plots: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = asdict(self)
        # Ensure all values are JSON serializable
        for key, value in config_dict.items():
            if hasattr(value, "tolist"):  # Handle torch tensors or numpy arrays
                config_dict[key] = value.tolist()
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = str(value)
        return config_dict

    def save(self, path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def get_experiment_dir(self) -> str:
        """Get the full path to the experiment directory"""
        return os.path.join(self.output_dir, self.experiment_name)

    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = ["=" * 60, "Training Configuration", "=" * 60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


class MetricsTracker:
    """
    Tracks and stores training metrics across epochs.
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_aucs = []
        self.val_aucs = []
        self.learning_rates = []
        self.epoch_times = []

        # Per-epoch metrics
        self.epoch_metrics = []

    def update_train(self, loss, accuracy, f1_score, auc):
        """Update training metrics"""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        self.train_f1_scores.append(f1_score)
        self.train_aucs.append(auc)

    def update_val(self, loss, accuracy, f1_score, auc):
        """Update validation metrics"""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1_score)
        self.val_aucs.append(auc)

    def update_lr(self, lr):
        """Update learning rate"""
        self.learning_rates.append(lr)

    def update_epoch_time(self, time):
        """Update epoch time"""
        self.epoch_times.append(time)

    def add_epoch_metrics(self, epoch, metrics_dict):
        """Add complete metrics for an epoch"""
        metrics_dict["epoch"] = epoch
        self.epoch_metrics.append(metrics_dict)

    def get_best_epoch(self, metric="val_accuracy"):
        """Get the epoch with the best performance"""
        if not self.epoch_metrics:
            return None

        if metric == "val_loss":
            # Lower is better for loss
            best_epoch = min(
                self.epoch_metrics, key=lambda x: x.get(metric, float("inf"))
            )
        else:
            # Higher is better for accuracy, f1, auc
            best_epoch = max(self.epoch_metrics, key=lambda x: x.get(metric, 0))

        return best_epoch

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics"""
        if not self.epoch_metrics:
            return {}
        return self.epoch_metrics[-1]

    def save(self, path: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        metrics_dict = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "train_f1_scores": self.train_f1_scores,
            "val_f1_scores": self.val_f1_scores,
            "train_aucs": self.train_aucs,
            "val_aucs": self.val_aucs,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "epoch_metrics": self.epoch_metrics,
        }

        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        print(f"Metrics saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load metrics from JSON file"""
        with open(path, "r") as f:
            metrics_dict = json.load(f)

        tracker = cls()
        tracker.train_losses = metrics_dict.get("train_losses", [])
        tracker.val_losses = metrics_dict.get("val_losses", [])
        tracker.train_accuracies = metrics_dict.get("train_accuracies", [])
        tracker.val_accuracies = metrics_dict.get("val_accuracies", [])
        tracker.train_f1_scores = metrics_dict.get("train_f1_scores", [])
        tracker.val_f1_scores = metrics_dict.get("val_f1_scores", [])
        tracker.train_aucs = metrics_dict.get("train_aucs", [])
        tracker.val_aucs = metrics_dict.get("val_aucs", [])
        tracker.learning_rates = metrics_dict.get("learning_rates", [])
        tracker.epoch_times = metrics_dict.get("epoch_times", [])
        tracker.epoch_metrics = metrics_dict.get("epoch_metrics", [])

        return tracker

    def summary(self) -> str:
        """Return a summary of the metrics"""
        if not self.epoch_metrics:
            return "No metrics recorded yet."

        best_val = self.get_best_epoch("val_accuracy")

        lines = [
            "=" * 60,
            "Training Summary",
            "=" * 60,
            f"Total epochs: {len(self.epoch_metrics)}",
            f"Best validation accuracy: {best_val.get('val_accuracy', 0):.4f} (epoch {best_val.get('epoch', 0)})",
            f"Best validation F1: {best_val.get('val_f1_score', 0):.4f}",
            f"Best validation AUC: {best_val.get('val_auc', 0):.4f}",
            f"Final train loss: {self.train_losses[-1]:.4f}",
            f"Final val loss: {self.val_losses[-1]:.4f}",
            f"Total training time: {sum(self.epoch_times):.2f}s",
            "=" * 60,
        ]

        return "\n".join(lines)


def test_config():
    """Test configuration"""
    config = TrainingConfig(
        experiment_name="test_exp",
        hidden_dims=[512, 256, 128],
        num_epochs=30,
        learning_rate=0.001,
    )

    print(config)

    # Test saving and loading
    test_path = "test_config.json"
    config.save(test_path)

    loaded_config = TrainingConfig.load(test_path)
    print("\nLoaded config:")
    print(loaded_config)

    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)


def test_metrics():
    """Test metrics tracker"""
    tracker = MetricsTracker()

    # Simulate training
    for epoch in range(5):
        tracker.update_train(
            loss=1.0 - epoch * 0.1,
            accuracy=0.5 + epoch * 0.05,
            f1_score=0.5 + epoch * 0.05,
            auc=0.6 + epoch * 0.05,
        )
        tracker.update_val(
            loss=1.2 - epoch * 0.1,
            accuracy=0.48 + epoch * 0.05,
            f1_score=0.48 + epoch * 0.05,
            auc=0.58 + epoch * 0.05,
        )
        tracker.update_lr(0.001 * (0.9**epoch))
        tracker.update_epoch_time(100.0)

        tracker.add_epoch_metrics(
            epoch,
            {
                "train_loss": tracker.train_losses[-1],
                "val_loss": tracker.val_losses[-1],
                "val_accuracy": tracker.val_accuracies[-1],
                "val_f1_score": tracker.val_f1_scores[-1],
                "val_auc": tracker.val_aucs[-1],
            },
        )

    print(tracker.summary())

    # Test saving
    test_path = "test_metrics.json"
    tracker.save(test_path)

    loaded_tracker = MetricsTracker.load(test_path)
    print("\nLoaded metrics summary:")
    print(loaded_tracker.summary())

    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)


if __name__ == "__main__":
    print("Testing Configuration:")
    test_config()
    print("\n" + "=" * 60 + "\n")
    print("Testing Metrics Tracker:")
    test_metrics()
