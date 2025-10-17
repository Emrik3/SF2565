"""
Trainer class for feature-based neural network.
Handles training loop, validation, checkpointing, and visualization.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import TrainingConfig, MetricsTracker
from model import FeatureNN
from feature_dataset import FeatureDataset


class FeatureNNTrainer:
    """
    Trainer for feature-based neural network classification.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: TrainingConfig object containing all hyperparameters
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create experiment directory
        self.exp_dir = config.get_experiment_dir()
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)

        # Save configuration
        config.save(os.path.join(self.exp_dir, "config.json"))

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Build model, dataloaders, optimizer, etc.
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_loss_function()

        # Early stopping
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

        if config.verbose:
            print(config)
            self._print_model_info()

    def _build_dataloaders(self):
        """Build train and validation dataloaders"""
        print(f"\nLoading dataset from {self.config.data_path}...")

        # Load full dataset
        full_dataset = FeatureDataset(self.config.data_path)

        # Split into train and validation
        total_size = len(full_dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.random_seed),
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        # Setup weighted sampler if requested
        train_sampler = None
        shuffle = True

        if self.config.use_weighted_sampler:
            print(f"\n{'=' * 60}")
            print("Setting up WeightedRandomSampler for class balance")
            print(f"{'=' * 60}")

            # Get labels from training set
            train_indices = self.train_dataset.indices
            all_labels = full_dataset.labels[train_indices].flatten()

            # Calculate class distribution
            num_class_0 = (all_labels == 0).sum().item()
            num_class_1 = (all_labels == 1).sum().item()

            print(f"Training set class distribution:")
            print(
                f"  Class 0 (Sick):    {num_class_0:,} ({100 * num_class_0 / len(all_labels):.1f}%)"
            )
            print(
                f"  Class 1 (Healthy): {num_class_1:,} ({100 * num_class_1 / len(all_labels):.1f}%)"
            )
            print(
                f"  Imbalance ratio:   {max(num_class_0, num_class_1) / min(num_class_0, num_class_1):.2f}:1"
            )

            # Calculate sample weights: weight = 1 / class_frequency
            class_counts = torch.tensor([num_class_0, num_class_1], dtype=torch.float)
            class_weights = 1.0 / class_counts

            # Assign weight to each sample based on its class
            sample_weights = class_weights[all_labels.long()]
            print(sample_weights)

            # Create weighted sampler
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,  # Allow oversampling
            )

            shuffle = False  # Don't shuffle when using sampler

            print(f"\nWeighted sampling enabled:")
            print(f"  Class 0 weight: {class_weights[0]:.6f}")
            print(f"  Class 1 weight: {class_weights[1]:.6f}")
            print(
                f"  Minority class will be oversampled {class_weights[1] / class_weights[0]:.2f}x"
            )
            print(f"{'=' * 60}\n")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def _build_model(self):
        """Build the neural network model"""
        self.model = FeatureNN(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            batch_norm=self.config.batch_norm,
        )
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        """Build optimizer"""
        if self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if not self.config.use_scheduler:
            self.scheduler = None
            return

        if self.config.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")

    def _build_loss_function(self):
        """Build loss function"""
        if self.config.pos_weight is not None:
            pos_weight = torch.tensor([self.config.pos_weight]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _print_model_info(self):
        """Print model information"""
        print("\n" + "=" * 60)
        print("Model Architecture")
        print("=" * 60)
        print(self.model)
        print(f"\nTotal parameters: {self.model.get_num_parameters():,}")
        print("=" * 60)

    def calculate_metrics(self, outputs, targets):
        """
        Calculate accuracy, F1 score, and AUC.

        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Dictionary containing metrics
        """
        # Convert to numpy
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        targets_np = targets.cpu().numpy().astype(int)

        # Flatten if needed
        preds = preds.flatten()
        targets_np = targets_np.flatten()
        probs = probs.flatten()

        # Calculate metrics
        accuracy = accuracy_score(targets_np, preds)
        f1 = f1_score(targets_np, preds, zero_division=0)

        # AUC (handle case where only one class is present)
        try:
            auc = roc_auc_score(targets_np, probs)
        except ValueError:
            auc = 0.0

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "predictions": preds,
            "probabilities": probs,
            "targets": targets_np,
        }

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()

        running_loss = 0.0
        all_outputs = []
        all_targets = []

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]"
        )

        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

            # Update progress bar
            if batch_idx % self.config.print_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.calculate_metrics(all_outputs, all_targets)
        metrics["loss"] = running_loss / len(self.train_loader)

        return metrics

    def validate_epoch(self, epoch):
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        all_outputs = []
        all_targets = []

        pbar = tqdm(
            self.val_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Val]  "
        )

        with torch.no_grad():
            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                # Track metrics
                running_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.calculate_metrics(all_outputs, all_targets)
        metrics["loss"] = running_loss / len(self.val_loader)

        return metrics

    def train(self):
        """
        Main training loop.
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate_epoch(epoch)

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Track metrics
            epoch_time = time.time() - epoch_start_time
            self.metrics.update_train(
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["f1_score"],
                train_metrics["auc"],
            )
            self.metrics.update_val(
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1_score"],
                val_metrics["auc"],
            )
            self.metrics.update_lr(current_lr)
            self.metrics.update_epoch_time(epoch_time)

            self.metrics.add_epoch_metrics(
                epoch,
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_score": train_metrics["f1_score"],
                    "train_auc": train_metrics["auc"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_score": val_metrics["f1_score"],
                    "val_auc": val_metrics["auc"],
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                },
            )

            # Print epoch summary
            self._print_epoch_summary(
                epoch, train_metrics, val_metrics, current_lr, epoch_time
            )

            # Save checkpoint
            is_best = val_metrics["accuracy"] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.config.save_best_only and is_best:
                self._save_checkpoint(epoch, is_best=True)
            elif not self.config.save_best_only:
                self._save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if (
                self.config.use_early_stopping
                and self.epochs_without_improvement
                >= self.config.early_stopping_patience
            ):
                print(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.config.early_stopping_patience} epochs)"
                )
                break

        # Training complete
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        self._print_final_summary()

        # Save final metrics
        self.metrics.save(os.path.join(self.exp_dir, "metrics.json"))

        # Generate plots
        if self.config.save_plots:
            self._generate_all_plots()

        return self.metrics

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """Print summary for current epoch"""
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{self.config.num_epochs} Summary")
        print(f"{'=' * 60}")
        print(f"Time: {epoch_time:.2f}s | LR: {lr:.6f}")
        print(
            f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1_score']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f}"
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_score']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )
        print(f"{'=' * 60}")

    def _print_final_summary(self):
        """Print final training summary"""
        print(self.metrics.summary())

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": self.metrics.get_current_metrics(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.exp_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.exp_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  → Best model saved (Val Acc: {self.best_val_accuracy:.4f})")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def _generate_all_plots(self):
        """Generate all visualization plots"""
        print("\nGenerating plots...")
        plots_dir = os.path.join(self.exp_dir, "plots")

        self._plot_loss_curves(plots_dir)
        self._plot_accuracy_curves(plots_dir)
        self._plot_f1_auc_curves(plots_dir)
        self._plot_learning_rate(plots_dir)
        self._plot_training_overview(plots_dir)

        print(f"Plots saved to {plots_dir}")

    def _plot_loss_curves(self, plots_dir):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics.train_losses) + 1)

        plt.plot(
            epochs, self.metrics.train_losses, "b-", label="Train Loss", linewidth=2
        )
        plt.plot(epochs, self.metrics.val_losses, "r-", label="Val Loss", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=300)
        plt.close()

    def _plot_accuracy_curves(self, plots_dir):
        """Plot training and validation accuracy curves"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics.train_accuracies) + 1)

        plt.plot(
            epochs,
            self.metrics.train_accuracies,
            "b-",
            label="Train Accuracy",
            linewidth=2,
        )
        plt.plot(
            epochs, self.metrics.val_accuracies, "r-", label="Val Accuracy", linewidth=2
        )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, "accuracy_curves.png"), dpi=300)
        plt.close()

    def _plot_f1_auc_curves(self, plots_dir):
        """Plot F1 score and AUC curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(self.metrics.train_f1_scores) + 1)

        # F1 Score
        ax1.plot(
            epochs, self.metrics.train_f1_scores, "b-", label="Train F1", linewidth=2
        )
        ax1.plot(epochs, self.metrics.val_f1_scores, "r-", label="Val F1", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("F1 Score", fontsize=12)
        ax1.set_title("F1 Score", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # AUC
        ax2.plot(epochs, self.metrics.train_aucs, "b-", label="Train AUC", linewidth=2)
        ax2.plot(epochs, self.metrics.val_aucs, "r-", label="Val AUC", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("AUC", fontsize=12)
        ax2.set_title("ROC AUC", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "f1_auc_curves.png"), dpi=300)
        plt.close()

    def _plot_learning_rate(self, plots_dir):
        """Plot learning rate schedule"""
        if not self.metrics.learning_rates:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics.learning_rates) + 1)

        plt.plot(epochs, self.metrics.learning_rates, "g-", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, "learning_rate.png"), dpi=300)
        plt.close()

    def _plot_training_overview(self, plots_dir):
        """Plot comprehensive training overview"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        epochs = range(1, len(self.metrics.train_losses) + 1)

        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.metrics.train_losses, "b-", label="Train", linewidth=2)
        ax1.plot(epochs, self.metrics.val_losses, "r-", label="Val", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            epochs, self.metrics.train_accuracies, "b-", label="Train", linewidth=2
        )
        ax2.plot(epochs, self.metrics.val_accuracies, "r-", label="Val", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # F1 Score
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.metrics.train_f1_scores, "b-", label="Train", linewidth=2)
        ax3.plot(epochs, self.metrics.val_f1_scores, "r-", label="Val", linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1 Score")
        ax3.set_title("F1 Score", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # AUC
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.metrics.train_aucs, "b-", label="Train", linewidth=2)
        ax4.plot(epochs, self.metrics.val_aucs, "r-", label="Val", linewidth=2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("AUC")
        ax4.set_title("ROC AUC", fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Learning Rate
        ax5 = fig.add_subplot(gs[2, 0])
        if self.metrics.learning_rates:
            ax5.plot(epochs, self.metrics.learning_rates, "g-", linewidth=2)
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("Learning Rate")
            ax5.set_title("Learning Rate", fontweight="bold")
            ax5.set_yscale("log")
            ax5.grid(True, alpha=0.3)

        # Epoch Time
        ax6 = fig.add_subplot(gs[2, 1])
        if self.metrics.epoch_times:
            ax6.plot(epochs, self.metrics.epoch_times, "m-", linewidth=2)
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Time (s)")
            ax6.set_title("Epoch Time", fontweight="bold")
            ax6.grid(True, alpha=0.3)

        fig.suptitle(
            f"Training Overview - {self.config.experiment_name}",
            fontsize=16,
            fontweight="bold",
        )

        plt.savefig(os.path.join(plots_dir, "training_overview.png"), dpi=300)
        plt.close()


def test_trainer():
    """Test the trainer with a small example"""
    # Create a simple config
    config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        experiment_name="test_run",
        hidden_dims=[512, 256],
        num_epochs=3,
        batch_size=32,
        verbose=True,
    )

    # Create trainer
    trainer = FeatureNNTrainer(config)

    # Train
    metrics = trainer.train()

    print("\nTraining test complete!")
    print(metrics.summary())


if __name__ == "__main__":
    test_trainer()
