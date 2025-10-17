"""
Evaluation and inference script for trained feature-based neural network.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import seaborn as sns

from config import TrainingConfig
from model import FeatureNN
from feature_dataset import FeatureDataset


class ModelEvaluator:
    """
    Evaluate trained model and generate comprehensive analysis.
    """

    def __init__(self, checkpoint_path, data_path=None):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            data_path: Path to test data (if different from training data)
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        self.config = TrainingConfig(**self.checkpoint["config"])

        # Override data path if provided
        if data_path is not None:
            self.config.data_path = data_path

        # Build model
        self._build_model()

        # Load test data
        self._load_data()

        print(f"Model loaded successfully on {self.device}")

    def _build_model(self):
        """Build and load model"""
        self.model = FeatureNN(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            batch_norm=self.config.batch_norm,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_data(self):
        """Load test/validation data"""
        dataset = FeatureDataset(self.config.data_path)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        print(f"Loaded {len(dataset)} samples for evaluation")

    def evaluate(self):
        """
        Run full evaluation on the dataset.

        Returns:
            Dictionary containing all evaluation metrics and predictions
        """
        print("\nRunning evaluation...")

        all_outputs = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for features, targets in self.dataloader:
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)
                probs = torch.sigmoid(outputs)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        # Flatten
        all_probs = all_probs.flatten()
        all_targets = all_targets.flatten().astype(int)

        # Get predictions (threshold at 0.5)
        all_preds = (all_probs > 0.5).astype(int)

        # Calculate metrics
        results = self._calculate_metrics(all_targets, all_preds, all_probs)

        # Add raw data
        results["targets"] = all_targets
        results["predictions"] = all_preds
        results["probabilities"] = all_probs

        return results

    def _calculate_metrics(self, targets, predictions, probabilities):
        """Calculate all evaluation metrics"""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["f1_score"] = f1_score(targets, predictions, zero_division=0)

        # Precision and Recall
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC
        try:
            metrics["auc"] = roc_auc_score(targets, probabilities)
        except ValueError:
            metrics["auc"] = 0.0

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(targets, predictions)

        # Classification report
        metrics["classification_report"] = classification_report(
            targets, predictions, target_names=["Healthy", "Sick"], output_dict=True
        )

        return metrics

    def print_results(self, results):
        """Print evaluation results in a nice format"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  F1 Score:    {results['f1_score']:.4f}")
        print(f"  Precision:   {results['precision']:.4f}")
        print(f"  Recall:      {results['recall']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        print(f"  AUC:         {results['auc']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = results["confusion_matrix"]
        print(f"                 Predicted")
        print(f"                 Neg    Pos")
        print(f"  Actual  Neg   {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"          Pos   {cm[1, 0]:5d}  {cm[1, 1]:5d}")

        print("\n" + "=" * 60)

    def plot_results(self, results, save_dir=None):
        """
        Generate comprehensive visualization plots.

        Args:
            results: Results dictionary from evaluate()
            save_dir: Directory to save plots (if None, uses checkpoint directory)
        """
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(self.checkpoint_path), "evaluation")

        os.makedirs(save_dir, exist_ok=True)

        print(f"\nGenerating evaluation plots...")

        # 1. Confusion Matrix
        self._plot_confusion_matrix(results, save_dir)

        # 2. ROC Curve
        self._plot_roc_curve(results, save_dir)

        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(results, save_dir)

        # 4. Probability Distribution
        self._plot_probability_distribution(results, save_dir)

        # 5. Comprehensive Overview
        self._plot_evaluation_overview(results, save_dir)

        print(f"Plots saved to {save_dir}")

    def _plot_confusion_matrix(self, results, save_dir):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))

        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Healthy", "Sick"],
            yticklabels=["Healthy", "Sick"],
            cbar_kws={"label": "Count"},
        )

        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

    def _plot_roc_curve(self, results, save_dir):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(results["targets"], results["probabilities"])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {results['auc']:.4f}")
        plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")

        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
        plt.close()

    def _plot_precision_recall_curve(self, results, save_dir):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(
            results["targets"], results["probabilities"]
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, "g-", linewidth=2)

        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300)
        plt.close()

    def _plot_probability_distribution(self, results, save_dir):
        """Plot distribution of predicted probabilities"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        targets = results["targets"]
        probs = results["probabilities"]

        # Histogram
        ax1.hist(
            probs[targets == 0],
            bins=50,
            alpha=0.7,
            label="Healthy",
            color="blue",
            edgecolor="black",
        )
        ax1.hist(
            probs[targets == 1],
            bins=50,
            alpha=0.7,
            label="Sick",
            color="red",
            edgecolor="black",
        )
        ax1.axvline(
            x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold"
        )
        ax1.set_xlabel("Predicted Probability", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Probability Distribution", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Box plot
        data_to_plot = [probs[targets == 0], probs[targets == 1]]
        ax2.boxplot(data_to_plot, labels=["Healthy", "Sick"])
        ax2.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="Threshold")
        ax2.set_ylabel("Predicted Probability", fontsize=12)
        ax2.set_title("Probability Box Plot", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "probability_distribution.png"), dpi=300)
        plt.close()

    def _plot_evaluation_overview(self, results, save_dir):
        """Plot comprehensive evaluation overview"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Healthy", "Sick"],
            yticklabels=["Healthy", "Sick"],
            ax=ax1,
            cbar=False,
        )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title("Confusion Matrix", fontweight="bold")

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(results["targets"], results["probabilities"])
        ax2.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {results['auc']:.3f}")
        ax2.plot([0, 1], [0, 1], "r--", linewidth=1)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision, recall, _ = precision_recall_curve(
            results["targets"], results["probabilities"]
        )
        ax3.plot(recall, precision, "g-", linewidth=2)
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.set_title("Precision-Recall Curve", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # 4. Metrics Bar Chart
        ax4 = fig.add_subplot(gs[1, 0])
        metrics = ["Accuracy", "F1", "Precision", "Recall", "Specificity"]
        values = [
            results["accuracy"],
            results["f1_score"],
            results["precision"],
            results["recall"],
            results["specificity"],
        ]
        bars = ax4.bar(
            metrics, values, color=["blue", "green", "orange", "red", "purple"]
        )
        ax4.set_ylabel("Score")
        ax4.set_title("Performance Metrics", fontweight="bold")
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        # 5. Probability Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        targets = results["targets"]
        probs = results["probabilities"]
        ax5.hist(probs[targets == 0], bins=30, alpha=0.7, label="Healthy", color="blue")
        ax5.hist(probs[targets == 1], bins=30, alpha=0.7, label="Sick", color="red")
        ax5.axvline(x=0.5, color="black", linestyle="--", linewidth=2)
        ax5.set_xlabel("Predicted Probability")
        ax5.set_ylabel("Count")
        ax5.set_title("Probability Distribution", fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Class Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        unique, counts = np.unique(results["targets"], return_counts=True)
        ax6.pie(
            counts,
            labels=["Healthy", "Sick"],
            autopct="%1.1f%%",
            colors=["lightblue", "lightcoral"],
            startangle=90,
        )
        ax6.set_title("Class Distribution", fontweight="bold")

        fig.suptitle("Evaluation Overview", fontsize=16, fontweight="bold")

        plt.savefig(os.path.join(save_dir, "evaluation_overview.png"), dpi=300)
        plt.close()

    def predict(self, features):
        """
        Make predictions on new feature data.

        Args:
            features: Tensor of shape (N, feature_dim)

        Returns:
            Dictionary with predictions and probabilities
        """
        self.model.eval()

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        features = features.to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

        return {
            "predictions": preds.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
        }


def evaluate_model(checkpoint_path, data_path=None):
    """
    Convenience function to evaluate a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to test data (optional)
    """
    evaluator = ModelEvaluator(checkpoint_path, data_path)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    evaluator.plot_results(results)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., experiments/my_exp/best_model.pt)",
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Path to test data (optional)"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save plots"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data if args.data else 'Using training data'}")
    print("=" * 60)

    # Run evaluation
    evaluator = ModelEvaluator(args.checkpoint, args.data)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    evaluator.plot_results(results, args.save_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
