"""
Compere reslts of experiments
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ExperimentComparator:
    """
    Compare multiple training experiments.
    """

    def __init__(self, experiments_dir="experiments/lr_sweep_20251015_120120"):
        self.experiments_dir = experiments_dir
        self.experiments = []
        self._load_experiments()

    def _load_experiments(self):
        """Load all experiments from the experiments directory"""
        if not os.path.exists(self.experiments_dir):
            print(f"Experiments directory '{self.experiments_dir}' not found!")
            return

        # Find all experiment directories
        for exp_name in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, exp_name)

            if not os.path.isdir(exp_path):
                continue

            metrics_path = os.path.join(exp_path, "metrics.json")
            config_path = os.path.join(exp_path, "config.json")

            if os.path.exists(metrics_path) and os.path.exists(config_path):
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    self.experiments.append(
                        {
                            "name": exp_name,
                            "path": exp_path,
                            "metrics": metrics,
                            "config": config,
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not load experiment '{exp_name}': {e}")

        print(f"Loaded {len(self.experiments)} experiments")

    def get_best_metrics(self, exp):
        """Extract best metrics from an experiment"""
        metrics = exp["metrics"]

        if not metrics.get("epoch_metrics"):
            return None

        epoch_metrics = metrics["epoch_metrics"]
        best_epoch = max(epoch_metrics, key=lambda x: x.get("val_accuracy", 0))

        return {
            "name": exp["name"],
            "best_val_accuracy": best_epoch.get("val_accuracy", 0),
            "best_val_f1_score": best_epoch.get("val_f1_score", 0),
            "best_val_auc": best_epoch.get("val_auc", 0),
            "best_val_loss": best_epoch.get("val_loss", float("inf")),
            "best_epoch": best_epoch.get("epoch", 0),
            "total_epochs": len(epoch_metrics),
            "final_val_accuracy": metrics["val_accuracies"][-1]
            if metrics["val_accuracies"]
            else 0,
            "training_time": sum(metrics["epoch_times"])
            if metrics["epoch_times"]
            else 0,
        }

    def print_summary(self):
        """Print a summary of all experiments"""
        if not self.experiments:
            print("No experiments found!")
            return

        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON")
        print("=" * 100)

        # Collect metrics
        results = []
        for exp in self.experiments:
            best = self.get_best_metrics(exp)
            if best:
                results.append(best)

        if not results:
            print("No valid results found!")
            return

        # Sort by validation accuracy
        results = sorted(results, key=lambda x: x["best_val_accuracy"], reverse=True)

        # Print table header
        print(
            f"\n{'Rank':<6}{'Experiment':<30}{'Val Acc':<10}{'Val F1':<10}{'Val AUC':<10}{'Epoch':<8}{'Time (s)':<10}"
        )
        print("-" * 100)

        # Print each experiment
        for i, result in enumerate(results, 1):
            print(
                f"{i:<6}"
                f"{result['name'][:28]:<30}"
                f"{result['best_val_accuracy']:.4f}    "
                f"{result['best_val_f1_score']:.4f}    "
                f"{result['best_val_auc']:.4f}    "
                f"{result['best_epoch']:<8}"
                f"{result['training_time']:.1f}"
            )

        print("=" * 100)

        # Print detailed info for top 3
        print("\n" + "=" * 100)
        print("TOP 3 CONFIGURATIONS")
        print("=" * 100)

        for i, result in enumerate(results[:3], 1):
            # Find the experiment
            exp = next(e for e in self.experiments if e["name"] == result["name"])
            config = exp["config"]

            print(f"\n{i}. {result['name']}")
            print(f"   Validation Metrics:")
            print(f"     - Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"     - F1 Score: {result['best_val_f1_score']:.4f}")
            print(f"     - AUC:      {result['best_val_auc']:.4f}")
            print(f"     - Loss:     {result['best_val_loss']:.4f}")
            print(f"   Best Epoch: {result['best_epoch']}/{result['total_epochs']}")
            print(f"   Training Time: {result['training_time']:.1f}s")
            print(f"   Configuration:")
            print(f"     - Architecture: {config.get('hidden_dims', 'N/A')}")
            print(f"     - Learning Rate: {config.get('learning_rate', 'N/A')}")
            print(f"     - Dropout: {config.get('dropout_rate', 'N/A')}")
            print(f"     - Batch Size: {config.get('batch_size', 'N/A')}")
            print(f"     - Optimizer: {config.get('optimizer', 'N/A')}")

        print("\n" + "=" * 100)

    def plot_comparison(self, metric="val_accuracy", top_n=5, save_path=None):
        """
        Plot comparison of experiments for a specific metric.

        Args:
            metric: Metric to plot ('val_accuracy', 'val_f1_score', 'val_loss', etc.)
            top_n: Number of top experiments to plot
            save_path: Path to save plot (optional)
        """
        if not self.experiments:
            print("No experiments to plot!")
            return

        # Collect best experiments
        results = []
        for exp in self.experiments:
            best = self.get_best_metrics(exp)
            if best:
                results.append((exp, best))

        # Sort by validation accuracy
        results = sorted(results, key=lambda x: x[1]["best_val_accuracy"], reverse=True)
        results = results[:top_n]

        # Create plot
        plt.figure(figsize=(14, 8))

        for exp, best in results:
            metrics = exp["metrics"]

            if metric in metrics:
                epochs = range(1, len(metrics[metric]) + 1)
                label = f"{exp['name'][:25]} (best: {best['best_val_accuracy']:.3f})"
                plt.plot(epochs, metrics[metric], linewidth=2, label=label, marker="o")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title(
            f"Comparison of Top {top_n} Experiments - {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_multi_metric_comparison(self, top_n=5, save_path=None):
        """
        Plot multiple metrics in a grid for easy comparison.

        Args:
            top_n: Number of top experiments to plot
            save_path: Path to save plot (optional)
        """
        if not self.experiments:
            print("No experiments to plot!")
            return

        # Collect best experiments
        results = []
        for exp in self.experiments:
            best = self.get_best_metrics(exp)
            if best:
                results.append((exp, best))

        # Sort by validation accuracy
        results = sorted(results, key=lambda x: x[1]["best_val_accuracy"], reverse=True)
        results = results[:top_n]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics_to_plot = [
            ("val_accuracies", "Validation Accuracy"),
            ("val_losses", "Validation Loss"),
            ("val_f1_scores", "Validation F1 Score"),
            ("val_aucs", "Validation AUC"),
        ]

        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]

            for exp, best in results:
                metrics = exp["metrics"]

                if metric_key in metrics and metrics[metric_key]:
                    epochs = range(1, len(metrics[metric_key]) + 1)
                    label = f"{exp['name'][:20]}"
                    ax.plot(epochs, metrics[metric_key], linewidth=2, label=label)

            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Comparison of balancing techniques",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_bar_comparison(self, save_path=None):
        """
        Plot bar chart comparing final metrics of all experiments.

        Args:
            save_path: Path to save plot (optional)
        """
        if not self.experiments:
            print("No experiments to plot!")
            return

        # Collect metrics
        results = []
        for exp in self.experiments:
            best = self.get_best_metrics(exp)
            if best:
                results.append(best)

        # Sort by validation accuracy
        results = sorted(results, key=lambda x: x["best_val_accuracy"], reverse=True)

        # Limit to top 10 for readability
        results = results[:10]

        # Create bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        metrics = [
            ("best_val_accuracy", "Validation Accuracy", axes[0, 0]),
            ("best_val_f1_score", "Validation F1 Score", axes[0, 1]),
            ("best_val_auc", "Validation AUC", axes[1, 0]),
            ("training_time", "Training Time (s)", axes[1, 1]),
        ]

        for metric_key, metric_name, ax in metrics:
            names = [r["name"][:15] for r in results]
            values = [r[metric_key] for r in results]

            bars = ax.barh(names, values, color="steelblue")

            # Color the best bar
            bars[0].set_color("green")

            ax.set_xlabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                if metric_key == "training_time":
                    label = f"{value:.1f}"
                else:
                    label = f"{value:.4f}"
                ax.text(
                    value,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    ha="left",
                    fontsize=9,
                )

        fig.suptitle("Experiment Comparison - Top 10", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_comparison_table(self, output_path="comparison_table.csv"):
        """Export comparison table to CSV"""
        import csv

        if not self.experiments:
            print("No experiments to export!")
            return

        results = []
        for exp in self.experiments:
            best = self.get_best_metrics(exp)
            if best:
                # Add config info
                config = exp["config"]
                best["hidden_dims"] = str(config.get("hidden_dims", "N/A"))
                best["learning_rate"] = config.get("learning_rate", "N/A")
                best["dropout_rate"] = config.get("dropout_rate", "N/A")
                best["batch_size"] = config.get("batch_size", "N/A")
                best["optimizer"] = config.get("optimizer", "N/A")
                results.append(best)

        # Sort by validation accuracy
        results = sorted(results, key=lambda x: x["best_val_accuracy"], reverse=True)

        # Write to CSV
        fieldnames = [
            "name",
            "best_val_accuracy",
            "best_val_f1_score",
            "best_val_auc",
            "best_val_loss",
            "best_epoch",
            "total_epochs",
            "training_time",
            "hidden_dims",
            "learning_rate",
            "dropout_rate",
            "batch_size",
            "optimizer",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Comparison table exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple training experiments"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Directory containing experiments",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_accuracy",
        help="Metric to plot (default: val_accuracy)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top experiments to plot (default: 5)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export comparison table to CSV file",
    )

    args = parser.parse_args()

    # Create comparator
    comparator = ExperimentComparator(args.experiments_dir)

    # Print summary
    comparator.print_summary()

    # Generate plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        comparator.plot_multi_metric_comparison(
            top_n=args.top_n, save_path="experiment_comparison.png"
        )
        comparator.plot_bar_comparison(save_path="experiment_bars.png")

    # Export to CSV if requested
    if args.export:
        comparator.export_comparison_table(args.export)


if __name__ == "__main__":
    main()
