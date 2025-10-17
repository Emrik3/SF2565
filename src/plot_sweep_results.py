"""
Plot parameter sweep results from saved JSON files.
Use this to visualize results without re-running experiments.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class SweepResultsPlotter:
    """
    Visualize parameter sweep results from saved data.
    """

    def __init__(self, sweep_dir):
        """
        Initialize plotter with sweep directory.

        Args:
            sweep_dir: Directory containing sweep_results.json
        """
        self.sweep_dir = sweep_dir
        self.results_path = os.path.join(sweep_dir, "sweep_results.json")
        self.results = []
        self.sweep_params = set()
        self._load_results()

    def _load_results(self):
        """Load results from JSON file"""
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        with open(self.results_path, "r") as f:
            self.results = json.load(f)

        # Filter successful experiments
        self.results = [r for r in self.results if "error" not in r]

        if not self.results:
            raise ValueError("No successful experiments found in results!")

        # Identify swept parameters (parameters that vary across experiments)
        if len(self.results) > 1:
            first_config = self.results[0]["config"]
            for key, value in first_config.items():
                # Skip experiment_name as it's not a real hyperparameter
                if key == "experiment_name":
                    continue
                if any(r["config"].get(key) != value for r in self.results[1:]):
                    self.sweep_params.add(key)

        print(f"Loaded {len(self.results)} successful experiments")
        print(f"Swept parameters: {self.sweep_params}")

    def print_summary(self):
        """Print summary of sweep results"""
        print("\n" + "=" * 100)
        print("PARAMETER SWEEP RESULTS SUMMARY")
        print("=" * 100)

        # Sort by validation accuracy
        sorted_results = sorted(
            self.results, key=lambda x: x["best_val_accuracy"], reverse=True
        )

        print(
            f"\n{'Rank':<6}{'Experiment':<35}{'Val Acc':<10}{'Val F1':<10}{'Val AUC':<10}{'Epoch':<8}"
        )
        print("-" * 100)

        for i, result in enumerate(sorted_results, 1):
            exp_name = result["experiment_name"][:33]
            print(
                f"{i:<6}"
                f"{exp_name:<35}"
                f"{result['best_val_accuracy']:.4f}    "
                f"{result['best_val_f1_score']:.4f}    "
                f"{result['best_val_auc']:.4f}    "
                f"{result['best_epoch']:<8}"
            )

        print("=" * 100)

        # Print top 3 configurations
        print("\n" + "=" * 100)
        print("TOP 3 CONFIGURATIONS")
        print("=" * 100)

        for i, result in enumerate(sorted_results[:3], 1):
            print(f"\n{i}. {result['experiment_name']}")
            print(f"   Validation Metrics:")
            print(f"     - Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"     - F1 Score: {result['best_val_f1_score']:.4f}")
            print(f"     - AUC:      {result['best_val_auc']:.4f}")
            print(f"     - Loss:     {result['best_val_loss']:.4f}")
            print(f"   Best Epoch: {result['best_epoch']}/{result['total_epochs']}")

            if self.sweep_params:
                print("   Swept Parameters:")
                for param in sorted(self.sweep_params):
                    value = result["config"].get(param)
                    print(f"     - {param}: {value}")

        print("\n" + "=" * 100)

    def plot_parameter_impact(self, metric="best_val_accuracy", save_path=None):
        """
        Plot how each parameter affects the chosen metric.

        Args:
            metric: Metric to analyze (default: 'best_val_accuracy')
            save_path: Path to save plot (optional)
        """
        if not self.sweep_params:
            print("No swept parameters found - nothing to plot!")
            return

        n_params = len(self.sweep_params)
        fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5), squeeze=False)
        axes = axes.flatten()

        for idx, param in enumerate(sorted(self.sweep_params)):
            ax = axes[idx]

            # Group results by parameter value
            param_values = {}
            for result in self.results:
                value = str(result["config"].get(param))
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(result[metric])

            # Calculate statistics
            labels = []
            means = []
            stds = []
            for value in sorted(param_values.keys()):
                labels.append(value)
                values = param_values[value]
                means.append(np.mean(values))
                stds.append(np.std(values) if len(values) > 1 else 0)

            # Create bar plot
            x_pos = np.arange(len(labels))
            bars = ax.bar(
                x_pos, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue"
            )

            # Highlight best
            best_idx = np.argmax(means)
            bars[best_idx].set_color("green")

            ax.set_xlabel(param, fontsize=11, fontweight="bold")
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.set_title(f"Impact of {param}", fontsize=12, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for i, (bar, mean) in enumerate(zip(bars, means)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{mean:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_comparison_bars(self, save_path=None):
        """
        Plot bar charts comparing all experiments.

        Args:
            save_path: Path to save plot (optional)
        """
        # Sort by validation accuracy
        sorted_results = sorted(
            self.results, key=lambda x: x["best_val_accuracy"], reverse=True
        )

        # Limit to top 15 for readability
        sorted_results = sorted_results[:15]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        metrics = [
            ("best_val_accuracy", "Validation Accuracy", axes[0, 0]),
            ("best_val_f1_score", "Validation F1 Score", axes[0, 1]),
            ("best_val_auc", "Validation AUC", axes[1, 0]),
            ("best_val_loss", "Validation Loss", axes[1, 1]),
        ]

        # Create labels based on swept parameters
        labels = []
        for r in sorted_results:
            if self.sweep_params:
                # Show swept parameter values
                param_parts = []
                for param in sorted(self.sweep_params):
                    value = r["config"].get(param)
                    # Format the value nicely
                    if isinstance(value, float):
                        param_parts.append(f"{value:.4g}")
                    elif isinstance(value, list):
                        param_parts.append(f"{value}")
                    else:
                        param_parts.append(f"{value}")
                labels.append(", ".join(param_parts))
            else:
                # Fallback to experiment name if no swept params
                labels.append(r["experiment_name"][:20])

        labels.sort()
        labels.insert(0, labels.pop())

        # labels.insert(-2, labels.pop(4))
        # labels.insert(-1, labels.pop(5))

        for metric_key, metric_name, ax in metrics:
            values = [r[metric_key] for r in sorted_results]

            bars = ax.barh(labels, values, color="steelblue")

            # Color the best bar (lowest for loss, highest for others)
            if metric_key == "best_val_loss":
                best_idx = np.argmin(values)
            else:
                best_idx = 0  # Already sorted by accuracy
            bars[best_idx].set_color("green")

            ax.set_xlabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")
            ax.invert_yaxis()  # Highest at top

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                label = f"{value:.4f}"
                ax.text(
                    value,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

        sweep_name = os.path.basename(self.sweep_dir)
        fig.suptitle(
            "Parameter Sweep Results - Learning Rate",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_scatter_matrix(self, save_path=None):
        """
        Plot scatter matrix showing relationships between swept parameters and metrics.

        Args:
            save_path: Path to save plot (optional)
        """
        if not self.sweep_params:
            print("No swept parameters found - nothing to plot!")
            return

        # Only plot numeric parameters
        numeric_params = []
        for param in self.sweep_params:
            # Check if parameter is numeric
            try:
                float(str(self.results[0]["config"].get(param)))
                numeric_params.append(param)
            except (ValueError, TypeError):
                pass

        if not numeric_params:
            print("No numeric swept parameters found for scatter plot!")
            return

        metrics = ["best_val_accuracy", "best_val_f1_score", "best_val_auc"]
        n_params = len(numeric_params)

        fig, axes = plt.subplots(
            len(metrics), n_params, figsize=(5 * n_params, 4 * len(metrics))
        )
        if n_params == 1:
            axes = axes.reshape(-1, 1)

        for i, metric in enumerate(metrics):
            for j, param in enumerate(numeric_params):
                ax = axes[i, j] if len(metrics) > 1 else axes[j]

                # Extract data
                x_values = []
                y_values = []
                for result in self.results:
                    try:
                        x_val = float(str(result["config"].get(param)))
                        y_val = result[metric]
                        x_values.append(x_val)
                        y_values.append(y_val)
                    except (ValueError, TypeError):
                        continue

                # Plot
                ax.scatter(x_values, y_values, alpha=0.6, s=100, color="steelblue")

                # Highlight best
                best_idx = np.argmax(y_values)
                ax.scatter(
                    [x_values[best_idx]],
                    [y_values[best_idx]],
                    color="green",
                    s=150,
                    marker="*",
                    zorder=5,
                    label="Best",
                )

                ax.set_xlabel(param, fontsize=10)
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend()

        sweep_name = os.path.basename(self.sweep_dir)
        fig.suptitle(
            f"Parameter vs Metric Relationships - {sweep_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_to_csv(self, output_path=None):
        """Export results to CSV"""
        import csv

        if output_path is None:
            output_path = os.path.join(self.sweep_dir, "sweep_results.csv")

        # Determine all unique keys
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
            all_keys.update(result["config"].keys())
        all_keys.discard("config")

        # Flatten results
        flattened = []
        for result in self.results:
            flat = {k: v for k, v in result.items() if k != "config"}
            for k, v in result["config"].items():
                flat[f"config_{k}"] = v
            flattened.append(flat)

        # Sort by validation accuracy
        flattened = sorted(
            flattened, key=lambda x: x.get("best_val_accuracy", 0), reverse=True
        )

        # Write CSV
        if flattened:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(flattened[0].keys()))
                writer.writeheader()
                writer.writerows(flattened)

            print(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot parameter sweep results from saved data"
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Directory containing sweep_results.json",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="best_val_accuracy",
        help="Metric to analyze (default: best_val_accuracy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as sweep_dir)",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to CSV",
    )
    parser.add_argument(
        "--all_plots",
        action="store_true",
        help="Generate all plot types",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.sweep_dir

    try:
        # Load and analyze results
        plotter = SweepResultsPlotter(args.sweep_dir)

        # Print summary
        plotter.print_summary()

        # Generate plots
        if args.all_plots:
            print("\nGenerating plots...")

            plotter.plot_parameter_impact(
                metric=args.metric,
                save_path=os.path.join(output_dir, "parameter_impact.png"),
            )

            plotter.plot_comparison_bars(
                save_path=os.path.join(output_dir, "comparison_bars.png")
            )

            plotter.plot_scatter_matrix(
                save_path=os.path.join(output_dir, "scatter_matrix.png")
            )

            print(f"\nAll plots saved to {output_dir}")

        # Export CSV if requested
        if args.export_csv:
            plotter.export_to_csv()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
