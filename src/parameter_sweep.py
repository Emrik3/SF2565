"""
Parameter sweep script for hyperparameter tuning.
Systematically tests different configurations to find the best parameters.
"""

import os
import json
import itertools
from datetime import datetime
import torch
import gc

from config import TrainingConfig
from trainer import FeatureNNTrainer

torch.multiprocessing.set_sharing_strategy("file_system")


class ParameterSweep:
    """
    Manages hyperparameter sweep experiments.
    """

    def __init__(
        self, base_config: TrainingConfig, sweep_params: dict, base_name: str = "sweep"
    ):
        """
        Initialize parameter sweep.

        Args:
            base_config: Base configuration to use
            sweep_params: Dictionary of parameters to sweep over
                         e.g., {'learning_rate': [0.001, 0.0001], 'dropout_rate': [0.3, 0.5]}
            base_name: Base name for sweep experiments
        """
        self.base_config = base_config
        self.sweep_params = sweep_params
        self.base_name = base_name
        self.results = []

        # Create sweep directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = os.path.join(
            base_config.output_dir, f"{base_name}_{timestamp}"
        )
        os.makedirs(self.sweep_dir, exist_ok=True)

        print(f"Parameter sweep directory: {self.sweep_dir}")

    def _generate_configs(self):
        """Generate all configuration combinations"""
        # Get parameter names and values
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())

        # Generate all combinations
        configs = []
        for i, values in enumerate(itertools.product(*param_values)):
            # Create new config based on base config
            config_dict = self.base_config.to_dict()

            # Update with sweep parameters
            for param_name, param_value in zip(param_names, values):
                config_dict[param_name] = param_value

            # Create unique experiment name
            param_str = "_".join([f"{k}{v}" for k, v in zip(param_names, values)])
            config_dict["experiment_name"] = f"{self.base_name}_{i:03d}_{param_str}"
            config_dict["output_dir"] = self.sweep_dir

            configs.append(TrainingConfig(**config_dict))

        return configs

    def run(self):
        """Run parameter sweep"""
        configs = self._generate_configs()
        total_experiments = len(configs)

        print("=" * 80)
        print(f"Starting Parameter Sweep: {total_experiments} experiments")
        print("=" * 80)
        print("\nSweep parameters:")
        for param, values in self.sweep_params.items():
            print(f"  {param}: {values}")
        print("=" * 80)

        for i, config in enumerate(configs, 1):
            print(f"\n{'#' * 80}")
            print(f"Experiment {i}/{total_experiments}: {config.experiment_name}")
            print(f"{'#' * 80}")

            try:
                # Train with this configuration
                trainer = FeatureNNTrainer(config)
                metrics = trainer.train()

                # Get best validation metrics
                best_metrics = metrics.get_best_epoch("val_accuracy")

                # Store results
                result = {
                    "experiment_id": i,
                    "experiment_name": config.experiment_name,
                    "config": config.to_dict(),
                    "best_val_accuracy": best_metrics.get("val_accuracy", 0),
                    "best_val_f1_score": best_metrics.get("val_f1_score", 0),
                    "best_val_auc": best_metrics.get("val_auc", 0),
                    "best_val_loss": best_metrics.get("val_loss", float("inf")),
                    "best_epoch": best_metrics.get("epoch", 0),
                    "total_epochs": len(metrics.epoch_metrics),
                    "final_val_accuracy": metrics.val_accuracies[-1]
                    if metrics.val_accuracies
                    else 0,
                }

                self.results.append(result)

                print(f"\nExperiment {i} completed successfully!")
                print(
                    f"  Best Val Accuracy: {result['best_val_accuracy']:.4f} (epoch {result['best_epoch']})"
                )
                print(f"  Best Val F1: {result['best_val_f1_score']:.4f}")
                print(f"  Best Val AUC: {result['best_val_auc']:.4f}")

                # Clean up memory
                del trainer
                del metrics
                del best_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\nExperiment {i} failed with error: {str(e)}")
                import traceback

                traceback.print_exc()
                result = {
                    "experiment_id": i,
                    "experiment_name": config.experiment_name,
                    "config": config.to_dict(),
                    "error": str(e),
                }
                self.results.append(result)

                # Clean up memory even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Save intermediate results after each experiment
            self._save_results()
            print(
                f"Progress saved to: {os.path.join(self.sweep_dir, 'sweep_results.json')}"
            )

        # Print final summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON file"""
        results_path = os.path.join(self.sweep_dir, "sweep_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def _print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP SUMMARY")
        print("=" * 80)

        # Filter successful experiments
        successful = [r for r in self.results if "error" not in r]

        if not successful:
            print("No successful experiments!")
            return

        # Sort by validation accuracy
        sorted_results = sorted(
            successful, key=lambda x: x["best_val_accuracy"], reverse=True
        )

        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")

        print("\n" + "-" * 80)
        print("TOP 5 CONFIGURATIONS (by validation accuracy)")
        print("-" * 80)

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. {result['experiment_name']}")
            print(f"   Val Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"   Val F1: {result['best_val_f1_score']:.4f}")
            print(f"   Val AUC: {result['best_val_auc']:.4f}")
            print(f"   Best Epoch: {result['best_epoch']}")

            # Print swept parameters
            print("   Parameters:")
            for param in self.sweep_params.keys():
                print(f"     {param}: {result['config'][param]}")

        print("\n" + "=" * 80)
        print(f"Results saved to: {os.path.join(self.sweep_dir, 'sweep_results.json')}")
        print("=" * 80)


def sweep_learning_rate():
    """Sweep over different learning rates"""
    base_config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        hidden_dims=[512, 256],
        num_epochs=30,
        batch_size=64,
        dropout_rate=0.5,
        verbose=False,
        use_weighted_sampler=True,
        num_workers=0,  # Prevent "too many open files" error
    )

    sweep_params = {
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
    }

    sweep = ParameterSweep(base_config, sweep_params, base_name="lr_sweep")
    results = sweep.run()

    return results


def sweep_architecture():
    """Sweep over different network architectures"""
    base_config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        num_epochs=30,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.5,
        verbose=False,
        use_weighted_sampler=True,
        num_workers=0,  # Prevent "too many open files" error
    )

    sweep_params = {
        "hidden_dims": [
            [512],
            [512, 256],
            [1024, 512],
            [512, 256, 128],
            [1024, 512, 256],
            [1024, 512, 256, 128],
        ],
    }

    sweep = ParameterSweep(base_config, sweep_params, base_name="arch_sweep")
    results = sweep.run()

    return results


def sweep_dropout():
    """Sweep over different dropout rates"""
    base_config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        hidden_dims=[512, 256],
        num_epochs=30,
        batch_size=64,
        learning_rate=0.001,
        verbose=False,
        use_weighted_sampler=True,
        num_workers=0,  # Prevent "too many open files" error
    )

    sweep_params = {
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }

    sweep = ParameterSweep(base_config, sweep_params, base_name="dropout_sweep")
    results = sweep.run()

    return results


def sweep_batch_size():
    """Sweep over different batch sizes"""
    base_config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        hidden_dims=[512, 256],
        num_epochs=30,
        learning_rate=0.001,
        dropout_rate=0.5,
        verbose=False,
        use_weighted_sampler=True,
        num_workers=0,  # Prevent "too many open files" error
    )

    sweep_params = {
        "batch_size": [16, 32, 64, 128, 256],
    }

    sweep = ParameterSweep(base_config, sweep_params, base_name="batch_sweep")
    results = sweep.run()

    return results


def sweep_comprehensive():
    """Comprehensive sweep over multiple parameters"""
    base_config = TrainingConfig(
        data_path="data/xray_features_frontal_only.pt",
        num_epochs=25,
        verbose=False,
        use_weighted_sampler=True,
        num_workers=0,  # Prevent "too many open files" error
    )

    sweep_params = {
        "hidden_dims": [[512], [512, 256], [1024, 512, 256]],
        "learning_rate": [0.001, 0.0001],
        "dropout_rate": [0.3, 0.5, 0.7],
        "batch_size": [64, 128],
    }

    sweep = ParameterSweep(base_config, sweep_params, base_name="comprehensive_sweep")
    results = sweep.run()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument(
        "--sweep",
        type=str,
        default="learning_rate",
        choices=[
            "learning_rate",
            "architecture",
            "dropout",
            "batch_size",
            "comprehensive",
        ],
        help="Type of parameter sweep to run",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PARAMETER SWEEP")
    print("=" * 80)
    print(f"Sweep type: {args.sweep}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)

    if args.sweep == "learning_rate":
        sweep_learning_rate()
    elif args.sweep == "architecture":
        sweep_architecture()
    elif args.sweep == "dropout":
        sweep_dropout()
    elif args.sweep == "batch_size":
        sweep_batch_size()
    elif args.sweep == "comprehensive":
        sweep_comprehensive()
    else:
        print(f"Unknown sweep type: {args.sweep}")


if __name__ == "__main__":
    main()
