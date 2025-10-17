"""
Time Complexity Analysis for Neural Network.
Measures and plots the computational time for different configurations.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

from model import FeatureNN


class TimeComplexityAnalyzer:
    """
    Analyzes and plots the time complexity of neural network operations.
    """

    def __init__(self, device: str = None):
        """
        Initialize the analyzer.

        Args:
            device: Device to run experiments on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Time Complexity Analyzer initialized on device: {self.device}")

    def measure_forward_pass_time(
        self, model: nn.Module, input_tensor: torch.Tensor, n_iterations: int = 100
    ) -> float:
        """
        Measure the average forward pass time.

        Args:
            model: Neural network model
            input_tensor: Input tensor
            n_iterations: Number of iterations to average over

        Returns:
            Average forward pass time in milliseconds
        """
        model.eval()
        times = []

        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = model(input_tensor)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Measure
            for _ in range(n_iterations):
                start_time = time.perf_counter()
                _ = model(input_tensor)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        return np.mean(times)

    def measure_backward_pass_time(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        n_iterations: int = 100,
    ) -> float:
        """
        Measure the average backward pass time.

        Args:
            model: Neural network model
            input_tensor: Input tensor
            target: Target tensor
            n_iterations: Number of iterations to average over

        Returns:
            Average backward pass time in milliseconds
        """
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        times = []

        # Warm up
        for _ in range(10):
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Measure
        for _ in range(n_iterations):
            output = model(input_tensor)
            loss = criterion(output, target)

            start_time = time.perf_counter()
            loss.backward()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            model.zero_grad()

        return np.mean(times)

    def analyze_batch_size_scaling(
        self,
        batch_sizes: List[int] = None,
        input_dim: int = 2048,
        hidden_dims: List[int] = None,
        n_iterations: int = 100,
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Analyze how time scales with batch size.

        Args:
            batch_sizes: List of batch sizes to test
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            n_iterations: Number of iterations for averaging

        Returns:
            Tuple of (batch_sizes, forward_times, backward_times)
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        if hidden_dims is None:
            hidden_dims = [512, 256]

        print(f"\nAnalyzing batch size scaling...")
        print(f"Hidden dims: {hidden_dims}")

        forward_times = []
        backward_times = []

        model = FeatureNN(
            input_dim=input_dim, hidden_dims=hidden_dims, num_classes=1
        ).to(self.device)

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")

            input_tensor = torch.randn(batch_size, input_dim).to(self.device)
            target = torch.randint(0, 2, (batch_size, 1)).float().to(self.device)

            forward_time = self.measure_forward_pass_time(
                model, input_tensor, n_iterations
            )
            backward_time = self.measure_backward_pass_time(
                model, input_tensor, target, n_iterations
            )

            forward_times.append(forward_time)
            backward_times.append(backward_time)

            print(f"  Forward: {forward_time:.4f} ms, Backward: {backward_time:.4f} ms")

        return batch_sizes, forward_times, backward_times

    def analyze_depth_scaling(
        self,
        depths: List[int] = None,
        layer_width: int = 512,
        batch_size: int = 64,
        input_dim: int = 2048,
        n_iterations: int = 100,
    ) -> Tuple[List[int], List[float], List[float], List[int]]:
        """
        Analyze how time scales with network depth.

        Args:
            depths: List of network depths to test
            layer_width: Width of each hidden layer
            batch_size: Batch size for testing
            input_dim: Input feature dimension
            n_iterations: Number of iterations for averaging

        Returns:
            Tuple of (depths, forward_times, backward_times, parameter_counts)
        """
        if depths is None:
            depths = [1, 2, 3, 4, 5, 6, 8, 10]

        print(f"\nAnalyzing network depth scaling...")
        print(f"Layer width: {layer_width}, Batch size: {batch_size}")

        forward_times = []
        backward_times = []
        parameter_counts = []

        input_tensor = torch.randn(batch_size, input_dim).to(self.device)
        target = torch.randint(0, 2, (batch_size, 1)).float().to(self.device)

        for depth in depths:
            print(f"Testing depth: {depth} layers")

            hidden_dims = [layer_width] * depth

            model = FeatureNN(
                input_dim=input_dim, hidden_dims=hidden_dims, num_classes=1
            ).to(self.device)

            num_params = model.get_num_parameters()
            parameter_counts.append(num_params)

            forward_time = self.measure_forward_pass_time(
                model, input_tensor, n_iterations
            )
            backward_time = self.measure_backward_pass_time(
                model, input_tensor, target, n_iterations
            )

            forward_times.append(forward_time)
            backward_times.append(backward_time)

            print(
                f"  Params: {num_params:,}, Forward: {forward_time:.4f} ms, Backward: {backward_time:.4f} ms"
            )

        return depths, forward_times, backward_times, parameter_counts

    def analyze_width_scaling(
        self,
        widths: List[int] = None,
        num_layers: int = 2,
        batch_size: int = 64,
        input_dim: int = 2048,
        n_iterations: int = 100,
    ) -> Tuple[List[int], List[float], List[float], List[int]]:
        """
        Analyze how time scales with layer width.

        Args:
            widths: List of layer widths to test
            num_layers: Number of hidden layers
            batch_size: Batch size for testing
            input_dim: Input feature dimension
            n_iterations: Number of iterations for averaging

        Returns:
            Tuple of (widths, forward_times, backward_times, parameter_counts)
        """
        if widths is None:
            widths = [64, 128, 256, 512, 1024, 2048, 4096]

        print(f"\nAnalyzing layer width scaling...")
        print(f"Number of layers: {num_layers}, Batch size: {batch_size}")

        forward_times = []
        backward_times = []
        parameter_counts = []

        input_tensor = torch.randn(batch_size, input_dim).to(self.device)
        target = torch.randint(0, 2, (batch_size, 1)).float().to(self.device)

        for width in widths:
            print(f"Testing width: {width}")

            hidden_dims = [width] * num_layers

            model = FeatureNN(
                input_dim=input_dim, hidden_dims=hidden_dims, num_classes=1
            ).to(self.device)

            num_params = model.get_num_parameters()
            parameter_counts.append(num_params)

            forward_time = self.measure_forward_pass_time(
                model, input_tensor, n_iterations
            )
            backward_time = self.measure_backward_pass_time(
                model, input_tensor, target, n_iterations
            )

            forward_times.append(forward_time)
            backward_times.append(backward_time)

            print(
                f"  Params: {num_params:,}, Forward: {forward_time:.4f} ms, Backward: {backward_time:.4f} ms"
            )

        return widths, forward_times, backward_times, parameter_counts

    def plot_batch_size_analysis(
        self,
        batch_sizes: List[int],
        forward_times: List[float],
        backward_times: List[float],
        save_path: str = None,
    ):
        """Plot batch size scaling analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Time vs Batch Size
        ax1.plot(
            batch_sizes,
            forward_times,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
        )
        ax1.plot(
            batch_sizes,
            backward_times,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
        )
        ax1.set_xlabel("Batch Size", fontsize=12)
        ax1.set_ylabel("Time (ms)", fontsize=12)
        ax1.set_title("Time Complexity vs Batch Size", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)

        # Plot 2: Time per Sample
        forward_per_sample = [ft / bs for ft, bs in zip(forward_times, batch_sizes)]
        backward_per_sample = [bt / bs for bt, bs in zip(backward_times, batch_sizes)]

        ax2.plot(
            batch_sizes,
            forward_per_sample,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
        )
        ax2.plot(
            batch_sizes,
            backward_per_sample,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
        )
        ax2.set_xlabel("Batch Size", fontsize=12)
        ax2.set_ylabel("Time per Sample (ms)", fontsize=12)
        ax2.set_title("Time per Sample vs Batch Size", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_depth_analysis(
        self,
        depths: List[int],
        forward_times: List[float],
        backward_times: List[float],
        parameter_counts: List[int],
        save_path: str = None,
    ):
        """Plot network depth scaling analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Time vs Depth
        ax1.plot(
            depths,
            forward_times,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
            color="blue",
        )
        ax1.plot(
            depths,
            backward_times,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
            color="red",
        )
        ax1.set_xlabel("Network Depth (Number of Hidden Layers)", fontsize=12)
        ax1.set_ylabel("Time (ms)", fontsize=12)
        ax1.set_title(
            "Time Complexity vs Network Depth", fontsize=14, fontweight="bold"
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Time vs Parameters
        ax2.plot(
            parameter_counts,
            forward_times,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
            color="blue",
        )
        ax2.plot(
            parameter_counts,
            backward_times,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
            color="red",
        )
        ax2.set_xlabel("Number of Parameters", fontsize=12)
        ax2.set_ylabel("Time (ms)", fontsize=12)
        ax2.set_title(
            "Time Complexity vs Number of Parameters", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_width_analysis(
        self,
        widths: List[int],
        forward_times: List[float],
        backward_times: List[float],
        parameter_counts: List[int],
        save_path: str = None,
    ):
        """Plot layer width scaling analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Time vs Width
        ax1.plot(
            widths,
            forward_times,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
            color="green",
        )
        ax1.plot(
            widths,
            backward_times,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
            color="orange",
        )
        ax1.set_xlabel("Layer Width", fontsize=12)
        ax1.set_ylabel("Time (ms)", fontsize=12)
        ax1.set_title("Time Complexity vs Layer Width", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)

        # Plot 2: Time vs Parameters
        ax2.plot(
            parameter_counts,
            forward_times,
            "o-",
            linewidth=2,
            markersize=8,
            label="Forward Pass",
            color="green",
        )
        ax2.plot(
            parameter_counts,
            backward_times,
            "s-",
            linewidth=2,
            markersize=8,
            label="Backward Pass",
            color="orange",
        )
        ax2.set_xlabel("Number of Parameters", fontsize=12)
        ax2.set_ylabel("Time (ms)", fontsize=12)
        ax2.set_title(
            "Time Complexity vs Number of Parameters", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_comprehensive_analysis(
        self,
        batch_data: Tuple,
        depth_data: Tuple,
        width_data: Tuple,
        save_path: str = None,
    ):
        """Create a comprehensive plot with all analyses"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Unpack data
        batch_sizes, batch_forward, batch_backward = batch_data
        depths, depth_forward, depth_backward, depth_params = depth_data
        widths, width_forward, width_backward, width_params = width_data

        # Plot 1: Batch Size - Time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(
            batch_sizes, batch_forward, "o-", linewidth=2, markersize=6, label="Forward"
        )
        ax1.plot(
            batch_sizes,
            batch_backward,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
        )
        ax1.set_xlabel("Batch Size", fontsize=11)
        ax1.set_ylabel("Time (ms)", fontsize=11)
        ax1.set_title("Time vs Batch Size", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)

        # Plot 2: Batch Size - Time per Sample
        ax2 = fig.add_subplot(gs[0, 1])
        forward_per_sample = [ft / bs for ft, bs in zip(batch_forward, batch_sizes)]
        backward_per_sample = [bt / bs for bt, bs in zip(batch_backward, batch_sizes)]
        ax2.plot(
            batch_sizes,
            forward_per_sample,
            "o-",
            linewidth=2,
            markersize=6,
            label="Forward",
        )
        ax2.plot(
            batch_sizes,
            backward_per_sample,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
        )
        ax2.set_xlabel("Batch Size", fontsize=11)
        ax2.set_ylabel("Time per Sample (ms)", fontsize=11)
        ax2.set_title("Time per Sample vs Batch Size", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)

        # Plot 3: Network Depth - Time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            depths,
            depth_forward,
            "o-",
            linewidth=2,
            markersize=6,
            label="Forward",
            color="blue",
        )
        ax3.plot(
            depths,
            depth_backward,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
            color="red",
        )
        ax3.set_xlabel("Network Depth (# Hidden Layers)", fontsize=11)
        ax3.set_ylabel("Time (ms)", fontsize=11)
        ax3.set_title("Time vs Network Depth", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Network Depth - Parameters
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(
            depth_params,
            depth_forward,
            "o-",
            linewidth=2,
            markersize=6,
            label="Forward",
            color="blue",
        )
        ax4.plot(
            depth_params,
            depth_backward,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
            color="red",
        )
        ax4.set_xlabel("Number of Parameters", fontsize=11)
        ax4.set_ylabel("Time (ms)", fontsize=11)
        ax4.set_title("Time vs Parameters (Depth)", fontsize=12, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        # Plot 5: Layer Width - Time
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(
            widths,
            width_forward,
            "o-",
            linewidth=2,
            markersize=6,
            label="Forward",
            color="green",
        )
        ax5.plot(
            widths,
            width_backward,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
            color="orange",
        )
        ax5.set_xlabel("Layer Width", fontsize=11)
        ax5.set_ylabel("Time (ms)", fontsize=11)
        ax5.set_title("Time vs Layer Width", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale("log", base=2)

        # Plot 6: Layer Width - Parameters
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(
            width_params,
            width_forward,
            "o-",
            linewidth=2,
            markersize=6,
            label="Forward",
            color="green",
        )
        ax6.plot(
            width_params,
            width_backward,
            "s-",
            linewidth=2,
            markersize=6,
            label="Backward",
            color="orange",
        )
        ax6.set_xlabel("Number of Parameters", fontsize=11)
        ax6.set_ylabel("Time (ms)", fontsize=11)
        ax6.set_title("Time vs Parameters (Width)", fontsize=12, fontweight="bold")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        fig.suptitle(
            "Neural Network Time Complexity Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comprehensive plot saved to {save_path}")

        plt.show()

    def run_full_analysis(self, output_dir: str = "experiments/time_complexity"):
        """
        Run complete time complexity analysis and generate all plots.

        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print("NEURAL NETWORK TIME COMPLEXITY ANALYSIS")
        print("=" * 70)

        # 1. Batch size analysis
        print("\n" + "=" * 70)
        print("1. BATCH SIZE SCALING ANALYSIS")
        print("=" * 70)
        batch_sizes, batch_forward, batch_backward = self.analyze_batch_size_scaling()
        self.plot_batch_size_analysis(
            batch_sizes,
            batch_forward,
            batch_backward,
            save_path=os.path.join(output_dir, "batch_size_analysis.png"),
        )

        # 2. Network depth analysis
        print("\n" + "=" * 70)
        print("2. NETWORK DEPTH SCALING ANALYSIS")
        print("=" * 70)
        depths, depth_forward, depth_backward, depth_params = (
            self.analyze_depth_scaling()
        )
        self.plot_depth_analysis(
            depths,
            depth_forward,
            depth_backward,
            depth_params,
            save_path=os.path.join(output_dir, "depth_analysis.png"),
        )

        # 3. Layer width analysis
        print("\n" + "=" * 70)
        print("3. LAYER WIDTH SCALING ANALYSIS")
        print("=" * 70)
        widths, width_forward, width_backward, width_params = (
            self.analyze_width_scaling()
        )
        self.plot_width_analysis(
            widths,
            width_forward,
            width_backward,
            width_params,
            save_path=os.path.join(output_dir, "width_analysis.png"),
        )

        # 4. Comprehensive plot
        print("\n" + "=" * 70)
        print("4. GENERATING COMPREHENSIVE ANALYSIS PLOT")
        print("=" * 70)
        self.plot_comprehensive_analysis(
            (batch_sizes, batch_forward, batch_backward),
            (depths, depth_forward, depth_backward, depth_params),
            (widths, width_forward, width_backward, width_params),
            save_path=os.path.join(output_dir, "comprehensive_analysis.png"),
        )

        # Save summary statistics
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("NEURAL NETWORK TIME COMPLEXITY ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("BATCH SIZE ANALYSIS:\n")
            f.write(f"  Batch sizes tested: {batch_sizes}\n")
            f.write(
                f"  Forward time range: {min(batch_forward):.4f} - {max(batch_forward):.4f} ms\n"
            )
            f.write(
                f"  Backward time range: {min(batch_backward):.4f} - {max(batch_backward):.4f} ms\n\n"
            )

            f.write("NETWORK DEPTH ANALYSIS:\n")
            f.write(f"  Depths tested: {depths}\n")
            f.write(
                f"  Parameter range: {min(depth_params):,} - {max(depth_params):,}\n"
            )
            f.write(
                f"  Forward time range: {min(depth_forward):.4f} - {max(depth_forward):.4f} ms\n"
            )
            f.write(
                f"  Backward time range: {min(depth_backward):.4f} - {max(depth_backward):.4f} ms\n\n"
            )

            f.write("LAYER WIDTH ANALYSIS:\n")
            f.write(f"  Widths tested: {widths}\n")
            f.write(
                f"  Parameter range: {min(width_params):,} - {max(width_params):,}\n"
            )
            f.write(
                f"  Forward time range: {min(width_forward):.4f} - {max(width_forward):.4f} ms\n"
            )
            f.write(
                f"  Backward time range: {min(width_backward):.4f} - {max(width_backward):.4f} ms\n\n"
            )

            f.write(f"Device used: {self.device}\n")

        print(f"\nSummary saved to {summary_path}")
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)


def main():
    """Main function to run time complexity analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze neural network time complexity"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run analysis on (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/time_complexity",
        help="Directory to save plots and results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick analysis with fewer iterations",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TimeComplexityAnalyzer(device=args.device)

    # Run full analysis
    if args.quick:
        print("Running quick analysis mode...")
        # You can customize this for faster testing
        analyzer.run_full_analysis(output_dir=args.output_dir)
    else:
        analyzer.run_full_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
