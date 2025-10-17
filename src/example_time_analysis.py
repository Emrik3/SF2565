"""
Some time complexity analysis for depth and width.
"""

import os
import torch
from plot_time_complexity import TimeComplexityAnalyzer


def network_depth():
    print("\n" + "=" * 70)

    analyzer = TimeComplexityAnalyzer()

    # Test different network depths
    depths = [1, 2, 3, 4, 5, 7, 10]

    depths_list, forward_times, backward_times, params = analyzer.analyze_depth_scaling(
        depths=depths, layer_width=512, batch_size=64, n_iterations=50
    )

    # Plot results
    output_dir = "experiments/time_complexity_example3"
    os.makedirs(output_dir, exist_ok=True)
    analyzer.plot_depth_analysis(
        depths_list,
        forward_times,
        backward_times,
        params,
        save_path=os.path.join(output_dir, "depth_analysis.png"),
    )


def layer_width():
    print("\n" + "=" * 70)

    analyzer = TimeComplexityAnalyzer()

    # Test different layer widths
    widths = [64, 128, 256, 512, 1024, 2048]

    widths_list, forward_times, backward_times, params = analyzer.analyze_width_scaling(
        widths=widths, num_layers=3, batch_size=64, n_iterations=50
    )

    # Plot results
    output_dir = "experiments/time_complexity_example4"
    os.makedirs(output_dir, exist_ok=True)
    analyzer.plot_width_analysis(
        widths_list,
        forward_times,
        backward_times,
        params,
        save_path=os.path.join(output_dir, "width_analysis.png"),
    )


def main():
    network_depth()
    layer_width()


if __name__ == "__main__":
    main()
