#!/usr/bin/env python3
"""
Full reproducibility script.

Running this file regenerates every table and figure in the paper.

Usage:
    python run_all.py
    python run_all.py --config config/custom.yaml
"""

import argparse
import yaml
from pathlib import Path
import time

from src.experiments.exp_selection_bias import run_selection_bias_experiment
from src.experiments.exp_model_vs_empirical import run_model_comparison_experiment
from src.experiments.exp_run_lengths import run_run_length_experiment
from src.plotting import (
    plot_tilt_grid,
    plot_run_length_comparison,
    plot_state_frequencies,
    plot_delta_heatmap
)


def main():
    parser = argparse.ArgumentParser(description='Run all twin prime experiments')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Twin Prime Selection Bias - Full Experiment Suite")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  K = {config['K']:,}")
    print(f"  P_grid = {config['P_grid']}")
    print(f"  block_size = {config['block_size']}")
    print(f"  seed = {config['seed']}")
    print()

    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # 1. Selection Bias Experiment (Paper Section 3)
    print("-" * 60)
    print("1. Selection Bias Experiment (Section 3)")
    print("-" * 60)
    start = time.time()
    df_bias = run_selection_bias_experiment(
        config['K'],
        output_dir,
        config['seed']
    )
    print(f"   Completed in {time.time() - start:.1f}s")
    print()

    # 2. Model vs Empirical Comparison (Sections 5-6)
    print("-" * 60)
    print("2. Model vs Empirical Comparison (Sections 5-6)")
    print("-" * 60)
    start = time.time()
    df_comparison = run_model_comparison_experiment(
        config['K'],
        config['P_grid'],
        output_dir
    )
    print(f"   Completed in {time.time() - start:.1f}s")
    print()

    # 3. Run Length Analysis (Appendix)
    print("-" * 60)
    print("3. Run Length Analysis (Appendix)")
    print("-" * 60)
    start = time.time()
    results_runs = run_run_length_experiment(
        config['K'],
        config['P_grid'],
        output_dir,
        config['seed']
    )
    print(f"   Completed in {time.time() - start:.1f}s")
    print()

    # 4. Generate Figures
    print("-" * 60)
    print("4. Generating Figures")
    print("-" * 60)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("  - Tilt grid comparison...")
    plot_tilt_grid(df_comparison, figures_dir / 'tilt_grid.png')

    print("  - State frequencies...")
    plot_state_frequencies(df_comparison, figures_dir / 'state_frequencies.png')

    print("  - Delta heatmap...")
    plot_delta_heatmap(df_comparison, figures_dir / 'delta_heatmap.png')

    print("  - Run length comparison...")
    plot_run_length_comparison(
        results_runs['comparison'],
        figures_dir / 'run_length_comparison.png'
    )

    print()

    # Summary
    total_time = time.time() - total_start
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")

    for f in sorted(output_dir.glob('*.csv')):
        print(f"  - {f.name}")

    print(f"\nFigures:")
    for f in sorted(figures_dir.glob('*.png')):
        print(f"  - figures/{f.name}")

    # Print key results
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)

    print("\nState distribution (empirical):")
    print(df_bias[['state', 'count', 'fraction', 'mean_omega_a', 'mean_omega_b']].to_string(index=False))

    pp_comp = df_comparison[df_comparison['state'] == 'PP'][
        ['P', 'emp_mean_omega_a', 'mod_mean_omega_a', 'delta_mean_a']
    ]
    print("\nPP state: Empirical vs Model (component a):")
    print(pp_comp.to_string(index=False))


if __name__ == '__main__':
    main()
