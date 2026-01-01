"""
Visualization utilities.

Responsibility: plots only. No logic, no computation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List


def plot_tilt_grid(df: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot model vs empirical tilt comparison grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from exp_model_vs_empirical with columns:
        P, state, emp_mean_omega_a, mod_mean_omega_a, etc.
    output_path : Path, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    states = ['PP', 'PC', 'CP', 'CC']
    P_values = sorted(df['P'].unique())

    for idx, state in enumerate(states):
        ax = axes[idx // 2, idx % 2]
        state_df = df[df['state'] == state]

        # Empirical vs model for mean omega_a
        ax.plot(P_values, state_df['emp_mean_omega_a'], 'o-', label='Empirical (a)')
        ax.plot(P_values, state_df['mod_mean_omega_a'], 's--', label='Model (a)')
        ax.plot(P_values, state_df['emp_mean_omega_b'], '^-', label='Empirical (b)')
        ax.plot(P_values, state_df['mod_mean_omega_b'], 'd--', label='Model (b)')

        ax.set_xlabel('P (prime cutoff)')
        ax.set_ylabel('Mean omega')
        ax.set_title(f'State {state}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_omega_histograms(omega_by_state: dict, state: str = 'PP',
                          output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot histograms of omega values for a given state.

    Parameters
    ----------
    omega_by_state : dict
        Dictionary from compute_omega_by_state with structure
        {state: {'a': array, 'b': array}}.
    state : str
        State to plot.
    output_path : Path, optional
        If provided, save figure.

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = omega_by_state[state]

    for idx, (component, label) in enumerate([('a', '6k-1'), ('b', '6k+1')]):
        ax = axes[idx]
        values = data[component]

        if len(values) > 0:
            bins = np.arange(values.min(), values.max() + 2) - 0.5
            ax.hist(values, bins=bins, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(values), color='red', linestyle='--',
                       label=f'Mean = {np.mean(values):.3f}')

        ax.set_xlabel(f'omega({label})')
        ax.set_ylabel('Density')
        ax.set_title(f'State {state}: {label} component')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_run_length_comparison(df: pd.DataFrame,
                               output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot run length comparison between empirical and null models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: model, mean, median, max, std, count.
    output_path : Path, optional
        If provided, save figure.

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df['model'].values
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, df['mean'], width, label='Mean', alpha=0.8)
    ax.bar(x + width/2, df['median'], width, label='Median', alpha=0.8)

    ax.set_ylabel('Run Length')
    ax.set_xlabel('Model')
    ax.set_title('PP Run Length: Empirical vs Null Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_state_frequencies(df: pd.DataFrame,
                           output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot state frequencies: empirical vs model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with state, emp_count/fraction, mod_probability.
    output_path : Path, optional
        If provided, save figure.

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get one row per state (first P value)
    states = ['PP', 'PC', 'CP', 'CC']
    first_P = df['P'].min()
    df_plot = df[df['P'] == first_P].set_index('state').loc[states]

    x = np.arange(len(states))
    width = 0.35

    total = df_plot['emp_count'].sum()
    emp_fracs = df_plot['emp_count'] / total
    mod_probs = df_plot['mod_probability']

    ax.bar(x - width/2, emp_fracs, width, label='Empirical', alpha=0.8)
    ax.bar(x + width/2, mod_probs, width, label='Model', alpha=0.8)

    ax.set_ylabel('Probability')
    ax.set_xlabel('State')
    ax.set_title('State Distribution: Empirical vs Model')
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_delta_heatmap(df: pd.DataFrame,
                       output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot heatmap of model-empirical deltas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with P, state, delta_mean_a, delta_mean_b.
    output_path : Path, optional
        If provided, save figure.

    Returns
    -------
    matplotlib.Figure
    """
    states = ['PP', 'PC', 'CP', 'CC']
    P_values = sorted(df['P'].unique())

    # Create matrices
    delta_a = np.zeros((len(states), len(P_values)))
    delta_b = np.zeros((len(states), len(P_values)))

    for i, state in enumerate(states):
        for j, P in enumerate(P_values):
            row = df[(df['state'] == state) & (df['P'] == P)]
            if len(row) > 0:
                delta_a[i, j] = row['delta_mean_a'].values[0]
                delta_b[i, j] = row['delta_mean_b'].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, delta, title in [(axes[0], delta_a, 'Delta Mean Omega (a)'),
                              (axes[1], delta_b, 'Delta Mean Omega (b)')]:
        im = ax.imshow(delta, aspect='auto', cmap='RdBu_r')
        ax.set_xticks(np.arange(len(P_values)))
        ax.set_xticklabels(P_values)
        ax.set_yticks(np.arange(len(states)))
        ax.set_yticklabels(states)
        ax.set_xlabel('P')
        ax.set_ylabel('State')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
