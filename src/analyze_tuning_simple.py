#!/usr/bin/env python3
"""
Simple Hyperparameter Tuning Analysis for GSAT

This script analyzes tuning results to identify the best hyperparameters based on:
1. Validation accuracy (model performance)
2. Validation AUROC (model performance)
3. Attention entropy (explanation quality - higher = more spread out/meaningful)

Memory-efficient: Only loads summary statistics, not full attention weights.

Usage:
    python analyze_tuning_simple.py --results_dir ~/hpc-share/ChemIntuit/MotifSAT/tuning_results
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def extract_experiment_metrics(exp_dir: Path) -> dict:
    """
    Extract key metrics from a single experiment directory.
    
    Returns dictionary with:
    - dataset, model, fold, seed
    - hyperparameters (pred_loss_coef, info_loss_coef, motif_loss_coef, init_r, final_r, etc.)
    - validation metrics (valid_acc, valid_roc)
    - attention distribution metrics (entropy, polarization)
    """
    # Load experiment summary
    summary_path = exp_dir / 'experiment_summary.json'
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load final metrics
    final_metrics_path = exp_dir / 'final_metrics.json'
    if not final_metrics_path.exists():
        return None
    
    with open(final_metrics_path, 'r') as f:
        final_metrics = json.load(f)
    
    # Get best epoch
    best_epoch = final_metrics.get('metric/best_clf_epoch', -1)
    
    # Extract attention distribution for best epoch (validation phase)
    attention_path = exp_dir / 'attention_distributions.jsonl'
    attention_metrics = {
        'entropy': np.nan,
        'pct_near_0': np.nan,
        'pct_near_1': np.nan,
        'pct_middle': np.nan,
        'polarization': np.nan,
        'mean': np.nan,
        'std': np.nan,
    }
    
    if attention_path.exists() and best_epoch >= 0:
        with open(attention_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Find the best epoch, validation phase
                if data.get('epoch') == best_epoch and data.get('phase') == 'valid':
                    attention_metrics['entropy'] = data.get('entropy', np.nan)
                    attention_metrics['pct_near_0'] = data.get('pct_near_0', np.nan)
                    attention_metrics['pct_near_1'] = data.get('pct_near_1', np.nan)
                    attention_metrics['pct_middle'] = data.get('pct_middle', np.nan)
                    attention_metrics['polarization'] = (
                        attention_metrics['pct_near_0'] + attention_metrics['pct_near_1']
                    )
                    attention_metrics['mean'] = data.get('mean', np.nan)
                    attention_metrics['std'] = data.get('std', np.nan)
                    break
    
    # Construct result dictionary
    result = {
        # Identifiers
        'exp_dir': str(exp_dir),
        'dataset': summary['dataset'],
        'model': summary['model'],
        'fold': summary['fold'],
        'seed': summary['seed'],
        'experiment_name': summary.get('experiment_name', 'unknown'),
        'tuning_id': summary.get('tuning_id', 'default'),
        
        # Hyperparameters
        'pred_loss_coef': summary['loss_coefficients']['pred_loss_coef'],
        'info_loss_coef': summary['loss_coefficients']['info_loss_coef'],
        'motif_loss_coef': summary['loss_coefficients']['motif_loss_coef'],
        'init_r': summary['weight_distribution_params']['init_r'],
        'final_r': summary['weight_distribution_params']['final_r'],
        'decay_r': summary['weight_distribution_params']['decay_r'],
        'decay_interval': summary['weight_distribution_params']['decay_interval'],
        
        # Validation performance
        'best_epoch': best_epoch,
        'valid_acc': final_metrics.get('metric/best_clf_valid', np.nan),
        'valid_roc': final_metrics.get('metric/best_clf_roc_valid', np.nan),
        'valid_clf_acc': final_metrics.get('metric/best_clf_acc_valid', np.nan),
        'valid_clf_roc': final_metrics.get('metric/best_clf_roc_valid', np.nan),
        
        # Test performance (for reference)
        'test_acc': final_metrics.get('metric/best_clf_test', np.nan),
        'test_roc': final_metrics.get('metric/best_clf_roc_test', np.nan),
        'test_clf_acc': final_metrics.get('metric/best_clf_acc_test', np.nan),
        'test_clf_roc': final_metrics.get('metric/best_clf_roc_test', np.nan),
        
        # Attention distribution (explanation quality)
        'attention_entropy': attention_metrics['entropy'],
        'attention_mean': attention_metrics['mean'],
        'attention_std': attention_metrics['std'],
        'attention_pct_near_0': attention_metrics['pct_near_0'],
        'attention_pct_near_1': attention_metrics['pct_near_1'],
        'attention_pct_middle': attention_metrics['pct_middle'],
        'attention_polarization': attention_metrics['polarization'],
    }
    
    return result


def scan_tuning_results(results_dir: Path, experiment_filter=None) -> pd.DataFrame:
    """
    Scan all experiment directories and extract metrics.
    
    Args:
        results_dir: Root directory containing tuning results
        experiment_filter: Optional string to filter by experiment name
    
    Returns:
        DataFrame with all experiment metrics
    """
    print(f"Scanning tuning results in: {results_dir}")
    
    # Find all experiment directories
    experiment_dirs = list(results_dir.rglob('experiment_summary.json'))
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Extract metrics from each experiment
    results = []
    for summary_path in experiment_dirs:
        exp_dir = summary_path.parent
        
        # Apply experiment filter if specified
        if experiment_filter and experiment_filter not in str(exp_dir):
            continue
        
        try:
            metrics = extract_experiment_metrics(exp_dir)
            if metrics is not None:
                results.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to process {exp_dir}: {e}")
    
    print(f"Successfully extracted metrics from {len(results)} experiments")
    
    return pd.DataFrame(results)


def create_hyperparameter_label(row, varying_params):
    """Create a readable label for hyperparameter configuration."""
    parts = []
    for param in varying_params:
        value = row[param]
        # Shorten parameter names
        param_short = param.replace('_loss_coef', '').replace('_r', '').replace('_', '')
        parts.append(f"{param_short}={value:.2f}" if isinstance(value, float) else f"{param_short}={value}")
    return '\n'.join(parts)


def plot_comparison_figure(df: pd.DataFrame, dataset: str, model: str, output_dir: Path):
    """
    Create comparison figure for a specific dataset-model combination.
    
    Shows 3 metrics (rows): validation accuracy, validation AUROC, attention entropy
    Each column represents a different hyperparameter configuration.
    """
    # Filter to specific dataset and model
    subset = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
    
    if len(subset) == 0:
        print(f"  No data for {dataset}-{model}")
        return
    
    # Identify which hyperparameters vary
    hyperparam_cols = ['pred_loss_coef', 'info_loss_coef', 'motif_loss_coef',
                       'init_r', 'final_r', 'decay_r', 'decay_interval']
    varying_params = [col for col in hyperparam_cols if subset[col].nunique() > 1]
    
    print(f"  {dataset}-{model}: {len(subset)} configs, varying params: {varying_params}")
    
    # Sort by validation AUROC (descending) for better visualization
    metric_col = 'valid_clf_roc' if 'valid_clf_roc' in subset.columns else 'valid_roc'
    subset = subset.sort_values(metric_col, ascending=False)
    
    # Limit to top 30 configs for readability
    if len(subset) > 30:
        print(f"    Showing top 30 configs (out of {len(subset)})")
        subset = subset.head(30)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(subset) * 0.5), 12))
    fig.suptitle(f'{dataset} - {model}\nHyperparameter Comparison', 
                 fontsize=16, fontweight='bold')
    
    # X-axis: config indices
    x = np.arange(len(subset))
    
    # Create labels for x-axis
    if len(varying_params) <= 2:
        labels = [create_hyperparameter_label(row, varying_params) 
                  for _, row in subset.iterrows()]
    else:
        # Too many params, just use tuning_id
        labels = subset['tuning_id'].tolist()
    
    # Metrics to plot
    metrics = [
        ('valid_clf_acc' if 'valid_clf_acc' in subset.columns else 'valid_acc', 
         'Validation Accuracy', 'green'),
        (metric_col, 'Validation AUROC', 'blue'),
        ('attention_entropy', 'Attention Entropy\n(Explanation Quality)', 'orange'),
    ]
    
    for ax, (metric, title, color) in zip(axes, metrics):
        values = subset[metric].values
        
        # Bar plot
        bars = ax.bar(x, values, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Highlight top configs
        if metric in [metrics[1][0]]:  # validation AUROC
            # Top 3 configs get gold border
            for i in range(min(3, len(bars))):
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(3)
        
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars for top configs
        for i in range(min(5, len(bars))):
            height = bars[i].get_height()
            if not np.isnan(height):
                ax.text(bars[i].get_x() + bars[i].get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set x-axis labels only on bottom plot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
    axes[-1].set_xlabel('Hyperparameter Configuration', fontsize=12, fontweight='bold')
    
    # Remove x-axis labels from top plots
    for ax in axes[:-1]:
        ax.set_xticks([])
    
    plt.tight_layout()
    
    # Save figure
    filename = f'comparison_{dataset}_{model}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")


def plot_aggregated_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create aggregated comparison across all experiments.
    Shows average performance by hyperparameter values.
    """
    # Identify varying hyperparameters across all experiments
    hyperparam_cols = ['pred_loss_coef', 'info_loss_coef', 'motif_loss_coef',
                       'init_r', 'final_r', 'decay_r', 'decay_interval']
    varying_params = [col for col in hyperparam_cols if df[col].nunique() > 1]
    
    if not varying_params:
        print("  No varying hyperparameters found")
        return
    
    # Create figure for each varying parameter
    for param in varying_params:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Effect of {param} on Performance', fontsize=16, fontweight='bold')
        
        # Group by parameter value and compute statistics
        grouped = df.groupby(param).agg({
            'valid_clf_acc': ['mean', 'std', 'count'],
            'valid_clf_roc': ['mean', 'std', 'count'],
            'attention_entropy': ['mean', 'std', 'count'],
        }).reset_index()
        
        # Plot 1: Validation Accuracy
        ax = axes[0]
        x = grouped[param].values
        y = grouped[('valid_clf_acc', 'mean')].values
        yerr = grouped[('valid_clf_acc', 'std')].values
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel(param, fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Validation Accuracy', fontsize=11)
        ax.grid(alpha=0.3)
        
        # Plot 2: Validation AUROC
        ax = axes[1]
        y = grouped[('valid_clf_roc', 'mean')].values
        yerr = grouped[('valid_clf_roc', 'std')].values
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2, markersize=8, color='blue')
        ax.set_xlabel(param, fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation AUROC', fontsize=12, fontweight='bold')
        ax.set_title('Validation AUROC', fontsize=11)
        ax.grid(alpha=0.3)
        
        # Plot 3: Attention Entropy
        ax = axes[2]
        y = grouped[('attention_entropy', 'mean')].values
        yerr = grouped[('attention_entropy', 'std')].values
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2, markersize=8, color='orange')
        ax.set_xlabel(param, fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Entropy', fontsize=12, fontweight='bold')
        ax.set_title('Attention Entropy (Explanation Quality)', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'effect_of_{param}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def identify_best_configs(df: pd.DataFrame, output_dir: Path):
    """
    Identify and save best configurations based on combined criteria:
    1. High validation AUROC (performance)
    2. High attention entropy (explanation quality)
    """
    print("\nIdentifying best configurations...")
    
    results = []
    
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
            
            if len(subset) == 0:
                continue
            
            # Use valid_clf_roc or valid_roc depending on what's available
            roc_col = 'valid_clf_roc' if 'valid_clf_roc' in subset.columns else 'valid_roc'
            
            # Normalize metrics to [0, 1] for comparison
            subset['roc_normalized'] = (subset[roc_col] - subset[roc_col].min()) / (subset[roc_col].max() - subset[roc_col].min() + 1e-8)
            subset['entropy_normalized'] = (subset['attention_entropy'] - subset['attention_entropy'].min()) / (subset['attention_entropy'].max() - subset['attention_entropy'].min() + 1e-8)
            
            # Combined score: 70% performance, 30% explanation quality
            subset['combined_score'] = 0.7 * subset['roc_normalized'] + 0.3 * subset['entropy_normalized']
            
            # Sort by combined score
            subset = subset.sort_values('combined_score', ascending=False)
            
            # Get top 3 configs
            for rank, (idx, row) in enumerate(subset.head(3).iterrows(), 1):
                results.append({
                    'rank': rank,
                    'dataset': dataset,
                    'model': model,
                    'tuning_id': row['tuning_id'],
                    'exp_dir': row['exp_dir'],
                    'valid_roc': row[roc_col],
                    'valid_acc': row.get('valid_clf_acc', row.get('valid_acc', np.nan)),
                    'attention_entropy': row['attention_entropy'],
                    'combined_score': row['combined_score'],
                    'pred_loss_coef': row['pred_loss_coef'],
                    'info_loss_coef': row['info_loss_coef'],
                    'motif_loss_coef': row['motif_loss_coef'],
                    'init_r': row['init_r'],
                    'final_r': row['final_r'],
                    'decay_r': row['decay_r'],
                    'decay_interval': row['decay_interval'],
                })
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = output_dir / 'best_configurations.csv'
    results_df.to_csv(output_file, index=False)
    print(f"  Saved best configurations to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TOP CONFIGURATIONS SUMMARY")
    print("="*80)
    for _, row in results_df.iterrows():
        if row['rank'] == 1:
            print(f"\n{row['dataset']} - {row['model']}:")
            print(f"  Config: {row['tuning_id']}")
            print(f"  Val AUROC: {row['valid_roc']:.4f}")
            print(f"  Val Acc: {row['valid_acc']:.4f}")
            print(f"  Attention Entropy: {row['attention_entropy']:.4f}")
            print(f"  Hyperparameters:")
            print(f"    pred_loss_coef={row['pred_loss_coef']}, info_loss_coef={row['info_loss_coef']}, motif_loss_coef={row['motif_loss_coef']}")
            print(f"    init_r={row['init_r']}, final_r={row['final_r']}, decay_r={row['decay_r']}, decay_interval={row['decay_interval']}")
    print("="*80)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Simple hyperparameter tuning analysis for GSAT',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing tuning results')
    
    parser.add_argument('--output_dir', type=str, default='./tuning_analysis',
                        help='Directory to save analysis outputs')
    
    parser.add_argument('--experiment_filter', type=str, default=None,
                        help='Filter by experiment name (e.g., weight_tuning)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GSAT HYPERPARAMETER TUNING ANALYSIS")
    print("="*80 + "\n")
    
    # Step 1: Scan and extract metrics
    df = scan_tuning_results(Path(args.results_dir), args.experiment_filter)
    
    if len(df) == 0:
        print("No experiments found!")
        return
    
    # Save raw data
    raw_data_file = output_dir / 'all_experiments.csv'
    df.to_csv(raw_data_file, index=False)
    print(f"\nSaved raw data to: {raw_data_file}")
    
    # Step 2: Create comparison plots for each dataset-model combination
    print("\nCreating comparison figures...")
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            plot_comparison_figure(df, dataset, model, output_dir)
    
    # Step 3: Create aggregated comparison plots
    print("\nCreating aggregated comparison plots...")
    plot_aggregated_comparison(df, output_dir)
    
    # Step 4: Identify best configurations
    best_configs = identify_best_configs(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()

