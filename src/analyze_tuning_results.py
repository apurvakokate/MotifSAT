#!/usr/bin/env python3
"""
Analyze tuning results from GSAT experiments.

This script comprehensively analyzes all saved metrics to answer the research questions:
1. What is the best model and explainer performance for each dataset?
2. How does motif_consistency_loss affect:
   - (i) Score consistency within motif nodes
   - (ii) Model prediction performance
   - (iii) Explainer performance (Pearson correlation)
   - (iv) Distribution of node weights

Usage:
    python analyze_tuning_results.py --results_dir tuning_results --output_dir analysis_results
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import pdb

class TuningResultsAnalyzer:
    """Comprehensive analyzer for tuning experiment results."""
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.all_experiments = []
        self.summary_df = None
        self.node_scores_df = None
        self.masked_impact_df = None
        self.masked_edge_impact_df = None
        self.epoch_metrics_df = None
        self.attention_dist_df = None
        
    def collect_all_experiments(self):
        """Recursively find all experiment directories and load their data."""
        print("Collecting all experiment results...")
        
        experiment_dirs = list(self.results_dir.rglob('experiment_summary.json'))
        print(f"Found {len(experiment_dirs)} experiments")
        
        for summary_path in experiment_dirs:
            exp_dir = summary_path.parent
            try:
                exp_data = self._load_experiment_data(exp_dir)
                self.all_experiments.append(exp_data)
            except Exception as e:
                print(f"Warning: Failed to load {exp_dir}: {e}")
        
        print(f"Successfully loaded {len(self.all_experiments)} experiments")
        return self.all_experiments

    def _plot_interaction_heatmap(self, df, param1, param2, metric, 
                               dataset, model, experiment_name):
        """Create heatmap showing interaction between two parameters."""
        
        # Remove NaN
        plot_data = df[[param1, param2, metric]].dropna()
        
        if len(plot_data) < 5:
            return
        
        # Create pivot table
        try:
            pivot = plot_data.pivot_table(
                values=metric, 
                index=param1, 
                columns=param2, 
                aggfunc='mean'
            )
        except:
            return
        
        if pivot.empty:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                    ax=ax, cbar_kws={'label': metric})
        
        ax.set_title(f'{dataset} - {model}\n{param1} × {param2} Interaction')
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        
        plt.tight_layout()
        
        exp_suffix = f'_{experiment_name}' if experiment_name else ''
        filename = f'interaction_{dataset}_{model}_{param1}_{param2}{exp_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved interaction plot: {filename}")
    
    def _load_experiment_data(self, exp_dir: Path) -> Dict:
        """Load all data from a single experiment directory."""
        
        # Load summary
        with open(exp_dir / 'experiment_summary.json', 'r') as f:
            summary = json.load(f)
        
        # Load final metrics
        final_metrics = {}
        final_metrics_path = exp_dir / 'final_metrics.json'
        if final_metrics_path.exists():
            with open(final_metrics_path, 'r') as f:
                final_metrics = json.load(f)
        
        # Load node scores
        node_scores = []
        node_scores_path = exp_dir / 'node_scores.jsonl'
        if node_scores_path.exists():
            with open(node_scores_path, 'r') as f:
                for line in f:
                    node_scores.append(json.loads(line))
        
        # Load masked impact
        masked_impact = []
        masked_impact_path = exp_dir / 'masked-impact.jsonl'
        if masked_impact_path.exists():
            with open(masked_impact_path, 'r') as f:
                for line in f:
                    masked_impact.append(json.loads(line))
        
        # Load masked edge impact
        masked_edge_impact = []
        masked_edge_impact_path = exp_dir / 'masked-edge-impact.jsonl'
        if masked_edge_impact_path.exists():
            with open(masked_edge_impact_path, 'r') as f:
                for line in f:
                    masked_edge_impact.append(json.loads(line))
        
        # Load epoch metrics
        epoch_metrics = []
        epoch_metrics_path = exp_dir / 'epoch_metrics.jsonl'
        if epoch_metrics_path.exists():
            with open(epoch_metrics_path, 'r') as f:
                for line in f:
                    epoch_metrics.append(json.loads(line))
        
        # Load attention distributions
        attention_dist = []
        attention_dist_path = exp_dir / 'attention_distributions.jsonl'
        if attention_dist_path.exists():
            with open(attention_dist_path, 'r') as f:
                for line in f:
                    attention_dist.append(json.loads(line))
        
        return {
            'exp_dir': str(exp_dir),
            'summary': summary,
            'final_metrics': final_metrics,
            'node_scores': node_scores,
            'masked_impact': masked_impact,
            'masked_edge_impact': masked_edge_impact,
            'epoch_metrics': epoch_metrics,
            'attention_dist': attention_dist,
        }
    
    def create_summary_dataframe(self):
        """Create a summary DataFrame of all experiments."""
        print("\nCreating summary DataFrame...")
        
        rows = []
        for exp in self.all_experiments:
            summary = exp['summary']
            final_metrics = exp['final_metrics']
            
            row = {
                # Experiment identifiers
                'exp_dir': exp['exp_dir'],
                'dataset': summary['dataset'],
                'model': summary['model'],
                'fold': summary['fold'],
                'seed': summary['seed'],
                'tuning_id': summary.get('tuning_id', 'default'),
                
                # Hyperparameters
                'pred_loss_coef': summary['loss_coefficients']['pred_loss_coef'],
                'info_loss_coef': summary['loss_coefficients']['info_loss_coef'],
                'motif_loss_coef': summary['loss_coefficients']['motif_loss_coef'],
                'init_r': summary['weight_distribution_params']['init_r'],
                'final_r': summary['weight_distribution_params']['final_r'],
                'decay_r': summary['weight_distribution_params']['decay_r'],
                'decay_interval': summary['weight_distribution_params']['decay_interval'],
                
                # Final metrics
                'best_epoch': final_metrics.get('metric/best_clf_epoch', np.nan),
                'train_acc': final_metrics.get('metric/best_clf_train', np.nan),
                'valid_acc': final_metrics.get('metric/best_clf_valid', np.nan),
                'test_acc': final_metrics.get('metric/best_clf_test', np.nan),
                'train_x_roc': final_metrics.get('metric/best_x_roc_train', np.nan),
                'valid_x_roc': final_metrics.get('metric/best_x_roc_valid', np.nan),
                'test_x_roc': final_metrics.get('metric/best_x_roc_test', np.nan),
                'train_x_precision': final_metrics.get('metric/best_x_precision_train', np.nan),
                'valid_x_precision': final_metrics.get('metric/best_x_precision_valid', np.nan),
                'test_x_precision': final_metrics.get('metric/best_x_precision_test', np.nan),
            }
            rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
        
        # Save to CSV
        output_path = self.output_dir / 'summary_all_experiments.csv'
        self.summary_df.to_csv(output_path, index=False)
        print(f"Saved summary to {output_path}")
        
        return self.summary_df
    
    def analyze_within_motif_consistency(self):
        """
        RQ 2(i): Analyze score consistency within motif nodes.
        
        Returns DataFrame with consistency metrics per experiment.
        """
        print("\nAnalyzing within-motif score consistency...")
        
        consistency_results = []
        
        for exp in self.all_experiments:
            if not exp['node_scores']:
                continue
            
            node_df = pd.DataFrame(exp['node_scores'])
            summary = exp['summary']
            
            # Calculate consistency for each split
            for split in ['train', 'valid', 'test']:
                split_data = node_df[node_df['split'] == split]
                
                if len(split_data) == 0:
                    continue
                
                # Group by graph and motif
                motif_groups = split_data.groupby(['graph_idx', 'motif_index'])
                
                # Calculate variance within each motif
                motif_variances = []
                motif_stds = []
                motif_ranges = []
                motif_coeffs_of_var = []
                
                for (graph_idx, motif_idx), group in motif_groups:
                    if len(group) > 1:  # Need at least 2 nodes
                        scores = group['score'].values
                        motif_variances.append(np.var(scores))
                        motif_stds.append(np.std(scores))
                        motif_ranges.append(np.max(scores) - np.min(scores))
                        
                        # Coefficient of variation (normalized std)
                        mean_score = np.mean(scores)
                        if mean_score > 0:
                            cv = np.std(scores) / mean_score
                            motif_coeffs_of_var.append(cv)
                
                if motif_variances:
                    result = {
                        'exp_dir': exp['exp_dir'],
                        'dataset': summary['dataset'],
                        'model': summary['model'],
                        'fold': summary['fold'],
                        'seed': summary['seed'],
                        'motif_loss_coef': summary['loss_coefficients']['motif_loss_coef'],
                        'split': split,
                        'num_motifs': len(motif_variances),
                        'avg_variance': np.mean(motif_variances),
                        'avg_std': np.mean(motif_stds),
                        'avg_range': np.mean(motif_ranges),
                        'avg_coeff_of_var': np.mean(motif_coeffs_of_var) if motif_coeffs_of_var else np.nan,
                        'median_variance': np.median(motif_variances),
                        'median_std': np.median(motif_stds),
                    }
                    consistency_results.append(result)
        
        consistency_df = pd.DataFrame(consistency_results)
        
        # Save results
        output_path = self.output_dir / 'within_motif_consistency.csv'
        consistency_df.to_csv(output_path, index=False)
        print(f"Saved consistency analysis to {output_path}")
        
        # Create visualization
        self._plot_consistency_vs_motif_loss(consistency_df)
        
        return consistency_df
    
    def analyze_explainer_performance(self):
        """
        RQ 2(iii): Analyze explainer performance using Pearson correlation
        between averaged node scores and motif impact.
        
        Returns DataFrame with correlation metrics per experiment.
        """
        print("\nAnalyzing explainer performance (Pearson correlation)...")
        
        explainer_results = []
        
        for exp in self.all_experiments:
            if not exp['node_scores'] or not exp['masked_edge_impact']:
                continue
            
            node_df = pd.DataFrame(exp['node_scores'])
            impact_df = pd.DataFrame(exp['masked_edge_impact'])
            summary = exp['summary']
            
            # Calculate prediction change from masking
            impact_df['prediction_change'] = np.abs(
                impact_df['old_prediction'] - impact_df['new_prediction']
            )
            
            # For each split
            for split in ['train', 'valid', 'test']:
                split_nodes = node_df[node_df['split'] == split]
                split_impact = impact_df[impact_df['split'] == split]
                
                if len(split_nodes) == 0 or len(split_impact) == 0:
                    continue
                
                # Calculate average node score per motif
                avg_motif_scores = split_nodes.groupby(['graph_idx', 'motif_index'])['score'].mean().reset_index()
                avg_motif_scores.columns = ['graph_idx', 'motif_index', 'avg_node_score']
                
                split_impact = split_impact.rename(columns={'motif_idx': 'motif_index'})
                
                # Merge with impact data
                merged = split_impact.merge(
                    avg_motif_scores,
                    on=['graph_idx', 'motif_index'],
                    how='inner'
                )
                
                if len(merged) > 2:  # Need at least 3 points for correlation
                    # Calculate Pearson correlation
                    corr, p_value = stats.pearsonr(
                        merged['avg_node_score'],
                        merged['prediction_change']
                    )
                    
                    # Calculate Spearman correlation (rank-based, more robust)
                    spearman_corr, spearman_p = stats.spearmanr(
                        merged['avg_node_score'],
                        merged['prediction_change']
                    )
                    
                    result = {
                        'exp_dir': exp['exp_dir'],
                        'dataset': summary['dataset'],
                        'model': summary['model'],
                        'fold': summary['fold'],
                        'seed': summary['seed'],
                        'motif_loss_coef': summary['loss_coefficients']['motif_loss_coef'],
                        'split': split,
                        'num_samples': len(merged),
                        'pearson_corr': corr,
                        'pearson_p_value': p_value,
                        'spearman_corr': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'mean_avg_node_score': merged['avg_node_score'].mean(),
                        'mean_prediction_change': merged['prediction_change'].mean(),
                    }
                    explainer_results.append(result)
        
        explainer_df = pd.DataFrame(explainer_results)
        
        # Save results
        output_path = self.output_dir / 'explainer_performance.csv'
        explainer_df.to_csv(output_path, index=False)
        print(f"Saved explainer performance to {output_path}")
        
        # Create visualization
        self._plot_explainer_performance(explainer_df)
        
        return explainer_df
    
    def analyze_weight_distribution(self):
        """
        RQ 2(iv): Analyze distribution of node weights.
        
        Returns DataFrame with distribution metrics per experiment.
        """
        print("\nAnalyzing weight distributions...")
        
        distribution_results = []
        
        for exp in self.all_experiments:
            if not exp['attention_dist']:
                continue
            
            dist_df = pd.DataFrame(exp['attention_dist'])
            summary = exp['summary']
            
            # Get final epoch distributions for each split
            final_epoch = dist_df['epoch'].max()
            final_dist = dist_df[dist_df['epoch'] == final_epoch]
            
            for _, row in final_dist.iterrows():
                result = {
                    'exp_dir': exp['exp_dir'],
                    'dataset': summary['dataset'],
                    'model': summary['model'],
                    'fold': summary['fold'],
                    'seed': summary['seed'],
                    'motif_loss_coef': summary['loss_coefficients']['motif_loss_coef'],
                    'init_r': summary['weight_distribution_params']['init_r'],
                    'final_r': summary['weight_distribution_params']['final_r'],
                    'phase': row['phase'],
                    'epoch': row['epoch'],
                    'mean': row['mean'],
                    'std': row['std'],
                    'entropy': row['entropy'],
                    'pct_near_0': row['pct_near_0'],
                    'pct_near_1': row['pct_near_1'],
                    'pct_middle': row['pct_middle'],
                    'polarization_score': row['pct_near_0'] + row['pct_near_1'],  # Combined polarization
                }
                distribution_results.append(result)
        
        distribution_df = pd.DataFrame(distribution_results)
        
        # Save results
        output_path = self.output_dir / 'weight_distributions.csv'
        distribution_df.to_csv(output_path, index=False)
        print(f"Saved weight distributions to {output_path}")
        
        # Create visualizations
        self._plot_weight_distributions(distribution_df)
        
        return distribution_df

    def analyze_hyperparameter_effects(self, experiment_name=None):
        """
        Analyze how hyperparameters affect model performance.
        
        For each experiment type, shows:
        - Main effects of each parameter
        - Interaction effects between parameters
        - Optimal parameter values
        - Statistical significance
        
        Args:
            experiment_name: Filter to specific experiment (e.g., 'weight_tuning')
        """
        print("\nAnalyzing hyperparameter effects...")
        
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        df = self.summary_df.copy()
        
        # Filter to specific experiment if requested
        if experiment_name:
            df = df[df['exp_dir'].str.contains(f'experiment_{experiment_name}')]
            print(f"Filtered to experiment: {experiment_name}")
        
        # Identify which parameters vary (these are the ones being tuned)
        param_columns = ['pred_loss_coef', 'info_loss_coef', 'motif_loss_coef',
                        'init_r', 'final_r', 'decay_r', 'decay_interval']
        
        varying_params = []
        for col in param_columns:
            if col in df.columns and df[col].nunique() > 1:
                varying_params.append(col)
        
        print(f"Parameters being tuned: {varying_params}")
        
        # Performance metrics to analyze
        perf_metrics = ['valid_acc', 'test_acc', 'test_x_roc']
        
        results = []
        
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
                
                if len(subset) < 3:  # Need enough data points
                    continue
                
                for param in varying_params:
                    for metric in perf_metrics:
                        if metric not in subset.columns:
                            continue
                        
                        # Remove NaN values
                        valid_data = subset[[param, metric]].dropna()
                        
                        if len(valid_data) < 3:
                            continue
                        
                        # Calculate correlation
                        corr, p_value = stats.pearsonr(valid_data[param], valid_data[metric])
                        
                        # Calculate Spearman (rank-based, more robust)
                        spearman_corr, spearman_p = stats.spearmanr(valid_data[param], valid_data[metric])
                        
                        # Find best parameter value
                        best_idx = valid_data[metric].idxmax()
                        best_param_value = valid_data.loc[best_idx, param]
                        best_metric_value = valid_data.loc[best_idx, metric]
                        
                        # Calculate mean performance at each parameter value
                        param_means = valid_data.groupby(param)[metric].agg(['mean', 'std', 'count'])
                        
                        results.append({
                            'dataset': dataset,
                            'model': model,
                            'parameter': param,
                            'metric': metric,
                            'correlation': corr,
                            'p_value': p_value,
                            'spearman_corr': spearman_corr,
                            'spearman_p': spearman_p,
                            'significant': p_value < 0.05,
                            'best_param_value': best_param_value,
                            'best_metric_value': best_metric_value,
                            'num_samples': len(valid_data),
                        })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.output_dir / 'hyperparameter_effects.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Saved hyperparameter effects to {output_path}")
        
        # Create visualizations
        self._plot_hyperparameter_effects(df, varying_params, perf_metrics, experiment_name)
        
        return results_df

    def _plot_hyperparameter_effects(self, df, varying_params, perf_metrics, experiment_name):
        """Create plots showing hyperparameter effects."""
        
        # For each dataset and model combination
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
                
                if len(subset) < 3:
                    continue
                
                # Create a figure with subplots for each parameter
                n_params = len(varying_params)
                n_metrics = len([m for m in perf_metrics if m in subset.columns])
                
                if n_params == 0 or n_metrics == 0:
                    continue
                
                fig, axes = plt.subplots(n_metrics, n_params, 
                                        figsize=(5*n_params, 4*n_metrics))
                
                if n_metrics == 1:
                    axes = axes.reshape(1, -1)
                if n_params == 1:
                    axes = axes.reshape(-1, 1)
                
                fig.suptitle(f'{dataset} - {model}\nHyperparameter Effects', fontsize=16)
                
                for i, metric in enumerate(perf_metrics):
                    if metric not in subset.columns:
                        continue
                        
                    for j, param in enumerate(varying_params):
                        ax = axes[i, j]
                        
                        # Remove NaN values
                        plot_data = subset[[param, metric]].dropna()
                        
                        if len(plot_data) < 2:
                            ax.text(0.5, 0.5, 'Insufficient data', 
                                ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{param} vs {metric}')
                            continue
                        
                        # If parameter has few unique values, use boxplot
                        if plot_data[param].nunique() <= 5:
                            plot_data.boxplot(column=metric, by=param, ax=ax)
                            ax.set_xlabel(param)
                            ax.set_ylabel(metric)
                            plt.sca(ax)
                            plt.xticks(rotation=45)
                        else:
                            # Otherwise use scatter plot with trend line
                            ax.scatter(plot_data[param], plot_data[metric], alpha=0.6)
                            
                            # Add trend line
                            z = np.polyfit(plot_data[param], plot_data[metric], 1)
                            p = np.poly1d(z)
                            ax.plot(plot_data[param], p(plot_data[param]), 
                                "r--", alpha=0.8, linewidth=2)
                            
                            # Add correlation
                            corr, p_val = stats.pearsonr(plot_data[param], plot_data[metric])
                            ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}',
                                transform=ax.transAxes, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                            
                            ax.set_xlabel(param)
                            ax.set_ylabel(metric)
                        
                        ax.set_title(f'{param} vs {metric}')
                        ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save figure
                exp_suffix = f'_{experiment_name}' if experiment_name else ''
                filename = f'hyperparameter_effects_{dataset}_{model}{exp_suffix}.png'
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot: {filename}")
    def analyze_parameter_interactions(self, experiment_name=None):
        """
        Analyze interaction effects between parameters.
        
        Shows if combinations of parameters matter more than individual effects.
        """
        print("\nAnalyzing parameter interactions...")
        
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        df = self.summary_df.copy()
        
        # Filter to specific experiment if requested
        if experiment_name:
            df = df[df['exp_dir'].str.contains(f'experiment_{experiment_name}')]
        
        # Identify varying parameters
        param_columns = ['pred_loss_coef', 'info_loss_coef', 'motif_loss_coef',
                        'init_r', 'final_r', 'decay_r', 'decay_interval']
        
        varying_params = [col for col in param_columns 
                        if col in df.columns and df[col].nunique() > 1]
        
        if len(varying_params) < 2:
            print("Need at least 2 varying parameters for interaction analysis")
            return
        
        # Create interaction heatmaps
        for dataset in df['dataset'].unique():
            for model in df['model'].unique():
                subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
                
                if len(subset) < 10:  # Need reasonable sample size
                    continue
                
                # For each pair of parameters
                for i, param1 in enumerate(varying_params):
                    for param2 in varying_params[i+1:]:
                        self._plot_interaction_heatmap(
                            subset, param1, param2, 'test_acc',
                            dataset, model, experiment_name
                        )
    
    def find_best_configurations(self):
        """
        RQ 1: Find best model and explainer performance for each dataset.
        
        Returns DataFrame with best configurations per dataset.
        """
        print("\nFinding best configurations per dataset...")
        
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        best_configs = []
        
        for dataset in self.summary_df['dataset'].unique():
            dataset_data = self.summary_df[self.summary_df['dataset'] == dataset]
            
            # Best by validation accuracy/performance
            best_by_valid = dataset_data.loc[dataset_data['valid_acc'].idxmax()]
            
            # Best by test explainer performance
            best_by_explainer = dataset_data.loc[dataset_data['test_x_roc'].idxmax()]
            
            best_configs.append({
                'dataset': dataset,
                'metric': 'Model Performance (Valid Acc)',
                'best_model': best_by_valid['model'],
                'motif_loss_coef': best_by_valid['motif_loss_coef'],
                'valid_acc': best_by_valid['valid_acc'],
                'test_acc': best_by_valid['test_acc'],
                'test_x_roc': best_by_valid['test_x_roc'],
            })
            
            best_configs.append({
                'dataset': dataset,
                'metric': 'Explainer Performance (Test X-ROC)',
                'best_model': best_by_explainer['model'],
                'motif_loss_coef': best_by_explainer['motif_loss_coef'],
                'valid_acc': best_by_explainer['valid_acc'],
                'test_acc': best_by_explainer['test_acc'],
                'test_x_roc': best_by_explainer['test_x_roc'],
            })
        
        best_df = pd.DataFrame(best_configs)
        
        # Save results
        output_path = self.output_dir / 'best_configurations.csv'
        best_df.to_csv(output_path, index=False)
        print(f"Saved best configurations to {output_path}")
        
        return best_df
    
    def compare_with_without_motif_loss(self):
        """
        Compare performance with and without motif consistency loss.
        
        Returns DataFrame with statistical comparisons.
        """
        print("\nComparing with/without motif loss...")
        
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        # Separate experiments with and without motif loss
        with_motif = self.summary_df[self.summary_df['motif_loss_coef'] > 0]
        without_motif = self.summary_df[self.summary_df['motif_loss_coef'] == 0]
        
        comparison_results = []
        
        for dataset in self.summary_df['dataset'].unique():
            for model in self.summary_df['model'].unique():
                with_subset = with_motif[
                    (with_motif['dataset'] == dataset) & 
                    (with_motif['model'] == model)
                ]
                without_subset = without_motif[
                    (without_motif['dataset'] == dataset) & 
                    (without_motif['model'] == model)
                ]
                
                if len(with_subset) == 0 or len(without_subset) == 0:
                    continue
                
                # Statistical tests
                metrics = ['test_acc', 'test_x_roc', 'test_x_precision']
                
                for metric in metrics:
                    with_values = with_subset[metric].dropna()
                    without_values = without_subset[metric].dropna()
                    
                    if len(with_values) > 0 and len(without_values) > 0:
                        # T-test
                        t_stat, p_value = stats.ttest_ind(with_values, without_values)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (with_values.var() + without_values.var()) / 2
                        )
                        cohens_d = (with_values.mean() - without_values.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        result = {
                            'dataset': dataset,
                            'model': model,
                            'metric': metric,
                            'with_motif_mean': with_values.mean(),
                            'with_motif_std': with_values.std(),
                            'without_motif_mean': without_values.mean(),
                            'without_motif_std': without_values.std(),
                            'difference': with_values.mean() - without_values.mean(),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'significant': p_value < 0.05,
                        }
                        comparison_results.append(result)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save results
        output_path = self.output_dir / 'motif_loss_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"Saved motif loss comparison to {output_path}")
        
        # Create visualization
        if not comparison_df.empty:
            self._plot_motif_loss_comparison(comparison_df)
        
        return comparison_df
    
    def _plot_consistency_vs_motif_loss(self, consistency_df):
        """Plot within-motif consistency vs motif loss coefficient."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Within-Motif Score Consistency vs Motif Loss Coefficient', fontsize=16)
        
        metrics = ['avg_variance', 'avg_std', 'avg_range', 'avg_coeff_of_var']
        titles = ['Average Variance', 'Average Std Dev', 'Average Range', 'Coefficient of Variation']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            for dataset in consistency_df['dataset'].unique():
                data = consistency_df[
                    (consistency_df['dataset'] == dataset) & 
                    (consistency_df['split'] == 'test')
                ]
                
                if len(data) > 0:
                    # Aggregate by motif_loss_coef
                    agg = data.groupby('motif_loss_coef')[metric].agg(['mean', 'std']).reset_index()
                    
                    ax.errorbar(
                        agg['motif_loss_coef'], 
                        agg['mean'], 
                        yerr=agg['std'],
                        marker='o', 
                        label=dataset,
                        capsize=5
                    )
            
            ax.set_xlabel('Motif Loss Coefficient')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'consistency_vs_motif_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: consistency_vs_motif_loss.png")
    
    def _plot_explainer_performance(self, explainer_df):
        """Plot explainer performance metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Explainer Performance: Correlation between Node Scores and Motif Impact', fontsize=14)
        
        test_data = explainer_df[explainer_df['split'] == 'test']
        
        # Pearson correlation
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['pearson_corr'].agg(['mean', 'std']).reset_index()
            
            axes[0].errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        axes[0].set_xlabel('Motif Loss Coefficient')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].set_title('Pearson Correlation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Spearman correlation
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['spearman_corr'].agg(['mean', 'std']).reset_index()
            
            axes[1].errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        axes[1].set_xlabel('Motif Loss Coefficient')
        axes[1].set_ylabel('Spearman Correlation')
        axes[1].set_title('Spearman Correlation (Rank-based)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'explainer_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: explainer_performance.png")
    
    def _plot_weight_distributions(self, distribution_df):
        """Plot weight distribution characteristics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attention Weight Distribution Analysis', fontsize=16)
        
        test_data = distribution_df[distribution_df['phase'] == 'test ']
        
        # Polarization score
        ax = axes[0, 0]
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['polarization_score'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        ax.set_xlabel('Motif Loss Coefficient')
        ax.set_ylabel('Polarization Score (% near 0 + % near 1)')
        ax.set_title('Weight Polarization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[0, 1]
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['entropy'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        ax.set_xlabel('Motif Loss Coefficient')
        ax.set_ylabel('Entropy')
        ax.set_title('Weight Distribution Entropy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Percentage in middle
        ax = axes[1, 0]
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['pct_middle'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        ax.set_xlabel('Motif Loss Coefficient')
        ax.set_ylabel('% Weights in [0.4, 0.6]')
        ax.set_title('Percentage of Uncertain Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Standard deviation
        ax = axes[1, 1]
        for dataset in test_data['dataset'].unique():
            data = test_data[test_data['dataset'] == dataset]
            agg = data.groupby('motif_loss_coef')['std'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['motif_loss_coef'],
                agg['mean'],
                yerr=agg['std'],
                marker='o',
                label=dataset,
                capsize=5
            )
        
        ax.set_xlabel('Motif Loss Coefficient')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Weight Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: weight_distributions.png")
    
    def _plot_motif_loss_comparison(self, comparison_df):
        """Plot comparison of with/without motif loss."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Effect of Motif Consistency Loss on Performance', fontsize=16)
        
        metrics = ['test_acc', 'test_x_roc', 'test_x_precision']
        titles = ['Model Performance (Test Acc)', 'Explainer ROC-AUC', 'Explainer Precision']
        
        for ax, metric, title in zip(axes, metrics, titles):
            metric_data = comparison_df[comparison_df['metric'] == metric]
            
            x = np.arange(len(metric_data))
            width = 0.35
            
            labels = [f"{row['dataset']}\n{row['model']}" for _, row in metric_data.iterrows()]
            
            ax.bar(x - width/2, metric_data['without_motif_mean'], width,
                   label='Without Motif Loss', yerr=metric_data['without_motif_std'],
                   capsize=5, alpha=0.8)
            
            ax.bar(x + width/2, metric_data['with_motif_mean'], width,
                   label='With Motif Loss', yerr=metric_data['with_motif_std'],
                   capsize=5, alpha=0.8)
            
            # Mark significant differences
            for i, (_, row) in enumerate(metric_data.iterrows()):
                if row['significant']:
                    y_pos = max(row['with_motif_mean'], row['without_motif_mean']) + row['with_motif_std']
                    ax.text(i, y_pos, '*', ha='center', va='bottom', fontsize=16)
            
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'motif_loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: motif_loss_comparison.png")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GSAT TUNING EXPERIMENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overview
            f.write("1. OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Experiments: {len(self.all_experiments)}\n")
            f.write(f"Datasets: {', '.join(self.summary_df['dataset'].unique())}\n")
            f.write(f"Models: {', '.join(self.summary_df['model'].unique())}\n")
            f.write(f"Folds: {', '.join(map(str, self.summary_df['fold'].unique()))}\n")
            f.write(f"Seeds: {', '.join(map(str, self.summary_df['seed'].unique()))}\n\n")
            
            # RQ1: Best configurations
            f.write("\n2. RESEARCH QUESTION 1: Best Model and Explainer Performance\n")
            f.write("-" * 80 + "\n")
            best_configs = self.find_best_configurations()
            f.write(best_configs.to_string())
            f.write("\n\n")
            
            # RQ2: Effect of motif loss
            f.write("\n3. RESEARCH QUESTION 2: Effect of Motif Consistency Loss\n")
            f.write("-" * 80 + "\n")
            
            # (i) Consistency
            f.write("\n3.1. Within-Motif Score Consistency\n")
            consistency_df = self.analyze_within_motif_consistency()
            summary_consistency = consistency_df.groupby('motif_loss_coef')[
                ['avg_variance', 'avg_std', 'avg_coeff_of_var']
            ].mean()
            f.write(summary_consistency.to_string())
            f.write("\n\n")
            
            # (ii) Model performance
            f.write("\n3.2. Model Prediction Performance\n")
            comparison_df = self.compare_with_without_motif_loss()
            if not comparison_df.empty:
                model_perf = comparison_df[comparison_df['metric'] == 'test_acc']
                f.write(model_perf.to_string())
                f.write("\n\n")
            
            # (iii) Explainer performance
            f.write("\n3.3. Explainer Performance (Pearson Correlation)\n")
            explainer_df = self.analyze_explainer_performance()
            summary_explainer = explainer_df.groupby('motif_loss_coef')[
                ['pearson_corr', 'spearman_corr']
            ].mean()
            f.write(summary_explainer.to_string())
            f.write("\n\n")
            
            # (iv) Weight distribution
            f.write("\n3.4. Weight Distribution Analysis\n")
            distribution_df = self.analyze_weight_distribution()
            summary_dist = distribution_df.groupby('motif_loss_coef')[
                ['mean', 'entropy', 'polarization_score', 'pct_middle']
            ].mean()
            f.write(summary_dist.to_string())
            f.write("\n\n")
            
            # Recommendations
            f.write("\n4. RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("Based on the analysis:\n\n")
            
            # Find optimal motif_loss_coef for each metric
            best_consistency = consistency_df.groupby('motif_loss_coef')['avg_variance'].mean().idxmin()
            best_model_perf = comparison_df[comparison_df['metric'] == 'test_acc'].groupby('motif_loss_coef')['with_motif_mean'].mean().idxmax() if len(comparison_df) > 0 else None
            best_explainer = explainer_df.groupby('motif_loss_coef')['pearson_corr'].mean().idxmax()
            
            f.write(f"- For best within-motif consistency: motif_loss_coef = {best_consistency}\n")
            if best_model_perf:
                f.write(f"- For best model performance: motif_loss_coef = {best_model_perf}\n")
            f.write(f"- For best explainer performance: motif_loss_coef = {best_explainer}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\nSaved comprehensive report to {report_path}")
    
    def run_full_analysis(self):
        """Run all analysis steps."""
        print("\n" + "="*80)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("="*80)
        
        # Step 1: Collect experiments
        self.collect_all_experiments()
        
        # Step 2: Create summary
        self.create_summary_dataframe()
        
        # Step 3: Run all analyses
        self.find_best_configurations()
        self.analyze_within_motif_consistency()
        self.analyze_explainer_performance()
        self.analyze_weight_distribution()
        self.compare_with_without_motif_loss()

        # Add after other analyses:
        print("\n[Running hyperparameter effects analysis...]")
        self.analyze_hyperparameter_effects()
        
        print("\n[Running parameter interaction analysis...]")
        self.analyze_parameter_interactions()
        
        # Step 4: Generate report
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob('*')):
            print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tuning results from GSAT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results_dir', type=str, default='tuning_results',
                        help='Directory containing tuning results')
    
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis outputs')
    
    parser.add_argument('--datasets', nargs='+',
                        help='Filter by specific datasets (optional)')
    
    parser.add_argument('--models', nargs='+',
                        help='Filter by specific models (optional)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TuningResultsAnalyzer(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir)
    )
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()

