#!/usr/bin/env python
"""
Collect Mutagenicity GSAT results and generate two summary tables:
  1. Model prediction performance (test ROC) by architecture, averaged over folds (and seeds).
  2. Explainer performance (motif attention–impact correlation) by architecture, averaged over folds (and seeds).

No retraining: reads from saved final_metrics.json under RESULTS_DIR/Mutagenicity/...
Run after run_mutagenicity_gsat_experiment.py has completed.

Usage:
  python collect_mutagenicity_tables.py --experiment_name mutagenicity_gsat_experiment [--results_dir ../tuning_results] [--output_dir .]
"""

import argparse
import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np


def find_mutagenicity_results(results_dir: Path, experiment_name: str):
    """Find all final_metrics.json under results_dir/Mutagenicity for the given experiment_name."""
    base = results_dir / 'Mutagenicity'
    if not base.exists():
        return []
    experiment_dir = f'experiment_{experiment_name}'
    records = []
    # Path pattern: .../Mutagenicity/model_{model}/experiment_{name}/tuning_{tuning_id}/.../fold{f}_seed{s}/final_metrics.json
    for fm_path in base.rglob('final_metrics.json'):
        try:
            parts = fm_path.parent.relative_to(base).parts
            if experiment_dir not in parts:
                continue
            # parts like ('model_GIN', 'experiment_mutagenicity_gsat_experiment', 'tuning_node_baseline', ...)
            model_name = None
            tuning_id = None
            fold = None
            seed = None
            for p in parts:
                if p.startswith('model_'):
                    model_name = p.replace('model_', '')
                elif p.startswith('tuning_'):
                    tuning_id = p.replace('tuning_', '')
                elif p.startswith('fold') and '_seed' in p:
                    m = re.match(r'fold(\d+)_seed(\d+)', p)
                    if m:
                        fold = int(m.group(1))
                        seed = int(m.group(2))
            if model_name is None or tuning_id is None or fold is None or seed is None:
                continue
            with open(fm_path) as f:
                metrics = json.load(f)
            records.append({
                'model': model_name,
                'variant': tuning_id,
                'fold': fold,
                'seed': seed,
                'metrics': metrics,
            })
        except Exception as e:
            continue
    return records


def build_prediction_table(records, metric_key='metric/best_clf_roc_test'):
    """Table: rows = variants, columns = architectures, values = mean (over folds and seeds)."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df['value'] = df['metrics'].apply(lambda m: m.get(metric_key, np.nan))
    agg = df.groupby(['variant', 'model'])['value'].agg(['mean', 'std', 'count']).reset_index()
    pivot_mean = agg.pivot(index='variant', columns='model', values='mean')
    pivot_std = agg.pivot(index='variant', columns='model', values='std')
    return pivot_mean, pivot_std


def build_explainer_table(records, metric_key='motif/att_impact_correlation'):
    """Table: rows = variants, columns = architectures, values = mean correlation (over folds and seeds)."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df['value'] = df['metrics'].apply(lambda m: m.get(metric_key, np.nan))
    agg = df.groupby(['variant', 'model'])['value'].agg(['mean', 'std', 'count']).reset_index()
    pivot_mean = agg.pivot(index='variant', columns='model', values='mean')
    pivot_std = agg.pivot(index='variant', columns='model', values='std')
    return pivot_mean, pivot_std


def main():
    parser = argparse.ArgumentParser(description='Collect Mutagenicity results into prediction and explainer tables')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name (e.g. mutagenicity_gsat_experiment). Only results under experiment_{name}/ are collected.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results dir (default: RESULTS_DIR env or ../tuning_results)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write CSV and summary (default: results_dir)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir or os.environ.get('RESULTS_DIR', '../tuning_results'))
    output_dir = Path(args.output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = find_mutagenicity_results(results_dir, args.experiment_name)
    if not records:
        print(f'No final_metrics.json found under {results_dir / "Mutagenicity"} for experiment "{args.experiment_name}".')
        print(f'Looked for paths containing: experiment_{args.experiment_name}')
        return

    print(f'Experiment: {args.experiment_name}')
    print(f'Found {len(records)} runs (fold/seed/model/variant).')

    # Table 1: Model prediction performance (test ROC), averaged by fold (and seed)
    pred_mean, pred_std = build_prediction_table(records, metric_key='metric/best_clf_roc_test')
    if pred_mean is not None:
        pred_path = output_dir / 'mutagenicity_prediction_performance.csv'
        pred_mean.to_csv(pred_path)
        print(f'\n--- Model prediction performance (test ROC, mean over folds/seeds) ---')
        print(pred_mean.to_string())
        print(f'\nSaved: {pred_path}')
        # Optional: mean ± std
        pred_path_std = output_dir / 'mutagenicity_prediction_performance_std.csv'
        pred_std.to_csv(pred_path_std)
        print(f'Saved (std): {pred_path_std}')

    # Table 2: Explainer performance (correlation between attention weights and motif impact)
    exp_mean, exp_std = build_explainer_table(records, metric_key='motif/att_impact_correlation')
    if exp_mean is not None:
        exp_path = output_dir / 'mutagenicity_explainer_correlation.csv'
        exp_mean.to_csv(exp_path)
        print(f'\n--- Explainer performance (motif att–impact correlation, mean over folds/seeds) ---')
        print(exp_mean.to_string())
        print(f'\nSaved: {exp_path}')
        exp_path_std = output_dir / 'mutagenicity_explainer_correlation_std.csv'
        exp_std.to_csv(exp_path_std)
        print(f'Saved (std): {exp_path_std}')
    else:
        print('\nNo motif/att_impact_correlation in any run (ensure explainer metrics were saved; run with latest run_gsat.py).')


if __name__ == '__main__':
    main()
