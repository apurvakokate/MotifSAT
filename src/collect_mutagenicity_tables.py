#!/usr/bin/env python
"""
Collect Mutagenicity GSAT results and generate summary tables.

Tables: columns = architectures, rows = the varying hyperparameter for that experiment.

Experiment-specific rows:
  r_impact_node / r_impact_edge       → rows = final_r  (0.4, 0.5, 0.6)
  within_motif_consistency_impact      → rows = motif_loss_coef  (1.0, 2.0)
  between_motif_consistency_impact     → rows = between_motif_coef (1.0, 2.0)

Reads from saved final_metrics.json and experiment_summary.json (no retraining).

Usage:
  python collect_mutagenicity_tables.py --experiment_name r_impact_node
  python collect_mutagenicity_tables.py --experiment_name within_motif_consistency_impact --verbose
"""

import argparse
import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Map experiment_name → (key in experiment_summary to read, display name)
EXPERIMENT_ROW_CONFIG = {
    'r_impact_node': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'r',
    },
    'r_impact_edge': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'r',
    },
    'within_motif_consistency_impact': {
        'summary_path': ('loss_coefficients', 'motif_loss_coef'),
        'row_label_prefix': 'motif_loss_coef',
    },
    'between_motif_consistency_impact': {
        'summary_path': ('loss_coefficients', 'between_motif_coef'),
        'row_label_prefix': 'between_motif_coef',
    },
}


def _get_row_value(summary: dict, experiment_name: str):
    """Extract the varying hyperparameter value from experiment_summary.json."""
    cfg = EXPERIMENT_ROW_CONFIG.get(experiment_name)
    if cfg is None:
        return 'unknown'
    section, key = cfg['summary_path']
    val = summary.get(section, {}).get(key, '?')
    return val


def _make_row_label(val, experiment_name: str):
    cfg = EXPERIMENT_ROW_CONFIG.get(experiment_name)
    prefix = cfg['row_label_prefix'] if cfg else 'param'
    return f'{prefix}={val}'


def _recover_truncated_json(raw: str):
    """Try to recover a truncated JSON object by finding the last valid closing brace."""
    raw = raw.strip()
    if not raw.startswith('{'):
        return None
    for i in range(len(raw) - 1, 0, -1):
        candidate = raw[:i]
        last_comma = candidate.rfind(',')
        if last_comma > 0:
            candidate = candidate[:last_comma]
        candidate = candidate.rstrip() + '\n}'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def find_results(results_dir: Path, experiment_name: str, verbose: bool = False):
    """Find all final_metrics.json under results_dir/Mutagenicity for the given experiment_name."""
    base = results_dir / 'Mutagenicity'
    if not base.exists():
        print(f'[WARN] Directory does not exist: {base}')
        return []
    experiment_dir = f'experiment_{experiment_name}'
    records = []
    all_final_metrics = list(base.rglob('final_metrics.json'))
    if verbose:
        print(f'[DEBUG] Found {len(all_final_metrics)} total final_metrics.json under {base}')
    skipped_experiment = 0
    skipped_parse = 0
    skipped_error = 0
    for fm_path in all_final_metrics:
        try:
            parts = fm_path.parent.relative_to(base).parts
            if experiment_dir not in parts:
                skipped_experiment += 1
                continue
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
                skipped_parse += 1
                if verbose:
                    print(f'  [SKIP parse] model={model_name} tuning={tuning_id} fold={fold} seed={seed} path={fm_path.parent.relative_to(base)}')
                continue
            with open(fm_path) as f:
                raw = f.read()
            try:
                metrics = json.loads(raw)
            except json.JSONDecodeError:
                metrics = _recover_truncated_json(raw)
                if metrics is None:
                    skipped_error += 1
                    if verbose:
                        print(f'  [ERROR] Unrecoverable JSON: {fm_path}')
                    continue
                if verbose:
                    print(f'  [RECOVERED] Partial JSON ({len(metrics)} keys): {fm_path}')
            # Read experiment_summary.json for the row value
            seed_dir = fm_path.parent
            summary_path = seed_dir / 'experiment_summary.json'
            summary = {}
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
            row_val = _get_row_value(summary, experiment_name)
            row_label = _make_row_label(row_val, experiment_name)
            records.append({
                'model': model_name,
                'variant': tuning_id,
                'row_val': row_val,
                'row': row_label,
                'fold': fold,
                'seed': seed,
                'metrics': metrics,
            })
        except Exception as e:
            skipped_error += 1
            if verbose:
                print(f'  [ERROR] {fm_path}: {e}')
            continue
    if verbose or skipped_error > 0:
        print(f'[DEBUG] Total: {len(all_final_metrics)}, '
              f'skipped {skipped_experiment} (wrong experiment), {skipped_parse} (parse), '
              f'{skipped_error} (error), collected {len(records)}')
    return records


def build_table(records, metric_key):
    """Build a pivot table: rows = hyperparameter values, columns = architectures."""
    if not records:
        return None, None
    df = pd.DataFrame(records)
    df['value'] = df['metrics'].apply(lambda m: m.get(metric_key, np.nan))
    agg = df.groupby(['row', 'row_val', 'model'])['value'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg.sort_values('row_val')
    pivot_mean = agg.pivot(index='row', columns='model', values='mean')
    pivot_std = agg.pivot(index='row', columns='model', values='std')
    # Sort rows by the numeric value
    row_order = agg.drop_duplicates('row').sort_values('row_val')['row'].tolist()
    pivot_mean = pivot_mean.reindex(row_order).dropna(how='all')
    pivot_std = pivot_std.reindex(row_order).dropna(how='all')
    return pivot_mean, pivot_std


def main():
    parser = argparse.ArgumentParser(description='Collect Mutagenicity experiment results into tables')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name (r_impact_node, r_impact_edge, within_motif_consistency_impact, between_motif_consistency_impact)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results dir (default: RESULTS_DIR env or ../tuning_results)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write CSV (default: results_dir)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir or os.environ.get('RESULTS_DIR', '../tuning_results'))
    output_dir = Path(args.output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = find_results(results_dir, args.experiment_name, verbose=args.verbose)
    if not records:
        print(f'No results found for experiment "{args.experiment_name}" under {results_dir / "Mutagenicity"}.')
        return

    print(f'Experiment: {args.experiment_name}')
    print(f'Found {len(records)} runs.')

    prefix = args.experiment_name

    # Table 1: Prediction performance (valid ROC)
    pred_mean, pred_std = build_table(records, metric_key='metric/best_clf_roc_valid')
    if pred_mean is not None:
        path = output_dir / f'{prefix}_prediction_valid_roc.csv'
        pred_mean.to_csv(path)
        print(f'\n--- Prediction performance (valid ROC, mean over folds/seeds) ---')
        print(pred_mean.to_string())
        print(f'\nSaved: {path}')
        path_std = output_dir / f'{prefix}_prediction_valid_roc_std.csv'
        pred_std.to_csv(path_std)

    # Table 2: Explainer performance (motif att–impact correlation)
    exp_mean, exp_std = build_table(records, metric_key='motif/att_impact_correlation')
    if exp_mean is not None and not exp_mean.isna().all().all():
        path = output_dir / f'{prefix}_explainer_correlation.csv'
        exp_mean.to_csv(path)
        print(f'\n--- Explainer (motif att–impact correlation, mean over folds/seeds) ---')
        print(exp_mean.to_string())
        print(f'\nSaved: {path}')
        path_std = output_dir / f'{prefix}_explainer_correlation_std.csv'
        exp_std.to_csv(path_std)
    else:
        print('\nNo motif/att_impact_correlation data found.')

    # Table 3: Motif edge attention range (max - min per motif)
    range_mean, range_std = build_table(records, metric_key='motif_edge_att/max_mean')
    if range_mean is not None and not range_mean.isna().all().all():
        path = output_dir / f'{prefix}_motif_edge_att_max.csv'
        range_mean.to_csv(path)
        print(f'\n--- Motif edge att max (mean over folds/seeds) ---')
        print(range_mean.to_string())
        print(f'\nSaved: {path}')

    min_mean, _ = build_table(records, metric_key='motif_edge_att/min_mean')
    if min_mean is not None and not min_mean.isna().all().all():
        path = output_dir / f'{prefix}_motif_edge_att_min.csv'
        min_mean.to_csv(path)
        print(f'\n--- Motif edge att min (mean over folds/seeds) ---')
        print(min_mean.to_string())
        print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
