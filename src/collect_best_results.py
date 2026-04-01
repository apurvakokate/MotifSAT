#!/usr/bin/env python
"""
Pick best (experiment × model) runs balancing valid ROC and score–impact correlations,
render score-vs-impact plots into best_results (via analyze_motif_consistency.plot_score_vs_impact),
and write summary CSVs under ../best_results/.

Selection uses valid split: composite = w_pred * roc_valid + w_m * r_motif + w_n * r_node
(nan correlations treated as 0). Default w_pred=0.6, w_m=0.2, w_n=0.2.

Table cells: after picking the winning hyperparam row (same composite criterion), values are
mean ± std (sample std, ddof=1) across folds — per fold, the best seed for that row by the same
composite. Single-fold rows show the mean only.

Requires: tuning_results with final_metrics.json and masking/attention jsonl under each seed_dir
(same artifacts training already writes for post-hoc analysis; no separate motif_consistency pass).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_mutagenicity_tables import (
    compute_node_score_impact_correlation,
    compute_posthoc_correlation,
    find_results,
)
from analyze_motif_consistency import plot_score_vs_impact


MODEL_ORDER = ['GAT', 'GCN', 'GIN', 'PNA', 'SAGE']

DEFAULT_EXPERIMENT_LABELS = {
    'vanilla_gnn': 'Vanilla',
    'base_gsat_fix_r': 'Base GSAT fix r',
    'base_gsat_decay_r': 'Base GSAT decay r',
    'base_gsat_decay_r_minority_global': 'Base GSAT decay r (minority_global)',
    'base_gsat_decay_r_injection': 'Decay r injection ablation',
    'base_gsat_motif_loss': 'Base GSAT motif loss',
    'motif_readout_decay_w_message': 'Motif readout decay w_message',
    'motif_readout_decay_injection_ablation': 'Motif readout injection ablation',
    'base_gsat_readout_intra_att': 'Base GSAT intra-motif att readout',
    'motif_readout_prior_node_gate': 'Motif readout prior node gate',
    'motif_readout_prior_node_gate_tanh_sched': 'Motif prior gate (tanh + shift schedule)',
    'motif_readout_weight_diversity': 'Motif weight diversity loss',
}


def experiment_label(key: str) -> str:
    return DEFAULT_EXPERIMENT_LABELS.get(key, key.replace('_', ' ').title())


def _render_score_impact_plots_for_split(
    seed_dir: Path,
    dataset: str,
    model: str,
    split: str,
    out_dir: Path,
    verbose: bool,
) -> None:
    """Write score_vs_impact*.png under out_dir from jsonl in seed_dir (training exports)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_score_vs_impact(
            seed_dir=seed_dir,
            split=split,
            output_dir=out_dir,
            model_name=model,
            dataset_name=dataset,
        )
    except Exception as e:
        print(f'[WARNING] plot_score_vs_impact failed {model} {split}: {e}')
    if verbose and not any(out_dir.glob('score_vs_impact*.png')):
        print(
            f'[INFO] No score_vs_impact PNGs for {model} split={split} '
            f'(check jsonl / split in {seed_dir})'
        )


def _format_float(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ''
    try:
        return f'{float(x):.4f}'
    except (TypeError, ValueError):
        return ''


def _format_mean_std(values: list[float]) -> str:
    """Mean ± sample std; one value → mean only."""
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return ''
    mean = float(np.mean(arr))
    if arr.size < 2:
        return f'{mean:.4f}'
    std = float(np.std(arr, ddof=1))
    return f'{mean:.4f} ± {std:.4f}'


def _composite(
    roc_valid: float,
    r_motif: float,
    r_node: float,
    w_pred: float,
    w_m: float,
    w_n: float,
) -> float:
    rm = float(r_motif) if np.isfinite(r_motif) else 0.0
    rn = float(r_node) if np.isfinite(r_node) else 0.0
    rv = float(roc_valid) if np.isfinite(roc_valid) else -1.0
    if rv < 0:
        return -1e9
    return w_pred * rv + w_m * rm + w_n * rn


def _best_run_for_model(
    records: list,
    model: str,
    w_pred: float,
    w_m: float,
    w_n: float,
) -> tuple[dict | None, float]:
    """Choose seed_dir maximizing composite on *valid* correlations."""
    best = None
    best_c = -1e18
    for r in records:
        if r['model'] != model:
            continue
        m = r['metrics']
        rv = m.get('metric/best_clf_roc_valid', np.nan)
        sd = r['seed_dir']
        rm, _, _ = compute_posthoc_correlation(sd, 'valid')
        rn, _, _ = compute_node_score_impact_correlation(sd, 'valid')
        c = _composite(rv, rm, rn, w_pred, w_m, w_n)
        if c > best_c:
            best_c = c
            best = r
    return best, best_c


def _per_fold_best_for_row(
    records: list,
    model: str,
    chosen_row: str,
    w_pred: float,
    w_m: float,
    w_n: float,
) -> list[dict]:
    """One record per fold: best composite among runs matching model and hyperparam row."""
    filtered = [r for r in records if r['model'] == model and r['row'] == chosen_row]
    by_fold: dict = {}
    for r in filtered:
        f = r['fold']
        by_fold.setdefault(f, []).append(r)
    out = []
    for fold in sorted(by_fold.keys()):
        best = None
        best_c = -1e18
        for r in by_fold[fold]:
            m = r['metrics']
            rv = m.get('metric/best_clf_roc_valid', np.nan)
            sd = r['seed_dir']
            rm, _, _ = compute_posthoc_correlation(sd, 'valid')
            rn, _, _ = compute_node_score_impact_correlation(sd, 'valid')
            c = _composite(rv, rm, rn, w_pred, w_m, w_n)
            if c > best_c:
                best_c = c
                best = r
        if best is not None:
            out.append(best)
    return out


def run(
    results_dir: Path,
    dataset: str,
    experiments: list[str],
    best_results_dir: Path,
    w_pred: float,
    w_m: float,
    w_n: float,
    verbose: bool = False,
):
    best_results_dir = best_results_dir.resolve()
    (best_results_dir / dataset).mkdir(parents=True, exist_ok=True)
    meta = {
        'dataset': dataset,
        'w_pred': w_pred,
        'w_motif_corr': w_m,
        'w_node_corr': w_n,
        'selection_split': 'valid',
        'note': (
            'Best hyperparam row per (experiment, model) via global max composite (valid ROC + correlations). '
            'Table numbers: mean ± std across folds; per fold, best seed for that row (same composite).'
        ),
    }
    with open(best_results_dir / dataset / 'selection_weights.json', 'w') as f:
        json.dump(meta, f, indent=2)
    train_dir = best_results_dir / 'train'
    valid_dir = best_results_dir / 'validation'
    test_dir = best_results_dir / 'test'
    for d in (train_dir, valid_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    perf_train_rows = []
    perf_valid_rows = []
    perf_test_rows = []
    exp_train_rows = []
    exp_valid_rows = []
    exp_test_rows = []
    hyperparam_rows = []

    for exp_key in experiments:
        label = experiment_label(exp_key)
        records = find_results(results_dir, exp_key, verbose=verbose, dataset=dataset)
        if not records:
            if verbose:
                print(f'[SKIP] No records for {exp_key}')
            continue

        row_labels = {str(r.get('row', '')) for r in records}
        has_scan = len(row_labels) > 1

        exp_plot_dir = best_results_dir / dataset / exp_key
        exp_plot_dir.mkdir(parents=True, exist_ok=True)

        perf_tr = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}
        perf_va = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}
        perf_te = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}
        ex_tr = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}
        ex_va = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}
        ex_te = {'Dataset': dataset, 'Experiment': label, 'Hyperparam_scan': 'yes' if has_scan else ''}

        for model in MODEL_ORDER:
            br, comp = _best_run_for_model(records, model, w_pred, w_m, w_n)
            if br is None:
                perf_tr[model] = ''
                perf_va[model] = ''
                perf_te[model] = ''
                ex_tr[f'{model}_motif_r'] = ''
                ex_tr[f'{model}_node_r'] = ''
                ex_va[f'{model}_motif_r'] = ''
                ex_va[f'{model}_node_r'] = ''
                ex_te[f'{model}_motif_r'] = ''
                ex_te[f'{model}_node_r'] = ''
                continue

            chosen_row = br.get('row', '')
            per_fold = _per_fold_best_for_row(records, model, chosen_row, w_pred, w_m, w_n)

            roc_tr = []
            roc_va = []
            roc_te = []
            motif_tr, motif_va, motif_te = [], [], []
            node_tr, node_va, node_te = [], [], []
            for pr in per_fold:
                pm = pr['metrics']
                psd: Path = pr['seed_dir']
                v = pm.get('metric/best_clf_roc_train')
                if v is not None and np.isfinite(v):
                    roc_tr.append(float(v))
                v = pm.get('metric/best_clf_roc_valid')
                if v is not None and np.isfinite(v):
                    roc_va.append(float(v))
                v = pm.get('metric/best_clf_roc_test')
                if v is not None and np.isfinite(v):
                    roc_te.append(float(v))
                rmt, _, _ = compute_posthoc_correlation(psd, 'train')
                rmv, _, _ = compute_posthoc_correlation(psd, 'valid')
                rme, _, _ = compute_posthoc_correlation(psd, 'test')
                if np.isfinite(rmt):
                    motif_tr.append(float(rmt))
                if np.isfinite(rmv):
                    motif_va.append(float(rmv))
                if np.isfinite(rme):
                    motif_te.append(float(rme))
                rnt, _, _ = compute_node_score_impact_correlation(psd, 'train')
                rnv, _, _ = compute_node_score_impact_correlation(psd, 'valid')
                rne, _, _ = compute_node_score_impact_correlation(psd, 'test')
                if np.isfinite(rnt):
                    node_tr.append(float(rnt))
                if np.isfinite(rnv):
                    node_va.append(float(rnv))
                if np.isfinite(rne):
                    node_te.append(float(rne))

            perf_tr[model] = _format_mean_std(roc_tr)
            perf_va[model] = _format_mean_std(roc_va)
            perf_te[model] = _format_mean_std(roc_te)
            ex_tr[f'{model}_motif_r'] = _format_mean_std(motif_tr)
            ex_tr[f'{model}_node_r'] = _format_mean_std(node_tr)
            ex_va[f'{model}_motif_r'] = _format_mean_std(motif_va)
            ex_va[f'{model}_node_r'] = _format_mean_std(node_va)
            ex_te[f'{model}_motif_r'] = _format_mean_std(motif_te)
            ex_te[f'{model}_node_r'] = _format_mean_std(node_te)

            sd: Path = br['seed_dir']
            hyperparam_rows.append({
                'dataset': dataset,
                'experiment': exp_key,
                'experiment_label': label,
                'model': model,
                'best_row': br.get('row', ''),
                'best_variant': br.get('variant', ''),
                'best_fold': br.get('fold'),
                'best_seed': br.get('seed'),
                'composite_valid': round(comp, 6),
                'seed_dir': str(sd),
                'n_folds_aggregated': len(per_fold),
                'folds_aggregated': ','.join(str(pr['fold']) for pr in sorted(per_fold, key=lambda x: x['fold'])),
            })

            dst_m = exp_plot_dir / f'model_{model}'
            for spl in ('test', 'valid'):
                out_sub = dst_m / f'{spl}_plots' / spl
                _render_score_impact_plots_for_split(sd, dataset, model, spl, out_sub, verbose)

        perf_train_rows.append(perf_tr)
        perf_valid_rows.append(perf_va)
        perf_test_rows.append(perf_te)
        exp_train_rows.append(ex_tr)
        exp_valid_rows.append(ex_va)
        exp_test_rows.append(ex_te)

    def _save_perf(rows, path: Path):
        if not rows:
            return
        df = pd.DataFrame(rows)
        cols = ['Dataset', 'Experiment', 'Hyperparam_scan'] + MODEL_ORDER
        df = df.reindex(columns=[c for c in cols if c in df.columns])
        df.to_csv(path, index=False)
        print(f'[INFO] Wrote {path}')

    def _save_expl(rows, path: Path):
        if not rows:
            return
        df = pd.DataFrame(rows)
        expl_cols = []
        for m in MODEL_ORDER:
            expl_cols.extend([f'{m}_motif_r', f'{m}_node_r'])
        cols = ['Dataset', 'Experiment', 'Hyperparam_scan'] + expl_cols
        df = df.reindex(columns=[c for c in cols if c in df.columns])
        df.to_csv(path, index=False)
        print(f'[INFO] Wrote {path}')

    _save_perf(perf_train_rows, train_dir / 'model_prediction_performance.csv')
    _save_perf(perf_valid_rows, valid_dir / 'model_prediction_performance.csv')
    _save_perf(perf_test_rows, test_dir / 'model_prediction_performance.csv')
    _save_expl(exp_train_rows, train_dir / 'explainer_score_impact_correlation.csv')
    _save_expl(exp_valid_rows, valid_dir / 'explainer_score_impact_correlation.csv')
    _save_expl(exp_test_rows, test_dir / 'explainer_score_impact_correlation.csv')

    if hyperparam_rows:
        hdf = pd.DataFrame(hyperparam_rows)
        hpath = best_results_dir / dataset / 'best_config_per_model.csv'
        hpath.parent.mkdir(parents=True, exist_ok=True)
        hdf.to_csv(hpath, index=False)
        print(f'[INFO] Wrote {hpath}')


def main():
    p = argparse.ArgumentParser(description='Collect best balanced runs and summary tables')
    p.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Mutagenicity'))
    p.add_argument('--results_dir', type=str, default=os.environ.get('RESULTS_DIR', '../tuning_results'))
    p.add_argument(
        '--best_results_dir',
        type=str,
        default=os.environ.get('BEST_RESULTS_DIR', '../best_results'),
    )
    p.add_argument('--experiments', nargs='*', default=None, help='Defaults to analysis.sh streamlined list')
    p.add_argument('--w_pred', type=float, default=0.6)
    p.add_argument('--w_motif_corr', type=float, default=0.2)
    p.add_argument('--w_node_corr', type=float, default=0.2)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.experiments:
        experiments = list(args.experiments)
    else:
        experiments = [
            'vanilla_gnn',
            'base_gsat_fix_r',
            'base_gsat_decay_r',
            'base_gsat_decay_r_minority_global',
            'base_gsat_decay_r_injection',
            'base_gsat_motif_loss',
            'motif_readout_decay_w_message',
            'motif_readout_decay_injection_ablation',
            'base_gsat_readout_intra_att',
            'motif_readout_prior_node_gate',
            'motif_readout_prior_node_gate_tanh_sched',
            'motif_readout_weight_diversity',
        ]

    run(
        results_dir=Path(args.results_dir),
        dataset=args.dataset,
        experiments=experiments,
        best_results_dir=Path(args.best_results_dir),
        w_pred=args.w_pred,
        w_m=args.w_motif_corr,
        w_n=args.w_node_corr,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
