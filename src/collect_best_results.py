#!/usr/bin/env python
"""
Pick best (experiment × model) runs, render score-vs-impact plots, write best_results CSVs.

Selection (valid split): --selection-by composite | motif_corr_valid (see analysis.sh SELECTION_BY).
  composite: w_pred*roc + w_m*r_motif + w_n*r_node on valid
  motif_corr_valid: max motif-level explainer r only (tie-break valid ROC)

Table cells: mean ± std across folds; per-fold seed tie-break matches anchor rule.

Optional SET_R: filter to that fix_r/final_r; with composite uses max valid ROC anchor; with
motif_corr_valid uses max motif-r anchor.

Requires: final_metrics.json + post-hoc jsonl under each seed_dir.
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
    EXPERIMENT_ROW_CONFIG,
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


def _parse_set_r_env() -> float | None:
    raw = os.environ.get('SET_R', '').strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _weight_distribution_params(record: dict) -> dict:
    w = record.get('metrics', {}).get('weight_distribution_params')
    if isinstance(w, dict) and any(v is not None for v in w.values()):
        return w
    path = record['seed_dir'] / 'experiment_summary.json'
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            w = data.get('weight_distribution_params')
            if isinstance(w, dict):
                return w
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _record_matches_set_r(record: dict, exp_key: str, set_r: float) -> bool:
    cfg = EXPERIMENT_ROW_CONFIG.get(exp_key) or {}
    px = cfg.get('path_extract')
    wdp = _weight_distribution_params(record)
    fix_r, final_r = wdp.get('fix_r'), wdp.get('final_r')
    try:
        fix_r = float(fix_r) if fix_r is not None else None
    except (TypeError, ValueError):
        fix_r = None
    try:
        final_r = float(final_r) if final_r is not None else None
    except (TypeError, ValueError):
        final_r = None
    tol = 1e-5
    if px == 'fix_r':
        return fix_r is not None and abs(fix_r - set_r) < tol
    if px == 'final_r':
        return final_r is not None and abs(final_r - set_r) < tol
    if px == 'vanilla':
        return True
    if px in ('injection_code', 'readout_sampling_mode', 'prior_gate_shift'):
        return final_r is not None and abs(final_r - set_r) < tol
    return True


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


def _best_run_max_valid_roc(records: list, model: str) -> tuple[dict | None, float]:
    """Anchor run: max valid ROC only (no correlation terms)."""
    best = None
    best_rv = -1.0
    for r in records:
        if r['model'] != model:
            continue
        rv = r['metrics'].get('metric/best_clf_roc_valid', np.nan)
        if not np.isfinite(rv):
            continue
        rv = float(rv)
        if rv > best_rv:
            best_rv = rv
            best = r
    if best is None:
        return None, -1.0
    return best, best_rv


def _best_run_motif_corr_valid(records: list, model: str) -> tuple[dict | None, float]:
    """Anchor run: max motif-level score–impact Pearson r on valid; tie-break max valid ROC."""
    best = None
    best_key = (-1e18, -1e18)
    for r in records:
        if r['model'] != model:
            continue
        sd = r['seed_dir']
        rm, _, _ = compute_posthoc_correlation(sd, 'valid')
        rv = r['metrics'].get('metric/best_clf_roc_valid', np.nan)
        rmc = float(rm) if np.isfinite(rm) else -1e9
        rvc = float(rv) if np.isfinite(rv) else -1e9
        key = (rmc, rvc)
        if key > best_key:
            best_key = key
            best = r
    if best is None:
        return None, float('nan')
    return best, float(best_key[0])


def _per_fold_best_for_row(
    records: list,
    model: str,
    chosen_row: str,
    w_pred: float,
    w_m: float,
    w_n: float,
    tie_break: str = 'composite',
) -> list[dict]:
    """One record per fold among runs matching model and hyperparam row."""
    filtered = [r for r in records if r['model'] == model and r['row'] == chosen_row]
    by_fold: dict = {}
    for r in filtered:
        f = r['fold']
        by_fold.setdefault(f, []).append(r)
    out = []
    for fold in sorted(by_fold.keys()):
        best = None
        best_c = -1e18
        best_key = (-1e18, -1e18)
        for r in by_fold[fold]:
            m = r['metrics']
            rv = m.get('metric/best_clf_roc_valid', np.nan)
            sd = r['seed_dir']
            if tie_break == 'valid_roc':
                c = float(rv) if np.isfinite(rv) else -1e9
                if c > best_c:
                    best_c = c
                    best = r
            elif tie_break == 'motif_corr':
                rm, _, _ = compute_posthoc_correlation(sd, 'valid')
                rmc = float(rm) if np.isfinite(rm) else -1e9
                rvc = float(rv) if np.isfinite(rv) else -1e9
                key = (rmc, rvc)
                if key > best_key:
                    best_key = key
                    best = r
            else:
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
    set_r: float | None = None,
    selection_by: str = 'composite',
):
    best_results_dir = best_results_dir.resolve()
    (best_results_dir / dataset).mkdir(parents=True, exist_ok=True)
    fixed_r_mode = set_r is not None
    use_motif_sel = selection_by == 'motif_corr_valid'
    if selection_by not in ('composite', 'motif_corr_valid'):
        raise ValueError(f"selection_by must be 'composite' or 'motif_corr_valid', got {selection_by!r}")

    if use_motif_sel:
        sel_label = 'motif_corr_valid_tie_roc'
    elif fixed_r_mode:
        sel_label = 'fixed_r_max_valid_roc'
    else:
        sel_label = 'composite_valid_correlations'

    meta = {
        'dataset': dataset,
        'set_r': set_r,
        'selection_by': selection_by,
        'selection_mode': sel_label,
        'w_pred': w_pred,
        'w_motif_corr': w_m,
        'w_node_corr': w_n,
        'selection_split': 'valid',
        'note': (
            (
                (
                    f'SET_R={set_r}: filter runs; anchor + per-fold = max motif explainer r on valid (tie-break ROC). '
                    if fixed_r_mode
                    else 'Anchor + per-fold = max motif-level explainer r on valid (tie-break valid ROC). '
                )
                if use_motif_sel
                else (
                    f'SET_R={set_r}: filter runs; anchor + per-fold = max valid ROC. '
                    if fixed_r_mode
                    else 'Anchor + per-fold = max composite (valid ROC + motif + node corrs on valid). '
                )
            )
            + 'Table numbers: mean ± std across folds.'
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

        if fixed_r_mode:
            before = len(records)
            records = [r for r in records if _record_matches_set_r(r, exp_key, set_r)]
            if verbose:
                print(f'[INFO] {exp_key}: SET_R={set_r} kept {len(records)}/{before} runs')
            if not records:
                if verbose:
                    print(f'[SKIP] No records left after SET_R filter for {exp_key}')
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

        if use_motif_sel:
            tie_break = 'motif_corr'
        elif fixed_r_mode:
            tie_break = 'valid_roc'
        else:
            tie_break = 'composite'

        for model in MODEL_ORDER:
            if use_motif_sel:
                br, comp = _best_run_motif_corr_valid(records, model)
            elif fixed_r_mode:
                br, comp = _best_run_max_valid_roc(records, model)
            else:
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
            per_fold = _per_fold_best_for_row(
                records, model, chosen_row, w_pred, w_m, w_n, tie_break=tie_break
            )

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
                'selection_by': selection_by,
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
    p.add_argument(
        '--set_r',
        type=float,
        default=None,
        help='If set (or env SET_R), keep only runs with this fix_r/final_r (per experiment type).',
    )
    p.add_argument(
        '--selection-by',
        type=str,
        choices=['composite', 'motif_corr_valid'],
        default=None,
        help='Anchor hyperparam row + per-fold seed: composite vs motif-level r on valid (default: env SELECTION_BY or composite).',
    )
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    set_r = args.set_r if args.set_r is not None else _parse_set_r_env()
    sel = (args.selection_by or os.environ.get('SELECTION_BY') or 'composite').strip().lower()
    if sel not in ('composite', 'motif_corr_valid'):
        print(f'[WARN] Invalid SELECTION_BY={sel!r}, using composite')
        sel = 'composite'

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
        set_r=set_r,
        selection_by=sel,
    )


if __name__ == '__main__':
    main()
