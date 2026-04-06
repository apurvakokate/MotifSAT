#!/usr/bin/env python
"""
Pick anchor (experiment × model) for plots, write best_results CSVs.

Selection (--selection-by) chooses which hyperparam row label to use per model; use encoder_branch to fix
emb_stop=encoder (motif_readout_pred_info_only). ROC and explainer
cells are the same numbers as collect_mutagenicity_tables for that row (build_table / build_posthoc_table).

Plots use the anchor run's seed_dir. Optional SET_R filters runs before building pivots.
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
    _sort_agg_by_row_val,
    build_posthoc_table,
    build_table,
    compute_node_score_impact_correlation,
    compute_posthoc_correlation,
    find_results,
    format_mean_std_count,
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
    'motif_readout_baseline_f07': 'Motif readout baseline (f_r=0.7, motif samp, prior gate)',
    'motif_readout_e1_logit_standardize': 'E1 per-graph logit standardization',
    'motif_readout_e2_temperature': 'E2 learned temperature on motif logits',
    'motif_readout_e3_max_pool': 'E3 max-pool motif readout',
    'motif_readout_e4_max_mean_pool': 'E4 max+mean concat readout',
    'motif_readout_e5_interp_head': 'E5 attention-statistics interp head',
    'motif_readout_e6_no_gate': 'E6 no node gate (motif logit only)',
    'motif_readout_e7_multiplicative_gate': 'E7 multiplicative gate',
    'motif_readout_e8_entropy_sweep': 'E8 node-attention entropy sweep',
    'motif_readout_e9_motif_ib_sweep': 'E9 motif-level IB sweep',
    'motif_readout_e10_align_sweep': 'E10 motif–node alignment sweep',
    'motif_readout_entropy_pool_sweep': 'Entropy bonus; pooling sweep (mean/max/max_mean/intra_att)',
    'motif_readout_maxmean_node_vs_edge_att': 'max_mean readout: node inj vs edge attention',
    'motif_readout_pred_info_only': 'L_pred+L_info; max_mean; sweep GNN layer for motif emb (α discriminability)',
    'factored_motif_attention_grid': 'Factored motif attention (M1–M4 × N1–N3; multi-z_k + factored logits + mean-α IB)',
    'factored_motif_additive': 'Factored motif additive (LN z^(1)||z^att; ℓ_k+δ intra; IB on σ(ℓ_k); r_f sweep)',
    'simplified_factored_motif_additive': 'Simplified factored additive (MLP(LN z^att)); 010; L_pred + motif L_info on σ(ℓ_k) (raw score); info_coef 0.01; motif_ib off; info_warmup 20; final_r=0.8',
    'simplified_motif_readout': 'Simplified motif readout: same as simplified_factored_motif_additive but node ℓ = ℓ_k only (broadcast; no intra-motif δ)',
    'simplified_motif_readout_maxmean': 'Like simplified_motif_readout; motif emb = max||mean pool (not intra-att) before motif MLP',
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


def _cell_from_pivots(mean_df, std_df, count_df, row: str, col: str) -> str:
    """Same formatted string as collect_mutagenicity_tables for that (row, model) cell."""
    if mean_df is None or row not in mean_df.index or col not in mean_df.columns:
        return ''
    combined = format_mean_std_count(mean_df, std_df, count_df)
    if combined is None:
        return ''
    if row not in combined.index or col not in combined.columns:
        return ''
    return str(combined.at[row, col])


def _build_node_posthoc_table(records: list, split: str = 'test'):
    """Same layout as build_posthoc_table but Pearson r from compute_node_score_impact_correlation."""
    rows = []
    for rec in records:
        r, _p, _n = compute_node_score_impact_correlation(rec['seed_dir'], split=split)
        rows.append({
            'model': rec['model'],
            'row': rec['row'],
            'row_val': rec['row_val'],
            'pearson_r': r,
        })
    if not rows:
        return None, None, None
    df = pd.DataFrame(rows)
    agg = df.groupby(['row', 'row_val', 'model'])['pearson_r'].agg(['mean', 'std', 'count']).reset_index()
    agg = _sort_agg_by_row_val(agg)
    pivot_mean = agg.pivot(index='row', columns='model', values='mean')
    pivot_std = agg.pivot(index='row', columns='model', values='std')
    pivot_count = agg.pivot(index='row', columns='model', values='count')
    row_order = agg.drop_duplicates('row')['row'].tolist()
    pivot_mean = pivot_mean.reindex(row_order).dropna(how='all')
    pivot_std = pivot_std.reindex(row_order).dropna(how='all')
    pivot_count = pivot_count.reindex(row_order).dropna(how='all')
    return pivot_mean, pivot_std, pivot_count


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


def _record_is_encoder_emb_stop(r: dict) -> bool:
    """motif_readout_pred_info_only (and similar): row label emb_stop=encoder from collect tables."""
    if str(r.get('row', '')) == 'emb_stop=encoder':
        return True
    rv = r.get('row_val')
    if rv is not None and str(rv).strip().lower() == 'encoder':
        return True
    return False


def _best_run_encoder_branch(records: list, model: str) -> tuple[dict | None, float]:
    """Anchor: only runs with motif GNN emb_stop=encoder; tie-break max valid ROC. Stored as composite_valid."""
    best = None
    best_rv = -1.0
    for r in records:
        if r['model'] != model:
            continue
        if not _record_is_encoder_emb_stop(r):
            continue
        rv = r['metrics'].get('metric/best_clf_roc_valid', np.nan)
        if not np.isfinite(rv):
            continue
        rv = float(rv)
        if rv > best_rv:
            best_rv = rv
            best = r
    if best is None:
        return None, float('nan')
    return best, best_rv


def _unpack_mean_std_count(t):
    """build_table / build_posthoc_table return (mean_df, std_df, count_df) or (None, None, None)."""
    if t is None or t[0] is None:
        return None, None, None
    return t[0], t[1], t[2]


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
    use_encoder_branch = selection_by == 'encoder_branch'
    if selection_by not in ('composite', 'motif_corr_valid', 'encoder_branch'):
        raise ValueError(
            f"selection_by must be 'composite', 'motif_corr_valid', or 'encoder_branch', got {selection_by!r}"
        )

    if use_encoder_branch:
        sel_label = 'encoder_emb_stop_tie_valid_roc'
    elif use_motif_sel:
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
            'Table values match collect_mutagenicity_tables pivots at the anchor row per model; '
            'selection only picks that row for plots. '
            + (f'SET_R={set_r} filters runs before pivots. ' if fixed_r_mode else '')
            + (
                'encoder_branch: anchor row is emb_stop=encoder only (no composite/motif-argmax). '
                if use_encoder_branch
                else ''
            )
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

        pred_tr_m, pred_tr_s, pred_tr_c = _unpack_mean_std_count(
            build_table(records, 'metric/best_clf_roc_train', verbose=verbose)
        )
        pred_va_m, pred_va_s, pred_va_c = _unpack_mean_std_count(
            build_table(records, 'metric/best_clf_roc_valid', verbose=verbose)
        )
        pred_te_m, pred_te_s, pred_te_c = _unpack_mean_std_count(
            build_table(records, 'metric/best_clf_roc_test', verbose=verbose)
        )
        ph_tr_m, ph_tr_s, ph_tr_c = _unpack_mean_std_count(build_posthoc_table(records, split='train'))
        ph_va_m, ph_va_s, ph_va_c = _unpack_mean_std_count(build_posthoc_table(records, split='valid'))
        ph_te_m, ph_te_s, ph_te_c = _unpack_mean_std_count(build_posthoc_table(records, split='test'))
        node_tr_m, node_tr_s, node_tr_c = _unpack_mean_std_count(_build_node_posthoc_table(records, split='train'))
        node_va_m, node_va_s, node_va_c = _unpack_mean_std_count(_build_node_posthoc_table(records, split='valid'))
        node_te_m, node_te_s, node_te_c = _unpack_mean_std_count(_build_node_posthoc_table(records, split='test'))

        for model in MODEL_ORDER:
            if use_encoder_branch:
                br, comp = _best_run_encoder_branch(records, model)
            elif use_motif_sel:
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
            perf_tr[model] = _cell_from_pivots(pred_tr_m, pred_tr_s, pred_tr_c, chosen_row, model)
            perf_va[model] = _cell_from_pivots(pred_va_m, pred_va_s, pred_va_c, chosen_row, model)
            perf_te[model] = _cell_from_pivots(pred_te_m, pred_te_s, pred_te_c, chosen_row, model)
            ex_tr[f'{model}_motif_r'] = _cell_from_pivots(ph_tr_m, ph_tr_s, ph_tr_c, chosen_row, model)
            ex_tr[f'{model}_node_r'] = _cell_from_pivots(node_tr_m, node_tr_s, node_tr_c, chosen_row, model)
            ex_va[f'{model}_motif_r'] = _cell_from_pivots(ph_va_m, ph_va_s, ph_va_c, chosen_row, model)
            ex_va[f'{model}_node_r'] = _cell_from_pivots(node_va_m, node_va_s, node_va_c, chosen_row, model)
            ex_te[f'{model}_motif_r'] = _cell_from_pivots(ph_te_m, ph_te_s, ph_te_c, chosen_row, model)
            ex_te[f'{model}_node_r'] = _cell_from_pivots(node_te_m, node_te_s, node_te_c, chosen_row, model)

            sd: Path = br['seed_dir']
            n_runs = ''
            if pred_va_c is not None and chosen_row in pred_va_c.index and model in pred_va_c.columns:
                nv = pred_va_c.at[chosen_row, model]
                if pd.notna(nv):
                    n_runs = int(nv)
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
                'n_runs_in_table_cell': n_runs,
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
        choices=['composite', 'motif_corr_valid', 'encoder_branch'],
        default=None,
        help='Anchor row: composite | motif_corr_valid | encoder_branch (emb_stop=encoder; tie valid ROC). '
        'Default: env SELECTION_BY or composite.',
    )
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    set_r = args.set_r if args.set_r is not None else _parse_set_r_env()
    sel = (args.selection_by or os.environ.get('SELECTION_BY') or 'composite').strip().lower()
    if sel not in ('composite', 'motif_corr_valid', 'encoder_branch'):
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
            'motif_readout_baseline_f07',
            'motif_readout_e1_logit_standardize',
            'motif_readout_e2_temperature',
            'motif_readout_e3_max_pool',
            'motif_readout_e4_max_mean_pool',
            'motif_readout_e5_interp_head',
            'motif_readout_e6_no_gate',
            'motif_readout_e7_multiplicative_gate',
            'motif_readout_e8_entropy_sweep',
            'motif_readout_e9_motif_ib_sweep',
            'motif_readout_e10_align_sweep',
            'motif_readout_entropy_pool_sweep',
            'motif_readout_maxmean_node_vs_edge_att',
            'motif_readout_pred_info_only',
            'factored_motif_attention_grid',
            'factored_motif_additive',
            'simplified_factored_motif_additive',
            'simplified_motif_readout',
            'simplified_motif_readout_maxmean',
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
