#!/usr/bin/env python
"""
Compare model attention-derived motif importance with motif-class statistics.

For each node_scores.jsonl under RESULTS_DIR:
1) aggregate mean |score| per motif *name* on a split,
2) join with precomputed association CSV by motif name,
3) compute score-vs-stat diagnostics (top/bottom-k precision for fisher_q mode) and Spearman.

Labeling modes:
- fisher_q (default): top/bottom-k precision against sign(delta) among statistically significant motifs
  (q < alpha), excluding ambiguous motifs (q >= alpha) from denominators.
- abs_delta_topq: positive motif iff |delta| >= quantile(|delta|, q)

Writes a CSV table (e.g. motif_importance_vs_fisher_roc.csv).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def _normalize_motif_name(name: str | None) -> str | None:
    if name is None:
        return None
    out = str(name).strip()
    if out == '' or out.lower() == 'nan':
        return None
    return out


def _motif_name_col(assoc: pd.DataFrame) -> str | None:
    for c in ('motif_smiles', 'motif', 'motif_name'):
        if c in assoc.columns:
            return c
    return None


def _prepare_association_by_motif_name(assoc: pd.DataFrame) -> pd.DataFrame:
    if assoc.empty:
        return pd.DataFrame(columns=['motif_name', 'abs_delta', 'fisher_q_bh'])
    if 'motif_id' not in assoc.columns:
        raise ValueError('association CSV must include motif_id')

    work = assoc.copy()
    motif_col = _motif_name_col(work)
    if motif_col is None:
        # Fallback keeps script running, though name-based join quality is best with motif_smiles.
        work['motif_name'] = work['motif_id'].map(lambda x: f'motif_id:{int(x)}')
    else:
        work['motif_name'] = work[motif_col].map(_normalize_motif_name)
        miss = work['motif_name'].isna()
        if miss.any():
            work.loc[miss, 'motif_name'] = work.loc[miss, 'motif_id'].map(lambda x: f'motif_id:{int(x)}')

    if 'abs_delta_p_motif' in work.columns:
        work['abs_delta'] = pd.to_numeric(work['abs_delta_p_motif'], errors='coerce')
    else:
        work['abs_delta'] = work.apply(motif_abs_delta, axis=1)

    if 'fisher_q_bh' in work.columns:
        work['fisher_q_bh'] = pd.to_numeric(work['fisher_q_bh'], errors='coerce')
    else:
        work['fisher_q_bh'] = np.nan

    grouped = (work.groupby('motif_name', as_index=False)
               .agg(abs_delta=('abs_delta', 'max'),
                    fisher_q_bh=('fisher_q_bh', 'min')))
    return grouped


def _motif_id_to_name_map(assoc: pd.DataFrame) -> dict[int, str]:
    if assoc.empty or 'motif_id' not in assoc.columns:
        return {}
    motif_col = _motif_name_col(assoc)
    mapping: dict[int, str] = {}
    for _, row in assoc.iterrows():
        try:
            mid = int(row['motif_id'])
        except Exception:
            continue
        if motif_col is None:
            mapping[mid] = f'motif_id:{mid}'
            continue
        mname = _normalize_motif_name(row.get(motif_col))
        if mname is None:
            mname = f'motif_id:{mid}'
        mapping[mid] = mname
    return mapping


def mean_abs_score_per_motif_name(
    jsonl_path: Path,
    motif_id_to_name: dict[int, str],
    split: str = 'test',
) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('split') != split:
                continue
            m = r.get('motif_index', r.get('motif_idx'))
            if m is None or int(m) < 0:
                continue
            m = int(m)
            motif_name = motif_id_to_name.get(m)
            if motif_name is None:
                continue
            sums[motif_name] += abs(float(r.get('score', 0.0)))
            cnt[motif_name] += 1
    return {m: sums[m] / cnt[m] for m in sums if cnt[m] > 0}


def parse_result_path(seed_dir: Path) -> dict:
    """Infer model / experiment from .../model_<ARCH>/experiment_<NAME>/.../fold*_seed*."""
    out = {'seed_dir': str(seed_dir)}
    for part in seed_dir.resolve().parts:
        if part.startswith('model_') and 'model' not in out:
            out['model'] = part[len('model_'):]
        if part.startswith('experiment_'):
            out['experiment'] = part[len('experiment_'):]
    return out


def motif_abs_delta(row: pd.Series) -> float | None:
    """
    |delta| = |P(motif|y=1) - P(motif|y=0)| from contingency:
              y=0   y=1
    absent      a     b
    present     c     d
    """
    need = [
        'contingency_a_absent_y0',
        'contingency_b_absent_y1',
        'contingency_c_present_y0',
        'contingency_d_present_y1',
    ]
    for k in need:
        if k not in row or pd.isna(row[k]):
            return None
    a = float(row['contingency_a_absent_y0'])
    b = float(row['contingency_b_absent_y1'])
    c = float(row['contingency_c_present_y0'])
    d = float(row['contingency_d_present_y1'])
    n0 = a + c
    n1 = b + d
    if n0 <= 0 or n1 <= 0:
        return None
    p_m_given_y0 = c / n0
    p_m_given_y1 = d / n1
    return float(abs(p_m_given_y1 - p_m_given_y0))


def paired_score_and_stats(
    assoc_by_name: pd.DataFrame,
    mean_score_by_name: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect aligned vectors by motif name: score, |delta|, fisher_q."""
    if assoc_by_name.empty or not mean_score_by_name:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    joined = assoc_by_name.copy()
    joined['score'] = joined['motif_name'].map(mean_score_by_name)
    joined = joined.dropna(subset=['score'])
    if joined.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    scores = joined['score'].to_numpy(dtype=float)
    deltas = joined['abs_delta'].to_numpy(dtype=float)
    qvals = joined['fisher_q_bh'].to_numpy(dtype=float)
    return scores, deltas, qvals


def auc_vs_fisher_q(
    scores: np.ndarray,
    qvals: np.ndarray,
    *,
    fisher_q_alpha: float,
) -> tuple[float | None, int, float]:
    """ROC AUC where positives are fisher_q_bh <= alpha."""
    if roc_auc_score is None:
        return None, 0, float('nan')
    if scores.size < 3 or qvals.size < 3:
        return float('nan'), int(min(scores.size, qvals.size)), fisher_q_alpha
    mask = np.isfinite(scores) & np.isfinite(qvals)
    if mask.sum() < 3:
        return float('nan'), int(mask.sum()), fisher_q_alpha
    scores = scores[mask]
    qvals = qvals[mask]
    y = (qvals <= fisher_q_alpha).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan'), int(len(y)), fisher_q_alpha
    return float(roc_auc_score(y, scores)), int(len(y)), fisher_q_alpha


def precision_top_bottom_vs_fisher_q(
    scores: np.ndarray,
    deltas: np.ndarray,
    qvals: np.ndarray,
    *,
    fisher_q_alpha: float,
    top_k: int,
) -> dict[str, float | int]:
    """
    Rank motifs by descending score. Evaluate:
      - top-k correctness:   delta > 0 and q < alpha
      - bottom-k correctness: delta < 0 and q < alpha
      - ambiguous motifs: q >= alpha (or non-finite q), excluded from denominators
    """
    if top_k <= 0:
        return {
            'precision_top': float('nan'),
            'precision_bottom': float('nan'),
            'precision_combined': float('nan'),
            'k_used': 0,
            'n_ranked': 0,
            'ambiguous_top': 0,
            'ambiguous_bottom': 0,
            'ambiguous_total': 0,
            'correct_top': 0,
            'correct_bottom': 0,
            'den_top': 0,
            'den_bottom': 0,
            'den_combined': 0,
        }

    mask = np.isfinite(scores) & np.isfinite(deltas)
    if mask.sum() == 0:
        return {
            'precision_top': float('nan'),
            'precision_bottom': float('nan'),
            'precision_combined': float('nan'),
            'k_used': 0,
            'n_ranked': 0,
            'ambiguous_top': 0,
            'ambiguous_bottom': 0,
            'ambiguous_total': 0,
            'correct_top': 0,
            'correct_bottom': 0,
            'den_top': 0,
            'den_bottom': 0,
            'den_combined': 0,
        }

    s = scores[mask]
    d = deltas[mask]
    q = qvals[mask]
    n_ranked = int(s.size)
    k_used = int(min(top_k, n_ranked))
    if k_used == 0:
        return {
            'precision_top': float('nan'),
            'precision_bottom': float('nan'),
            'precision_combined': float('nan'),
            'k_used': 0,
            'n_ranked': n_ranked,
            'ambiguous_top': 0,
            'ambiguous_bottom': 0,
            'ambiguous_total': 0,
            'correct_top': 0,
            'correct_bottom': 0,
            'den_top': 0,
            'den_bottom': 0,
            'den_combined': 0,
        }

    ranked_idx = np.argsort(-s)
    top_idx = ranked_idx[:k_used]
    bot_idx = ranked_idx[-k_used:]

    is_sig = np.isfinite(q) & (q < fisher_q_alpha)

    ambiguous_top = int((~is_sig[top_idx]).sum())
    ambiguous_bottom = int((~is_sig[bot_idx]).sum())
    ambiguous_total = ambiguous_top + ambiguous_bottom

    correct_top = int(((d[top_idx] > 0) & is_sig[top_idx]).sum())
    correct_bottom = int(((d[bot_idx] < 0) & is_sig[bot_idx]).sum())

    den_top = int(k_used - ambiguous_top)
    den_bottom = int(k_used - ambiguous_bottom)
    den_combined = int(2 * k_used - ambiguous_total)

    precision_top = (correct_top / den_top) if den_top > 0 else float('nan')
    precision_bottom = (correct_bottom / den_bottom) if den_bottom > 0 else float('nan')
    precision_combined = (
        (correct_top + correct_bottom) / den_combined if den_combined > 0 else float('nan')
    )

    return {
        'precision_top': float(precision_top),
        'precision_bottom': float(precision_bottom),
        'precision_combined': float(precision_combined),
        'k_used': k_used,
        'n_ranked': n_ranked,
        'ambiguous_top': ambiguous_top,
        'ambiguous_bottom': ambiguous_bottom,
        'ambiguous_total': ambiguous_total,
        'correct_top': correct_top,
        'correct_bottom': correct_bottom,
        'den_top': den_top,
        'den_bottom': den_bottom,
        'den_combined': den_combined,
    }


def auc_vs_delta_top_quantile(
    scores: np.ndarray,
    deltas: np.ndarray,
    *,
    delta_label_quantile: float,
) -> tuple[float | None, int, float]:
    """
    ROC AUC where positive motifs are top-|delta| motifs by quantile threshold.
    """
    if roc_auc_score is None:
        return None, 0, float('nan')
    if scores.size < 3 or deltas.size < 3:
        return float('nan'), int(min(scores.size, deltas.size)), float('nan')
    thr = float(np.nanquantile(deltas, delta_label_quantile))
    y = (deltas >= thr).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan'), int(len(y)), thr
    return float(roc_auc_score(y, scores)), int(len(y)), thr


def spearman_score_vs_delta(scores: np.ndarray, deltas: np.ndarray) -> tuple[float | None, float | None, int]:
    """Spearman rank correlation between score and |delta|."""
    if spearmanr is None:
        return None, None, 0
    if scores.size < 3 or deltas.size < 3:
        return float('nan'), float('nan'), int(min(scores.size, deltas.size))
    rho, pval = spearmanr(scores, deltas)
    rho = float(rho) if np.isfinite(rho) else float('nan')
    pval = float(pval) if np.isfinite(pval) else float('nan')
    return rho, pval, int(len(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--association_csv', type=str, required=True, help='*_motif_class_association.csv from compute_motif_class_association')
    p.add_argument('--results_dir', type=str, default=os.environ.get('RESULTS_DIR', '../tuning_results'))
    p.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Mutagenicity'))
    p.add_argument(
        '--experiments',
        nargs='*',
        default=None,
        help='Optional experiment-name filter (values from path segment experiment_<name>).',
    )
    p.add_argument('--split', type=str, default='test', help='Which split in node_scores.jsonl')
    p.add_argument(
        '--label_mode',
        type=str,
        default='fisher_q',
        choices=['fisher_q', 'abs_delta_topq'],
        help='How positive motifs are labeled for ROC (default: fisher_q).',
    )
    p.add_argument(
        '--fisher_q_alpha',
        type=float,
        default=0.05,
        help='Positive motif threshold for fisher_q mode: fisher_q_bh <= alpha.',
    )
    p.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='For fisher_q mode: evaluate top-k and bottom-k motifs by score.',
    )
    p.add_argument(
        '--delta_label_quantile',
        type=float,
        default=0.75,
        help='Positive motif label in abs_delta_topq mode is |delta| >= this quantile.',
    )
    p.add_argument('--out_csv', type=str, default='motif_importance_vs_fisher_roc.csv')
    args = p.parse_args()

    assoc_path = Path(args.association_csv)
    if not assoc_path.is_file():
        raise FileNotFoundError(assoc_path)
    assoc = pd.read_csv(assoc_path)
    if assoc.empty:
        print('[WARN] Association table empty; nothing to score.')
        return

    assoc_by_name = _prepare_association_by_motif_name(assoc)
    motif_id_to_name = _motif_id_to_name_map(assoc)
    if assoc_by_name.empty:
        print('[WARN] Association table has no motifs after preprocessing; nothing to score.')
        return

    results_root = Path(args.results_dir) / args.dataset
    if not results_root.is_dir():
        print(f'[WARN] No results dir {results_root}')
        return

    exp_filter = set(args.experiments) if args.experiments else None
    if exp_filter:
        print(f'[INFO] Filtering experiments: {sorted(exp_filter)}')

    rows = []
    for jsonl in sorted(results_root.glob('**/node_scores.jsonl')):
        seed_dir = jsonl.parent
        meta = parse_result_path(seed_dir)
        exp_name = meta.get('experiment')
        if exp_filter is not None and exp_name not in exp_filter:
            continue
        mean_score = mean_abs_score_per_motif_name(jsonl, motif_id_to_name, split=args.split)
        if not mean_score:
            continue
        scores, deltas, qvals = paired_score_and_stats(assoc_by_name, mean_score)
        if args.label_mode == 'fisher_q':
            fisher_prec = precision_top_bottom_vs_fisher_q(
                scores,
                deltas,
                qvals,
                fisher_q_alpha=args.fisher_q_alpha,
                top_k=args.top_k,
            )
            stat_metric = fisher_prec['precision_combined']
            n_m_stat = fisher_prec['n_ranked']
            thr = args.fisher_q_alpha
        else:
            auc_d, n_m_auc, thr = auc_vs_delta_top_quantile(
                scores,
                deltas,
                delta_label_quantile=args.delta_label_quantile,
            )
            fisher_prec = None
            stat_metric = auc_d
            n_m_stat = n_m_auc
        rho_d, rho_p, n_m_corr = spearman_score_vs_delta(scores, deltas)
        row = {
            **meta,
            'node_scores_jsonl': str(jsonl),
            'n_motifs_with_scores': len(mean_score),
            'label_mode': args.label_mode,
            'fisher_q_alpha': float(args.fisher_q_alpha),
            'top_k': int(args.top_k),
            'delta_label_quantile': float(args.delta_label_quantile),
            'label_threshold_value': thr,
            'auc_score_vs_stat_label': stat_metric,
            'n_motifs_used_auc_delta': n_m_stat,
            'spearman_score_vs_abs_delta': rho_d,
            'spearman_pvalue_score_vs_abs_delta': rho_p,
            'n_motifs_used_spearman': n_m_corr,
        }
        if fisher_prec is not None:
            row.update({
                'precision_top': fisher_prec['precision_top'],
                'precision_bottom': fisher_prec['precision_bottom'],
                'precision_combined': fisher_prec['precision_combined'],
                'k_used': fisher_prec['k_used'],
                'n_ranked': fisher_prec['n_ranked'],
                'ambiguous_top': fisher_prec['ambiguous_top'],
                'ambiguous_bottom': fisher_prec['ambiguous_bottom'],
                'ambiguous_total': fisher_prec['ambiguous_total'],
                'correct_top': fisher_prec['correct_top'],
                'correct_bottom': fisher_prec['correct_bottom'],
                'den_top': fisher_prec['den_top'],
                'den_bottom': fisher_prec['den_bottom'],
                'den_combined': fisher_prec['den_combined'],
            })
        rows.append(row)

    if not rows:
        print('[WARN] No node_scores.jsonl found or all empty.')
        return

    out = Path(args.out_csv)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'[INFO] Wrote {out.resolve()} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
