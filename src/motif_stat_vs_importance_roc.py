#!/usr/bin/env python
"""
Compare model attention-derived motif importance with motif-class statistics.

For each node_scores.jsonl under RESULTS_DIR:
1) aggregate mean |score| per motif *name* on a split,
2) join with precomputed association CSV by motif name,
3) compute ROC AUC and Spearman diagnostics.

Labeling modes for ROC:
- fisher_q (default): positive motif iff fisher_q_bh <= alpha
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

    rows = []
    for jsonl in sorted(results_root.glob('**/node_scores.jsonl')):
        seed_dir = jsonl.parent
        mean_score = mean_abs_score_per_motif_name(jsonl, motif_id_to_name, split=args.split)
        if not mean_score:
            continue
        meta = parse_result_path(seed_dir)
        scores, deltas, qvals = paired_score_and_stats(assoc_by_name, mean_score)
        if args.label_mode == 'fisher_q':
            auc_d, n_m_auc, thr = auc_vs_fisher_q(
                scores,
                qvals,
                fisher_q_alpha=args.fisher_q_alpha,
            )
        else:
            auc_d, n_m_auc, thr = auc_vs_delta_top_quantile(
                scores,
                deltas,
                delta_label_quantile=args.delta_label_quantile,
            )
        rho_d, rho_p, n_m_corr = spearman_score_vs_delta(scores, deltas)
        row = {
            **meta,
            'node_scores_jsonl': str(jsonl),
            'n_motifs_with_scores': len(mean_score),
            'label_mode': args.label_mode,
            'fisher_q_alpha': float(args.fisher_q_alpha),
            'delta_label_quantile': float(args.delta_label_quantile),
            'label_threshold_value': thr,
            'auc_score_vs_stat_label': auc_d,
            'n_motifs_used_auc_delta': n_m_auc,
            'spearman_score_vs_abs_delta': rho_d,
            'spearman_pvalue_score_vs_abs_delta': rho_p,
            'n_motifs_used_spearman': n_m_corr,
        }
        rows.append(row)

    if not rows:
        print('[WARN] No node_scores.jsonl found or all empty.')
        return

    out = Path(args.out_csv)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'[INFO] Wrote {out.resolve()} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
