#!/usr/bin/env python
"""
Compare model attention-derived motif importance with class-discriminative motifs (|delta|).

For each node_scores.jsonl under RESULTS_DIR, aggregate mean |score| per global motif_id on a split,
join with precomputed association CSV, and compute:
- ROC AUC where positive label = top-|delta| motifs (by quantile)
- Spearman correlation between mean |score| and |delta|

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


def mean_abs_score_per_motif(jsonl_path: Path, split: str = 'test') -> dict[int, float]:
    sums: dict[int, float] = defaultdict(float)
    cnt: dict[int, int] = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('split') != split:
                continue
            m = r.get('motif_index', r.get('motif_idx'))
            if m is None or int(m) < 0:
                continue
            m = int(m)
            sums[m] += abs(float(r.get('score', 0.0)))
            cnt[m] += 1
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


def paired_score_and_delta(
    assoc: pd.DataFrame,
    mean_score: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect aligned vectors: score = mean |attention|, target = |delta|."""
    s: list[float] = []
    d: list[float] = []
    for _, row in assoc.iterrows():
        mid = int(row['motif_id'])
        if mid not in mean_score:
            continue
        dd = motif_abs_delta(row)
        if dd is None or not np.isfinite(dd):
            continue
        s.append(float(mean_score[mid]))
        d.append(float(dd))
    return np.asarray(s, dtype=float), np.asarray(d, dtype=float)


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
        '--delta_label_quantile',
        type=float,
        default=0.75,
        help='Positive motif label for ROC is |delta| >= this quantile (default: top 25%%).',
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

    results_root = Path(args.results_dir) / args.dataset
    if not results_root.is_dir():
        print(f'[WARN] No results dir {results_root}')
        return

    rows = []
    for jsonl in sorted(results_root.glob('**/node_scores.jsonl')):
        seed_dir = jsonl.parent
        mean_score = mean_abs_score_per_motif(jsonl, split=args.split)
        if not mean_score:
            continue
        meta = parse_result_path(seed_dir)
        scores, deltas = paired_score_and_delta(assoc, mean_score)
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
            'delta_label_quantile': float(args.delta_label_quantile),
            'delta_threshold_value': thr,
            'auc_score_vs_abs_delta_topq': auc_d,
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
