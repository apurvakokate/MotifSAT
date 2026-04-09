#!/usr/bin/env python
"""
Compare model attention-derived motif importance with Fisher-significant motifs (ROC / AUC).

For each node_scores.jsonl under RESULTS_DIR, aggregate mean |score| per global motif_id on a split,
join with precomputed association CSV, and compute ROC AUC treating Fisher FDR significance as label.

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


def one_auc(
    assoc: pd.DataFrame,
    mean_score: dict[int, float],
    *,
    use_fdr_q: bool,
) -> tuple[float | None, int]:
    """ROC AUC: positive label = Fisher-significant motif; score = mean |attention|."""
    if roc_auc_score is None:
        return None, 0
    y = []
    s = []
    for _, row in assoc.iterrows():
        mid = int(row['motif_id'])
        if mid not in mean_score:
            continue
        if use_fdr_q:
            if 'fisher_q_bh' not in row or pd.isna(row['fisher_q_bh']):
                continue
            lab = float(row['fisher_q_bh']) < 0.05
        else:
            if 'fisher_p' not in row or pd.isna(row['fisher_p']):
                continue
            lab = float(row['fisher_p']) < 0.05
        y.append(int(lab))
        s.append(float(mean_score[mid]))
    if len(y) < 3 or len(set(y)) < 2:
        return float('nan'), len(y)
    return float(roc_auc_score(y, s)), len(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--association_csv', type=str, required=True, help='*_motif_class_association.csv from compute_motif_class_association')
    p.add_argument('--results_dir', type=str, default=os.environ.get('RESULTS_DIR', '../tuning_results'))
    p.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'Mutagenicity'))
    p.add_argument('--split', type=str, default='test', help='Which split in node_scores.jsonl')
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
        auc_p, n_m = one_auc(assoc, mean_score, use_fdr_q=False)
        auc_q, n_mq = one_auc(assoc, mean_score, use_fdr_q=True)
        row = {
            **meta,
            'node_scores_jsonl': str(jsonl),
            'n_motifs_with_scores': len(mean_score),
            'auc_score_vs_fisher_p_lt005': auc_p,
            'auc_score_vs_fisher_q_lt005': auc_q,
            'n_motifs_used_auc_p': n_m,
            'n_motifs_used_auc_q': n_mq,
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
