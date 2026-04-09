#!/usr/bin/env python
"""
Precompute motif–class association (Fisher / chi-squared) on the training split and save CSV + p-value tensor.

Typical use (from repo `src/`):
  python compute_motif_class_association.py --dataset Mutagenicity --fold 0

Outputs under --out_dir (default: <data_dir>/motif_association):
  {stem}_motif_class_association.csv
  {stem}_motif_pvalues.pt   # torch tensor [V] for AddMotifAssocP
  {stem}_meta.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataLoader import CHOSEN_THRESHOLD, get_setup_files_with_folds
from utils.get_data_loaders import DATASET_COLUMN, DATASET_TYPE
from motif_class_association import (
    association_table,
    build_graph_motif_presence,
    motif_pvalue_vector,
    save_association_artifacts,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Mutagenicity')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--algorithm', type=str, default='BRICS')
    p.add_argument(
        '--dictionary_path',
        type=str,
        default='/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DICTIONARY',
    )
    p.add_argument('--dictionary_fold_variant', type=str, default='nofilter')
    p.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Project data dir (default: ../data next to src from global_config)',
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Override output directory (default: <data_dir>/motif_association)',
    )
    p.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='FOLDS CSV with columns group, smiles, labels (default: DomainDrivenGlobalExpl .../FOLDS/<ds>_<fold>.csv)',
    )
    p.add_argument('--min_support', type=int, default=5, help='Min present & absent graphs per motif')
    p.add_argument(
        '--which_split',
        type=str,
        default='training',
        choices=['training', 'valid', 'test', 'train_valid', 'all'],
        help='Contingency table built from this CSV `group` (default: training only, no test leakage).',
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir or (Path(__file__).resolve().parent.parent / 'data'))
    out_root = Path(args.out_dir) if args.out_dir else (data_dir / 'motif_association')
    ds = args.dataset
    algo = args.algorithm
    thr_key = CHOSEN_THRESHOLD.get('BRICS', {}).get(ds, 0.2)
    date_tag = f'{algo}{thr_key}'

    lookup, motif_list, *_rest = get_setup_files_with_folds(
        ds, date_tag, args.fold, algo,
        path=args.dictionary_path,
        dictionary_fold_variant=args.dictionary_fold_variant,
    )

    # Same default as utils/get_data_loaders.py (FOLDS CSV under DomainDrivenGlobalExpl)
    default_csv = (
        Path('/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DomainDrivenGlobalExpl/datasets/FOLDS')
        / f'{ds}_{args.fold}.csv'
    )
    csv_path = Path(args.csv_file) if args.csv_file else default_csv
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Fold CSV not found at {csv_path}. Pass --csv_file /path/to/{ds}_{args.fold}.csv'
        )

    df = pd.read_csv(csv_path)
    label_col = DATASET_COLUMN[ds][0] if isinstance(DATASET_COLUMN[ds], list) else DATASET_COLUMN[ds]

    if args.which_split == 'all':
        sub = df
    elif args.which_split == 'train_valid':
        sub = df[df['group'].isin(['training', 'valid'])]
    else:
        sub = df[df['group'] == args.which_split]

    smiles_list = sub['smiles'].values.tolist()
    y_raw = sub[label_col].values
    if DATASET_TYPE[ds] != 'BinaryClass':
        raise SystemExit(f'Dataset {ds} is not BinaryClass; association is only implemented for binary labels.')

    y = np.asarray(y_raw).astype(int).ravel()
    # Handle possible [0,1] or {-1,1}
    if set(np.unique(y)) == {-1, 1}:
        y = (y > 0).astype(int)
    elif y.max() > 1 or y.min() < 0:
        print(f'[WARN] Unexpected label range {np.unique(y)}; using y>0 as positive class.')
        y = (y > 0).astype(int)

    X, y2, motif_ids = build_graph_motif_presence(smiles_list, y, lookup)
    assert np.array_equal(y2, y)

    motif_smiles = list(motif_list) if motif_list is not None else None
    tab = association_table(X, y, motif_ids, motif_smiles, min_support=args.min_support)
    vocab_size = len(motif_list) if motif_list is not None else (max(motif_ids) + 1 if motif_ids else 0)
    if tab.empty:
        print('[WARN] Empty association table (increase data or lower --min_support).')
        pvec = np.full(vocab_size, np.nan, dtype=float)
    else:
        pvec = motif_pvalue_vector(tab, vocab_size)

    stem = f'{ds}_fold{args.fold}_{args.which_split}'
    meta = {
        'dataset': ds,
        'fold': args.fold,
        'which_split': args.which_split,
        'n_graphs': int(len(y)),
        'n_motifs_tested': int(len(motif_ids)),
        'n_motifs_rows': int(len(tab)),
        'vocab_size': int(vocab_size),
        'algorithm': algo,
        'dictionary_fold_variant': args.dictionary_fold_variant,
        'csv_source': str(csv_path),
    }
    save_association_artifacts(out_root, stem, tab, pvec, meta)
    print(f'[INFO] Wrote {out_root / (stem + "_motif_class_association.csv")}')
    print(f'[INFO] Wrote {out_root / (stem + "_motif_pvalues.pt")}')


if __name__ == '__main__':
    main()
