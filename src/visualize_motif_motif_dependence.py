#!/usr/bin/env python
"""
Motif–motif dependence heatmap among the **top label-associated** motifs.

Loads the same split as ``compute_motif_class_association.py``, builds graph–motif
presence **X**, ranks motifs by Fisher *p*, BH *q*, or |log₂ OR|, takes the top **K**,
then plots a symmetric heatmap of pairwise **Phi** (Pearson correlation of binary
presence vectors) or **Jaccard** co-occurrence on graphs.

Typical use (from ``src/``)::

    python visualize_motif_motif_dependence.py \\
      --dataset Mutagenicity --fold 0 --which_split training \\
      --top_k 40 --out ../figures/motif_motif_top_label_assoc.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataLoader import CHOSEN_THRESHOLD, get_setup_files_with_folds
from utils.get_data_loaders import DATASET_COLUMN, DATASET_TYPE
from motif_class_association import association_table, build_graph_motif_presence

_DEFAULT_DICTIONARY_PATH = os.environ.get(
    'MOTIFSAT_DICTIONARY_PATH',
    '/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_CREATE',
)


def _odds_ratio_y1(a: float, b: float, c: float, d: float) -> float:
    eps = 0.5
    aa, bb, cc, dd = a + eps, b + eps, c + eps, d + eps
    return (aa * dd) / (bb * cc)


def _add_log2_or(tab: pd.DataFrame) -> pd.DataFrame:
    out = tab.copy()
    out['log2_or'] = out.apply(
        lambda r: math.log2(
            _odds_ratio_y1(
                r['contingency_a_absent_y0'],
                r['contingency_b_absent_y1'],
                r['contingency_c_present_y0'],
                r['contingency_d_present_y1'],
            )
        ),
        axis=1,
    )
    return out


def _rank_top_motifs(tab: pd.DataFrame, rank_by: str, top_k: int) -> pd.DataFrame:
    if tab.empty:
        return tab
    df = _add_log2_or(tab)
    if rank_by == 'fisher_p':
        df = df.sort_values('fisher_p', ascending=True, na_position='last')
    elif rank_by == 'fisher_q_bh':
        if 'fisher_q_bh' not in df.columns:
            df = df.sort_values('fisher_p', ascending=True, na_position='last')
        else:
            df = df.sort_values('fisher_q_bh', ascending=True, na_position='last')
    elif rank_by == 'abs_log2_or':
        df = df.assign(_a=np.abs(df['log2_or'])).sort_values('_a', ascending=False).drop(columns=['_a'])
    else:
        raise ValueError(rank_by)
    return df.head(int(top_k))


def _phi_matrix(X_sub: np.ndarray) -> np.ndarray:
    """Pairwise Pearson (Phi) correlation; constant columns -> 0 off-diagonal."""
    Xf = X_sub.astype(np.float64)
    if Xf.shape[1] <= 1:
        return np.ones((Xf.shape[1], Xf.shape[1]))
    C = np.corrcoef(Xf.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def _jaccard_matrix(X_sub: np.ndarray) -> np.ndarray:
    """Jaccard similarity |A∩B|/|A∪B| on sets of graphs where each motif is present."""
    V = X_sub.astype(np.float64)
    inter = V.T @ V
    s = V.sum(axis=0)
    union = s[:, None] + s[None, :] - inter
    J = inter / np.maximum(union, 1e-12)
    np.fill_diagonal(J, 1.0)
    return J


def _cluster_order(M: np.ndarray) -> np.ndarray:
    """Hierarchical clustering order (average linkage on 1 - |Phi|)."""
    n = M.shape[0]
    if n <= 1:
        return np.arange(n)
    D = 1.0 - np.abs(M)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='average')
    return np.array(leaves_list(Z))


def _motif_labels(
    rows: pd.DataFrame,
    max_len: int = 22,
    *,
    tick_style: str = 'two_line',
) -> list[str]:
    """
    tick_style:
      two_line — long SMILES split across two lines (no motif id).
      one_line — single line of truncated SMILES only.
      id_only — motif id only (dense grids; cross-ref CSV for SMILES).
    """
    if tick_style not in ('two_line', 'one_line', 'id_only'):
        raise ValueError(f'tick_style must be two_line|one_line|id_only, got {tick_style!r}')
    out: list[str] = []
    for _, r in rows.iterrows():
        mid = int(r['motif_id'])
        smi = r.get('motif_smiles')
        if tick_style == 'id_only':
            out.append(str(mid))
            continue
        if pd.isna(smi) or smi is None:
            out.append('—')
            continue
        t = str(smi)
        if tick_style == 'one_line':
            if len(t) <= max_len:
                out.append(t)
            else:
                out.append(t[: max_len - 1] + '…')
            continue
        # two_line: two chunks of SMILES, max_len ≈ chars per line
        if len(t) <= max_len:
            out.append(t)
        elif len(t) <= 2 * max_len:
            out.append(t[:max_len] + '\n' + t[max_len:])
        else:
            out.append(t[:max_len] + '\n' + t[max_len : 2 * max_len - 1] + '…')
    return out


def plot_motif_motif_heatmap(
    *,
    M: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
    fig_size: tuple[float, float],
    colorbar_label: str,
):
    n = M.shape[0]
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, constrained_layout=True)
    im = ax.imshow(M, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        labels, fontsize=6, rotation=90, ha='center', va='top', rotation_mode='anchor',
    )
    ax.set_yticklabels(labels, fontsize=6, ha='right')
    ax.tick_params(axis='x', which='major', pad=3, length=3)
    ax.tick_params(axis='y', which='major', pad=3, length=3)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, label=colorbar_label)
    ax.set_title(title, fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[INFO] Wrote {out_path.resolve()}')


def load_dataset_motif_matrix(
    *,
    dataset: str,
    fold: int,
    which_split: str,
    algorithm: str = 'BRICS',
    dictionary_path: str | None = None,
    dictionary_fold_variant: str = 'nofilter',
    csv_file: str | None = None,
    min_support: int = 5,
) -> tuple[np.ndarray, pd.DataFrame, dict[int, int], int, list | None]:
    """
    Returns:
        X, association_table, mid_to_col, n_graphs, motif_list (or None)
    """
    dictionary_path = dictionary_path or _DEFAULT_DICTIONARY_PATH
    ds = dataset
    algo = algorithm
    thr_key = CHOSEN_THRESHOLD.get('BRICS', {}).get(ds, 0.2)
    date_tag = f'{algo}{thr_key}'

    lookup, motif_list, *_rest = get_setup_files_with_folds(
        ds, date_tag, fold, algo,
        path=dictionary_path,
        dictionary_fold_variant=dictionary_fold_variant,
    )

    default_csv = (
        Path('/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DomainDrivenGlobalExpl/datasets/FOLDS')
        / f'{ds}_{fold}.csv'
    )
    csv_path = Path(csv_file) if csv_file else default_csv
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Fold CSV not found at {csv_path}. Pass csv_file=/path/to/{ds}_{fold}.csv',
        )

    df = pd.read_csv(csv_path)
    label_col = DATASET_COLUMN[ds][0] if isinstance(DATASET_COLUMN[ds], list) else DATASET_COLUMN[ds]

    if which_split == 'all':
        sub = df
    elif which_split == 'train_valid':
        sub = df[df['group'].isin(['training', 'valid'])]
    else:
        sub = df[df['group'] == which_split]

    smiles_list = sub['smiles'].values.tolist()
    y_raw = sub[label_col].values
    if DATASET_TYPE[ds] != 'BinaryClass':
        raise ValueError(f'Dataset {ds} is not BinaryClass.')

    y = np.asarray(y_raw).astype(int).ravel()
    if set(np.unique(y)) == {-1, 1}:
        y = (y > 0).astype(int)
    elif y.max() > 1 or y.min() < 0:
        y = (y > 0).astype(int)

    X, y2, motif_ids = build_graph_motif_presence(smiles_list, y, lookup)
    assert np.array_equal(y2, y)
    motif_smiles_list = list(motif_list) if motif_list is not None else None
    tab = association_table(X, y, motif_ids, motif_smiles_list, min_support=min_support)
    mid_to_col = {m: j for j, m in enumerate(motif_ids)}
    return X, tab, mid_to_col, len(smiles_list), motif_list


def load_split_and_matrix(args: argparse.Namespace):
    return load_dataset_motif_matrix(
        dataset=args.dataset,
        fold=args.fold,
        which_split=args.which_split,
        algorithm=args.algorithm,
        dictionary_path=args.dictionary_path,
        dictionary_fold_variant=args.dictionary_fold_variant,
        csv_file=args.csv_file,
        min_support=args.min_support,
    )[:4]


def main():
    p = argparse.ArgumentParser(description='Motif–motif dependence heatmap for top label-associated motifs')
    p.add_argument('--dataset', type=str, default='Mutagenicity')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--algorithm', type=str, default='BRICS')
    p.add_argument('--dictionary_path', type=str, default=None)
    p.add_argument('--dictionary_fold_variant', type=str, default='nofilter')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--csv_file', type=str, default=None)
    p.add_argument(
        '--which_split',
        type=str,
        default='training',
        choices=['training', 'valid', 'test', 'train_valid', 'all'],
    )
    p.add_argument('--min_support', type=int, default=5)
    p.add_argument('--top_k', type=int, default=40, help='Number of most label-associated motifs to include')
    p.add_argument(
        '--rank_by',
        type=str,
        default='fisher_p',
        choices=['fisher_p', 'fisher_q_bh', 'abs_log2_or'],
        help='How to rank motifs before taking top_k',
    )
    p.add_argument(
        '--dependence',
        type=str,
        default='phi',
        choices=['phi', 'jaccard'],
        help='phi = Pearson correlation of binary presence; jaccard = |A∩B|/|A∪B| on graphs',
    )
    p.add_argument(
        '--no_cluster',
        action='store_true',
        help='Keep motifs in label-association rank order (default: hierarchical cluster order)',
    )
    p.add_argument('--out', type=str, required=True, help='Output PNG path')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--fig_max', type=float, default=14.0, help='Max width/height in inches (figure is square)')
    args = p.parse_args()
    do_cluster = not args.no_cluster

    X, tab, mid_to_col, n_graphs = load_split_and_matrix(args)
    if tab.empty:
        raise SystemExit('Empty association table; lower --min_support or check data.')

    top = _rank_top_motifs(tab, args.rank_by, args.top_k)
    if top.empty:
        raise SystemExit('No motifs after ranking.')

    cols = []
    kept_rows = []
    for _, r in top.iterrows():
        mid = int(r['motif_id'])
        if mid not in mid_to_col:
            continue
        cols.append(mid_to_col[mid])
        kept_rows.append(r)
    if not cols:
        raise SystemExit('No overlapping motif columns for top motifs (unexpected).')

    top = pd.DataFrame(kept_rows)
    X_sub = X[:, cols]
    if args.dependence == 'phi':
        M = _phi_matrix(X_sub)
        cbar = r'$\Phi$ (Pearson corr. of presence)'
        vmin, vmax = -1.0, 1.0
        cmap = 'RdBu_r'
    else:
        M = _jaccard_matrix(X_sub)
        cbar = 'Jaccard similarity'
        vmin, vmax = 0.0, 1.0
        cmap = 'viridis'

    order = np.arange(M.shape[0])
    if do_cluster and M.shape[0] > 2:
        order = _cluster_order(M)
    M = M[order][:, order]
    labels = _motif_labels(top.iloc[order])

    title = (
        f'{args.dataset} fold{args.fold} {args.which_split}  |  n_graphs={n_graphs}  |  '
        f'top {len(cols)} motifs by {args.rank_by}\n'
        f'{args.dependence} co-occurrence across graphs'
        + (' (clustered)' if do_cluster else '')
    )

    n_m = len(labels)
    fig_size = (min(args.fig_max, 4.0 + 0.22 * n_m), min(args.fig_max, 4.0 + 0.22 * n_m))

    plot_motif_motif_heatmap(
        M=M,
        labels=labels,
        title=title,
        out_path=Path(args.out),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        dpi=args.dpi,
        fig_size=fig_size,
        colorbar_label=cbar,
    )


if __name__ == '__main__':
    main()
