#!/usr/bin/env python
"""
Single entry point: **compare** motif–class association views across **multiple** datasets.

Produces (under ``--out_dir``):

1. ``compare_descriptive.png`` — same four row types side-by-side (one column per dataset):
   summary + label bar, prevalence histogram, significance histogram, volcano (shared y-scale
   on volcanos for comparability).
2. ``compare_motif_motif_<dependence>.png`` — motif–motif dependence heatmaps (Φ or Jaccard),
   one panel per dataset, **shared** color limits.

For the descriptive figure, expects ``*_motif_class_association.csv`` under
``<data_dir>/motif_association/``. If a file is missing, this script **runs**
``compute_motif_class_association.py`` for that dataset (unless ``--no-compute-association``).
Heatmaps load **live** graph–motif presence (see ``visualize_motif_motif_dependence.py``).

Example::

    python visualize_motif_association_compare.py \\
      --dataset Mutagenicity --dataset BBBP \\
      --fold 0 --which_split training \\
      --out_dir ../figures/motif_compare

Heatmap ticks default to ``--heatmap_tick_style one_line`` (truncated **SMILES only**, no motif id).
Use ``--heatmap_tick_style id_only`` for dense grids. Tune layout with ``--descriptive_fig_height`` and
``--volcano_annotate_fontsize``.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_motif_discrimination import (
    annotate_volcano_fisher_p,
    dataset_class_counts,
    prepare_association_df,
    _lor_color,
)
from visualize_motif_motif_dependence import (
    _cluster_order,
    _jaccard_matrix,
    _motif_labels,
    _phi_matrix,
    _rank_top_motifs,
    load_dataset_motif_matrix,
)


def default_association_csv(data_dir: Path, dataset: str, fold: int, which_split: str) -> Path:
    return data_dir / 'motif_association' / f'{dataset}_fold{fold}_{which_split}_motif_class_association.csv'


def ensure_association_csv(
    *,
    dataset: str,
    csv_path: Path,
    data_dir: Path,
    fold: int,
    which_split: str,
    algorithm: str,
    dictionary_path: str | None,
    dictionary_fold_variant: str,
    folds_csv: str | None,
    min_support: int,
) -> None:
    """Run ``compute_motif_class_association.py`` if ``csv_path`` is missing."""
    if csv_path.is_file():
        return

    src_dir = Path(__file__).resolve().parent
    compute_script = src_dir / 'compute_motif_class_association.py'
    if not compute_script.is_file():
        raise FileNotFoundError(
            f'Missing association CSV {csv_path} and compute script not found at {compute_script}',
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        sys.executable,
        str(compute_script),
        '--dataset',
        dataset,
        '--fold',
        str(fold),
        '--which_split',
        which_split,
        '--data_dir',
        str(data_dir),
        '--algorithm',
        algorithm,
        '--dictionary_fold_variant',
        dictionary_fold_variant,
        '--min_support',
        str(min_support),
    ]
    if dictionary_path:
        cmd.extend(['--dictionary_path', dictionary_path])
    if folds_csv:
        cmd.extend(['--csv_file', folds_csv])

    print('=' * 60)
    print(f'  Computing motif–class association → {csv_path}')
    print('=' * 60)
    r = subprocess.run(cmd, cwd=str(src_dir))
    if r.returncode != 0:
        raise RuntimeError(
            f'compute_motif_class_association failed (exit {r.returncode}). '
            'Check FOLDS CSV / dictionary paths, or pass --csv_file to your fold CSV.',
        )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Association CSV still missing after compute: {csv_path}',
        )


def _draw_compare_summary_panel(
    ax_sum,
    *,
    name: str,
    title_suffix: str,
    n_g: int,
    n0: int,
    n1: int,
    n_motifs: int,
    min_count: int,
) -> None:
    """First row of compare_descriptive: dataset title, full-width stats, then a wide short bar below."""
    ax_sum.set_xlim(0, 1)
    ax_sum.set_ylim(0, 1)
    ax_sum.axis('off')

    ts = title_suffix.strip()
    title = f'{name}{(" · " + ts) if ts else ""}'

    ax_sum.text(
        0.5,
        0.98,
        title,
        ha='center',
        va='top',
        fontsize=11,
        fontweight='600',
        color='#1a237e',
        transform=ax_sum.transAxes,
    )

    pct0 = 100.0 * n0 / max(n_g, 1)
    pct1 = 100.0 * n1 / max(n_g, 1)
    stats = (
        f'Graphs  {n_g:,}\n'
        f'Negative (y=0)  {n0:,}  ({pct0:.1f}%)\n'
        f'Positive (y=1)  {n1:,}  ({pct1:.1f}%)\n'
        f'Motifs in table  {n_motifs:,}  ·  min n_present ≥ {min_count}'
    )
    # Centered block above the bar — uses nearly full width so lines are not clipped.
    ax_sum.text(
        0.5,
        0.78,
        stats,
        ha='center',
        va='top',
        fontsize=9,
        linespacing=1.45,
        color='#263238',
        transform=ax_sum.transAxes,
    )

    # Short, wide stacked bar below all text: [left, bottom, width, height] in parent axes coords.
    ax_bar = ax_sum.inset_axes([0.04, 0.06, 0.92, 0.20])
    ax_bar.set_facecolor('#fafafa')
    ax_bar.barh([0], [n0 / n_g], left=0.0, height=0.45, color='#546e7a', edgecolor='white', linewidth=0.8)
    ax_bar.barh([0], [n1 / n_g], left=n0 / n_g, height=0.45, color='#ef6c00', edgecolor='white', linewidth=0.8)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.65, 0.65)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.tick_params(axis='x', labelsize=7, colors='#424242')
    ax_bar.set_xlabel('Class fraction', fontsize=7.5, color='#424242', labelpad=4)
    ax_bar.set_title('Label balance', fontsize=8.5, color='#37474f', pad=8)


def _sig_array(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, str]:
    if value_col == 'neglog10_p':
        return df['neglog10_p'].values, r'$-\log_{10}(p_\mathrm{Fisher})$'
    if value_col == 'neglog10_q':
        if 'neglog10_q' not in df.columns or df['neglog10_q'].isna().all():
            return df['neglog10_p'].values, r'$-\log_{10}(p)$ (q N/A)'
        return df['neglog10_q'].values, r'$-\log_{10}(q_\mathrm{BH})$'
    raise ValueError(value_col)


def plot_compare_descriptive(
    *,
    columns: list[tuple[str, Path]],
    out_path: Path,
    min_count: int,
    max_motifs: int | None,
    value_col: str,
    clip_log2: float,
    dpi: int,
    fig_width_per_col: float,
    fig_height: float,
    title_suffix: str,
    annotate_min_neglog10_p: float | None = 10.0,
    volcano_annotate_fontsize: float = 4.5,
):
    n = len(columns)
    if n == 0:
        raise SystemExit('No datasets.')

    prepared: list[tuple[str, pd.DataFrame]] = []
    for name, path in columns:
        df = prepare_association_df(path, min_count=min_count, max_motifs=max_motifs)
        prepared.append((name, df))

    ymax = -math.log10(0.05)
    for _, df in prepared:
        if df.empty:
            continue
        sig, _ = _sig_array(df, value_col)
        if len(sig):
            ymax = max(ymax, float(np.nanmax(sig)) * 1.08)

    ymax = max(ymax, 0.5)
    ts = (' ' + title_suffix) if title_suffix else ''

    cmap_pos = '#c62828'
    cmap_neg = '#1565c0'
    cmap_neu = '#78909c'

    fig_w = fig_width_per_col * n
    # Taller volcano row than histograms so scatter is readable; summary row sized for text + label bar.
    fig, axes = plt.subplots(
        4,
        n,
        figsize=(fig_w, fig_height),
        squeeze=False,
        gridspec_kw={
            'height_ratios': [0.95, 1.0, 1.0, 2.65],
            'hspace': 0.32,
            'wspace': 0.32,
        },
    )
    if n == 1:
        axes = np.asarray(axes)

    for j, ((name, path), (_, df)) in enumerate(zip(columns, prepared)):
        ax_sum = axes[0, j]
        ax_hp = axes[1, j]
        ax_hs = axes[2, j]
        ax_vol = axes[3, j]

        if df.empty:
            ax_sum.axis('off')
            ax_sum.text(0.5, 0.5, f'No rows\n{name}{ts}', ha='center', va='center', transform=ax_sum.transAxes)
            ax_hp.set_visible(False)
            ax_hs.set_visible(False)
            ax_vol.set_visible(False)
            continue

        n_g, n0, n1 = dataset_class_counts(df.iloc[0])
        _draw_compare_summary_panel(
            ax_sum,
            name=name,
            title_suffix=title_suffix,
            n_g=n_g,
            n0=n0,
            n1=n1,
            n_motifs=len(df),
            min_count=min_count,
        )

        prev = df['n_present'].values.astype(float)
        ax_hp.hist(
            prev, bins=min(48, max(12, int(np.sqrt(len(prev)) + 5))),
            color='#455a64', edgecolor='#263238', linewidth=0.55, alpha=0.92,
        )
        ax_hp.set_yscale('symlog', linthresh=1.0)
        ax_hp.set_xlabel(r'$n_{\mathrm{present}}$', fontsize=8)
        ax_hp.set_ylabel('Motif count (symlog y)', fontsize=8)
        ax_hp.set_title('Prevalence', fontsize=9)

        sig, sig_lab = _sig_array(df, value_col)
        ax_hs.hist(
            sig, bins=min(40, max(15, len(df) // 30 + 10)),
            color='#6a1b9a', edgecolor='#38006b', linewidth=0.55, alpha=0.92,
        )
        ax_hs.set_yscale('symlog', linthresh=1.0)
        ax_hs.set_xlabel(sig_lab, fontsize=8)
        ax_hs.set_ylabel('Motif count (symlog y)', fontsize=8)
        ax_hs.set_title(r'Significance ($-\log_{10}$)', fontsize=9)

        lor = df['log2_or'].values.astype(float)
        lor_plot = np.clip(lor, -clip_log2, clip_log2)
        colors = [_lor_color(float(x)) for x in lor]
        ax_vol.scatter(
            lor_plot, sig, c=colors, s=np.clip(8000.0 / max(len(df), 1), 3.0, 26.0),
            alpha=0.45, linewidths=0, rasterized=True,
        )
        annotate_volcano_fisher_p(
            ax_vol,
            df,
            lor_plot,
            sig,
            annotate_min_neglog10_p=annotate_min_neglog10_p,
            fontsize=volcano_annotate_fontsize,
        )
        ax_vol.axhline(-math.log10(0.05), color='#37474f', linestyle='--', linewidth=0.85, alpha=0.75)
        ax_vol.axvline(0.0, color='#90a4ae', linestyle='-', linewidth=0.55, alpha=0.8)
        ax_vol.set_xlim(-clip_log2 * 1.02, clip_log2 * 1.02)
        ax_vol.set_ylim(0.0, ymax)
        ax_vol.set_xlabel(r'$\log_2(\mathrm{OR})$', fontsize=8)
        ax_vol.set_ylabel(sig_lab, fontsize=8)
        ax_vol.set_title('Volcano (all motifs)', fontsize=9)
        if j == n - 1:
            leg = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_pos, markersize=6, label='enr. y=1'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_neg, markersize=6, label='dep. y=1'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_neu, markersize=6, label='neutral'),
            ]
            ax_vol.legend(handles=leg, loc='upper right', fontsize=6, framealpha=0.92)

    fig.suptitle(
        'Motif–class association compared across datasets (BRICS)',
        fontsize=12,
        fontweight='600',
        y=0.995,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f'[INFO] Wrote {out_path.resolve()}')


def plot_compare_motif_motif(
    *,
    datasets: list[str],
    out_path: Path,
    fold: int,
    which_split: str,
    algorithm: str,
    dictionary_path: str | None,
    dictionary_fold_variant: str,
    csv_file: str | None,
    min_support: int,
    top_k: int,
    rank_by: str,
    dependence: str,
    no_cluster: bool,
    dpi: float,
    fig_h_per_panel: float,
    heatmap_tick_style: str,
):
    n = len(datasets)
    if n == 0:
        raise SystemExit('No datasets.')

    mats: list[np.ndarray] = []
    labels_list: list[list[str]] = []
    titles: list[str] = []

    do_cluster = not no_cluster
    for ds in datasets:
        X, tab, mid_to_col, n_g, _ml = load_dataset_motif_matrix(
            dataset=ds,
            fold=fold,
            which_split=which_split,
            algorithm=algorithm,
            dictionary_path=dictionary_path,
            dictionary_fold_variant=dictionary_fold_variant,
            csv_file=csv_file,
            min_support=min_support,
        )
        if tab.empty:
            mats.append(np.array([[1.0]]))
            labels_list.append(['—'])
            titles.append(f'{ds}\n(no motifs)')
            continue

        top = _rank_top_motifs(tab, rank_by, top_k)
        cols: list[int] = []
        kept: list = []
        for _, r in top.iterrows():
            mid = int(r['motif_id'])
            if mid not in mid_to_col:
                continue
            cols.append(mid_to_col[mid])
            kept.append(r)
        if not cols:
            mats.append(np.array([[1.0]]))
            labels_list.append(['—'])
            titles.append(f'{ds}\n(no overlap)')
            continue

        top_df = pd.DataFrame(kept)
        X_sub = X[:, cols]
        if dependence == 'phi':
            M = _phi_matrix(X_sub)
        else:
            M = _jaccard_matrix(X_sub)

        order = np.arange(M.shape[0])
        if do_cluster and M.shape[0] > 2:
            order = _cluster_order(M)
        M = M[order][:, order]
        if heatmap_tick_style == 'one_line':
            ml = 32
        elif heatmap_tick_style == 'id_only':
            ml = 14
        else:
            ml = 14
        labels_list.append(
            _motif_labels(top_df.iloc[order], max_len=ml, tick_style=heatmap_tick_style),
        )
        mats.append(M)
        sub = (
            f'n={n_g:,} graphs · top {M.shape[0]} by {rank_by} · {dependence}'
            + (' · clustered' if do_cluster else '')
        )
        id_hint = (
            '\nAxes: motif id — full SMILES in association CSV'
            if heatmap_tick_style == 'id_only'
            else ''
        )
        titles.append(f'{ds}\n{sub}{id_hint}')

    if dependence == 'phi':
        vmin, vmax = -1.0, 1.0
        cmap_name = 'RdBu_r'
        cbar_lbl = r'$\Phi$'
    else:
        vmin, vmax = 0.0, 1.0
        cmap_name = 'viridis'
        cbar_lbl = 'Jaccard'

    max_w = max(m.shape[0] for m in mats)
    fig_w = min(4.2 + 1.18 * max_w, 20.0) * n * 0.52
    fig_h = min(fig_h_per_panel + 0.16 * max_w, 22.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    axr = axes[0, :] if n > 1 else [axes[0, 0]]

    for ax, M, lab, tit in zip(axr, mats, labels_list, titles):
        im = ax.imshow(M, cmap=cmap_name, aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(M.shape[0]))
        ax.set_yticks(np.arange(M.shape[0]))
        fs = max(4, min(8, int(560 // max(M.shape[0], 1))))
        # Keep labels outside cells: y right-aligned; x labels below tick line (bottom axis).
        ax.set_xticklabels(
            lab, fontsize=fs, rotation=90, ha='center', va='top', rotation_mode='anchor',
        )
        ax.set_yticklabels(lab, fontsize=fs, ha='right')
        ax.tick_params(axis='x', which='major', pad=6, length=3)
        ax.tick_params(axis='y', which='major', pad=4, length=3)
        ax.set_title(tit, fontsize=9, linespacing=1.15)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.06, label=cbar_lbl)

    fig.suptitle(
        'Motif–motif dependence (top label-associated motifs)',
        fontsize=11,
        y=0.998,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.06, facecolor='white')
    plt.close(fig)
    print(f'[INFO] Wrote {out_path.resolve()}')


def main():
    p = argparse.ArgumentParser(
        description='Compare motif association descriptive plots and heatmaps across datasets',
    )
    p.add_argument('--dataset', action='append', default=[], metavar='NAME', help='Repeat for each dataset')
    p.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list (alternative to repeating --dataset)',
    )
    p.add_argument('--fold', type=int, default=0)
    p.add_argument(
        '--which_split',
        type=str,
        default='training',
        choices=['training', 'valid', 'test', 'train_valid', 'all'],
    )
    p.add_argument('--data_dir', type=str, default=None, help='Project data dir (default: ../data)')
    p.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Output directory (default: <data_dir>/motif_association_compare)',
    )
    p.add_argument('--algorithm', type=str, default='BRICS')
    p.add_argument('--dictionary_path', type=str, default=None)
    p.add_argument('--dictionary_fold_variant', type=str, default='nofilter')
    p.add_argument('--csv_file', type=str, default=None, help='Single FOLDS CSV for all datasets (unusual)')
    p.add_argument('--min_support', type=int, default=5)
    p.add_argument('--min_count', type=int, default=5, help='For descriptive CSV filter (n_present)')
    p.add_argument('--max-motifs', type=int, default=None, help='Optional cap on motifs in descriptive plots')
    p.add_argument(
        '--value',
        type=str,
        default='neglog10_p',
        choices=['neglog10_p', 'neglog10_q'],
    )
    p.add_argument('--clip-log2', type=float, default=6.0)
    p.add_argument('--top_k', type=int, default=32, help='Motifs in each heatmap panel')
    p.add_argument(
        '--rank_by',
        type=str,
        default='fisher_p',
        choices=['fisher_p', 'fisher_q_bh', 'abs_log2_or'],
    )
    p.add_argument('--dependence', type=str, default='phi', choices=['phi', 'jaccard'])
    p.add_argument(
        '--heatmap_tick_style',
        type=str,
        default='one_line',
        choices=['id_only', 'one_line', 'two_line'],
        help='Heatmap ticks: one_line (truncated SMILES only, default), two_line, or id_only',
    )
    p.add_argument(
        '--fig_h_per_panel',
        type=float,
        default=7.5,
        help='Height per heatmap panel in compare_motif_motif_*.png',
    )
    p.add_argument('--no_cluster', action='store_true')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument(
        '--fig_width_per_col',
        type=float,
        default=4.75,
        help='Width (inches) per dataset column in compare_descriptive.png',
    )
    p.add_argument(
        '--descriptive_fig_height',
        type=float,
        default=15.0,
        help='Total figure height for compare_descriptive.png (volcano row scales with this)',
    )
    p.add_argument(
        '--volcano_annotate_fontsize',
        type=float,
        default=4.5,
        help='Font size for volcano point labels in compare_descriptive (smaller = less overlap)',
    )
    p.add_argument('--title_suffix', type=str, default='')
    p.add_argument(
        '--annotate-min-neglog10-p',
        type=float,
        default=10.0,
        help='Volcano: label motifs with −log10(Fisher p) ≥ this (0 disables)',
    )
    p.add_argument(
        '--no-compute-association',
        action='store_true',
        help='If association CSV is missing, exit with error instead of running compute_motif_class_association.py',
    )
    p.add_argument(
        '--skip_descriptive',
        action='store_true',
        help='Only write motif–motif comparison (skips descriptive plot and CSV check)',
    )
    p.add_argument(
        '--skip_heatmap',
        action='store_true',
        help='Only write descriptive comparison',
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir or (Path(__file__).resolve().parent.parent / 'data'))
    out_dir = Path(args.out_dir) if args.out_dir else (data_dir / 'motif_association_compare')
    out_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    if args.datasets:
        names.extend(s.strip() for s in args.datasets.split(',') if s.strip())
    names.extend(args.dataset)
    if not names:
        raise SystemExit('Pass --dataset NAME one or more times or --datasets A,B,C')
    names = list(dict.fromkeys(names))

    columns: list[tuple[str, Path]] = []
    if not args.skip_descriptive:
        folds_for_compute = args.csv_file if len(names) == 1 else None
        for ds in names:
            csv_assoc = default_association_csv(data_dir, ds, args.fold, args.which_split)
            if not csv_assoc.is_file():
                if args.no_compute_association:
                    raise FileNotFoundError(
                        f'Missing association CSV: {csv_assoc}\n'
                        f'Run: python compute_motif_class_association.py --dataset {ds} --fold {args.fold} '
                        f'--which_split {args.which_split} --data_dir {data_dir}',
                    )
                ensure_association_csv(
                    dataset=ds,
                    csv_path=csv_assoc,
                    data_dir=data_dir,
                    fold=args.fold,
                    which_split=args.which_split,
                    algorithm=args.algorithm,
                    dictionary_path=args.dictionary_path,
                    dictionary_fold_variant=args.dictionary_fold_variant,
                    folds_csv=folds_for_compute,
                    min_support=args.min_support,
                )
            columns.append((ds, csv_assoc))

    if not args.skip_descriptive:
        ann = None if args.annotate_min_neglog10_p <= 0 else float(args.annotate_min_neglog10_p)
        plot_compare_descriptive(
            columns=columns,
            out_path=out_dir / 'compare_descriptive.png',
            min_count=args.min_count,
            max_motifs=args.max_motifs,
            value_col=args.value,
            clip_log2=args.clip_log2,
            dpi=args.dpi,
            fig_width_per_col=args.fig_width_per_col,
            fig_height=args.descriptive_fig_height,
            title_suffix=args.title_suffix,
            annotate_min_neglog10_p=ann,
            volcano_annotate_fontsize=args.volcano_annotate_fontsize,
        )

    if not args.skip_heatmap:
        csv_for_heat = args.csv_file if len(names) == 1 else None
        if args.csv_file and len(names) > 1:
            print('[WARN] Ignoring --csv_file for heatmaps when multiple datasets (using per-dataset FOLDS paths).')
        plot_compare_motif_motif(
            datasets=names,
            out_path=out_dir / f'compare_motif_motif_{args.dependence}.png',
            fold=args.fold,
            which_split=args.which_split,
            algorithm=args.algorithm,
            dictionary_path=args.dictionary_path,
            dictionary_fold_variant=args.dictionary_fold_variant,
            csv_file=csv_for_heat,
            min_support=args.min_support,
            top_k=args.top_k,
            rank_by=args.rank_by,
            dependence=args.dependence,
            no_cluster=args.no_cluster,
            dpi=args.dpi,
            fig_h_per_panel=args.fig_h_per_panel,
            heatmap_tick_style=args.heatmap_tick_style,
        )


if __name__ == '__main__':
    main()
