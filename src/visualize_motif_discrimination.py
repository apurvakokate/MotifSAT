#!/usr/bin/env python
"""
Descriptive plots for class–motif association tables (Fisher / BH-FDR).

Input: CSV files from ``compute_motif_class_association.py`` (*_motif_class_association.csv).

For each ``--panel`` dataset, produces **one vertical block** with:

1. **Dataset summary** — graph count, class balance (from contingency totals), motif count after filters.
2. **Motif prevalence** — histogram of ``n_present`` (how often each motif appears in the split).
3. **Significance distribution** — histogram of :math:`-\\log_{10}(p)` or :math:`-\\log_{10}(q)`.
4. **Volcano** — **every motif** as a point: x = :math:`\\log_2` odds ratio (y=1 | present vs absent),
   y = :math:`-\\log_{10}(p)` (or BH q). Motifs with :math:`-\\log_{10}(p_\\mathrm{Fisher}) \\geq 10` get a
   text label (truncated SMILES only; see ``--annotate-min-neglog10-p``; use ``0`` to turn off).

Example::

    python visualize_motif_discrimination.py \\
      --panel Mutagenicity=../data/motif_association/Mutagenicity_fold0_training_motif_class_association.csv \\
      --out ../figures/motif_association_descriptive.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def odds_ratio_y1_enrichment(a: float, b: float, c: float, d: float) -> float:
    """
    Contingency (see motif_class_association):
              y=0   y=1
    absent      a     b
    present     c     d
    OR = (d/c) / (b/a) = (a*d) / (b*c).
    """
    eps = 0.5
    aa, bb, cc, dd = a + eps, b + eps, c + eps, d + eps
    return (aa * dd) / (bb * cc)


def _neglog10(p: float, cap: float = 50.0) -> float:
    if p is None or not math.isfinite(float(p)) or float(p) <= 0:
        return 0.0
    return min(-math.log10(float(p)), cap)


def dataset_class_counts(row: pd.Series) -> tuple[int, int, int]:
    """Return (n_graphs, n_y0, n_y1); same for every motif row in a valid association CSV."""
    a = int(row['contingency_a_absent_y0'])
    b = int(row['contingency_b_absent_y1'])
    c = int(row['contingency_c_present_y0'])
    d = int(row['contingency_d_present_y1'])
    n_y0 = a + c
    n_y1 = b + d
    n = n_y0 + n_y1
    if 'n_graphs' in row.index and pd.notna(row['n_graphs']) and int(row['n_graphs']) != n:
        warnings.warn(
            f'n_graphs={row["n_graphs"]} != a+b+c+d={n}; using contingency sum.',
            stacklevel=2,
        )
    return n, n_y0, n_y1


def prepare_association_df(
    csv_path: Path,
    *,
    min_count: int,
    max_motifs: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    need = [
        'motif_id', 'fisher_p', 'contingency_a_absent_y0', 'contingency_b_absent_y1',
        'contingency_c_present_y0', 'contingency_d_present_y1',
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f'{csv_path}: missing column {c!r}')
    df = df.copy()
    df['neglog10_p'] = df['fisher_p'].map(_neglog10)
    if 'fisher_q_bh' in df.columns:
        df['neglog10_q'] = df['fisher_q_bh'].map(_neglog10)
    else:
        df['neglog10_q'] = np.nan

    df['log2_or'] = df.apply(
        lambda r: math.log2(
            odds_ratio_y1_enrichment(
                r['contingency_a_absent_y0'],
                r['contingency_b_absent_y1'],
                r['contingency_c_present_y0'],
                r['contingency_d_present_y1'],
            )
        ),
        axis=1,
    )
    df['n_present'] = df['n_present'].astype(int)
    df = df[df['n_present'] >= min_count].sort_values('motif_id', ascending=True)
    if max_motifs is not None and len(df) > max_motifs:
        warnings.warn(
            f'{csv_path}: subsampling {max_motifs} of {len(df)} motifs (--max-motifs).',
            stacklevel=2,
        )
        df = df.sample(n=max_motifs, random_state=0).sort_values('motif_id', ascending=True)
    return df


def _lor_color(lor: float) -> str:
    if lor > 0.05:
        return '#c62828'
    if lor < -0.05:
        return '#1565c0'
    return '#78909c'


def _volcano_motif_label(row: pd.Series, max_len: int = 26) -> str:
    smi = row.get('motif_smiles')
    if pd.isna(smi) or smi is None or str(smi).strip() == '':
        return '—'
    s = str(smi).strip()
    if len(s) > max_len:
        s = s[: max_len - 1] + '…'
    return s


def annotate_volcano_fisher_p(
    ax,
    df: pd.DataFrame,
    lor_plot: np.ndarray,
    sig: np.ndarray,
    *,
    annotate_min_neglog10_p: float | None,
    fontsize: float = 5.0,
) -> None:
    """Draw text for points with ``−log10(Fisher p)`` ≥ threshold (Fisher only; not BH q)."""
    if annotate_min_neglog10_p is None or annotate_min_neglog10_p <= 0:
        return
    thr = float(annotate_min_neglog10_p)
    for i, (_, row) in enumerate(df.iterrows()):
        nlp = float(row['neglog10_p'])
        if not math.isfinite(nlp) or nlp < thr:
            continue
        lx = float(lor_plot[i])
        py = float(sig[i])
        if not math.isfinite(lx) or not math.isfinite(py):
            continue
        lbl = _volcano_motif_label(row)
        ax.annotate(
            lbl,
            (lx, py),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=fontsize,
            color='#111111',
            alpha=0.92,
            bbox=dict(
                boxstyle='round,pad=0.15',
                facecolor='white',
                edgecolor='#bdbdbd',
                linewidth=0.35,
                alpha=0.9,
            ),
            zorder=6,
        )


def plot_descriptive_panels(
    panels: list[tuple[str, Path]],
    out_path: Path,
    *,
    min_count: int,
    max_motifs: int | None,
    value_col: str,
    clip_log2: float,
    fig_width: float,
    dpi: int,
    title_suffix: str,
    annotate_min_neglog10_p: float | None = 10.0,
):
    n = len(panels)
    if n == 0:
        raise SystemExit('No --panel entries.')

    cmap_pos = '#c62828'
    cmap_neg = '#1565c0'
    cmap_neu = '#78909c'

    fig_h = 5.4 * n + 0.6
    fig = plt.figure(figsize=(fig_width, fig_h), constrained_layout=False)
    outer = fig.add_gridspec(n, 1, hspace=0.42, top=0.93, bottom=0.05, left=0.09, right=0.97)

    for i, (name, path) in enumerate(panels):
        inner = outer[i].subgridspec(
            3, 2,
            height_ratios=[0.2, 0.42, 1.05],
            width_ratios=[1.0, 1.0],
            hspace=0.4,
            wspace=0.28,
        )
        ax_sum = fig.add_subplot(inner[0, :])
        ax_hist_prev = fig.add_subplot(inner[1, 0])
        ax_hist_sig = fig.add_subplot(inner[1, 1])
        ax_vol = fig.add_subplot(inner[2, :])

        df = prepare_association_df(path, min_count=min_count, max_motifs=max_motifs)
        ts = (' ' + title_suffix) if title_suffix else ''

        if df.empty:
            ax_sum.axis('off')
            ax_sum.text(0.5, 0.5, f'No rows after filters\n{name}{ts}', ha='center', va='center')
            for ax in (ax_hist_prev, ax_hist_sig, ax_vol):
                ax.set_visible(False)
            continue

        n_g, n0, n1 = dataset_class_counts(df.iloc[0])
        summary = (
            f'{name}{ts}\n'
            f'Graphs: {n_g}  |  Class 0: {n0} ({100.0 * n0 / max(n_g, 1):.1f}%)  '
            f'|  Class 1: {n1} ({100.0 * n1 / max(n_g, 1):.1f}%)\n'
            f'Motifs plotted: {len(df)}  (min n_present ≥ {min_count})'
        )
        ax_sum.axis('off')
        ax_sum.text(0.0, 0.5, summary, ha='left', va='center', fontsize=10, family='monospace')
        # Class balance bar (full width under text — small inset)
        ax_bar = ax_sum.inset_axes([0.58, 0.12, 0.4, 0.35])
        ax_bar.barh(
            [0], [n0 / n_g], left=0.0, height=0.85, color='#546e7a', label='y=0', edgecolor='white',
        )
        ax_bar.barh(
            [0], [n1 / n_g], left=n0 / n_g, height=0.85, color='#ef6c00', label='y=1', edgecolor='white',
        )
        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0, 0.5, 1.0])
        ax_bar.set_xlabel('Class fraction', fontsize=7)
        ax_bar.set_title('Label distribution', fontsize=8)

        prev = df['n_present'].values.astype(float)
        ax_hist_prev.hist(
            prev, bins=min(48, max(12, int(np.sqrt(len(prev)) + 5))),
            color='#455a64', edgecolor='#263238', linewidth=0.55, alpha=0.92,
        )
        ax_hist_prev.set_yscale('symlog', linthresh=1.0)
        ax_hist_prev.set_xlabel(r'$n_{\mathrm{present}}$ (graphs with motif)', fontsize=8)
        ax_hist_prev.set_ylabel('Motif count (symlog y)', fontsize=8)
        ax_hist_prev.set_title('Prevalence of motifs', fontsize=9)

        if value_col == 'neglog10_p':
            sig = df['neglog10_p'].values
            sig_lab = r'$-\log_{10}(p_\mathrm{Fisher})$'
        elif value_col == 'neglog10_q':
            if 'neglog10_q' not in df.columns or df['neglog10_q'].isna().all():
                sig = df['neglog10_p'].values
                sig_lab = r'$-\log_{10}(p)$ (q N/A)'
            else:
                sig = df['neglog10_q'].values
                sig_lab = r'$-\log_{10}(q_\mathrm{BH})$'
        else:
            raise ValueError(value_col)

        ax_hist_sig.hist(
            sig, bins=min(40, max(15, len(df) // 30 + 10)),
            color='#6a1b9a', edgecolor='#38006b', linewidth=0.55, alpha=0.92,
        )
        ax_hist_sig.set_yscale('symlog', linthresh=1.0)
        ax_hist_sig.set_xlabel(sig_lab, fontsize=8)
        ax_hist_sig.set_ylabel('Motif count (symlog y)', fontsize=8)
        ax_hist_sig.set_title('Significance across all motifs', fontsize=9)

        lor = df['log2_or'].values.astype(float)
        lor_plot = np.clip(lor, -clip_log2, clip_log2)
        colors = [_lor_color(float(x)) for x in lor]
        y_sig = sig
        ax_vol.scatter(
            lor_plot, y_sig, c=colors, s=np.clip(8000.0 / max(len(df), 1), 3.0, 28.0),
            alpha=0.45, linewidths=0, edgecolors='none', rasterized=True,
        )
        annotate_volcano_fisher_p(
            ax_vol, df, lor_plot, y_sig, annotate_min_neglog10_p=annotate_min_neglog10_p,
        )
        ax_vol.axhline(-math.log10(0.05), color='#37474f', linestyle='--', linewidth=0.9, alpha=0.75, label=r'$p=0.05$')
        ax_vol.axvline(0.0, color='#90a4ae', linestyle='-', linewidth=0.6, alpha=0.8)
        ax_vol.set_xlim(-clip_log2 * 1.02, clip_log2 * 1.02)
        ymax = max(float(np.nanmax(y_sig)) if len(y_sig) else 0.0, -math.log10(0.05)) * 1.08
        ax_vol.set_ylim(0.0, max(ymax, 0.5))
        ax_vol.set_xlabel(r'$\log_2(\mathrm{OR})$ for $y{=}1$ | motif (clipped to $\pm$' + f'{clip_log2})', fontsize=9)
        ax_vol.set_ylabel(sig_lab, fontsize=9)
        ax_vol.set_title(
            f'All motifs (volcano){ts}\n'
            f'red: higher $y{{=}}1$ odds when present; blue: lower; gray: $|\\log_2 \\mathrm{{OR}}|\\lesssim 0$',
            fontsize=9,
        )
        leg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_pos, markersize=7, label='enriched class 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_neg, markersize=7, label='depleted class 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_neu, markersize=7, label='~neutral'),
        ]
        ax_vol.legend(handles=leg, loc='upper right', fontsize=7, framealpha=0.9)

    fig.suptitle(
        'Motif–class association: dataset distribution and all motifs (BRICS vocabulary)',
        fontsize=11,
        y=0.995,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[INFO] Wrote {out_path.resolve()}')


def main():
    p = argparse.ArgumentParser(
        description='Dataset-level and motif-level descriptive plots (all motifs, volcano + histograms)',
    )
    p.add_argument(
        '--panel',
        action='append',
        metavar='NAME=CSV',
        default=[],
        help='Repeat per dataset: Name=/path/to/*_motif_class_association.csv',
    )
    p.add_argument('--out', type=str, required=True, help='Output PNG path')
    p.add_argument(
        '--max-motifs',
        type=int,
        default=None,
        help='Optional cap on number of motifs (subsample; default: use all)',
    )
    p.add_argument(
        '--value',
        type=str,
        default='neglog10_p',
        choices=['neglog10_p', 'neglog10_q'],
        help='Vertical axis in volcano and significance histogram',
    )
    p.add_argument('--min_count', type=int, default=5, help='Min n_present to include a motif')
    p.add_argument('--clip-log2', type=float, default=6.0, help='Volcano x-axis limit for log2(OR)')
    p.add_argument('--fig_width', type=float, default=11.0)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--title_suffix', type=str, default='', help='Append to panel titles')
    p.add_argument(
        '--annotate-min-neglog10-p',
        type=float,
        default=10.0,
        help='Annotate volcano when −log10(Fisher p) ≥ this (0 disables)',
    )
    args = p.parse_args()

    panels: list[tuple[str, Path]] = []
    for raw in args.panel:
        if '=' not in raw:
            raise SystemExit(f'--panel must be NAME=PATH, got: {raw!r}')
        name, path = raw.split('=', 1)
        name, path = name.strip(), path.strip()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        panels.append((name, path))

    ann = None if args.annotate_min_neglog10_p <= 0 else float(args.annotate_min_neglog10_p)
    plot_descriptive_panels(
        panels,
        Path(args.out),
        min_count=args.min_count,
        max_motifs=args.max_motifs,
        value_col=args.value,
        clip_log2=args.clip_log2,
        fig_width=args.fig_width,
        dpi=args.dpi,
        title_suffix=args.title_suffix,
        annotate_min_neglog10_p=ann,
    )


if __name__ == '__main__':
    main()
