"""
Motif–class association (Fisher exact / chi-squared) for molecular graphs with BRICS motifs.

Used when there is no human ground-truth explanation: identify subgraphs whose *presence*
correlates with the label (e.g. mutagenicity). Results are cached to disk; optionally
attached to each `Data` via a PyG transform (per-node Fisher p for that node's motif id).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, fisher_exact


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (two-sided tests); NaNs preserved."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if not m.any():
        return out
    pv = p[m]
    n = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = sorted_p * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_full = np.empty(n, dtype=float)
    q_full[order] = q_sorted
    out[m] = q_full
    return out


def unique_motifs_in_graph(lookup: dict, smiles: str) -> set[int]:
    """Global motif indices present in `smiles` (excludes unmapped / padding)."""
    if lookup is None or smiles not in lookup:
        return set()
    out: set[int] = set()
    for _node, (_mstr, mid) in lookup[smiles].items():
        if mid is None:
            continue
        try:
            k = int(mid)
        except (TypeError, ValueError):
            continue
        if k >= 0:
            out.add(k)
    return out


def build_graph_motif_presence(
    smiles_list: Sequence[str],
    y: np.ndarray,
    lookup: dict,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Returns:
        X: (n_graphs, n_motif_ids) bool — motif m appears anywhere in graph g
        y: (n_graphs,) int {0,1}
        motif_ids: list of motif column ids (global indices)
    """
    y = np.asarray(y).astype(int).ravel()
    assert len(smiles_list) == len(y)
    all_ids: set[int] = set()
    per_graph: list[set[int]] = []
    for smi in smiles_list:
        mids = unique_motifs_in_graph(lookup, smi)
        per_graph.append(mids)
        all_ids |= mids
    motif_ids = sorted(all_ids)
    if not motif_ids:
        return np.zeros((len(y), 0), dtype=bool), y, []
    mid_to_col = {m: j for j, m in enumerate(motif_ids)}
    X = np.zeros((len(y), len(motif_ids)), dtype=bool)
    for i, mids in enumerate(per_graph):
        for m in mids:
            if m in mid_to_col:
                X[i, mid_to_col[m]] = True
    return X, y, motif_ids


def _safe_fisher_chi2(
    present: np.ndarray,
    y: np.ndarray,
    min_cell: float = 1.0,
) -> tuple[float | None, float | None, tuple[int, int, int, int]]:
    """
    present: (n,) bool — graph contains motif
    y: (n,) int 0/1
    Table:
              y=0   y=1
    absent      a     b
    present     c     d
    """
    y = np.asarray(y).astype(int).ravel()
    absent = ~present
    a = int(np.sum(absent & (y == 0)))
    b = int(np.sum(absent & (y == 1)))
    c = int(np.sum(present & (y == 0)))
    d = int(np.sum(present & (y == 1)))
    table = [[a, b], [c, d]]
    fisher_p: float | None = None
    chi2_p: float | None = None
    try:
        _, fisher_p = fisher_exact(table, alternative='two-sided')
    except ValueError:
        fisher_p = None
    try:
        chi2, p_chi, _, _ = chi2_contingency(table, correction=False)
        if not math.isfinite(chi2):
            chi2_p = None
        else:
            # expected frequency check
            exp = np.asarray(table, dtype=float)
            row_s = exp.sum(axis=1, keepdims=True)
            col_s = exp.sum(axis=0, keepdims=True)
            tot = exp.sum()
            if tot <= 0:
                chi2_p = None
            else:
                expected = row_s @ col_s / tot
                if expected.min() < min_cell:
                    chi2_p = None
                else:
                    chi2_p = float(p_chi)
    except (ValueError, np.linalg.LinAlgError):
        chi2_p = None
    return fisher_p, chi2_p, (a, b, c, d)


def association_table(
    X: np.ndarray,
    y: np.ndarray,
    motif_ids: list[int],
    motif_smiles: Sequence[str | None] | None,
    min_support: int = 5,
) -> pd.DataFrame:
    """One row per motif id with Fisher / chi2 p-values and counts."""
    rows = []
    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    for j, mid in enumerate(motif_ids):
        col = X[:, j]
        n_pos = int(col.sum())
        n_neg = n - n_pos
        if n_pos < min_support or n_neg < min_support:
            continue
        fp, cp, (a, b, c, d) = _safe_fisher_chi2(col, y)
        smi = None
        if motif_smiles is not None and 0 <= int(mid) < len(motif_smiles):
            smi = motif_smiles[int(mid)]
        rows.append({
            'motif_id': int(mid),
            'motif_smiles': smi,
            'n_graphs': n,
            'n_present': n_pos,
            'n_absent': n_neg,
            'contingency_a_absent_y0': a,
            'contingency_b_absent_y1': b,
            'contingency_c_present_y0': c,
            'contingency_d_present_y1': d,
            'fisher_p': fp,
            'chi2_p': cp,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    fp = df['fisher_p'].astype(float)
    df['fisher_q_bh'] = _benjamini_hochberg(fp.to_numpy())
    return df


def motif_pvalue_vector(
    df: pd.DataFrame,
    vocab_size: int,
) -> np.ndarray:
    """Dense length vocab_size; NaN if motif not in table."""
    out = np.full(vocab_size, np.nan, dtype=float)
    for _, r in df.iterrows():
        mid = int(r['motif_id'])
        if 0 <= mid < vocab_size and pd.notna(r.get('fisher_p')):
            out[mid] = float(r['fisher_p'])
    return out


class AddMotifAssocP:
    """
    Adds per-node Fisher p-values from a dense vector indexed by global motif id:
      data.node_motif_assoc_p[i] = pvec[nodes_to_motifs[i]] (or NaN if unmapped).
    Optional: data.graph_min_motif_assoc_p — min p over nodes with valid motifs.
    """

    def __init__(self, pvec: torch.Tensor, sig_alpha: float = 0.05):
        self.pvec = pvec  # (V,) float, can contain nan
        self.sig_alpha = float(sig_alpha)

    def __call__(self, data):
        ntm = getattr(data, 'nodes_to_motifs', None)
        if ntm is None:
            return data
        pv = self.pvec.to(device=ntm.device, dtype=torch.float32)
        n = ntm.numel()
        out = torch.full((n,), float('nan'), device=ntm.device, dtype=torch.float32)
        valid = ntm >= 0
        if valid.any():
            idx = ntm[valid].clamp(min=0, max=pv.numel() - 1)
            out[valid] = pv[idx]
        data.node_motif_assoc_p = out
        finite = out[valid & torch.isfinite(out)]
        if finite.numel() > 0:
            data.graph_min_motif_assoc_p = finite.min()
            data.graph_has_stat_sig_motif = bool((finite < self.sig_alpha).any().item())
        else:
            data.graph_min_motif_assoc_p = torch.tensor(float('nan'), device=ntm.device)
            data.graph_has_stat_sig_motif = False
        return data


def save_association_artifacts(
    out_dir: Path,
    stem: str,
    df: pd.DataFrame,
    pvec: np.ndarray,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{stem}_motif_class_association.csv'
    df.to_csv(csv_path, index=False)
    torch.save(torch.from_numpy(pvec).float(), out_dir / f'{stem}_motif_pvalues.pt')
    with open(out_dir / f'{stem}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_pvalue_tensor(out_dir: Path, stem: str) -> torch.Tensor | None:
    p = out_dir / f'{stem}_motif_pvalues.pt'
    if not p.exists():
        return None
    return torch.load(p, weights_only=True)
