#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

DNF_RULE_CONFIGS = {
    "Mutagenicity": {"floor_anchor": 0.70, "floor_fill": 0.55, "sup_lo": 0.005},
    "Benzene": {"floor_anchor": 0.90, "floor_fill": 0.60, "sup_lo": 0.003},
    "BBBP": {"floor_anchor": 0.90, "floor_fill": 0.80, "sup_lo": 0.010},
    "hERG": {"floor_anchor": 0.65, "floor_fill": 0.60, "sup_lo": 0.010},
    "Alkane_Carbonyl": {"floor_anchor": 0.50, "floor_fill": 0.50, "sup_lo": 0.010},
}
DNF_RULE_DEFAULT = {"floor_anchor": 0.70, "floor_fill": 0.55, "sup_lo": 0.010}


def _as_data_list(dataset) -> list:
    return [dataset[i].clone() for i in range(len(dataset))]


def _normalize_binary_y(y_values: np.ndarray) -> np.ndarray:
    y = np.asarray(y_values).reshape(-1).astype(float)
    if y.size == 0:
        return np.zeros((0,), dtype=int)
    yn = y[~np.isnan(y)]
    if yn.size == 0:
        return np.zeros_like(y, dtype=int)
    if yn.min() >= 0 and yn.max() <= 1:
        return (y >= 0.5).astype(int)
    if set(np.unique(yn).tolist()) == {-1.0, 1.0}:
        return (y > 0).astype(int)
    return (y > 0).astype(int)


def _records_from_splits(split_data_lists: dict[str, list], motif_list: list[Any] | None) -> list[dict[str, Any]]:
    rows = []
    for split_name in ("train", "valid", "test"):
        for split_idx, data in enumerate(split_data_lists[split_name]):
            n2m = getattr(data, "nodes_to_motifs", None)
            motifs = []
            motif_names = []
            if n2m is not None:
                motifs = sorted({int(v) for v in n2m.detach().cpu().tolist() if int(v) >= 0})
                motif_names = sorted({_motif_name(int(v), motif_list) for v in motifs})
            y0 = float(torch.as_tensor(getattr(data, "y")).view(-1)[0].item())
            rows.append(
                {
                    "split": split_name,
                    "split_idx": int(split_idx),
                    "smiles": str(getattr(data, "smiles", "")),
                    "motif_ids": motifs,
                    "motif_names": motif_names,
                    "y_orig": y0,
                }
            )
    return rows


def _presence(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    motif_names = sorted({m for r in records for m in r["motif_names"]})
    motif_to_col = {m: j for j, m in enumerate(motif_names)}
    X = np.zeros((len(records), len(motif_names)), dtype=np.uint8)
    for i, row in enumerate(records):
        for m in row["motif_names"]:
            X[i, motif_to_col[m]] = 1
    y = _normalize_binary_y(np.array([r["y_orig"] for r in records], dtype=float))
    col_to_motif = {j: m for m, j in motif_to_col.items()}
    return X, y, col_to_motif


def _candidate_pool(X: np.ndarray, y: np.ndarray, floor_lo: float, floor_hi: float, sup_lo: float) -> list[dict[str, Any]]:
    n, p = X.shape
    if n == 0 or p == 0:
        return []
    cands = []
    support = X.mean(axis=0)
    usable = [int(j) for j in np.where((support >= sup_lo) & (support <= 0.99))[0]]
    for j in usable:
        mask = X[:, j].astype(bool)
        cnt = int(mask.sum())
        if cnt < 8:
            continue
        prec = float((mask & (y == 1)).sum() / max(cnt, 1))
        if floor_lo <= prec < floor_hi:
            cands.append({"motif_cols": (j,), "k": 1, "mask": mask, "prec": prec, "score": float(mask.mean()) * prec})

    usable_pairs = [j for j in usable if support[j] >= 0.02]
    usable_pairs = sorted(usable_pairs, key=lambda j: -support[j])[:120]
    for j1, j2 in combinations(usable_pairs, 2):
        mask = X[:, j1].astype(bool) & X[:, j2].astype(bool)
        cnt = int(mask.sum())
        if cnt < 15:
            continue
        prec = float((mask & (y == 1)).sum() / max(cnt, 1))
        if floor_lo <= prec < floor_hi:
            cands.append({"motif_cols": (j1, j2), "k": 2, "mask": mask, "prec": prec, "score": float(mask.mean()) * prec})
    cands.sort(key=lambda c: -c["score"])
    return cands


def _greedy(cands: list[dict[str, Any]], covered: np.ndarray, target_cov: float = 0.50) -> tuple[list[dict[str, Any]], np.ndarray]:
    out = []
    n = int(covered.size)
    for c in cands:
        if float(covered.mean()) >= target_cov:
            break
        gain = int((c["mask"] & ~covered).sum())
        if gain < max(8, int(n * 0.002)):
            continue
        covered = covered | c["mask"]
        out.append({**c, "gain": gain, "cumul_cov": float(covered.mean())})
    return out, covered


def _fit_rules(records: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    cfg = DNF_RULE_CONFIGS.get(dataset_name, DNF_RULE_DEFAULT)
    X, y, col_to_motif_name = _presence(records)
    covered = np.zeros((X.shape[0],), dtype=bool)
    fa, ff, sup_lo = float(cfg["floor_anchor"]), float(cfg["floor_fill"]), float(cfg["sup_lo"])
    anchor_pool = _candidate_pool(X, y, fa, 2.0, sup_lo)
    fill_pool = _candidate_pool(X, y, ff, fa, sup_lo) if ff < fa else []
    a_sel, covered = _greedy(anchor_pool, covered, 0.50)
    f_sel, covered = _greedy(fill_pool, covered, 0.50)
    clauses = [{**c, "tier": "anchor"} for c in a_sel] + [{**c, "tier": "fill"} for c in f_sel]
    for c in clauses:
        # Canonicalized by motif names (fold-invariant where vocab names match).
        c["motif_names"] = sorted([str(col_to_motif_name[j]) for j in c["motif_cols"]])
    return {
        "X": X,
        "clauses": clauses,
        "col_to_motif_name": col_to_motif_name,
        "n_rows": int(X.shape[0]),
        "n_motifs": int(X.shape[1]),
        "n_anchor_candidates": int(len(anchor_pool)),
        "n_fill_candidates": int(len(fill_pool)),
    }


def _fired(row_x: np.ndarray, clauses: list[dict[str, Any]], col_to_motif_name: dict[int, str]) -> tuple[list[int], list[str]]:
    fired_idx = []
    active_motif_names = set()
    for ci, c in enumerate(clauses):
        cols = c["motif_cols"]
        ok = bool(row_x[cols[0]]) if len(cols) == 1 else bool(row_x[cols[0]] and row_x[cols[1]])
        if ok:
            fired_idx.append(int(ci))
            for cc in cols:
                active_motif_names.add(str(col_to_motif_name[int(cc)]))
    return fired_idx, sorted(active_motif_names)


def _edge_label(data, active_motifs: set[int]) -> tuple[torch.Tensor, int]:
    edge_label = torch.zeros(data.edge_index.size(1), dtype=torch.float32)
    n2m = getattr(data, "nodes_to_motifs", None)
    if n2m is None or not active_motifs:
        return edge_label, 0
    node_is_active = torch.tensor([int(v) in active_motifs for v in n2m.detach().cpu().tolist()], dtype=torch.bool)
    src, dst = data.edge_index.detach().cpu().long()
    pos = node_is_active[src] | node_is_active[dst]
    edge_label[pos] = 1.0
    return edge_label, int(pos.sum().item())


def _motif_name(motif_id: int, motif_list: list[Any] | None) -> str:
    if motif_list is not None and 0 <= int(motif_id) < len(motif_list):
        raw = motif_list[int(motif_id)]
        return str(raw)
    return f"motif_{int(motif_id)}"


def _motif_name_to_ids(motif_list: list[Any] | None) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    if motif_list is None:
        return out
    for mid, raw in enumerate(motif_list):
        key = str(raw)
        if key not in out:
            out[key] = set()
        out[key].add(int(mid))
    return out


def _resolve_active_ids(active_ids: set[int], active_names: set[str], motif_name_to_ids: dict[str, set[int]]) -> set[int]:
    resolved = set(int(v) for v in active_ids)
    for name in active_names:
        resolved.update(motif_name_to_ids.get(str(name), set()))
    return resolved


def _debug_plot(df: pd.DataFrame, fig_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(df["edge_pos_frac"].to_numpy(dtype=float), bins=30, color="#4c72b0", alpha=0.85)
    axes[0].set_title("Per-graph edge GT fraction")
    axes[1].hist(df["n_active_rule_motifs"].to_numpy(dtype=float), bins=20, color="#dd8452", alpha=0.85)
    axes[1].set_title("Active rule motifs per graph")
    fig.suptitle(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def load_or_build_ground_truth_splits(
    *,
    dataset_name: str,
    fold: int,
    split_datasets: dict[str, Any],
    motif_list: list[Any] | None = None,
    cache_root: Path,
    dictionary_fold_variant: str = "nofilter",
    force_rebuild: bool = False,
    relabel_graphs_with_ground_truth: bool = True,
) -> tuple[dict[str, list], dict[str, dict[str, Any]]]:
    cache_dir = Path(cache_root) / dataset_name / f"fold{int(fold)}" / str(dictionary_fold_variant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    relabel_tag = "relabel1" if relabel_graphs_with_ground_truth else "relabel0"
    cache_files = {s: cache_dir / f"{s}_dataset_with_ground_truth_dnf_{relabel_tag}.pt" for s in ("train", "valid", "test")}
    debug_json = {s: cache_dir / f"{s}_ground_truth_debug_dnf_{relabel_tag}.json" for s in ("train", "valid", "test")}
    debug_csv = {s: cache_dir / f"{s}_ground_truth_debug_dnf_{relabel_tag}.csv" for s in ("train", "valid", "test")}
    debug_fig = {s: cache_dir / f"{s}_ground_truth_debug_dnf_{relabel_tag}.png" for s in ("train", "valid", "test")}
    rules_json = cache_dir / f"dnf_rules_{relabel_tag}.json"

    if (not force_rebuild) and all(p.is_file() for p in cache_files.values()) and rules_json.is_file():
        out = {s: torch.load(cache_files[s], weights_only=False) for s in ("train", "valid", "test")}
        dbg = {}
        for s in ("train", "valid", "test"):
            d = {}
            if debug_json[s].is_file():
                with open(debug_json[s]) as f:
                    d = json.load(f)
            d["loaded_from_cache"] = True
            d["cache_file"] = str(cache_files[s])
            dbg[s] = d
        return out, dbg

    split_data_lists = {s: _as_data_list(split_datasets[s]) for s in ("train", "valid", "test")}
    motif_name_to_ids = _motif_name_to_ids(motif_list)
    records = _records_from_splits(split_data_lists, motif_list=motif_list)
    model = _fit_rules(records, dataset_name)
    X, clauses, col_to_motif_name = model["X"], model["clauses"], model["col_to_motif_name"]
    with open(rules_json, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "fold": int(fold),
                "n_rows": model["n_rows"],
                "n_motifs": model["n_motifs"],
                "n_anchor_candidates": model["n_anchor_candidates"],
                "n_fill_candidates": model["n_fill_candidates"],
                "n_selected_clauses": len(clauses),
                "clauses": [
                    {
                        "tier": c["tier"],
                        "k": int(c["k"]),
                        "motif_names": [str(v) for v in c["motif_names"]],
                        # Backward-compatible metadata for existing analysis scripts.
                        "motif_ids": sorted(
                            {
                                int(mid)
                                for name in c["motif_names"]
                                for mid in motif_name_to_ids.get(str(name), set())
                            }
                        ),
                        "prec": float(c["prec"]),
                        "score": float(c["score"]),
                        "gain": int(c["gain"]),
                        "cumul_cov": float(c["cumul_cov"]),
                    }
                    for c in clauses
                ],
            },
            f,
            indent=2,
        )

    out = {"train": [], "valid": [], "test": []}
    dbg = {}
    cursor = 0
    for split_name in ("train", "valid", "test"):
        rows = []
        agg = {
            "n_graphs_total": int(len(split_data_lists[split_name])),
            "n_graphs_with_pos_edges": 0,
            "n_total_edges": 0,
            "n_total_pos_edges": 0,
            "n_graphs_rule_positive": 0,
            "n_graphs_relabelled": 0,
            "n_selected_clauses": int(len(clauses)),
            "rules_file": str(rules_json),
            "loaded_from_cache": False,
            "cache_file": str(cache_files[split_name]),
        }
        for data in split_data_lists[split_name]:
            fired_idx, active_motif_names_list = _fired(X[cursor], clauses, col_to_motif_name)
            active_motif_names = set(active_motif_names_list)
            active_motif_ids_resolved = _resolve_active_ids(set(), active_motif_names, motif_name_to_ids)
            gt_y = 1.0 if fired_idx else 0.0
            edge_label, n_pos = _edge_label(data, active_motif_ids_resolved)
            data.edge_label = edge_label
            old_y = float(torch.as_tensor(getattr(data, "y")).view(-1)[0].item())
            if relabel_graphs_with_ground_truth:
                data.y = torch.tensor([gt_y], dtype=torch.float32)
                if old_y != gt_y:
                    agg["n_graphs_relabelled"] += 1
            n_edges = int(edge_label.numel())
            if gt_y > 0.5:
                agg["n_graphs_rule_positive"] += 1
            if n_pos > 0:
                agg["n_graphs_with_pos_edges"] += 1
            agg["n_total_edges"] += n_edges
            agg["n_total_pos_edges"] += int(n_pos)
            rows.append(
                {
                    "smiles": str(getattr(data, "smiles", "")),
                    "n_edges": n_edges,
                    "n_positive_edges": int(n_pos),
                    "edge_pos_frac": (float(n_pos) / n_edges) if n_edges > 0 else 0.0,
                    "n_fired_clauses": int(len(fired_idx)),
                    "fired_clause_indices": ",".join(str(v) for v in fired_idx),
                    "n_active_rule_motifs": int(len(active_motif_ids_resolved)),
                    "active_rule_motif_names": ",".join(sorted(active_motif_names)),
                    # Backward-compatible metadata for existing analysis scripts.
                    "active_rule_motif_ids": ",".join(str(v) for v in sorted(active_motif_ids_resolved)),
                    "old_graph_label": old_y,
                    "gt_graph_label": gt_y,
                    "new_graph_label": gt_y if relabel_graphs_with_ground_truth else old_y,
                }
            )
            out[split_name].append(data)
            cursor += 1
        agg["edge_positive_fraction_global"] = float(agg["n_total_pos_edges"]) / float(agg["n_total_edges"]) if agg["n_total_edges"] > 0 else 0.0
        df = pd.DataFrame(rows)
        df.to_csv(debug_csv[split_name], index=False)
        _debug_plot(df, debug_fig[split_name], f"{dataset_name} fold{fold} {split_name}: DNF-rule GT diagnostics")
        with open(debug_json[split_name], "w") as f:
            json.dump(agg, f, indent=2)
        torch.save(out[split_name], cache_files[split_name])
        dbg[split_name] = agg
    return out, dbg


def _normalize_label_col(label_col):
    if isinstance(label_col, list):
        return label_col[0]
    return label_col


def _build_split_datasets_for_cli(*, csv_file: str, label_col, lookup, test_lookup, dataset_type: str):
    from DataLoader import MolDataset
    label_col = _normalize_label_col(label_col)
    if dataset_type == "Regression":
        train_set = MolDataset(root=".", split="training", csv_file=csv_file, label_col=[label_col], normalize=True, mean=None, std=None, lookup=lookup)
        valid_set = MolDataset(root=".", split="valid", csv_file=csv_file, label_col=[label_col], normalize=True, mean=train_set.mean, std=train_set.std, lookup=lookup)
        test_set = MolDataset(root=".", split="test", csv_file=csv_file, label_col=[label_col], normalize=True, mean=train_set.mean, std=train_set.std, lookup=test_lookup)
    else:
        train_set = MolDataset(root=".", split="training", csv_file=csv_file, label_col=[label_col], normalize=False, lookup=lookup)
        valid_set = MolDataset(root=".", split="valid", csv_file=csv_file, label_col=[label_col], normalize=False, lookup=lookup)
        test_set = MolDataset(root=".", split="test", csv_file=csv_file, label_col=[label_col], normalize=False, lookup=test_lookup)
    return {"train": train_set, "valid": valid_set, "test": test_set}


def main():
    parser = argparse.ArgumentParser(description="Build cached DNF-rule GT splits.")
    parser.add_argument("--datasets", nargs="+", default=["Mutagenicity", "BBBP", "hERG", "Benzene", "Alkane_Carbonyl"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--algorithm", type=str, default="BRICS")
    parser.add_argument("--dictionary_fold_variant", type=str, default="nofilter")
    parser.add_argument("--dictionary_path", type=str, default=os.environ.get("MOTIFSAT_DICTIONARY_PATH", "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_CREATE"))
    parser.add_argument("--csv_root", type=str, default="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DomainDrivenGlobalExpl/datasets/FOLDS")
    parser.add_argument("--cache_root", type=str, default="../data/ground_truth_cache")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--no_relabel_graphs_with_ground_truth", action="store_true", default=False)
    args = parser.parse_args()

    from DataLoader import CHOSEN_THRESHOLD, get_setup_files_with_folds
    from utils.get_data_loaders import DATASET_COLUMN, DATASET_TYPE

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset in args.datasets:
        if dataset not in DATASET_COLUMN:
            continue
        for fold in args.folds:
            thr = CHOSEN_THRESHOLD[args.algorithm][dataset]
            date_tag = f"{args.algorithm}{thr:g}"
            csv_file = Path(args.csv_root) / f"{dataset}_{fold}.csv"
            if not csv_file.is_file():
                continue
            setup = get_setup_files_with_folds(
                dataset_name=dataset,
                date_tag=date_tag,
                fold=fold,
                algorithm=args.algorithm,
                path=args.dictionary_path,
                dictionary_fold_variant=args.dictionary_fold_variant,
            )
            lookup = setup[0]
            motif_list = setup[1]
            test_lookup = setup[6]
            split_sets = _build_split_datasets_for_cli(
                csv_file=str(csv_file),
                label_col=DATASET_COLUMN[dataset],
                lookup=lookup,
                test_lookup=test_lookup,
                dataset_type=DATASET_TYPE[dataset],
            )
            _, split_debug = load_or_build_ground_truth_splits(
                dataset_name=dataset,
                fold=fold,
                split_datasets=split_sets,
                motif_list=motif_list,
                cache_root=cache_root,
                dictionary_fold_variant=args.dictionary_fold_variant,
                force_rebuild=args.force_rebuild,
                relabel_graphs_with_ground_truth=not args.no_relabel_graphs_with_ground_truth,
            )
            summary.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "train_edge_positive_fraction_global": split_debug["train"].get("edge_positive_fraction_global", np.nan),
                    "valid_edge_positive_fraction_global": split_debug["valid"].get("edge_positive_fraction_global", np.nan),
                    "test_edge_positive_fraction_global": split_debug["test"].get("edge_positive_fraction_global", np.nan),
                    "train_rule_positive_graphs": split_debug["train"].get("n_graphs_rule_positive", np.nan),
                }
            )
    if summary:
        out_path = cache_root / "ground_truth_cache_summary.csv"
        pd.DataFrame(summary).to_csv(out_path, index=False)
        print(f"[INFO] Summary saved: {out_path}")


if __name__ == "__main__":
    main()
