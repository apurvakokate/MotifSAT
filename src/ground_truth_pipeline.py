#!/usr/bin/env python
"""
Build and cache molecular datasets with edge-level ground truth labels.

Ground-truth generation is intentionally derived from existing masked-feature
pickle artifacts (train/valid/test *_dataset_masked.pickle contents) without
changing how masked features are produced.

This module provides:
  1) Reusable helpers to load/build cached split datasets for training pipeline
  2) A CLI to precompute all requested datasets/folds and save debug artifacts
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_data_list(dataset) -> list:
    """Materialize a split dataset into a plain Data list."""
    return [dataset[i].clone() for i in range(len(dataset))]


def _extract_mask_index(mask_data: Any) -> tuple[dict[int, list[torch.Tensor]], dict[str, int]]:
    """
    Convert masked pickle payload into graph_idx -> list(masked_x tensors).

    Expected masked payload structure (unchanged upstream):
      mask_data[0]: motif_idx -> graph_idx -> masked node-feature tensor
      mask_data[1]: original graph list (optional, unused here)
    """
    stats = {
        "n_mask_entries": 0,
        "n_graphs_with_masks": 0,
        "n_motif_keys": 0,
    }
    per_graph_masks: dict[int, list[torch.Tensor]] = defaultdict(list)
    if not isinstance(mask_data, (list, tuple)) or len(mask_data) == 0:
        return per_graph_masks, stats

    motif_to_graph = mask_data[0]
    if not isinstance(motif_to_graph, dict):
        return per_graph_masks, stats

    stats["n_motif_keys"] = int(len(motif_to_graph))
    for _, graph_map in motif_to_graph.items():
        if not isinstance(graph_map, dict):
            continue
        for graph_idx, masked_x in graph_map.items():
            try:
                gi = int(graph_idx)
            except Exception:
                continue
            if masked_x is None:
                continue
            if not torch.is_tensor(masked_x):
                try:
                    masked_x = torch.as_tensor(masked_x)
                except Exception:
                    continue
            per_graph_masks[gi].append(masked_x)
            stats["n_mask_entries"] += 1

    stats["n_graphs_with_masks"] = int(len(per_graph_masks))
    return per_graph_masks, stats


def _edge_label_from_masked_features(data, masked_x_list: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Build edge ground truth from masked node-feature tensors.

    Ground-truth derivation follows existing masking semantics:
      - a node is treated as masked if its features become all-zero
      - an edge is positive when both endpoints are masked in a motif mask
      - final edge_label is union across motif masks for that graph
    """
    edge_label = torch.zeros(data.edge_index.size(1), dtype=torch.float32)
    src, dst = data.edge_index

    n_used_masks = 0
    n_bad_shape = 0
    n_no_masked_nodes = 0

    data_x = data.x.detach().cpu()
    for masked_x in masked_x_list:
        mx = masked_x.detach().cpu()
        if mx.shape != data_x.shape:
            n_bad_shape += 1
            continue
        masked_nodes = (mx.abs().sum(dim=1) == 0) & (data_x.abs().sum(dim=1) > 0)
        if int(masked_nodes.sum().item()) == 0:
            n_no_masked_nodes += 1
            continue
        motif_edge_mask = masked_nodes[src] & masked_nodes[dst]
        edge_label[motif_edge_mask] = 1.0
        n_used_masks += 1

    dbg = {
        "n_used_masks": int(n_used_masks),
        "n_bad_shape": int(n_bad_shape),
        "n_no_masked_nodes": int(n_no_masked_nodes),
    }
    return edge_label, dbg


def _debug_plot(split_df: pd.DataFrame, fig_path: Path, title: str):
    """Save simple diagnostics figure for edge-label coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    pos_frac = split_df["edge_pos_frac"].to_numpy(dtype=float)
    axes[0].hist(pos_frac, bins=30, color="#4c72b0", alpha=0.85)
    axes[0].set_title("Per-graph edge GT fraction")
    axes[0].set_xlabel("positive edges / total edges")
    axes[0].set_ylabel("count")

    axes[1].scatter(
        split_df["n_mask_tensors"].to_numpy(dtype=float),
        split_df["n_positive_edges"].to_numpy(dtype=float),
        s=14,
        alpha=0.6,
        color="#dd8452",
    )
    axes[1].set_title("Mask tensors vs positive edges")
    axes[1].set_xlabel("# motif mask tensors")
    axes[1].set_ylabel("# positive edges")

    fig.suptitle(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def load_or_build_ground_truth_split(
    *,
    dataset_name: str,
    fold: int,
    split_name: str,
    dataset,
    mask_data,
    cache_root: Path,
    dictionary_fold_variant: str = "nofilter",
    force_rebuild: bool = False,
    relabel_graphs_with_ground_truth: bool = True,
) -> tuple[list, dict[str, Any]]:
    """
    Return split dataset with edge_label GT, using on-disk cache when available.
    """
    cache_dir = (
        Path(cache_root)
        / dataset_name
        / f"fold{int(fold)}"
        / str(dictionary_fold_variant)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    relabel_tag = "relabel1" if relabel_graphs_with_ground_truth else "relabel0"
    cache_file = cache_dir / f"{split_name}_dataset_with_ground_truth_{relabel_tag}.pt"
    debug_csv = cache_dir / f"{split_name}_ground_truth_debug_{relabel_tag}.csv"
    debug_json = cache_dir / f"{split_name}_ground_truth_debug_{relabel_tag}.json"
    debug_fig = cache_dir / f"{split_name}_ground_truth_debug_{relabel_tag}.png"

    if cache_file.is_file() and not force_rebuild:
        data_list = torch.load(cache_file, weights_only=False)
        debug = {}
        if debug_json.is_file():
            try:
                with open(debug_json) as f:
                    debug = json.load(f)
            except Exception:
                debug = {}
        debug["loaded_from_cache"] = True
        debug["cache_file"] = str(cache_file)
        return data_list, debug

    data_list = _as_data_list(dataset)
    per_graph_masks, mask_stats = _extract_mask_index(mask_data)

    debug_rows: list[dict[str, Any]] = []
    aggregate = {
        "n_graphs_total": int(len(data_list)),
        "n_graphs_with_pos_edges": 0,
        "n_total_edges": 0,
        "n_total_pos_edges": 0,
        "n_graphs_relabelled": 0,
        "n_used_masks_total": 0,
        "n_bad_shape_total": 0,
        "n_no_masked_nodes_total": 0,
        **mask_stats,
    }

    for gi, data in enumerate(data_list):
        masked_x_list = per_graph_masks.get(gi, [])
        edge_label, dbg = _edge_label_from_masked_features(data, masked_x_list)
        data.edge_label = edge_label.float()

        n_edges = int(edge_label.numel())
        n_pos = int(edge_label.sum().item())
        edge_pos_frac = (float(n_pos) / n_edges) if n_edges > 0 else 0.0
        gt_graph_label = 1.0 if n_pos > 0 else 0.0

        old_y = getattr(data, "y", None)
        old_y_scalar = np.nan
        if old_y is not None:
            try:
                old_y_scalar = float(torch.as_tensor(old_y).view(-1)[0].item())
            except Exception:
                old_y_scalar = np.nan

        if relabel_graphs_with_ground_truth:
            data.y = torch.tensor([gt_graph_label], dtype=torch.float32)
            if not np.isnan(old_y_scalar) and float(old_y_scalar) != float(gt_graph_label):
                aggregate["n_graphs_relabelled"] += 1

        aggregate["n_total_edges"] += n_edges
        aggregate["n_total_pos_edges"] += n_pos
        aggregate["n_used_masks_total"] += dbg["n_used_masks"]
        aggregate["n_bad_shape_total"] += dbg["n_bad_shape"]
        aggregate["n_no_masked_nodes_total"] += dbg["n_no_masked_nodes"]
        if n_pos > 0:
            aggregate["n_graphs_with_pos_edges"] += 1

        debug_rows.append(
            {
                "graph_idx": int(gi),
                "smiles": str(getattr(data, "smiles", "")),
                "n_edges": n_edges,
                "n_positive_edges": n_pos,
                "edge_pos_frac": edge_pos_frac,
                "n_mask_tensors": int(len(masked_x_list)),
                "n_used_masks": int(dbg["n_used_masks"]),
                "n_bad_shape": int(dbg["n_bad_shape"]),
                "n_no_masked_nodes": int(dbg["n_no_masked_nodes"]),
                "old_graph_label": old_y_scalar,
                "new_graph_label": gt_graph_label if relabel_graphs_with_ground_truth else old_y_scalar,
                "gt_graph_label": gt_graph_label,
            }
        )

    aggregate["edge_positive_fraction_global"] = (
        float(aggregate["n_total_pos_edges"]) / float(aggregate["n_total_edges"])
        if aggregate["n_total_edges"] > 0
        else 0.0
    )
    aggregate["loaded_from_cache"] = False
    aggregate["cache_file"] = str(cache_file)
    aggregate["split"] = split_name
    aggregate["dataset"] = dataset_name
    aggregate["fold"] = int(fold)
    aggregate["relabel_graphs_with_ground_truth"] = bool(relabel_graphs_with_ground_truth)

    torch.save(data_list, cache_file)
    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(debug_csv, index=False)
    _debug_plot(
        debug_df,
        debug_fig,
        title=f"{dataset_name} fold{fold} {split_name}: edge GT diagnostics",
    )
    with open(debug_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    return data_list, aggregate


def _normalize_label_col(label_col):
    if isinstance(label_col, list):
        return label_col[0]
    return label_col


def _build_split_datasets_for_cli(
    *,
    dataset_name: str,
    fold: int,
    csv_file: str,
    label_col,
    lookup,
    test_lookup,
    dataset_type: str,
):
    from DataLoader import MolDataset

    label_col = _normalize_label_col(label_col)
    if dataset_type == "Regression":
        train_set = MolDataset(
            root=".",
            split="training",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=True,
            mean=None,
            std=None,
            lookup=lookup,
        )
        valid_set = MolDataset(
            root=".",
            split="valid",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=True,
            mean=train_set.mean,
            std=train_set.std,
            lookup=lookup,
        )
        test_set = MolDataset(
            root=".",
            split="test",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=True,
            mean=train_set.mean,
            std=train_set.std,
            lookup=test_lookup,
        )
    else:
        train_set = MolDataset(
            root=".",
            split="training",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=False,
            lookup=lookup,
        )
        valid_set = MolDataset(
            root=".",
            split="valid",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=False,
            lookup=lookup,
        )
        test_set = MolDataset(
            root=".",
            split="test",
            csv_file=csv_file,
            label_col=[label_col],
            normalize=False,
            lookup=test_lookup,
        )
    return {"train": train_set, "valid": valid_set, "test": test_set}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build cached dataset splits with edge-level ground truth from "
            "masked motif pickles, plus debug artifacts."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "Mutagenicity",
            "BBBP",
            "hERG",
            "Benzene",
            "Alkane_Carbonyl",
            "Fluoride_Carbonyl",
            "esol",
            "Lipophilicity",
        ],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--algorithm", type=str, default="BRICS")
    parser.add_argument("--dictionary_fold_variant", type=str, default="nofilter")
    parser.add_argument(
        "--dictionary_path",
        type=str,
        default=os.environ.get(
            "MOTIFSAT_DICTIONARY_PATH",
            "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_CREATE",
        ),
    )
    parser.add_argument(
        "--csv_root",
        type=str,
        default="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DomainDrivenGlobalExpl/datasets/FOLDS",
    )
    parser.add_argument("--cache_root", type=str, default="../data/ground_truth_cache")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument(
        "--no_relabel_graphs_with_ground_truth",
        action="store_true",
        default=False,
        help="Keep original graph labels even when edge-level ground truth is present.",
    )
    args = parser.parse_args()

    from DataLoader import CHOSEN_THRESHOLD, get_setup_files_with_folds
    from utils.get_data_loaders import DATASET_COLUMN, DATASET_TYPE

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        if dataset not in DATASET_COLUMN:
            print(f"[WARN] Unknown dataset in DATASET_COLUMN: {dataset}")
            continue
        for fold in args.folds:
            print(f"\n[INFO] Processing {dataset} fold={fold}")
            try:
                thr = CHOSEN_THRESHOLD[args.algorithm][dataset]
                date_tag = f"{args.algorithm}{thr:g}"
            except Exception as e:
                print(f"[WARN] Missing threshold for {dataset}: {e}")
                continue

            csv_file = Path(args.csv_root) / f"{dataset}_{fold}.csv"
            if not csv_file.is_file():
                print(f"[WARN] Missing CSV: {csv_file}")
                continue

            (
                lookup,
                _motif_list,
                _motif_counts,
                _motif_lengths,
                _motif_class_count,
                _graph_to_motifs,
                test_lookup,
                _test_graph_to_motifs,
                train_mask_data,
                val_mask_data,
                test_mask_data,
            ) = get_setup_files_with_folds(
                dataset_name=dataset,
                date_tag=date_tag,
                fold=fold,
                algorithm=args.algorithm,
                path=args.dictionary_path,
                dictionary_fold_variant=args.dictionary_fold_variant,
            )

            split_sets = _build_split_datasets_for_cli(
                dataset_name=dataset,
                fold=fold,
                csv_file=str(csv_file),
                label_col=DATASET_COLUMN[dataset],
                lookup=lookup,
                test_lookup=test_lookup,
                dataset_type=DATASET_TYPE[dataset],
            )
            split_masks = {
                "train": train_mask_data,
                "valid": val_mask_data,
                "test": test_mask_data,
            }

            split_debug = {}
            for split_name in ("train", "valid", "test"):
                _, dbg = load_or_build_ground_truth_split(
                    dataset_name=dataset,
                    fold=fold,
                    split_name=split_name,
                    dataset=split_sets[split_name],
                    mask_data=split_masks[split_name],
                    cache_root=cache_root,
                    dictionary_fold_variant=args.dictionary_fold_variant,
                    force_rebuild=args.force_rebuild,
                    relabel_graphs_with_ground_truth=not args.no_relabel_graphs_with_ground_truth,
                )
                split_debug[split_name] = dbg
                print(
                    f"[INFO] {dataset} fold={fold} {split_name}: "
                    f"graphs={dbg.get('n_graphs_total', 'na')} "
                    f"pos_graphs={dbg.get('n_graphs_with_pos_edges', 'na')} "
                    f"relabelled={dbg.get('n_graphs_relabelled', 'na')} "
                    f"edge_pos_frac={dbg.get('edge_positive_fraction_global', float('nan')):.4f} "
                    f"cache={dbg.get('cache_file', '')}"
                )

            run_rows.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "dictionary_fold_variant": args.dictionary_fold_variant,
                    "train_edge_positive_fraction_global": split_debug["train"].get("edge_positive_fraction_global", np.nan),
                    "valid_edge_positive_fraction_global": split_debug["valid"].get("edge_positive_fraction_global", np.nan),
                    "test_edge_positive_fraction_global": split_debug["test"].get("edge_positive_fraction_global", np.nan),
                }
            )

    if run_rows:
        summary_path = cache_root / "ground_truth_cache_summary.csv"
        pd.DataFrame(run_rows).to_csv(summary_path, index=False)
        print(f"\n[INFO] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
