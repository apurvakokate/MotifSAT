#!/usr/bin/env python3
"""
Generate full experiment report tables and score-impact correlation grids.

Outputs:
  1) Tables (train/validation/test) with one row per (experiment, model, config variant),
     one column per flattened config parameter, plus:
       - classifier ROC
       - ground-truth explainer ROC (x_roc)
       - motif score-impact Pearson correlation
       - node score-impact Pearson correlation
  2) Combined motif-level score-impact scatter grids (train/validation/test),
     with one row per table row (not just best config), columns = architectures.

Usage:
  python src/generate_full_experiment_report.py --dataset Mutagenicity
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_motif_consistency import get_motif_level_score_impact_points
from collect_mutagenicity_tables import (
    compute_node_score_impact_correlation,
    compute_posthoc_correlation,
    find_results,
)
from motif_stat_vs_importance_roc import (
    _motif_id_to_name_map,
    _prepare_association_by_motif_name,
    auc_vs_delta_top_quantile,
    auc_vs_fisher_q,
    mean_abs_score_per_motif_name,
    paired_score_and_stats,
)


MODEL_ORDER = ["GAT", "GCN", "GIN", "PNA", "SAGE"]


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def _safe_mean_std(xs: list[float]) -> tuple[float | None, float | None, int]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def _fmt_mean_std_n(mean: float | None, std: float | None, n: int) -> str:
    if mean is None or n <= 0:
        return ""
    if std is None:
        std = 0.0
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _split_keys(split_name: str) -> tuple[str, str]:
    if split_name == "train":
        return "metric/best_clf_roc_train", "metric/best_x_roc_train"
    if split_name in ("valid", "validation"):
        return "metric/best_clf_roc_valid", "metric/best_x_roc_valid"
    return "metric/best_clf_roc_test", "metric/best_x_roc_test"


def _default_association_csv(dataset: str) -> Path:
    return (Path("../data") / "motif_association" / f"{dataset}_fold0_all_motif_class_association.csv").resolve()


def _build_xroc_fallback_context(
    dataset: str,
    association_csv: str | None,
    label_mode: str,
    fisher_q_alpha: float,
    delta_label_quantile: float,
) -> dict[str, Any] | None:
    assoc_path = Path(association_csv).expanduser().resolve() if association_csv else _default_association_csv(dataset)
    if not assoc_path.exists():
        return None
    try:
        assoc = pd.read_csv(assoc_path)
    except Exception:
        return None
    if assoc.empty:
        return None
    assoc_by_name = _prepare_association_by_motif_name(assoc)
    motif_id_to_name = _motif_id_to_name_map(assoc)
    if assoc_by_name.empty or not motif_id_to_name:
        return None
    return {
        "assoc_path": str(assoc_path),
        "assoc_by_name": assoc_by_name,
        "motif_id_to_name": motif_id_to_name,
        "label_mode": label_mode,
        "fisher_q_alpha": float(fisher_q_alpha),
        "delta_label_quantile": float(delta_label_quantile),
        "cache": {},
        "hits": 0,
    }


def _compute_xroc_fallback(
    seed_dir: Path,
    split: str,
    ctx: dict[str, Any] | None,
) -> float:
    if ctx is None:
        return float("nan")
    key = (str(seed_dir), split)
    cache: dict[tuple[str, str], float] = ctx["cache"]
    if key in cache:
        return cache[key]
    node_scores = seed_dir / "node_scores.jsonl"
    if not node_scores.exists():
        cache[key] = float("nan")
        return cache[key]
    mean_score = mean_abs_score_per_motif_name(node_scores, ctx["motif_id_to_name"], split=split)
    if not mean_score:
        cache[key] = float("nan")
        return cache[key]
    scores, deltas, qvals = paired_score_and_stats(ctx["assoc_by_name"], mean_score)
    if ctx["label_mode"] == "fisher_q":
        auc, _, _ = auc_vs_fisher_q(scores, qvals, fisher_q_alpha=ctx["fisher_q_alpha"])
    else:
        auc, _, _ = auc_vs_delta_top_quantile(
            scores,
            deltas,
            delta_label_quantile=ctx["delta_label_quantile"],
        )
    out = float(auc) if auc is not None and np.isfinite(auc) else float("nan")
    cache[key] = out
    return out


def build_group_rows(
    results_dir: Path,
    dataset: str,
    experiments: list[str],
    xroc_fallback_ctx: dict[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_experiment_records: dict[str, list[dict[str, Any]]] = {}

    for exp in experiments:
        recs = find_results(results_dir, exp, verbose=verbose, dataset=dataset)
        by_experiment_records[exp] = recs
        for r in recs:
            key = (exp, str(r.get("variant", "")))
            grouped[key].append(r)

    rows: list[dict[str, Any]] = []
    row_counter = 1
    for (exp, variant), recs in sorted(grouped.items()):
        if not recs:
            continue
        seed_dirs = [r["seed_dir"] for r in recs if "seed_dir" in r]
        summary = _read_json(seed_dirs[0] / "experiment_summary.json") if seed_dirs else {}
        flat_cfg = flatten_dict(summary)

        recs_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            recs_by_model[str(r.get("model", ""))].append(r)

        row_id = f"R{row_counter:04d}"
        row_counter += 1
        row = {
            "row_id": row_id,
            "experiment": exp,
            "variant": variant,
            "num_runs": len(recs),
            "num_models": len(recs_by_model),
            "models_present": ",".join(sorted(recs_by_model.keys())),
            "seed_dirs": ";".join(str(sd) for sd in seed_dirs),
        }
        row.update(flat_cfg)

        for model in MODEL_ORDER:
            model_recs = recs_by_model.get(model, [])
            if not model_recs:
                continue
            split_metric_buckets: dict[str, list[dict[str, float]]] = {
                "train": [],
                "validation": [],
                "test": [],
            }
            for r in model_recs:
                metrics = r.get("metrics", {}) or {}
                sd = r.get("seed_dir")
                if sd is None:
                    continue
                for split in ("train", "validation", "test"):
                    clf_key, x_key = _split_keys(split)
                    clf = metrics.get(clf_key, np.nan)
                    corr_split = "valid" if split == "validation" else split
                    xroc_raw = metrics.get(x_key, np.nan)
                    xroc = float(xroc_raw) if np.isfinite(xroc_raw) else np.nan
                    if (not np.isfinite(xroc) or xroc == 0.0) and xroc_fallback_ctx is not None:
                        xroc_fb = _compute_xroc_fallback(Path(sd), corr_split, xroc_fallback_ctx)
                        if np.isfinite(xroc_fb):
                            xroc = float(xroc_fb)
                            xroc_fallback_ctx["hits"] += 1
                    motif_r, _, motif_n = compute_posthoc_correlation(sd, split=corr_split)
                    node_r, _, node_n = compute_node_score_impact_correlation(sd, split=corr_split)
                    split_metric_buckets[split].append(
                        {
                            "clf_roc": float(clf) if np.isfinite(clf) else np.nan,
                            "x_roc": float(xroc) if np.isfinite(xroc) else np.nan,
                            "motif_r": float(motif_r) if np.isfinite(motif_r) else np.nan,
                            "node_r": float(node_r) if np.isfinite(node_r) else np.nan,
                            "motif_n": int(motif_n),
                            "node_n": int(node_n),
                        }
                    )

            for split in ("train", "validation", "test"):
                bucket = split_metric_buckets[split]
                for key in ("clf_roc", "x_roc", "motif_r", "node_r"):
                    mean, std, n = _safe_mean_std([b.get(key, np.nan) for b in bucket])
                    row[f"{split}_{key}_{model}_mean"] = mean
                    row[f"{split}_{key}_{model}_std"] = std
                    row[f"{split}_{key}_{model}_n"] = n
                    row[f"{split}_{key}_{model}"] = _fmt_mean_std_n(mean, std, n)

                motif_ns = [int(b.get("motif_n", 0)) for b in bucket]
                node_ns = [int(b.get("node_n", 0)) for b in bucket]
                row[f"{split}_motif_points_{model}_mean"] = float(np.mean(motif_ns)) if motif_ns else np.nan
                row[f"{split}_node_points_{model}_mean"] = float(np.mean(node_ns)) if node_ns else np.nan

        rows.append(row)

    return rows, by_experiment_records


def _plot_grid_for_split(
    split: str,
    row_entries: list[dict[str, Any]],
    output_png: Path,
) -> None:
    if not row_entries:
        return
    n_rows = len(row_entries)
    n_cols = len(MODEL_ORDER)
    fig_w = max(12, 2.5 * n_cols + 2.0)
    fig_h = max(6, 1.6 * n_rows + 1.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"Motif score vs impact ({split}) — all configs", fontsize=12, fontweight="bold")

    rg_split = "valid" if split == "validation" else split
    for i, row in enumerate(row_entries):
        label = f"{row['row_id']} | {row['experiment']} | {row['variant']}"
        for j, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Mean motif score", fontsize=7)
            if j == 0:
                ax.set_ylabel("Impact", fontsize=7)
                ax.text(
                    -0.45,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    fontsize=7,
                    va="center",
                    ha="left",
                )
            if i == 0:
                ax.set_title(model, fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=6)

            seed_dir = row.get("seed_dir_by_model", {}).get(model)
            if seed_dir is None:
                ax.text(0.5, 0.5, "No run", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                continue

            xs, ys = get_motif_level_score_impact_points(Path(seed_dir), split=rg_split)
            if xs is None or ys is None or len(xs) == 0:
                ax.text(0.5, 0.5, "No points", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                continue
            ax.scatter(xs, ys, s=8, alpha=0.3, edgecolors="none")

    fig.tight_layout(rect=[0.06, 0.03, 1.0, 0.97])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def build_plot_rows(rows_df: pd.DataFrame, by_experiment_records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # Map (exp, variant, model) -> representative seed_dir (best valid clf_roc if available)
    rep_seed_dir: dict[tuple[str, str, str], Path] = {}
    for exp, recs in by_experiment_records.items():
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            buckets[(str(r.get("model", "")), str(r.get("variant", "")))].append(r)
        for (model, variant), grp in buckets.items():
            best = None
            best_val = -1e18
            for r in grp:
                v = r.get("metrics", {}).get("metric/best_clf_roc_valid", np.nan)
                score = float(v) if np.isfinite(v) else -1e18
                if score > best_val:
                    best_val = score
                    best = r
            if best is not None:
                rep_seed_dir[(exp, variant, model)] = best["seed_dir"]

    plot_rows: list[dict[str, Any]] = []
    for _, r in rows_df.sort_values(["row_id"]).iterrows():
        exp = str(r["experiment"])
        variant = str(r["variant"])
        seed_dir_by_model: dict[str, Path] = {}
        for model in MODEL_ORDER:
            sd = rep_seed_dir.get((exp, variant, model))
            if sd is not None:
                seed_dir_by_model[model] = sd
        plot_rows.append(
            {
                "row_id": str(r["row_id"]),
                "experiment": exp,
                "variant": variant,
                "seed_dir_by_model": seed_dir_by_model,
            }
        )
    return plot_rows


def write_split_tables(rows_df: pd.DataFrame, out_dir: Path) -> None:
    base_cols = ["row_id", "experiment", "variant", "num_runs", "num_models", "models_present"]
    metric_cols_by_split = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split in ("train", "validation", "test"):
        for model in MODEL_ORDER:
            metric_cols_by_split[split].extend(
                [
                    f"{split}_clf_roc_{model}",
                    f"{split}_x_roc_{model}",
                    f"{split}_motif_r_{model}",
                    f"{split}_node_r_{model}",
                    f"{split}_motif_points_{model}_mean",
                    f"{split}_node_points_{model}_mean",
                ]
            )

    all_metric_cols = set(sum(metric_cols_by_split.values(), []))
    all_metric_cols |= set(c for c in rows_df.columns if c.endswith("_mean") or c.endswith("_std") or c.endswith("_n"))
    exclude_cols = set(base_cols) | all_metric_cols | {"seed_dirs"}
    cfg_cols = sorted([c for c in rows_df.columns if c not in exclude_cols])

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, metric_cols in metric_cols_by_split.items():
        cols = [c for c in base_cols + metric_cols + cfg_cols if c in rows_df.columns]
        split_df = rows_df[[c for c in cols if c in rows_df.columns]].copy()
        split_df.to_csv(out_dir / f"full_report_{split}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full MotifSAT report tables and correlation grids.")
    parser.add_argument("--dataset", type=str, default="Mutagenicity")
    parser.add_argument("--results_dir", type=str, default=str(Path("../tuning_results")))
    parser.add_argument("--output_dir", type=str, default=str(Path("../full_experiment_report")))
    parser.add_argument("--experiments", nargs="*", default=None, help="Defaults to ALL_EXPERIMENT_NAMES")
    parser.add_argument(
        "--association_csv",
        type=str,
        default=None,
        help="Optional motif association CSV for x_roc fallback (default: ../data/motif_association/<dataset>_fold0_all_motif_class_association.csv).",
    )
    parser.add_argument(
        "--xroc_label_mode",
        type=str,
        default="fisher_q",
        choices=["fisher_q", "abs_delta_topq"],
        help="Label mode for fallback x_roc computation (mirrors motif_stat_vs_importance_roc.py).",
    )
    parser.add_argument(
        "--fisher_q_alpha",
        type=float,
        default=0.05,
        help="Positive motif threshold for fisher_q fallback mode.",
    )
    parser.add_argument(
        "--delta_label_quantile",
        type=float,
        default=0.75,
        help="Positive motif threshold quantile for abs_delta_topq fallback mode.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        from run_mutagenicity_gsat_experiment import ALL_EXPERIMENT_NAMES

        default_experiments = list(ALL_EXPERIMENT_NAMES)
    except Exception:
        default_experiments = []
    experiments = args.experiments if args.experiments else default_experiments
    if not experiments:
        raise ValueError("No experiments specified and ALL_EXPERIMENT_NAMES could not be loaded.")

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    xroc_fallback_ctx = _build_xroc_fallback_context(
        dataset=args.dataset,
        association_csv=args.association_csv,
        label_mode=args.xroc_label_mode,
        fisher_q_alpha=args.fisher_q_alpha,
        delta_label_quantile=args.delta_label_quantile,
    )
    if xroc_fallback_ctx is None:
        print("[INFO] x_roc fallback disabled (association CSV unavailable/invalid).")
    else:
        print(f"[INFO] x_roc fallback enabled using: {xroc_fallback_ctx['assoc_path']}")

    rows, by_experiment_records = build_group_rows(
        results_dir,
        args.dataset,
        experiments,
        xroc_fallback_ctx=xroc_fallback_ctx,
        verbose=args.verbose,
    )
    if not rows:
        print("[WARN] No rows generated. Check dataset/results_dir/experiments.")
        return

    rows_df = pd.DataFrame(rows).sort_values(["experiment", "variant", "model"]).reset_index(drop=True)
    rows_df.to_csv(output_dir / "full_report_all_rows.csv", index=False)
    write_split_tables(rows_df, output_dir)

    plot_rows = build_plot_rows(rows_df, by_experiment_records)
    for split in ("train", "validation", "test"):
        _plot_grid_for_split(split, plot_rows, output_dir / f"motif_level_score_vs_impact_grid_{split}_all_rows.png")

    print(f"[INFO] Wrote report under: {output_dir}")
    print("[INFO] Files:")
    print("  - full_report_all_rows.csv")
    print("  - full_report_train.csv")
    print("  - full_report_validation.csv")
    print("  - full_report_test.csv")
    print("  - motif_level_score_vs_impact_grid_train_all_rows.png")
    print("  - motif_level_score_vs_impact_grid_validation_all_rows.png")
    print("  - motif_level_score_vs_impact_grid_test_all_rows.png")
    if xroc_fallback_ctx is not None:
        print(f"[INFO] x_roc fallback substitutions: {xroc_fallback_ctx['hits']}")


if __name__ == "__main__":
    main()

