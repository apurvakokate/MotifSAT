#!/usr/bin/env python3
"""
Build per-dataset artifacts for motif_readout_info0_motif_noise_add_temp1_compare_rerun.

Outputs one artifact directory per dataset with:
  - Prediction performance (train/valid/test) by pipeline x model
  - Explainer score-impact correlations (motif/node)
  - Explainer ROC (all edges; correctly predicted class-1 edges)
  - Top-10 motifs tables
  - MLP logit and motif-noise stats
  - Motif/node importance-vs-impact plot grids
and logs them to W&B.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from collect_mutagenicity_tables import (
    compute_node_score_impact_correlation,
    compute_posthoc_correlation,
    find_results,
)

try:
    import wandb
except Exception:
    wandb = None


EXPERIMENT_KEY = "motif_readout_info0_motif_noise_add_temp1_compare_rerun"
MODEL_ORDER = ["GAT", "GCN", "GIN", "PNA", "SAGE"]
PIPELINE_ORDER = ["beta_clamped", "beta_unclamped", "base_decay_r07"]
PIPELINE_LABEL = {
    "beta_clamped": "beta_clamped",
    "beta_unclamped": "beta_unclamped",
    "base_decay_r07": "base_decay_r07",
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _mean_std(xs: list[float]) -> tuple[float | None, float | None, int]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), int(arr.size)


def _dataset_to_results_tag(dataset_name: str) -> str:
    ds = str(dataset_name).strip()
    if ds.endswith("_GT_relabled"):
        return ds.replace("_GT_relabled", "_GT_relabeled")
    return ds


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _edge_roc(seed_dir: Path, split: str = "test") -> tuple[float | None, float | None]:
    rows = _read_jsonl(seed_dir / "edge_scores.jsonl")
    if not rows:
        return None, None
    all_y, all_s = [], []
    pos_y, pos_s = [], []
    for r in rows:
        if r.get("split") != split:
            continue
        y = r.get("edge_label")
        s = r.get("score")
        if y is None or s is None:
            continue
        y = int(float(y) > 0.5)
        s = float(s)
        all_y.append(y)
        all_s.append(s)
        is_cls1_correct = int(r.get("graph_label", -1)) == 1 and int(r.get("graph_correct", 0)) == 1
        if is_cls1_correct:
            pos_y.append(y)
            pos_s.append(s)
    auc_all = None
    auc_pos = None
    if len(set(all_y)) > 1:
        auc_all = float(roc_auc_score(all_y, all_s))
    if len(set(pos_y)) > 1:
        auc_pos = float(roc_auc_score(pos_y, pos_s))
    return auc_all, auc_pos


def _mlp_logit_stats(seed_dir: Path, split: str = "test") -> dict[str, float | None]:
    rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    vals = []
    for r in rows:
        if r.get("split") != split:
            continue
        v = r.get("motif_logit")
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _noise_stats(seed_dir: Path) -> dict[str, float | None]:
    stats_rows = _read_jsonl(seed_dir / "motif_sampling_stats.jsonl")
    for r in reversed(stats_rows):
        if r.get("phase") == "train" and "motif_noise_mean" in r:
            return {
                "mean": float(r.get("motif_noise_mean")),
                "std": float(r.get("motif_noise_std")),
                "min": float(r.get("motif_noise_min")),
                "max": float(r.get("motif_noise_max")),
            }
    fm = seed_dir / "final_metrics.json"
    if fm.exists():
        try:
            with fm.open("r") as f:
                m = json.load(f)
            return {
                "mean": m.get("motif_sampling/train/motif_noise_mean"),
                "std": m.get("motif_sampling/train/motif_noise_std"),
                "min": m.get("motif_sampling/train/motif_noise_min"),
                "max": m.get("motif_sampling/train/motif_noise_max"),
            }
        except Exception:
            pass
    return {"mean": None, "std": None, "min": None, "max": None}


def _top10_motifs(seed_dir: Path, split: str = "test") -> pd.DataFrame:
    node_rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    impact_rows = _read_jsonl(seed_dir / "Motif_level_node_and_edge_masking_impact.jsonl")
    score_by_m = defaultdict(list)
    for r in node_rows:
        if r.get("split") != split:
            continue
        m = r.get("motif_index")
        s = r.get("motif_score", r.get("score"))
        if m is None or s is None:
            continue
        score_by_m[int(m)].append(float(s))
    impact_by_m = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        m = r.get("motif_idx", r.get("motif_index"))
        if m is None:
            continue
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if old_p is None or new_p is None:
            continue
        imp = abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p)))
        impact_by_m[int(m)].append(float(imp))
    rows = []
    for m, sc in score_by_m.items():
        rows.append(
            {
                "motif_index": int(m),
                "score_mean": float(np.mean(sc)),
                "score_std": float(np.std(sc, ddof=0)),
                "impact_mean": float(np.mean(impact_by_m[m])) if impact_by_m[m] else np.nan,
                "n_nodes": int(len(sc)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["rank", "motif_index", "score_mean", "score_std", "impact_mean", "n_nodes"])
    df = pd.DataFrame(rows).sort_values("score_mean", ascending=False).head(10).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _node_points(seed_dir: Path, split: str = "test") -> tuple[np.ndarray | None, np.ndarray | None]:
    rows = _read_jsonl(seed_dir / "Individual_node_node_masking_impact.jsonl")
    xs, ys = [], []
    for r in rows:
        if r.get("split") != split:
            continue
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        s = r.get("score")
        if old_p is None or new_p is None or s is None:
            continue
        xs.append(float(s))
        ys.append(abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p))))
    if not xs:
        return None, None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _motif_points(seed_dir: Path, split: str = "test") -> tuple[np.ndarray | None, np.ndarray | None]:
    node_rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    impact_rows = _read_jsonl(seed_dir / "Motif_level_node_and_edge_masking_impact.jsonl")
    score_by_m = defaultdict(list)
    for r in node_rows:
        if r.get("split") != split:
            continue
        m = r.get("motif_index")
        s = r.get("motif_score", r.get("score"))
        if m is None or s is None:
            continue
        score_by_m[int(m)].append(float(s))
    impact_by_m = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        m = r.get("motif_idx", r.get("motif_index"))
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if m is None or old_p is None or new_p is None:
            continue
        impact_by_m[int(m)].append(abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p))))
    common = sorted(set(score_by_m.keys()) & set(impact_by_m.keys()))
    if not common:
        return None, None
    xs = np.asarray([float(np.mean(score_by_m[m])) for m in common], dtype=float)
    ys = np.asarray([float(np.mean(impact_by_m[m])) for m in common], dtype=float)
    if xs.size == 0 or ys.size == 0:
        return None, None
    return xs, ys


def _plot_grid(rep: dict[tuple[str, str], Path], pipelines: list[str], out_png: Path, level: str = "motif") -> None:
    n_rows = len(pipelines)
    n_cols = len(MODEL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle(f"{level.title()} importance vs impact (test)", fontsize=12, fontweight="bold")
    for i, p in enumerate(pipelines):
        for j, m in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            sd = rep.get((p, m))
            if i == 0:
                ax.set_title(m, fontsize=9)
            if j == 0:
                ax.set_ylabel(PIPELINE_LABEL.get(p, p), fontsize=8)
            if sd is None:
                ax.text(0.5, 0.5, "No run", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if level == "motif":
                xs, ys = _motif_points(sd, split="test")
            else:
                xs, ys = _node_points(sd, split="test")
            if xs is None or ys is None or len(xs) == 0:
                ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.scatter(xs, ys, s=8, alpha=0.3, edgecolors="none")
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=6)
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_dataset_artifact(
    results_dir: Path,
    output_root: Path,
    dataset_name: str,
    log_to_wandb: bool = True,
) -> Path:
    dataset_tag = _dataset_to_results_tag(dataset_name)
    recs = find_results(results_dir, EXPERIMENT_KEY, dataset=dataset_tag, verbose=False)
    out_dir = output_root / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not recs:
        with (out_dir / "README.txt").open("w") as f:
            f.write(f"No records found for {dataset_tag} / {EXPERIMENT_KEY}\n")
        return out_dir

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rep_seed: dict[tuple[str, str], tuple[Path, float]] = {}
    for r in recs:
        p = str(r.get("row_val", "unknown"))
        m = str(r.get("model", ""))
        if p not in PIPELINE_ORDER or m not in MODEL_ORDER:
            continue
        grouped[(p, m)].append(r)
        rv = r.get("metrics", {}).get("metric/best_clf_roc_valid", np.nan)
        rv = float(rv) if np.isfinite(rv) else -1e18
        cur = rep_seed.get((p, m))
        if cur is None or rv > cur[1]:
            rep_seed[(p, m)] = (Path(r["seed_dir"]), rv)

    pred_rows, corr_rows, roc_rows, stat_rows = [], [], [], []
    top_tables = {}
    rep_paths: dict[tuple[str, str], Path] = {}

    for p in PIPELINE_ORDER:
        for m in MODEL_ORDER:
            cell = grouped.get((p, m), [])
            if not cell:
                continue
            for split, key in (
                ("train", "metric/best_clf_roc_train"),
                ("validation", "metric/best_clf_roc_valid"),
                ("test", "metric/best_clf_roc_test"),
            ):
                mu, sd, n = _mean_std([c.get("metrics", {}).get(key, np.nan) for c in cell])
                pred_rows.append(
                    {"dataset": dataset_tag, "pipeline": p, "model": m, "split": split, "mean": mu, "std": sd, "n_runs": n}
                )

            for split in ("train", "valid", "test"):
                motif_vals, node_vals = [], []
                for c in cell:
                    sd_dir = Path(c["seed_dir"])
                    motif_r, _, _ = compute_posthoc_correlation(sd_dir, split=split)
                    node_r, _, _ = compute_node_score_impact_correlation(sd_dir, split=split)
                    motif_vals.append(motif_r)
                    node_vals.append(node_r)
                m_mu, m_sd, m_n = _mean_std(motif_vals)
                n_mu, n_sd, n_n = _mean_std(node_vals)
                corr_rows.append(
                    {
                        "dataset": dataset_tag,
                        "pipeline": p,
                        "model": m,
                        "split": "validation" if split == "valid" else split,
                        "motif_corr_mean": m_mu,
                        "motif_corr_std": m_sd,
                        "motif_corr_n": m_n,
                        "node_corr_mean": n_mu,
                        "node_corr_std": n_sd,
                        "node_corr_n": n_n,
                    }
                )

            auc_all_vals, auc_pos_vals = [], []
            for c in cell:
                a_all, a_pos = _edge_roc(Path(c["seed_dir"]), split="test")
                auc_all_vals.append(a_all if a_all is not None else np.nan)
                auc_pos_vals.append(a_pos if a_pos is not None else np.nan)
            a_mu, a_sd, a_n = _mean_std(auc_all_vals)
            p_mu, p_sd, p_n = _mean_std(auc_pos_vals)
            roc_rows.append(
                {
                    "dataset": dataset_tag,
                    "pipeline": p,
                    "model": m,
                    "split": "test",
                    "explainer_roc_all_mean": a_mu,
                    "explainer_roc_all_std": a_sd,
                    "explainer_roc_all_n": a_n,
                    "explainer_roc_cls1_correct_mean": p_mu,
                    "explainer_roc_cls1_correct_std": p_sd,
                    "explainer_roc_cls1_correct_n": p_n,
                }
            )

            best = rep_seed.get((p, m))
            if best is None:
                continue
            sd_path = best[0]
            rep_paths[(p, m)] = sd_path
            top_df = _top10_motifs(sd_path, split="test")
            top_tables[(p, m)] = top_df
            if not top_df.empty:
                top_df.to_csv(out_dir / f"top10_motifs_{p}_{m}.csv", index=False)

            mlp_stats = _mlp_logit_stats(sd_path, split="test")
            noise_stats = _noise_stats(sd_path)
            stat_rows.append(
                {
                    "dataset": dataset_tag,
                    "pipeline": p,
                    "model": m,
                    "mlp_logit_mean": mlp_stats["mean"],
                    "mlp_logit_std": mlp_stats["std"],
                    "mlp_logit_min": mlp_stats["min"],
                    "mlp_logit_max": mlp_stats["max"],
                    "noise_logit_mean": noise_stats["mean"],
                    "noise_logit_std": noise_stats["std"],
                    "noise_logit_min": noise_stats["min"],
                    "noise_logit_max": noise_stats["max"],
                    "representative_seed_dir": str(sd_path),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    corr_df = pd.DataFrame(corr_rows)
    roc_df = pd.DataFrame(roc_rows)
    stats_df = pd.DataFrame(stat_rows)
    pred_df.to_csv(out_dir / "prediction_performance.csv", index=False)
    corr_df.to_csv(out_dir / "explainer_correlation.csv", index=False)
    roc_df.to_csv(out_dir / "explainer_roc.csv", index=False)
    stats_df.to_csv(out_dir / "logit_and_noise_stats.csv", index=False)

    _plot_grid(rep_paths, PIPELINE_ORDER, out_dir / "motif_level_importance_vs_impact.png", level="motif")
    _plot_grid(rep_paths, PIPELINE_ORDER, out_dir / "node_level_importance_vs_impact.png", level="node")

    summary = {
        "dataset": dataset_tag,
        "experiment": EXPERIMENT_KEY,
        "pipelines": PIPELINE_ORDER,
        "models": MODEL_ORDER,
        "files": sorted([p.name for p in out_dir.iterdir()]),
    }
    with (out_dir / "artifact_manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if log_to_wandb and wandb is not None:
        try:
            run = wandb.init(
                project=f"GSAT-{dataset_tag}-posthoc",
                name=f"{dataset_tag}-motif-noise-compare-artifact",
                reinit=True,
                config={"dataset": dataset_tag, "experiment": EXPERIMENT_KEY},
            )
            if run is not None:
                if not pred_df.empty:
                    wandb.log({"posthoc/prediction_performance": wandb.Table(dataframe=pred_df)})
                if not corr_df.empty:
                    wandb.log({"posthoc/explainer_correlation": wandb.Table(dataframe=corr_df)})
                if not roc_df.empty:
                    wandb.log({"posthoc/explainer_roc": wandb.Table(dataframe=roc_df)})
                if not stats_df.empty:
                    wandb.log({"posthoc/logit_noise_stats": wandb.Table(dataframe=stats_df)})
                for img_name in ("motif_level_importance_vs_impact.png", "node_level_importance_vs_impact.png"):
                    p = out_dir / img_name
                    if p.exists():
                        wandb.log({f"posthoc/{img_name}": wandb.Image(str(p))})
                artifact = wandb.Artifact(
                    name=f"{dataset_tag}_motif_noise_compare",
                    type="analysis",
                    metadata={"dataset": dataset_tag, "experiment": EXPERIMENT_KEY},
                )
                artifact.add_dir(str(out_dir))
                wandb.log_artifact(artifact)
                wandb.finish()
        except Exception as e:
            print(f"[WARN] W&B logging failed for {dataset_tag}: {e}")

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Collect dataset artifacts for motif-noise compare experiment.")
    p.add_argument("--results_dir", type=str, default="../tuning_results")
    p.add_argument("--output_dir", type=str, default="../dataset_artifacts/motif_noise_compare")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "Mutagenicity",
            "Mutagenicity_GT_relabled",
            "Benzene",
            "Benzene_GT_relabled",
            "BBBP",
            "BBBP_GT_relabled",
        ],
    )
    p.add_argument("--no_wandb", action="store_true", default=False)
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        out = build_dataset_artifact(
            results_dir=results_dir,
            output_root=output_dir,
            dataset_name=ds,
            log_to_wandb=(not args.no_wandb),
        )
        print(f"[INFO] Wrote dataset artifact: {out}")


if __name__ == "__main__":
    main()

