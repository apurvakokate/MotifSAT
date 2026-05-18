#!/usr/bin/env python3
"""
Build per-dataset artifacts for motif-noise compare experiments.

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
import re
from collections import Counter, defaultdict
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
    compute_node_score_impact_correlation_per_graph,
    compute_posthoc_correlation,
    compute_posthoc_correlation_per_graph,
    find_results,
    _resolve_motif_name_from_seed,
)

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_EXPERIMENT_KEYS = [
    "motif_readout_info0_motif_noise_add_temp1_compare_rerun",
    "motif_readout_info0_motif_noise_add_temp1_compare_gt_only",
]
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


def _parse_motif_id(rec: dict[str, Any]) -> int | None:
    raw = rec.get("motif_idx", rec.get("motif_index"))
    if raw is None:
        return None
    try:
        motif_id = int(raw)
    except (TypeError, ValueError):
        return None
    if motif_id < 0:
        return None
    return motif_id


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
    name_by_id: dict[int, str] = {}
    for r in node_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        s = r.get("motif_score", r.get("score"))
        if m is None or s is None:
            continue
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        score_by_m[(m, motif_name)].append(float(s))
    impact_by_m = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        if m is None:
            continue
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if old_p is None or new_p is None:
            continue
        imp = abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p)))
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        impact_by_m[(m, motif_name)].append(float(imp))
    rows = []
    for (motif_id, motif_name), sc in score_by_m.items():
        rows.append(
            {
                "motif_id": int(motif_id),
                "motif_name": motif_name,
                "score_mean": float(np.mean(sc)),
                "score_std": float(np.std(sc, ddof=0)),
                "impact_mean": float(np.mean(impact_by_m[(motif_id, motif_name)]))
                if impact_by_m[(motif_id, motif_name)]
                else np.nan,
                "n_nodes": int(len(sc)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["rank", "motif_id", "motif_name", "score_mean", "score_std", "impact_mean", "n_nodes"])
    df = pd.DataFrame(rows).sort_values("score_mean", ascending=False).head(10).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _bottom10_motifs(seed_dir: Path, split: str = "test") -> pd.DataFrame:
    node_rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    impact_rows = _read_jsonl(seed_dir / "Motif_level_node_and_edge_masking_impact.jsonl")
    score_by_m = defaultdict(list)
    name_by_id: dict[int, str] = {}
    for r in node_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        s = r.get("motif_score", r.get("score"))
        if m is None or s is None:
            continue
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        score_by_m[(m, motif_name)].append(float(s))
    impact_by_m = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        if m is None:
            continue
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if old_p is None or new_p is None:
            continue
        imp = abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p)))
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        impact_by_m[(m, motif_name)].append(float(imp))
    rows = []
    for (motif_id, motif_name), sc in score_by_m.items():
        rows.append(
            {
                "motif_id": int(motif_id),
                "motif_name": motif_name,
                "score_mean": float(np.mean(sc)),
                "score_std": float(np.std(sc, ddof=0)),
                "impact_mean": float(np.mean(impact_by_m[(motif_id, motif_name)]))
                if impact_by_m[(motif_id, motif_name)]
                else np.nan,
                "n_nodes": int(len(sc)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["rank", "motif_id", "motif_name", "score_mean", "score_std", "impact_mean", "n_nodes"])
    df = pd.DataFrame(rows).sort_values("score_mean", ascending=True).head(10).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _gt_vs_non_gt_motif_stats(seed_dir: Path, split: str = "test") -> pd.DataFrame:
    node_rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    impact_rows = _read_jsonl(seed_dir / "Motif_level_node_and_edge_masking_impact.jsonl")
    edge_rows = _read_jsonl(seed_dir / "edge_scores.jsonl")

    node_to_motif: dict[tuple[int, int], tuple[int, str]] = {}
    score_by_inst = defaultdict(list)
    name_by_id: dict[int, str] = {}
    for r in node_rows:
        if r.get("split") != split:
            continue
        g = r.get("graph_idx")
        ni = r.get("node_index")
        m = _parse_motif_id(r)
        s = r.get("motif_score", r.get("score"))
        if g is None or ni is None or m is None or s is None:
            continue
        g = int(g)
        ni = int(ni)
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        node_to_motif[(g, ni)] = (m, motif_name)
        score_by_inst[(g, m, motif_name)].append(float(s))

    impact_by_inst = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        g = r.get("graph_idx", r.get("graph_id"))
        m = _parse_motif_id(r)
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if g is None or m is None or old_p is None or new_p is None:
            continue
        g = int(g)
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        impact_by_inst[(g, m, motif_name)].append(abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p))))

    gt_motifs_by_graph: dict[int, set[int]] = defaultdict(set)
    for r in edge_rows:
        if r.get("split") != split:
            continue
        g = r.get("graph_idx")
        y = r.get("edge_label")
        src = r.get("source")
        dst = r.get("target")
        if g is None or y is None or src is None or dst is None:
            continue
        if float(y) <= 0.5:
            continue
        g = int(g)
        src = int(src)
        dst = int(dst)
        ms = node_to_motif.get((g, src))
        md = node_to_motif.get((g, dst))
        if ms is not None:
            gt_motifs_by_graph[g].add(int(ms[0]))
        if md is not None:
            gt_motifs_by_graph[g].add(int(md[0]))

    gt_scores, gt_impacts = [], []
    non_scores, non_impacts = [], []
    for key, sc in score_by_inst.items():
        g, m, _ = key
        if key not in impact_by_inst:
            continue
        score_mean = float(np.mean(sc))
        impact_mean = float(np.mean(impact_by_inst[key]))
        if m in gt_motifs_by_graph.get(g, set()):
            gt_scores.append(score_mean)
            gt_impacts.append(impact_mean)
        else:
            non_scores.append(score_mean)
            non_impacts.append(impact_mean)

    rows = []
    for kind, scores, impacts in (
        ("gt_motif", gt_scores, gt_impacts),
        ("non_gt_motif", non_scores, non_impacts),
    ):
        s_arr = np.asarray(scores, dtype=float) if scores else np.asarray([], dtype=float)
        i_arr = np.asarray(impacts, dtype=float) if impacts else np.asarray([], dtype=float)
        rows.append(
            {
                "group": kind,
                "n_motif_instances": int(s_arr.size),
                "importance_mean": float(np.mean(s_arr)) if s_arr.size else np.nan,
                "importance_std": float(np.std(s_arr, ddof=0)) if s_arr.size else np.nan,
                "impact_mean": float(np.mean(i_arr)) if i_arr.size else np.nan,
                "impact_std": float(np.std(i_arr, ddof=0)) if i_arr.size else np.nan,
            }
        )
    if gt_scores and non_scores:
        rows.append(
            {
                "group": "delta_gt_minus_non_gt",
                "n_motif_instances": int(len(gt_scores) + len(non_scores)),
                "importance_mean": float(np.mean(gt_scores) - np.mean(non_scores)),
                "importance_std": np.nan,
                "impact_mean": float(np.mean(gt_impacts) - np.mean(non_impacts)),
                "impact_std": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _base_dataset_name(dataset_tag: str) -> str:
    ds = str(dataset_tag)
    for suffix in ("_GT_relabeled", "_GT_relabled", "_GT"):
        if ds.endswith(suffix):
            return ds[: -len(suffix)]
    return ds


def _load_selected_rule(seed_dir: Path, dataset_tag: str) -> dict[str, Any] | None:
    rc_path = seed_dir / "run_config_full.json"
    if not rc_path.exists():
        return None
    try:
        with rc_path.open("r") as f:
            rc = json.load(f)
    except Exception:
        return None
    data_cfg = ((rc.get("run_config") or {}).get("data") or {})
    cache_root = data_cfg.get("ground_truth_cache_root", None)
    if not cache_root:
        return None
    dictionary_fold_variant = str(data_cfg.get("dictionary_fold_variant", "nofilter"))
    relabel = bool(data_cfg.get("ground_truth_relabel_graphs", True))
    relabel_tag = "relabel1" if relabel else "relabel0"
    fold_invariant = bool(data_cfg.get("ground_truth_fold_invariant", True))
    ref_fold = int(data_cfg.get("ground_truth_reference_fold", 0))
    ds_base = _base_dataset_name(dataset_tag)
    sm_path = seed_dir / "experiment_summary.json"
    fold = None
    if sm_path.exists():
        try:
            with sm_path.open("r") as f:
                fold = int(json.load(f).get("fold"))
        except Exception:
            fold = None
    if fold is None:
        fold = 0
    src_fold = ref_fold if fold_invariant else int(fold)
    fold_tag = f"inv{int(fold_invariant)}_src{int(src_fold)}"

    paths = []
    if fold_invariant:
        paths.append(
            Path(cache_root)
            / ds_base
            / "_shared_rules"
            / dictionary_fold_variant
            / f"motif_label_results_{relabel_tag}_inv1_src{int(src_fold)}.json"
        )
    paths.append(
        Path(cache_root)
        / ds_base
        / f"fold{int(fold)}"
        / dictionary_fold_variant
        / f"motif_label_results_{relabel_tag}_{fold_tag}.json"
    )
    for p in paths:
        if p.exists():
            try:
                with p.open("r") as f:
                    payload = json.load(f)
                sr = payload.get("selected_rule")
                if isinstance(sr, dict):
                    sr["_rules_file"] = str(p)
                    return sr
            except Exception:
                continue
    return None


def _rule_satisfaction_tables(seed_dir: Path, dataset_tag: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rule = _load_selected_rule(seed_dir, dataset_tag)
    node_rows = _read_jsonl(seed_dir / "node_scores.jsonl")
    if selected_rule is None:
        return (
            pd.DataFrame([{"note": "selected_rule_not_found", "rules_file": None}]),
            pd.DataFrame([{"note": "selected_rule_not_found", "reason": "no_rule_file", "count": 0}]),
        )

    g_to_motifs: dict[int, set[str]] = defaultdict(set)
    for r in node_rows:
        if r.get("split") != split:
            continue
        g = r.get("graph_idx")
        m = _parse_motif_id(r)
        if g is None or m is None:
            continue
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, {})
        g_to_motifs[int(g)].add(motif_name)

    group_motifs = selected_rule.get("group_motifs", [])
    gates = [str(g).upper() for g in selected_rule.get("gates", [])]
    cross = str(selected_rule.get("cross", "OR")).upper()
    if not group_motifs or not gates or len(group_motifs) != len(gates):
        return (
            pd.DataFrame([{"note": "selected_rule_invalid", "rule": selected_rule.get("rule")}]),
            pd.DataFrame([{"note": "selected_rule_invalid", "reason": "shape_mismatch", "count": 0}]),
        )

    total = max(1, len(g_to_motifs))
    group_ok_counts = Counter()
    miss_reason_counts = Counter()
    missing_motif_counts = Counter()
    rule_ok_count = 0
    for _, present in g_to_motifs.items():
        group_oks = []
        for i, motifs in enumerate(group_motifs):
            motif_set = {str(x) for x in motifs}
            if gates[i] == "AND":
                missing = sorted([m for m in motif_set if m not in present])
                ok = len(missing) == 0
                if not ok:
                    miss_reason_counts[f"group{i}_AND_missing"] += 1
                    for mm in missing:
                        missing_motif_counts[(f"group{i}", mm)] += 1
            else:
                ok = any(m in present for m in motif_set)
                if not ok:
                    miss_reason_counts[f"group{i}_OR_no_match"] += 1
            group_oks.append(ok)
            if ok:
                group_ok_counts[i] += 1
        rule_ok = all(group_oks) if cross == "AND" else any(group_oks)
        if rule_ok:
            rule_ok_count += 1
        else:
            miss_reason_counts[f"cross_{cross}_not_satisfied"] += 1

    sat_rows = [
        {
            "rule": selected_rule.get("rule"),
            "rules_file": selected_rule.get("_rules_file"),
            "metric": "rule_satisfied",
            "count": int(rule_ok_count),
            "pct": float(rule_ok_count) * 100.0 / float(total),
        }
    ]
    for gi in range(len(group_motifs)):
        c = int(group_ok_counts.get(gi, 0))
        sat_rows.append(
            {
                "rule": selected_rule.get("rule"),
                "rules_file": selected_rule.get("_rules_file"),
                "metric": f"group{gi}_satisfied_{gates[gi]}",
                "count": c,
                "pct": float(c) * 100.0 / float(total),
            }
        )
    miss_rows = []
    for reason, c in miss_reason_counts.most_common():
        miss_rows.append({"reason": reason, "count": int(c)})
    for (gk, mm), c in missing_motif_counts.most_common(25):
        miss_rows.append({"reason": f"{gk}_missing_motif::{mm}", "count": int(c)})
    if not miss_rows:
        miss_rows = [{"reason": "none", "count": 0}]
    return pd.DataFrame(sat_rows), pd.DataFrame(miss_rows)


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
    name_by_id: dict[int, str] = {}
    for r in node_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        s = r.get("motif_score", r.get("score"))
        if m is None or s is None:
            continue
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        score_by_m[(m, motif_name)].append(float(s))
    impact_by_m = defaultdict(list)
    for r in impact_rows:
        if r.get("split") != split:
            continue
        m = _parse_motif_id(r)
        old_p = r.get("old_prediction")
        new_p = r.get("new_prediction")
        if m is None or old_p is None or new_p is None:
            continue
        motif_name = _resolve_motif_name_from_seed(seed_dir, r, m, name_by_id)
        impact_by_m[(m, motif_name)].append(abs(_sigmoid(float(new_p)) - _sigmoid(float(old_p))))
    common = sorted(set(score_by_m.keys()) & set(impact_by_m.keys()))
    if not common:
        return None, None
    xs = np.asarray([float(np.mean(score_by_m[m])) for m in common], dtype=float)
    ys = np.asarray([float(np.mean(impact_by_m[m])) for m in common], dtype=float)
    if xs.size == 0 or ys.size == 0:
        return None, None
    return xs, ys


def _sanitize_tag(x: str) -> str:
    return str(x).replace("/", "_").replace(" ", "_")


def _plot_grid(rep: dict[tuple[str, str], Path], row_keys: list[str], out_png: Path, level: str = "motif") -> None:
    n_rows = len(row_keys)
    n_cols = len(MODEL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle(f"{level.title()} importance vs impact (test)", fontsize=12, fontweight="bold")
    for i, rk in enumerate(row_keys):
        for j, m in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            ax.set_xlim(0.0, 1.0)
            sd = rep.get((rk, m))
            if i == 0:
                ax.set_title(m, fontsize=9)
            if j == 0:
                ax.set_ylabel(rk, fontsize=8)
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
    experiment_keys: list[str],
    log_to_wandb: bool = True,
) -> Path:
    dataset_tag = _dataset_to_results_tag(dataset_name)
    recs: list[dict[str, Any]] = []
    for exp in experiment_keys:
        rows = find_results(results_dir, exp, dataset=dataset_tag, verbose=False)
        for r in rows:
            rr = dict(r)
            rr["experiment"] = exp
            recs.append(rr)
    out_dir = output_root / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not recs:
        with (out_dir / "README.txt").open("w") as f:
            f.write(f"No records found for {dataset_tag} / {experiment_keys}\n")
        return out_dir

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rep_seed: dict[tuple[str, str, str], tuple[Path, float]] = {}
    row_keys: list[str] = []
    for r in recs:
        exp = str(r.get("experiment", "unknown_experiment"))
        p = str(r.get("row_val", "unknown"))
        m = str(r.get("model", ""))
        if p not in PIPELINE_ORDER or m not in MODEL_ORDER:
            continue
        rk = f"{exp} | {PIPELINE_LABEL.get(p, p)}"
        if rk not in row_keys:
            row_keys.append(rk)
        grouped[(exp, p, m)].append(r)
        rv = r.get("metrics", {}).get("metric/best_clf_roc_valid", np.nan)
        rv = float(rv) if np.isfinite(rv) else -1e18
        cur = rep_seed.get((exp, p, m))
        if cur is None or rv > cur[1]:
            rep_seed[(exp, p, m)] = (Path(r["seed_dir"]), rv)

    pred_rows, corr_rows, roc_rows, stat_rows = [], [], [], []
    gt_non_gt_rows = []
    rule_sat_rows = []
    rule_miss_rows = []
    top_tables = {}
    rep_paths: dict[tuple[str, str], Path] = {}

    for exp in experiment_keys:
        for p in PIPELINE_ORDER:
            for m in MODEL_ORDER:
                cell = grouped.get((exp, p, m), [])
                if not cell:
                    continue
                row_key = f"{exp} | {PIPELINE_LABEL.get(p, p)}"
                for split, key in (
                    ("train", "metric/best_clf_roc_train"),
                    ("validation", "metric/best_clf_roc_valid"),
                    ("test", "metric/best_clf_roc_test"),
                ):
                    mu, sd, n = _mean_std([c.get("metrics", {}).get(key, np.nan) for c in cell])
                    pred_rows.append(
                        {
                            "dataset": dataset_tag,
                            "experiment": exp,
                            "pipeline": p,
                            "model": m,
                            "split": split,
                            "mean": mu,
                            "std": sd,
                            "n_runs": n,
                        }
                    )

                for split in ("train", "valid", "test"):
                    motif_vals, node_vals = [], []
                    motif_graph_vals, node_graph_vals = [], []
                    motif_graph_counts, node_graph_counts = [], []
                    for c in cell:
                        sd_dir = Path(c["seed_dir"])
                        motif_r, _, _ = compute_posthoc_correlation(sd_dir, split=split)
                        node_r, _, _ = compute_node_score_impact_correlation(sd_dir, split=split)
                        motif_graph_r, _, motif_graph_n = compute_posthoc_correlation_per_graph(sd_dir, split=split)
                        node_graph_r, _, node_graph_n = compute_node_score_impact_correlation_per_graph(sd_dir, split=split)
                        motif_vals.append(motif_r)
                        node_vals.append(node_r)
                        motif_graph_vals.append(motif_graph_r)
                        node_graph_vals.append(node_graph_r)
                        motif_graph_counts.append(motif_graph_n)
                        node_graph_counts.append(node_graph_n)
                    m_mu, m_sd, m_n = _mean_std(motif_vals)
                    n_mu, n_sd, n_n = _mean_std(node_vals)
                    mg_mu, mg_sd, mg_n = _mean_std(motif_graph_vals)
                    ng_mu, ng_sd, ng_n = _mean_std(node_graph_vals)
                    corr_rows.append(
                        {
                            "dataset": dataset_tag,
                            "experiment": exp,
                            "pipeline": p,
                            "model": m,
                            "split": "validation" if split == "valid" else split,
                            "motif_corr_mean": m_mu,
                            "motif_corr_std": m_sd,
                            "motif_corr_n": m_n,
                            "node_corr_mean": n_mu,
                            "node_corr_std": n_sd,
                            "node_corr_n": n_n,
                            "motif_graph_corr_mean": mg_mu,
                            "motif_graph_corr_std": mg_sd,
                            "motif_graph_corr_n_runs": mg_n,
                            "motif_graph_corr_n_graphs_total": int(sum(int(v) for v in motif_graph_counts if v is not None)),
                            "node_graph_corr_mean": ng_mu,
                            "node_graph_corr_std": ng_sd,
                            "node_graph_corr_n_runs": ng_n,
                            "node_graph_corr_n_graphs_total": int(sum(int(v) for v in node_graph_counts if v is not None)),
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
                        "experiment": exp,
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

                best = rep_seed.get((exp, p, m))
                if best is None:
                    continue
                sd_path = best[0]
                rep_paths[(row_key, m)] = sd_path
                top_df = _top10_motifs(sd_path, split="test")
                top_tables[(exp, p, m)] = top_df
                if not top_df.empty:
                    top_df.to_csv(
                        out_dir / f"top10_motifs_{_sanitize_tag(exp)}_{_sanitize_tag(p)}_{m}.csv",
                        index=False,
                    )
                bottom_df = _bottom10_motifs(sd_path, split="test")
                if not bottom_df.empty:
                    bottom_df.to_csv(
                        out_dir / f"bottom10_motifs_{_sanitize_tag(exp)}_{_sanitize_tag(p)}_{m}.csv",
                        index=False,
                    )

                gt_df = _gt_vs_non_gt_motif_stats(sd_path, split="test")
                if not gt_df.empty:
                    gt_df = gt_df.copy()
                    gt_df.insert(0, "dataset", dataset_tag)
                    gt_df.insert(1, "experiment", exp)
                    gt_df.insert(2, "pipeline", p)
                    gt_df.insert(3, "model", m)
                    gt_non_gt_rows.extend(gt_df.to_dict(orient="records"))
                    gt_df.to_csv(
                        out_dir / f"gt_vs_non_gt_motif_stats_{_sanitize_tag(exp)}_{_sanitize_tag(p)}_{m}.csv",
                        index=False,
                    )

                sat_df, miss_df = _rule_satisfaction_tables(sd_path, dataset_tag, split="test")
                if not sat_df.empty:
                    sat_df = sat_df.copy()
                    sat_df.insert(0, "dataset", dataset_tag)
                    sat_df.insert(1, "experiment", exp)
                    sat_df.insert(2, "pipeline", p)
                    sat_df.insert(3, "model", m)
                    rule_sat_rows.extend(sat_df.to_dict(orient="records"))
                    sat_df.to_csv(
                        out_dir / f"rule_satisfaction_{_sanitize_tag(exp)}_{_sanitize_tag(p)}_{m}.csv",
                        index=False,
                    )
                if not miss_df.empty:
                    miss_df = miss_df.copy()
                    miss_df.insert(0, "dataset", dataset_tag)
                    miss_df.insert(1, "experiment", exp)
                    miss_df.insert(2, "pipeline", p)
                    miss_df.insert(3, "model", m)
                    rule_miss_rows.extend(miss_df.to_dict(orient="records"))
                    miss_df.to_csv(
                        out_dir / f"rule_miss_reasons_{_sanitize_tag(exp)}_{_sanitize_tag(p)}_{m}.csv",
                        index=False,
                    )

                mlp_stats = _mlp_logit_stats(sd_path, split="test")
                noise_stats = _noise_stats(sd_path)
                stat_rows.append(
                    {
                        "dataset": dataset_tag,
                        "experiment": exp,
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
    gt_non_gt_df = pd.DataFrame(gt_non_gt_rows)
    rule_sat_df = pd.DataFrame(rule_sat_rows)
    rule_miss_df = pd.DataFrame(rule_miss_rows)
    pred_df.to_csv(out_dir / "prediction_performance.csv", index=False)
    corr_df.to_csv(out_dir / "explainer_correlation.csv", index=False)
    roc_df.to_csv(out_dir / "explainer_roc.csv", index=False)
    stats_df.to_csv(out_dir / "logit_and_noise_stats.csv", index=False)
    gt_non_gt_df.to_csv(out_dir / "gt_vs_non_gt_motif_stats.csv", index=False)
    rule_sat_df.to_csv(out_dir / "rule_satisfaction.csv", index=False)
    rule_miss_df.to_csv(out_dir / "rule_miss_reasons.csv", index=False)

    _plot_grid(rep_paths, row_keys, out_dir / "motif_level_importance_vs_impact.png", level="motif")
    _plot_grid(rep_paths, row_keys, out_dir / "node_level_importance_vs_impact.png", level="node")

    summary = {
        "dataset": dataset_tag,
        "experiments": experiment_keys,
        "pipelines": PIPELINE_ORDER,
        "row_keys": row_keys,
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
                config={"dataset": dataset_tag, "experiments": experiment_keys},
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
                if not gt_non_gt_df.empty:
                    wandb.log({"posthoc/gt_vs_non_gt_motif_stats": wandb.Table(dataframe=gt_non_gt_df)})
                if not rule_sat_df.empty:
                    wandb.log({"posthoc/rule_satisfaction": wandb.Table(dataframe=rule_sat_df)})
                if not rule_miss_df.empty:
                    wandb.log({"posthoc/rule_miss_reasons": wandb.Table(dataframe=rule_miss_df)})
                for img_name in ("motif_level_importance_vs_impact.png", "node_level_importance_vs_impact.png"):
                    p = out_dir / img_name
                    if p.exists():
                        wandb.log({f"posthoc/{img_name}": wandb.Image(str(p))})
                artifact = wandb.Artifact(
                    name=f"{dataset_tag}_motif_noise_compare",
                    type="analysis",
                    metadata={"dataset": dataset_tag, "experiments": experiment_keys},
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
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENT_KEYS,
        help="Experiment names to include in merged artifact tables/plots.",
    )
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
            experiment_keys=[str(x) for x in args.experiments],
            log_to_wandb=(not args.no_wandb),
        )
        print(f"[INFO] Wrote dataset artifact: {out}")


if __name__ == "__main__":
    main()

