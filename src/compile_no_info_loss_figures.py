#!/usr/bin/env python3
"""
Compile no-info-loss comparison figures from local `epoch_metrics.jsonl` / `final_metrics.json`
and optional Weights & Biases edge-attention histograms.

Row ↔ tuning_id mapping (stochastic attention, experiment `no_info_loss`):
  - Row 0 — base GSAT: `no_info_loss_base`
  - Row 1 — motif readout max_mean + node-level motif sampling: `no_info_loss_maxmean_node_samp`
  - Row 2 — motif readout max_mean + motif-level sampling: `no_info_loss_maxmean_motif_samp`

Deterministic attention (`no_attention_sampling=True`, experiment `no_info_loss_deterministic_attn`)
has only two registered variants (no separate deterministic motif-sampling row):
  - Row 0 — base: `no_info_loss_det_base`
  - Row 1 — readout max_mean + node-level motif sampling: `no_info_loss_det_maxmean`

See `EXPERIMENT_GROUPS['no_info_loss']` and `EXPERIMENT_GROUPS['no_info_loss_deterministic_attn']`
in `run_mutagenicity_gsat_experiment.py`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Experiment × variant registry
# ---------------------------------------------------------------------------

STOCHASTIC_ROWS: List[Tuple[str, str, str]] = [
    ("Base", "no_info_loss", "no_info_loss_base"),
    ("Readout + node samp.", "no_info_loss", "no_info_loss_maxmean_node_samp"),
    ("Readout + motif samp.", "no_info_loss", "no_info_loss_maxmean_motif_samp"),
]

DETERMINISTIC_ROWS: List[Tuple[str, str, str]] = [
    ("Base", "no_info_loss_deterministic_attn", "no_info_loss_det_base"),
    ("Readout + node samp.", "no_info_loss_deterministic_attn", "no_info_loss_det_maxmean"),
]

CURVE_METRICS: Tuple[str, ...] = ("train_loss", "valid_loss", "train_auroc", "valid_auroc")
ALL_METRICS: Tuple[str, ...] = CURVE_METRICS + ("edge_dist",)


def _default_results_dir() -> str:
    return os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "tuning_results"))


def _default_out_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "figures")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_seed_dir(
    results_dir: str,
    dataset: str,
    model: str,
    experiment_name: str,
    tuning_id: str,
    fold: int,
    seed: int,
) -> Optional[str]:
    """
    Glob under tuning_id (deep path with method_* / pred* / init* segments) for fold{fold}_seed{seed}.
    If multiple matches, pick the shortest path and warn on stderr.
    """
    pattern = os.path.join(
        results_dir,
        dataset,
        f"model_{model}",
        f"experiment_{experiment_name}",
        f"tuning_{tuning_id}",
        "**",
        f"fold{fold}_seed{seed}",
    )
    matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    if not matches:
        return None
    matches.sort(key=lambda p: (len(p), p))
    chosen = matches[0]
    if len(matches) > 1:
        print(
            f"[WARN] Multiple seed dirs for {dataset} {model} {experiment_name} {tuning_id} "
            f"fold{fold}_seed{seed}; using shortest path:\n  {chosen}\n"
            f"  (also: {matches[1:3]}{'...' if len(matches) > 3 else ''})",
            file=sys.stderr,
        )
    return chosen


# ---------------------------------------------------------------------------
# epoch_metrics.jsonl
# ---------------------------------------------------------------------------


def load_epoch_curves(path: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Returns:
      {'train': {'loss': [(epoch, v), ...], 'clf_roc': [...]},
       'valid': {...}}
    Multiple records per (epoch, phase) keep the last seen value for that epoch.
    """
    by_phase: Dict[str, Dict[str, Dict[int, float]]] = {
        "train": {"loss": {}, "clf_roc": {}},
        "valid": {"loss": {}, "clf_roc": {}},
    }
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ph = rec.get("phase")
            if ph not in ("train", "valid"):
                continue
            ep = int(rec["epoch"])
            if "loss" in rec and rec["loss"] is not None:
                by_phase[ph]["loss"][ep] = float(rec["loss"])
            if rec.get("clf_roc") is not None:
                by_phase[ph]["clf_roc"][ep] = float(rec["clf_roc"])
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for ph in ("train", "valid"):
        out[ph] = {}
        for key in ("loss", "clf_roc"):
            d = by_phase[ph][key]
            out[ph][key] = sorted((e, d[e]) for e in sorted(d.keys()))
    return out


# ---------------------------------------------------------------------------
# attention_distributions.jsonl (local fallback)
# ---------------------------------------------------------------------------


def load_last_attention_dist_summary(seed_dir: str, phase: str = "valid") -> Optional[Dict[str, Any]]:
    path = os.path.join(seed_dir, "attention_distributions.jsonl")
    if not os.path.isfile(path):
        return None
    last: Optional[Dict[str, Any]] = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") == phase:
                last = rec
    return last


# ---------------------------------------------------------------------------
# W&B edge histogram
# ---------------------------------------------------------------------------


def _try_import_wandb():
    try:
        import wandb  # noqa: F401

        return wandb
    except ImportError:
        return None


def wandb_display_name(model: str, fold: int, seed: int, tuning_id: str) -> str:
    return f"{model}-fold{fold}-seed{seed}-{tuning_id}"


def _decode_wandb_histogram_value(val: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (counts, bin_edges) for plotting with plt.stairs or bar centers.
    Handles several wandb / API serializations.
    """
    if val is None:
        return None
    # Summary / history: nested dict
    if isinstance(val, dict):
        if val.get("_type") == "histogram" and "values" in val:
            # Often counts only; bins may be separate
            vals = val["values"]
            bins = val.get("bins")
            if bins is not None and len(bins) >= 2:
                c = np.asarray(vals, dtype=float)
                e = np.asarray(bins, dtype=float)
                if c.size == e.size - 1:
                    return c, e
        # bins + values / counts
        if "bins" in val and ("values" in val or "counts" in val):
            e = np.asarray(val["bins"], dtype=float)
            c = np.asarray(val.get("values", val.get("counts")), dtype=float)
            if c.size == e.size - 1:
                return c, e
        # np_histogram style
        if "np_histogram" in val:
            nh = val["np_histogram"]
            if isinstance(nh, (list, tuple)) and len(nh) == 2:
                c, e = nh
                c = np.asarray(c, dtype=float)
                e = np.asarray(e, dtype=float)
                if c.size == e.size - 1:
                    return c, e
    return None


def fetch_edge_histogram_wandb(
    wandb_mod: Any,
    entity: Optional[str],
    project: str,
    display_name: str,
    phase: str = "valid",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if wandb_mod is None:
        return None
    try:
        api = wandb_mod.Api()
    except Exception:
        return None
    if entity:
        path = f"{entity}/{project}"
    else:
        path = project
    try:
        runs = api.runs(path, filters={"display_name": display_name})
    except Exception:
        return None
    if not runs:
        return None
    run = runs[0]
    key = f"{phase}/edge_att/distribution"
    # Prefer summary (last logged value often aggregated)
    summary = getattr(run, "summary", None) or {}
    if key in summary:
        h = _decode_wandb_histogram_value(summary[key])
        if h is not None:
            return h
    # History: last row with non-null histogram
    try:
        hist = run.history(keys=[key], pandas=False)
    except Exception:
        hist = None
    if hist:
        last_val = None
        for row in hist:
            v = row.get(key)
            if v is not None:
                last_val = v
        if last_val is not None:
            h = _decode_wandb_histogram_value(last_val)
            if h is not None:
                return h
    return None


def plot_edge_histogram_on_axis(
    ax: plt.Axes,
    seed_dir: str,
    wandb_mod: Any,
    entity: Optional[str],
    project: str,
    display_name: str,
    color: str = "C0",
    alpha: float = 0.85,
) -> str:
    """
    Plot edge attention distribution on ax.
    Returns how the panel was filled: 'wandb', 'local', or '' if nothing drawn.
    """
    wh = fetch_edge_histogram_wandb(wandb_mod, entity, project, display_name, phase="valid")
    if wh is not None:
        counts, edges = wh
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        ax.bar(centers, counts, width=widths, align="center", alpha=alpha, color=color)
        return "wandb"
    summ = load_last_attention_dist_summary(seed_dir, phase="valid")
    if summ is None:
        return ""
    mean = summ.get("mean")
    std = summ.get("std")
    if mean is None:
        return ""
    ax.bar([0], [mean], yerr=[std] if std is not None else None, capsize=4, color=color, alpha=alpha)
    ax.set_xticks([0])
    ax.set_xticklabels(["mean±std"])
    return "local"


# ---------------------------------------------------------------------------
# final_metrics + motif_readout_analysis.json
# ---------------------------------------------------------------------------


def load_final_metrics(seed_dir: str) -> Dict[str, Any]:
    p = os.path.join(seed_dir, "final_metrics.json")
    with open(p, "r") as f:
        return json.load(f)


def load_motif_readout_json(seed_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(seed_dir, "motif_readout_analysis.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def get_marginal_scalar(
    fm: Dict[str, Any],
    motif_json: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[float]]:
    """explainer/fidelity_minus for base; else motif readout Pearson/Spearman from metrics or JSON."""
    v = fm.get("explainer/fidelity_minus")
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        return "explainer/fidelity_minus", float(v)
    for k in (
        "motif_readout/pearson_sigma_m_impact",
        "motif_readout/spearman_sigma_m_impact",
    ):
        x = fm.get(k)
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            return k, float(x)
    if motif_json:
        for k in (
            "pearson_sigma_m_impact",
            "spearman_sigma_m_impact",
            "motif_readout/pearson_sigma_m_impact",
        ):
            if k in motif_json and motif_json[k] is not None:
                return k, float(motif_json[k])
    return "marginal", None


def best_run_max_valid(
    rows: Sequence[Tuple[str, str, str]],
    results_dir: str,
    dataset: str,
    model: str,
    folds: Sequence[int],
    seed: int,
) -> Optional[Tuple[float, str, Dict[str, Any], Optional[Dict[str, Any]]]]:
    """(best_valid, seed_dir, final_metrics, motif_json)"""
    best_val = -1.0
    best_dir: Optional[str] = None
    best_fm: Optional[Dict[str, Any]] = None
    best_mj: Optional[Dict[str, Any]] = None
    for _, exp, tid in rows:
        for fold in folds:
            sd = resolve_seed_dir(results_dir, dataset, model, exp, tid, fold, seed)
            if sd is None or not os.path.isfile(os.path.join(sd, "final_metrics.json")):
                continue
            fm = load_final_metrics(sd)
            v = fm.get("metric/best_clf_roc_valid")
            if v is None:
                continue
            v = float(v)
            if v > best_val:
                best_val = v
                best_dir = sd
                best_fm = fm
                best_mj = load_motif_readout_json(sd)
    if best_dir is None or best_fm is None:
        return None
    return best_val, best_dir, best_fm, best_mj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_curve_metric(
    ax: plt.Axes,
    metric_key: str,
    fold_curves: Dict[int, Dict[str, Dict[str, List[Tuple[int, float]]]]],
    ylabel: str,
) -> None:
    style = [("-", "C0"), ("--", "C1")]
    for i, fold in enumerate(sorted(fold_curves.keys())):
        curves = fold_curves[fold]
        if metric_key == "train_loss":
            series = curves["train"]["loss"]
        elif metric_key == "valid_loss":
            series = curves["valid"]["loss"]
        elif metric_key == "train_auroc":
            series = curves["train"]["clf_roc"]
        elif metric_key == "valid_auroc":
            series = curves["valid"]["clf_roc"]
        else:
            series = []
        ls, col = style[i % len(style)]
        if series:
            xs = [t[0] for t in series]
            ys = [t[1] for t in series]
            ax.plot(xs, ys, linestyle=ls, color=col, label=f"fold {fold}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7, loc="best")


def compile_metric_figure(
    metric: str,
    rows: Sequence[Tuple[str, str, str]],
    results_dir: str,
    dataset: str,
    model: str,
    folds: Sequence[int],
    seed: int,
    wandb_mod: Any,
    wandb_entity: Optional[str],
    wandb_project: str,
    title_prefix: str,
) -> plt.Figure:
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 1, figsize=(6.5, 2.4 * nrows), squeeze=False)
    metric_titles = {
        "train_loss": ("Train loss", "loss"),
        "valid_loss": ("Valid loss", "loss"),
        "train_auroc": ("Train AUROC", "clf_roc"),
        "valid_auroc": ("Valid AUROC", "clf_roc"),
        "edge_dist": ("Edge att. distribution (valid)", ""),
    }
    for r, (row_label, exp, tid) in enumerate(rows):
        ax = axes[r, 0]
        if metric == "edge_dist":
            sd0 = resolve_seed_dir(results_dir, dataset, model, exp, tid, folds[0], seed)
            kind0 = ""
            if sd0 is not None:
                wname0 = wandb_display_name(model, folds[0], seed, tid)
                kind0 = plot_edge_histogram_on_axis(
                    ax, sd0, wandb_mod, wandb_entity, wandb_project, wname0, color="C0", alpha=0.85
                )
                if kind0 == "wandb":
                    ax.set_ylabel("count (fold 0)")
                elif kind0 == "local":
                    ax.set_ylabel("edge att (approx., fold 0)")
            if len(folds) > 1:
                sd1 = resolve_seed_dir(results_dir, dataset, model, exp, tid, folds[1], seed)
                if sd1 is not None:
                    ax1 = ax.twinx()
                    wname1 = wandb_display_name(model, folds[1], seed, tid)
                    kind1 = plot_edge_histogram_on_axis(
                        ax1, sd1, wandb_mod, wandb_entity, wandb_project, wname1, color="C1", alpha=0.55
                    )
                    if kind1 == "wandb":
                        ax1.set_ylabel("count (fold 1)", color="C1", fontsize=8)
                    elif kind1 == "local":
                        ax1.set_ylabel("edge att (approx., fold 1)", color="C1", fontsize=8)
            if not kind0 and sd0 is None:
                ax.text(0.5, 0.5, "missing run dir", ha="center", va="center", transform=ax.transAxes)
            elif not kind0:
                ax.text(0.5, 0.5, "no edge histogram", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{row_label} — {tid}", fontsize=9)
        else:
            fold_curves: Dict[int, Dict[str, Dict[str, List[Tuple[int, float]]]]] = {}
            for fold in folds:
                sd = resolve_seed_dir(results_dir, dataset, model, exp, tid, fold, seed)
                if sd is None:
                    continue
                ep = os.path.join(sd, "epoch_metrics.jsonl")
                if not os.path.isfile(ep):
                    continue
                fold_curves[fold] = load_epoch_curves(ep)
            if not fold_curves:
                ax.text(0.5, 0.5, "missing epoch_metrics", ha="center", va="center", transform=ax.transAxes)
            else:
                _plot_curve_metric(ax, metric, fold_curves, metric_titles[metric][1])
            ax.set_title(f"{row_label} — {tid}", fontsize=9)
    fig.suptitle(f"{title_prefix} — {metric_titles[metric][0]}", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def compile_overview_figure(
    rows: Sequence[Tuple[str, str, str]],
    results_dir: str,
    dataset: str,
    model: str,
    folds: Sequence[int],
    seed: int,
    wandb_mod: Any,
    wandb_entity: Optional[str],
    wandb_project: str,
    title: str,
) -> plt.Figure:
    """rows × 5 columns: train_loss, valid_loss, train_auroc, valid_auroc, edge_dist."""
    nrows = len(rows)
    ncols = len(ALL_METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.3 * nrows), squeeze=False)
    for r, (row_label, exp, tid) in enumerate(rows):
        for c, metric in enumerate(ALL_METRICS):
            ax = axes[r, c]
            if metric == "edge_dist":
                any_drawn = False
                # Overlay all requested folds in the same edge-distribution cell.
                # First fold uses the base axis; additional folds use twinx to avoid
                # conflicting y-scales when histogram counts differ by fold.
                for fi, fold in enumerate(folds):
                    sd = resolve_seed_dir(results_dir, dataset, model, exp, tid, fold, seed)
                    if sd is None:
                        continue
                    target_ax = ax if fi == 0 else ax.twinx()
                    wname = wandb_display_name(model, fold, seed, tid)
                    k = plot_edge_histogram_on_axis(
                        target_ax,
                        sd,
                        wandb_mod,
                        wandb_entity,
                        wandb_project,
                        wname,
                        color=f"C{fi % 10}",
                        alpha=0.65 if fi == 0 else 0.45,
                    )
                    if k:
                        any_drawn = True
                        if k == "wandb":
                            target_ax.set_ylabel(f"count (f{fold})", fontsize=7)
                        else:
                            target_ax.set_ylabel(f"approx. (f{fold})", fontsize=7)
                if not any_drawn:
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
            else:
                fold_curves = {}
                for fold in folds:
                    sd = resolve_seed_dir(results_dir, dataset, model, exp, tid, fold, seed)
                    if sd is None:
                        continue
                    ep = os.path.join(sd, "epoch_metrics.jsonl")
                    if os.path.isfile(ep):
                        fold_curves[fold] = load_epoch_curves(ep)
                if fold_curves:
                    ylab = {"train_loss": "loss", "valid_loss": "loss", "train_auroc": "AUROC", "valid_auroc": "AUROC"}[
                        metric
                    ]
                    _plot_curve_metric(ax, metric, fold_curves, ylab)
                else:
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
            if r == 0:
                ax.set_title(metric.replace("_", " "), fontsize=8)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def render_scalar_table(
    regime: str,
    rows: Sequence[Tuple[str, str, str]],
    results_dir: str,
    dataset: str,
    model: str,
    folds: Sequence[int],
    seed: int,
    out_png: str,
    csv_path: Optional[str],
) -> None:
    br = best_run_max_valid(rows, results_dir, dataset, model, folds, seed)
    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.axis("off")
    if br is None:
        ax.text(0.5, 0.5, "No final_metrics found for grid.", ha="center", va="center")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    _, sd, fm, mj = br
    marg_k, marg_v = get_marginal_scalar(fm, mj)
    metric_keys = [
        "metric/best_clf_roc_train",
        "metric/best_clf_roc_valid",
        "metric/best_clf_roc_test",
        "metric/best_x_roc_test",
    ]
    cols = ["best run path"] + metric_keys + [marg_k]
    display_path = sd if len(sd) <= 96 else sd[:44] + " … " + sd[-44:]
    row_vals: List[str] = [display_path]
    for k in metric_keys:
        v = fm.get(k)
        row_vals.append(f"{v:.4f}" if isinstance(v, (float, int)) else str(v))
    row_vals.append(f"{marg_v:.4f}" if marg_v is not None else "—")

    table_data = [cols, row_vals]
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax.set_title(f"{regime} — best by metric/best_clf_roc_valid (all variants × folds)", fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if csv_path:
        csv_row = [sd] + row_vals[1:]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow(csv_row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default=_default_results_dir())
    p.add_argument("--dataset", type=str, default="Mutagenicity")
    p.add_argument("--model", type=str, default="GIN", help="Subfolder model_<name> in results tree.")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=_default_out_dir())
    p.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-project", type=str, default=None, help="Default: GSAT-<dataset>")
    p.add_argument("--no-wandb", action="store_true", help="Skip W&B API; edge panels use local fallback only.")
    p.add_argument("--overview", action="store_true", help="Also write no_info_loss_overview_{stoch|det}.png (rows×5 cols).")
    p.add_argument("--csv", action="store_true", help="Write scalar table CSV next to PNG.")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_mod = None if args.no_wandb else _try_import_wandb()
    if not args.no_wandb and wandb_mod is None:
        print("[WARN] wandb not installed; using local edge fallbacks only.", file=sys.stderr)

    project = args.wandb_project or f"GSAT-{args.dataset}"

    def run_regime(
        name: str,
        rows: List[Tuple[str, str, str]],
        tag: str,
    ) -> None:
        title_p = f"{args.dataset} {args.model} seed={args.seed} ({name})"
        for metric in ALL_METRICS:
            fig = compile_metric_figure(
                metric,
                rows,
                args.results_dir,
                args.dataset,
                args.model,
                args.folds,
                args.seed,
                wandb_mod,
                args.wandb_entity,
                project,
                title_p,
            )
            outp = out_dir / f"no_info_loss_{metric}_{tag}.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Wrote {outp}")
        if args.overview:
            fig_o = compile_overview_figure(
                rows,
                args.results_dir,
                args.dataset,
                args.model,
                args.folds,
                args.seed,
                wandb_mod,
                args.wandb_entity,
                project,
                title_p + " overview",
            )
            op = out_dir / f"no_info_loss_overview_{tag}.png"
            fig_o.savefig(op, dpi=150, bbox_inches="tight")
            plt.close(fig_o)
            print(f"[INFO] Wrote {op}")
        csv_path = (out_dir / f"no_info_loss_scalar_table_{tag}.csv") if args.csv else None
        render_scalar_table(
            name,
            rows,
            args.results_dir,
            args.dataset,
            args.model,
            args.folds,
            args.seed,
            str(out_dir / f"no_info_loss_scalar_table_{tag}.png"),
            str(csv_path) if csv_path else None,
        )
        print(f"[INFO] Wrote {out_dir / f'no_info_loss_scalar_table_{tag}.png'}")

    run_regime("no_info_loss (stochastic attention)", STOCHASTIC_ROWS, "stochastic")
    run_regime("no_info_loss_deterministic_attn", DETERMINISTIC_ROWS, "deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
