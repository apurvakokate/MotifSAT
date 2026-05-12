#!/bin/bash
set -euo pipefail

# Detect and recover corrupted JSONL artifacts caused by duplicate launches.
#
# Usage:
#   # 1) Dry-run scan + rerun commands (no deletion)
#   bash src/recover_motif_noise_compare_duplicates.sh
#
#   # 2) Delete only export artifacts in affected seed dirs
#   APPLY=1 bash src/recover_motif_noise_compare_duplicates.sh
#
#   # 3) Restrict to one dataset tag under RESULTS_DIR
#   DATASET_FILTER=Mutagenicity bash src/recover_motif_noise_compare_duplicates.sh
#
# Environment:
#   RESULTS_DIR      root results directory (default: ~/hpc-share/ChemIntuit/MotifSAT/tuning_results)
#   EXPERIMENT_NAME  experiment directory key without "experiment_" prefix
#                    (default: motif_readout_info0_motif_noise_add_temp1_compare)
#   APPLY            1 => delete stale export JSONLs in affected seed dirs
#   DATASET_FILTER   optional exact dataset tag under RESULTS_DIR (e.g., Mutagenicity_GT_relabeled)

RESULTS_DIR="${RESULTS_DIR:-$HOME/hpc-share/ChemIntuit/MotifSAT/tuning_results}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-motif_readout_info0_motif_noise_add_temp1_compare}"
APPLY="${APPLY:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"

echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[INFO] APPLY=${APPLY}"
echo "[INFO] DATASET_FILTER=${DATASET_FILTER:-<none>}"
echo

python3 - <<'PY' "$RESULTS_DIR" "$EXPERIMENT_NAME" "$APPLY" "$DATASET_FILTER"
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

results_dir = Path(sys.argv[1]).expanduser().resolve()
exp_name = sys.argv[2]
apply = str(sys.argv[3]).strip() == "1"
dataset_filter = sys.argv[4].strip()

if not results_dir.exists():
    print(f"[ERROR] RESULTS_DIR does not exist: {results_dir}")
    sys.exit(1)

exp_dir_token = f"experiment_{exp_name}"

def parse_seed_dir_from_jsonl_path(p: Path):
    """
    Extract (dataset_tag, model, fold, seed, seed_dir) from:
      .../<dataset>/model_<M>/experiment_<E>/.../fold<k>_seed<s>/<file>.jsonl
    """
    parts = p.parts
    fold_idx = None
    for i, x in enumerate(parts):
        if re.fullmatch(r"fold\d+_seed\d+", x):
            fold_idx = i
            break
    if fold_idx is None:
        return None
    seed_dir = Path(*parts[: fold_idx + 1])
    if fold_idx < 1:
        return None
    fold_seed = parts[fold_idx]
    m = re.fullmatch(r"fold(\d+)_seed(\d+)", fold_seed)
    if not m:
        return None
    fold = int(m.group(1))
    seed = int(m.group(2))

    dataset_tag = None
    model = None
    for i, x in enumerate(parts):
        if x.startswith("model_"):
            model = x.replace("model_", "", 1)
            if i - 1 >= 0:
                dataset_tag = parts[i - 1]
            break
    if dataset_tag is None or model is None:
        return None
    return dataset_tag, model, fold, seed, seed_dir


def base_dataset_and_gt_args(dataset_tag: str):
    """
    Map result namespace tag back to run_mutagenicity_gsat_experiment.py args.
    """
    if dataset_tag.endswith("_GT_relabeled"):
        base = dataset_tag[: -len("_GT_relabeled")]
        extra = []
    elif dataset_tag.endswith("_GT"):
        base = dataset_tag[: -len("_GT")]
        extra = ["--no_ground_truth_relabel_graphs"]
    else:
        base = dataset_tag
        extra = ["--no_ground_truth_cache"]
    return base, extra


# Scan malformed lines
bad_lines_by_file = {}
affected = {}  # key -> metadata
jsonl_files = list(results_dir.rglob("*.jsonl"))
for jf in jsonl_files:
    s = str(jf)
    if exp_dir_token not in s:
        continue
    if dataset_filter and f"/{dataset_filter}/" not in s:
        continue
    bad = 0
    try:
        with jf.open("r") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except Exception:
                    bad += 1
    except Exception:
        # unreadable file counts as bad
        bad += 1
    if bad:
        bad_lines_by_file[jf] = bad
        parsed = parse_seed_dir_from_jsonl_path(jf)
        if parsed is None:
            continue
        dataset_tag, model, fold, seed, seed_dir = parsed
        key = (dataset_tag, model, fold, seed, str(seed_dir))
        affected[key] = {
            "dataset_tag": dataset_tag,
            "model": model,
            "fold": fold,
            "seed": seed,
            "seed_dir": seed_dir,
        }

print(f"[INFO] Malformed JSONL files: {len(bad_lines_by_file)}")
for p, n in sorted(bad_lines_by_file.items()):
    print(f"  {n:4d} bad lines :: {p}")
print()

if not affected:
    print("[INFO] No affected seed dirs detected.")
    sys.exit(0)

print(f"[INFO] Affected seed dirs: {len(affected)}")

grouped = defaultdict(list)
for k, v in affected.items():
    grouped[(v["dataset_tag"], v["model"], v["fold"], v["seed"])].append(v["seed_dir"])

for (dataset_tag, model, fold, seed), dirs in sorted(grouped.items()):
    print(f"  - dataset={dataset_tag} model={model} fold={fold} seed={seed}  (dirs={len(dirs)})")
print()

# Optional cleanup
targets = [
    "node_scores.jsonl",
    "edge_scores.jsonl",
    "Individual_node_node_masking_impact.jsonl",
    "Individual_edge_node_and_edge_masking_impact.jsonl",
    "Motif_level_node_masking_impact.jsonl",
    "Motif_level_node_and_edge_masking_impact.jsonl",
    "motif_sampling_stats.jsonl",
]

if apply:
    removed = 0
    print("[INFO] APPLY=1 => deleting export artifacts in affected seed dirs...")
    for meta in affected.values():
        sd = meta["seed_dir"]
        for name in targets:
            p = sd / name
            if p.exists():
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"[WARN] Failed to delete {p}: {e}")
    print(f"[INFO] Deleted files: {removed}")
    print()
else:
    print("[INFO] Dry-run only. Set APPLY=1 to delete stale export artifacts.")
    print()

# Print rerun commands grouped by dataset/model/fold/seed
print("[INFO] Rerun commands (copy/paste):")
for (dataset_tag, model, fold, seed), _dirs in sorted(grouped.items()):
    base_ds, extra = base_dataset_and_gt_args(dataset_tag)
    extra_s = " ".join(extra)
    cmd = (
        "python3 run_mutagenicity_gsat_experiment.py "
        f"--dataset \"{base_ds}\" "
        f"--experiments \"{exp_name}\" "
        f"--folds {fold} "
        f"--models {model} "
        f"--seeds {seed} "
        "--cuda 0 "
        "--embedding_viz_every 10 "
        f"{extra_s}"
    ).strip()
    print(cmd)

print()
print("[NOTE] The experiment group reruns all 3 variants for that dataset/model/fold/seed.")
print("[NOTE] If only affected seed dirs were cleaned, unaffected variants should be skipped by artifact checks.")
PY

