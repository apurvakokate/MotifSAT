#!/bin/bash
# GSAT decay-r grid using motif dictionaries under FOLDS/minority_global/..._*_minority_global
# (see DataLoader.DICTIONARY_FOLD_VARIANTS and experiment base_gsat_decay_r_minority_global).
# The same experiment is included in run_mutagenicity_gsat_experiment.sh (full EXPERIMENTS list).
# Official GSAT reference: https://github.com/Graph-COM/GSAT
#
# Usage (from MotifSAT/src, after conda activate):
#   ./run_mutagenicity_gsat_minority_global.sh
# Or with sbatch: copy SBATCH headers from run_mutagenicity_gsat_experiment.sh as needed.

set -euo pipefail

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate l2xgnn 2>/dev/null || true

mkdir -p logs
export RESULTS_DIR="${RESULTS_DIR:-../tuning_results}"
export WANDB_DIR="${WANDB_DIR:-../wandb}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

EMBEDDING_VIZ_EVERY="${EMBEDDING_VIZ_EVERY:-10}"

python run_mutagenicity_gsat_experiment.py \
  --dataset "${DATASET:-Mutagenicity}" \
  --experiments base_gsat_decay_r_minority_global \
  --models "${MODELS:-GIN}" \
  --seeds "${SEEDS:-0}" \
  --cuda "${CUDA:-0}" \
  --embedding_viz_every "$EMBEDDING_VIZ_EVERY"

echo "[INFO] Done. Results under RESULTS_DIR=$RESULTS_DIR"
