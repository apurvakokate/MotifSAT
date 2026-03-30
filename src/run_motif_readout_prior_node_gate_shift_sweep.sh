#!/bin/bash
# motif_readout_prior_node_gate: runs 4 variants (motif_prior_shift_scale 0, 0.1, 0.5, 1.0)
# in one driver invocation (see EXPERIMENT_GROUPS in run_mutagenicity_gsat_experiment.py).
#
# Usage (from MotifSAT/src):
#   ./run_motif_readout_prior_node_gate_shift_sweep.sh
# Env: DATASET, MODELS, SEEDS, CUDA, EMBEDDING_VIZ_EVERY, MOTIFSAT_SRC_DIR, RESULTS_DIR, WANDB_DIR

set -euo pipefail

if [[ -n "${MOTIFSAT_SRC_DIR:-}" ]]; then
  cd "$MOTIFSAT_SRC_DIR" || exit 1
fi

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate l2xgnn 2>/dev/null || true

mkdir -p logs
export RESULTS_DIR="${RESULTS_DIR:-../tuning_results}"
export WANDB_DIR="${WANDB_DIR:-../wandb}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="${DATASET:-Mutagenicity}"
MODELS="${MODELS:-GIN PNA GAT SAGE GCN}"
SEEDS="${SEEDS:-0}"
CUDA="${CUDA:-0}"
EMBEDDING_VIZ_EVERY="${EMBEDDING_VIZ_EVERY:-10}"

echo "[INFO] dataset=${DATASET} experiment=motif_readout_prior_node_gate (shift scales 0, 0.1, 0.5, 1.0)"
# shellcheck disable=SC2086
python run_mutagenicity_gsat_experiment.py \
  --dataset "${DATASET}" \
  --experiments motif_readout_prior_node_gate \
  --models ${MODELS} \
  --seeds ${SEEDS} \
  --cuda "${CUDA}" \
  --embedding_viz_every "${EMBEDDING_VIZ_EVERY}"

echo "[INFO] Done."
