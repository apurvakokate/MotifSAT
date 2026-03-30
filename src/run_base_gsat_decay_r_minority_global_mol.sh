#!/bin/bash
# base_gsat_decay_r_minority_global on molecular MolDataset targets (minority_global motif pickles).
# Folds default to [0,1] in the driver for DATASETS_WITH_MOTIFS (see run_mutagenicity_gsat_experiment.py).
#
# Usage (from MotifSAT/src):
#   ./run_base_gsat_decay_r_minority_global_mol.sh
# Optional env: MODELS="GIN PNA" SEEDS="0 1" CUDA=0 EMBEDDING_VIZ_EVERY=10 MOTIFSAT_SRC_DIR=...

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

EMBEDDING_VIZ_EVERY="${EMBEDDING_VIZ_EVERY:-10}"
MODELS="${MODELS:-GIN PNA GAT SAGE GCN}"
SEEDS="${SEEDS:-0}"
CUDA="${CUDA:-0}"

DATASETS=(
  Mutagenicity
  hERG
  BBBP
  Benzene
  Alkane_Carbonyl
  Fluoride_Carbonyl
  Lipophilicity
  esol
)

echo "[INFO] experiment=base_gsat_decay_r_minority_global"
echo "[INFO] datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "[INFO] models: ${MODELS}"
echo "[INFO] seeds: ${SEEDS} cuda=${CUDA}"

for DATASET in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[INFO] dataset=${DATASET}"
  echo "============================================================"
  # shellcheck disable=SC2086
  python run_mutagenicity_gsat_experiment.py \
    --dataset "${DATASET}" \
    --experiments base_gsat_decay_r_minority_global \
    --models ${MODELS} \
    --seeds ${SEEDS} \
    --cuda "${CUDA}" \
    --embedding_viz_every "${EMBEDDING_VIZ_EVERY}"
done

echo "[INFO] All datasets finished. RESULTS_DIR=${RESULTS_DIR}"
