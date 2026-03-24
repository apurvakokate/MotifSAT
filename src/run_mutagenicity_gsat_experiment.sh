#!/bin/bash
#SBATCH --job-name=gsat_motifsat_full
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Full grid: all experiment groups × SUPPORTED_DATASETS × ARCHITECTURES × dataset-specific folds.
# W&B: PCA node/motif embedding panels + per-motif wandb.Table (full motif_name) for distribution checks.
# Override: EMBEDDING_VIZ_EVERY=0 to disable embedding logs (helps if the job is OOM-killed).
# Mem 128G: embedding PCA + large batches can spike RSS; use 0 or lower EMBEDDING_VIZ_EVERY if needed.

set -euo pipefail

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn

mkdir -p logs
mkdir -p ~/hpc-share/ChemIntuit/MotifSAT/tuning_results
mkdir -p ~/hpc-share/ChemIntuit/MotifSAT/wandb

export RESULTS_DIR=~/hpc-share/ChemIntuit/MotifSAT/tuning_results
export WANDB_DIR=~/hpc-share/ChemIntuit/MotifSAT/wandb

EMBEDDING_VIZ_EVERY="${EMBEDDING_VIZ_EVERY:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Populate arrays from Python (keeps shell in sync with run_mutagenicity_gsat_experiment.py / experiment_configs)
EXPERIMENTS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && EXPERIMENTS+=("$line")
done < <(python3 -c "from run_mutagenicity_gsat_experiment import EXPERIMENT_GROUPS; print('\n'.join(EXPERIMENT_GROUPS.keys()))")

DATASETS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && DATASETS+=("$line")
done < <(python3 -c "from run_mutagenicity_gsat_experiment import SUPPORTED_DATASETS; print('\n'.join(SUPPORTED_DATASETS))")

MODELS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && MODELS+=("$line")
done < <(python3 -c "from experiment_configs import ARCHITECTURES; print('\n'.join(ARCHITECTURES))")

echo "[INFO] Experiments (${#EXPERIMENTS[@]}): ${EXPERIMENTS[*]}"
echo "[INFO] Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "[INFO] Models (${#MODELS[@]}): ${MODELS[*]}"
echo "[INFO] embedding_viz_every=${EMBEDDING_VIZ_EVERY}"

for DATASET in "${DATASETS[@]}"; do
  if python3 -c "from run_mutagenicity_gsat_experiment import DATASETS_WITH_MOTIFS; import sys; sys.exit(0 if '${DATASET}' in DATASETS_WITH_MOTIFS else 1)"; then
    FOLDS=(0 1)
  else
    FOLDS=(0)
  fi

  for FOLD in "${FOLDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      echo "============================================================"
      echo "dataset=${DATASET} fold=${FOLD} model=${MODEL}"
      echo "RESULTS_DIR=${RESULTS_DIR} WANDB_DIR=${WANDB_DIR}"
      echo "============================================================"

      python3 run_mutagenicity_gsat_experiment.py \
        --dataset "${DATASET}" \
        --folds "${FOLD}" \
        --models "${MODEL}" \
        --experiments "${EXPERIMENTS[@]}" \
        --seeds 0 \
        --cuda 0 \
        --embedding_viz_every "${EMBEDDING_VIZ_EVERY}"
    done
  done
done

echo "[INFO] Full grid finished."
