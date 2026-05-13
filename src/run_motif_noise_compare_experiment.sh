#!/bin/bash
#SBATCH --job-name=motif_noise_compare
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn

mkdir -p logs
mkdir -p ~/hpc-share/ChemIntuit/MotifSAT/tuning_results
mkdir -p ~/hpc-share/ChemIntuit/MotifSAT/wandb

export RESULTS_DIR=~/hpc-share/ChemIntuit/MotifSAT/tuning_results
export WANDB_DIR=~/hpc-share/ChemIntuit/MotifSAT/wandb

# Slurm runs a copy; anchor to submit dir unless overridden.
if [[ -n "${MOTIFSAT_SRC_DIR:-}" ]]; then
  cd "$MOTIFSAT_SRC_DIR" || exit 1
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT="motif_readout_info0_motif_noise_add_temp1_compare_rerun"
MODELS=(GIN PNA GAT SAGE GCN)
BASE_DATASETS=(Mutagenicity Benzene BBBP)
FOLDS=(0 1)
SEEDS=(0)
# Set GT_ONLY=1 to skip base (non-GT) branch.
GT_ONLY="${GT_ONLY:-0}"

echo "[INFO] Running experiment: ${EXPERIMENT}"
echo "[INFO] Base datasets: ${BASE_DATASETS[*]}"
echo "[INFO] Models: ${MODELS[*]}"
echo "[INFO] Folds: ${FOLDS[*]}"
echo "[INFO] GT_ONLY=${GT_ONLY}"

for DATASET in "${BASE_DATASETS[@]}"; do
  for GT_RELABEL in 0 1; do
    if [[ "$GT_ONLY" -eq 1 && "$GT_RELABEL" -eq 0 ]]; then
      continue
    fi
    if [[ "$GT_RELABEL" -eq 1 ]]; then
      DATASET_ALIAS="${DATASET}_GT_relabeled"
      GT_ARGS=()
      echo "[INFO] Dataset alias=${DATASET_ALIAS} -> run dataset=${DATASET} (GT cache + relabel ON)"
    else
      DATASET_ALIAS="${DATASET}"
      GT_ARGS=(--no_ground_truth_cache)
      echo "[INFO] Dataset alias=${DATASET_ALIAS} -> run dataset=${DATASET} (GT cache OFF)"
    fi

    for FOLD in "${FOLDS[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          echo "============================================================"
          echo "alias=${DATASET_ALIAS} dataset=${DATASET} fold=${FOLD} model=${MODEL} seed=${SEED}"
          echo "============================================================"
          python3 run_mutagenicity_gsat_experiment.py \
            --dataset "${DATASET}" \
            --experiments "${EXPERIMENT}" \
            --folds "${FOLD}" \
            --models "${MODEL}" \
            --seeds "${SEED}" \
            --cuda 0 \
            --embedding_viz_every 10 \
            "${GT_ARGS[@]}"
        done
      done
    done
  done
done

echo "[INFO] Finished motif_noise_compare grid."

