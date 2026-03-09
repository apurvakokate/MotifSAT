#!/bin/bash
#SBATCH --job-name=gsat_motifsat
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
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

EXPERIMENTS=(vanilla_gnn_node_repaired base_gsat_fix_r_node_repaired base_gsat_decay_r_node_repaired)
DATASETS=(Mutagenicity)
MODELS=(GAT)

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Running dataset=${DATASET} model=${MODEL}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "============================================================"

    python run_mutagenicity_gsat_experiment.py \
      --dataset "${DATASET}" \
      --models "${MODEL}" \
      --experiments "${EXPERIMENTS[@]}" \
      --seeds 0 \
      --cuda 0
  done
done