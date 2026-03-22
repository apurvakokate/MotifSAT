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

EXPERIMENTS=(
  # Original explainer experiments
  base_gsat_decay_r_explainer
  motif_readout_decay_r_mean_explainer
  motif_readout_decay_r_mean_sampling_explainer
  # Motif-level info loss variants
  base_gsat_decay_r_explainer_motif_info
  motif_readout_decay_r_mean_explainer_motif_info
  motif_readout_decay_r_mean_sampling_explainer_motif_info
  # Warmup variants (conservative info loss schedule)
  base_gsat_decay_r_explainer_warmup
  motif_readout_decay_r_mean_explainer_warmup
  motif_readout_decay_r_mean_sampling_explainer_warmup
  # Injection point ablation (motif-level sampling)
  motif_injection_node
  motif_injection_node_readout
  motif_injection_readout_only
  motif_injection_edge_readout
  # Info loss coefficient sweep (motif readout + sampling)
  motif_readout_sampling_info_coef_sweep
  # Extractor MLP capacity sweep
  motif_readout_sampling_extractor_sweep
  # Rich motif readout (mean+max+sum concatenation)
  motif_readout_sampling_rich_pool
)
DATASETS=(Mutagenicity)
MODELS=(GCN SAGE)

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