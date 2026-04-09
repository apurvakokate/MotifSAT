#!/bin/bash
#SBATCH --job-name=gsat_motifsat_full
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Grid is defined below in bash only (edit arrays to balance cluster load). No Python imports for lists.
# Fold rule: datasets in DATASETS_TWO_FOLDS get folds 0 and 1; others get fold 0 only (matches driver DATASETS_WITH_MOTIFS).
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

# Slurm runs a *copy* of this script under /var/spool/slurmd/job*/ — never cd to dirname(BASH_SOURCE).
# Default job cwd is $SLURM_SUBMIT_DIR (where you ran sbatch), like your older script that never cd'd.
if [[ -n "${MOTIFSAT_SRC_DIR:-}" ]]; then
  cd "$MOTIFSAT_SRC_DIR" || exit 1
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
echo "[INFO] MotifSAT cwd: $(pwd)"

# --- Edit these arrays (must match ALL_EXPERIMENT_NAMES in run_mutagenicity_gsat_experiment.py) ---
EXPERIMENTS=(
  vanilla_gnn
  base_gsat_fix_r
  base_gsat_decay_r
  base_gsat_decay_r_minority_global
  base_gsat_decay_r_injection
  base_gsat_motif_loss
  motif_readout_decay_w_message
  no_info_loss
  motif_readout_decay_injection_ablation
  base_gsat_readout_intra_att
  motif_readout_prior_node_gate
  motif_readout_prior_node_gate_tanh_sched
  motif_readout_weight_diversity
  motif_readout_baseline_f07
  motif_readout_e1_logit_standardize
  motif_readout_e2_temperature
  motif_readout_e3_max_pool
  motif_readout_e4_max_mean_pool
  motif_readout_e5_interp_head
  motif_readout_e6_no_gate
  motif_readout_e7_multiplicative_gate
  motif_readout_e8_entropy_sweep
  motif_readout_e9_motif_ib_sweep
  motif_readout_e10_align_sweep
  motif_readout_entropy_pool_sweep
  motif_readout_maxmean_node_vs_edge_att
  motif_readout_pred_info_only
  factored_motif_attention_grid
  factored_motif_additive
  simplified_factored_motif_additive
  simplified_motif_readout
  simplified_motif_readout_maxmean
  simplified_motif_readout_maxmean_z1
  simplified_motif_readout_maxmean_injection_ablation
  simplified_motif_readout_maxmean_info_loss_ablation
)

# Subset or reorder for load balancing. Full SUPPORTED_DATASETS order in driver: molecular + OGB + PAPER_DATASETS (deduped).
DATASETS=(
  Mutagenicity
  BBBP
  hERG
  Benzene
  Alkane_Carbonyl
  Fluoride_Carbonyl
  ogbg_molhiv
  ogbg_molbace
  ogbg_molbbbp
  ogbg_molclintox
  ogbg_moltox21
  ogbg_molsider
  ba_2motifs
  mutag
  mnist
  spmotif_0.5
  spmotif_0.7
  spmotif_0.9
  Graph-SST2
)

MODELS=(
  GIN
  PNA
  GAT
  SAGE
  GCN
)

# Datasets that use fold 0 and 1 (see DATASETS_WITH_MOTIFS in run_mutagenicity_gsat_experiment.py).
DATASETS_TWO_FOLDS=(
  Mutagenicity
  BBBP
  hERG
  Benzene
  Alkane_Carbonyl
  Fluoride_Carbonyl
  esol
  Lipophilicity
)

_dataset_uses_two_folds() {
  local d="$1"
  local x
  for x in "${DATASETS_TWO_FOLDS[@]}"; do
    [[ "$x" == "$d" ]] && return 0
  done
  return 1
}

echo "[INFO] Experiments (${#EXPERIMENTS[@]}): ${EXPERIMENTS[*]}"
echo "[INFO] Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "[INFO] Models (${#MODELS[@]}): ${MODELS[*]}"
echo "[INFO] embedding_viz_every=${EMBEDDING_VIZ_EVERY}"

for DATASET in "${DATASETS[@]}"; do
  if _dataset_uses_two_folds "$DATASET"; then
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
