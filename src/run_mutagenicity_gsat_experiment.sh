#!/usr/bin/env bash
# Run GSAT on Mutagenicity: all folds, all architectures, 4 GSAT variants.
# Variants:
#   1. Node attention, no motif (loss/readout/graph)
#   2. Edge attention, no motif
#   3. Node attention + motif consistency loss (very high)
#   4. Node attention + motif consistency loss (comparable)
#
# Wandb: prediction performances (best/clf_roc_*) and per-motif edge weight
#        distribution (motif_edge_att/min_mean, motif_edge_att/max_mean) are logged.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="Mutagenicity"
FOLDS="${FOLDS:-0 1}"
MODELS="${MODELS:-GIN PNA GAT SAGE GCN}"
CUDA="${CUDA:-0}"
CONFIG_DIR="$SCRIPT_DIR/configs/mutagenicity_gsat"

# Variant name -> config file, and optional --learn_edge_att
run_variant() {
  local fold=$1
  local model=$2
  local variant_name=$3
  local config_file=$4
  local use_edge_att=$5  # "1" = pass --learn_edge_att, "0" = node attention

  echo "=============================================="
  echo "Mutagenicity fold=$fold model=$model variant=$variant_name"
  echo "=============================================="

  CMD="python run_gsat.py --dataset $DATASET --fold $fold --backbone $model --config $CONFIG_DIR/$config_file --cuda $CUDA"
  if [[ "$use_edge_att" == "1" ]]; then
    CMD="$CMD --learn_edge_att"
  fi
  $CMD
}

for fold in $FOLDS; do
  for model in $MODELS; do
    # 1. Node attention, no motif
    run_variant "$fold" "$model" "node_baseline" "node_baseline.yml" "0"
    # 2. Edge attention, no motif
    run_variant "$fold" "$model" "edge_baseline" "edge_baseline.yml" "1"
    # 3. Node + motif loss (high)
    run_variant "$fold" "$model" "node_motif_high" "node_motif_loss_high.yml" "0"
    # 4. Node + motif loss (comparable)
    run_variant "$fold" "$model" "node_motif_comparable" "node_motif_loss_comparable.yml" "0"
  done
done

echo "Done. Check wandb project GSAT-Mutagenicity for best/clf_roc_* and motif_edge_att/*."
