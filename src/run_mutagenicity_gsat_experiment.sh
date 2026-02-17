#!/usr/bin/env bash
# Run GSAT on Mutagenicity using the same configs as run_gsat_replication
# (via experiment_configs.get_base_config). All folds, all architectures, 4 variants.
#
# Variants: node_baseline, edge_baseline, node_motif_loss_high, node_motif_loss_comparable
# Wandb: best/clf_roc_*, motif_edge_att/*

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FOLDS="${FOLDS:-0 1}"
MODELS="${MODELS:-GIN PNA GAT SAGE GCN}"
CUDA="${CUDA:-0}"
SEEDS="${SEEDS:-0}"

python run_mutagenicity_gsat_experiment.py \
  --folds $FOLDS \
  --models $MODELS \
  --seeds $SEEDS \
  --cuda "$CUDA"
