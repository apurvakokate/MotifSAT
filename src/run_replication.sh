#!/bin/bash
# GSAT Replication Experiment Runner
# Based on "Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism" (ICML 2022)
#
# This script runs the baseline GSAT (without motif modifications) across multiple
# architectures and paper datasets to verify compatibility.

set -e

# Configuration
CUDA_DEVICE=${CUDA:-0}
RESULTS_DIR=${RESULTS_DIR:-"../replication_results"}
SEEDS=${SEEDS:-"0 1 2"}

echo "========================================"
echo "GSAT Replication Experiment"
echo "========================================"
echo "CUDA Device: $CUDA_DEVICE"
echo "Results Dir: $RESULTS_DIR"
echo "Seeds: $SEEDS"
echo "========================================"

# Create results directory
mkdir -p $RESULTS_DIR

# Paper datasets
DATASETS=(
    "ba_2motifs"
    "mutag"
    "mnist"
    "spmotif_0.5"
    "spmotif_0.7"
    "spmotif_0.9"
    "Graph-SST2"
    "ogbg_molhiv"
)

# Architectures to test
MODELS=("GIN" "PNA" "GAT" "SAGE" "GCN")

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in $SEEDS; do
            echo ""
            echo "========================================"
            echo "Running: $MODEL on $DATASET (seed $SEED)"
            echo "========================================"
            
            python run_gsat_replication.py \
                --single \
                --model $MODEL \
                --dataset $DATASET \
                --seed $SEED \
                --cuda $CUDA_DEVICE \
                --results_dir $RESULTS_DIR \
                2>&1 | tee -a $RESULTS_DIR/log_${MODEL}_${DATASET}_seed${SEED}.txt
            
            # Check if experiment succeeded
            if [ $? -ne 0 ]; then
                echo "[WARNING] Experiment failed: $MODEL on $DATASET (seed $SEED)"
            fi
        done
    done
done

echo ""
echo "========================================"
echo "Replication Complete!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
