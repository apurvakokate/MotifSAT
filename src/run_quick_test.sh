#!/bin/bash
# Quick Test: Run one experiment per architecture on BA-2Motifs
# Use this to verify all architectures work before full replication

set -e

CUDA_DEVICE=${CUDA:-0}
RESULTS_DIR="../replication_test"
DATASET="ba_2motifs"
SEED=0

echo "========================================"
echo "GSAT Quick Architecture Test"
echo "========================================"
echo "Dataset: $DATASET"
echo "CUDA Device: $CUDA_DEVICE"
echo "========================================"

mkdir -p $RESULTS_DIR

MODELS=("GIN" "PNA" "GAT" "SAGE" "GCN")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Testing $MODEL..."
    
    python run_gsat_replication.py \
        --single \
        --model $MODEL \
        --dataset $DATASET \
        --seed $SEED \
        --cuda $CUDA_DEVICE \
        --results_dir $RESULTS_DIR
    
    if [ $? -eq 0 ]; then
        echo "[OK] $MODEL completed successfully"
    else
        echo "[FAIL] $MODEL failed"
    fi
done

echo ""
echo "========================================"
echo "Quick Test Complete!"
echo "========================================"
