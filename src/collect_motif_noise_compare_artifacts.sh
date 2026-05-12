#!/bin/bash
set -euo pipefail

export RESULTS_DIR="${RESULTS_DIR:-$HOME/hpc-share/ChemIntuit/MotifSAT/tuning_results}"
export OUTPUT_DIR="${OUTPUT_DIR:-../dataset_artifacts/motif_noise_compare}"

python3 collect_motif_noise_compare_artifacts.py \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --datasets \
    Mutagenicity Mutagenicity_GT_relabled \
    Benzene Benzene_GT_relabled \
    BBBP BBBP_GT_relabled

echo "[INFO] Done. Dataset artifacts are under ${OUTPUT_DIR}"

