#!/bin/bash
set -euo pipefail

export RESULTS_DIR=${RESULTS_DIR:-~/hpc-share/ChemIntuit/MotifSAT/tuning_results}
OUTPUT_BASE=../motif_consistency_results

EXPERIMENTS=(
  base_gsat_decay_r_explainer
  motif_readout_decay_r_mean_explainer
  motif_readout_decay_r_mean_sampling_explainer
)
MODELS=(GCN SAGE)
FINAL_RS=(0.8 0.7)

# ─── 1. Score-vs-Impact plots + Full consistency analysis ───
for EXP in "${EXPERIMENTS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for R in "${FINAL_RS[@]}"; do
      # Find the seed_dir using glob (intermediate dirs vary by experiment)
      PATTERN="${RESULTS_DIR}/Mutagenicity/model_${MODEL}/experiment_${EXP}/*/method_*/pred*/init0.9_final${R}_decay0.1/fold0_seed0"
      SEED_DIR=$(ls -d $PATTERN 2>/dev/null | head -1)

      if [ -z "$SEED_DIR" ]; then
        echo "[SKIP] No seed_dir found for ${EXP} ${MODEL} r=${R}"
        continue
      fi

      OUT="${OUTPUT_BASE}/${EXP}/${MODEL}_r${R}"
      echo "============================================================"
      echo "  ${EXP}  model=${MODEL}  final_r=${R}"
      echo "  seed_dir: ${SEED_DIR}"
      echo "============================================================"

      # Score-vs-Impact plot (standalone, reads impact JSONLs)
      python analyze_motif_consistency.py \
        --score_vs_impact "${SEED_DIR}" \
        --dataset Mutagenicity --model "${MODEL}" --split test \
        --output_dir "${OUT}"

      # Full consistency analysis (reads node_scores.jsonl)
      if [ -f "${SEED_DIR}/node_scores.jsonl" ]; then
        python analyze_motif_consistency.py \
          --from_jsonl "${SEED_DIR}/node_scores.jsonl" \
          --dataset Mutagenicity --model "${MODEL}" --split test --fold 0 \
          --output_dir "${OUT}"
      else
        echo "[SKIP] No node_scores.jsonl in ${SEED_DIR}"
      fi
    done
  done
done

# ─── 2. Collect summary tables (one per experiment) ───
for EXP in "${EXPERIMENTS[@]}"; do
  echo "============================================================"
  echo "  Collecting tables for ${EXP}"
  echo "============================================================"
  python collect_mutagenicity_tables.py \
    --experiment_name "${EXP}" \
    --dataset Mutagenicity
done

echo ""
echo "Done. Results in ${OUTPUT_BASE}/"
