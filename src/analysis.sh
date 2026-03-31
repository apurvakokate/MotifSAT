#!/bin/bash
# Post-hoc analysis for Mutagenicity runs under RESULTS_DIR.
#
# Streamlined experiments (run_mutagenicity_gsat_experiment.py) use experiment_* names below.
# Legacy sweeps still live under the old experiment_* names; add them to EXPERIMENTS if needed.
#
# Requirements per GSAT seed_dir (molecular loader with datasets + masked_data):
#   - node_scores.jsonl, Motif_level_node_and_edge_masking_impact.jsonl (last-epoch export in run_gsat.train)
# vanilla_gnn uses train_vanilla_gnn_one_seed: empty node_scores.jsonl → consistency step is skipped.

set -euo pipefail

export RESULTS_DIR="${RESULTS_DIR:-$HOME/hpc-share/ChemIntuit/MotifSAT/tuning_results}"
OUTPUT_BASE="${OUTPUT_BASE:-../motif_consistency_results}"
BEST_RESULTS_DIR="${BEST_RESULTS_DIR:-../best_results}"
DATASET="${DATASET:-Mutagenicity}"

# Must match ALL_EXPERIMENT_NAMES in run_mutagenicity_gsat_experiment.py (streamlined groups).
EXPERIMENTS=(
  vanilla_gnn
  base_gsat_fix_r
  base_gsat_decay_r
  base_gsat_decay_r_minority_global
  base_gsat_decay_r_injection
  base_gsat_motif_loss
  motif_readout_decay_w_message
  motif_readout_decay_injection_ablation
  base_gsat_readout_intra_att
  motif_readout_prior_node_gate   # collect_mutagenicity_tables: one row per motif_prior_shift_scale (0, 0.1, 0.5, 1)
  motif_readout_prior_node_gate_tanh_sched
  motif_readout_weight_diversity
)

MODELS=(GIN PNA GAT SAGE GCN)

echo "RESULTS_DIR=${RESULTS_DIR}"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo "BEST_RESULTS_DIR=${BEST_RESULTS_DIR}"
echo ""

# # ─── 1. Score-vs-Impact + consistency (every fold0_seed0 under each experiment) ───
# # Skipped by default (slow). Uncomment to populate ${OUTPUT_BASE} with plots + consistency.
# for EXP in "${EXPERIMENTS[@]}"; do
#   for MODEL in "${MODELS[@]}"; do
#     EXP_ROOT="${RESULTS_DIR}/${DATASET}/model_${MODEL}/experiment_${EXP}"
#     if [[ ! -d "$EXP_ROOT" ]]; then
#       echo "[SKIP] No directory: ${EXP_ROOT}"
#       continue
#     fi

#     while IFS= read -r SEED_DIR; do
#       [[ -d "$SEED_DIR" ]] || continue

#       # Unique output folder from path tail (tuning_*, method_*, pred*, init*)
#       REL="${SEED_DIR#"${RESULTS_DIR}/${DATASET}/model_${MODEL}/"}"
#       SAFE_REL=$(echo "$REL" | tr '/' '_')

#       OUT="${OUTPUT_BASE}/${EXP}/${MODEL}/${SAFE_REL}"
#       echo "============================================================"
#       echo "  ${EXP}  model=${MODEL}"
#       echo "  seed_dir: ${SEED_DIR}"
#       echo "============================================================"

#       python analyze_motif_consistency.py \
#         --score_vs_impact "${SEED_DIR}" \
#         --dataset "${DATASET}" --model "${MODEL}" --split test \
#         --output_dir "${OUT}" || echo "[ERROR] score_vs_impact failed for ${SEED_DIR}"

#       if [[ -f "${SEED_DIR}/node_scores.jsonl" ]] && [[ -s "${SEED_DIR}/node_scores.jsonl" ]]; then
#         python analyze_motif_consistency.py \
#           --from_jsonl "${SEED_DIR}/node_scores.jsonl" \
#           --dataset "${DATASET}" --model "${MODEL}" --split test --fold 0 \
#           --output_dir "${OUT}" || echo "[ERROR] consistency analysis failed for ${SEED_DIR}"
#       else
#         echo "[SKIP] Missing or empty node_scores.jsonl in ${SEED_DIR}"
#       fi

#       echo ""
#     done < <(find "${EXP_ROOT}" -type d \( -name 'fold0_seed0' -o -name 'fold1_seed0' \) 2>/dev/null | sort)
#   done
# done

# ─── 2. Collect summary tables (one invocation per experiment name) ───
for EXP in "${EXPERIMENTS[@]}"; do
  echo "============================================================"
  echo "  Collecting tables for ${EXP}"
  echo "============================================================"
  python collect_mutagenicity_tables.py \
    --experiment_name "${EXP}" \
    --dataset "${DATASET}" || echo "[ERROR] collect tables failed for ${EXP}"
  echo ""
done

# ─── 3. Best balanced (pred + score–impact) tables and plots ───
# Picks best run per (experiment × model) via composite on valid ROC + valid correlations.
# Tables: ${BEST_RESULTS_DIR}/{train,validation,test}/model_prediction_performance.csv and explainer_score_impact_correlation.csv
# Plots: rendered under ${BEST_RESULTS_DIR}/${DATASET}/<experiment>/model_<ARCH>/{test,valid}_plots/<split>/
# from jsonl in each best seed_dir (training exports only; does not use OUTPUT_BASE).
echo "============================================================"
echo "  collect_best_results → ${BEST_RESULTS_DIR}"
echo "============================================================"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
python collect_best_results.py \
  --dataset "${DATASET}" \
  --results_dir "${RESULTS_DIR}" \
  --best_results_dir "${BEST_RESULTS_DIR}" || echo "[ERROR] collect_best_results failed"

echo "Done. Summary tables (collect_mutagenicity_tables); best tables + score-vs-impact plots under ${BEST_RESULTS_DIR}/"
