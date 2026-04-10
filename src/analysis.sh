#!/bin/bash
# Post-hoc analysis for Mutagenicity runs under RESULTS_DIR.
#
# Streamlined experiments (run_mutagenicity_gsat_experiment.py) use experiment_* names below.
# Legacy sweeps still live under the old experiment_* names; add them to EXPERIMENTS if needed.
#
# Requirements per GSAT seed_dir (molecular loader with datasets + masked_data):
#   - node_scores.jsonl, Motif_level_node_and_edge_masking_impact.jsonl (last-epoch export in run_gsat.train)
# vanilla_gnn uses train_vanilla_gnn_one_seed: empty node_scores.jsonl → consistency step is skipped.
#
# Motif–class Fisher table: compute_motif_class_association.py caches under ${DATA_DIR}/motif_association/
# (training split by default). If present, get_data_loaders attaches node_motif_assoc_p to each graph.
# motif_stat_vs_importance_roc.py writes ${OUTPUT_BASE}/motif_importance_vs_fisher_roc_${DATASET}.csv

set -euo pipefail

export RESULTS_DIR="${RESULTS_DIR:-$HOME/hpc-share/ChemIntuit/MotifSAT/tuning_results}"
OUTPUT_BASE="${OUTPUT_BASE:-../motif_consistency_results}"
BEST_RESULTS_DIR="${BEST_RESULTS_DIR:-../best_results}"
DATASET="${DATASET:-Mutagenicity}"
# Optional: SET_R=0.8 → keep only runs matching that fix_r/final_r (per experiment).
SET_R="${SET_R:-}"
# Best row + per-fold seeds: motif_corr_valid = max motif-level explainer r on valid (tie-break ROC).
# Override: SELECTION_BY=composite for the old ROC+motif+node weighted score.
export SELECTION_BY="${SELECTION_BY:-motif_corr_valid}"

# Must match ALL_EXPERIMENT_NAMES in run_mutagenicity_gsat_experiment.py (streamlined groups).
EXPERIMENTS=(
  vanilla_gnn
  base_gsat_fix_r
  base_gsat_decay_r
  base_gsat_decay_r_minority_global
  base_gsat_decay_r_injection
  base_gsat_motif_loss
  motif_readout_decay_w_message
  no_info_loss
  no_info_loss_deterministic_attn
  motif_readout_decay_injection_ablation
  base_gsat_readout_intra_att
  motif_readout_prior_node_gate   # collect_mutagenicity_tables: one row per motif_prior_shift_scale (0, 0.1, 0.5, 1)
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
  # factored_motif_additive: one table per experiment name; rows split by motif_ib_final_r
  # (variants tuning_factored_reg_ibf070|050|030 → 0.7, 0.5, 0.3). See EXPERIMENT_ROW_CONFIG in collect_mutagenicity_tables.py
  factored_motif_additive
  simplified_factored_motif_additive
  simplified_motif_readout
  simplified_motif_readout_maxmean
  simplified_motif_readout_maxmean_z1
  simplified_motif_readout_maxmean_injection_ablation
  simplified_motif_readout_maxmean_info_loss_ablation
)

MODELS=(GIN PNA GAT SAGE GCN)

echo "RESULTS_DIR=${RESULTS_DIR}"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo "BEST_RESULTS_DIR=${BEST_RESULTS_DIR}"
echo "SET_R=${SET_R:-<unset>}"
echo "SELECTION_BY=${SELECTION_BY}"
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
# SELECTION_BY picks which hyperparam row (per model) to use for plots; CSV cells match collect_mutagenicity_tables
# pivots for that row (same mean±std as the big tables). SET_R filters runs before pivots if set.
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
  --best_results_dir "${BEST_RESULTS_DIR}" \
  ${SET_R:+--set_r "${SET_R}"} || echo "[ERROR] collect_best_results failed"

# ─── 4. Motif–class Fisher association (cached) + ROC: model importance vs statistical motifs ───
FOLD="${FOLD:-0}"
DATA_DIR="${DATA_DIR:-../data}"
ASSOC_DIR="${DATA_DIR}/motif_association"
STEM="${DATASET}_fold${FOLD}_training"
ASSOC_CSV="${ASSOC_DIR}/${STEM}_motif_class_association.csv"
mkdir -p "${ASSOC_DIR}" "${OUTPUT_BASE}"
if [[ ! -f "${ASSOC_CSV}" ]]; then
  echo "============================================================"
  echo "  Computing motif–class association (Fisher / chi-sq) → ${ASSOC_CSV}"
  echo "============================================================"
  python compute_motif_class_association.py \
    --dataset "${DATASET}" \
    --fold "${FOLD}" \
    --data_dir "${DATA_DIR}" || echo "[WARN] compute_motif_class_association failed (set --csv_file to your FOLDS CSV if needed)"
else
  echo "[INFO] Using cached motif association table: ${ASSOC_CSV}"
fi
ROC_OUT="${OUTPUT_BASE}/motif_importance_vs_fisher_roc_${DATASET}.csv"
if [[ -f "${ASSOC_CSV}" ]]; then
  echo "============================================================"
  echo "  Motif importance vs Fisher significance ROC → ${ROC_OUT}"
  echo "============================================================"
  python motif_stat_vs_importance_roc.py \
    --association_csv "${ASSOC_CSV}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --out_csv "${ROC_OUT}" || echo "[WARN] motif_stat_vs_importance_roc failed"
  echo "[INFO] ROC table: ${ROC_OUT}"
fi

echo "Done. Summary tables (collect_mutagenicity_tables); best tables + score-vs-impact plots under ${BEST_RESULTS_DIR}/"
