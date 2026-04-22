#!/usr/bin/env python
"""
Collect Mutagenicity GSAT results and generate summary tables.

Supports both legacy experiments and newer path layouts like:
  Mutagenicity/model_GCN/experiment_base_gsat_fix_r_node/tuning_fix_r0.5/...
  .../init0.9_final0.5_decay0.1/fold0_seed0/

Tables: columns = architectures, rows = the varying hyperparameter for that experiment.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


EXPERIMENT_ROW_CONFIG = {
    # legacy
    'r_impact_node': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'r',
        'path_extract': 'final_r',
    },
    'r_impact_edge': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'r',
        'path_extract': 'final_r',
    },
    'within_motif_consistency_impact': {
        'summary_path': ('loss_coefficients', 'motif_loss_coef'),
        'row_label_prefix': 'motif_loss_coef',
        'path_extract': 'motif_loss_coef',
    },
    'between_motif_consistency_impact': {
        'summary_path': ('loss_coefficients', 'between_motif_coef'),
        'row_label_prefix': 'between_motif_coef',
        'path_extract': 'between_motif_coef',
    },

    # new experiments
    'base_gsat_fix_r_node': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'base_gsat_fix_r_node_repaired': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'base_gsat_decay_r_node_repaired': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_node': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_fix_r': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'vanilla_gnn_node': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'vanilla',
    },
    'vanilla_gnn_node_repaired': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'vanilla',
    },
    'vanilla_gnn_clean': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'vanilla',
    },
    'motif_readout_fix_r_repaired': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'motif_readout_fix_r_mean': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'motif_readout_fix_r_sum': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'motif_readout_decay_r_mean': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_sum': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_explainer': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_explainer': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_sampling_explainer': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_sampling_info_coef_sweep': {
        'summary_path': None,
        'row_label_prefix': '',
        'path_extract': 'info_coef_final_r',
    },
    'motif_readout_sampling_extractor_sweep': {
        'summary_path': None,
        'row_label_prefix': '',
        'path_extract': 'ext_mult_final_r',
    },
    'motif_readout_sampling_rich_pool': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_explainer_motif_info': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_explainer_motif_info': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_sampling_explainer_motif_info': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_explainer_warmup': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_explainer_warmup': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_r_mean_sampling_explainer_warmup': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_injection_node': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_injection_node_readout': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_injection_readout_only': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_injection_edge_readout': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },

    # Current run_mutagenicity_gsat_experiment.py EXPERIMENT_GROUPS (streamlined)
    'vanilla_gnn': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'vanilla',
    },
    'base_gsat_fix_r': {
        'summary_path': ('weight_distribution_params', 'fix_r'),
        'row_label_prefix': 'fix_r',
        'path_extract': 'fix_r',
    },
    'base_gsat_decay_r': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_minority_global': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'base_gsat_decay_r_injection': {
        'summary_path': None,
        'row_label_prefix': 'inj',
        'path_extract': 'injection_code',
    },
    'base_gsat_motif_loss': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_decay_w_message': {
        'summary_path': None,
        'row_label_prefix': 'sampling',
        'path_extract': 'readout_sampling_mode',
    },
    'no_info_loss': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'no_info_loss_variant',
    },
    'no_info_loss_deterministic_attn': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'no_info_loss_det_variant',
    },
    'motif_readout_decay_injection_ablation': {
        'summary_path': None,
        'row_label_prefix': 'inj',
        'path_extract': 'injection_code',
    },
    'base_gsat_readout_intra_att': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_prior_node_gate': {
        'summary_path': ('motif_incorporation', 'motif_prior_shift_scale'),
        'row_label_prefix': 'shift_scale',
        'path_extract': 'prior_gate_shift',
    },
    'motif_readout_prior_node_gate_tanh_sched': {
        'summary_path': ('motif_incorporation', 'motif_prior_shift_scale'),
        'row_label_prefix': 'shift_scale',
        'path_extract': 'prior_gate_shift',
    },
    'motif_readout_weight_diversity': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_baseline_f07': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e1_logit_standardize': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e2_temperature': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e3_max_pool': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e4_max_mean_pool': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e5_interp_head': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e6_no_gate': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e7_multiplicative_gate': {
        'summary_path': ('weight_distribution_params', 'final_r'),
        'row_label_prefix': 'final_r',
        'path_extract': 'final_r',
    },
    'motif_readout_e8_entropy_sweep': {
        'summary_path': ('loss_coefficients', 'motif_entropy_coef'),
        'row_label_prefix': 'entropy_gamma',
        'path_extract': 'final_r',
    },
    'motif_readout_e9_motif_ib_sweep': {
        'summary_path': None,
        'row_label_prefix': 'motif_ib',
        'path_extract': 'motif_ib_sweep',
    },
    'motif_readout_e10_align_sweep': {
        'summary_path': ('loss_coefficients', 'motif_align_loss_coef'),
        'row_label_prefix': 'align_lambda',
        'path_extract': 'final_r',
    },
    'motif_readout_entropy_pool_sweep': {
        'summary_path': ('motif_incorporation', 'motif_pooling_method'),
        'row_label_prefix': 'pool',
        'path_extract': 'final_r',
    },
    'motif_readout_maxmean_node_vs_edge_att': {
        'summary_path': None,
        'row_label_prefix': 'usage',
        'path_extract': 'maxmean_score_usage',
    },
    'motif_readout_pred_info_only': {
        'summary_path': ('motif_readout_ablation', 'motif_readout_emb_stop'),
        'row_label_prefix': 'emb_stop',
        'path_extract': 'final_r',
    },
    'factored_motif_attention_grid': {
        # Row = M{1-4}_N{1-3}; prefer experiment_summary.json; else parse tuning_factored_M*_N* from path
        'summary_path': None,
        'row_label_prefix': 'cell',
        'path_extract': 'factored_motif_cell',
    },
    'factored_motif_additive': {
        # Row = motif_ib_final_r sweep (0.7, 0.5, 0.3); paths tuning_factored_reg_ibf070 | ibf050 | ibf030
        'summary_path': ('motif_readout_ablation', 'motif_ib_final_r'),
        'row_label_prefix': 'motif_ib_final_r',
        'path_extract': 'factored_additive_ibf',
    },
    'simplified_factored_motif_additive': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'simplified_factored_additive',
    },
    'simplified_motif_readout': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'simplified_motif_readout',
    },
    'simplified_motif_readout_maxmean': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'simplified_motif_readout_maxmean',
    },
    'simplified_motif_readout_maxmean_z1': {
        'summary_path': None,
        'row_label_prefix': 'variant',
        'path_extract': 'simplified_motif_readout_maxmean_z1',
    },
    'simplified_motif_readout_maxmean_injection_ablation': {
        'summary_path': None,
        'row_label_prefix': 'injection',
        'path_extract': 'maxmean_inj_code',
    },
    'simplified_motif_readout_maxmean_info_loss_ablation': {
        'summary_path': ('loss_coefficients', 'info_loss_coef'),
        'row_label_prefix': 'info_loss_coef',
        'path_extract': 'info_loss_coef',
    },
    'test_gradient_factored_between_within_tau2': {
        'summary_path': ('motif_incorporation', 'method'),
        'row_label_prefix': 'variant',
        'path_extract': 'variant',
    },
}


def _coerce_number(x):
    if x is None:
        return None
    try:
        return int(x) if float(x).is_integer() else float(x)
    except Exception:
        return x


def _normalize_motif_emb_stop_row_val(val):
    """
    experiment_summary may store motif_readout_emb_stop as int (layer) or str (encoder/final).
    Mixed types break pandas sort_values('row_val'); canonicalize to str.
    """
    if val is None:
        return 'unknown'
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return str(int(val))
    if isinstance(val, float):
        if not math.isfinite(val):
            return 'unknown'
        return str(int(val)) if val == int(val) else str(val)
    s = str(val).strip().lower()
    if s in ('final', 'none', ''):
        return 'final'
    if s == 'encoder':
        return 'encoder'
    if s.isdigit():
        return str(int(s))
    return s


def _sort_agg_by_row_val(agg: pd.DataFrame) -> pd.DataFrame:
    """Sort by row_val without TypeError when JSON mixes int and str (e.g. emb_stop sweep)."""
    out = agg.copy()
    out['_row_val_sort'] = out['row_val'].map(lambda v: str(v) if v is not None else '')
    out = out.sort_values('_row_val_sort')
    return out.drop(columns=['_row_val_sort'])


def _get_nested(summary: dict, path_tuple):
    if not path_tuple:
        return None
    cur = summary
    for key in path_tuple:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_value_from_parts(parts, mode):
    """
    Recover the row variable directly from path parts.

    Examples:
      tuning_fix_r0.5                       -> fix_r = 0.5
      init0.9_final0.5_decay0.1            -> final_r = 0.5
      pred1_info1_motif2.0_between0        -> motif_loss_coef = 2.0
      pred1_info1_motif1.0_between2.0      -> between_motif_coef = 2.0
    """
    joined = "/".join(parts)

    if mode == 'fix_r':
        for p in parts:
            m = re.match(r'tuning_fix_r([0-9.]+)', p)
            if m:
                return _coerce_number(m.group(1))

    elif mode == 'final_r':
        for p in parts:
            m = re.search(r'init([0-9.]+)_final([0-9.]+)_decay([0-9.]+)', p)
            if m:
                return _coerce_number(m.group(2))
        # fallback for legacy paths
        for p in parts:
            m = re.search(r'(?:node_r|edge_r)([0-9.]+)', p)
            if m:
                return _coerce_number(m.group(1))

    elif mode == 'motif_loss_coef':
        for p in parts:
            m = re.search(r'motif([0-9.]+)_between([0-9.]+)', p)
            if m:
                return _coerce_number(m.group(1))

    elif mode == 'between_motif_coef':
        for p in parts:
            m = re.search(r'motif([0-9.]+)_between([0-9.]+)', p)
            if m:
                return _coerce_number(m.group(2))

    elif mode == 'info_coef_final_r':
        info_val = None
        final_val = None
        for p in parts:
            m = re.search(r'pred[0-9.]+_info([0-9.]+)_motif', p)
            if m:
                info_val = _coerce_number(m.group(1))
            m2 = re.search(r'init[0-9.]+_final([0-9.]+)_decay', p)
            if m2:
                final_val = _coerce_number(m2.group(1))
        if info_val is not None and final_val is not None:
            return f'info={info_val}, r={final_val}'

    elif mode == 'ext_mult_final_r':
        mult_val = None
        final_val = None
        for p in parts:
            m = re.search(r'tuning_ext([0-9]+)_final([0-9.]+)', p)
            if m:
                mult_val = _coerce_number(m.group(1))
                final_val = _coerce_number(m.group(2))
        if mult_val is not None and final_val is not None:
            return f'ext={mult_val}x, r={final_val}'

    elif mode == 'vanilla':
        return 'no_attention'

    elif mode == 'injection_code':
        for p in parts:
            m = re.search(r'inj([0-9]{3})', p)
            if m:
                return m.group(1)
        return None

    elif mode == 'readout_sampling_mode':
        for p in parts:
            if 'node_samp' in p:
                return 'node_samp'
            if 'motif_samp' in p:
                return 'motif_samp'
        return None

    elif mode == 'no_info_loss_variant':
        joined = '/'.join(parts)
        if 'no_info_loss_base' in joined:
            return 'base_gsat'
        if 'no_info_loss_maxmean_node_samp' in joined:
            return 'maxmean_node_samp'
        if 'no_info_loss_maxmean_motif_samp' in joined:
            return 'maxmean_motif_samp'
        return None

    elif mode == 'no_info_loss_det_variant':
        joined = '/'.join(parts)
        if 'no_info_loss_det_base' in joined:
            return 'base_gsat'
        if 'no_info_loss_det_maxmean' in joined:
            return 'maxmean_readout'
        return None

    elif mode == 'prior_gate_shift':
        # tuning_...readout_prior_gate_s{scale} or ...prior_tanhsched_s{scale}
        for p in parts:
            m = re.search(r'readout_prior_gate_s([0-9.]+)', p) or re.search(
                r'prior_tanhsched_s([0-9.]+)', p
            )
            if m:
                return _coerce_number(m.group(1))
        # Pre-sweep runs: tuning_* ended with ..._readout_prior_gate (no _s{scale}); default was 0.1
        for p in parts:
            if re.search(r'readout_prior_gate(?:_|$)', p) and not re.search(
                r'readout_prior_gate_s[0-9.]', p
            ):
                return 0.1
        return None

    elif mode == 'maxmean_score_usage':
        joined = '/'.join(parts)
        if 'maxmean_node_inj' in joined:
            return 'node_inj'
        if 'maxmean_edge_att' in joined:
            return 'edge_att'
        return None

    elif mode == 'factored_motif_variant':
        joined = '/'.join(parts)
        m = re.search(r'factored_([A-Z][0-9])_([A-Z][0-9])', joined)
        if m:
            return f'{m.group(1)}_{m.group(2)}'
        return None

    elif mode == 'factored_additive_ibf':
        # tuning_factored_reg_ibf070 -> 0.7, ibf050 -> 0.5, ibf030 -> 0.3
        joined = '/'.join(parts)
        m = re.search(r'factored_reg_ibf(\d{3})', joined)
        if m:
            return int(m.group(1)) / 100.0
        return None

    elif mode == 'simplified_factored_additive':
        for p in parts:
            if 'simplified_factored_additive' in p:
                return 'simplified_factored_additive'
        return None

    elif mode == 'simplified_motif_readout':
        for p in parts:
            if 'simplified_motif_readout' in p and 'maxmean' not in p:
                return 'simplified_motif_readout'
        return None

    elif mode == 'simplified_motif_readout_maxmean':
        for p in parts:
            if 'simplified_motif_readout_maxmean_z1' in p:
                continue
            if 'simplified_motif_readout_maxmean' in p:
                return 'simplified_motif_readout_maxmean'
        return None

    elif mode == 'simplified_motif_readout_maxmean_z1':
        for p in parts:
            if 'simplified_motif_readout_maxmean_z1' in p:
                return 'simplified_motif_readout_maxmean_z1'
        return None

    elif mode == 'maxmean_inj_code':
        joined = '/'.join(parts)
        m = re.search(r'simplified_maxmean_inj(\d{3})', joined)
        if m:
            return m.group(1)
        return None

    elif mode == 'info_loss_coef':
        for p in parts:
            m = re.search(r'pred[0-9.]+_info([0-9.]+)_motif', p)
            if m:
                return _coerce_number(m.group(1))
        return None

    return None


def _get_row_value(summary: dict, experiment_name: str, parts=None):
    cfg = EXPERIMENT_ROW_CONFIG.get(experiment_name)
    if cfg is None:
        return 'unknown'

    extract_mode = cfg.get('path_extract')

    # Composite modes: build from multiple summary fields
    if extract_mode == 'motif_ib_sweep':
        bm = _get_nested(summary, ('loss_coefficients', 'motif_level_ib_coef'))
        fr = _get_nested(summary, ('motif_readout_ablation', 'motif_ib_final_r'))
        if bm is not None and fr is not None:
            return f'bm={_coerce_number(bm)},rfr={_coerce_number(fr)}'
        if parts is not None:
            joined = '/'.join(parts)
            m = re.search(r'e9_motifib_bm([0-9.]+)_rfr([0-9.]+)', joined)
            if m:
                return f'bm={_coerce_number(m.group(1))},rfr={_coerce_number(m.group(2))}'
        return 'unknown'

    # Composite modes: build from multiple summary fields
    if extract_mode == 'info_coef_final_r':
        info_val = _get_nested(summary, ('loss_coefficients', 'info_loss_coef'))
        final_val = _get_nested(summary, ('weight_distribution_params', 'final_r'))
        if info_val is not None and final_val is not None:
            return f'info={_coerce_number(info_val)}, r={_coerce_number(final_val)}'
        if parts is not None:
            val = _extract_value_from_parts(parts, extract_mode)
            if val is not None:
                return val
        return 'unknown'

    if extract_mode == 'ext_mult_final_r':
        final_val = _get_nested(summary, ('weight_distribution_params', 'final_r'))
        if parts is not None:
            val = _extract_value_from_parts(parts, extract_mode)
            if val is not None:
                return val
        if final_val is not None:
            return f'ext=?x, r={_coerce_number(final_val)}'
        return 'unknown'

    if extract_mode == 'factored_motif_cell':
        zk = _get_nested(summary, ('motif_readout_ablation', 'factored_motif_zk_axis'))
        nd = _get_nested(summary, ('motif_readout_ablation', 'factored_node_logit_axis'))
        if zk is not None and nd is not None:
            return f'{zk}_{nd}'
        if parts is not None:
            val_fb = _extract_value_from_parts(parts, 'factored_motif_variant')
            if val_fb is not None:
                return val_fb
        return 'unknown'

    val = None
    if cfg.get('summary_path') is not None:
        val = _get_nested(summary, cfg['summary_path'])

    if val is None and parts is not None:
        val = _extract_value_from_parts(parts, extract_mode)

    if val is None:
        return 'unknown'
    if cfg.get('summary_path') == ('motif_readout_ablation', 'motif_readout_emb_stop'):
        val = _normalize_motif_emb_stop_row_val(val)
    return val


def _make_row_label(val, experiment_name: str):
    cfg = EXPERIMENT_ROW_CONFIG.get(experiment_name)
    prefix = cfg['row_label_prefix'] if cfg else 'param'
    return f'{prefix}={val}'


def _recover_truncated_json(raw: str):
    raw = raw.strip()
    if not raw.startswith('{'):
        return None
    for i in range(len(raw) - 1, 0, -1):
        candidate = raw[:i]
        last_comma = candidate.rfind(',')
        if last_comma > 0:
            candidate = candidate[:last_comma]
        candidate = candidate.rstrip() + '\n}'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _read_json(path: Path):
    with open(path) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _recover_truncated_json(raw)


def _read_jsonl(path: Path):
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _infer_model_from_parts(parts):
    for p in parts:
        if p.startswith('model_'):
            return p.replace('model_', '')
    return None


def _infer_tuning_from_parts(parts):
    for p in parts:
        if p.startswith('tuning_'):
            return p.replace('tuning_', '')
    return None


def _infer_fold_seed_from_parts(parts):
    for p in parts:
        m = re.match(r'fold(\d+)_seed(\d+)', p)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _compute_attention_min_max_from_jsonl(seed_dir: Path):
    """
    Fallback for motif_edge_att/max_mean and motif_edge_att/min_mean if not present
    in final_metrics.json.

    Assumes attention_distributions.jsonl has one attention value per record and
    some motif identifier. You may need to tweak the key names if your JSONL differs.
    """
    path = seed_dir / 'attention_distributions.jsonl'
    if not path.exists():
        return np.nan, np.nan

    groups = defaultdict(list)

    for rec in _read_jsonl(path):
        motif_idx = rec.get('motif_idx', rec.get('motif_index'))
        if motif_idx is None:
            continue

        att = None
        for k in ['attention', 'att', 'edge_attention', 'score', 'value']:
            if k in rec:
                att = rec[k]
                break

        if att is None:
            continue

        try:
            att = float(att)
        except Exception:
            continue

        groups[motif_idx].append(att)

    if not groups:
        return np.nan, np.nan

    per_motif_max = [max(v) for v in groups.values() if len(v) > 0]
    per_motif_min = [min(v) for v in groups.values() if len(v) > 0]

    if not per_motif_max or not per_motif_min:
        return np.nan, np.nan

    return float(np.mean(per_motif_max)), float(np.mean(per_motif_min))


def find_results(results_dir: Path, experiment_name: str, verbose: bool = False, dataset: str = 'Mutagenicity'):
    """
    Find run directories for the given experiment_name.
    Prefers final_metrics.json, but can still register a run if only attention_distributions.jsonl exists.
    """
    base = results_dir / dataset
    if not base.exists():
        print(f'[WARN] Directory does not exist: {base}')
        return []

    experiment_dirs = {f'experiment_{experiment_name}'}
    # Some groups use a custom on-disk experiment_name in run_mutagenicity_gsat_experiment.py.
    # Keep analysis calls stable by accepting either group key or configured experiment_name.
    try:
        from run_mutagenicity_gsat_experiment import EXPERIMENT_GROUPS  # type: ignore

        cfg = EXPERIMENT_GROUPS.get(experiment_name)
        if isinstance(cfg, dict):
            cfg_exp_name = cfg.get('experiment_name')
            if isinstance(cfg_exp_name, str) and cfg_exp_name:
                experiment_dirs.add(f'experiment_{cfg_exp_name}')
    except Exception:
        pass
    records = []

    candidate_seed_dirs = set()

    for p in base.rglob('final_metrics.json'):
        if any(ed in p.parts for ed in experiment_dirs):
            candidate_seed_dirs.add(p.parent)

    for p in base.rglob('attention_distributions.jsonl'):
        if any(ed in p.parts for ed in experiment_dirs):
            candidate_seed_dirs.add(p.parent)

    if verbose:
        print(f'[DEBUG] Found {len(candidate_seed_dirs)} candidate seed dirs for {experiment_name}')

    skipped_parse = 0
    skipped_error = 0

    for seed_dir in sorted(candidate_seed_dirs):
        try:
            parts = seed_dir.relative_to(base).parts

            model_name = _infer_model_from_parts(parts)
            tuning_id = _infer_tuning_from_parts(parts)
            fold, seed = _infer_fold_seed_from_parts(parts)

            if model_name is None or fold is None or seed is None:
                skipped_parse += 1
                if verbose:
                    print(f'  [SKIP parse] path={seed_dir}')
                continue

            fm_path = seed_dir / 'final_metrics.json'
            metrics = {}
            if fm_path.exists():
                metrics = _read_json(fm_path) or {}

            # fallback-compute motif edge stats if absent
            if 'motif_edge_att/max_mean' not in metrics or 'motif_edge_att/min_mean' not in metrics:
                max_mean, min_mean = _compute_attention_min_max_from_jsonl(seed_dir)
                if not np.isnan(max_mean):
                    metrics.setdefault('motif_edge_att/max_mean', max_mean)
                if not np.isnan(min_mean):
                    metrics.setdefault('motif_edge_att/min_mean', min_mean)

            summary_path = seed_dir / 'experiment_summary.json'
            summary = {}
            if summary_path.exists():
                summary = _read_json(summary_path) or {}

            row_val = _get_row_value(summary, experiment_name, parts=parts)
            row_label = _make_row_label(row_val, experiment_name)

            records.append({
                'model': model_name,
                'variant': tuning_id,
                'row_val': row_val,
                'row': row_label,
                'fold': fold,
                'seed': seed,
                'metrics': metrics,
                'seed_dir': seed_dir,
            })
        except Exception as e:
            skipped_error += 1
            if verbose:
                print(f'  [ERROR] {seed_dir}: {e}')

    if verbose:
        print(f'[DEBUG] skipped_parse={skipped_parse}, skipped_error={skipped_error}, collected={len(records)}')

    return records


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_posthoc_correlation(seed_dir: Path, split: str = 'test'):
    node_scores_path = seed_dir / 'node_scores.jsonl'
    impact_path = seed_dir / 'Motif_level_node_and_edge_masking_impact.jsonl'
    if not impact_path.exists():
        impact_path = seed_dir / 'masked-edge-impact.jsonl'  # legacy fallback

    if not node_scores_path.exists() or not impact_path.exists():
        return np.nan, np.nan, 0

    scores = defaultdict(list)
    for rec in _read_jsonl(node_scores_path):
        if rec.get('split') != split:
            continue
        motif_idx = rec.get('motif_index', rec.get('motif_idx'))
        if motif_idx is None or motif_idx < 0:
            continue
        scores[motif_idx].append(rec['score'])

    impacts = defaultdict(list)
    for rec in _read_jsonl(impact_path):
        if rec.get('split') != split:
            continue
        motif_idx = rec.get('motif_idx', rec.get('motif_index'))
        if motif_idx is None or motif_idx < 0:
            continue
        imp = abs(_sigmoid(rec['new_prediction']) - _sigmoid(rec['old_prediction']))
        impacts[motif_idx].append(imp)

    common = set(scores.keys()) & set(impacts.keys())
    if len(common) < 3:
        return np.nan, np.nan, len(common)

    avg_scores = np.array([np.mean(scores[m]) for m in sorted(common)])
    avg_impacts = np.array([np.mean(impacts[m]) for m in sorted(common)])

    if np.allclose(avg_scores, avg_scores[0]) or np.allclose(avg_impacts, avg_impacts[0]):
        return np.nan, np.nan, len(common)

    r, p = pearsonr(avg_scores, avg_impacts)
    return float(r), float(p), len(common)


def compute_node_score_impact_correlation(seed_dir: Path, split: str = 'test'):
    """
    Pearson r between per-node attention score and |Δ sigmoid(pred)| from individual-node masking.
    Matches analyze_motif_consistency.plot_score_vs_impact individual-node branch.
    """
    path = seed_dir / 'Individual_node_node_masking_impact.jsonl'
    if not path.exists():
        path = seed_dir / 'masked-node-impact.jsonl'
    if not path.exists():
        return np.nan, np.nan, 0
    xs, ys = [], []
    for rec in _read_jsonl(path):
        if rec.get('split') != split:
            continue
        xs.append(float(rec['score']))
        ys.append(abs(_sigmoid(rec['new_prediction']) - _sigmoid(rec['old_prediction'])))
    if len(xs) < 3:
        return np.nan, np.nan, len(xs)
    r, p = pearsonr(xs, ys)
    return float(r), float(p), len(xs)


def build_posthoc_table(records, split='test'):
    rows = []
    for rec in records:
        r, p, n = compute_posthoc_correlation(rec['seed_dir'], split=split)
        rows.append({
            'model': rec['model'],
            'row': rec['row'],
            'row_val': rec['row_val'],
            'pearson_r': r,
            'p_value': p,
            'n_motifs': n,
        })

    if not rows:
        return None, None, None

    df = pd.DataFrame(rows)
    agg = df.groupby(['row', 'row_val', 'model'])['pearson_r'].agg(['mean', 'std', 'count']).reset_index()
    agg = _sort_agg_by_row_val(agg)

    pivot_mean = agg.pivot(index='row', columns='model', values='mean')
    pivot_std = agg.pivot(index='row', columns='model', values='std')
    pivot_count = agg.pivot(index='row', columns='model', values='count')

    row_order = agg.drop_duplicates('row')['row'].tolist()
    pivot_mean = pivot_mean.reindex(row_order).dropna(how='all')
    pivot_std = pivot_std.reindex(row_order).dropna(how='all')
    pivot_count = pivot_count.reindex(row_order).dropna(how='all')
    return pivot_mean, pivot_std, pivot_count


def build_node_posthoc_table(records, split='test'):
    rows = []
    for rec in records:
        r, p, n = compute_node_score_impact_correlation(rec['seed_dir'], split=split)
        rows.append({
            'model': rec['model'],
            'row': rec['row'],
            'row_val': rec['row_val'],
            'pearson_r': r,
            'p_value': p,
            'n_nodes': n,
        })

    if not rows:
        return None, None, None

    df = pd.DataFrame(rows)
    agg = df.groupby(['row', 'row_val', 'model'])['pearson_r'].agg(['mean', 'std', 'count']).reset_index()
    agg = _sort_agg_by_row_val(agg)

    pivot_mean = agg.pivot(index='row', columns='model', values='mean')
    pivot_std = agg.pivot(index='row', columns='model', values='std')
    pivot_count = agg.pivot(index='row', columns='model', values='count')

    row_order = agg.drop_duplicates('row')['row'].tolist()
    pivot_mean = pivot_mean.reindex(row_order).dropna(how='all')
    pivot_std = pivot_std.reindex(row_order).dropna(how='all')
    pivot_count = pivot_count.reindex(row_order).dropna(how='all')
    return pivot_mean, pivot_std, pivot_count


def build_table(records, metric_key, verbose=False):
    if not records:
        return None, None, None

    df = pd.DataFrame(records)
    df['has_key'] = df['metrics'].apply(lambda m: metric_key in m)
    df['value'] = df['metrics'].apply(lambda m: m.get(metric_key, np.nan))

    if verbose:
        total = df.groupby(['row', 'model']).size().rename('total_runs')
        present = df.groupby(['row', 'model'])['has_key'].sum().rename('has_metric')
        diag = pd.concat([total, present], axis=1)
        missing = diag[diag['has_metric'] < diag['total_runs']]
        if not missing.empty:
            print(f'  [DIAG] Metric "{metric_key}" missing in some runs:')
            for (row, model), r in missing.iterrows():
                print(f'    {row} / {model}: {int(r["has_metric"])}/{int(r["total_runs"])} runs have the key')

    agg = df.groupby(['row', 'row_val', 'model'])['value'].agg(['mean', 'std', 'count']).reset_index()
    agg = _sort_agg_by_row_val(agg)

    pivot_mean = agg.pivot(index='row', columns='model', values='mean')
    pivot_std = agg.pivot(index='row', columns='model', values='std')
    pivot_count = agg.pivot(index='row', columns='model', values='count')

    row_order = agg.drop_duplicates('row')['row'].tolist()
    pivot_mean = pivot_mean.reindex(row_order).dropna(how='all')
    pivot_std = pivot_std.reindex(row_order).dropna(how='all')
    pivot_count = pivot_count.reindex(row_order).dropna(how='all')
    return pivot_mean, pivot_std, pivot_count


def format_mean_std_count(mean_df, std_df, count_df):
    """Combine mean, std, and count into 'mean +/- std' formatted DataFrame.

    Blank / missing-std cells get a diagnostic suffix explaining WHY:
      [no_runs]           – no seed directories matched this (row, model) combo
      [metric_NaN,n=K]    – K runs found but the metric was NaN in all of them
      [n=1,no_std]        – only 1 valid value so std is undefined (ddof=1)
      [std=NaN,n=K]       – K>1 values but std still NaN (all identical?)
    """
    if mean_df is None:
        return None
    combined = mean_df.copy().astype(object)
    for col in mean_df.columns:
        for idx in mean_df.index:
            m = mean_df.at[idx, col]
            s = std_df.at[idx, col] if idx in std_df.index and col in std_df.columns else np.nan
            n = count_df.at[idx, col] if idx in count_df.index and col in count_df.columns else np.nan
            if pd.isna(m):
                if pd.isna(n):
                    combined.at[idx, col] = '[no_runs]'
                elif int(n) == 0:
                    combined.at[idx, col] = '[metric_NaN,n=0]'
                else:
                    combined.at[idx, col] = f'[metric_NaN,n={int(n)}]'
            elif pd.isna(s):
                if pd.isna(n) or int(n) == 1:
                    combined.at[idx, col] = f'{m:.4f} [n=1,no_std]'
                else:
                    combined.at[idx, col] = f'{m:.4f} [std=NaN,n={int(n)}]'
            else:
                combined.at[idx, col] = f'{m:.4f} +/- {s:.4f}'
    return combined


def _print_and_save_table(label, mean_df, std_df, count_df, prefix, suffix, output_dir):
    """Helper to print formatted table and save CSVs."""
    if mean_df is None or mean_df.isna().all().all():
        print(f'\nNo data for: {label}')
        return
    formatted = format_mean_std_count(mean_df, std_df, count_df)
    print(f'\n--- {label} ---')
    print(formatted.to_string())
    path = output_dir / f'{prefix}_{suffix}.csv'
    mean_df.to_csv(path)
    std_df.to_csv(output_dir / f'{prefix}_{suffix}_std.csv')
    count_df.to_csv(output_dir / f'{prefix}_{suffix}_count.csv')
    print(f'Saved: {path}')


MOTIF_READOUT_EXPERIMENTS = {
    ('fix', 'mean'): 'motif_readout_fix_r_mean',
    ('fix', 'sum'):  'motif_readout_fix_r_sum',
    ('decay', 'mean'): 'motif_readout_decay_r_mean',
    ('decay', 'sum'):  'motif_readout_decay_r_sum',
}


def build_combined_motif_readout_table(results_dir, metric_key, dataset='Mutagenicity', verbose=False):
    """
    Build a combined table across all 4 motif readout experiments.

    Returns a DataFrame with:
      Row MultiIndex: (r_value, pooling)  e.g. (0.9, 'mean'), (0.9, 'sum')
      Column MultiIndex: (model, schedule) e.g. ('GCN', 'fix'), ('GCN', 'decay')
    """
    all_rows = []

    for (schedule, pooling), exp_name in MOTIF_READOUT_EXPERIMENTS.items():
        records = find_results(results_dir, exp_name, verbose=verbose, dataset=dataset)
        if not records:
            continue

        for rec in records:
            val = rec['metrics'].get(metric_key, np.nan)
            r_value = rec['row_val']
            all_rows.append({
                'r': r_value,
                'pooling': pooling,
                'schedule': schedule,
                'model': rec['model'],
                'fold': rec['fold'],
                'seed': rec['seed'],
                'value': val,
            })

    if not all_rows:
        return None, None, None

    df = pd.DataFrame(all_rows)

    agg = df.groupby(['r', 'pooling', 'model', 'schedule'])['value'].agg(
        ['mean', 'std', 'count']).reset_index()

    mean_records, std_records, count_records = [], [], []
    for _, row in agg.iterrows():
        key = {'r': row['r'], 'pooling': row['pooling']}
        col = (row['model'], row['schedule'])
        mean_records.append({**key, 'col': col, 'val': row['mean']})
        std_records.append({**key, 'col': col, 'val': row['std']})
        count_records.append({**key, 'col': col, 'val': row['count']})

    def _pivot(records_list):
        if not records_list:
            return None
        rows_idx = sorted({(r['r'], r['pooling']) for r in records_list},
                          key=lambda x: (-x[0], x[1]))
        cols = sorted({r['col'] for r in records_list}, key=lambda x: (x[0], x[1]))
        multi_idx = pd.MultiIndex.from_tuples(rows_idx, names=['r', 'pooling'])
        multi_col = pd.MultiIndex.from_tuples(cols, names=['model', 'schedule'])
        result = pd.DataFrame(np.nan, index=multi_idx, columns=multi_col)
        for r in records_list:
            result.at[(r['r'], r['pooling']), r['col']] = r['val']
        return result

    return _pivot(mean_records), _pivot(std_records), _pivot(count_records)


def format_combined_mean_std(mean_df, std_df, count_df):
    """Format combined table as 'mean +/- std' with diagnostic reasons for gaps."""
    if mean_df is None:
        return None
    combined = mean_df.copy().astype(object)
    for col in mean_df.columns:
        for idx in mean_df.index:
            m = mean_df.at[idx, col]
            s = std_df.at[idx, col] if std_df is not None else np.nan
            n = count_df.at[idx, col] if count_df is not None else np.nan
            if pd.isna(m):
                if pd.isna(n):
                    combined.at[idx, col] = '[no_runs]'
                elif int(n) == 0:
                    combined.at[idx, col] = '[metric_NaN,n=0]'
                else:
                    combined.at[idx, col] = f'[metric_NaN,n={int(n)}]'
            elif pd.isna(s):
                if pd.isna(n) or int(n) == 1:
                    combined.at[idx, col] = f'{m:.4f} [n=1,no_std]'
                else:
                    combined.at[idx, col] = f'{m:.4f} [std=NaN,n={int(n)}]'
            else:
                combined.at[idx, col] = f'{m:.4f} +/- {s:.4f}'
    return combined


def run_combined_motif_readout(results_dir, output_dir, dataset='Mutagenicity', verbose=False):
    """Generate combined motif readout tables for all key metrics."""
    metrics = [
        ('metric/best_clf_roc_valid', 'Valid ROC', 'valid_roc'),
        ('metric/best_clf_roc_test', 'Test ROC', 'test_roc'),
        ('motif/att_impact_correlation', 'Explainer Correlation', 'explainer_corr'),
    ]

    for metric_key, label, suffix in metrics:
        mean_df, std_df, count_df = build_combined_motif_readout_table(
            results_dir, metric_key, dataset=dataset, verbose=verbose)

        if mean_df is None:
            print(f'\nNo data for combined motif readout: {label}')
            continue

        formatted = format_combined_mean_std(mean_df, std_df, count_df)
        print(f'\n=== Combined Motif Readout: {label} ({dataset}) ===')
        print(formatted.to_string())

        path = output_dir / f'combined_motif_readout_{dataset}_{suffix}.csv'
        mean_df.to_csv(path)
        std_df.to_csv(output_dir / f'combined_motif_readout_{dataset}_{suffix}_std.csv')
        count_df.to_csv(output_dir / f'combined_motif_readout_{dataset}_{suffix}_count.csv')
        print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Collect GSAT experiment results into tables')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Single experiment name to collect')
    parser.add_argument('--combine_motif_readout', action='store_true',
                        help='Build combined motif readout table across fix/decay x mean/sum')
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        help='Dataset name (default: Mutagenicity)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results dir (default: RESULTS_DIR env or ../tuning_results)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir or os.environ.get('RESULTS_DIR', '../tuning_results'))
    output_dir = Path(args.output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine_motif_readout:
        run_combined_motif_readout(results_dir, output_dir, dataset=args.dataset, verbose=args.verbose)
        return

    if args.experiment_name is None:
        parser.error('--experiment_name is required unless --combine_motif_readout is used')

    records = find_results(results_dir, args.experiment_name, verbose=args.verbose, dataset=args.dataset)
    if not records:
        print(f'No results found for experiment "{args.experiment_name}" under {results_dir / args.dataset}.')
        return

    print(f'Experiment: {args.experiment_name} (dataset: {args.dataset})')
    print(f'Found {len(records)} runs.')

    prefix = args.experiment_name

    verbose = args.verbose

    pred_mean, pred_std, pred_count = build_table(records, metric_key='metric/best_clf_roc_valid', verbose=verbose)
    _print_and_save_table('Prediction performance (valid ROC, mean +/- std)',
                          pred_mean, pred_std, pred_count, prefix, 'prediction_valid_roc', output_dir)

    test_mean, test_std, test_count = build_table(records, metric_key='metric/best_clf_roc_test', verbose=verbose)
    _print_and_save_table('Prediction performance (test ROC, mean +/- std)',
                          test_mean, test_std, test_count, prefix, 'prediction_test_roc', output_dir)

    exp_mean, exp_std, exp_count = build_table(records, metric_key='motif/att_impact_correlation', verbose=verbose)
    _print_and_save_table('Explainer (motif att-impact correlation, mean +/- std)',
                          exp_mean, exp_std, exp_count, prefix, 'explainer_correlation', output_dir)

    node_mean, node_std, node_count = build_node_posthoc_table(records, split='test')
    _print_and_save_table('Explainer (node score-impact correlation, test, mean +/- std)',
                          node_mean, node_std, node_count, prefix, 'node_score_impact_correlation', output_dir)

    range_mean, range_std, range_count = build_table(records, metric_key='motif_edge_att/max_mean', verbose=verbose)
    _print_and_save_table('Motif edge att max (mean +/- std)',
                          range_mean, range_std, range_count, prefix, 'motif_edge_att_max', output_dir)

    min_mean, min_std, min_count = build_table(records, metric_key='motif_edge_att/min_mean', verbose=verbose)
    _print_and_save_table('Motif edge att min (mean +/- std)',
                          min_mean, min_std, min_count, prefix, 'motif_edge_att_min', output_dir)

    for split in ['train', 'valid', 'test']:
        ph_mean, ph_std, ph_count = build_posthoc_table(records, split=split)
        _print_and_save_table(f'Post-hoc score-impact correlation ({split}, mean +/- std)',
                              ph_mean, ph_std, ph_count, prefix, f'posthoc_correlation_{split}', output_dir)


if __name__ == '__main__':
    main()