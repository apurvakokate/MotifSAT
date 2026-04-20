#!/usr/bin/env python
"""
GSAT experiment driver for paper-style and molecular (motif) datasets.

Active groups (EXPERIMENT_GROUPS):
  vanilla_gnn — vanilla GNN only
  base_gsat_fix_r — fix_r in {1.0..0.5}, w_message only (010: w_feat=F, w_message=T, w_readout=F)
  base_gsat_decay_r — decay schedule, final_r in {1.0..0.5}, w_message only
  base_gsat_decay_r_injection — decay, final_r=0.8, injection ablation (100,010,001,101,011)
  base_gsat_motif_loss — decay, final_r=0.8, w_message only, motif_method=loss
  motif_readout_decay_w_message — decay, final_r=0.8, w_message only; node- vs motif-level sampling
  motif_readout_decay_injection_ablation — decay, final_r=0.8, node-level sampling; injection ablation (100..011)
  base_gsat_readout_intra_att — decay, final_r=0.8, w_message only; readout with intra-motif attention pooling only
  motif_readout_prior_node_gate — readout prior-gate; shift_scale sweep {0, 0.1, 0.5, 1.0} (see variant_id *_s*)
  motif_readout_prior_node_gate_tanh_sched — tanh-bounded gate shift + warmup_linear s schedule; same s sweep
  motif_readout_weight_diversity — readout + motif_weight_diversity_coef (penalize identical motif scores within a graph)
  motif_readout_baseline_f07 — decay final_r=0.7, motif-level sampling, prior node gate (fixed baseline for E1–E10)
  motif_readout_e1_logit_standardize … motif_readout_e10_align_sweep — single-factor ablations (see EXPERIMENT_GROUPS)
  motif_readout_entropy_pool_sweep — no node gate; entropy bonus; pooling sweep mean | max | max_mean | intra_att
  motif_readout_maxmean_node_vs_edge_att — max_mean + entropy; node-injection vs edge_atten downstream usage
  motif_readout_pred_info_only — L_pred + L_info only; max_mean + motif-level sampling; sweep motif_readout_emb_stop
    (encoder | layer 0..2 | final) to find discriminative motif α without extra losses
  test_gradients — focused readout run with motif grad probe enabled (pred vs KL/info per motif)
  test_gradient_info_coef0.2_tau2 — test_gradients variant with info_loss_coef=0.2 and sampling temperature τ=2.0
  base_gsat_decay_r_minority_global — same as base_gsat_decay_r but motif pickles from FOLDS/minority_global/...
  factored_motif_attention_grid — 12 variants (M1–M4 × N1–N3): multi-granularity z_k, factored node logits, motif IB on mean node α (see experiment_factored_motif.py)
  factored_motif_additive — LN(z^(1)||z^att), MLP motif ℓ_k, node ℓ=ℓ_k+δ(intra), IB on σ(ℓ_k); sweep motif_ib_final_r ∈ {0.7,0.5,0.3}
  simplified_factored_motif_additive — MLP(LN(z^att)) only; 010; L_pred + motif-level L_info on σ(ℓ_k) (use_raw_score_loss); info_loss_coef≈motif_ib scale; info_warmup 20; final_r=0.8
  simplified_motif_readout — same as simplified_factored_motif_additive but node logit = ℓ_k only (no intra-motif δ); motif score broadcast to all nodes in motif
  simplified_motif_readout_maxmean — same as simplified_motif_readout but motif embedding = concat(max, mean) over nodes per motif (no intra-attention pool for z_k)
  simplified_motif_readout_maxmean_z1 — same as simplified_motif_readout_maxmean but motif MLP input z_k = LN(z^(1)) || LN(max||mean) (emb_stop=0 mean-pool per motif + max_mean)
  simplified_motif_readout_maxmean_injection_ablation — maxmean + no info warmup; sweep injection 010 / 101 / 011 / 111
  simplified_motif_readout_maxmean_info_loss_ablation — maxmean + 010 + no warmup; sweep info_loss_coef ∈ {0.01, 0.1, 0.3}
  no_info_loss — info_loss_coef=0, 011 injection; (1) base GSAT decay final_r=0.8, (2) motif readout max_mean + node sampling, (3) motif readout max_mean + motif sampling
  no_info_loss_deterministic_attn — same as no_info_loss intent but no_attention_sampling=True (σ(logits) only, no Concrete noise); (1) base GSAT, (2) motif readout max_mean (node-level motif sampling)

Injection codes map to GSAT flags (w_node ≡ w_feat): 100=w_feat only, 010=w_message only, 001=w_readout only, 111=all three.

Pre-streamlining registry: run_mutagenicity_gsat_experiment_legacy_experiment_groups.EXPERIMENT_GROUPS_LEGACY
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from experiment_configs import get_base_config, ARCHITECTURES, PAPER_DATASETS, MOL_REGRESSION_DATASETS
from utils import set_seed

SUPPORTED_DATASETS = list(dict.fromkeys([
    'Mutagenicity', 'BBBP', 'hERG', 'Benzene',
    'Alkane_Carbonyl', 'Fluoride_Carbonyl', 'esol', 'Lipophilicity',
    'ogbg_molhiv',
    'ogbg_molbace', 'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_moltox21', 'ogbg_molsider',
] + list(PAPER_DATASETS)))
DATASETS_WITH_MOTIFS = [
    'Mutagenicity', 'BBBP', 'hERG', 'Benzene',
    'Alkane_Carbonyl', 'Fluoride_Carbonyl', 'esol', 'Lipophilicity',
]
DATASET = 'Mutagenicity'
MOTIF_SCORES_TEMPLATE = '/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/All0.5_learn_unk+motif_scores/{dataset}_{model}_motif_scores.csv'

# Injection ablation: user notation 100 = w_feat (node input) only, 010 = w_message only, 001 = w_readout only.
INJECTION_PRESETS = {
    '100': {'w_feat': True, 'w_message': False, 'w_readout': False},
    '010': {'w_feat': False, 'w_message': True, 'w_readout': False},
    '001': {'w_feat': False, 'w_message': False, 'w_readout': True},
    '101': {'w_feat': True, 'w_message': False, 'w_readout': True},
    '011': {'w_feat': False, 'w_message': True, 'w_readout': True},
    '111': {'w_feat': True, 'w_message': True, 'w_readout': True},
}

# motif_readout_pred_info_only: which backbone depth feeds motif pooling (see run_gsat motif_readout_emb_stop).
_PRED_INFO_LAYER_VARIANTS = (
    ('enc', 'encoder'),
    ('l0', 0),
    ('l1', 1),
    ('l2', 2),
    ('final', 'final'),
)

# Shared decay kwargs (decay_interval left to get_base_config: 10 for most backbones, 5 for PNA)
_DECAY_R_BASE = {
    'fix_r': False,
    'init_r': 0.9,
    'decay_r': 0.1,
}

_BASE_GSAT_NONE = {
    'motif_incorporation_method': None,
    'motif_loss_coef': 0,
    'between_motif_coef': 0,
    'pred_loss_coef': 1.0,
    'info_loss_coef': 1.0,
}

# ---------------------------------------------------------------------------
# Experiment group definitions (streamlined)
# Full pre-cleanup registry: run_mutagenicity_gsat_experiment_legacy_experiment_groups.EXPERIMENT_GROUPS_LEGACY
# ---------------------------------------------------------------------------

EXPERIMENT_GROUPS = {
    'vanilla_gnn': {
        'experiment_name': 'vanilla_gnn',
        'variants': [
            {
                'variant_id': 'vanilla',
                'gsat_overrides': {
                    'tuning_id': 'vanilla',
                    'fix_r': 1.0,
                },
                'learn_edge_att': False,
                'vanilla_clean': True,
            },
        ],
    },
    'base_gsat_fix_r': {
        'experiment_name': 'base_gsat_fix_r',
        'variants': [
            {
                'variant_id': f'fix_r{r}_w010',
                'gsat_overrides': {
                    'tuning_id': f'fix_r{r}_w010',
                    'fix_r': r,
                    **_BASE_GSAT_NONE,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },
    'base_gsat_decay_r': {
        'experiment_name': 'base_gsat_decay_r',
        'variants': [
            {
                'variant_id': f'decay_final{fr}_w010',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}_w010',
                    **_DECAY_R_BASE,
                    'final_r': fr,
                    **_BASE_GSAT_NONE,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for fr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },
    'base_gsat_decay_r_injection': {
        'experiment_name': 'base_gsat_decay_r_injection',
        'variants': [
            {
                'variant_id': f'decay_f0.8_inj{code}',
                'gsat_overrides': {
                    'tuning_id': f'decay_f0.8_inj{code}',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    **_BASE_GSAT_NONE,
                    **INJECTION_PRESETS[code],
                },
                'learn_edge_att': False,
            }
            for code in ['100', '010', '001', '101', '011']
        ],
    },
    'base_gsat_motif_loss': {
        'experiment_name': 'base_gsat_motif_loss',
        'variants': [
            {
                'variant_id': 'motif_loss_decay_f0.8_w010',
                'gsat_overrides': {
                    'tuning_id': 'motif_loss_decay_f0.8_w010',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'loss',
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_decay_w_message': {
        'experiment_name': 'motif_readout_decay_w_message',
        'variants': [
            {
                'variant_id': 'readout_decay_f0.8_w010_node_samp',
                'gsat_overrides': {
                    'tuning_id': 'readout_decay_f0.8_w010_node_samp',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': False,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'readout_decay_f0.8_w010_motif_samp',
                'gsat_overrides': {
                    'tuning_id': 'readout_decay_f0.8_w010_motif_samp',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'test_gradients': {
        'experiment_name': 'test_gradients',
        'variants': [
            {
                'variant_id': 'gradprobe_readout_maxmean_motifsamp_inj111',
                'gsat_overrides': {
                    'tuning_id': 'gradprobe_readout_maxmean_motifsamp_inj111',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_level_info_loss': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_grad_probe': True,
                    'motif_grad_probe_every': 1,
                    'motif_grad_probe_max_batches': 1,
                    'motif_grad_probe_epochs': -1,
                    'motif_grad_probe_start_epoch': 10,
                    'motif_grad_probe_epoch_every': 10,
                    **INJECTION_PRESETS['111'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'test_gradient_info_coef0.2_tau2': {
        'experiment_name': 'test_gradient_info_coef0.2_tau2',
        'variants': [
            {
                'variant_id': 'gradprobe_readout_maxmean_motifsamp_inj111_info0.2_tau2',
                'gsat_overrides': {
                    'tuning_id': 'gradprobe_readout_maxmean_motifsamp_inj111_info0.2_tau2',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_level_info_loss': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 0.2,
                    'attention_sampling_temp': 2.0,
                    'motif_grad_probe': True,
                    'motif_grad_probe_every': 1,
                    'motif_grad_probe_max_batches': 1,
                    'motif_grad_probe_epochs': -1,
                    'motif_grad_probe_start_epoch': 10,
                    'motif_grad_probe_epoch_every': 10,
                    **INJECTION_PRESETS['111'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'no_info_loss': {
        'experiment_name': 'no_info_loss_inj011',
        'variants': [
            {
                'variant_id': 'no_info_loss_base',
                'gsat_overrides': {
                    'tuning_id': 'no_info_loss_base',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    **_BASE_GSAT_NONE,
                    'info_loss_coef': 0.0,
                    **INJECTION_PRESETS['011'],
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'no_info_loss_maxmean_node_samp',
                'gsat_overrides': {
                    'tuning_id': 'no_info_loss_maxmean_node_samp',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': False,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 0.0,
                    **INJECTION_PRESETS['011'],
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'no_info_loss_maxmean_motif_samp',
                'gsat_overrides': {
                    'tuning_id': 'no_info_loss_maxmean_motif_samp',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 0.0,
                    **INJECTION_PRESETS['011'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'no_info_loss_deterministic_attn': {
        'experiment_name': 'no_info_loss_deterministic_attn_inj011',
        'variants': [
            {
                'variant_id': 'no_info_loss_det_base',
                'gsat_overrides': {
                    'tuning_id': 'no_info_loss_det_base',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    **_BASE_GSAT_NONE,
                    'info_loss_coef': 0.0,
                    'no_attention_sampling': True,
                    **INJECTION_PRESETS['011'],
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'no_info_loss_det_maxmean',
                'gsat_overrides': {
                    'tuning_id': 'no_info_loss_det_maxmean',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': False,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 0.0,
                    'no_attention_sampling': True,
                    **INJECTION_PRESETS['011'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_decay_injection_ablation': {
        'experiment_name': 'motif_readout_decay_injection_ablation',
        'variants': [
            {
                'variant_id': f'readout_decay_f0.8_node_samp_inj{code}',
                'gsat_overrides': {
                    'tuning_id': f'readout_decay_f0.8_node_samp_inj{code}',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': False,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS[code],
                },
                'learn_edge_att': False,
            }
            for code in ['100', '010', '001', '101', '011']
        ],
    },
    'base_gsat_readout_intra_att': {
        'experiment_name': 'base_gsat_readout_intra_att',
        'variants': [
            {
                'variant_id': 'decay_f0.8_w010_readout_intra_att',
                'gsat_overrides': {
                    'tuning_id': 'decay_f0.8_w010_readout_intra_att',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'intra_att',
                    'motif_level_sampling': False,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    # Motif extractor ℓ_m; node logit = ℓ_m(i) + shift_scale * f([h||z||α]) (legacy: motif_prior_gate_full_mlp).
    'motif_readout_prior_node_gate': {
        'experiment_name': 'motif_readout_prior_node_gate',
        'variants': [
            {
                'variant_id': f'decay_f0.8_w010_readout_prior_gate_s{shift_scale:g}',
                'gsat_overrides': {
                    'tuning_id': f'decay_f0.8_w010_readout_prior_gate_s{shift_scale:g}',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': shift_scale,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for shift_scale in (0.0, 0.1, 0.5, 1.0)
        ],
    },
    # Bounded shift (tanh) + s(epoch): 0 until warmup, linear ramp to shift_scale, then hold (see run_gsat.effective_motif_prior_shift_scale).
    'motif_readout_prior_node_gate_tanh_sched': {
        'experiment_name': 'motif_readout_prior_node_gate_tanh_sched',
        'variants': [
            {
                'variant_id': f'decay_f0.8_w010_prior_tanhsched_s{shift_scale:g}',
                'gsat_overrides': {
                    'tuning_id': f'decay_f0.8_w010_prior_tanhsched_s{shift_scale:g}',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_prior_node_gate': True,
                    'motif_prior_gate_tanh': True,
                    'motif_prior_shift_schedule': 'warmup_linear',
                    'motif_prior_shift_warmup_epochs': 25,
                    'motif_prior_shift_ramp_epochs': 45,
                    'motif_prior_shift_scale': shift_scale,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for shift_scale in (0.0, 0.1, 0.5, 1.0)
        ],
    },
    'motif_readout_weight_diversity': {
        'experiment_name': 'motif_readout_weight_diversity',
        'variants': [
            {
                'variant_id': 'decay_f0.8_w010_readout_motif_div',
                'gsat_overrides': {
                    'tuning_id': 'decay_f0.8_w010_readout_motif_div',
                    **_DECAY_R_BASE,
                    'final_r': 0.8,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': False,
                    'motif_weight_diversity_coef': 1.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    # -------------------------------------------------------------------------
    # Motif readout ablations (E1–E10): decay to final_r=0.7, motif-level sampling,
    # mean pool + MLP + sigmoid baseline with prior node gate (additive s=0.1).
    # Run with: --models GCN SAGE GAT GIN --experiments <group_name>
    # -------------------------------------------------------------------------
    'motif_readout_baseline_f07': {
        'experiment_name': 'motif_readout_baseline_f07',
        'variants': [
            {
                'variant_id': 'baseline_decay_f0.7_w010_motif_samp_prior_gate',
                'gsat_overrides': {
                    'tuning_id': 'baseline_decay_f0.7_w010_motif_samp_prior_gate',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e1_logit_standardize': {
        'experiment_name': 'motif_readout_e1_logit_standardize',
        'variants': [
            {
                'variant_id': 'e1_std_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e1_std_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_logit_standardize_per_graph': True,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e2_temperature': {
        'experiment_name': 'motif_readout_e2_temperature',
        'variants': [
            {
                'variant_id': 'e2_temp_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e2_temp_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_logit_temperature_learned': True,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e3_max_pool': {
        'experiment_name': 'motif_readout_e3_max_pool',
        'variants': [
            {
                'variant_id': 'e3_max_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e3_max_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e4_max_mean_pool': {
        'experiment_name': 'motif_readout_e4_max_mean_pool',
        'variants': [
            {
                'variant_id': 'e4_maxmean_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e4_maxmean_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e5_interp_head': {
        'experiment_name': 'motif_readout_e5_interp_head',
        'variants': [
            {
                'variant_id': 'e5_interp_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e5_interp_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_readout_interp_head': True,
                    'motif_interp_distill_coef': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e6_no_gate': {
        'experiment_name': 'motif_readout_e6_no_gate',
        'variants': [
            {
                'variant_id': 'e6_nogate_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e6_nogate_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': False,
                    'motif_readout_no_gate': True,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e7_multiplicative_gate': {
        'experiment_name': 'motif_readout_e7_multiplicative_gate',
        'variants': [
            {
                'variant_id': 'e7_multgate_decay_f0.7',
                'gsat_overrides': {
                    'tuning_id': 'e7_multgate_decay_f0.7',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_gate_mode': 'multiplicative',
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_e8_entropy_sweep': {
        'experiment_name': 'motif_readout_e8_entropy_sweep',
        'variants': [
            {
                'variant_id': f'e8_entropy_g{gamma:g}',
                'gsat_overrides': {
                    'tuning_id': f'e8_entropy_g{gamma:g}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_entropy_coef': gamma,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for gamma in (0.05, 0.1, 0.5)
        ],
    },
    'motif_readout_e9_motif_ib_sweep': {
        'experiment_name': 'motif_readout_e9_motif_ib_sweep',
        'variants': [
            {
                'variant_id': f'e9_motifib_bm{bm:g}_rfr{fr:g}',
                'gsat_overrides': {
                    'tuning_id': f'e9_motifib_bm{bm:g}_rfr{fr:g}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_level_ib_coef': bm,
                    'motif_ib_final_r': fr,
                    'motif_ib_init_r': 0.9,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for bm in (0.1, 0.5, 1.0)
            for fr in (0.3, 0.5, 0.7)
        ],
    },
    'motif_readout_e10_align_sweep': {
        'experiment_name': 'motif_readout_e10_align_sweep',
        'variants': [
            {
                'variant_id': f'e10_align_lam{lam:g}',
                'gsat_overrides': {
                    'tuning_id': f'e10_align_lam{lam:g}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': True,
                    'motif_prior_shift_scale': 0.1,
                    'motif_align_loss_coef': lam,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for lam in (0.1, 0.5, 1.0)
        ],
    },
    # No prior node gate; entropy bonus for spread; motif-level sampling → downstream GNN (w_message 010).
    # Pooling sweep: mean | max | max_mean | intra_att (attention pooling within motif).
    'motif_readout_entropy_pool_sweep': {
        'experiment_name': 'motif_readout_entropy_pool_sweep',
        'variants': [
            {
                'variant_id': f'entropypool_{pool}_w010',
                'gsat_overrides': {
                    'tuning_id': f'entropypool_{pool}_w010',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': pool,
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': False,
                    'motif_readout_no_gate': True,
                    'motif_logit_temperature_learned': False,
                    'motif_logit_standardize_per_graph': False,
                    'motif_entropy_coef': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for pool in ('mean', 'max', 'max_mean', 'intra_att')
        ],
    },
    # max_mean readout fixed; compare node-injection vs full forward with edge attention from lifted motif scores.
    'motif_readout_maxmean_node_vs_edge_att': {
        'experiment_name': 'motif_readout_maxmean_node_vs_edge_att',
        'variants': [
            {
                'variant_id': 'maxmean_node_inj_w010',
                'gsat_overrides': {
                    'tuning_id': 'maxmean_node_inj_w010',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': False,
                    'motif_readout_no_gate': True,
                    'motif_logit_temperature_learned': False,
                    'motif_logit_standardize_per_graph': False,
                    'motif_entropy_coef': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'maxmean_edge_att_w010',
                'gsat_overrides': {
                    'tuning_id': 'maxmean_edge_att_w010',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_prior_node_gate': False,
                    'motif_readout_no_gate': True,
                    'motif_logit_temperature_learned': False,
                    'motif_logit_standardize_per_graph': False,
                    'motif_entropy_coef': 0.1,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': True,
            },
        ],
    },
    # Only prediction + GSAT info (motif-level att_for_loss); all other readout auxiliaries off.
    'motif_readout_pred_info_only': {
        'experiment_name': 'motif_readout_pred_info_only',
        'variants': [
            {
                'variant_id': f'predinfo_maxmean_{tag}_w010',
                'gsat_overrides': {
                    'tuning_id': f'predinfo_maxmean_{tag}_w010',
                    'fix_r': False,
                    'init_r': 0.9,
                    'decay_r': 0.1,
                    'final_r': 0.7,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'max_mean',
                    'motif_level_sampling': True,
                    'motif_level_info_loss': True,
                    'motif_prior_node_gate': False,
                    'motif_readout_no_gate': True,
                    'motif_logit_temperature_learned': False,
                    'motif_logit_standardize_per_graph': False,
                    'motif_readout_emb_stop': stop,
                    'pred_loss_coef': 1.0,
                    'info_loss_coef': 1.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                    'motif_weight_diversity_coef': 0.0,
                    'motif_entropy_coef': 0.0,
                    'motif_level_ib_coef': 0.0,
                    'motif_align_loss_coef': 0.0,
                    'motif_interp_distill_coef': 0.0,
                    **INJECTION_PRESETS['010'],
                },
                'learn_edge_att': False,
            }
            for tag, stop in _PRED_INFO_LAYER_VARIANTS
        ],
    },
}

# Same hyperparameter grid as base_gsat_decay_r, but loads motif dictionaries from
# path/FOLDS/minority_global/{dataset}_{algo}_fold_{k}_{algo}_minority_global (see DataLoader.py).
_base_decay_r = EXPERIMENT_GROUPS['base_gsat_decay_r']
EXPERIMENT_GROUPS['base_gsat_decay_r_minority_global'] = {
    'experiment_name': 'base_gsat_decay_r_minority_global',
    'variants': [
        {**v, 'data_overrides': {'dictionary_fold_variant': 'minority_global'}}
        for v in _base_decay_r['variants']
    ],
}

# Factored Motif Attention Pipeline: multi-granularity z_k × factored node logits (12 cells: M1–M4 × N1–N3).
# See run_gsat.GSAT._factored_motif_prepare and experiment docstring in experiment_factored_motif.py
_FACTORED_MOTIF_BASE_GSAT = {
    'motif_incorporation_method': 'readout',
    'factored_motif_attention': True,
    'motif_pooling_method': 'intra_att',
    'motif_gate_mode': 'none',
    'motif_readout_no_gate': True,
    'motif_prior_node_gate': False,
    'info_loss_coef': 0.0,
    'use_raw_score_loss': False,
    'use_motif_ib_mean_node_alpha': True,
    'motif_level_ib_coef': 1.0,
    'motif_ib_init_r': 0.9,
    'motif_ib_final_r': 0.7,
    'fix_r': False,
    'init_r': 0.9,
    'decay_r': 0.1,
    'final_r': 0.7,
    'pred_loss_coef': 1.0,
    'motif_loss_coef': 0.0,
    'between_motif_coef': 0.0,
    'w_feat': True,
    'w_message': True,
    'w_readout': True,
}

# Regularized factored pipeline (mutually exclusive with factored_motif_attention); see run_gsat.GSAT
_FACTORED_MOTIF_REG_BASE_GSAT = {
    'motif_incorporation_method': 'readout',
    'factored_motif_regularized': True,
    'factored_motif_attention': False,
    'motif_pooling_method': 'intra_att',
    'motif_gate_mode': 'none',
    'motif_readout_no_gate': True,
    'motif_prior_node_gate': False,
    'info_loss_coef': 0.0,
    'info_warmup_epochs': 20,
    'ib_ramp_epochs': 20,
    'use_raw_score_loss': False,
    'use_motif_ib_mean_node_alpha': False,
    'motif_level_ib_coef': 0.01,
    'motif_ib_init_r': 0.9,
    'decay_interval': 10,
    'factored_motif_zk_dropout_p': 0.3,
    'factored_motif_node_logit_clamp': 4.0,
    'fix_r': False,
    'init_r': 0.9,
    'decay_r': 0.1,
    'final_r': 0.7,
    'pred_loss_coef': 1.0,
    'motif_loss_coef': 0.0,
    'between_motif_coef': 0.0,
    'w_feat': True,
    'w_message': True,
    'w_readout': True,
}

# IB prior r(t) uses motif_ib_final_r with get_r(decay_interval=10, decay_r=0.1)
FACTORED_MOTIF_IB_FINAL_R_VARIANTS = (
    ('factored_reg_ibf070', 0.7),
    ('factored_reg_ibf050', 0.5),
    ('factored_reg_ibf030', 0.3),
)

EXPERIMENT_GROUPS['factored_motif_additive'] = {
    'experiment_name': 'factored_motif_additive',
    'variants': [
        {
            'variant_id': vid,
            'gsat_overrides': {
                'tuning_id': vid,
                **_DECAY_R_BASE,
                **{**_FACTORED_MOTIF_REG_BASE_GSAT, 'motif_ib_final_r': ibf},
            },
            'learn_edge_att': False,
        }
        for vid, ibf in FACTORED_MOTIF_IB_FINAL_R_VARIANTS
    ],
}

# Like factored_motif_additive but: ℓ_k from MLP(LN(z^att)) only; 010; L_info on σ(ℓ_k) (use_raw_score_loss); coef ~ motif_level_ib (0.01); not motif_ib; info_warmup 20; final_r=0.8
_SIMPLIFIED_FACTORED_MOTIF_ADDITIVE_GSAT = {
    **_FACTORED_MOTIF_REG_BASE_GSAT,
    'factored_motif_zk_zatt_only': True,
    'motif_level_info_loss': True,
    'use_raw_score_loss': True,
    'info_loss_coef': 0.01,
    'info_warmup_epochs': 20,
    'ib_ramp_epochs': 0,
    'motif_level_ib_coef': 0.0,
    'final_r': 0.8,
    **INJECTION_PRESETS['010'],
}

EXPERIMENT_GROUPS['simplified_factored_motif_additive'] = {
    'experiment_name': 'simplified_factored_motif_additive',
    'variants': [
        {
            'variant_id': 'simplified_factored_additive',
            'gsat_overrides': {
                'tuning_id': 'simplified_factored_additive',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_FACTORED_MOTIF_ADDITIVE_GSAT,
            },
            'learn_edge_att': False,
        },
    ],
}

# Broadcast-only node logits (no δ); else identical to simplified_factored_motif_additive
_SIMPLIFIED_MOTIF_READOUT_GSAT = {
    **_SIMPLIFIED_FACTORED_MOTIF_ADDITIVE_GSAT,
    'factored_motif_no_intra_delta': True,
}

EXPERIMENT_GROUPS['simplified_motif_readout'] = {
    'experiment_name': 'simplified_motif_readout',
    'variants': [
        {
            'variant_id': 'simplified_motif_readout',
            'gsat_overrides': {
                'tuning_id': 'simplified_motif_readout',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_MOTIF_READOUT_GSAT,
            },
            'learn_edge_att': False,
        },
    ],
}

_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_GSAT = {
    **_SIMPLIFIED_MOTIF_READOUT_GSAT,
    'motif_pooling_method': 'max_mean',
}

# max_mean readout + motif L_info from epoch 0 (no prediction-only warmup)
_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_NO_WARMUP_GSAT = {
    **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_GSAT,
    'info_warmup_epochs': 0,
}

# 010=edge(message) only; 101=node+readout; 011=edge+readout; 111=node+edge+readout
_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_INJECTION_ABLATION_CODES = (
    ('010', 'edge_only'),
    ('101', 'node_readout'),
    ('011', 'edge_readout'),
    ('111', 'node_edge_readout'),
)

EXPERIMENT_GROUPS['simplified_motif_readout_maxmean_injection_ablation'] = {
    'experiment_name': 'simplified_motif_readout_maxmean_injection_ablation',
    'variants': [
        {
            'variant_id': f'maxmean_inj_{tag}',
            'gsat_overrides': {
                'tuning_id': f'simplified_maxmean_inj{code}',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_NO_WARMUP_GSAT,
                **INJECTION_PRESETS[code],
            },
            'learn_edge_att': False,
        }
        for code, tag in _SIMPLIFIED_MOTIF_READOUT_MAXMEAN_INJECTION_ABLATION_CODES
    ],
}

# Edge-only masking (010); sweep L_info strength
_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_INFO_LOSS_ABLATION_VARIANTS = (
    ('info_coef_001', 0.01),
    ('info_coef_010', 0.1),
    ('info_coef_030', 0.3),
)

EXPERIMENT_GROUPS['simplified_motif_readout_maxmean_info_loss_ablation'] = {
    'experiment_name': 'simplified_motif_readout_maxmean_info_loss_ablation',
    'variants': [
        {
            'variant_id': f'maxmean_{vid}',
            'gsat_overrides': {
                'tuning_id': f'simplified_maxmean_{vid}',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_NO_WARMUP_GSAT,
                **INJECTION_PRESETS['010'],
                'info_loss_coef': coef,
            },
            'learn_edge_att': False,
        }
        for vid, coef in _SIMPLIFIED_MOTIF_READOUT_MAXMEAN_INFO_LOSS_ABLATION_VARIANTS
    ],
}

EXPERIMENT_GROUPS['simplified_motif_readout_maxmean'] = {
    'experiment_name': 'simplified_motif_readout_maxmean',
    'variants': [
        {
            'variant_id': 'simplified_motif_readout_maxmean',
            'gsat_overrides': {
                'tuning_id': 'simplified_motif_readout_maxmean',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_GSAT,
            },
            'learn_edge_att': False,
        },
    ],
}

# max_mean z^att + layer-0 mean-pooled z^(1) before motif MLP (RegularizedMotifScoringMLP in_dim = 3H)
_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_Z1_GSAT = {
    **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_GSAT,
    'factored_motif_zk_zatt_only': False,
}

EXPERIMENT_GROUPS['simplified_motif_readout_maxmean_z1'] = {
    'experiment_name': 'simplified_motif_readout_maxmean_z1',
    'variants': [
        {
            'variant_id': 'simplified_motif_readout_maxmean_z1',
            'gsat_overrides': {
                'tuning_id': 'simplified_motif_readout_maxmean_z1',
                **_DECAY_R_BASE,
                **_SIMPLIFIED_MOTIF_READOUT_MAXMEAN_Z1_GSAT,
            },
            'learn_edge_att': False,
        },
    ],
}

EXPERIMENT_GROUPS['factored_motif_attention_grid'] = {
    'experiment_name': 'factored_motif_attention_grid',
    'variants': [
        {
            'variant_id': f'factored_{zk}_{n}',
            'gsat_overrides': {
                'tuning_id': f'factored_{zk}_{n}',
                **_DECAY_R_BASE,
                **_FACTORED_MOTIF_BASE_GSAT,
                'factored_motif_zk_axis': zk,
                'factored_node_logit_axis': n,
            },
            'learn_edge_att': False,
        }
        for zk in ('M1', 'M2', 'M3', 'M4')
        for n in ('N1', 'N2', 'N3')
    ],
}

ALL_EXPERIMENT_NAMES = list(EXPERIMENT_GROUPS.keys())


def run_one(model_name, fold, variant, experiment_name, seed, cuda_id, data_dir, dataset_name, embedding_viz_every=10):
    """Run a single experiment: one model, one fold, one variant, one seed."""
    config = get_base_config(model_name, dataset_name, gsat_overrides=variant['gsat_overrides'])
    config['model_config']['use_edge_attr'] = False
    config['shared_config']['learn_edge_att'] = variant['learn_edge_att']
    # W&B embedding PCA (run_gsat.log_valid_embedding_viz_wandb); 0 disables
    config['shared_config']['embedding_viz_every'] = int(embedding_viz_every)
    if 'shared_overrides' in variant:
        config['shared_config'].update(variant['shared_overrides'])
    if 'data_overrides' in variant:
        config['data_config'].update(variant['data_overrides'])
    config['GSAT_config']['experiment_name'] = experiment_name
    # Log variant identity on W&B with full GSAT config (train_gsat_one_seed run_config.gsat).
    config['GSAT_config']['variant_id'] = variant['variant_id']

    if 'model_overrides' in variant:
        config['model_config'].update(variant['model_overrides'])

    scores_path = config['GSAT_config'].get('motif_scores_path')
    if scores_path and '{' in scores_path:
        config['GSAT_config']['motif_scores_path'] = scores_path.format(
            dataset=dataset_name, model=model_name
        )

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    variant_id = variant['variant_id']
    log_dir = data_dir / f'{dataset_name}-fold{fold}' / 'logs' / f'{model_name}-seed{seed}-GSAT-{variant_id}'

    set_seed(seed)

    task_type = 'regression' if dataset_name in MOL_REGRESSION_DATASETS else 'classification'

    if variant.get('vanilla_clean', False):
        from run_gsat import train_vanilla_gnn_one_seed
        hparam_dict, metric_dict = train_vanilla_gnn_one_seed(
            config, data_dir, log_dir, model_name, dataset_name,
            device, seed, fold=fold, task_type=task_type
        )
    else:
        from run_gsat import train_gsat_one_seed
        hparam_dict, metric_dict = train_gsat_one_seed(
            config, data_dir, log_dir, model_name, dataset_name,
            'GSAT', device, seed, fold=fold, task_type=task_type
        )
    return metric_dict


def main():
    parser = argparse.ArgumentParser(description='GSAT experiments for molecular datasets')
    parser.add_argument('--dataset', type=str, default=DATASET,
                        choices=SUPPORTED_DATASETS, help='Dataset to run experiments on')
    parser.add_argument('--experiments', type=str, nargs='+', default=ALL_EXPERIMENT_NAMES,
                        choices=ALL_EXPERIMENT_NAMES, help='Which experiment groups to run')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='Folds to run (default: [0,1] for MolDatasets, [0] for OGB)')
    parser.add_argument('--models', type=str, nargs='+', default=ARCHITECTURES, help='Models (backbones)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id (-1 for CPU)')
    parser.add_argument('--embedding_viz_every', type=int, default=10,
                        help='Log W&B PCA embedding panels + motif tables every N valid epochs (0=off). '
                             'Binary non-multilabel GSAT only. Default: 10.')
    parser.add_argument('--wandb_lite', action='store_true', default=False,
                        help='Set MOTIFSAT_WANDB_LITE=1: smaller local W&B run directory (see run_gsat wandb helpers).')
    parser.add_argument('--wandb_log_every', type=int, default=None,
                        help='Set MOTIFSAT_WANDB_LOG_EVERY (throttle wandb.log; default 50 with --wandb_lite).')
    args = parser.parse_args()

    if args.wandb_lite:
        os.environ['MOTIFSAT_WANDB_LITE'] = '1'
    if args.wandb_log_every is not None:
        os.environ['MOTIFSAT_WANDB_LOG_EVERY'] = str(max(1, int(args.wandb_log_every)))

    dataset_name = args.dataset

    # Default folds: [0,1] for MolDatasets with folds, [0] for OGB (no fold splitting)
    if args.folds is not None:
        folds = args.folds
    elif dataset_name in DATASETS_WITH_MOTIFS:
        folds = [0, 1]
    else:
        folds = [0]

    # Warn if motif-requiring experiments are selected for OGB datasets
    motif_experiments = {
        'base_gsat_motif_loss',
        'motif_readout_decay_w_message',
        'motif_readout_decay_injection_ablation',
        'motif_readout_prior_node_gate',
        'motif_readout_prior_node_gate_tanh_sched',
        'motif_readout_weight_diversity',
        'motif_readout_baseline_f07',
        'motif_readout_e1_logit_standardize',
        'motif_readout_e2_temperature',
        'motif_readout_e3_max_pool',
        'motif_readout_e4_max_mean_pool',
        'motif_readout_e5_interp_head',
        'motif_readout_e6_no_gate',
        'motif_readout_e7_multiplicative_gate',
        'motif_readout_e8_entropy_sweep',
        'motif_readout_e9_motif_ib_sweep',
        'motif_readout_e10_align_sweep',
        'motif_readout_entropy_pool_sweep',
        'motif_readout_maxmean_node_vs_edge_att',
        'motif_readout_pred_info_only',
        'factored_motif_attention_grid',
        'factored_motif_additive',
        'simplified_factored_motif_additive',
        'simplified_motif_readout',
        'simplified_motif_readout_maxmean',
        'simplified_motif_readout_maxmean_z1',
        'simplified_motif_readout_maxmean_injection_ablation',
        'simplified_motif_readout_maxmean_info_loss_ablation',
        'test_gradients',
        'test_gradient_info_coef0.2_tau2',
        'no_info_loss',
        'no_info_loss_deterministic_attn',
    }
    if dataset_name not in DATASETS_WITH_MOTIFS:
        skipped = [e for e in args.experiments if e in motif_experiments]
        if skipped:
            print(f'[WARNING] Skipping motif experiments for {dataset_name} (no motif info): {skipped}')
            args.experiments = [e for e in args.experiments if e not in motif_experiments]

    config_dir = Path(__file__).resolve().parent / 'configs'
    with open(config_dir / 'global_config.yml') as f:
        global_config = yaml.safe_load(f)
    data_dir = Path(global_config['data_dir'])

    total_variants = sum(len(EXPERIMENT_GROUPS[e]['variants']) for e in args.experiments)
    total = len(folds) * len(args.models) * total_variants * len(args.seeds)
    n = 0

    print(f'\n[INFO] Dataset: {dataset_name}')
    print(f'[INFO] Folds: {folds}')
    print(f'[INFO] embedding_viz_every={args.embedding_viz_every} (W&B PCA panels; use 0 to disable)')

    for exp_key in args.experiments:
        group = EXPERIMENT_GROUPS[exp_key]
        experiment_name = group['experiment_name']
        print(f'\n{"#" * 80}')
        print(f'# Experiment group: {exp_key} (experiment_name={experiment_name})')
        print(f'{"#" * 80}')

        for fold in folds:
            for model_name in args.models:
                for variant in group['variants']:
                    for seed in args.seeds:
                        n += 1
                        vid = variant['variant_id']
                        print('=' * 80)
                        print(f'[{n}/{total}] {experiment_name} dataset={dataset_name} fold={fold} model={model_name} variant={vid} seed={seed}')
                        print('=' * 80)
                        try:
                            run_one(
                                model_name, fold, variant, experiment_name, seed, args.cuda, data_dir, dataset_name,
                                embedding_viz_every=args.embedding_viz_every,
                            )
                        except Exception as e:
                            print(f'[ERROR] {e}')
                            import traceback
                            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
