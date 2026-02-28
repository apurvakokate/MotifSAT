#!/usr/bin/env python
"""
Run GSAT on Mutagenicity with structured experiment groups.

Experiment groups (each has its own experiment_name):
  1. r_impact_node:  Base GSAT, node attention, varying final_r in {0.4, 0.5, 0.6}
  2. r_impact_edge:  Base GSAT, edge attention, varying final_r in {0.4, 0.5, 0.6}
  3. within_motif_consistency_impact:  Node attention, motif_loss_coef in {1.0, 2.0}
  4. between_motif_consistency_impact: Node attention, motif_loss_coef=1.0, between_motif_coef in {1.0, 2.0}
  5. motif_readout_info_loss: Motif readout with motif-level info loss (r=0.5, learn_edge_att=False)
  6. motif_readout_adaptive_r: Motif readout with graph-adaptive r (target_k in {1, 2})
  7. motif_readout_score_r_by_dataset_model: Motif readout with fixed per-motif score r (no decay)
  8. motif_readout_score_r_interpolate: Score r with interpolation (init_r → score_r over decay)
  9. motif_readout_score_r_max: Score r with max schedule (global_r until it drops below score_r)
 10. motif_loss_motif_sampling: Motif consistency loss with motif-level stochastic sampling
     (pool node logits to motif, sample once per motif, broadcast back)
 11. att_injection_point: Node attention injection ablation (W_FEAT / W_MESSAGE / W_READOUT)
     4 variants: feat_only, message_only, readout_only, feat_readout

Usage:
  python run_mutagenicity_gsat_experiment.py --experiments r_impact_node r_impact_edge
  python run_mutagenicity_gsat_experiment.py --experiments within_motif_consistency_impact between_motif_consistency_impact
  python run_mutagenicity_gsat_experiment.py  # runs all 4
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from experiment_configs import get_base_config, ARCHITECTURES
from utils import set_seed

SUPPORTED_DATASETS = ['Mutagenicity', 'BBBP', 'hERG', 'Benzene', 'Alkane_Carbonyl', 'Fluoride_Carbonyl']
DATASET = 'Mutagenicity'  # default, overridden by --dataset CLI arg
MOTIF_SCORES_TEMPLATE = '/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/All0.5_learn_unk+motif_scores/{dataset}_{model}_motif_scores.csv'

# ---------------------------------------------------------------------------
# Experiment group definitions
# Each group is a dict with:
#   experiment_name: str  (used in results path)
#   variants: list of dicts, each with:
#       variant_id: str (tuning_id)
#       gsat_overrides: dict
#       learn_edge_att: bool
# ---------------------------------------------------------------------------

EXPERIMENT_GROUPS = {
    'r_impact_node_no_encoder_2_linear_clf': {
        'experiment_name': 'r_impact_node_no_encoder_2_linear_clf',
        'variants': [
            {
                'variant_id': 'node_r0.5',
                'gsat_overrides': {
                    'tuning_id': 'node_r0.5',
                    'final_r': 0.5,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],

    },
    'r_impact_node': {
        'experiment_name': 'r_impact_node',
        'variants': [
            {
                'variant_id': 'node_r0.4',
                'gsat_overrides': {
                    'tuning_id': 'node_r0.4',
                    'final_r': 0.4,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'node_r0.5',
                'gsat_overrides': {
                    'tuning_id': 'node_r0.5',
                    'final_r': 0.5,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'node_r0.6',
                'gsat_overrides': {
                    'tuning_id': 'node_r0.6',
                    'final_r': 0.6,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'r_impact_edge': {
        'experiment_name': 'r_impact_edge',
        'variants': [
            {
                'variant_id': 'edge_r0.4',
                'gsat_overrides': {
                    'tuning_id': 'edge_r0.4',
                    'final_r': 0.4,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': True,
            },
            {
                'variant_id': 'edge_r0.5',
                'gsat_overrides': {
                    'tuning_id': 'edge_r0.5',
                    'final_r': 0.5,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': True,
            },
            {
                'variant_id': 'edge_r0.6',
                'gsat_overrides': {
                    'tuning_id': 'edge_r0.6',
                    'final_r': 0.6,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': True,
            },
        ],
    },
    'within_motif_consistency_impact': {
        'experiment_name': 'within_motif_consistency_impact',
        'variants': [
            {
                'variant_id': 'within_w1',
                'gsat_overrides': {
                    'tuning_id': 'within_w1',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'within_w2',
                'gsat_overrides': {
                    'tuning_id': 'within_w2',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 2.0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'between_motif_consistency_impact': {
        'experiment_name': 'between_motif_consistency_impact',
        'variants': [
            {
                'variant_id': 'fisher_w1_b1',
                'gsat_overrides': {
                    'tuning_id': 'fisher_w1_b1',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 1.0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'fisher_w1_b2',
                'gsat_overrides': {
                    'tuning_id': 'fisher_w1_b2',
                    'motif_incorporation_method': 'loss',
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 2.0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_info_loss': {
        'experiment_name': 'motif_readout_info_loss',
        'variants': [
            {
                'variant_id': 'readout_motif_info_r0.5',
                'gsat_overrides': {
                    'tuning_id': 'readout_motif_info_r0.5',
                    'final_r': 0.5,
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_adaptive_r': {
        'experiment_name': 'motif_readout_adaptive_r',
        'variants': [
            {
                'variant_id': 'readout_targetk1',
                'gsat_overrides': {
                    'tuning_id': 'readout_targetk1',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'target_k': 1.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'readout_targetk2',
                'gsat_overrides': {
                    'tuning_id': 'readout_targetk2',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'target_k': 2.0,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_score_r_by_dataset_model': {
        'experiment_name': 'motif_readout_score_r_by_dataset_model',
        'variants': [
            {
                'variant_id': 'readout_score_r',
                'gsat_overrides': {
                    'tuning_id': 'readout_score_r',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'motif_scores_path': MOTIF_SCORES_TEMPLATE,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_score_r_interpolate': {
        'experiment_name': 'motif_readout_score_r_interpolate',
        'variants': [
            {
                'variant_id': 'readout_score_r_interp',
                'gsat_overrides': {
                    'tuning_id': 'readout_score_r_interp',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'motif_scores_path': MOTIF_SCORES_TEMPLATE,
                    'score_r_schedule': 'interpolate',
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_readout_score_r_max': {
        'experiment_name': 'motif_readout_score_r_max',
        'variants': [
            {
                'variant_id': 'readout_score_r_max',
                'gsat_overrides': {
                    'tuning_id': 'readout_score_r_max',
                    'motif_incorporation_method': 'readout',
                    'motif_level_info_loss': True,
                    'motif_scores_path': MOTIF_SCORES_TEMPLATE,
                    'score_r_schedule': 'max',
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },
    'motif_loss_motif_sampling': {
        'experiment_name': 'motif_loss_motif_sampling',
        'variants': [
            {
                'variant_id': 'motif_samp_w1_b1',
                'gsat_overrides': {
                    'tuning_id': 'motif_samp_w1_b1',
                    'motif_incorporation_method': 'loss',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 1.0,
                    'between_motif_coef': 1.0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'motif_samp_w0_b1',
                'gsat_overrides': {
                    'tuning_id': 'motif_samp_w0_b1',
                    'motif_incorporation_method': 'loss',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0.0,
                    'between_motif_coef': 1.0,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'motif_samp_w0_b2',
                'gsat_overrides': {
                    'tuning_id': 'motif_samp_w0_b2',
                    'motif_incorporation_method': 'loss',
                    'motif_level_sampling': True,
                    'motif_loss_coef': 0.0,
                    'between_motif_coef': 2.0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'att_injection_point': {
        'experiment_name': 'att_injection_point',
        'variants': [
            {
                'variant_id': 'w_feat_only',
                'gsat_overrides': {
                    'tuning_id': 'w_feat_only',
                    'w_feat': True,
                    'w_message': False,
                    'w_readout': False,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_message_only',
                'gsat_overrides': {
                    'tuning_id': 'w_message_only',
                    'w_feat': False,
                    'w_message': True,
                    'w_readout': False,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_readout_only',
                'gsat_overrides': {
                    'tuning_id': 'w_readout_only',
                    'w_feat': False,
                    'w_message': False,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            },
            {
                'variant_id': 'w_feat_readout',
                'gsat_overrides': {
                    'tuning_id': 'w_feat_readout',
                    'w_feat': True,
                    'w_message': False,
                    'w_readout': True,
                },
                'learn_edge_att': False,
            },
        ],
    },
}

ALL_EXPERIMENT_NAMES = list(EXPERIMENT_GROUPS.keys())


def run_one(model_name, fold, variant, experiment_name, seed, cuda_id, data_dir, dataset_name):
    """Run a single experiment: one model, one fold, one variant, one seed."""
    from run_gsat import train_gsat_one_seed

    config = get_base_config(model_name, dataset_name, gsat_overrides=variant['gsat_overrides'])
    config['shared_config']['learn_edge_att'] = variant['learn_edge_att']
    config['GSAT_config']['experiment_name'] = experiment_name

    # Resolve motif_scores_path template with dataset and model name
    scores_path = config['GSAT_config'].get('motif_scores_path')
    if scores_path and '{' in scores_path:
        config['GSAT_config']['motif_scores_path'] = scores_path.format(
            dataset=dataset_name, model=model_name
        )

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    variant_id = variant['variant_id']
    log_dir = data_dir / f'{dataset_name}-fold{fold}' / 'logs' / f'{model_name}-seed{seed}-GSAT-{variant_id}'

    set_seed(seed)
    hparam_dict, metric_dict = train_gsat_one_seed(
        config, data_dir, log_dir, model_name, dataset_name,
        'GSAT', device, seed, fold=fold, task_type='classification'
    )
    return metric_dict


def main():
    parser = argparse.ArgumentParser(description='GSAT experiments for molecular datasets')
    parser.add_argument('--dataset', type=str, default=DATASET,
                        choices=SUPPORTED_DATASETS, help='Dataset to run experiments on')
    parser.add_argument('--experiments', type=str, nargs='+', default=ALL_EXPERIMENT_NAMES,
                        choices=ALL_EXPERIMENT_NAMES, help='Which experiment groups to run')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1], help='Folds to run')
    parser.add_argument('--models', type=str, nargs='+', default=ARCHITECTURES, help='Models (backbones)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id (-1 for CPU)')
    args = parser.parse_args()
    
    dataset_name = args.dataset

    config_dir = Path(__file__).resolve().parent / 'configs'
    with open(config_dir / 'global_config.yml') as f:
        global_config = yaml.safe_load(f)
    data_dir = Path(global_config['data_dir'])

    total_variants = sum(len(EXPERIMENT_GROUPS[e]['variants']) for e in args.experiments)
    total = len(args.folds) * len(args.models) * total_variants * len(args.seeds)
    n = 0

    print(f'\n[INFO] Dataset: {dataset_name}')

    for exp_key in args.experiments:
        group = EXPERIMENT_GROUPS[exp_key]
        experiment_name = group['experiment_name']
        print(f'\n{"#" * 80}')
        print(f'# Experiment group: {exp_key} (experiment_name={experiment_name})')
        print(f'{"#" * 80}')

        for fold in args.folds:
            for model_name in args.models:
                for variant in group['variants']:
                    for seed in args.seeds:
                        n += 1
                        vid = variant['variant_id']
                        print('=' * 80)
                        print(f'[{n}/{total}] {experiment_name} dataset={dataset_name} fold={fold} model={model_name} variant={vid} seed={seed}')
                        print('=' * 80)
                        try:
                            run_one(model_name, fold, variant, experiment_name, seed, args.cuda, data_dir, dataset_name)
                        except Exception as e:
                            print(f'[ERROR] {e}')
                            import traceback
                            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
