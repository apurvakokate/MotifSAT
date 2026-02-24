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
  7. motif_readout_score_r: Motif readout with pre-computed motif score r values

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

DATASET = 'Mutagenicity'
MOTIF_SCORES_TEMPLATE = 'hpc-share/ChemIntuit/MOSE-GNN/All0.5_learn_unk+motif_scores/{dataset}_{model}_motif_scores.csv'

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
}

ALL_EXPERIMENT_NAMES = list(EXPERIMENT_GROUPS.keys())


def run_one(model_name, fold, variant, experiment_name, seed, cuda_id, data_dir):
    """Run a single experiment: one model, one fold, one variant, one seed."""
    from run_gsat import train_gsat_one_seed

    config = get_base_config(model_name, DATASET, gsat_overrides=variant['gsat_overrides'])
    config['shared_config']['learn_edge_att'] = variant['learn_edge_att']
    config['GSAT_config']['experiment_name'] = experiment_name

    # Resolve motif_scores_path template with dataset and model name
    scores_path = config['GSAT_config'].get('motif_scores_path')
    if scores_path and '{' in scores_path:
        config['GSAT_config']['motif_scores_path'] = scores_path.format(
            dataset=DATASET, model=model_name
        )

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    variant_id = variant['variant_id']
    log_dir = data_dir / f'{DATASET}-fold{fold}' / 'logs' / f'{model_name}-seed{seed}-GSAT-{variant_id}'

    set_seed(seed)
    hparam_dict, metric_dict = train_gsat_one_seed(
        config, data_dir, log_dir, model_name, DATASET,
        'GSAT', device, seed, fold=fold, task_type='classification'
    )
    return metric_dict


def main():
    parser = argparse.ArgumentParser(description='Mutagenicity GSAT experiments')
    parser.add_argument('--experiments', type=str, nargs='+', default=ALL_EXPERIMENT_NAMES,
                        choices=ALL_EXPERIMENT_NAMES, help='Which experiment groups to run')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1], help='Folds to run')
    parser.add_argument('--models', type=str, nargs='+', default=ARCHITECTURES, help='Models (backbones)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id (-1 for CPU)')
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parent / 'configs'
    with open(config_dir / 'global_config.yml') as f:
        global_config = yaml.safe_load(f)
    data_dir = Path(global_config['data_dir'])

    total_variants = sum(len(EXPERIMENT_GROUPS[e]['variants']) for e in args.experiments)
    total = len(args.folds) * len(args.models) * total_variants * len(args.seeds)
    n = 0

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
                        print(f'[{n}/{total}] {experiment_name} fold={fold} model={model_name} variant={vid} seed={seed}')
                        print('=' * 80)
                        try:
                            run_one(model_name, fold, variant, experiment_name, seed, args.cuda, data_dir)
                        except Exception as e:
                            print(f'[ERROR] {e}')
                            import traceback
                            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
