#!/usr/bin/env python
"""
Run GSAT on Mutagenicity: all folds, all architectures, 4 GSAT variants.
Uses the same base config as run_gsat_replication (via experiment_configs.get_base_config).

Variants:
  1. Node attention, no motif (loss/readout/graph)
  2. Edge attention, no motif
  3. Node attention + motif consistency loss (very high)
  4. Node attention + motif consistency loss (comparable)

Wandb: best/clf_roc_* and motif_edge_att/* (per-motif edge weight min/max) are logged.
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

# Variant id -> (GSAT overrides, use_edge_att)
MUTAGENICITY_VARIANTS = {
    # --- Baselines (no motif loss) ---
    'node_baseline': {
        'gsat_overrides': {
            'tuning_id': 'node_baseline',
            'motif_incorporation_method': None,
            'motif_loss_coef': 0,
            'between_motif_coef': 0,
        },
        'learn_edge_att': False,
    },
    'edge_baseline': {
        'gsat_overrides': {
            'tuning_id': 'edge_baseline',
            'motif_incorporation_method': None,
            'motif_loss_coef': 0,
            'between_motif_coef': 0,
        },
        'learn_edge_att': True,
    },
    # --- Within-only motif consistency (original formulation) ---
    'node_motif_loss_high': {
        'gsat_overrides': {
            'tuning_id': 'node_motif_loss_high',
            'motif_incorporation_method': 'loss',
            'motif_loss_coef': 10,
            'between_motif_coef': 0,
        },
        'learn_edge_att': False,
    },
    'node_motif_loss_comparable': {
        'gsat_overrides': {
            'tuning_id': 'node_motif_loss_comparable',
            'motif_incorporation_method': 'loss',
            'motif_loss_coef': 1,
            'between_motif_coef': 0,
        },
        'learn_edge_att': False,
    },
    # --- Fisher-style: within-motif consistency + between-motif discrimination ---
    'node_motif_fisher_balanced': {
        'gsat_overrides': {
            'tuning_id': 'node_motif_fisher_balanced',
            'motif_incorporation_method': 'loss',
            'motif_loss_coef': 1,
            'between_motif_coef': 1,
        },
        'learn_edge_att': False,
    },
    'node_motif_fisher_high': {
        'gsat_overrides': {
            'tuning_id': 'node_motif_fisher_high',
            'motif_incorporation_method': 'loss',
            'motif_loss_coef': 10,
            'between_motif_coef': 10,
        },
        'learn_edge_att': False,
    },
}


def run_one(model_name, fold, variant_id, seed, cuda_id, data_dir):
    """Run a single experiment: one model, one fold, one variant, one seed."""
    from run_gsat import train_gsat_one_seed

    variant = MUTAGENICITY_VARIANTS[variant_id]
    config = get_base_config(model_name, DATASET, gsat_overrides=variant['gsat_overrides'])
    config['shared_config']['learn_edge_att'] = variant['learn_edge_att']
    config['GSAT_config']['experiment_name'] = 'test_motif_loss'

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    log_dir = data_dir / f'{DATASET}-fold{fold}' / 'logs' / f'{model_name}-seed{seed}-GSAT-{variant_id}'

    set_seed(seed)
    hparam_dict, metric_dict = train_gsat_one_seed(
        config, data_dir, log_dir, model_name, DATASET,
        'GSAT', device, seed, fold=fold, task_type='classification'
    )
    return metric_dict


def main():
    parser = argparse.ArgumentParser(description='Mutagenicity GSAT experiment (all folds, architectures, 4 variants)')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1], help='Folds to run')
    parser.add_argument('--models', type=str, nargs='+', default=ARCHITECTURES, help='Models (backbones)')
    parser.add_argument('--variants', type=str, nargs='+', default=list(MUTAGENICITY_VARIANTS.keys()),
                        help='Variant ids')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id (-1 for CPU)')
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parent / 'configs'
    with open(config_dir / 'global_config.yml') as f:
        global_config = yaml.safe_load(f)
    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config.get('num_seeds', 1)
    seeds = args.seeds if args.seeds else list(range(num_seeds))

    total = len(args.folds) * len(args.models) * len(args.variants) * len(seeds)
    n = 0
    for fold in args.folds:
        for model_name in args.models:
            for variant_id in args.variants:
                for seed in seeds:
                    n += 1
                    print('=' * 80)
                    print(f'[{n}/{total}] Mutagenicity fold={fold} model={model_name} variant={variant_id} seed={seed}')
                    print('=' * 80)
                    try:
                        run_one(model_name, fold, variant_id, seed, args.cuda, data_dir)
                    except Exception as e:
                        print(f'[ERROR] {e}')
                        import traceback
                        traceback.print_exc()

    print('Done. Check wandb project GSAT-Mutagenicity for best/clf_roc_* and motif_edge_att/*.')


if __name__ == '__main__':
    main()
