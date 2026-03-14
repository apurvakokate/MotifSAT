#!/usr/bin/env python
"""
Run GSAT on molecular datasets with structured experiment groups.

Experiment groups:
  1. vanilla_gnn: No attention weighting (true vanilla GNN baseline)
  2. base_gsat_fix_r: GSAT with fixed r values {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}
  3. base_gsat_decay_r: GSAT with decay schedule, final_r in {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}
  4. motif_readout_fix_r: Motif-level readout with fixed r {1.0, 0.9, 0.8, 0.7}
     (only for datasets with motif info: Mutagenicity, Benzene, etc.)

  Legacy groups (kept for reference):
  5. r_impact_node: Base GSAT node attention, varying final_r
  6. r_impact_edge: Base GSAT edge attention, varying final_r
  7. within_motif_consistency_impact: Motif consistency loss ablation
  8. between_motif_consistency_impact: Between-motif consistency loss ablation
  9. motif_readout_info_loss: Motif readout with node-level perturbation
  10. motif_readout_adaptive_r: Motif readout with graph-adaptive r
  11. att_injection_point: W_FEAT / W_MESSAGE / W_READOUT ablation

Usage:
  python run_mutagenicity_gsat_experiment.py --experiments vanilla_gnn base_gsat_fix_r
  python run_mutagenicity_gsat_experiment.py --dataset ogbg_molhiv --experiments vanilla_gnn base_gsat_fix_r
  python run_mutagenicity_gsat_experiment.py  # runs all groups
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

SUPPORTED_DATASETS = [
    'Mutagenicity', 'BBBP', 'hERG', 'Benzene',
    'Alkane_Carbonyl', 'Fluoride_Carbonyl',
    'ogbg_molhiv',
]
DATASETS_WITH_MOTIFS = ['Mutagenicity', 'BBBP', 'hERG', 'Benzene', 'Alkane_Carbonyl', 'Fluoride_Carbonyl']
DATASET = 'Mutagenicity'
MOTIF_SCORES_TEMPLATE = '/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/All0.5_learn_unk+motif_scores/{dataset}_{model}_motif_scores.csv'

# ---------------------------------------------------------------------------
# Experiment group definitions
# ---------------------------------------------------------------------------

EXPERIMENT_GROUPS = {

    # =========================================================================
    # NEW: Incremental R-value sweep experiments
    # =========================================================================

    'vanilla_gnn_node_repaired': {
        'experiment_name': 'vanilla_gnn_node_repaired',
        'variants': [
            {
                'variant_id': 'no_attention',
                'gsat_overrides': {
                    'tuning_id': 'no_attention',
                    'no_attention': True,
                    'fix_r': 1.0,
                    'info_loss_coef': 0,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            },
        ],
    },

    'vanilla_gnn_clean': {
        'experiment_name': 'vanilla_gnn_clean',
        'variants': [
            {
                'variant_id': 'clean',
                'gsat_overrides': {
                    'tuning_id': 'clean',
                    'fix_r': 1.0,
                },
                'learn_edge_att': False,
                'vanilla_clean': True,
            },
        ],
    },

    'base_gsat_fix_r_node_repaired': {
        'experiment_name': 'base_gsat_fix_r_node_repaired',
        'variants': [
            {
                'variant_id': f'fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },

    'base_gsat_decay_r_node_repaired': {
        'experiment_name': 'base_gsat_decay_r_node_repaired',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        ],
    },

    'motif_readout_fix_r_repaired': {
        'experiment_name': 'motif_readout_fix_r_repaired',
        'variants': [
            {
                'variant_id': f'readout_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [1.0, 0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_fix_r_mean': {
        'experiment_name': 'motif_readout_fix_r_mean',
        'variants': [
            {
                'variant_id': f'readout_mean_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_fix_r_sum': {
        'experiment_name': 'motif_readout_fix_r_sum',
        'variants': [
            {
                'variant_id': f'readout_sum_fix_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'readout_sum_fix_r{r}',
                    'fix_r': r,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'sum',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean': {
        'experiment_name': 'motif_readout_decay_r_mean',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.9, 0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_sum': {
        'experiment_name': 'motif_readout_decay_r_sum',
        'variants': [
            {
                'variant_id': f'readout_sum_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_sum_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'sum',
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.9, 0.8, 0.7]
        ],
    },

    # =========================================================================
    # Explainer analysis experiments (score-vs-impact with r highlight)
    # New names to avoid overwriting previous results.
    # =========================================================================

    'base_gsat_decay_r_explainer': {
        'experiment_name': 'base_gsat_decay_r_explainer',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_explainer': {
        'experiment_name': 'motif_readout_decay_r_mean_explainer',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': False,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_sampling_explainer': {
        'experiment_name': 'motif_readout_decay_r_mean_sampling_explainer',
        'variants': [
            {
                'variant_id': f'readout_mean_sampling_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_sampling_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Motif-level info loss variants (compare with the originals above)
    # Same as base/readout/sampling explainer but with motif_level_info_loss=True
    # =========================================================================

    'base_gsat_decay_r_explainer_motif_info': {
        'experiment_name': 'base_gsat_decay_r_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_explainer_motif_info': {
        'experiment_name': 'motif_readout_decay_r_mean_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'readout_mean_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': False,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    'motif_readout_decay_r_mean_sampling_explainer_motif_info': {
        'experiment_name': 'motif_readout_decay_r_mean_sampling_explainer_motif_info',
        'variants': [
            {
                'variant_id': f'readout_mean_sampling_decay_final{fr}',
                'gsat_overrides': {
                    'tuning_id': f'readout_mean_sampling_decay_final{fr}',
                    'fix_r': False,
                    'init_r': 0.9,
                    'final_r': fr,
                    'decay_r': 0.1,
                    'motif_incorporation_method': 'readout',
                    'motif_pooling_method': 'mean',
                    'motif_loss_coef': 0,
                    'motif_level_sampling': True,
                    'motif_level_info_loss': True,
                },
                'learn_edge_att': False,
            }
            for fr in [0.8, 0.7]
        ],
    },

    # =========================================================================
    # Legacy experiments (kept for reference, still functional)
    # =========================================================================

    'r_impact_node': {
        'experiment_name': 'r_impact_node',
        'variants': [
            {
                'variant_id': f'node_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'node_r{r}',
                    'final_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': False,
            }
            for r in [0.4, 0.5, 0.6]
        ],
    },

    'r_impact_edge': {
        'experiment_name': 'r_impact_edge',
        'variants': [
            {
                'variant_id': f'edge_r{r}',
                'gsat_overrides': {
                    'tuning_id': f'edge_r{r}',
                    'final_r': r,
                    'motif_incorporation_method': None,
                    'motif_loss_coef': 0,
                    'between_motif_coef': 0,
                },
                'learn_edge_att': True,
            }
            for r in [0.4, 0.5, 0.6]
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
    config = get_base_config(model_name, dataset_name, gsat_overrides=variant['gsat_overrides'])
    config['shared_config']['learn_edge_att'] = variant['learn_edge_att']
    config['GSAT_config']['experiment_name'] = experiment_name

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

    if variant.get('vanilla_clean', False):
        from run_gsat import train_vanilla_gnn_one_seed
        hparam_dict, metric_dict = train_vanilla_gnn_one_seed(
            config, data_dir, log_dir, model_name, dataset_name,
            device, seed, fold=fold, task_type='classification'
        )
    else:
        from run_gsat import train_gsat_one_seed
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
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='Folds to run (default: [0,1] for MolDatasets, [0] for OGB)')
    parser.add_argument('--models', type=str, nargs='+', default=ARCHITECTURES, help='Models (backbones)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id (-1 for CPU)')
    args = parser.parse_args()
    
    dataset_name = args.dataset

    # Default folds: [0,1] for MolDatasets with folds, [0] for OGB (no fold splitting)
    if args.folds is not None:
        folds = args.folds
    elif dataset_name in DATASETS_WITH_MOTIFS:
        folds = [0, 1]
    else:
        folds = [0]

    # Warn if motif-requiring experiments are selected for OGB datasets
    motif_experiments = {'motif_readout_fix_r', 'motif_readout_fix_r_repaired',
                         'motif_readout_fix_r_mean', 'motif_readout_fix_r_sum',
                         'motif_readout_decay_r_mean', 'motif_readout_decay_r_sum',
                         'motif_readout_decay_r_mean_explainer',
                         'motif_readout_decay_r_mean_sampling_explainer',
                         'motif_readout_info_loss', 'motif_readout_adaptive_r',
                         'within_motif_consistency_impact', 'between_motif_consistency_impact'}
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
                            run_one(model_name, fold, variant, experiment_name, seed, args.cuda, data_dir, dataset_name)
                        except Exception as e:
                            print(f'[ERROR] {e}')
                            import traceback
                            traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
