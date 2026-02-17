#!/usr/bin/env python
"""
GSAT Replication Experiment

This script runs the baseline GSAT (no motif modifications) across multiple
architectures (GIN, PNA, GAT, SAGE, GCN) and paper datasets to verify the
core GSAT workflow is compatible with different architectures.

Paper datasets tested:
- BA-2Motifs
- Mutag
- MNIST-75sp
- Spurious-Motif (b=0.5, 0.7, 0.9)
- Graph-SST2
- OGBG-Molhiv

Based on: "Interpretable and Generalizable Graph Learning via Stochastic
Attention Mechanism" (ICML 2022, Miao et al.)
"""

import yaml
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import argparse
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Writer, set_seed, get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, init_metric_dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy

from experiment_configs import PAPER_DATASETS, ARCHITECTURES, get_base_config


def run_single_experiment(model_name, dataset_name, seed, cuda_id, results_dir):
    """
    Run a single GSAT experiment with baseline configuration.
    
    Returns:
        dict: Metric dictionary with results
    """
    from run_gsat import train_gsat_one_seed, ExtractorMLP
    
    print('=' * 80)
    print(f'Running: {model_name} on {dataset_name} (seed={seed})')
    print('=' * 80)
    
    # Get configuration
    config = get_base_config(model_name, dataset_name)
    
    # Setup paths
    config_dir = Path('./configs')
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    data_dir = Path(global_config['data_dir'])
    
    # Setup device
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    
    # Log directory
    time_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    log_dir = data_dir / dataset_name / 'logs' / f'replication-{time_str}-{model_name}-seed{seed}'
    
    try:
        # Run training
        hparam_dict, metric_dict = train_gsat_one_seed(
            config, data_dir, log_dir, model_name, dataset_name,
            'GSAT', device, seed, fold=0, task_type='classification'
        )
        
        # Add experiment info
        metric_dict['model'] = model_name
        metric_dict['dataset'] = dataset_name
        metric_dict['seed'] = seed
        metric_dict['status'] = 'success'
        
    except Exception as e:
        print(f'[ERROR] Experiment failed: {e}')
        import traceback
        traceback.print_exc()
        metric_dict = {
            'model': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'status': 'failed',
            'error': str(e),
        }
    
    return metric_dict


def run_replication_experiments(
    datasets=None,
    models=None,
    seeds=None,
    cuda_id=0,
    results_dir='../replication_results',
):
    """
    Run full replication experiment across all model-dataset combinations.
    """
    datasets = datasets or PAPER_DATASETS
    models = models or ARCHITECTURES
    seeds = seeds or [0, 1, 2]  # 3 seeds for statistical significance
    
    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'gsat_replication_{timestamp}'
    
    # Store all results
    all_results = []
    summary = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'datasets': datasets,
        'models': models,
        'seeds': seeds,
        'results': {},
    }
    
    total_experiments = len(datasets) * len(models) * len(seeds)
    current_exp = 0
    
    for dataset in datasets:
        summary['results'][dataset] = {}
        
        for model in models:
            summary['results'][dataset][model] = {
                'seeds': {},
                'mean': {},
                'std': {},
            }
            
            seed_results = []
            
            for seed in seeds:
                current_exp += 1
                print(f'\n[{current_exp}/{total_experiments}] Running {model} on {dataset} (seed {seed})')
                
                result = run_single_experiment(
                    model, dataset, seed, cuda_id, results_dir
                )
                
                all_results.append(result)
                seed_results.append(result)
                summary['results'][dataset][model]['seeds'][seed] = result
                
                # Save intermediate results
                with open(results_dir / f'{experiment_name}_all_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
            
            # Compute statistics across seeds
            if all(r.get('status') == 'success' for r in seed_results):
                for metric in ['metric/best_clf_test', 'metric/best_x_roc_test']:
                    values = [r.get(metric, 0) for r in seed_results if r.get(metric) is not None]
                    if values:
                        summary['results'][dataset][model]['mean'][metric] = float(np.mean(values))
                        summary['results'][dataset][model]['std'][metric] = float(np.std(values))
    
    # Save final summary
    with open(results_dir / f'{experiment_name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print results table
    print_results_table(summary)
    
    return summary


def print_results_table(summary):
    """Print a formatted results table."""
    print('\n' + '=' * 100)
    print('GSAT REPLICATION RESULTS')
    print('=' * 100)
    
    datasets = summary['datasets']
    models = summary['models']
    
    # Print header
    header = f"{'Dataset':<20}" + "".join([f"{m:<15}" for m in models])
    print(header)
    print('-' * len(header))
    
    # Print results
    for dataset in datasets:
        row = f"{dataset:<20}"
        for model in models:
            try:
                mean = summary['results'][dataset][model]['mean'].get('metric/best_clf_test', 0)
                std = summary['results'][dataset][model]['std'].get('metric/best_clf_test', 0)
                if mean > 0:
                    row += f"{mean:.3f}±{std:.3f}  "
                else:
                    row += f"{'FAILED':<15}"
            except:
                row += f"{'N/A':<15}"
        print(row)
    
    print('=' * 100)
    
    # Print explanation metrics
    print('\nExplanation AUROC:')
    print('-' * len(header))
    for dataset in datasets:
        row = f"{dataset:<20}"
        for model in models:
            try:
                mean = summary['results'][dataset][model]['mean'].get('metric/best_x_roc_test', 0)
                std = summary['results'][dataset][model]['std'].get('metric/best_x_roc_test', 0)
                if mean > 0:
                    row += f"{mean:.3f}±{std:.3f}  "
                else:
                    row += f"{'FAILED':<15}"
            except:
                row += f"{'N/A':<15}"
        print(row)
    
    print('=' * 100)


def main():
    parser = argparse.ArgumentParser(description='GSAT Replication Experiment')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to test (default: all paper datasets)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to test (default: GIN, PNA, GAT, SAGE, GCN)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds to use (default: 0 1 2)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='../replication_results',
                        help='Directory to save results')
    parser.add_argument('--single', action='store_true',
                        help='Run a single experiment (requires --model and --dataset)')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model to run (with --single)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset to run (with --single)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Single seed to run (with --single)')
    
    args = parser.parse_args()
    
    torch.set_num_threads(5)
    
    if args.single:
        # Run single experiment
        if not args.model or not args.dataset:
            print('ERROR: --single requires --model and --dataset')
            return
        
        result = run_single_experiment(
            args.model, args.dataset, args.seed, args.cuda, args.results_dir
        )
        print(f'\nResult: {json.dumps(result, indent=2, default=str)}')
    else:
        # Run full replication
        run_replication_experiments(
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
            cuda_id=args.cuda,
            results_dir=args.results_dir,
        )


if __name__ == '__main__':
    main()
