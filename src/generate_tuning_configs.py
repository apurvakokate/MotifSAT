#!/usr/bin/env python3
"""
Generate configuration files for hyperparameter tuning experiments.

This script creates a grid of hyperparameter combinations to systematically
test the effect of loss coefficients and weight distribution parameters on
model and explainer performance.

Base configuration is loaded from configs/total.config.yml, which contains
all default values for the GSAT training pipeline.

Usage:
    python generate_tuning_configs.py --experiment baseline --output_dir configs/tuning
    python generate_tuning_configs.py --experiment baseline --base_config configs/total.config.yml
"""

import argparse
import itertools
import yaml
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy
import os


# Default path to the base configuration file
DEFAULT_BASE_CONFIG_PATH = Path(__file__).parent / 'configs' / 'total_config.yml'


def load_base_config(config_path=None):
    """
    Load the base configuration from total.config.yml.
    
    Args:
        config_path: Path to the base config file. If None, uses default path.
        
    Returns:
        Dictionary containing the GSAT_config section with all default values.
    """
    if config_path is None:
        config_path = DEFAULT_BASE_CONFIG_PATH
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Base config file not found: {config_path}\n"
            f"Please ensure total.config.yml exists in configs/ directory."
        )
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract GSAT_config section as the base for tuning
    gsat_config = full_config.get('GSAT_config', {})
    
    # Extract tunable parameters with their defaults
    base_tuning_config = {
        # Loss coefficients
        'pred_loss_coef': gsat_config.get('pred_loss_coef', 1.0),
        'info_loss_coef': gsat_config.get('info_loss_coef', 1.0),
        'motif_loss_coef': gsat_config.get('motif_loss_coef', 2.0),
        
        # Weight distribution parameters (r parameter)
        'init_r': gsat_config.get('init_r', 0.9),
        'final_r': gsat_config.get('final_r', 0.5),
        'decay_r': gsat_config.get('decay_r', 0.1),
        'decay_interval': gsat_config.get('decay_interval', 10),
        
        # Training parameters (can also be tuned)
        'epochs': gsat_config.get('epochs', 100),
        'lr': gsat_config.get('lr', 1.0e-3),
        'weight_decay': gsat_config.get('weight_decay', 0),
        
        # Optional parameters
        'fix_r': gsat_config.get('fix_r', False),
        'from_scratch': gsat_config.get('from_scratch', True),
    }
    
    print(f"[INFO] Loaded base config from: {config_path}")
    print(f"[INFO] Base GSAT config values:")
    for key, value in base_tuning_config.items():
        print(f"       {key}: {value}")
    
    return base_tuning_config


def generate_config_grid(base_config, param_grid, experiment_overrides=None):
    """
    Generate all combinations of parameters from the grid.
    
    Args:
        base_config: Base configuration dictionary (from total.config.yml)
        param_grid: Dictionary of parameter names to lists of values to tune
        experiment_overrides: Optional dict of fixed overrides for this experiment
                             (applied before param_grid variations)
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Start with base config
    working_base = deepcopy(base_config)
    
    # Apply experiment-specific overrides (fixed values for this experiment type)
    if experiment_overrides:
        working_base.update(experiment_overrides)
    
    # Get all parameter names and their values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # Generate all combinations
    for idx, values in enumerate(itertools.product(*param_values)):
        config = deepcopy(working_base)
        config['tuning_id'] = f'config_{idx:04d}'
        
        # Update with current parameter values from grid
        param_dict = dict(zip(param_names, values))
        config.update(param_dict)
        
        configs.append(config)
    
    return configs


def create_method_comparison_experiment(base_config):
    """
    Compare all motif incorporation methods.
    
    Research Question: Which motif incorporation method works best?
    
    Methods:
        - None: Baseline GSAT (no motif information)
        - 'loss': Motif consistency loss (node-level attention with variance loss)
        - 'readout': Motif-level readout (pool embeddings, score motifs, broadcast)
        - 'graph': Motif-level graph (construct coarsened graph, run GNN on it)
          - With shared vs separate GNN for motif processing
          - With/without auxiliary motif graph training
    
    Uses defaults from total.config.yml for other parameters.
    """
    configs = []
    config_idx = 0
    
    # Method 1: None (baseline GSAT - no motif incorporation)
    config_none = deepcopy(base_config)
    config_none['tuning_id'] = f'config_{config_idx:04d}'
    config_none['motif_incorporation_method'] = None
    config_none['train_motif_graph'] = False
    config_none['separate_motif_model'] = False
    config_none['motif_loss_coef'] = 0.0  # No motif loss for baseline
    configs.append(config_none)
    config_idx += 1
    
    # Method 2: 'loss' (motif consistency loss)
    config_loss = deepcopy(base_config)
    config_loss['tuning_id'] = f'config_{config_idx:04d}'
    config_loss['motif_incorporation_method'] = 'loss'
    config_loss['train_motif_graph'] = False
    config_loss['separate_motif_model'] = False
    # Uses motif_loss_coef from base_config
    configs.append(config_loss)
    config_idx += 1
    
    # Method 3: 'readout' (motif-level attention readout)
    config_readout = deepcopy(base_config)
    config_readout['tuning_id'] = f'config_{config_idx:04d}'
    config_readout['motif_incorporation_method'] = 'readout'
    config_readout['train_motif_graph'] = False
    config_readout['separate_motif_model'] = False
    config_readout['motif_loss_coef'] = 0.0  # Not applicable for readout
    configs.append(config_readout)
    config_idx += 1
    
    # Method 4: 'graph' with SHARED model (no auxiliary training)
    config_graph_shared = deepcopy(base_config)
    config_graph_shared['tuning_id'] = f'config_{config_idx:04d}'
    config_graph_shared['motif_incorporation_method'] = 'graph'
    config_graph_shared['train_motif_graph'] = False
    config_graph_shared['separate_motif_model'] = False  # Shared parameters
    config_graph_shared['motif_loss_coef'] = 0.0  # No auxiliary loss
    configs.append(config_graph_shared)
    config_idx += 1
    
    # Method 5: 'graph' with SHARED model + auxiliary training
    config_graph_shared_train = deepcopy(base_config)
    config_graph_shared_train['tuning_id'] = f'config_{config_idx:04d}'
    config_graph_shared_train['motif_incorporation_method'] = 'graph'
    config_graph_shared_train['train_motif_graph'] = True
    config_graph_shared_train['separate_motif_model'] = False  # Shared parameters
    # Uses motif_loss_coef from base_config for auxiliary loss weight
    configs.append(config_graph_shared_train)
    config_idx += 1
    
    # Method 6: 'graph' with SEPARATE model (no auxiliary training)
    config_graph_separate = deepcopy(base_config)
    config_graph_separate['tuning_id'] = f'config_{config_idx:04d}'
    config_graph_separate['motif_incorporation_method'] = 'graph'
    config_graph_separate['train_motif_graph'] = False
    config_graph_separate['separate_motif_model'] = True  # Separate GNN for motif graph
    config_graph_separate['motif_loss_coef'] = 0.0  # No auxiliary loss
    configs.append(config_graph_separate)
    config_idx += 1
    
    # Method 7: 'graph' with SEPARATE model + auxiliary training
    config_graph_separate_train = deepcopy(base_config)
    config_graph_separate_train['tuning_id'] = f'config_{config_idx:04d}'
    config_graph_separate_train['motif_incorporation_method'] = 'graph'
    config_graph_separate_train['train_motif_graph'] = True
    config_graph_separate_train['separate_motif_model'] = True  # Separate GNN for motif graph
    # Uses motif_loss_coef from base_config for auxiliary loss weight
    configs.append(config_graph_separate_train)
    config_idx += 1
    
    return configs


def create_loss_coefficient_tuning(base_config):
    """
    Tune loss coefficients while keeping weight distribution fixed.
    
    Research Question: What are the optimal loss coefficient ratios?
    
    Uses weight distribution parameters from total.config.yml.
    """
    # No overrides - weight distribution comes from base config
    experiment_overrides = {}
    
    param_grid = {
        'pred_loss_coef': [0.5, 1.0, 2.0],
        'info_loss_coef': [0.5, 1.0, 2.0],
        'motif_loss_coef': [0.0, 0.5, 1.0, 2.0, 5.0],
    }
    
    return generate_config_grid(base_config, param_grid, experiment_overrides)


def create_weight_distribution_tuning(base_config):
    """
    Tune weight distribution parameters while keeping loss coefficients fixed.
    
    Research Question: How do weight distribution parameters affect polarization?
    
    Uses loss coefficients from total.config.yml.
    """
    # No overrides - loss coefficients come from base config
    experiment_overrides = {}
    
    param_grid = {
        'init_r': [0.7, 0.9],  # Starting distribution
        'final_r': [0.1, 0.3, 0.5],  # Final distribution
        'decay_r': [0.05, 0.1, 0.2],  # Decay rate
        'decay_interval': [5, 10, 15],  # Decay frequency
    }
    
    return generate_config_grid(base_config, param_grid, experiment_overrides)


def create_combined_tuning(base_config):
    """
    Fine-grained tuning of both loss coefficients and weight parameters.
    
    Research Question: What is the optimal combination of all hyperparameters?
    
    Tunes all major hyperparameters simultaneously.
    """
    # No overrides - tune everything
    experiment_overrides = {}
    
    param_grid = {
        'pred_loss_coef': [0.5, 1.0],
        'info_loss_coef': [0.5, 1.0],
        'motif_loss_coef': [0.0, 1.0, 2.0],
        'init_r': [0.8, 0.9],
        'final_r': [0.1, 0.3],
        'decay_r': [0.1],
        'decay_interval': [10],
    }
    
    return generate_config_grid(base_config, param_grid, experiment_overrides)


def create_motif_loss_sensitivity(base_config):
    """
    Fine-grained sensitivity analysis of motif loss coefficient.
    
    Research Question: How sensitive is performance to motif_loss_coef?
    
    Uses all other parameters from total.config.yml.
    """
    # No overrides - use base config for everything except motif_loss_coef
    experiment_overrides = {}
    
    param_grid = {
        'motif_loss_coef': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    }
    
    return generate_config_grid(base_config, param_grid, experiment_overrides)


def create_learning_rate_tuning(base_config):
    """
    Tune learning rate and related training parameters.
    
    Research Question: What is the optimal learning rate for GSAT training?
    
    Uses loss coefficients and weight distribution from total.config.yml.
    """
    experiment_overrides = {}
    
    param_grid = {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3],
        'weight_decay': [0, 1e-5, 1e-4],
    }
    
    return generate_config_grid(base_config, param_grid, experiment_overrides)


EXPERIMENT_TYPES = {
    'method_comparison': {
        'description': 'Compare all motif incorporation methods with shared/separate models (7 configs)',
        'generator': create_method_comparison_experiment,
    },
    'loss_tuning': {
        'description': 'Tune loss coefficients (3×3×5 = 45 configs)',
        'generator': create_loss_coefficient_tuning,
    },
    'weight_tuning': {
        'description': 'Tune weight distribution parameters (2×3×3×3 = 54 configs)',
        'generator': create_weight_distribution_tuning,
    },
    'combined': {
        'description': 'Combined tuning of all parameters (2×2×3×2×2×1×1 = 48 configs)',
        'generator': create_combined_tuning,
    },
    'sensitivity': {
        'description': 'Sensitivity analysis of motif_loss_coef (11 configs)',
        'generator': create_motif_loss_sensitivity,
    },
    'lr_tuning': {
        'description': 'Tune learning rate and weight decay (4×3 = 12 configs)',
        'generator': create_learning_rate_tuning,
    },
}


def save_configs(configs, output_dir, experiment_name, base_config_path=None):
    """
    Save configurations to files and create a manifest.
    
    Args:
        configs: List of configuration dictionaries
        output_dir: Directory to save configs
        experiment_name: Name of the experiment
        base_config_path: Path to the base config file used (for documentation)
    """
    output_dir = Path(output_dir)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'experiment_name': experiment_name,
        'num_configs': len(configs),
        'created_at': datetime.now().isoformat(),
        'base_config': str(base_config_path or DEFAULT_BASE_CONFIG_PATH),
        'configs': []
    }
    
    # Save individual config files
    for config in configs:
        # Add experiment name to config
        config['experiment_name'] = experiment_name
        
        config_id = config['tuning_id']
        config_file = experiment_dir / f'{config_id}.yaml'
        
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        # Store path relative to src/ directory (where scripts will be run from)
        # If output_dir is configs/tuning, then config_file is configs/tuning/experiment/config.yaml
        relative_path = config_file

        manifest['configs'].append({
            'config_id': config_id,
            'file': str(relative_path),
            'params': config
        })
            
    # Save manifest
    manifest_file = experiment_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save a summary table (CSV format)
    summary_file = experiment_dir / 'summary.csv'
    with open(summary_file, 'w') as f:
        # Header
        keys = configs[0].keys()
        f.write(','.join(keys) + '\n')
        
        # Data rows
        for config in configs:
            values = [str(config[k]) for k in keys]
            f.write(','.join(values) + '\n')
    
    print(f"Generated {len(configs)} configurations in {experiment_dir}")
    print(f"  - Config files: {config_id}.yaml")
    print(f"  - Manifest: {manifest_file}")
    print(f"  - Summary: {summary_file}")
    
    return manifest


def create_run_scripts(manifest, output_dir, datasets, models, folds, seeds):
    """
    Create shell scripts to run all experiments - parallelized by dataset AND fold.
    Scripts will be placed in src/ directory for easy execution from HPC.
    
    Creates one script per (dataset, fold) combination for maximum parallelization.
    
    Args:
        manifest: Experiment manifest
        output_dir: Directory containing configs
        datasets: List of dataset names
        models: List of model names
        folds: List of fold numbers
        seeds: List of random seeds
    """
    experiment_name = manifest['experiment_name']
    experiment_dir = Path(output_dir) / experiment_name
    src_dir = Path('.')  # Current directory (src/) for run scripts
    scripts_dir = src_dir / 'run_scripts'
    scripts_dir.mkdir(exist_ok=True)
    
    created_scripts = []
    
    # Create individual scripts for each (dataset, fold) combination
    for dataset in datasets:
        for fold in folds:
            script_name = f'run_{dataset}_fold{fold}.sh'
            script_path = scripts_dir / script_name
            
            with open(script_path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'# Run experiments for {dataset} - Fold {fold}\n')
                f.write(f'# Experiment: {experiment_name}\n')
                f.write(f'# Organization: Architecture -> Config -> Seed\n')
                f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
                
                for model in models:
                    f.write(f'\n{"-"*80}\n')
                    f.write(f'# Architecture: {model}\n')
                    f.write(f'{"-"*80}\n\n')
                    
                    for config_info in manifest['configs']:
                        config_id = config_info['config_id']
                        config_file = config_info['file']
                        params = config_info['params']
                        
                        f.write(f'\n# Config: {config_id} | Params: {params}\n')
                        
                        for seed in seeds:
                            f.write(f'# Seed {seed}\n')
                            f.write('python run_gsat.py \\\n')
                            f.write(f'  --dataset {dataset} \\\n')
                            f.write(f'  --backbone {model} \\\n')
                            f.write(f'  --fold {fold} \\\n')
                            f.write(f'  --seed {seed} \\\n')
                            f.write(f'  --cuda 0 \\\n')
                            f.write(f'  --config {config_file}\n\n')
            
            script_path.chmod(0o755)
            created_scripts.append(script_name)
            print(f"Created: {script_path}")
    
    # Create per-dataset master scripts (runs all folds for a dataset sequentially)
    for dataset in datasets:
        dataset_script = scripts_dir / f'run_{dataset}_all_folds.sh'
        with open(dataset_script, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Run all folds for {dataset}\n')
            f.write(f'# Experiment: {experiment_name}\n')
            f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
            
            for fold in folds:
                f.write(f'echo "Starting {dataset} - Fold {fold}"\n')
                f.write(f'bash run_scripts/run_{dataset}_fold{fold}.sh\n\n')
        
        dataset_script.chmod(0o755)
        print(f"Created: {dataset_script}")
    
    # Create master script that runs everything
    master_script = src_dir / 'run_all.sh'
    with open(master_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Master script to run all tuning experiments\n')
        f.write(f'# Experiment: {experiment_name}\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
        f.write(f'# Total scripts: {len(created_scripts)} (one per dataset-fold combination)\n')
        f.write(f'# Datasets: {", ".join(datasets)}\n')
        f.write(f'# Folds: {", ".join(map(str, folds))}\n\n')
        
        for dataset in datasets:
            for fold in folds:
                f.write(f'echo "Starting {dataset} - Fold {fold}"\n')
                f.write(f'bash run_scripts/run_{dataset}_fold{fold}.sh\n\n')
    
    master_script.chmod(0o755)
    
    print(f"\n{'='*60}")
    print(f"Created {len(created_scripts)} individual scripts in {scripts_dir}")
    print(f"Run individual (dataset, fold) pairs in parallel:")
    print(f"  bash run_scripts/run_<dataset>_fold<N>.sh")
    print(f"Or run all folds for a dataset:")
    print(f"  bash run_scripts/run_<dataset>_all_folds.sh")
    print(f"Or run everything sequentially:")
    print(f"  bash run_all.sh")
    print(f"{'='*60}")


def create_slurm_scripts(manifest, output_dir, datasets, models, folds, seeds, base_path=None):
    """
    Create SLURM batch scripts for HPC cluster - one job per (dataset, fold) for parallel execution.
    
    Creates one script per (dataset, fold) combination for maximum parallelization.
    
    Args:
        manifest: Experiment manifest
        output_dir: Directory containing configs
        datasets: List of dataset names
        models: List of model names
        folds: List of fold numbers
        seeds: List of random seeds
        base_path: Absolute base path for HPC (e.g., /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifSAT)
                   If None, uses relative paths
    """
    experiment_name = manifest['experiment_name']
    experiment_dir = Path(output_dir) / experiment_name
    slurm_dir = experiment_dir / 'slurm_scripts'
    slurm_dir.mkdir(exist_ok=True)
    
    # Default base path if not provided
    if base_path is None:
        base_path = '/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifSAT'
    
    created_scripts = []
    
    # Create one SLURM script per (dataset, fold) combination
    for dataset in datasets:
        for fold in folds:
            job_name = f'{experiment_name}_{dataset}_fold{fold}'
            script_path = slurm_dir / f'{job_name}.sh'
            
            with open(script_path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'#SBATCH --job-name={job_name}\n')
                f.write(f'#SBATCH --output=logs/{job_name}_%j.out\n')
                f.write(f'#SBATCH --error=logs/{job_name}_%j.err\n')
                f.write('#SBATCH --time=1-00:00:00\n')  # Shorter time for single fold
                f.write('#SBATCH --partition=gpu\n')
                f.write('#SBATCH --gres=gpu:1\n')
                f.write('#SBATCH --cpus-per-task=8\n')
                f.write('#SBATCH --mem=64G\n\n')
                
                f.write('# Initialize Conda\n')
                f.write('source ~/hpc-share/anaconda3/etc/profile.d/conda.sh\n\n')
                
                f.write('# Activate the desired environment\n')
                f.write('conda activate l2xgnn\n\n')
                
                f.write('# Set absolute paths for HPC\n')
                f.write(f'export BASE_PATH="{base_path}"\n')
                f.write('export RESULTS_DIR="${BASE_PATH}/tuning_results"\n')
                f.write('export WANDB_DIR="${BASE_PATH}/wandb"\n\n')
                
                f.write('# Navigate to src directory\n')
                f.write(f'cd {base_path}/src\n\n')
                
                f.write(f'# Dataset: {dataset}, Fold: {fold}\n')
                f.write('# Organization: Architecture -> Config -> Seed\n\n')
                
                for model in models:
                    f.write(f'\n{"-"*80}\n')
                    f.write(f'# Architecture: {model}\n')
                    f.write(f'{"-"*80}\n\n')
                    
                    for config_info in manifest['configs']:
                        config_id = config_info['config_id']
                        config_file = config_info['file']
                        params = config_info['params']
                        
                        f.write(f'\n# Config: {config_id} | Params: {params}\n')
                        
                        for seed in seeds:
                            f.write(f'echo "Running {dataset} - fold {fold} - {model} - {config_id} - seed {seed}"\n')
                            f.write('python run_gsat.py \\\n')
                            f.write(f'  --dataset {dataset} \\\n')
                            f.write(f'  --backbone {model} \\\n')
                            f.write(f'  --fold {fold} \\\n')
                            f.write(f'  --seed {seed} \\\n')
                            f.write(f'  --cuda 0 \\\n')
                            f.write(f'  --config {config_file}\n\n')
            
            script_path.chmod(0o755)
            created_scripts.append(job_name)
    
    # Create per-dataset submission scripts (submits all folds for a dataset)
    for dataset in datasets:
        submit_dataset = slurm_dir / f'submit_{dataset}.sh'
        with open(submit_dataset, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Submit all folds for {dataset}\n\n')
            
            for fold in folds:
                job_name = f'{experiment_name}_{dataset}_fold{fold}'
                f.write(f'echo "Submitting job for {dataset} - Fold {fold}"\n')
                f.write(f'sbatch {slurm_dir}/{job_name}.sh\n')
                f.write('sleep 1\n\n')
        
        submit_dataset.chmod(0o755)
    
    # Create master submission script
    submit_all = slurm_dir / 'submit_all.sh'
    with open(submit_all, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'# Submit all SLURM jobs (one per dataset-fold combination)\n')
        f.write(f'# Total jobs: {len(created_scripts)}\n')
        f.write(f'# Datasets: {", ".join(datasets)}\n')
        f.write(f'# Folds: {", ".join(map(str, folds))}\n\n')
        
        for dataset in datasets:
            f.write(f'\n# {dataset}\n')
            for fold in folds:
                job_name = f'{experiment_name}_{dataset}_fold{fold}'
                f.write(f'echo "Submitting {dataset} - Fold {fold}"\n')
                f.write(f'sbatch {slurm_dir}/{job_name}.sh\n')
                f.write('sleep 1\n')
    
    submit_all.chmod(0o755)
    
    print(f"\n{'='*60}")
    print(f"Created {len(created_scripts)} SLURM scripts in {slurm_dir}")
    print(f"Submit individual (dataset, fold) jobs:")
    print(f"  sbatch {slurm_dir}/{experiment_name}_<dataset>_fold<N>.sh")
    print(f"Submit all folds for a dataset:")
    print(f"  bash {slurm_dir}/submit_<dataset>.sh")
    print(f"Submit all jobs:")
    print(f"  bash {submit_all}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate tuning configuration files for GSAT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiment types:
""" + '\n'.join(f"  {name:15s} - {info['description']}" 
                for name, info in EXPERIMENT_TYPES.items()) + """

Base Configuration:
  All experiments use configs/total.config.yml as the base configuration.
  This ensures consistent defaults across all tuning experiments.
  You can override the base config path with --base_config.
"""
    )
    
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENT_TYPES.keys()),
                        help='Type of experiment to generate')
    
    parser.add_argument('--base_config', type=str, default=None,
                        help='Path to base configuration file (default: configs/total.config.yml)')
    
    parser.add_argument('--output_dir', type=str, default='configs/tuning',
                        help='Directory to save configuration files')
    
    parser.add_argument('--datasets', nargs='+', 
                        default=['Mutagenicity', 'hERG', 'BBBP', 'Benzene'],
                        help='List of datasets to run experiments on')
    
    parser.add_argument('--models', nargs='+',
                        default=['GIN', 'GCN', 'GAT', 'SAGE'],
                        help='List of model architectures to test')
    
    parser.add_argument('--folds', nargs='+', type=int,
                        default=[0, 1, 2, 3, 4],
                        help='List of fold numbers for cross-validation')
    
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[0],
                        help='List of random seeds for multiple runs')
    
    parser.add_argument('--create_run_scripts', action='store_true',
                        help='Create shell scripts to run all experiments')
    
    parser.add_argument('--create_slurm_scripts', action='store_true',
                        help='Create SLURM batch scripts for HPC')
    
    parser.add_argument('--hpc_base_path', type=str,
                        default='/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifSAT',
                        help='Absolute base path for HPC cluster (for SLURM scripts)')
    
    args = parser.parse_args()
    
    # Load base configuration from total.config.yml
    print("\n" + "="*80)
    print("LOADING BASE CONFIGURATION")
    print("="*80)
    base_config = load_base_config(args.base_config)
    
    # Generate configurations
    print("\n" + "="*80)
    print("GENERATING EXPERIMENT CONFIGURATIONS")
    print("="*80)
    print(f"Experiment type: {args.experiment}")
    exp_info = EXPERIMENT_TYPES[args.experiment]
    print(f"Description: {exp_info['description']}")
    
    # Pass base_config to the generator function
    configs = exp_info['generator'](base_config)
    
    # Save configurations
    manifest = save_configs(configs, args.output_dir, args.experiment)
    
    # Also save a copy of the base config used for reference
    base_config_copy_path = Path(args.output_dir) / args.experiment / 'base_config_used.yaml'
    with open(base_config_copy_path, 'w') as f:
        yaml.safe_dump(base_config, f, sort_keys=False)
    print(f"  - Base config reference: {base_config_copy_path}")
    
    # Create run scripts if requested
    if args.create_run_scripts:
        create_run_scripts(manifest, args.output_dir, args.datasets, 
                          args.models, args.folds, args.seeds)
    
    if args.create_slurm_scripts:
        create_slurm_scripts(manifest, args.output_dir, args.datasets,
                           args.models, args.folds, args.seeds, 
                           base_path=args.hpc_base_path)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Base config: {args.base_config or str(DEFAULT_BASE_CONFIG_PATH)}")
    print(f"Configurations: {len(configs)}")
    print(f"Datasets: {len(args.datasets)} - {', '.join(args.datasets)}")
    print(f"Models: {len(args.models)} - {', '.join(args.models)}")
    print(f"Folds: {len(args.folds)} - {', '.join(map(str, args.folds))}")
    print(f"Seeds: {len(args.seeds)} - {', '.join(map(str, args.seeds))}")
    print(f"\nTotal experiments: {len(configs) * len(args.datasets) * len(args.models) * len(args.folds) * len(args.seeds)}")
    print("="*80)


if __name__ == '__main__':
    main()

