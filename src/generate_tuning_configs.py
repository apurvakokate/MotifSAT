#!/usr/bin/env python3
"""
Generate configuration files for hyperparameter tuning experiments.

This script creates a grid of hyperparameter combinations to systematically
test the effect of loss coefficients and weight distribution parameters on
model and explainer performance.

Usage:
    python generate_tuning_configs.py --output_dir configs/tuning --experiment_name baseline
"""

import argparse
import itertools
import yaml
from pathlib import Path
import json
from datetime import datetime
from pathlib import Path
import os


def generate_config_grid(base_config, param_grid):
    """
    Generate all combinations of parameters from the grid.
    
    Args:
        base_config: Base configuration dictionary
        param_grid: Dictionary of parameter names to lists of values
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Get all parameter names and their values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # Generate all combinations
    for idx, values in enumerate(itertools.product(*param_values)):
        config = base_config.copy()
        config['tuning_id'] = f'config_{idx:04d}'
        
        # Update with current parameter values
        param_dict = dict(zip(param_names, values))
        config.update(param_dict)
        
        configs.append(config)
    
    return configs


def create_baseline_experiment():
    """
    Create baseline experiment: test with and without motif loss.
    
    Research Question: Does adding motif_consistency_loss improve performance?
    """
    base_config = {
        'pred_loss_coef': 1.0,
        'info_loss_coef': 1.0,
        'init_r': 0.9,
        'final_r': 0.1,
        'decay_r': 0.1,
        'decay_interval': 10,
    }
    
    param_grid = {
        'motif_loss_coef': [0.0, 0.5, 1.0, 2.0],  # Test without and with motif loss
    }
    
    return generate_config_grid(base_config, param_grid)


def create_loss_coefficient_tuning():
    """
    Tune loss coefficients while keeping weight distribution fixed.
    
    Research Question: What are the optimal loss coefficient ratios?
    """
    base_config = {
        'init_r': 0.9,
        'final_r': 0.1,
        'decay_r': 0.1,
        'decay_interval': 10,
    }
    
    param_grid = {
        'pred_loss_coef': [0.5, 1.0, 2.0],
        'info_loss_coef': [0.5, 1.0, 2.0],
        'motif_loss_coef': [0.0, 0.5, 1.0, 2.0, 5.0],
    }
    
    return generate_config_grid(base_config, param_grid)


def create_weight_distribution_tuning():
    """
    Tune weight distribution parameters while keeping loss coefficients fixed.
    
    Research Question: How do weight distribution parameters affect polarization?
    """
    base_config = {
        'pred_loss_coef': 1.0,
        'info_loss_coef': 1.0,
        'motif_loss_coef': 1.0,
    }
    
    param_grid = {
        'init_r': [0.7, 0.9],  # Starting distribution
        'final_r': [0.1, 0.3, 0.5],  # Final distribution
        'decay_r': [0.05, 0.1, 0.2],  # Decay rate
        'decay_interval': [5, 10, 15],  # Decay frequency
    }
    
    return generate_config_grid(base_config, param_grid)


def create_combined_tuning():
    """
    Fine-grained tuning of both loss coefficients and weight parameters.
    
    Research Question: What is the optimal combination of all hyperparameters?
    """
    base_config = {}
    
    param_grid = {
        'pred_loss_coef': [0.5, 1.0],
        'info_loss_coef': [0.5, 1.0],
        'motif_loss_coef': [0.0, 1.0, 2.0],
        'init_r': [0.8, 0.9],
        'final_r': [0.1, 0.3],
        'decay_r': [0.1],
        'decay_interval': [10],
    }
    
    return generate_config_grid(base_config, param_grid)


def create_motif_loss_sensitivity():
    """
    Fine-grained sensitivity analysis of motif loss coefficient.
    
    Research Question: How sensitive is performance to motif_loss_coef?
    """
    base_config = {
        'pred_loss_coef': 1.0,
        'info_loss_coef': 1.0,
        'init_r': 0.9,
        'final_r': 0.1,
        'decay_r': 0.1,
        'decay_interval': 10,
    }
    
    param_grid = {
        'motif_loss_coef': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    }
    
    return generate_config_grid(base_config, param_grid)


EXPERIMENT_TYPES = {
    'baseline': {
        'description': 'Test with and without motif loss (4 configs)',
        'generator': create_baseline_experiment,
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
}


def save_configs(configs, output_dir, experiment_name):
    """
    Save configurations to files and create a manifest.
    
    Args:
        configs: List of configuration dictionaries
        output_dir: Directory to save configs
        experiment_name: Name of the experiment
    """
    output_dir = Path(output_dir)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'experiment_name': experiment_name,
        'num_configs': len(configs),
        'created_at': datetime.now().isoformat(),
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
    Create shell scripts to run all experiments - parallelized by dataset.
    Scripts will be placed in src/ directory for easy execution from HPC.
    
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
    
    # Create individual scripts for each dataset (for parallel execution)
    # Prioritize: Fold -> Model (Architecture) -> Config -> Seed
    for dataset in datasets:
        dataset_script = src_dir / f'run_{dataset}.sh'
        
        with open(dataset_script, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Run all experiments for {dataset}\n')
            f.write(f'# Experiment: {experiment_name}\n')
            f.write(f'# Organization: Fold -> Architecture -> Config -> Seed\n')
            f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
            
            # Prioritize fold first, then model (architecture)
            for fold in folds:
                f.write(f'\n{"="*80}\n')
                f.write(f'# FOLD {fold}\n')
                f.write(f'{"="*80}\n\n')
                
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
        
        dataset_script.chmod(0o755)
        print(f"Created: {dataset_script}")
    
    # Create master script in src/ that runs all dataset scripts
    master_script = src_dir / 'run_all.sh'
    with open(master_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Master script to run all tuning experiments\n')
        f.write(f'# Experiment: {experiment_name}\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n\n')
        
        for dataset in datasets:
            f.write(f'echo "Starting experiments for {dataset}"\n')
            f.write(f'bash run_{dataset}.sh\n\n')
    
    master_script.chmod(0o755)
    print(f"\nCreated master script: {master_script}")
    print(f"Run individual datasets in parallel using: bash run_<dataset>.sh or sbatch")


def create_slurm_scripts(manifest, output_dir, datasets, models, folds, seeds, base_path=None):
    """
    Create SLURM batch scripts for HPC cluster - one job per dataset for parallel execution.
    
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
    
    # Create one SLURM script per dataset (for parallel job submission)
    for dataset in datasets:
        job_name = f'{experiment_name}_{dataset}'
        script_path = slurm_dir / f'{job_name}.sh'
        
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name={job_name}\n')
            f.write(f'#SBATCH --output=logs/{job_name}_%j.out\n')
            f.write(f'#SBATCH --error=logs/{job_name}_%j.err\n')
            f.write('#SBATCH --time=2-00:00:00\n')
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
            
            f.write('# Organization: Fold -> Architecture -> Config -> Seed\n\n')
            
            # Add all experiments for this dataset
            # Prioritize fold first, then model (architecture)
            for fold in folds:
                f.write(f'\n{"="*80}\n')
                f.write(f'# FOLD {fold}\n')
                f.write(f'{"="*80}\n\n')
                
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
    
    # Create submission script
    submit_all = slurm_dir / 'submit_all.sh'
    with open(submit_all, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Submit all SLURM jobs (one per dataset for parallel execution)\n\n')
        
        for dataset in datasets:
            job_name = f'{experiment_name}_{dataset}'
            f.write(f'echo "Submitting job for {dataset}"\n')
            f.write(f'sbatch slurm_scripts/{job_name}.sh\n')
            f.write('sleep 1\n\n')
    
    submit_all.chmod(0o755)
    print(f"\nCreated SLURM scripts in {slurm_dir}")
    print(f"Submit all jobs: bash {submit_all}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate tuning configuration files for GSAT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiment types:
""" + '\n'.join(f"  {name:15s} - {info['description']}" 
                for name, info in EXPERIMENT_TYPES.items())
    )
    
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENT_TYPES.keys()),
                        help='Type of experiment to generate')
    
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
    
    # Generate configurations
    print(f"\nGenerating {args.experiment} experiment configurations...")
    exp_info = EXPERIMENT_TYPES[args.experiment]
    print(f"Description: {exp_info['description']}")
    
    configs = exp_info['generator']()
    
    # Save configurations
    manifest = save_configs(configs, args.output_dir, args.experiment)
    
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
    print(f"Configurations: {len(configs)}")
    print(f"Datasets: {len(args.datasets)} - {', '.join(args.datasets)}")
    print(f"Models: {len(args.models)} - {', '.join(args.models)}")
    print(f"Folds: {len(args.folds)} - {', '.join(map(str, args.folds))}")
    print(f"Seeds: {len(args.seeds)} - {', '.join(map(str, args.seeds))}")
    print(f"\nTotal experiments: {len(configs) * len(args.datasets) * len(args.models) * len(args.folds) * len(args.seeds)}")
    print("="*80)


if __name__ == '__main__':
    main()

