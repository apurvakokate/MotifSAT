#!/usr/bin/env python
"""
Generate a shell script to rerun experiments with missing metrics.

Scans the same results directory that collect_mutagenicity_tables.py uses,
identifies seed directories where required metric keys are absent from
final_metrics.json, then emits a bash script that:

  1. Renames each incomplete seed directory  (→ …_incomplete_YYYYMMDD_HHMMSS)
     so that check_artifacts_exist() will not skip the rerun.
  2. Invokes run_mutagenicity_gsat_experiment.py with the minimal set of
     --experiments / --models / --seeds / --folds needed.

Usage:
  python generate_rerun_script.py \
      --experiment_name base_gsat_decay_r_node_repaired \
      --results_dir ../tuning_results \
      --dataset Mutagenicity \
      --output_script rerun_missing.sh \
      --verbose

  # Then inspect and run:
  bash rerun_missing.sh
"""

import argparse
import json
import os
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_REQUIRED_METRICS = [
    'motif/att_impact_correlation',
    'motif_edge_att/max_mean',
    'motif_edge_att/min_mean',
]


def _read_json(path: Path):
    with open(path) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _infer_model_from_parts(parts):
    for p in parts:
        if p.startswith('model_'):
            return p.replace('model_', '')
    return None


def _infer_fold_seed_from_parts(parts):
    for p in parts:
        m = re.match(r'fold(\d+)_seed(\d+)', p)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def find_incomplete_seeds(results_dir: Path, experiment_name: str, dataset: str,
                          required_metrics: list, verbose: bool = False):
    """
    Walk the results tree and return a list of dicts describing every seed
    directory that is missing at least one required metric key.

    Each dict: {seed_dir, model, fold, seed, missing_keys, present_keys}
    """
    base = results_dir / dataset
    experiment_dir = f'experiment_{experiment_name}'

    candidate_seed_dirs = set()
    for p in base.rglob('final_metrics.json'):
        if experiment_dir in p.parts:
            candidate_seed_dirs.add(p.parent)

    if verbose:
        print(f'[INFO] Scanning {len(candidate_seed_dirs)} seed dirs for experiment={experiment_name}')

    incomplete = []
    complete = 0

    for seed_dir in sorted(candidate_seed_dirs):
        parts = seed_dir.relative_to(base).parts
        model = _infer_model_from_parts(parts)
        fold, seed = _infer_fold_seed_from_parts(parts)
        if model is None or fold is None or seed is None:
            if verbose:
                print(f'  [SKIP] Cannot parse path: {seed_dir}')
            continue

        fm_path = seed_dir / 'final_metrics.json'
        metrics = _read_json(fm_path) or {}

        missing = [k for k in required_metrics if k not in metrics]

        if missing:
            incomplete.append({
                'seed_dir': seed_dir,
                'model': model,
                'fold': fold,
                'seed': seed,
                'missing_keys': missing,
                'present_keys': [k for k in required_metrics if k in metrics],
            })
            if verbose:
                print(f'  [INCOMPLETE] {seed_dir.name}  model={model} fold={fold} seed={seed}')
                print(f'               missing: {missing}')
        else:
            complete += 1

    if verbose:
        print(f'[INFO] Complete: {complete}, Incomplete: {len(incomplete)}')

    return incomplete


def generate_script(incomplete, experiment_name: str, dataset: str,
                    cuda: int, output_path: Path, verbose: bool = False):
    """
    Write a bash script that renames incomplete dirs and reruns the experiment.
    """
    if not incomplete:
        print('[INFO] All runs are complete — nothing to rerun.')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    models = sorted({r['model'] for r in incomplete})
    folds = sorted({r['fold'] for r in incomplete})
    seeds = sorted({r['seed'] for r in incomplete})

    lines = [
        '#!/usr/bin/env bash',
        f'# Auto-generated rerun script — {datetime.now().isoformat()}',
        f'# Experiment : {experiment_name}',
        f'# Dataset    : {dataset}',
        f'# Incomplete : {len(incomplete)} seed dirs',
        f'# Models     : {" ".join(models)}',
        f'# Folds      : {" ".join(str(f) for f in folds)}',
        f'# Seeds      : {" ".join(str(s) for s in seeds)}',
        '',
        'set -euo pipefail',
        '',
        '# ------------------------------------------------------------------',
        '# 1. Rename incomplete seed directories so they are not skipped',
        '#    (check_artifacts_exist looks for final_metrics.json + scores)',
        '# ------------------------------------------------------------------',
    ]

    missing_summary = defaultdict(list)
    for rec in incomplete:
        sd = rec['seed_dir']
        backup = sd.parent / f'{sd.name}_incomplete_{timestamp}'
        lines.append(f'echo "Renaming {sd.name} → {backup.name}"')
        lines.append(f'mv "{sd}" "{backup}"')
        for k in rec['missing_keys']:
            missing_summary[k].append(f"{rec['model']}/fold{rec['fold']}/seed{rec['seed']}")

    lines += [
        '',
        '# ------------------------------------------------------------------',
        '# 2. Rerun the experiment (only renamed dirs will be retrained;',
        '#    complete runs still have artifacts and will be skipped)',
        '# ------------------------------------------------------------------',
        '',
        f'echo ""',
        f'echo "Rerunning {len(incomplete)} incomplete runs …"',
        f'echo ""',
        '',
    ]

    # Build the python command
    models_arg = ' '.join(models)
    folds_arg = ' '.join(str(f) for f in folds)
    seeds_arg = ' '.join(str(s) for s in seeds)
    cmd = (
        f'python run_mutagenicity_gsat_experiment.py \\\n'
        f'    --experiment {experiment_name} \\\n'
        f'    --dataset {dataset} \\\n'
        f'    --models {models_arg} \\\n'
        f'    --folds {folds_arg} \\\n'
        f'    --seeds {seeds_arg} \\\n'
        f'    --cuda {cuda}'
    )
    lines.append(cmd)
    lines.append('')
    lines.append(f'echo ""')
    lines.append(f'echo "Rerun complete."')
    lines.append('')

    script_text = '\n'.join(lines) + '\n'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_text)
    os.chmod(output_path, 0o755)

    print(f'\nGenerated: {output_path}')
    print(f'  {len(incomplete)} incomplete seed dirs to rerun')
    print(f'  Models : {models_arg}')
    print(f'  Folds  : {folds_arg}')
    print(f'  Seeds  : {seeds_arg}')
    print()
    print('Missing metrics summary:')
    for metric_key, locations in sorted(missing_summary.items()):
        print(f'  {metric_key}: missing in {len(locations)} runs')
        for loc in locations:
            print(f'    - {loc}')
    print()
    print(f'Review the script, then run:  bash {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate a shell script to rerun experiments with missing metrics')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name (e.g. base_gsat_decay_r_node_repaired)')
    parser.add_argument('--dataset', type=str, default='Mutagenicity')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results dir (default: RESULTS_DIR env or ../tuning_results)')
    parser.add_argument('--required_metrics', type=str, nargs='+',
                        default=DEFAULT_REQUIRED_METRICS,
                        help='Metric keys that must be present in final_metrics.json')
    parser.add_argument('--output_script', type=str, default=None,
                        help='Path for the generated shell script (default: rerun_{experiment_name}.sh)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device for the rerun command')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir or os.environ.get('RESULTS_DIR', '../tuning_results'))
    output_script = Path(args.output_script or f'rerun_{args.experiment_name}.sh')

    incomplete = find_incomplete_seeds(
        results_dir, args.experiment_name, args.dataset,
        args.required_metrics, verbose=args.verbose)

    generate_script(incomplete, args.experiment_name, args.dataset,
                    args.cuda, output_script, verbose=args.verbose)


if __name__ == '__main__':
    main()
