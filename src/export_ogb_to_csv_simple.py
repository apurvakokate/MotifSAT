#!/usr/bin/env python3
"""
Simple script to export OGB molecular datasets to CSV format for BRICS motif generation.

This is a standalone script that only requires ogb and pandas.

Usage:
    python export_ogb_to_csv_simple.py --output_dir ../dataset_csvs
    
    # Or specific datasets
    python export_ogb_to_csv_simple.py --datasets ogbg-molhiv ogbg-molbbbp
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    import torch
except ImportError:
    print("Error: Please install required packages:")
    print("  pip install ogb torch pandas")
    exit(1)


def export_ogb_dataset(dataset_name: str, data_dir: Path, output_dir: Path):
    """
    Export OGB molecular dataset to CSV.
    
    Args:
        dataset_name: OGB dataset name (e.g., 'ogbg-molhiv', 'ogbg-molbbbp')
        data_dir: Directory to store/load raw data
        output_dir: Directory to save CSV files
    """
    print("\n" + "="*80)
    print(f"Exporting {dataset_name}")
    print("="*80)
    
    try:
        # Load dataset
        print(f"Loading dataset from {data_dir}...")
        dataset = PygGraphPropPredDataset(root=str(data_dir), name=dataset_name)
        print(f"✓ Loaded {len(dataset)} graphs")
        
        # Get split indices
        split_idx = dataset.get_idx_split()
        print(f"  Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
        
        # Extract SMILES and labels
        print("Extracting SMILES and labels...")
        smiles_list = []
        labels_list = []
        split_list = []
        
        # Process all splits
        for split_name, indices in split_idx.items():
            for idx in tqdm(indices, desc=f"  {split_name:5s}"):
                idx = int(idx)
                
                # Get SMILES
                try:
                    smiles = dataset.smiles[idx]
                except:
                    data = dataset[idx]
                    if hasattr(data, 'smiles'):
                        smiles = data.smiles
                    elif hasattr(data, 'smile'):
                        smiles = data.smile
                    else:
                        print(f"    Warning: No SMILES for idx {idx}, skipping...")
                        continue
                
                # Get label
                data = dataset[idx]
                if hasattr(data, 'y'):
                    label = data.y.squeeze()
                    # Handle multi-dimensional labels
                    if label.dim() > 0:
                        if label.numel() == 1:
                            label = float(label.item())
                        else:
                            # Multi-task: convert to list
                            label = label.cpu().numpy().tolist()
                    else:
                        label = float(label.item())
                else:
                    print(f"    Warning: No label for idx {idx}, skipping...")
                    continue
                
                smiles_list.append(smiles)
                labels_list.append(label)
                split_list.append(split_name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'smiles': smiles_list,
            'label': labels_list,
            'split': split_list
        })
        
        # Save to CSV
        output_file = output_dir / f'{dataset_name.replace("-", "_")}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Exported {len(df)} samples to {output_file}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Split distribution:")
        print(f"{df['split'].value_counts().to_string(header=False)}")
        
        # Print label info
        if isinstance(df['label'].iloc[0], list):
            num_tasks = len(df['label'].iloc[0])
            print(f"  Multi-task dataset with {num_tasks} tasks")
        else:
            print(f"  Label statistics:")
            print(f"    Count: {df['label'].count()}")
            print(f"    Unique: {df['label'].nunique()}")
            if df['label'].nunique() <= 10:
                print(f"  Label distribution:")
                print(f"{df['label'].value_counts().sort_index().to_string()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to export {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export OGB molecular datasets to CSV format for BRICS motif generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available OGB datasets:
  ogbg-molhiv       - HIV virus replication inhibition
  ogbg-molbbbp      - Blood-brain barrier penetration
  ogbg-molbace      - β-secretase inhibition
  ogbg-molclintox   - Clinical trial toxicity
  ogbg-molsider     - Side effects
  ogbg-moltox21     - Toxicity (21 tasks)
  
Example:
  python export_ogb_to_csv_simple.py --datasets ogbg-molhiv ogbg-molbbbp
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory to store/load OGB data files')
    
    parser.add_argument('--output_dir', type=str, default='../dataset_csvs',
                        help='Directory to save CSV files')
    
    parser.add_argument('--datasets', nargs='+',
                        default=['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace',
                                'ogbg-molclintox', 'ogbg-molsider', 'ogbg-moltox21'],
                        help='List of OGB datasets to export')
    
    args = parser.parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("OGB DATASET TO CSV EXPORTER")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets to export: {', '.join(args.datasets)}")
    
    # Track results
    results = {}
    
    # Export each dataset
    for dataset_name in args.datasets:
        success = export_ogb_dataset(dataset_name, data_dir, output_dir)
        results[dataset_name] = success
    
    # Print summary
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    
    for dataset_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        csv_name = dataset_name.replace('-', '_') + '.csv'
        print(f"{status:12s} - {dataset_name:20s} → {csv_name}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nTotal: {successful}/{total} datasets exported successfully")
    
    if successful > 0:
        print(f"\n✓ CSV files saved to: {output_dir}")
        print("\nGenerated files:")
        for f in sorted(output_dir.glob('*.csv')):
            size_mb = f.stat().st_size / (1024 * 1024)
            num_lines = sum(1 for _ in open(f)) - 1  # -1 for header
            print(f"  - {f.name:30s} ({size_mb:6.2f} MB, {num_lines:,} samples)")
    
    print("\n" + "="*80)
    print("\n💡 Next steps:")
    print("  1. Use these CSV files to generate BRICS motifs")
    print("  2. Run motif extraction: python generate_brics_motifs.py")
    print("  3. Use motifs in MotifSAT training")


if __name__ == '__main__':
    main()
