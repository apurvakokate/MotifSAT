#!/usr/bin/env python3
"""
Export OGB molecular datasets to CSV (smiles, label, split).

SMILES and labels are read from the dataset mapping file:
  {data_dir}/{dataset_name}/mapping/mol.csv.gz

Format (see OGB readme): columns are [task_1, ... task_n], SMILES, mol_id.
The i-th PyG data object corresponds to the i-th row in mol.csv.gz.

Usage:
  python export_ogb_to_csv_simple.py --output_dir ../dataset_csvs
  python export_ogb_to_csv_simple.py --datasets ogbg-molhiv ogbg-molbbbp --verify 5
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

try:
    from graph_to_smiles_utils import verify_smiles_vs_graph
    _VERIFY_AVAILABLE = True
except ImportError:
    _VERIFY_AVAILABLE = False

try:
    from torch_geometric.utils import to_smiles as pyg_to_smiles
    _PYG_TO_SMILES_AVAILABLE = True
except ImportError:
    _PYG_TO_SMILES_AVAILABLE = False


def _canonical_smiles(smiles):
    """Return canonical SMILES or None if invalid."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None

# Patch for OGB bug with NaN in metadata
def patch_ogb_metadata():
    """
    Patch OGB to handle NaN values in metadata.
    
    This fixes the error: 'float' object has no attribute 'split'
    which occurs when metadata contains NaN instead of empty string.
    """
    import ogb.graphproppred.dataset_pyg as ogb_pyg
    original_process = ogb_pyg.PygGraphPropPredDataset.process
    
    def patched_process(self):
        # Fix NaN values in meta_info before processing
        if hasattr(self, 'meta_info'):
            for key in ['additional node files', 'additional edge files']:
                if key in self.meta_info:
                    value = self.meta_info[key]
                    # Replace NaN with empty string
                    if isinstance(value, float) and value != value:  # Check for NaN
                        self.meta_info[key] = ''
        return original_process(self)
    
    ogb_pyg.PygGraphPropPredDataset.process = patched_process

# Apply patch
try:
    patch_ogb_metadata()
except:
    pass  # If patching fails, continue anyway


def _load_mol_csv(data_dir: Path, dataset_name: str):
    """Load mapping/mol.csv.gz. Columns: [task_1,...,task_n], smiles, mol_id. i-th row = i-th graph."""
    mapping_path = data_dir / dataset_name.replace('-', '_') / 'mapping' / 'mol.csv.gz'
    if not mapping_path.exists():
        return None, f"File not found: {mapping_path}"
    df = pd.read_csv(mapping_path)
    if 'smiles' not in df.columns:
        return None, f"No 'smiles' column in {mapping_path}. Columns: {list(df.columns)}"
    return df, None


def export_ogb_dataset(dataset_name: str, data_dir: Path, output_dir: Path, verify_sample: int = 0):
    """
    Export OGB molecular dataset to CSV from mapping/mol.csv.gz.
    i-th data object = i-th row in mol.csv.gz (see OGB readme).
    """
    print("\n" + "="*80)
    print(f"Exporting {dataset_name}")
    print("="*80)

    try:
        # Load mapping file first (single source of SMILES)
        df_mol, err = _load_mol_csv(data_dir, dataset_name)
        if df_mol is None:
            print(f"\n✗ {err}")
            return False
        print(f"✓ Loaded {len(df_mol)} rows from mapping/mol.csv.gz")

        # Load dataset
        print(f"Loading dataset from {data_dir}...")
        try:
            dataset = PygGraphPropPredDataset(root=str(data_dir), name=dataset_name)
        except AttributeError as e:
            if "'float' object has no attribute 'split'" in str(e):
                import shutil
                cache_dir = data_dir / dataset_name.replace('-', '_')
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    print("   Cleared cache, re-downloading...")
                dataset = PygGraphPropPredDataset(root=str(data_dir), name=dataset_name)
            else:
                raise

        n_data = len(dataset)
        if len(df_mol) != n_data:
            print(f"\n✗ Row count mismatch: mol.csv.gz has {len(df_mol)}, dataset has {n_data}")
            return False
        print(f"✓ Loaded {n_data} graphs")

        split_idx = dataset.get_idx_split()
        print(f"  Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")

        # Extract SMILES and labels (i-th row = i-th graph)
        print("\nExtracting SMILES and labels...")
        smiles_list = []
        labels_list = []
        split_list = []

        for split_name, indices in split_idx.items():
            for idx in tqdm(indices, desc=f"  {split_name:5s}"):
                idx = int(idx)
                smiles = df_mol['smiles'].iloc[idx]
                if pd.isna(smiles) or not str(smiles).strip():
                    continue
                data = dataset[idx]
                if not hasattr(data, 'y'):
                    continue
                label = data.y.squeeze()
                if label.dim() > 0:
                    label = float(label.item()) if label.numel() == 1 else label.cpu().numpy().tolist()
                else:
                    label = float(label.item())
                smiles_list.append(smiles)
                labels_list.append(label)
                split_list.append(split_name)

        if len(smiles_list) == 0:
            print("\n✗ No valid SMILES/labels extracted")
            return False

        # Optional: verify random sample (SMILES→graph and graph→SMILES)
        if verify_sample > 0:
            rng = np.random.default_rng()
            n_verify = min(verify_sample, n_data)
            indices = rng.choice(n_data, size=n_verify, replace=False)
            print(f"\n🔍 Verifying {n_verify} randomly sampled rows (SMILES↔graph)...")
            ok_s2g = 0
            ok_g2s = 0
            for k, idx in enumerate(indices):
                idx = int(idx)
                smiles_orig = df_mol['smiles'].iloc[idx]
                data = dataset[idx]
                # (a) SMILES → graph: does CSV SMILES match dataset graph?
                if _VERIFY_AVAILABLE:
                    v = verify_smiles_vs_graph(smiles_orig, data)
                    s2g = v.get('match')
                    if s2g:
                        ok_s2g += 1
                    status_s2g = "✓" if s2g else "✗"
                else:
                    status_s2g = "?"
                    s2g = None
                # (b) Graph → SMILES: round-trip and compare canonical SMILES
                g2s_ok = False
                if _PYG_TO_SMILES_AVAILABLE and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    try:
                        smiles_from_graph = pyg_to_smiles(data)
                        can_orig = _canonical_smiles(str(smiles_orig))
                        can_graph = _canonical_smiles(smiles_from_graph) if smiles_from_graph else None
                        g2s_ok = (can_orig is not None and can_graph is not None and can_orig == can_graph)
                        if g2s_ok:
                            ok_g2s += 1
                    except Exception:
                        pass
                status_g2s = "✓" if g2s_ok else "✗"
                print(f"  [{k+1}/{n_verify}] idx {idx}: SMILES→graph {status_s2g}  graph→SMILES {status_g2s}")
            if _VERIFY_AVAILABLE:
                print(f"  SMILES→graph match: {ok_s2g}/{n_verify}")
            if _PYG_TO_SMILES_AVAILABLE:
                print(f"  graph→SMILES match: {ok_g2s}/{n_verify}")

        # Write CSV
        df = pd.DataFrame({'smiles': smiles_list, 'label': labels_list, 'split': split_list})
        output_file = output_dir / f'{dataset_name.replace("-", "_")}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Exported {len(df)} samples to {output_file}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Split distribution:\n{df['split'].value_counts().to_string(header=False)}")
        if len(df) > 0 and isinstance(df['label'].iloc[0], list):
            print(f"  Multi-task: {len(df['label'].iloc[0])} tasks")
        elif df['label'].nunique() <= 10:
            print(f"  Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")

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

    parser.add_argument('--verify', type=int, default=0, metavar='N',
                        help='Verify first N SMILES against graph (uses graph_to_smiles_utils); 0=off')

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
        success = export_ogb_dataset(dataset_name, data_dir, output_dir, verify_sample=args.verify)
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
