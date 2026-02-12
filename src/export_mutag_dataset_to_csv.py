#!/usr/bin/env python3
"""
Export Mutag dataset to CSV format for BRICS motif generation.

Output CSV columns: smiles, label, [verified].

By default, SMILES are derived from the graph (node types + edge_index) using
shared graph_to_smiles logic. Optionally, provide an external SMILES file;
then SMILES are loaded from file and verified against the graph.

Usage:
    # Default: graph-derived SMILES (no external file needed)
    python export_mutag_dataset_to_csv.py --data_dir ../data --output_dir ../dataset_csvs

    # With external SMILES file (e.g. from chrsmrrs)
    python export_mutag_dataset_to_csv.py --smiles_file path/to/mutag_smiles.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from graph_to_smiles_utils import mutag_graph_to_smiles, verify_smiles_vs_graph, RDKIT_AVAILABLE
from datasets import Mutag


def load_external_smiles(smiles_file: Path):
    """
    Load SMILES from an external file.
    
    Expected format:
    - One SMILES per line, OR
    - CSV with 'smiles' column, OR
    - TSV with graph_id and smiles columns
    
    Args:
        smiles_file: Path to SMILES file
        
    Returns:
        List of SMILES strings
    """
    print(f"Loading SMILES from external file: {smiles_file}")
    
    if smiles_file.suffix == '.csv':
        df = pd.read_csv(smiles_file)
        if 'smiles' in df.columns:
            return df['smiles'].tolist()
        else:
            return df.iloc[:, 0].tolist()  # First column
    elif smiles_file.suffix in ['.txt', '.smi']:
        with open(smiles_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            # Check if it's tab-separated (graph_id \t smiles)
            if '\t' in lines[0]:
                return [line.split('\t')[1] for line in lines]
            else:
                return lines
    else:
        raise ValueError(f"Unsupported file format: {smiles_file.suffix}")


def check_for_smiles_file(data_dir: Path):
    """
    Check if SMILES file exists in the data directory.
    
    Args:
        data_dir: Data directory
        
    Returns:
        Path to SMILES file if found, else None
    """
    # Common SMILES file names
    candidates = [
        'mutag_smiles.txt',
        'mutag_smiles.csv',
        'Mutagenicity_smiles.txt',
        'Mutagenicity_smiles.csv',
        'smiles.txt',
        'smiles.csv'
    ]
    
    mutag_dir = data_dir / 'mutag' / 'raw'
    if mutag_dir.exists():
        for candidate in candidates:
            smiles_file = mutag_dir / candidate
            if smiles_file.exists():
                return smiles_file
    
    return None


def export_mutag(data_dir: Path, output_dir: Path, smiles_file: Path = None):
    """
    Export Mutag dataset to CSV (smiles, label, verified).

    By default uses graph-derived SMILES (node types + edge_index). If
    smiles_file is provided (or found in data_dir), uses external SMILES
    and verifies them against the graph.
    """
    print("\n" + "="*80)
    print("Exporting Mutag dataset")
    print("="*80)

    try:
        dataset = Mutag(root=data_dir / 'mutag')
        print(f"✓ Loaded {len(dataset)} graphs from Mutag dataset")

        use_external = False
        if smiles_file is not None:
            use_external = True
        else:
            smiles_file = check_for_smiles_file(data_dir)
            if smiles_file is not None:
                use_external = True

        if use_external:
            # ---- External SMILES: load and verify ----
            print(f"✓ Using external SMILES file: {smiles_file}")
            smiles_list = load_external_smiles(smiles_file)
            print(f"✓ Loaded {len(smiles_list)} SMILES strings")
            if len(smiles_list) != len(dataset):
                print(f"⚠️  SMILES count ({len(smiles_list)}) != dataset count ({len(dataset)})")

            print("\n🔍 Verifying SMILES vs graph structures...")
            verified_data = []
            match_count = 0
            mismatch_count = 0
            invalid_smiles_count = 0
            max_idx = min(len(smiles_list), len(dataset))

            for i in tqdm(range(max_idx), desc="Verifying SMILES-graph pairs"):
                smiles = smiles_list[i]
                graph_data = dataset[i]
                if not hasattr(graph_data, 'y'):
                    mismatch_count += 1
                    continue
                label = graph_data.y.item() if hasattr(graph_data.y, 'item') else graph_data.y
                verification = verify_smiles_vs_graph(smiles, graph_data)

                if verification['match'] is None:
                    verified_data.append({
                        'graph_id': i, 'smiles': smiles, 'label': label,
                        'verified': False, 'reason': verification['reason']
                    })
                elif verification['match']:
                    verified_data.append({
                        'graph_id': i, 'smiles': smiles, 'label': label,
                        'verified': True, 'reason': 'Verified'
                    })
                    match_count += 1
                else:
                    mismatch_count += 1
                    if 'Invalid SMILES' in verification['reason']:
                        invalid_smiles_count += 1
                    verified_data.append({
                        'graph_id': i, 'smiles': smiles, 'label': label,
                        'verified': False, 'reason': verification['reason']
                    })

            print(f"\n📊 Verification: {match_count} matched, {mismatch_count} mismatched, {invalid_smiles_count} invalid SMILES")
            if mismatch_count > 0:
                mismatches = [d for d in verified_data if not d['verified']]
                for m in mismatches[:2]:
                    print(f"   Graph {m['graph_id']}: {m['reason']}")
            if match_count == 0 and RDKIT_AVAILABLE:
                print("❌ No SMILES-graph pairs matched. Aborting.")
                return False
            if mismatch_count > max_idx * 0.5:
                r = input(f"⚠️  {100*mismatch_count/max_idx:.0f}% mismatched. Continue anyway? (yes/no): ").lower()
                if r not in ('yes', 'y'):
                    return False

            smiles_out = [d['smiles'] for d in verified_data]
            labels_out = [d['label'] for d in verified_data]
            verified_flags = [d['verified'] for d in verified_data]
        else:
            # ---- Graph-derived SMILES (default) ----
            if not RDKIT_AVAILABLE:
                print("❌ RDKit is required for graph-to-SMILES conversion. Install rdkit or use --smiles_file.")
                return False
            print("✓ Using graph-derived SMILES (no external file)")
            verified_data = []
            failed = 0
            for i in tqdm(range(len(dataset)), desc="Converting graphs to SMILES"):
                data = dataset[i]
                if not hasattr(data, 'node_type'):
                    failed += 1
                    continue
                smiles = mutag_graph_to_smiles(data.edge_index, data.node_type)
                if smiles is None:
                    failed += 1
                    continue
                label = data.y.item() if hasattr(data.y, 'item') else data.y
                verified_data.append({
                    'graph_id': i, 'smiles': smiles, 'label': label,
                    'verified': True, 'reason': 'Graph-derived'
                })
            if failed:
                print(f"⚠️  {failed} graphs could not be converted to SMILES (skipped)")
            if len(verified_data) == 0:
                print("❌ No graphs could be converted. Check dataset has node_type and RDKit is available.")
                return False

            smiles_out = [d['smiles'] for d in verified_data]
            labels_out = [d['label'] for d in verified_data]
            verified_flags = [d['verified'] for d in verified_data]

        # ---- Common: write CSV ----
        df = pd.DataFrame({
            'smiles': smiles_out,
            'label': labels_out,
            'verified': verified_flags
        })
        output_file = output_dir / 'mutag.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Exported {len(df)} samples to {output_file}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
        if verified_flags and not all(verified_flags):
            print(f"  Verified: {sum(verified_flags)} / {len(verified_flags)}")
        return True

    except Exception as e:
        print(f"\n✗ Failed to export Mutag: {e}")
        import traceback
        traceback.print_exc()
        return False




def main():
    parser = argparse.ArgumentParser(
        description='Export Mutag dataset to CSV format for BRICS motif generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default SMILES are derived from the graph. Optionally use an external file:

  # Graph-derived SMILES (default)
  python export_mutag_dataset_to_csv.py --data_dir ../data --output_dir ../dataset_csvs

  # External SMILES file
  python export_mutag_dataset_to_csv.py --smiles_file path/to/mutag_smiles.txt

External file formats: .txt/.smi (one SMILES per line), .csv (with 'smiles' column).
For OGB datasets (native SMILES): python export_ogb_to_csv_simple.py
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing Mutag dataset files')
    
    parser.add_argument('--output_dir', type=str, default='../dataset_csvs',
                        help='Directory to save CSV files')
    
    parser.add_argument('--smiles_file', type=str, default=None,
                        help='Path to external SMILES file (required if not in data_dir)')
    
    args = parser.parse_args()
    
    # Create output directory
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get SMILES file path
    smiles_file = Path(args.smiles_file) if args.smiles_file else None
    
    print("\n" + "="*80)
    print("MUTAG DATASET TO CSV EXPORTER")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    if smiles_file:
        print(f"SMILES file: {smiles_file}")
    else:
        print("SMILES: graph-derived (default); use --smiles_file for external file")
    
    # Export Mutag
    success = export_mutag(data_dir, output_dir, smiles_file)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    
    if success:
        print("✓ SUCCESS - Mutag dataset exported to CSV")
        print(f"\n✓ CSV file saved to: {output_dir}")
        
        output_file = output_dir / 'mutag.csv'
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            num_lines = sum(1 for _ in open(output_file)) - 1  # -1 for header
            print(f"\nGenerated file:")
            print(f"  - mutag.csv ({size_mb:.2f} MB, {num_lines:,} samples)")
            
        print("\n💡 Next steps:")
        print("  1. Verify the CSV: head dataset_csvs/mutag.csv")
        print("  2. Generate BRICS motifs from the CSV")
        print("  3. Use motifs in MotifSAT training")
    else:
        print("✗ FAILED - Could not export Mutag dataset")
        print("\n💡 Possible solutions:")
        print("  1. Provide SMILES file: --smiles_file path/to/mutag_smiles.txt")
        print("  2. Place SMILES file in: data/mutag/raw/mutag_smiles.txt")
        print("  3. Use OGB datasets: python export_ogb_to_csv_simple.py")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
