#!/usr/bin/env python3
"""
Export Mutag dataset to CSV format for BRICS motif generation.

This script attempts to export the Mutag dataset to CSV with:
- Column 1: smiles (SMILES string)
- Column 2: label (graph label)

IMPORTANT NOTE: 
The Mutag dataset from GSAT does NOT contain SMILES strings in its original format.
It only has:
  - Node features (atom types)
  - Edge indices
  - Graph labels
  - Ground-truth edge labels

This script will attempt to find SMILES from alternative sources:
1. Check if a SMILES file exists in the raw data directory
2. Try to load from an external SMILES mapping file

If SMILES cannot be found, the script will fail and suggest alternatives.

Usage:
    python export_dataset_to_csv.py --data_dir ../data --output_dir ../dataset_csvs
    
    # With external SMILES file
    python export_dataset_to_csv.py --smiles_file path/to/mutag_smiles.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. SMILES verification will be limited.")

# Import dataset loaders
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


def verify_smiles_matches_graph(smiles: str, graph_data, verbose: bool = False):
    """
    Verify that a SMILES string matches the molecular graph structure.
    
    This function checks:
    1. SMILES can be parsed by RDKit
    2. Number of atoms matches
    3. Number of bonds matches (accounting for undirected graph)
    
    Args:
        smiles: SMILES string
        graph_data: PyG Data object with x (node features) and edge_index
        verbose: Print detailed comparison info
        
    Returns:
        dict: {'match': bool, 'reason': str, 'scores': dict}
    """
    if not RDKIT_AVAILABLE:
        return {
            'match': None, 
            'reason': 'RDKit not available, cannot verify',
            'scores': {}
        }
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'match': False,
            'reason': 'Invalid SMILES string',
            'scores': {}
        }
    
    # Get graph properties
    num_atoms_smiles = mol.GetNumAtoms()
    num_bonds_smiles = mol.GetNumBonds()
    num_nodes_graph = graph_data.x.shape[0]
    num_edges_graph = graph_data.edge_index.shape[1] // 2  # Undirected, so divide by 2
    
    scores = {
        'atoms_smiles': num_atoms_smiles,
        'atoms_graph': num_nodes_graph,
        'bonds_smiles': num_bonds_smiles,
        'bonds_graph': num_edges_graph,
        'atoms_match': num_atoms_smiles == num_nodes_graph,
        'bonds_match': num_bonds_smiles == num_edges_graph
    }
    
    if verbose:
        print(f"  SMILES: {num_atoms_smiles} atoms, {num_bonds_smiles} bonds")
        print(f"  Graph:  {num_nodes_graph} nodes, {num_edges_graph} edges")
    
    # Check if structure matches
    if num_atoms_smiles != num_nodes_graph:
        return {
            'match': False,
            'reason': f'Atom count mismatch: SMILES has {num_atoms_smiles}, graph has {num_nodes_graph}',
            'scores': scores
        }
    
    if num_bonds_smiles != num_edges_graph:
        return {
            'match': False,
            'reason': f'Bond count mismatch: SMILES has {num_bonds_smiles}, graph has {num_edges_graph}',
            'scores': scores
        }
    
    return {
        'match': True,
        'reason': 'Structure matches',
        'scores': scores
    }


def export_mutag(data_dir: Path, output_dir: Path, smiles_file: Path = None):
    """
    Export Mutag dataset to CSV.
    
    NOTE: Mutag dataset from GSAT does not contain SMILES strings natively.
          This function requires an external SMILES file to work.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save CSV files
        smiles_file: Optional path to external SMILES file
    """
    print("\n" + "="*80)
    print("Exporting Mutag dataset")
    print("="*80)
    
    try:
        # Load dataset
        dataset = Mutag(root=data_dir / 'mutag')
        print(f"✓ Loaded {len(dataset)} graphs from Mutag dataset")
        
        # Try to find SMILES
        if smiles_file is None:
            print("\n🔍 Searching for SMILES file in data directory...")
            smiles_file = check_for_smiles_file(data_dir)
        
        if smiles_file is None:
            print("\n" + "="*80)
            print("❌ ERROR: No SMILES file found!")
            print("="*80)
            print("\nThe Mutag dataset does NOT contain SMILES strings in the GSAT format.")
            print("It only contains:")
            print("  • Node features (atom types)")
            print("  • Edge indices (molecular graph structure)")
            print("  • Graph labels (mutagenic or not)")
            print("  • Ground-truth edge labels")
            print("\nTo export Mutag to CSV, you need to provide SMILES strings from an")
            print("external source. Options:")
            print("\n1. Download original Mutagenicity dataset with SMILES:")
            print("   https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip")
            print("\n2. Use a SMILES mapping file:")
            print("   python export_dataset_to_csv.py --smiles_file path/to/mutag_smiles.txt")
            print("\n3. Use OGB molecular datasets instead (they include SMILES):")
            print("   python export_ogb_to_csv_simple.py")
            print("\n" + "="*80)
            return False
        
        # Load SMILES
        print(f"✓ Found SMILES file: {smiles_file}")
        smiles_list = load_external_smiles(smiles_file)
        print(f"✓ Loaded {len(smiles_list)} SMILES strings")
        
        # Verify count matches
        if len(smiles_list) != len(dataset):
            print(f"\n⚠️  WARNING: SMILES count ({len(smiles_list)}) != dataset count ({len(dataset)})")
            print("    Will verify each SMILES-graph pair individually")
        
        # Verify SMILES match graph structures
        print("\n🔍 Verifying SMILES match graph structures...")
        print("    This ensures data integrity by checking molecular structure correspondence")
        
        verified_data = []
        match_count = 0
        mismatch_count = 0
        invalid_smiles_count = 0
        
        # Check how many we can verify
        max_idx = min(len(smiles_list), len(dataset))
        
        for i in tqdm(range(max_idx), desc="Verifying SMILES-graph pairs"):
            smiles = smiles_list[i]
            graph_data = dataset[i]
            
            # Get label
            if hasattr(graph_data, 'y'):
                label = graph_data.y.item() if hasattr(graph_data.y, 'item') else graph_data.y
            else:
                mismatch_count += 1
                continue
            
            # Verify SMILES matches graph
            verification = verify_smiles_matches_graph(smiles, graph_data)
            
            if verification['match'] is None:
                # RDKit not available, proceed with warning
                verified_data.append({
                    'graph_id': i,
                    'smiles': smiles,
                    'label': label,
                    'verified': False,
                    'reason': verification['reason']
                })
            elif verification['match']:
                # Match confirmed
                verified_data.append({
                    'graph_id': i,
                    'smiles': smiles,
                    'label': label,
                    'verified': True,
                    'reason': 'Verified'
                })
                match_count += 1
            else:
                # Mismatch detected
                mismatch_count += 1
                if 'Invalid SMILES' in verification['reason']:
                    invalid_smiles_count += 1
                
                # Store but mark as unverified
                verified_data.append({
                    'graph_id': i,
                    'smiles': smiles,
                    'label': label,
                    'verified': False,
                    'reason': verification['reason']
                })
        
        # Report verification results
        print(f"\n📊 Verification Results:")
        print(f"   ✓ Matched:          {match_count:4d} ({100*match_count/max_idx:.1f}%)")
        print(f"   ✗ Mismatched:       {mismatch_count:4d} ({100*mismatch_count/max_idx:.1f}%)")
        print(f"   ? Invalid SMILES:   {invalid_smiles_count:4d}")
        print(f"   ━ Total processed:  {max_idx:4d}")
        
        if mismatch_count > 0:
            print(f"\n⚠️  WARNING: {mismatch_count} SMILES-graph pairs did not match!")
            print("    This indicates the SMILES file may not be properly aligned with the dataset.")
            print("    Possible causes:")
            print("      • Different molecule ordering")
            print("      • SMILES from a different version of Mutag")
            print("      • Corrupted or incomplete SMILES file")
            
            # Show first few mismatches
            mismatches = [d for d in verified_data if not d['verified']]
            if mismatches:
                print(f"\n    First {min(3, len(mismatches))} mismatches:")
                for mismatch in mismatches[:3]:
                    print(f"      Graph {mismatch['graph_id']}: {mismatch['reason']}")
                    print(f"        SMILES: {mismatch['smiles'][:60]}...")
        
        # Decide whether to proceed
        if match_count == 0 and RDKIT_AVAILABLE:
            print("\n❌ CRITICAL ERROR: No SMILES-graph pairs matched!")
            print("    Cannot safely export data. Please check your SMILES file.")
            return False
        
        if mismatch_count > max_idx * 0.5:  # More than 50% mismatch
            print(f"\n⚠️  WARNING: {100*mismatch_count/max_idx:.0f}% of pairs mismatched.")
            user_input = input("    Continue anyway? (yes/no): ").lower()
            if user_input not in ['yes', 'y']:
                print("    Export cancelled.")
                return False
        
        # Extract verified data
        smiles_list = [d['smiles'] for d in verified_data]
        labels_list = [d['label'] for d in verified_data]
        verified_flags = [d['verified'] for d in verified_data]
        
        # Create DataFrame
        df = pd.DataFrame({
            'smiles': smiles_list,
            'label': labels_list,
            'verified': verified_flags
        })
        
        # Save to CSV
        output_file = output_dir / 'mutag.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully exported {len(df)} samples to {output_file}")
        print(f"  Columns: {list(df.columns)}")
        
        # Verification summary
        verified_count = df['verified'].sum()
        print(f"\n  Verification status:")
        print(f"    Verified:   {verified_count:4d} ({100*verified_count/len(df):.1f}%)")
        print(f"    Unverified: {len(df)-verified_count:4d} ({100*(len(df)-verified_count)/len(df):.1f}%)")
        
        print(f"\n  Label distribution:")
        print(df['label'].value_counts().sort_index().to_string())
        
        # Sample preview
        print(f"\n📋 Sample preview (first 3 verified rows):")
        verified_samples = df[df['verified']].head(3)
        if len(verified_samples) > 0:
            print(verified_samples[['smiles', 'label']].to_string(index=False))
        else:
            print("  (No verified samples available)")
            print(f"\n  First 3 rows (unverified):")
            print(df[['smiles', 'label']].head(3).to_string(index=False))
        
        # Warning about unverified data
        if verified_count < len(df):
            print(f"\n⚠️  Note: CSV includes 'verified' column indicating which SMILES matched graph structure")
            print(f"   You may want to filter: df = df[df['verified'] == True]")
        
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
IMPORTANT NOTE:
The Mutag dataset does NOT contain SMILES strings in its original GSAT format.
You must provide an external SMILES file using the --smiles_file option.

Expected SMILES file formats:
  1. Plain text file (.txt or .smi): One SMILES per line
  2. CSV file (.csv): With 'smiles' column
  3. TSV file (.txt): With graph_id and smiles columns (tab-separated)

Examples:
  # With external SMILES file
  python export_dataset_to_csv.py --smiles_file path/to/mutag_smiles.txt
  
  # Auto-search for SMILES in data directory
  python export_dataset_to_csv.py --data_dir ../data --output_dir ../dataset_csvs

Alternative:
  For datasets with native SMILES support, use:
  python export_ogb_to_csv_simple.py  # For OGB molecular datasets
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
        print("SMILES file: Auto-search in data directory")
    
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
