#!/usr/bin/env python3
"""
Shared utilities for converting graph datasets to SMILES and verifying consistency.

Used by:
  - export_mutag_dataset_to_csv.py (graph-derived SMILES, optional verification)
  - export_ogb_to_csv_simple.py (optional verification of native SMILES)
"""

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Mutag node label -> atomic number (same as datasets.mutag.Mutag.graph_to_smiles)
MUTAG_ATOM_TYPE_MAP = {
    0: 6,   # C (Carbon)
    1: 7,   # N (Nitrogen)
    2: 8,   # O (Oxygen)
    3: 9,   # F (Fluorine)
    4: 15,  # P (Phosphorus)
    5: 16,  # S (Sulfur)
    6: 17,  # Cl (Chlorine)
    7: 35,  # Br (Bromine)
    8: 53,  # I (Iodine)
}


def mutag_graph_to_smiles(edge_index, node_type):
    """
    Convert a Mutag-format molecular graph to a SMILES string.

    Uses RDKit RWMol: node types (0-8) map to atomic numbers, all bonds
    are treated as single. Same logic as datasets.mutag.Mutag.graph_to_smiles.

    Args:
        edge_index: Tensor or array of shape [2, num_edges] (PyG style).
        node_type: Tensor or list of integer node labels (0-8 for C,N,O,F,P,S,Cl,Br,I).

    Returns:
        SMILES string, or None if conversion fails (unknown atom type, sanitization, etc.).
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.RWMol()
        atom_idx_map = {}

        for idx, atom_type_idx in enumerate(node_type):
            atom_type_idx = int(atom_type_idx)
            if atom_type_idx in MUTAG_ATOM_TYPE_MAP:
                atomic_num = MUTAG_ATOM_TYPE_MAP[atom_type_idx]
                rdkit_idx = mol.AddAtom(Chem.Atom(atomic_num))
                atom_idx_map[idx] = rdkit_idx
            else:
                return None

        added_bonds = set()
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i].item() if hasattr(edge_index[0, i], 'item') else edge_index[0, i])
            dst = int(edge_index[1, i].item() if hasattr(edge_index[1, i], 'item') else edge_index[1, i])
            if src in atom_idx_map and dst in atom_idx_map:
                bond_tuple = tuple(sorted([src, dst]))
                if bond_tuple not in added_bonds:
                    mol.AddBond(atom_idx_map[src], atom_idx_map[dst], Chem.BondType.SINGLE)
                    added_bonds.add(bond_tuple)

        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def verify_smiles_vs_graph(smiles: str, graph_data, verbose: bool = False):
    """
    Verify that a SMILES string matches a molecular graph (atom and bond counts).

    Checks: SMILES parses; num_atoms(SMILES) == num_nodes(graph);
    num_bonds(SMILES) == num_edges(graph) (undirected: edge_index.shape[1] // 2).

    Args:
        smiles: SMILES string.
        graph_data: PyG Data with .x (node count from shape[0]) and .edge_index.
        verbose: If True, print counts.

    Returns:
        dict with keys: 'match' (bool or None if RDKit unavailable),
        'reason' (str), 'scores' (dict of counts).
    """
    if not RDKIT_AVAILABLE:
        return {
            'match': None,
            'reason': 'RDKit not available, cannot verify',
            'scores': {}
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'match': False,
            'reason': 'Invalid SMILES string',
            'scores': {}
        }

    num_atoms_smiles = mol.GetNumAtoms()
    num_bonds_smiles = mol.GetNumBonds()
    num_nodes_graph = graph_data.x.shape[0]
    num_edges_graph = graph_data.edge_index.shape[1] // 2

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
