# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, BRICS
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. SMILES conversion will be skipped.")


class Mutag(InMemoryDataset):
    def __init__(self, root, add_smiles=True, add_motifs=True):
        """
        Args:
            root: Root directory
            add_smiles: If True, convert graphs to SMILES and add to Data objects
            add_motifs: If True, add BRICS motif decomposition and node-to-motif mapping
        """
        self.add_smiles = add_smiles and RDKIT_AVAILABLE
        self.add_motifs = add_motifs and RDKIT_AVAILABLE
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # Load motif vocabulary if it exists
        if self.add_motifs:
            motif_vocab_file = Path(self.processed_dir) / 'motif_vocabulary.pt'
            if motif_vocab_file.exists():
                self.motif_vocabulary = torch.load(motif_vocab_file)
            else:
                self.motif_vocabulary = {}

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        files = ['data.pt']
        if self.add_motifs:
            files.append('motif_vocabulary.pt')
        return files

    def download(self):
        pass
        # raise NotImplementedError

    def graph_to_smiles(self, edge_index, node_type):
        """
        Convert molecular graph to SMILES string.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            node_type: Node types (atom types)
            
        Returns:
            SMILES string or None if conversion fails
        """
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            # Create RDKit molecule
            mol = Chem.RWMol()
            
            # Map node type to atomic number
            # Based on Mutagenicity dataset encoding
            atom_type_map = {
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
            
            # Add atoms
            atom_idx_map = {}
            for idx, atom_type_idx in enumerate(node_type):
                atom_type_idx = int(atom_type_idx)
                if atom_type_idx in atom_type_map:
                    atomic_num = atom_type_map[atom_type_idx]
                    atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
                    atom_idx_map[idx] = atom_idx
                else:
                    # Unknown atom type
                    return None
            
            # Add bonds (avoid duplicates for undirected graph)
            added_bonds = set()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src in atom_idx_map and dst in atom_idx_map:
                    bond_tuple = tuple(sorted([src, dst]))
                    if bond_tuple not in added_bonds:
                        mol.AddBond(atom_idx_map[src], atom_idx_map[dst], Chem.BondType.SINGLE)
                        added_bonds.add(bond_tuple)
            
            # Convert to SMILES
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            
            return smiles
            
        except Exception as e:
            # Conversion failed
            return None
    
    def extract_brics_motifs_with_atom_mapping(self, smiles):
        """
        Extract BRICS motifs and track which atoms belong to which motif.
        
        Args:
            smiles: SMILES string
            
        Returns:
            motifs: List of motif SMILES
            atom_to_motif: List mapping atom index to motif index (within this molecule)
        """
        if not RDKIT_AVAILABLE:
            return [], []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [], []
            
            # Get BRICS fragments with atom tracking
            # BRICSDecompose returns fragment SMILES with dummy atoms marked as [*]
            frags = BRICS.BRICSDecompose(mol, returnMols=True)
            
            if not frags:
                # No fragmentation possible, treat whole molecule as one motif
                return [smiles], [0] * mol.GetNumAtoms()
            
            # Convert fragments back to list
            frag_list = list(frags)
            motif_smiles = []
            
            for frag_mol in frag_list:
                # Convert fragment to canonical SMILES (with dummy atoms)
                frag_smi = Chem.MolToSmiles(frag_mol)
                motif_smiles.append(frag_smi)
            
            # Map each atom to a motif index
            # This is an approximation - we'll assign atoms to motifs based on substructure matching
            atom_to_motif = [-1] * mol.GetNumAtoms()
            
            for motif_idx, frag_mol in enumerate(frag_list):
                # Remove dummy atoms for matching
                frag_pattern = Chem.MolFromSmiles(Chem.MolToSmiles(frag_mol).replace('[*]', '[H]'))
                if frag_pattern is None:
                    continue
                
                # Find substructure matches
                matches = mol.GetSubstructMatches(frag_pattern)
                if matches:
                    # Use first match
                    for atom_idx in matches[0]:
                        if atom_to_motif[atom_idx] == -1:  # Not yet assigned
                            atom_to_motif[atom_idx] = motif_idx
            
            # Assign remaining unassigned atoms to motif 0 (or create a default motif)
            for i in range(len(atom_to_motif)):
                if atom_to_motif[i] == -1:
                    atom_to_motif[i] = 0  # Assign to first motif as fallback
            
            return motif_smiles, atom_to_motif
            
        except Exception as e:
            # If BRICS fails, treat whole molecule as one motif
            return [smiles], [0] * Chem.MolFromSmiles(smiles).GetNumAtoms()

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

        data_list = []
        skipped_count = 0
        smiles_conversion_failed = 0
        
        # First pass: collect all unique motifs to build vocabulary
        motif_vocabulary = {}  # motif_smiles -> motif_index
        motif_index = 0
        
        if self.add_motifs:
            print("\nFirst pass: Building motif vocabulary...")
            temp_smiles_list = []
            
            for i in range(original_labels.shape[0]):
                num_nodes = len(node_type_lists[i])
                edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T
                y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
                
                # Check if should skip
                edge_label = torch.tensor(edge_label_lists[i]).float()
                if y.item() != 0:
                    edge_label = torch.zeros_like(edge_label).float()
                signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
                
                if y.item() == 0 and len(signal_nodes) == 0:
                    continue
                
                # Convert to SMILES
                smiles = self.graph_to_smiles(edge_index, torch.tensor(node_type_lists[i]))
                if smiles is None:
                    continue
                
                temp_smiles_list.append(smiles)
                
                # Extract motifs
                motifs, _ = self.extract_brics_motifs_with_atom_mapping(smiles)
                for motif in motifs:
                    if motif not in motif_vocabulary:
                        motif_vocabulary[motif] = motif_index
                        motif_index += 1
            
            print(f"Found {len(motif_vocabulary)} unique motifs across {len(temp_smiles_list)} molecules")
        
        # Second pass: process data with motif assignments
        for i in range(original_labels.shape[0]):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                skipped_count += 1
                continue

            # Convert graph to SMILES
            smiles = None
            if self.add_smiles or self.add_motifs:
                smiles = self.graph_to_smiles(edge_index, torch.tensor(node_type_lists[i]))
                
                if smiles is None:
                    # Skip graphs that can't be converted to SMILES
                    smiles_conversion_failed += 1
                    continue
            
            # Extract motifs and create node-to-motif mapping
            node_to_motifs = None
            if self.add_motifs and smiles is not None:
                motifs, atom_to_motif_local = self.extract_brics_motifs_with_atom_mapping(smiles)
                
                # Map local motif indices to global vocabulary indices
                node_to_motifs = []
                for local_motif_idx in atom_to_motif_local:
                    if local_motif_idx < len(motifs):
                        motif_smiles = motifs[local_motif_idx]
                        global_motif_idx = motif_vocabulary.get(motif_smiles, 0)
                        node_to_motifs.append(global_motif_idx)
                    else:
                        node_to_motifs.append(0)  # Default motif
                
                # Convert to tensor
                node_to_motifs = torch.tensor(node_to_motifs, dtype=torch.long)
                
                # Verify length matches number of nodes
                if len(node_to_motifs) != x.shape[0]:
                    print(f"Warning: Graph {i} motif mapping size mismatch. "
                          f"Nodes: {x.shape[0]}, Motifs: {len(node_to_motifs)}")
                    # Pad or truncate to match
                    if len(node_to_motifs) < x.shape[0]:
                        node_to_motifs = torch.cat([
                            node_to_motifs,
                            torch.zeros(x.shape[0] - len(node_to_motifs), dtype=torch.long)
                        ])
                    else:
                        node_to_motifs = node_to_motifs[:x.shape[0]]
            
            # Create Data object with SMILES and motifs
            data = Data(
                x=x, 
                y=y, 
                edge_index=edge_index, 
                node_label=node_label, 
                edge_label=edge_label, 
                node_type=torch.tensor(node_type_lists[i])
            )
            
            # Add SMILES as attribute (not as tensor)
            if smiles is not None:
                data.smiles = smiles
            
            # Add node-to-motif mapping
            if node_to_motifs is not None:
                data.node_to_motifs = node_to_motifs
                
            data_list.append(data)

        print(f"\nMutag dataset processing:")
        print(f"  Total graphs: {original_labels.shape[0]}")
        print(f"  Skipped (no signal nodes): {skipped_count}")
        print(f"  SMILES conversion failed: {smiles_conversion_failed}")
        print(f"  Final dataset size: {len(data_list)}")
        if self.add_motifs:
            print(f"  Unique motifs in vocabulary: {len(motif_vocabulary)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Save motif vocabulary
        if self.add_motifs:
            motif_vocab_file = Path(self.processed_dir) / 'motif_vocabulary.pt'
            torch.save(motif_vocabulary, motif_vocab_file)
            print(f"  Saved motif vocabulary to {motif_vocab_file}")

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1] = len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start, t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists
