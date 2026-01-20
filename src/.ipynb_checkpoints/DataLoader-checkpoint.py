import numpy as np
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pandas as pd
import pdb


ATOMS = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Cu': 10, 'Bi': 11, 'B': 12, 'Zn': 13, 'Hg': 14, 'Ti': 15, 'Fe': 16, 'Au': 17, 'Mn': 18, 'Tl': 19, 'As': 20, 'Ca': 21, 'Si': 22, 'Co': 23, 'Al': 24, 'Na': 25, 'Ni': 26, 'K': 27, 'Sn': 28, 'Cr': 29, 'Dy': 30, 'Zr': 31, 'Sb': 32, 'In': 33, 'Yb': 34, 'Nd': 35, 'Be': 36, 'Se': 37, 'Cd': 38, 'Li': 39, 'Mg': 40, 'Pt': 41, 'Gd': 42, 'V': 43, 'Ge': 44, 'Mo': 45, 'Ag': 46, 'Ba': 47, 'Pb': 48, 'Sr': 49, 'Pd': 50}

BONDS = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}

STEREO = {'STEREOZ': 0, 'STEREOE': 1, 'STEREOANY': 2, 'STEREONONE': 3}


def load_required_files(base_path):
    
    with open(base_path + '_graph_lookup.pickle', 'rb') as file:
            lookup = pickle.load(file)
    with open(base_path + '_motif_list.pickle', 'rb') as file:
        motif_list = list(pickle.load(file))
    with open(base_path + '_motif_counts.pickle', 'rb') as file:
        motif_counts = pickle.load(file)
    with open(base_path + '_motif_length.pickle', 'rb') as file:
        motif_lengths = pickle.load(file)
    with open(base_path + '_motif_class.pickle', 'rb') as file:
        motif_class_count = pickle.load(file)
    with open(base_path + '_graph_motifidx.pickle', 'rb') as file:
        graph_to_motifs = pickle.load(file)
    with open(base_path + '_test_graph_lookup.pickle', 'rb') as file:
        test_data_lookup = pickle.load(file)
    with open(base_path + '_test_graph_motifidx.pickle', 'rb') as file:
        test_graph_to_motifs = pickle.load(file)
    with open(base_path + '_test_dataset_masked.pickle', 'rb') as file:
        test_mask_data = pickle.load(file)
    with open(base_path + '_train_dataset_masked.pickle', 'rb') as file:
        train_mask_data = pickle.load(file)
    with open(base_path + '_validation_dataset_masked.pickle', 'rb') as file:
        val_mask_data = pickle.load(file)

    return lookup,motif_list,motif_counts,motif_lengths,motif_class_count,graph_to_motifs,test_data_lookup,test_graph_to_motifs, train_mask_data,val_mask_data,test_mask_data
    


def get_setup_files_with_folds(dataset_name, date_tag, fold, algorithm):
    algorithm = 'RBRICS' if algorithm == 'None' else algorithm
    least_count_dict = {'1225':
                            {'Mutagenicity':{'RBRICS':3, 'MGSSL':3}, 
                           'hERG':{'RBRICS':10,'MGSSL':15}, 
                           'BBBP':{'RBRICS':6,'MGSSL':15},
                           'Lipophilicity':{'RBRICS':10,'MGSSL':15},
                           'tox21':{'RBRICS':5,'MGSSL':3},#{'RBRICS':20,'MGSSL':10},#
                           'esol':{'RBRICS':5,'MGSSL':5}},
                        '0205':
                        {'Mutagenicity':{'RBRICS':35, 'MGSSL':3}, 
                       'hERG':{'RBRICS':45, 'MGSSL':15}, 
                       'BBBP':{'RBRICS':9, 'MGSSL':15},
                       'Lipophilicity':{'RBRICS':19, 'MGSSL':15},
                       'esol':{'RBRICS':5, 'MGSSL':5}, 
                       'tox21':{'RBRICS':35, 'MGSSL':4}},
                        '0208':
                        {'Mutagenicity':{'RBRICS':35, 'MGSSL':35}, 
                       'hERG':{'RBRICS':45, 'MGSSL':45}, 
                       'BBBP':{'RBRICS':9, 'MGSSL':9},
                       'Lipophilicity':{'RBRICS':19, 'MGSSL':19},
                       'esol':{'RBRICS':5, 'MGSSL':5}, 
                       'tox21':{'RBRICS':35, 'MGSSL':35}},
                        '0201':
                            {'tox21':{'RBRICS':7,'MGSSL':4}}
                       }
    #Absolute path is used to run on HPC cluster
    path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DICTIONARY"
    
    # print(algorithm)
    if date_tag in least_count_dict:
        least_count = least_count_dict[date_tag][dataset_name][algorithm]
        base_path = f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}'
        
            
    else:
        base_path = f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_{date_tag}'
    
    
    return load_required_files(base_path)


# def get_setup_files(dataset_name, date_tag):
#     with open(f'dictionary/{dataset_name}_graph_lookup_{date_tag}.pickle', 'rb') as file:
#         lookup = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_motif_list_{date_tag}.pickle', 'rb') as file:
#         motif_list = list(pickle.load(file))
#     with open(f'dictionary/{dataset_name}_motif_counts_{date_tag}.pickle', 'rb') as file:
#         motif_counts = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_motif_class_{date_tag}.pickle', 'rb') as file:
#         motif_class_count = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_graph_motifidx_{date_tag}.pickle', 'rb') as file:
#         graph_to_motifs = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_test_graph_lookup_{date_tag}.pickle', 'rb') as file:
#         # Serialize and save the object to the file
#         test_data_lookup = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_test_graph_motifidx_{date_tag}.pickle', 'rb') as file:
#         test_graph_to_motifs = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_test_dataset_masked_{date_tag}.pickle', 'rb') as file:
#         test_mask_data = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_train_dataset_masked_{date_tag}.pickle', 'rb') as file:
#         train_mask_data = pickle.load(file)
#     with open(f'dictionary/{dataset_name}_validation_dataset_masked_{date_tag}.pickle', 'rb') as file:
#         val_mask_data = pickle.load(file)
        
#     return lookup, motif_list, motif_counts, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data


def get_atom_features(atom):
    atom_idx = ATOMS.get(str(atom.GetSymbol()), None)
    if atom_idx is None:
        return None

    return F.one_hot(torch.tensor(atom_idx), num_classes=len(ATOMS))


def get_bond_features(bond, use_stereochemistry=True):
    bond_idx = BONDS.get(bond.GetBondType(), None)
    if bond_idx is None:
        return None

    if use_stereochemistry:
        stereo_idx = STEREO.get(str(bond.GetStereo()), None)
        if stereo_idx is None:
            return None

    bond_one_hot = F.one_hot(torch.tensor(bond_idx), num_classes=len(BONDS))
    stereo_one_hot = F.one_hot(torch.tensor(stereo_idx), num_classes=len(STEREO))

    return torch.cat([bond_one_hot, stereo_one_hot], dim=-1)


def build_graph(smiles, y, lookup):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  

    # Atom features
    type_idx = []
    for atom in mol.GetAtoms():
        atom_idx = ATOMS.get(str(atom.GetSymbol()), None)
        if atom_idx is None:
            raise ValueError(f'Error processing {smiles} at atom {atom.GetSymbol()}. Verify before continuing')
        type_idx.append(atom_idx)

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOMS)).float()

    # Edge indices
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    edge_index = torch.stack(
        [torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)],
        dim=0
    )

    # Edge attributes
    edge_attr = []
    for i, j in zip(rows, cols):
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        edge_feat = get_bond_features(bond)
        if edge_feat is None:
            raise ValueError(f'Error processing {smiles} at bond between {i}-{j}. Verify before continuing')
        edge_attr.append(edge_feat)

    if len(edge_attr) > 0:
        edge_attr = torch.stack(edge_attr, dim=0)
    else:
        edge_attr = torch.tensor([], dtype=torch.float)  # Handle cases with no edges
        
    
    # In your Dataset's __getitem__ method:
    num_nodes = x.size(0)
    node_to_motif_tensor = torch.full((num_nodes,), -1, dtype=torch.long)  # -1 for unmapped nodes
    if lookup is not None:
        nodes_to_motif = lookup[smiles]
        for node_idx, (motif,motif_idx) in nodes_to_motif.items():
            if motif_idx is not None:
                node_to_motif_tensor[node_idx] = motif_idx

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles, nodes_to_motifs=node_to_motif_tensor)


class MolDataset(InMemoryDataset):
    def __init__(self, root, csv_file, split, label_col, normalize=False, mean=None, std=None, 
                 transform=None, pre_transform=None, pre_filter=None, num_classes=None, lookup=None):
        self.csv_file = csv_file
        self.split = split
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['group'] == split]
        self.smiles_list = self.df['smiles'].values
        self.y = self.df[label_col].values
        self.lookup = lookup
        self.num_classes_ = num_classes
        
        # Normalization setup
        self.normalize = normalize
        if self.normalize:
            if mean is None and std is None:
                self.mean = torch.tensor(np.nanmean(self.y), dtype=torch.float)
                self.std = torch.tensor(np.nanstd(self.y), dtype=torch.float)
            else:
                self.mean = torch.tensor(mean, dtype=torch.float)
                self.std = torch.tensor(std, dtype=torch.float)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def num_classes(self):
        return self.num_classes_ if hasattr(self, "num_classes_") else elf.num_classes

    @property
    def processed_file_names(self):
        return f'{self.split}_{Path(self.csv_file).stem}.pt'

    def process(self):
        data_list = []
        for idx, smiles in enumerate(tqdm(self.smiles_list)):
            y = torch.tensor(self.y[idx], dtype=torch.float)
    
            if self.normalize:
                y = (y - self.mean) / self.std
                
            if len(y) > 1:
                y = y.unsqueeze(0)
                

            data = build_graph(smiles, y, self.lookup)
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if data is None: 
                continue
            
            if data.num_nodes == 0:
                print(f'Skipping molecule {smiles} since it resulted in zero atoms')
                continue
            
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])