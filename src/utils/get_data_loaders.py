import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Batch, InMemoryDataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from DataLoader import MolDataset, get_setup_files_with_folds
from ogb.graphproppred import PygGraphPropPredDataset
from datasets import SynGraphDataset, Mutag, SPMotif, MNIST75sp, graph_sst2


class OGBDatasetWithSmiles:
    """Wraps an OGB PyG dataset so each Data has a .smiles attribute from mapping/mol.csv.gz."""

    def __init__(self, ogb_dataset, smiles_list):
        self._dataset = ogb_dataset
        self._smiles = list(smiles_list)
        assert len(self._smiles) == len(self._dataset), (
            f"smiles length {len(self._smiles)} != dataset length {len(self._dataset)}"
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # OGB split_idx can be tensors; ensure Python int for indexing
        idx = int(idx.item() if hasattr(idx, 'item') else idx)
        data = self._dataset[idx]
        data.smiles = self._smiles[idx]
        return data

    def get_idx_split(self):
        return self._dataset.get_idx_split()

    def copy(self, indices):
        """Return a Subset so that test_set[i] still yields Data with .smiles (for visualization)."""
        from torch.utils.data import Subset
        return Subset(self, indices)

DATASET_COLUMN = {
                  'Mutagenicity':['Mutagenicity'], 
                  'hERG':['hERG'], 
                  'BBBP':['BBBP'],
                  'Lipophilicity':['Lipophilicity'],
                  'tox21': ['NR-AR', 'NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD', 
                            'NR-PPAR-gamma', 'SR-ARE','SR-ATAD5', 'SR-HSE','SR-MMP','SR-p53'],
                  'esol':['measured log solubility in mols per litre'],
                  'Benzene':['label'],
                  'Alkane_Carbonyl':['label'],
                  'Fluoride_Carbonyl':['label'],
                }

DATASET_TYPE = {
                  'Mutagenicity':'BinaryClass', 
                  'hERG':'BinaryClass', 
                  'BBBP':'BinaryClass',
                  'Lipophilicity':'Regression',
                  'tox21': 'MultiTask',
                  'esol':'Regression',
                  'Benzene':'BinaryClass',
                  'Alkane_Carbonyl':'BinaryClass',
                  'Fluoride_Carbonyl':'BinaryClass',
                }

CHOSEN_THRESHOLD = {'BRICS': 
                    {'Mutagenicity':0.2,
                     'hERG':0.5,
                     'BBBP':0.6,
                     'Benzene':0.5,
                     'Alkane_Carbonyl':0.5,
                     'Fluoride_Carbonyl':0.5,
                     'esol':0.2,
                     'Lipophilicity':0.5,
                     'tox21':0.2}
                   }

def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False, fold=None, path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DICTIONARY"):
    multi_label = False
    assert dataset_name in ['ba_2motifs', 'mutag', 'Graph-SST2', 'mnist',
                            'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
                            'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
                            'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider',
                            'BBBP', 'esol', 'Lipophilicity', 'Mutagenicity', 'hERG', 
                            'Fluoride_Carbonyl', 'Alkane_Carbonyl', 'Benzene']

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
        # Attach SMILES from mapping/mol.csv.gz (i-th row = i-th graph)
        mol_csv = Path(data_dir) / dataset_name / 'mapping' / 'mol.csv.gz'
        if mol_csv.exists():
            import pandas as pd
            df_mol = pd.read_csv(mol_csv)
            if 'smiles' in df_mol.columns and len(df_mol) == len(dataset):
                smiles_list = df_mol['smiles'].astype(str).tolist()
                dataset = OGBDatasetWithSmiles(dataset, smiles_list)
                print('[INFO] Attached data.smiles from mapping/mol.csv.gz')
            else:
                print('[WARNING] mapping/mol.csv.gz missing or length mismatch; data.smiles not set')
        else:
            print('[WARNING] mapping/mol.csv.gz not found; data.smiles not set')
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'Graph-SST2':
        dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
        dataloader, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(dataset, batch_size=batch_size, degree_bias=True, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization

    elif 'spmotif' in dataset_name:
        b = float(dataset_name.split('_')[-1])
        train_set = SPMotif(root=data_dir / dataset_name, b=b, mode='train')
        valid_set = SPMotif(root=data_dir / dataset_name, b=b, mode='val')
        test_set = SPMotif(root=data_dir / dataset_name, b=b, mode='test')
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})

    elif dataset_name == 'mnist':
        n_train_data, n_val_data = 20000, 5000
        train_val = MNIST75sp(data_dir / 'mnist', mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(random_state))
        train_val = train_val[perm_idx]

        train_set, valid_set = train_val[:n_train_data], train_val[-n_val_data:]
        test_set = MNIST75sp(data_dir / 'mnist', mode='test')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})
        print('[INFO] Using default splits!')
        
    elif dataset_name in ['BBBP','Mutagenicity','hERG','Benzene','Alkane_Carbonyl','Fluoride_Carbonyl', 'esol', 'Lipophilicity']:
        algorithm = "BRICS"
        date_tag = f"{algorithm}{CHOSEN_THRESHOLD[algorithm][dataset_name]}"

        base_name = dataset_name
        csv_path = f"/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DomainDrivenGlobalExpl/datasets/FOLDS/{base_name}_{fold}.csv"

        lookup, _, _, _, _, _, test_data_lookup, _, train_mask_data,val_mask_data,test_mask_data = get_setup_files_with_folds(
            base_name, date_tag, fold, algorithm, path=path
        )
        
        if DATASET_TYPE[dataset_name] == 'Regression':
            # Access training and validation data
            train_set = MolDataset(
                root=".", split='training',csv_file=csv_path, 
                label_col = DATASET_COLUMN[dataset_name], normalize = True, mean = None, std = None, lookup = lookup)
            valid_set = MolDataset(
                root=".", split='valid',csv_file=csv_path, 
                label_col = DATASET_COLUMN[dataset_name], normalize = True, mean = train_set.mean, std = train_set.std, lookup = lookup)
            test_set = MolDataset(
                root=".", split='test',csv_file=csv_path, 
                label_col = DATASET_COLUMN[dataset_name], normalize = True, mean = train_set.mean, std = train_set.std, lookup = test_data_lookup)
            num_class = 1
        else:
            train_set = MolDataset(
                root=".", csv_file=csv_path, split='training',
                label_col=DATASET_COLUMN[dataset_name], normalize=False, lookup=lookup
            )
            valid_set = MolDataset(
                root=".", csv_file=csv_path, split='valid',
                label_col=DATASET_COLUMN[dataset_name], normalize=False, lookup=lookup
            )
            test_set = MolDataset(
                root=".", csv_file=csv_path, split='test',
                label_col=DATASET_COLUMN[dataset_name], normalize=False, lookup=test_data_lookup
            )
            num_class =2

        loaders = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "valid": DataLoader(valid_set, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_set, batch_size=batch_size, shuffle=False)
        }
        # Also return original datasets for score saving
        datasets = {
            'train': train_set,
            'valid': valid_set,
            'test': test_set
        }

        masked_data_features = {
            'train':train_mask_data, 
            'valid':val_mask_data, 
            'test':test_mask_data,
        }
        x_dim = train_set[0].x.shape[1]
        edge_attr_dim = train_set[0].edge_attr.shape[1]
        batched_train = Batch.from_data_list([train_set[i] for i in range(len(train_set))])
        d = degree(batched_train.edge_index[1], num_nodes=batched_train.num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=10)
        aux_info = {'deg': deg, 'multi_label': False}
        return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features
    else:
        raise ValueError(f"[ERROR] Unknown dataset: {dataset_name}")

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print(f'[INFO] Num Classes {num_class} : Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    # degree computation
    try:
        batched_train_set = Batch.from_data_list(train_set)
        d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=10)
    except Exception as e:
        print(f"[WARNING] Failed to compute degree: {e}")
        deg = torch.zeros(10, dtype=torch.long)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
