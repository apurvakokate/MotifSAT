"""
Shared experiment configuration for GSAT runs.

Used by:
- run_gsat_replication.py (paper datasets, baseline GSAT)
- run_mutagenicity_gsat_experiment.py (Mutagenicity, all folds/architectures, 4 variants)

Provides: PAPER_DATASETS, ARCHITECTURES, PAPER_HYPERPARAMS, get_base_config().
"""

# Paper datasets (replication)
PAPER_DATASETS = [
    'ba_2motifs',
    'mutag',
    'mnist',
    'spmotif_0.5',
    'spmotif_0.7',
    'spmotif_0.9',
    'Graph-SST2',
    'ogbg_molhiv',
    'ogbg_molbace',
    'ogbg_molbbbp',
    'ogbg_molclintox',
    'ogbg_moltox21',
    'ogbg_molsider',
]

# Molecular datasets using MolDataset + folds (Mutagenicity, BBBP, hERG, etc.)
MOL_DATASETS_WITH_FOLDS = ['Mutagenicity', 'BBBP', 'hERG', 'Benzene', 'Alkane_Carbonyl', 'Fluoride_Carbonyl', 'esol', 'Lipophilicity']

# Architectures
ARCHITECTURES = ['GIN', 'PNA', 'GAT', 'SAGE', 'GCN']

# Default hyperparameters from the paper (and for molecular datasets)
PAPER_HYPERPARAMS = {
    'ba_2motifs': {
        'GIN': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.5, 'epochs': 50, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
    },
    'mutag': {
        'GIN': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.5, 'epochs': 50, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.5, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
    },
    'mnist': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 256},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 256},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 256},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 256},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 256},
    },
    'spmotif_0.5': {
        'GIN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 3e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
    },
    'spmotif_0.7': {
        'GIN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 3e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
    },
    'spmotif_0.9': {
        'GIN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 3e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 100, 'lr': 3e-3, 'batch_size': 128},
    },
    'Graph-SST2': {
        'GIN': {'r': 0.7, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 3e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_molhiv': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_molbace': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_molbbbp': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_molclintox': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_moltox21': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'ogbg_molsider': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    # Molecular datasets with folds (Mutagenicity has GT explanations → r=0.5 like mutag)
    'Mutagenicity': {
        'GIN': {'r': 0.5, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.5, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.5, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.5, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.5, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'BBBP': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
    'hERG': {
        'GIN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'PNA': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GAT': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'SAGE': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
        'GCN': {'r': 0.7, 'epochs': 200, 'lr': 1e-3, 'batch_size': 128},
    },
}


def get_base_config(model_name, dataset_name, gsat_overrides=None):
    """
    Get base configuration for a model-dataset combination.
    Uses paper hyperparameters where available.
    Same config structure as used in run_gsat_replication.

    Args:
        model_name: One of GIN, PNA, GAT, SAGE, GCN
        dataset_name: e.g. 'Mutagenicity', 'ogbg_molhiv', 'ba_2motifs', ...
        gsat_overrides: Optional dict to merge into GSAT_config (e.g. motif_loss_coef, tuning_id).

    Returns:
        Full config dict with data_config, model_config, shared_config, GSAT_config.
    """
    # Dataset-specific configurations (based on MAGE ICML'24 paper + DIR ICLR'22)
    if 'ba_2motif' in dataset_name.lower() or 'spmotif' in dataset_name.lower():
        gcn_hidden = 20
        gcn_normalize = False
        gcn_dropout = 0.0
        sage_hidden = 20
        sage_aggr = 'sum'
        sage_dropout = 0.0
        gat_hidden = 64
        backbone_dropout = 0.3
    else:
        gcn_hidden = 64
        gcn_normalize = True
        gcn_dropout = 0.4 if dataset_name == 'ogbg_molbace' else 0.3
        sage_hidden = 64
        sage_aggr = 'mean'
        sage_dropout = 0.4 if dataset_name == 'ogbg_molbace' else 0.3
        gat_hidden = 64
        backbone_dropout = 0.4 if dataset_name == 'ogbg_molbace' else 0.3

    model_defaults = {
        'GIN': {'hidden_size': 64, 'n_layers': 2, 'dropout_p': backbone_dropout},
        'PNA': {
            'hidden_size': 80,
            'n_layers': 4,
            'dropout_p': backbone_dropout,
            'aggregators': ['mean', 'min', 'max', 'std', 'sum'],
            'scalers': False,
        },
        'GAT': {'hidden_size': gat_hidden, 'n_layers': 3, 'dropout_p': backbone_dropout},
        'SAGE': {'hidden_size': sage_hidden, 'n_layers': 3, 'dropout_p': sage_dropout, 'sage_aggr': sage_aggr},
        'GCN': {'hidden_size': gcn_hidden, 'n_layers': 3, 'dropout_p': gcn_dropout, 'gcn_normalize': gcn_normalize},
    }

    hp = PAPER_HYPERPARAMS.get(dataset_name, {}).get(model_name, {'r': 0.7, 'epochs': 100, 'lr': 1e-3, 'batch_size': 128})

    use_atom_encoder = 'ogbg' in dataset_name
    use_edge_attr = (
        'ogbg' in dataset_name
        or 'spmotif' in dataset_name
        or dataset_name in MOL_DATASETS_WITH_FOLDS
    )

    data_config = {
        'splits': {'train': 0.8, 'valid': 0.1, 'test': 0.1},
        'batch_size': hp['batch_size'],
        'mutag_x': dataset_name == 'mutag',
    }

    model_config = {
        'model_name': model_name,
        **model_defaults[model_name],
        'atom_encoder': use_atom_encoder,
        'use_edge_attr': use_edge_attr,
        'pretrain_lr': 1e-3,
        'pretrain_epochs': 100 if model_name != 'PNA' else 50,
    }

    shared_config = {
        'learn_edge_att': True,
        'precision_k': 5,
        'num_viz_samples': 0,
        'viz_interval': 10,
        'viz_norm_att': True,
        'extractor_dropout_p': 0.6 if dataset_name == 'ogbg_molbace' else 0.5,
    }

    GSAT_config = {
        'method_name': 'GSAT',
        'model_name': model_name,
        'pred_loss_coef': 1,
        'info_loss_coef': 1,
        'motif_loss_coef': 0,
        'between_motif_coef': 0,
        'epochs': hp['epochs'],
        'lr': hp['lr'],
        'from_scratch': True,
        'fix_r': False,
        'decay_interval': 10 if model_name != 'PNA' else 5,
        'decay_r': 0.1,
        'init_r': 0.9,
        'final_r': hp['r'],
        'motif_incorporation_method': None,
        'train_motif_graph': False,
        'separate_motif_model': False,
        'experiment_name': 'gsat_replication',
        'tuning_id': f'{model_name}_{dataset_name}',
    }

    if gsat_overrides:
        GSAT_config = {**GSAT_config, **gsat_overrides}

    config = {
        'data_config': data_config,
        'model_config': model_config,
        'shared_config': shared_config,
        'GSAT_config': GSAT_config,
    }
    return config
