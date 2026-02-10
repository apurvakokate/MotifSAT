#!/usr/bin/env python3
"""
Test: Train GCN on BA-2Motifs WITHOUT GSAT attention.
This should replicate literature results (97-100% accuracy).
"""

import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.loader import DataLoader
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import get_data_loaders, get_model, set_seed
from pretrain_clf import train_clf_one_seed

def test_baseline_gcn():
    print("="*80)
    print("TEST: GCN on BA-2Motifs WITHOUT GSAT")
    print("="*80)
    print("\nThis tests if GCN can achieve 97-100% like literature")
    print("by training WITHOUT the GSAT attention mechanism.\n")
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'ba_2motifs'
    model_name = 'GCN'
    seed = 0
    
    set_seed(seed)
    
    # Configuration matching GOAt paper
    config = {
        'model_config': {
            'model_name': 'GCN',
            'hidden_size': 32,  # GOAt paper uses 32 for BA-2Motifs
            'n_layers': 3,      # Literature uses 3
            'dropout_p': 0.3,
            'pretrain_epochs': 100,
            'pretrain_lr': 1e-3,
            'atom_encoder': False,
            'use_edge_attr': False,
        },
        'data_config': {
            'splits': {'train': 0.8, 'valid': 0.1, 'test': 0.1},
            'batch_size': 128,
            'mutag_x': False,
        },
    }
    
    print(f"Config:")
    print(f"  Model: {model_name}")
    print(f"  Hidden size: {config['model_config']['hidden_size']}")
    print(f"  Layers: {config['model_config']['n_layers']}")
    print(f"  Device: {device}")
    
    # Load data
    print(f"\nLoading {dataset_name}...")
    data_dir = Path('../data')
    batch_size = config['data_config']['batch_size']
    
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, _ = get_data_loaders(
        data_dir, dataset_name, batch_size, None, seed, False, 0
    )
    
    print(f"✓ Dataset loaded")
    print(f"  Train: {len(datasets['train'])} graphs")
    print(f"  Valid: {len(datasets['valid'])} graphs")
    print(f"  Test: {len(datasets['test'])} graphs")
    print(f"  Num classes: {num_class}")
    print(f"  Node feature dim: {x_dim}")
    
    # Check label distribution
    train_labels = torch.stack([d.y for d in datasets['train']])
    print(f"  Train label distribution: {torch.unique(train_labels, return_counts=True)}")
    
    # Build model
    config['model_config']['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], 
                     config['model_config'], device)
    
    print(f"\n✓ Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train WITHOUT GSAT (just standard classifier training)
    print(f"\n" + "="*80)
    print("TRAINING (Standard GCN, NO GSAT attention)")
    print("="*80)
    
    log_dir = data_dir / dataset_name / 'logs' / f'baseline_test_{model_name}_seed{seed}'
    
    hparam_dict, metric_dict = train_clf_one_seed(
        config, data_dir, log_dir, model_name, dataset_name, 
        device, seed,
        model=model, loaders=loaders, num_class=num_class, aux_info=aux_info
    )
    
    # Results
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nBest Epoch: {metric_dict.get('metric/best_clf_epoch', 'N/A')}")
    print(f"\nClassification Accuracy:")
    print(f"  Train: {metric_dict.get('metric/best_clf_train', 0):.4f}")
    print(f"  Valid: {metric_dict.get('metric/best_clf_valid', 0):.4f}")
    print(f"  Test:  {metric_dict.get('metric/best_clf_test', 0):.4f}")
    
    print(f"\nClassification ROC-AUC:")
    print(f"  Train: {metric_dict.get('metric/best_x_roc_train', 0):.4f}")
    print(f"  Valid: {metric_dict.get('metric/best_x_roc_valid', 0):.4f}")
    print(f"  Test:  {metric_dict.get('metric/best_x_roc_test', 0):.4f}")
    
    # Compare with literature
    print(f"\n" + "="*80)
    print("COMPARISON WITH LITERATURE")
    print("="*80)
    
    test_acc = metric_dict.get('metric/best_clf_test', 0)
    
    print(f"\nLiterature (GOAt paper): GCN = 100%")
    print(f"Literature (MAGE paper): GCN = 97%")
    print(f"Your result:            GCN = {test_acc*100:.1f}%")
    
    if test_acc >= 0.95:
        print(f"\n✅ SUCCESS! Matches literature (≥95%)")
        print(f"   → GCN CAN solve BA-2Motifs without GSAT!")
    elif test_acc >= 0.80:
        print(f"\n⚠️  PARTIAL SUCCESS ({test_acc*100:.1f}%)")
        print(f"   → Close but not literature level")
        print(f"   → May need hyperparameter tuning")
    else:
        print(f"\n❌ FAILURE ({test_acc*100:.1f}%)")
        print(f"   → Even baseline GCN not working")
        print(f"   → Check data loading or model implementation")
    
    print(f"\n" + "="*80)
    
    return metric_dict


if __name__ == '__main__':
    test_baseline_gcn()
