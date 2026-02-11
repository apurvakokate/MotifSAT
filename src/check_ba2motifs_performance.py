#!/usr/bin/env python3
"""
Check what's causing train performance to be worse than test
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import SynGraphDataset
from utils import get_data_loaders

# Load BA-2Motifs
data_dir = Path('../data')
splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

result = get_data_loaders(data_dir, 'ba_2motifs', 64, splits, 0, False, 0)

if len(result) == 8:
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, _ = result
else:
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = result
    datasets = {
        'train': loaders['train'].dataset,
        'valid': loaders['valid'].dataset,
        'test': loaders['test'].dataset
    }

print("="*80)
print("BA-2MOTIFS DATASET ANALYSIS")
print("="*80)

print(f"\nDataset sizes:")
print(f"  Train: {len(datasets['train'])} graphs")
print(f"  Valid: {len(datasets['valid'])} graphs")
print(f"  Test: {len(datasets['test'])} graphs")

# Check label distribution in each split
for split_name in ['train', 'valid', 'test']:
    labels = torch.stack([d.y for d in datasets[split_name]]).squeeze()
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"\n{split_name.capitalize()} label distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label.item()}: {count.item()} ({count.item()/len(labels)*100:.1f}%)")

# Check feature dimensions
sample = datasets['train'][0]
print(f"\nFeature info:")
print(f"  Node feature dim: {sample.x.shape[1]}")
print(f"  Sample graph nodes: {sample.x.shape[0]}")
print(f"  Sample graph edges: {sample.edge_index.shape[1]}")

# Check if features vary
print(f"\nFeature variance check:")
all_features = torch.cat([d.x for d in datasets['train']])
print(f"  Mean: {all_features.mean():.4f}")
print(f"  Std: {all_features.std():.4f}")
print(f"  Unique values: {len(torch.unique(all_features))}")

# Check graph size distribution
train_sizes = [d.x.shape[0] for d in datasets['train']]
test_sizes = [d.x.shape[0] for d in datasets['test']]
print(f"\nGraph size distribution:")
print(f"  Train avg nodes: {sum(train_sizes)/len(train_sizes):.1f}")
print(f"  Test avg nodes: {sum(test_sizes)/len(test_sizes):.1f}")

print("\n" + "="*80)
