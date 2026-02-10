#!/usr/bin/env python3
"""
Debug script to check BA-2Motifs data loading
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import SynGraphDataset

# Load BA-2Motifs
data_dir = Path('../data')
dataset = SynGraphDataset(data_dir, 'ba_2motifs')

print("="*80)
print("BA-2MOTIFS DATASET DEBUG")
print("="*80)

print(f"\nDataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")

# Check first 10 samples
print("\nFirst 10 samples:")
for i in range(min(10, len(dataset))):
    data = dataset[i]
    print(f"  Sample {i}:")
    print(f"    Nodes: {data.x.shape[0]}")
    print(f"    Edges: {data.edge_index.shape[1]}")
    print(f"    Features shape: {data.x.shape}")
    print(f"    Label: {data.y.item()} (shape: {data.y.shape}, dtype: {data.y.dtype})")
    print(f"    Node features sample: {data.x[0, :5]}")  # First 5 features of first node

# Check label distribution
all_labels = torch.stack([d.y for d in dataset])
unique, counts = torch.unique(all_labels, return_counts=True)
print(f"\nLabel distribution:")
for label, count in zip(unique, counts):
    print(f"  Class {label.item()}: {count.item()} samples ({count.item()/len(dataset)*100:.1f}%)")

# Check if labels are balanced
if len(unique) == 2:
    ratio = counts[0].item() / counts[1].item()
    print(f"\nClass balance ratio: {ratio:.2f}:1")
    if 0.8 <= ratio <= 1.2:
        print("✓ Classes are balanced")
    else:
        print("⚠️  Classes are imbalanced!")

# Check features
print(f"\nFeature statistics:")
all_features = torch.cat([d.x for d in dataset])
print(f"  Mean: {all_features.mean():.4f}")
print(f"  Std: {all_features.std():.4f}")
print(f"  Min: {all_features.min():.4f}")
print(f"  Max: {all_features.max():.4f}")

# Check if features are constant/zero
if all_features.std() < 0.01:
    print("❌ Features have very low variance - might be constant!")
else:
    print("✓ Features have reasonable variance")

# Check node labels (ground truth for explainability)
print(f"\nGround truth labels:")
sample = dataset[0]
print(f"  Node labels: {sample.node_label[:30]}")  # First 30 nodes
print(f"  Edge labels (first 20): {sample.edge_label[:20]}")

print("\n" + "="*80)
