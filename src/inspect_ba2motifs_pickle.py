#!/usr/bin/env python3
"""
Inspect the actual node features in BA-2Motifs pickle file
"""

import pickle
import numpy as np
from pathlib import Path

# Load the pickle file
data_path = Path('../data/ba_2motifs/raw/BA_2Motifs.pkl')

if not data_path.exists():
    print(f"❌ Pickle file not found at: {data_path}")
    print("   Download it first or check the path")
    exit(1)

print("="*80)
print("BA-2MOTIFS PICKLE FILE INSPECTION")
print("="*80)

with open(data_path, 'rb') as f:
    dense_edges, node_features, graph_labels = pickle.load(f)

print(f"\nLoaded data:")
print(f"  dense_edges shape: {dense_edges.shape}")
print(f"  node_features shape: {node_features.shape}")
print(f"  graph_labels shape: {graph_labels.shape}")

# Check node features
print(f"\nNode features inspection:")
print(f"  Dtype: {node_features.dtype}")
print(f"  Mean: {node_features.mean():.6f}")
print(f"  Std: {node_features.std():.6f}")
print(f"  Min: {node_features.min():.6f}")
print(f"  Max: {node_features.max():.6f}")

# Check first graph
print(f"\nFirst graph (index 0):")
print(f"  Features shape: {node_features[0].shape}")
print(f"  Num nodes: {node_features[0].shape[0]}")
print(f"  Feature dim: {node_features[0].shape[1] if len(node_features[0].shape) > 1 else 1}")
print(f"  Sample features (first 5 nodes):")
for i in range(min(5, node_features[0].shape[0])):
    print(f"    Node {i}: {node_features[0][i]}")

# Check if features are constant
unique_vals = np.unique(node_features[0])
print(f"\nUnique values in first graph: {unique_vals}")
if len(unique_vals) == 1:
    print(f"  ❌ CONSTANT FEATURES! All nodes have value {unique_vals[0]}")
    print(f"  This is why GCN cannot learn!")
elif len(unique_vals) < 5:
    print(f"  ⚠️  Very few unique values ({len(unique_vals)})")
else:
    print(f"  ✓ Features have {len(unique_vals)} unique values")

# Check across all graphs
all_unique = np.unique(node_features)
print(f"\nUnique values across ALL graphs: {len(all_unique)} values")
print(f"  Range: [{all_unique.min():.4f}, {all_unique.max():.4f}]")

# Check label distribution
print(f"\nLabel distribution:")
labels = np.where(graph_labels)[1]
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  Class {label}: {count} graphs ({count/len(labels)*100:.1f}%)")

print("\n" + "="*80)
