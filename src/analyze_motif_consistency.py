#!/usr/bin/env python
"""
Post-hoc Motif Node Score Consistency Analysis

Evaluates whether nodes belonging to the same motif receive consistent
attention scores across different graph examples.

Supports two families of datasets:
  1. Synthetic (BA-2Motifs, SPMotif): ground-truth node_label (0=base, 1=motif),
     grouped by graph label y (motif type).
  2. Molecular (Mutagenicity, BBBP, …): BRICS-decomposed nodes_to_motifs mapping,
     grouped by motif_index. Same motif fragment should score similarly across molecules.

Two data sources:
  A. From checkpoint: load model + extractor, run inference.
  B. From node_scores.jsonl: read pre-computed per-node scores (fast, no GPU needed).

Usage:
    # Synthetic — from checkpoint
    python analyze_motif_consistency.py \
        --checkpoint_dir /path/to/seed_dir \
        --dataset ba_2motifs --model GIN

    # Molecular — from pre-computed JSONL (recommended)
    python analyze_motif_consistency.py \
        --from_jsonl /path/to/seed_dir/node_scores.jsonl \
        --dataset Mutagenicity --model GIN

    # Molecular — from checkpoint (re-extracts scores)
    python analyze_motif_consistency.py \
        --checkpoint_dir /path/to/seed_dir \
        --dataset Mutagenicity --model GIN --fold 0
"""

import os
import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_model, load_checkpoint
from utils.get_data_loaders import get_data_loaders
from experiment_configs import get_base_config, MOL_DATASETS_WITH_FOLDS
from run_gsat import ExtractorMLP

warnings.filterwarnings('ignore', category=UserWarning)

SYNTHETIC_DATASETS = ['ba_2motifs', 'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9']
ALL_DATASETS = SYNTHETIC_DATASETS + MOL_DATASETS_WITH_FOLDS

MOTIF_TYPE_NAMES = {
    'ba_2motifs': {0: 'House', 1: 'Cycle'},
    'spmotif_0.5': {0: 'House', 1: 'Cycle', 2: 'Crane'},
    'spmotif_0.7': {0: 'House', 1: 'Cycle', 2: 'Crane'},
    'spmotif_0.9': {0: 'House', 1: 'Cycle', 2: 'Crane'},
}


# ---------------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------------

def find_best_checkpoint_epoch(checkpoint_dir):
    """Find the latest checkpoint epoch in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    clf_files = sorted(checkpoint_dir.glob('gsat_clf_epoch_*.pt'))
    if not clf_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    epochs = []
    for f in clf_files:
        try:
            epoch = int(f.stem.split('_')[-1])
            att_file = checkpoint_dir / f'gsat_att_epoch_{epoch}.pt'
            if att_file.exists():
                epochs.append(epoch)
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError(f"No matching clf+att checkpoint pairs in {checkpoint_dir}")
    return max(epochs)


def load_trained_model(checkpoint_dir, model_name, dataset_name, device,
                       epoch=None, fold=0):
    """
    Load trained classifier and extractor from checkpoint.

    Reads shared_config.yaml and model_config.yaml saved during training so
    that learn_edge_att, hidden_size, etc. exactly match the checkpoint.
    Falls back to get_base_config() defaults for any file that doesn't exist.
    """
    import yaml

    checkpoint_dir = Path(checkpoint_dir)

    # Start from defaults, then override with configs saved during training
    config = get_base_config(model_name, dataset_name)
    data_config = config['data_config']
    model_config = config['model_config']
    shared_config = config['shared_config']

    # Override shared_config (learn_edge_att, extractor_dropout_p, …)
    saved_shared = checkpoint_dir / 'shared_config.yaml'
    if saved_shared.exists():
        with open(saved_shared, 'r') as f:
            saved = yaml.safe_load(f) or {}
        shared_config.update(saved)
        print(f"[INFO] Loaded shared_config from {saved_shared}")
    else:
        print(f"[WARNING] No shared_config.yaml in {checkpoint_dir}, using defaults")

    # Override model_config (hidden_size, n_layers, gcn_normalize, …)
    saved_model_cfg = checkpoint_dir / 'model_config.yaml'
    if saved_model_cfg.exists():
        with open(saved_model_cfg, 'r') as f:
            saved = yaml.safe_load(f) or {}
        model_config.update(saved)
        print(f"[INFO] Loaded model_config from {saved_model_cfg}")
    else:
        print(f"[WARNING] No model_config.yaml in {checkpoint_dir}, using defaults")

    config_dir = Path('./configs')
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    data_dir = Path(global_config['data_dir'])

    batch_size = data_config['batch_size']
    splits = data_config.get('splits', None)

    is_mol = dataset_name in MOL_DATASETS_WITH_FOLDS
    loader_result = get_data_loaders(
        data_dir, dataset_name, batch_size, splits,
        random_state=0,
        mutag_x=data_config.get('mutag_x', False),
        fold=fold if is_mol else None,
    )

    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = loader_result[:6]

    model_config['deg'] = aux_info['deg']
    clf = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)

    if epoch is None:
        epoch = find_best_checkpoint_epoch(checkpoint_dir)
    print(f"[INFO] Loading checkpoint epoch {epoch} from {checkpoint_dir}")
    print(f"[INFO] ExtractorMLP: learn_edge_att={shared_config['learn_edge_att']}, "
          f"hidden_size={model_config['hidden_size']}")

    load_checkpoint(clf, checkpoint_dir, f'gsat_clf_epoch_{epoch}')
    load_checkpoint(extractor, checkpoint_dir, f'gsat_att_epoch_{epoch}')

    clf.eval()
    extractor.eval()

    learn_edge_att = shared_config['learn_edge_att']
    return clf, extractor, loaders, test_set, learn_edge_att, config


# ---------------------------------------------------------------------------
# Attention extraction — from model inference
# ---------------------------------------------------------------------------

def extract_node_attention(clf, extractor, data_loader, device, learn_edge_att,
                           dataset_name):
    """
    Run inference and extract per-graph node attention scores + metadata.

    For synthetic datasets: returns node_label (0/1).
    For molecular datasets: returns nodes_to_motifs (motif ID per node).
    """
    results = []
    graph_idx = 0
    is_mol = dataset_name in MOL_DATASETS_WITH_FOLDS

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            edge_attr = getattr(batch_data, 'edge_attr', None)

            emb = clf.get_emb(batch_data.x, batch_data.edge_index,
                              batch_data.batch, edge_attr=edge_attr)
            att_log_logits = extractor(emb, batch_data.edge_index, batch_data.batch)
            att = att_log_logits.sigmoid().squeeze()

            if learn_edge_att:
                src, dst = batch_data.edge_index
                node_att = torch.zeros(batch_data.num_nodes, device=device)
                node_count = torch.zeros(batch_data.num_nodes, device=device)
                node_att.scatter_add_(0, src, att)
                node_att.scatter_add_(0, dst, att)
                node_count.scatter_add_(0, src, torch.ones_like(att))
                node_count.scatter_add_(0, dst, torch.ones_like(att))
                node_att = node_att / node_count.clamp(min=1)
            else:
                node_att = att

            node_att_np = node_att.cpu().numpy()
            batch_np = batch_data.batch.cpu().numpy()
            y_np = batch_data.y.cpu().numpy().flatten()

            node_label_np = (batch_data.node_label.cpu().numpy()
                             if hasattr(batch_data, 'node_label') and batch_data.node_label is not None
                             else None)
            nodes_to_motifs_np = (batch_data.nodes_to_motifs.cpu().numpy()
                                  if hasattr(batch_data, 'nodes_to_motifs') and batch_data.nodes_to_motifs is not None
                                  else None)
            smiles_list = None
            if hasattr(batch_data, 'smiles'):
                smiles_list = batch_data.smiles

            unique_graphs = np.unique(batch_np)
            for g in unique_graphs:
                mask = batch_np == g
                r = {
                    'graph_idx': graph_idx,
                    'y': int(y_np[g]) if g < len(y_np) else -1,
                    'node_att': node_att_np[mask],
                    'num_nodes': int(mask.sum()),
                }
                if node_label_np is not None:
                    r['node_label'] = node_label_np[mask]
                if nodes_to_motifs_np is not None:
                    r['nodes_to_motifs'] = nodes_to_motifs_np[mask]
                if smiles_list is not None:
                    r['smiles'] = smiles_list[g] if isinstance(smiles_list, (list, tuple)) else None
                results.append(r)
                graph_idx += 1

    return results


# ---------------------------------------------------------------------------
# Loading from pre-computed node_scores.jsonl
# ---------------------------------------------------------------------------

def load_from_jsonl(jsonl_path, split='test'):
    """
    Load per-node scores from node_scores.jsonl.

    Returns list of dicts grouped by graph:
        {
            'graph_idx': int,
            'smiles': str,
            'nodes_to_motifs': np.ndarray,  # motif_index per node
            'node_att': np.ndarray,          # score per node
        }
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"node_scores.jsonl not found at {jsonl_path}")

    by_graph = defaultdict(lambda: {'node_indices': [], 'motif_indices': [], 'scores': [], 'smiles': None})

    with open(jsonl_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('split') != split:
                continue
            gidx = rec['graph_idx']
            by_graph[gidx]['node_indices'].append(rec['node_index'])
            by_graph[gidx]['motif_indices'].append(rec.get('motif_index', -1))
            by_graph[gidx]['scores'].append(rec['score'])
            if by_graph[gidx]['smiles'] is None:
                by_graph[gidx]['smiles'] = rec.get('smiles')

    results = []
    for gidx in sorted(by_graph.keys()):
        g = by_graph[gidx]
        order = np.argsort(g['node_indices'])
        results.append({
            'graph_idx': gidx,
            'smiles': g['smiles'],
            'nodes_to_motifs': np.array(g['motif_indices'])[order],
            'node_att': np.array(g['scores'])[order],
        })

    print(f"[INFO] Loaded {len(results)} graphs from {jsonl_path} (split={split})")
    return results


# ---------------------------------------------------------------------------
# Metrics — synthetic datasets (BA-2Motifs, SPMotif)
# ---------------------------------------------------------------------------

def compute_synthetic_consistency(graph_results, dataset_name):
    """
    Consistency metrics for synthetic datasets with binary node_label.
    Groups graphs by motif type (graph label y).
    """
    motif_names = MOTIF_TYPE_NAMES.get(dataset_name, {})
    by_type = defaultdict(list)
    for r in graph_results:
        by_type[r['y']].append(r)

    rows = []
    detailed = {}

    for motif_type, graphs in sorted(by_type.items()):
        type_name = motif_names.get(motif_type, f'Type {motif_type}')

        motif_means, nonmotif_means = [], []
        motif_stds_within = []
        all_motif_scores = []
        motif_vectors = []

        for g in graphs:
            nl = g.get('node_label')
            if nl is None:
                continue
            m_mask = nl > 0
            nm_mask = nl == 0
            m_scores = g['node_att'][m_mask]
            nm_scores = g['node_att'][nm_mask]
            if len(m_scores) == 0:
                continue

            motif_means.append(float(m_scores.mean()))
            motif_stds_within.append(float(m_scores.std()) if len(m_scores) > 1 else 0.0)
            all_motif_scores.extend(m_scores.tolist())
            motif_vectors.append(m_scores)
            if len(nm_scores) > 0:
                nonmotif_means.append(float(nm_scores.mean()))

        motif_means = np.array(motif_means)
        nonmotif_means = np.array(nonmotif_means)
        motif_stds_within = np.array(motif_stds_within)
        all_motif_scores = np.array(all_motif_scores)

        cross_mean = float(motif_means.mean()) if len(motif_means) else np.nan
        cross_std = float(motif_means.std()) if len(motif_means) > 1 else np.nan
        cv = cross_std / cross_mean if cross_mean > 1e-8 else np.nan
        avg_within = float(motif_stds_within.mean()) if len(motif_stds_within) else np.nan
        gap = float(motif_means.mean() - nonmotif_means.mean()) if len(nonmotif_means) else np.nan

        cohens_d = np.nan
        if len(motif_means) > 1 and len(nonmotif_means) > 1:
            ps = np.sqrt((motif_means.var() + nonmotif_means.var()) / 2)
            if ps > 1e-8:
                cohens_d = float((motif_means.mean() - nonmotif_means.mean()) / ps)

        cosine_mean, cosine_std = np.nan, np.nan
        sizes = [len(v) for v in motif_vectors]
        if len(set(sizes)) == 1 and len(motif_vectors) >= 2:
            vecs = np.stack(motif_vectors)
            sims = []
            idx = np.random.choice(len(vecs), size=min(500, len(vecs)), replace=False)
            for i, j in combinations(idx, 2):
                d = np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
                if d > 1e-8:
                    sims.append(float(np.dot(vecs[i], vecs[j]) / d))
            if sims:
                cosine_mean, cosine_std = float(np.mean(sims)), float(np.std(sims))

        icc = np.nan
        if len(set(sizes)) == 1 and len(motif_vectors) >= 3:
            mat = np.stack(motif_vectors)
            ng, nn = mat.shape
            if nn >= 2:
                gm = mat.mean()
                rm = mat.mean(axis=1)
                cm = mat.mean(axis=0)
                ss_r = nn * ((rm - gm) ** 2).sum()
                ss_c = ng * ((cm - gm) ** 2).sum()
                ss_t = ((mat - gm) ** 2).sum()
                ss_e = ss_t - ss_r - ss_c
                ms_r = ss_r / (ng - 1)
                ms_e = ss_e / ((ng - 1) * (nn - 1)) if ng > 1 and nn > 1 else 0
                denom = ms_r + (nn - 1) * ms_e
                icc = float((ms_r - ms_e) / denom) if denom > 1e-8 else np.nan

        rows.append({
            'Motif Type': type_name,
            'N Graphs': len(graphs),
            'Mean Motif Att': round(cross_mean, 4),
            'Cross-Graph Std': round(cross_std, 4),
            'CV': round(cv, 4),
            'Avg Within-Graph Std': round(avg_within, 4),
            'Mean Non-Motif Att': round(float(nonmotif_means.mean()), 4) if len(nonmotif_means) else np.nan,
            'Gap': round(gap, 4),
            "Cohen's d": round(cohens_d, 4),
            'Cosine Sim': round(cosine_mean, 4),
            'ICC': round(icc, 4),
        })

        detailed[motif_type] = {
            'type_name': type_name,
            'motif_means': motif_means,
            'nonmotif_means': nonmotif_means,
            'motif_stds_within': motif_stds_within,
            'all_motif_scores': all_motif_scores,
            'motif_vectors': motif_vectors,
        }

    return pd.DataFrame(rows), detailed


# ---------------------------------------------------------------------------
# Metrics — molecular datasets (Mutagenicity, BBBP, …)
# ---------------------------------------------------------------------------

def compute_mol_consistency(graph_results, dataset_name, min_occurrences=3,
                            top_k=30):
    """
    Consistency metrics for molecular datasets with nodes_to_motifs.

    Groups all nodes by their motif_index across all graphs.
    For each motif type that appears in ≥ min_occurrences graphs, computes:
      - Per-instance mean attention (one value per graph that contains this motif)
      - Cross-graph std / CV of that per-instance mean
      - Within-instance std (avg std of attention within one occurrence)
    """
    # Collect: motif_id -> list of (graph_idx, [scores], smiles)
    motif_instances = defaultdict(list)
    unmapped_scores = []

    for g in graph_results:
        n2m = g.get('nodes_to_motifs')
        if n2m is None:
            continue
        att = g['node_att']
        for mid in np.unique(n2m):
            mask = n2m == mid
            scores = att[mask]
            if mid < 0:
                unmapped_scores.extend(scores.tolist())
                continue
            motif_instances[int(mid)].append({
                'graph_idx': g['graph_idx'],
                'scores': scores,
                'smiles': g.get('smiles'),
            })

    # Per-motif statistics
    rows = []
    per_motif_detail = {}

    for mid in sorted(motif_instances.keys()):
        instances = motif_instances[mid]
        n_graphs = len(instances)
        if n_graphs < min_occurrences:
            continue

        instance_means = np.array([inst['scores'].mean() for inst in instances])
        instance_stds = np.array([inst['scores'].std() if len(inst['scores']) > 1 else 0.0
                                  for inst in instances])
        all_scores = np.concatenate([inst['scores'] for inst in instances])
        total_nodes = sum(len(inst['scores']) for inst in instances)

        cross_mean = float(instance_means.mean())
        cross_std = float(instance_means.std()) if n_graphs > 1 else 0.0
        cv = cross_std / cross_mean if cross_mean > 1e-8 else np.nan
        avg_within_std = float(instance_stds.mean())

        rows.append({
            'Motif ID': mid,
            'N Graphs': n_graphs,
            'Total Nodes': total_nodes,
            'Mean Att': round(cross_mean, 4),
            'Cross-Graph Std': round(cross_std, 4),
            'CV': round(cv, 4),
            'Avg Within-Instance Std': round(avg_within_std, 4),
            'Min Instance Mean': round(float(instance_means.min()), 4),
            'Max Instance Mean': round(float(instance_means.max()), 4),
        })

        per_motif_detail[mid] = {
            'instance_means': instance_means,
            'instance_stds': instance_stds,
            'all_scores': all_scores,
            'n_graphs': n_graphs,
        }

    detail_df = pd.DataFrame(rows)
    if len(detail_df) == 0:
        print("[WARNING] No motifs found with sufficient occurrences.")
        return pd.DataFrame(), pd.DataFrame(), per_motif_detail, unmapped_scores

    detail_df = detail_df.sort_values('N Graphs', ascending=False)

    # Global summary table
    all_cvs = detail_df['CV'].dropna()
    all_cross_stds = detail_df['Cross-Graph Std'].dropna()
    all_within_stds = detail_df['Avg Within-Instance Std'].dropna()
    unmapped_mean = float(np.mean(unmapped_scores)) if unmapped_scores else np.nan

    summary_rows = [{
        'Dataset': dataset_name,
        'N Motif Types': len(detail_df),
        'N Total Graphs': len(graph_results),
        'Avg Cross-Graph Std': round(float(all_cross_stds.mean()), 4),
        'Median CV': round(float(all_cvs.median()), 4),
        'Avg Within-Instance Std': round(float(all_within_stds.mean()), 4),
        'Mean Motif Att (all)': round(float(detail_df['Mean Att'].mean()), 4),
        'Mean Unmapped Att': round(unmapped_mean, 4),
        f'Top-{top_k} Most Consistent (avg CV)': round(
            float(detail_df.nsmallest(min(top_k, len(detail_df)), 'CV')['CV'].mean()), 4
        ),
    }]
    summary_df = pd.DataFrame(summary_rows)

    return summary_df, detail_df, per_motif_detail, unmapped_scores


# ---------------------------------------------------------------------------
# Plotting — synthetic datasets
# ---------------------------------------------------------------------------

def plot_synthetic_consistency(detailed, dataset_name, model_name, output_dir, split_name):
    """Plots for synthetic datasets (BA-2Motifs, SPMotif)."""
    output_dir = Path(output_dir)
    n_types = len(detailed)
    if n_types == 0:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    tag = f'{model_name}_{dataset_name}_{split_name}'

    # -- Figure 1: Violin + Motif vs Non-Motif box plot --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    violin_data, violin_labels = [], []
    for mt in sorted(detailed.keys()):
        d = detailed[mt]
        violin_data.append(d['all_motif_scores'])
        violin_labels.append(d['type_name'])

    parts = axes[0].violinplot(violin_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('red')
    axes[0].set_xticks(range(1, len(violin_labels) + 1))
    axes[0].set_xticklabels(violin_labels)
    axes[0].set_ylabel('Node Attention Score')
    axes[0].set_title('Motif Node Scores (all nodes, all graphs)')

    box_m, box_nm, box_labels = [], [], []
    for mt in sorted(detailed.keys()):
        d = detailed[mt]
        box_m.append(d['motif_means'])
        box_nm.append(d['nonmotif_means'])
        box_labels.append(d['type_name'])

    pos_m = np.arange(1, n_types + 1) - 0.15
    pos_nm = np.arange(1, n_types + 1) + 0.15
    bp1 = axes[1].boxplot(box_m, positions=pos_m, widths=0.25, patch_artist=True, showfliers=False)
    bp2 = axes[1].boxplot(box_nm, positions=pos_nm, widths=0.25, patch_artist=True, showfliers=False)
    for p in bp1['boxes']:
        p.set_facecolor('#2196F3'); p.set_alpha(0.7)
    for p in bp2['boxes']:
        p.set_facecolor('#BDBDBD'); p.set_alpha(0.7)
    axes[1].set_xticks(range(1, n_types + 1))
    axes[1].set_xticklabels(box_labels)
    axes[1].set_ylabel('Mean Attention per Graph')
    axes[1].set_title('Motif (blue) vs Non-Motif (gray)')
    axes[1].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Motif', 'Non-Motif'], loc='upper right')

    fig.suptitle(f'{model_name} on {dataset_name} ({split_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'synth_distributions_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -- Figure 2: Cross-graph histogram --
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), squeeze=False)
    axes = axes[0]
    for i, mt in enumerate(sorted(detailed.keys())):
        d = detailed[mt]
        axes[i].hist(d['motif_means'], bins=30, color=colors[i % len(colors)],
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[i].axvline(d['motif_means'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f"mean={d['motif_means'].mean():.3f}")
        axes[i].set_xlabel('Mean Motif Att per Graph')
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{d["type_name"]}')
        axes[i].legend(fontsize=9)
    fig.suptitle(f'{model_name} on {dataset_name} ({split_name}) — Cross-Graph Consistency',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'synth_cross_graph_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -- Figure 3: Heatmap (fixed-size motifs) --
    all_fixed = all(
        len(set(len(v) for v in d['motif_vectors'])) == 1 and d['motif_vectors']
        for d in detailed.values()
    )
    if all_fixed:
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5), squeeze=False)
        axes = axes[0]
        for i, mt in enumerate(sorted(detailed.keys())):
            d = detailed[mt]
            if not d['motif_vectors']:
                continue
            mat = np.stack(d['motif_vectors'])
            n_show = min(100, mat.shape[0])
            idx = np.random.choice(mat.shape[0], n_show, replace=False)
            im = axes[i].imshow(mat[idx], aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            axes[i].set_xlabel('Motif Node Position')
            axes[i].set_ylabel(f'Graphs (n={n_show})')
            axes[i].set_title(d['type_name'])
            plt.colorbar(im, ax=axes[i], label='Attention')
        fig.suptitle(f'{model_name} on {dataset_name} ({split_name}) — Positional Heatmap',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'synth_heatmap_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # -- Figure 4: Within-graph scatter --
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, mt in enumerate(sorted(detailed.keys())):
        d = detailed[mt]
        ax.scatter(d['motif_means'], d['motif_stds_within'], alpha=0.4, s=15,
                   color=colors[i % len(colors)], label=d['type_name'])
    ax.set_xlabel('Mean Motif Attention')
    ax.set_ylabel('Within-Graph Std')
    ax.set_title(f'{model_name} on {dataset_name} ({split_name}) — Within-Graph Variability')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f'synth_within_scatter_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Synthetic plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Plotting — molecular datasets
# ---------------------------------------------------------------------------

def plot_mol_consistency(summary_df, detail_df, per_motif_detail, unmapped_scores,
                         dataset_name, model_name, output_dir, split_name,
                         top_k=20):
    """Plots for molecular datasets grouped by BRICS motif_index."""
    output_dir = Path(output_dir)
    if detail_df.empty:
        print("[INFO] No motif data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    tag = f'{model_name}_{dataset_name}_{split_name}'

    # -- Figure 1: CV distribution across motif types --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cvs = detail_df['CV'].dropna()
    axes[0].hist(cvs, bins=40, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(cvs.median(), color='red', linestyle='--', linewidth=2,
                    label=f'median={cvs.median():.3f}')
    axes[0].set_xlabel('Coefficient of Variation (CV)')
    axes[0].set_ylabel('Number of Motif Types')
    axes[0].set_title('CV Distribution Across Motif Types')
    axes[0].legend()

    cross_stds = detail_df['Cross-Graph Std'].dropna()
    axes[1].hist(cross_stds, bins=40, color='#FF5722', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].axvline(cross_stds.median(), color='red', linestyle='--', linewidth=2,
                    label=f'median={cross_stds.median():.3f}')
    axes[1].set_xlabel('Cross-Graph Std')
    axes[1].set_ylabel('Number of Motif Types')
    axes[1].set_title('Cross-Graph Std Distribution')
    axes[1].legend()

    within_stds = detail_df['Avg Within-Instance Std'].dropna()
    axes[2].hist(within_stds, bins=40, color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[2].axvline(within_stds.median(), color='red', linestyle='--', linewidth=2,
                    label=f'median={within_stds.median():.3f}')
    axes[2].set_xlabel('Avg Within-Instance Std')
    axes[2].set_ylabel('Number of Motif Types')
    axes[2].set_title('Within-Instance Std Distribution')
    axes[2].legend()

    fig.suptitle(f'{model_name} on {dataset_name} ({split_name}) — Motif Consistency Overview',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'mol_consistency_overview_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -- Figure 2: Top-K most/least consistent motifs (bar chart) --
    k = min(top_k, len(detail_df))
    most_consistent = detail_df.nsmallest(k, 'CV')
    least_consistent = detail_df.nlargest(k, 'CV')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(range(k), most_consistent['CV'].values, color='#4CAF50', alpha=0.8)
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([f"Motif {mid}" for mid in most_consistent['Motif ID'].values], fontsize=8)
    axes[0].set_xlabel('CV (lower = more consistent)')
    axes[0].set_title(f'Top-{k} Most Consistent Motifs')
    axes[0].invert_yaxis()

    axes[1].barh(range(k), least_consistent['CV'].values, color='#FF5722', alpha=0.8)
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([f"Motif {mid}" for mid in least_consistent['Motif ID'].values], fontsize=8)
    axes[1].set_xlabel('CV (higher = less consistent)')
    axes[1].set_title(f'Top-{k} Least Consistent Motifs')
    axes[1].invert_yaxis()

    fig.suptitle(f'{model_name} on {dataset_name} ({split_name}) — Most/Least Consistent Motifs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'mol_top_bottom_motifs_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -- Figure 3: Mean attention vs CV scatter --
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(detail_df['Mean Att'], detail_df['CV'],
                         s=detail_df['N Graphs'].clip(upper=200) * 0.5,
                         c=detail_df['N Graphs'], cmap='viridis', alpha=0.6, edgecolors='gray', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='N Graphs containing motif')
    ax.set_xlabel('Mean Attention Score')
    ax.set_ylabel('CV (Cross-Graph Variability)')
    ax.set_title(f'{model_name} on {dataset_name} ({split_name}) — Mean Att vs Consistency')
    plt.tight_layout()
    fig.savefig(output_dir / f'mol_mean_vs_cv_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -- Figure 4: Box plot of per-instance means for top frequent motifs --
    frequent = detail_df.nlargest(min(15, len(detail_df)), 'N Graphs')
    if len(frequent) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        bp_data = []
        bp_labels = []
        for _, row in frequent.iterrows():
            mid = row['Motif ID']
            if mid in per_motif_detail:
                bp_data.append(per_motif_detail[mid]['instance_means'])
                bp_labels.append(f"M{mid}\n(n={int(row['N Graphs'])})")

        if bp_data:
            bp = ax.boxplot(bp_data, patch_artist=True, showfliers=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(plt.cm.tab20(i / len(bp['boxes'])))
                patch.set_alpha(0.7)
            ax.set_xticklabels(bp_labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylabel('Per-Graph Mean Attention')
            ax.set_title(f'{model_name} on {dataset_name} ({split_name}) — '
                         f'Top-{len(bp_data)} Most Frequent Motifs')
            plt.tight_layout()
            fig.savefig(output_dir / f'mol_frequent_boxplot_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"[INFO] Molecular plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main analysis entry points
# ---------------------------------------------------------------------------

def run_analysis(checkpoint_dir=None, from_jsonl=None, dataset_name=None,
                 model_name=None, output_dir='../motif_consistency_results',
                 epoch=None, seed=0, split='test', device=None, fold=0,
                 min_occurrences=3, top_k=30):
    """
    Run the full motif consistency analysis pipeline.

    Exactly one of checkpoint_dir or from_jsonl must be provided.
    """
    if checkpoint_dir is None and from_jsonl is None:
        raise ValueError("Provide either --checkpoint_dir or --from_jsonl")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_mol = dataset_name in MOL_DATASETS_WITH_FOLDS
    is_synth = dataset_name in SYNTHETIC_DATASETS

    print(f"[INFO] Dataset: {dataset_name} ({'molecular' if is_mol else 'synthetic'})")
    print(f"[INFO] Model: {model_name}, Split: {split}")

    # ── Load data ──
    if from_jsonl is not None:
        graph_results = load_from_jsonl(from_jsonl, split=split)
        source_label = f"JSONL: {from_jsonl}"
    else:
        print(f"[INFO] Loading model from checkpoint: {checkpoint_dir}")
        clf, extractor, loaders, test_set, learn_edge_att, config = load_trained_model(
            checkpoint_dir, model_name, dataset_name, device, epoch, fold
        )
        if split not in loaders:
            raise ValueError(f"Split '{split}' not found. Available: {list(loaders.keys())}")
        print(f"[INFO] Extracting node attention on '{split}' set...")
        graph_results = extract_node_attention(
            clf, extractor, loaders[split], device, learn_edge_att, dataset_name
        )
        source_label = f"Checkpoint: {checkpoint_dir}"

    print(f"[INFO] {len(graph_results)} graphs loaded")

    # ── Route to appropriate analysis ──
    if is_synth:
        has_labels = any(r.get('node_label') is not None for r in graph_results)
        if not has_labels:
            print("[ERROR] Synthetic dataset but no node_label found in data.")
            return None, None

        summary_df, detailed = compute_synthetic_consistency(graph_results, dataset_name)

        print("\n" + "=" * 100)
        print(f"MOTIF NODE SCORE CONSISTENCY — {model_name} on {dataset_name} ({split})")
        print(f"Source: {source_label}")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100)

        csv_path = output_dir / f'consistency_table_{model_name}_{dataset_name}_{split}.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"[INFO] Table saved to {csv_path}")

        plot_synthetic_consistency(detailed, dataset_name, model_name, output_dir, split)

        raw_path = output_dir / f'consistency_raw_{model_name}_{dataset_name}_{split}.json'
        raw = {str(k): {'type_name': v['type_name'],
                        'motif_means': v['motif_means'].tolist(),
                        'nonmotif_means': v['nonmotif_means'].tolist()}
               for k, v in detailed.items()}
        with open(raw_path, 'w') as f:
            json.dump(raw, f, indent=2)

        return summary_df, detailed

    elif is_mol:
        has_motifs = any(r.get('nodes_to_motifs') is not None for r in graph_results)
        if not has_motifs:
            print("[ERROR] Molecular dataset but no nodes_to_motifs found. "
                  "Use --from_jsonl with node_scores.jsonl, or ensure the data loader "
                  "provides nodes_to_motifs (requires fold and lookup).")
            return None, None

        summary_df, detail_df, per_motif_detail, unmapped = compute_mol_consistency(
            graph_results, dataset_name,
            min_occurrences=min_occurrences, top_k=top_k,
        )

        print("\n" + "=" * 100)
        print(f"MOTIF CONSISTENCY SUMMARY — {model_name} on {dataset_name} ({split})")
        print(f"Source: {source_label}")
        print("=" * 100)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
        print("=" * 100)

        print(f"\nPER-MOTIF DETAIL (top 30 by frequency):")
        print("-" * 100)
        if not detail_df.empty:
            print(detail_df.head(30).to_string(index=False))
        print("=" * 100)

        # Save tables
        summary_csv = output_dir / f'mol_summary_{model_name}_{dataset_name}_{split}.csv'
        detail_csv = output_dir / f'mol_detail_{model_name}_{dataset_name}_{split}.csv'
        summary_df.to_csv(summary_csv, index=False)
        detail_df.to_csv(detail_csv, index=False)
        print(f"[INFO] Summary saved to {summary_csv}")
        print(f"[INFO] Detail saved to {detail_csv}")

        plot_mol_consistency(summary_df, detail_df, per_motif_detail, unmapped,
                             dataset_name, model_name, output_dir, split, top_k)

        return summary_df, detail_df

    else:
        print(f"[ERROR] Unknown dataset type: {dataset_name}")
        return None, None

    # ── Score-vs-impact plot (runs whenever seed_dir data is available) ──
    seed_dir = None
    if from_jsonl:
        seed_dir = Path(from_jsonl).parent
    elif checkpoint_dir:
        seed_dir = Path(checkpoint_dir)
    if seed_dir and seed_dir.exists():
        try:
            plot_score_vs_impact(seed_dir, split=split, output_dir=output_dir,
                                model_name=model_name, dataset_name=dataset_name)
        except Exception as e:
            print(f"[WARNING] Score-vs-impact plot failed: {e}")


def run_multi_model_analysis(checkpoint_dirs=None, jsonl_paths=None,
                             dataset_name=None, model_names=None,
                             output_dir='../motif_consistency_results',
                             epoch=None, seed=0, split='test', device=None,
                             fold=0, min_occurrences=3, top_k=30):
    """
    Run consistency analysis for multiple models. Produces combined tables.

    Provide either checkpoint_dirs (dict model->path) or jsonl_paths (dict model->path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for model_name in model_names:
        ckpt = (checkpoint_dirs or {}).get(model_name)
        jpath = (jsonl_paths or {}).get(model_name)

        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}...")
        print(f"{'='*60}")
        try:
            result = run_analysis(
                checkpoint_dir=ckpt, from_jsonl=jpath,
                dataset_name=dataset_name, model_name=model_name,
                output_dir=output_dir, epoch=epoch, seed=seed, split=split,
                device=device, fold=fold, min_occurrences=min_occurrences,
                top_k=top_k,
            )
            if result and result[0] is not None:
                df = result[0].copy()
                df.insert(0, 'Model', model_name)
                all_summaries.append(df)
        except Exception as e:
            print(f"[ERROR] Failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        print("\n" + "=" * 120)
        print(f"COMBINED CONSISTENCY — {dataset_name} ({split})")
        print("=" * 120)
        print(combined.to_string(index=False))
        print("=" * 120)

        csv_path = output_dir / f'consistency_combined_{dataset_name}_{split}.csv'
        combined.to_csv(csv_path, index=False)
        print(f"[INFO] Combined table saved to {csv_path}")


# ---------------------------------------------------------------------------
# Score-vs-Impact explainer analysis (with r-value highlight)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _read_jsonl(path, split='test'):
    records = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('split') == split:
                records.append(rec)
    return records


def _get_final_r(seed_dir):
    """Read final_r from saved configs in seed_dir."""
    import yaml
    seed_dir = Path(seed_dir)

    # Try method_config.yaml first
    mc = seed_dir / 'method_config.yaml'
    if mc.exists():
        with open(mc) as f:
            cfg = yaml.safe_load(f) or {}
        if 'final_r' in cfg:
            return float(cfg['final_r'])

    # Try experiment_summary.json
    es = seed_dir / 'experiment_summary.json'
    if es.exists():
        with open(es) as f:
            summary = json.load(f)
        wdp = summary.get('weight_distribution_params', {})
        if 'final_r' in wdp:
            return float(wdp['final_r'])

    return None


def get_motif_level_score_impact_points(seed_dir, split='test'):
    """
    Motif-level scatter data: x = mean node attention score per (graph, motif instance),
    y = |Δ sigmoid(pred)| from motif-level masking. Same pairing as the motif branch of
    plot_score_vs_impact and consistent with compute_posthoc_correlation's per-motif story.

    Returns:
        (xs, ys) as 1d float arrays, or (None, None) if files missing or no overlap.
    """
    from collections import defaultdict

    seed_dir = Path(seed_dir)
    node_scores_path = seed_dir / 'node_scores.jsonl'
    motif_impact_path = seed_dir / 'Motif_level_node_and_edge_masking_impact.jsonl'
    if not motif_impact_path.exists():
        motif_impact_path = seed_dir / 'masked-edge-impact.jsonl'
    if not node_scores_path.exists() or not motif_impact_path.exists():
        return None, None

    node_recs = _read_jsonl(node_scores_path, split)
    impact_recs = _read_jsonl(motif_impact_path, split)

    motif_scores = defaultdict(list)
    for rec in node_recs:
        midx = rec.get('motif_index', rec.get('motif_idx', -1))
        if midx is None or midx < 0:
            continue
        motif_scores[(rec['graph_idx'], midx)].append(float(rec['score']))

    if not motif_scores:
        return None, None

    motif_avg_score = {k: float(np.mean(v)) for k, v in motif_scores.items()}
    motif_impacts = {}
    for rec in impact_recs:
        midx = rec.get('motif_idx', rec.get('motif_index', -1))
        if midx is None or midx < 0:
            continue
        imp = abs(_sigmoid(rec['new_prediction']) - _sigmoid(rec['old_prediction']))
        motif_impacts[(rec['graph_idx'], midx)] = imp

    common_keys = set(motif_avg_score.keys()) & set(motif_impacts.keys())
    if not common_keys:
        return None, None
    xs = np.array([motif_avg_score[k] for k in common_keys], dtype=float)
    ys = np.array([motif_impacts[k] for k in common_keys], dtype=float)
    return xs, ys


def plot_score_vs_impact(seed_dir, split='test', output_dir=None, model_name=None,
                         dataset_name=None):
    """
    Plot node/motif attention score (x) vs masking impact (y) with r-value
    highlighted as a vertical line.

    Reads from the seed_dir:
      - node_scores.jsonl                                    (per-node scores with motif_index)
      - Motif_level_node_and_edge_masking_impact.jsonl       (per-motif: zero features + remove edges)
      - Individual_node_node_masking_impact.jsonl             (per-node: zero features + remove edges)
      - Individual_edge_node_and_edge_masking_impact.jsonl    (per-edge: remove one undirected edge)
      - method_config.yaml                                   (to read final_r)

    Generates two scatter plots:
      1. Motif-level: x = mean node score per motif instance, y = motif masking impact
      2. Node-level:  x = individual node score, y = node masking impact
    """
    seed_dir = Path(seed_dir)
    output_dir = Path(output_dir) if output_dir else seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label = f'{model_name or ""}{"/" if model_name and dataset_name else ""}{dataset_name or ""}'
    tag = f'{model_name or "model"}_{dataset_name or "data"}_{split}'

    final_r = _get_final_r(seed_dir)
    print(f"[INFO] final_r = {final_r}")

    plt.style.use('seaborn-v0_8-whitegrid')

    # ── Motif-level plot ──
    node_scores_path = seed_dir / 'node_scores.jsonl'
    motif_impact_path = seed_dir / 'Motif_level_node_and_edge_masking_impact.jsonl'
    if not motif_impact_path.exists():
        motif_impact_path = seed_dir / 'masked-edge-impact.jsonl'  # legacy fallback

    has_motif = node_scores_path.exists() and motif_impact_path.exists()
    if has_motif:
        xs, ys = get_motif_level_score_impact_points(seed_dir, split)
        if xs is not None and ys is not None and len(xs):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(xs, ys, alpha=0.3, s=8, c='#2196F3', edgecolors='none')
            if final_r is not None:
                ax.axvline(final_r, color='red', linestyle='--', linewidth=2,
                           label=f'r = {final_r}')
            ax.set_xlabel('Mean Motif Node Attention Score')
            ax.set_ylabel('Motif Masking Impact  |Δ sigmoid(pred)|')
            ax.set_title(f'Motif-Level Score vs Impact — {label} ({split})')
            ax.legend(fontsize=11)
            plt.tight_layout()
            fig.savefig(output_dir / f'score_vs_impact_motif_{tag}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            corr, pval = sp_stats.pearsonr(xs, ys) if len(xs) > 2 else (np.nan, np.nan)
            print(f"[INFO] Motif-level: {len(xs)} points, Pearson r={corr:.4f}, p={pval:.2e}")
        else:
            print("[WARNING] No matching motif entries between node_scores and impact")
    else:
        print(f"[INFO] Skipping motif-level plot (missing files in {seed_dir})")

    # ── Individual node plot (per-node: zero features + remove edges) ──
    indiv_node_path = seed_dir / 'Individual_node_node_masking_impact.jsonl'
    if not indiv_node_path.exists():
        indiv_node_path = seed_dir / 'masked-node-impact.jsonl'  # legacy fallback
    indiv_node_data = None
    if indiv_node_path.exists():
        recs = _read_jsonl(indiv_node_path, split)
        if recs:
            xs = np.array([rec['score'] for rec in recs])
            ys = np.array([abs(_sigmoid(rec['new_prediction']) - _sigmoid(rec['old_prediction']))
                           for rec in recs])
            indiv_node_data = (xs, ys)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(xs, ys, alpha=0.15, s=4, c='#FF5722', edgecolors='none')
            if final_r is not None:
                ax.axvline(final_r, color='red', linestyle='--', linewidth=2, label=f'r = {final_r}')
            ax.set_xlabel('Node Attention Score')
            ax.set_ylabel('Masking Impact  |Δ sigmoid(pred)|')
            ax.set_title(f'Individual Node (Feature+Edge) — {label} ({split})')
            ax.legend(fontsize=11)
            plt.tight_layout()
            fig.savefig(output_dir / f'score_vs_impact_indiv_node_{tag}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            corr, pval = sp_stats.pearsonr(xs, ys) if len(xs) > 2 else (np.nan, np.nan)
            print(f"[INFO] Individual Node: {len(recs)} points, Pearson r={corr:.4f}, p={pval:.2e}")
    else:
        print(f"[INFO] Individual_node_node_masking_impact.jsonl not found — skipping")

    # ── Individual edge plot (per-edge: remove one undirected edge) ──
    indiv_edge_path = seed_dir / 'Individual_edge_node_and_edge_masking_impact.jsonl'
    indiv_edge_data = None
    if indiv_edge_path.exists():
        recs = _read_jsonl(indiv_edge_path, split)
        if recs:
            ys = np.array([abs(_sigmoid(rec['new_prediction']) - _sigmoid(rec['old_prediction']))
                           for rec in recs])
            xs_idx = np.arange(len(ys))

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(xs_idx, ys, alpha=0.2, s=4, c='#9C27B0', edgecolors='none')
            ax.set_xlabel('Edge Index (sorted by graph)')
            ax.set_ylabel('Edge Removal Impact  |Δ sigmoid(pred)|')
            ax.set_title(f'Individual Edge Removal — {label} ({split})')
            plt.tight_layout()
            fig.savefig(output_dir / f'score_vs_impact_indiv_edge_{tag}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Histogram of edge removal impacts
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.hist(ys, bins=50, color='#9C27B0', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Edge Removal Impact  |Δ sigmoid(pred)|')
            ax.set_ylabel('Count')
            ax.set_title(f'Edge Removal Impact Distribution — {label} ({split})')
            plt.tight_layout()
            fig.savefig(output_dir / f'edge_removal_impact_hist_{tag}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"[INFO] Individual Edge: {len(recs)} edges, "
                  f"mean impact={ys.mean():.4f}, max={ys.max():.4f}")
            indiv_edge_data = ys
    else:
        print(f"[INFO] Individual_edge_node_and_edge_masking_impact.jsonl not found — skipping")

    # ── Combined figure (motif-level + individual node) ──
    panels = []
    if has_motif and common_keys:
        xs_m = np.array([motif_avg_score[k] for k in common_keys])
        ys_m = np.array([motif_impacts[k] for k in common_keys])
        panels.append(('Motif-Level (Feature+Edge)', xs_m, ys_m, '#2196F3', 8, 0.3))
    if indiv_node_data is not None:
        panels.append(('Individual Node (Feature+Edge)',
                        indiv_node_data[0], indiv_node_data[1], '#FF5722', 4, 0.15))

    if len(panels) >= 2:
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ax, (title, x, y, c, sz, alpha_v) in zip(axes, panels):
            ax.scatter(x, y, alpha=alpha_v, s=sz, c=c, edgecolors='none')
            if final_r is not None:
                ax.axvline(final_r, color='red', linestyle='--', linewidth=2, label=f'r = {final_r}')
            ax.set_xlabel('Attention Score')
            ax.set_ylabel('Masking Impact')
            ax.set_title(title)
            ax.legend()
        fig.suptitle(f'Score vs Impact — {label} ({split})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'score_vs_impact_combined_{tag}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"[INFO] Score-vs-impact plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Post-hoc Motif Node Score Consistency Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic — from checkpoint
  python analyze_motif_consistency.py \\
      --checkpoint_dir /path/to/seed_dir \\
      --dataset ba_2motifs --model GIN

  # Molecular — from pre-computed JSONL (fast, no GPU)
  python analyze_motif_consistency.py \\
      --from_jsonl /path/to/seed_dir/node_scores.jsonl \\
      --dataset Mutagenicity --model GIN

  # Molecular — from checkpoint (re-extracts scores)
  python analyze_motif_consistency.py \\
      --checkpoint_dir /path/to/seed_dir \\
      --dataset Mutagenicity --model GIN --fold 0

  # Compare train vs test
  python analyze_motif_consistency.py \\
      --from_jsonl /path/to/node_scores.jsonl \\
      --dataset BBBP --model GIN --split train
        """
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with gsat_clf_epoch_*.pt / gsat_att_epoch_*.pt')
    source.add_argument('--from_jsonl', type=str, default=None,
                        help='Path to node_scores.jsonl (fast, no model loading)')
    source.add_argument('--score_vs_impact', type=str, default=None,
                        metavar='SEED_DIR',
                        help='Standalone: just generate the score-vs-impact plot '
                             'from a seed_dir (no consistency analysis)')

    parser.add_argument('--dataset', type=str, default=None, choices=ALL_DATASETS,
                        help='Dataset name')
    parser.add_argument('--model', type=str, default=None,
                        choices=['GIN', 'PNA', 'GAT', 'SAGE', 'GCN'],
                        help='Model architecture')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Checkpoint epoch (default: latest)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number (molecular datasets)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test', 'training'])
    parser.add_argument('--output_dir', type=str, default='../motif_consistency_results')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--min_occurrences', type=int, default=3,
                        help='Min graphs a motif must appear in (molecular)')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Top-K motifs for detailed plots')

    args = parser.parse_args()

    if args.score_vs_impact is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_score_vs_impact(
            seed_dir=args.score_vs_impact,
            split=args.split,
            output_dir=out,
            model_name=args.model,
            dataset_name=args.dataset,
        )
    else:
        if args.dataset is None or args.model is None:
            parser.error('--dataset and --model are required for consistency analysis')
        run_analysis(
            checkpoint_dir=args.checkpoint_dir,
            from_jsonl=args.from_jsonl,
            dataset_name=args.dataset,
            model_name=args.model,
            output_dir=args.output_dir,
            epoch=args.epoch,
            seed=args.seed,
            split=args.split,
            device=args.device,
            fold=args.fold,
            min_occurrences=args.min_occurrences,
            top_k=args.top_k,
        )


if __name__ == '__main__':
    main()
