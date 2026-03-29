import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import os
import json
import io
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
import wandb
from PIL import Image

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict

from torch_geometric.utils import scatter, softmax  # or from torch_scatter import scatter
import pandas as pd


# =============================================================================
# MOTIF INCORPORATION HELPER FUNCTIONS
# =============================================================================

# def load_motif_scores_as_r(csv_path, fold, motif_list):
#     """
#     Load pre-computed motif importance scores from CSV and build an r-value
#     tensor indexed by motif ID.

#     Args:
#         csv_path: Path to the motif scores CSV (columns: dataset, motif string,
#                   fold_0, fold_1, ..., fold_4).
#         fold: Current fold number (0-4). Used to pick the fold-specific column;
#               if a value is missing, the mean across available folds is used.
#         motif_list: List of motif strings where motif_list[i] is the SMILES for
#                     motif index i (loaded from *_motif_list.pickle).

#     Returns:
#         motif_r: FloatTensor of shape [len(motif_list)] mapping motif_id -> r value.
#     """
#     df = pd.read_csv(csv_path)

#     fold_cols = [c for c in df.columns if c.startswith('fold_')]
#     fold_col = f'fold_{fold}'

#     # Build motif_string -> r mapping
#     score_map = {}
#     unk_r = 0.5  # fallback if UNK row is also missing
#     for _, row in df.iterrows():
#         motif_str = str(row['motif string']).strip()
#         if fold_col in df.columns and pd.notna(row.get(fold_col)):
#             r_val = float(row[fold_col])
#         else:
#             vals = [float(row[c]) for c in fold_cols if pd.notna(row.get(c))]
#             r_val = float(np.mean(vals)) if vals else 0.5
#         score_map[motif_str] = r_val
#         if motif_str == 'UNK':
#             unk_r = r_val

#     motif_r = torch.full((len(motif_list),), unk_r, dtype=torch.float)
#     for idx, motif_str in enumerate(motif_list):
#         if motif_str in score_map:
#             motif_r[idx] = score_map[motif_str]

#     motif_r = motif_r.clamp(min=0.01, max=0.99) #Todo: Is this needed?
#     print(f'[INFO] Loaded motif scores from {csv_path} (fold={fold})')
#     print(f'[INFO]   {len(score_map)} motifs in CSV, {len(motif_list)} in vocabulary, UNK r={unk_r:.4f}')
#     print(f'[INFO]   r range: [{motif_r.min().item():.4f}, {motif_r.max().item():.4f}]')
#     return motif_r


def check_no_unmapped_nodes(nodes_to_motifs):
    """
    Raise error if any node is unmapped (has nodes_to_motifs == -1).
    All nodes must belong to a motif for motif incorporation methods.
    """
    if (nodes_to_motifs == -1).any():
        num_unmapped = (nodes_to_motifs == -1).sum().item()
        raise ValueError(
            f"Found {num_unmapped} unmapped nodes (nodes_to_motifs == -1). "
            f"All nodes must belong to a motif for motif incorporation methods."
        )


def motif_consistency_loss(att, nodes_to_motifs, batch):
    """
    Compute within-motif variance and between-motif variance at GRAPH level.
    
    Within-motif: for each motif, variance of node attentions (want low).
    Between-motif: for each graph, variance of motif mean attentions (want high).
    
    Args:
        att: [N, 1] or [N] node attentions
        nodes_to_motifs: [N] LongTensor with global motif id per node
        batch: [N] LongTensor with graph id per node
    
    Returns:
        (within_var, between_var) — both scalar tensors on the same device as att.
        within_var: mean intra-motif variance (lower = more consistent within motifs)
        between_var: mean inter-motif variance per graph (higher = more discriminative)
    """
    att = att.view(-1)
    device = att.device

    nodes_to_motifs = nodes_to_motifs.to(device=device, dtype=torch.long)
    batch = batch.to(device=device, dtype=torch.long)

    max_motif_id = nodes_to_motifs.max().item() + 1
    graph_motif_id = batch * max_motif_id + nodes_to_motifs

    unique_graph_motifs = graph_motif_id.unique()

    # --- Within-motif variance ---
    within_total = att.new_tensor(0.0)
    within_count = 0
    # Also collect (graph_id, motif_mean) for between-motif variance
    graph_ids_for_motifs = []
    motif_means = []

    for gm_id in unique_graph_motifs:
        mask = (graph_motif_id == gm_id)
        num_nodes = mask.sum()
        vals = att[mask]
        mean_val = vals.mean()

        graph_id = gm_id.item() // max_motif_id
        graph_ids_for_motifs.append(graph_id)
        motif_means.append(mean_val)

        if num_nodes > 1:
            within_total = within_total + (vals - mean_val).pow(2).mean()
            within_count += 1

    within_var = within_total / within_count if within_count > 0 else att.new_tensor(0.0)

    # --- Between-motif variance (per graph, then averaged) ---
    if len(motif_means) < 2:
        return within_var, att.new_tensor(0.0)

    motif_means_t = torch.stack(motif_means)
    graph_ids_t = torch.tensor(graph_ids_for_motifs, device=device, dtype=torch.long)
    unique_graphs = graph_ids_t.unique()

    between_total = att.new_tensor(0.0)
    between_count = 0
    for g_id in unique_graphs:
        g_mask = (graph_ids_t == g_id)
        if g_mask.sum() < 2:
            continue
        g_means = motif_means_t[g_mask]
        between_total = between_total + g_means.var()
        between_count += 1

    between_var = between_total / between_count if between_count > 0 else att.new_tensor(0.0)

    return within_var, between_var


def compute_motif_inverse_indices(nodes_to_motifs, batch):
    """
    Map each node to a dense motif row index (per-graph motif instances).

    Returns:
        inverse_indices: [N] in 0..M-1
        motif_batch: [M]
        motif_ids: [M] global motif id within vocabulary
    """
    max_motif_id = int(nodes_to_motifs.max().item()) + 1
    graph_motif_id = batch.long() * max_motif_id + nodes_to_motifs.long()
    unique_graph_motifs, inverse_indices = graph_motif_id.unique(return_inverse=True)
    motif_batch = unique_graph_motifs // max_motif_id
    motif_ids = unique_graph_motifs % max_motif_id
    return inverse_indices, motif_batch, motif_ids


class IntraMotifAttentionPool(nn.Module):
    """
    Attention-weighted motif embedding + mean/max/size features for readout.

    Per motif instance m: softmax over nodes in V_m, r_m = sum_v alpha_v h_v;
    concat mean(h), max(h), log(1+|V_m|) -> [3H+1] per motif row.
    """

    def __init__(self, hidden_size, att_hidden=64):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, att_hidden)
        self.lin2 = nn.Linear(att_hidden, 1)

    def forward(self, emb, inverse_indices):
        q = torch.tanh(self.lin1(emb))
        logits = self.lin2(q).squeeze(-1)
        num_motifs = int(inverse_indices.max().item()) + 1
        alpha = softmax(logits, inverse_indices, num_nodes=num_motifs)
        weighted = emb * alpha.unsqueeze(-1)
        r_m = scatter(weighted, inverse_indices, dim=0, reduce='sum')
        mean_m = scatter(emb, inverse_indices, dim=0, reduce='mean')
        max_m = scatter(emb, inverse_indices, dim=0, reduce='max')
        ones = torch.ones(emb.size(0), device=emb.device, dtype=emb.dtype)
        counts = scatter(ones, inverse_indices, dim=0, reduce='sum')
        size_f = torch.log(1.0 + counts).unsqueeze(-1)
        return torch.cat([r_m, mean_m, max_m, size_f], dim=-1)


class MotifPriorNodeGateMLP(nn.Module):
    """
    Node gate conditioned on local structure and motif context (readout + prior).

    g_i = f_node([h_i || z~_{a(i)} || α_{a(i)}]),  α_i = σ(g_i) after GSAT sampling.

    z~_{a(i)} is the pooled motif embedding row for node i's motif instance; α_{a(i)} is the
    motif-level score σ(extractor logit) used as a chemical prior; f_node restores per-node flexibility.
    """

    def __init__(self, hidden_size, motif_dim, shared_config):
        super().__init__()
        mult = shared_config.get('extractor_hidden_mult', 1)
        dropout_p = shared_config['extractor_dropout_p']
        in_dim = hidden_size + motif_dim + 1
        self.mlp = MLP([in_dim, hidden_size * 2 * mult, hidden_size * mult, 1], dropout=dropout_p)

    def forward(self, h, z_tilde, alpha_motif, batch):
        """
        Args:
            h: [N, H] node embeddings
            z_tilde: [N, D_motif] motif pooled features gathered per node
            alpha_motif: [N, 1] motif-level prior scores gathered per node
            batch: [N] graph id per node (for InstanceNorm in MLP)
        Returns:
            gate_logit: [N, 1]
        """
        x = torch.cat([h, z_tilde, alpha_motif], dim=-1)
        return self.mlp(x, batch)


def sample_motif_readout_with_prior_node_gate(
    sampling_fn,
    motif_att_log_logits,
    motif_emb,
    emb,
    inverse_indices,
    batch,
    epoch,
    training,
    use_raw_score_loss,
    gate_module,
    detach_alpha,
    detach_z,
):
    """
    Motif extractor defines prior α_m; node gate MLP defines stochastic node attention.
    Info loss applies to node gates (att is node-level).
    """
    motif_alpha = motif_att_log_logits.sigmoid()
    if detach_alpha:
        motif_alpha = motif_alpha.detach()
    z_node = motif_emb[inverse_indices]
    if detach_z:
        z_node = z_node.detach()
    alpha_node = motif_alpha[inverse_indices]
    gate_logit = gate_module(emb, z_node, alpha_node, batch)
    node_att = sampling_fn(gate_logit, epoch, training)
    att = node_att
    raw_att_for_loss = gate_logit.sigmoid() if use_raw_score_loss else None
    return node_att, att, raw_att_for_loss


def motif_pooling(emb, nodes_to_motifs, batch, reduce='mean'):
    """
    Pool node embeddings to motif level.
    
    Handles non-consecutive global motif indices by remapping to local
    consecutive indices within the batch. Motifs are kept separate per-graph
    to ensure proper graph-level processing.
    
    Args:
        emb: [N, hidden] node embeddings
        nodes_to_motifs: [N] global motif indices (can be non-consecutive)
        batch: [N] batch assignment for nodes
        reduce: Aggregation method ('mean', 'sum', or 'multi' for
                concatenated mean+max+sum giving 3× the embedding width).
                ('intra_att' is handled by IntraMotifAttentionPool on GSAT, not here.)
    
    Returns:
        motif_emb: [M, hidden] (or [M, 3*hidden] for 'multi')
        motif_batch: [M] batch assignment for motifs
        inverse_indices: [N] mapping from each node to its motif index (0..M-1)
        motif_ids: [M] original global motif index for each pooled motif
    """
    inverse_indices, motif_batch, motif_ids = compute_motif_inverse_indices(nodes_to_motifs, batch)

    if reduce == 'multi':
        motif_mean = scatter(emb, inverse_indices, dim=0, reduce='mean')
        motif_max = scatter(emb, inverse_indices, dim=0, reduce='max')
        motif_sum = scatter(emb, inverse_indices, dim=0, reduce='sum')
        motif_emb = torch.cat([motif_mean, motif_max, motif_sum], dim=1)
    else:
        motif_emb = scatter(emb, inverse_indices, dim=0, reduce=reduce)
    return motif_emb, motif_batch, inverse_indices, motif_ids


def motif_mean_pooling(emb, nodes_to_motifs, batch):
    """Backward-compatible wrapper."""
    return motif_pooling(emb, nodes_to_motifs, batch, reduce='mean')


def lift_motif_att_to_node_att(motif_att, inverse_indices):
    """
    Broadcast motif attention scores back to nodes.
    
    Each node gets the attention score of its corresponding motif.
    
    Args:
        motif_att: [M, 1] attention per motif
        inverse_indices: [N] mapping from nodes to local motif index (0..M-1)
    
    Returns:
        node_att: [N, 1] attention per node
    """
    return motif_att[inverse_indices]


def forward_clf_with_node_attention_injection(
    clf, data, node_att, edge_att, learn_edge_att, w_feat, w_message, w_readout, edge_attr=None,
):
    """
    <USAGE> Shared GSAT classifier forward: W_FEAT / W_MESSAGE / W_READOUT when using node-level attention.

    Matches the node-attention branch in forward_pass (previously duplicated at ~1266–1273 and ~1311–1318).
    When learn_edge_att is True, falls back to full-graph forward with edge_atten (legacy; disabled in streamlined configs).
    """
    if learn_edge_att:
        return clf(data.x, data.edge_index, data.batch, edge_attr=edge_attr, edge_atten=edge_att)
    x_clf = data.x * node_att if w_feat else data.x
    edge_atten_mp = edge_att if w_message else None
    clf_emb = clf.get_emb(
        x_clf, data.edge_index, batch=data.batch, edge_attr=edge_attr, edge_atten=edge_atten_mp,
    )
    if w_readout:
        clf_emb = clf_emb * node_att
    return clf.get_pred_from_emb(clf_emb, data.batch)


def sample_node_level_gsat(sampling_fn, att_log_logits, epoch, training, use_raw_score_loss):
    """
    <SAMPLING> Node-level Concrete / sigmoid sampling (baseline GSAT path).
    sampling_fn: typically model.sampling bound method.
    """
    att = sampling_fn(att_log_logits, epoch, training)
    raw_att_for_loss = att_log_logits.sigmoid() if use_raw_score_loss else None
    return att, raw_att_for_loss


def sample_motif_readout_branch(
    sampling_fn,
    motif_att_log_logits,
    inverse_indices,
    motif_level_sampling,
    motif_level_info_loss,
    epoch,
    training,
    use_raw_score_loss,
):
    """
    <SAMPLING> Motif readout: either sample at motif level then broadcast, or lift logits to nodes then sample.
    Returns (node_att, att_for_loss_scalar_path, raw_att_for_loss).
    """
    if motif_level_sampling:
        motif_att = sampling_fn(motif_att_log_logits, epoch, training)
        node_att = lift_motif_att_to_node_att(motif_att, inverse_indices)
        att = motif_att if motif_level_info_loss else node_att
        if use_raw_score_loss:
            raw_motif = motif_att_log_logits.sigmoid()
            raw_att_for_loss = (
                raw_motif if motif_level_info_loss else lift_motif_att_to_node_att(raw_motif, inverse_indices)
            )
        else:
            raw_att_for_loss = None
    else:
        node_att_log_logits = lift_motif_att_to_node_att(motif_att_log_logits, inverse_indices)
        node_att = sampling_fn(node_att_log_logits, epoch, training)
        att = node_att
        raw_att_for_loss = node_att_log_logits.sigmoid() if use_raw_score_loss else None
    return node_att, att, raw_att_for_loss


def construct_motif_graph(x, edge_index, edge_attr, nodes_to_motifs, batch):
    """
    Construct a coarsened motif-level graph from the node-level graph.
    
    Each motif becomes a single node in the new graph. An edge exists between
    two motif-nodes if ANY node in one motif is connected to ANY node in the
    other motif in the original graph.
    
    IMPORTANT: Motifs are kept separate per-graph in the batch. Even if two graphs
    have nodes with the same global motif ID, they become separate motif nodes.
    This ensures the number of motif nodes per graph matches graph-level labels.
    
    Args:
        x: [N, feat] node features
        edge_index: [2, E] edge indices
        edge_attr: [E, attr_dim] edge attributes (can be None)
        nodes_to_motifs: [N] global motif indices
        batch: [N] batch assignment for nodes
    
    Returns:
        motif_x: [M, feat] aggregated motif features (mean pooled)
        motif_edge_index: [2, E'] unique edges between motifs
        motif_edge_attr: [E', attr_dim] aggregated edge attributes (or None)
        motif_batch: [M] batch assignment for motifs (which graph each motif belongs to)
        unique_motifs: [M] original global motif IDs (note: may have duplicates across graphs)
        inverse_indices: [N] mapping from nodes to local motif index (0..M-1)
    """
    # Create unique (graph_id, motif_id) pairs to keep motifs separate per-graph
    # This prevents merging same motif ID across different graphs in the batch
    max_motif_id = nodes_to_motifs.max().item() + 1
    graph_motif_id = batch * max_motif_id + nodes_to_motifs
    
    # Get unique graph-motif combinations
    unique_graph_motifs, inverse_indices = graph_motif_id.unique(return_inverse=True)
    num_motifs = unique_graph_motifs.size(0)
    
    # Decode back to get motif batch assignment and original motif IDs
    motif_batch = unique_graph_motifs // max_motif_id
    unique_motifs = unique_graph_motifs % max_motif_id
    
    # Aggregate node features to motifs using mean pooling
    motif_x = scatter(x, inverse_indices, dim=0, reduce='mean')
    
    # Map edges to motif level
    src, dst = edge_index
    src_motif = inverse_indices[src]
    dst_motif = inverse_indices[dst]
    
    # Deduplicate motif edges using integer encoding trick
    # Encode (src_motif, dst_motif) as single integer for efficient deduplication
    combined = src_motif * num_motifs + dst_motif
    unique_combined, edge_mapping = combined.unique(return_inverse=True)
    
    # Decode back to edge pairs
    motif_edge_index = torch.stack([
        unique_combined // num_motifs,  # src motif
        unique_combined % num_motifs    # dst motif
    ], dim=0)
    
    # Aggregate edge attributes if present
    # When multiple original edges map to same motif edge, take the mean
    if edge_attr is not None:
        motif_edge_attr = scatter(edge_attr.float(), edge_mapping, dim=0, reduce='mean')
    else:
        motif_edge_attr = None
    
    return motif_x, motif_edge_index, motif_edge_attr, motif_batch, unique_motifs, inverse_indices


def map_motif_att_to_edge_att(motif_att, edge_index, inverse_indices):
    """
    Map motif attention scores to edge attention on the original graph.
    
    edge_att[e] = motif_att[src_motif] * motif_att[dst_motif]
    
    For intra-motif edges (both endpoints in same motif), this results in
    squared attention values.
    
    Args:
        motif_att: [M, 1] attention per motif
        edge_index: [2, E] original edge indices
        inverse_indices: [N] mapping from nodes to local motif index
    
    Returns:
        edge_att: [E, 1] attention per edge
    """
    src, dst = edge_index
    src_motif_idx = inverse_indices[src]
    dst_motif_idx = inverse_indices[dst]
    
    src_att = motif_att[src_motif_idx]
    dst_att = motif_att[dst_motif_idx]
    
    return src_att * dst_att

# =============================================================================
# TEST GRAPH PIPELINE - DETAILED OUTPUT FOR EACH METHOD
# =============================================================================

def run_test_graphs_pipeline(n_graphs, model, extractor, data_loader, device, output_file, model_config):
    """
    Run N graphs through the pipeline and log detailed outputs at each stage.
    
    This function traces through all 4 motif incorporation methods to show:
    - Input graph structure
    - Node embeddings  
    - Method-specific intermediate results
    - Final attention and predictions
    
    Args:
        n_graphs: Number of graphs to process
        model: The GNN classifier model
        extractor: The attention extractor MLP
        data_loader: DataLoader to get test graphs from
        device: torch device
        output_file: Path to save detailed outputs
        model_config: Model configuration dict
    """
    model.eval()
    extractor.eval()
    
    # Collect n_graphs from the data loader
    test_graphs = []
    for batch_data in data_loader:
        # Process individual graphs from the batch
        batch_data = batch_data.to(device)
        num_in_batch = batch_data.batch.max().item() + 1
        
        for graph_idx in range(num_in_batch):
            if len(test_graphs) >= n_graphs:
                break
            
            # Extract single graph from batch
            node_mask = (batch_data.batch == graph_idx)
            graph_nodes = node_mask.sum().item()
            
            # Get node indices for this graph
            node_indices = torch.where(node_mask)[0]
            
            # Create edge mask for edges within this graph
            src, dst = batch_data.edge_index
            edge_mask = node_mask[src] & node_mask[dst]
            
            # Store graph info
            graph_info = {
                'graph_idx': len(test_graphs),
                'batch_data': batch_data,
                'node_mask': node_mask,
                'edge_mask': edge_mask,
                'node_indices': node_indices,
                'num_nodes': graph_nodes,
                'num_edges': edge_mask.sum().item()
            }
            test_graphs.append(graph_info)
        
        if len(test_graphs) >= n_graphs:
            break
    
    if len(test_graphs) == 0:
        print("[WARNING] No graphs found in data loader for testing")
        return
    
    print(f"\n{'='*80}")
    print(f"RUNNING TEST GRAPH PIPELINE FOR {len(test_graphs)} GRAPHS")
    print(f"{'='*80}\n")
    
    # Open output file
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MOTIFSAT TEST GRAPH PIPELINE - DETAILED OUTPUT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Number of graphs: {len(test_graphs)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Process each graph through all methods
        for graph_info in test_graphs:
            batch_data = graph_info['batch_data']
            graph_idx = graph_info['graph_idx']
            node_mask = graph_info['node_mask']
            edge_mask = graph_info['edge_mask']
            
            f.write("\n" + "#" * 100 + "\n")
            f.write(f"GRAPH {graph_idx}\n")
            f.write("#" * 100 + "\n\n")
            
            # =====================================================================
            # SECTION 1: INPUT GRAPH STRUCTURE
            # =====================================================================
            f.write("-" * 80 + "\n")
            f.write("1. INPUT GRAPH STRUCTURE\n")
            f.write("-" * 80 + "\n\n")
            
            num_nodes = node_mask.sum().item()
            num_edges = edge_mask.sum().item()
            
            f.write(f"Number of nodes: {num_nodes}\n")
            f.write(f"Number of edges: {num_edges}\n")
            f.write(f"Node feature dim: {batch_data.x.shape[1]}\n")
            
            if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
                f.write(f"Edge attr dim: {batch_data.edge_attr.shape[1]}\n")
            else:
                f.write("Edge attr: None\n")
            
            # Node features summary
            node_x = batch_data.x[node_mask]
            f.write(f"\nNode features:\n")
            for i in range(num_nodes):
                f.write(f"  Node {i}: {node_x[i].cpu().numpy()}\n")
            
            # Edge structure
            src, dst = batch_data.edge_index[:, edge_mask]
            # Remap to local indices
            node_remap = torch.zeros(batch_data.x.size(0), dtype=torch.long, device=device)
            node_remap[node_mask] = torch.arange(num_nodes, device=device)
            local_src = node_remap[src]
            local_dst = node_remap[dst]
            
            f.write(f"\nEdge list:\n")
            for i in range(num_edges):
                f.write(f"  Edge {i}: {local_src[i].item()} -> {local_dst[i].item()}\n")
            
            # Motif structure
            if hasattr(batch_data, 'nodes_to_motifs'):
                nodes_to_motifs = batch_data.nodes_to_motifs[node_mask]
                unique_motifs = nodes_to_motifs.unique()
                f.write(f"\nMotif structure:\n")
                f.write(f"  Number of unique motifs: {len(unique_motifs)}\n")
                f.write(f"  Motif IDs: {unique_motifs.cpu().numpy()}\n")
                f.write(f"  Node-to-motif mapping: {nodes_to_motifs.cpu().numpy()}\n")
                
                # Motif sizes
                f.write(f"\n  Motif sizes:\n")
                for motif_id in unique_motifs:
                    size = (nodes_to_motifs == motif_id).sum().item()
                    f.write(f"    Motif {motif_id.item()}: {size} nodes\n")
            else:
                f.write("\nMotif structure: Not available\n")
            
            # Label
            if hasattr(batch_data, 'y'):
                # Get label for this specific graph
                if batch_data.y.dim() > 1:
                    label = batch_data.y[graph_idx].cpu().numpy()
                else:
                    label = batch_data.y[graph_idx].item()
                f.write(f"\nGraph label: {label}\n")
            
            # =====================================================================
            # SECTION 2: NODE EMBEDDINGS
            # =====================================================================
            f.write("\n" + "-" * 80 + "\n")
            f.write("2. NODE EMBEDDINGS (from GNN backbone)\n")
            f.write("-" * 80 + "\n\n")
            
            with torch.no_grad():
                edge_attr = batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
                emb = model.get_emb(batch_data.x, batch_data.edge_index, batch=batch_data.batch, edge_attr=edge_attr)
                node_emb = emb[node_mask]
            
            f.write(f"Embedding shape: {node_emb.shape}\n")
            f.write(f"Embedding stats: mean={node_emb.mean().item():.4f}, std={node_emb.std().item():.4f}, "
                   f"min={node_emb.min().item():.4f}, max={node_emb.max().item():.4f}\n")
            
            f.write(f"\nEmbedding samples:\n")
            for i in range(num_nodes):
                f.write(f"  Node {i}: {node_emb[i].cpu().numpy()}\n")
            
            # =====================================================================
            # SECTION 3: METHOD = None (Baseline)
            # =====================================================================
            f.write("\n" + "-" * 80 + "\n")
            f.write("3. METHOD: None (BASELINE)\n")
            f.write("-" * 80 + "\n\n")
            f.write("Description: Standard GSAT - node-level attention, no motif information used.\n\n")
            
            with torch.no_grad():
                # Get node attention
                att_log_logits = extractor(emb, batch_data.edge_index, batch_data.batch)
                node_att = torch.sigmoid(att_log_logits)
                
                graph_node_att = node_att[node_mask]
                
                f.write(f"Node attention shape: {graph_node_att.shape}\n")
                f.write(f"Node attention stats: mean={graph_node_att.mean().item():.4f}, "
                       f"std={graph_node_att.std().item():.4f}, "
                       f"min={graph_node_att.min().item():.4f}, max={graph_node_att.max().item():.4f}\n")
                f.write(f"\nNode attention values:\n")
                for i in range(num_nodes):
                    f.write(f"  Node {i}: {graph_node_att[i].item():.4f}\n")
                
                # Lift to edge attention
                src_full, dst_full = batch_data.edge_index
                edge_att_full = node_att[src_full] * node_att[dst_full]
                graph_edge_att = edge_att_full[edge_mask]
                
                f.write(f"\nEdge attention (lifted from nodes):\n")
                f.write(f"  Shape: {graph_edge_att.shape}\n")
                f.write(f"  Stats: mean={graph_edge_att.mean().item():.4f}, "
                       f"std={graph_edge_att.std().item():.4f}\n")
                f.write(f"\n  All edges:\n")
                for i in range( num_edges):
                    f.write(f"    Edge {local_src[i].item()}->{local_dst[i].item()}: {graph_edge_att[i].item():.4f}\n")
            
            # =====================================================================
            # SECTION 4: METHOD = 'loss' (Motif Consistency Loss)
            # =====================================================================
            f.write("\n" + "-" * 80 + "\n")
            f.write("4. METHOD: 'loss' (MOTIF CONSISTENCY LOSS)\n")
            f.write("-" * 80 + "\n\n")
            f.write("Description: Same attention as baseline, but adds motif consistency loss\n")
            f.write("             to encourage similar attention within motifs.\n\n")
            
            if hasattr(batch_data, 'nodes_to_motifs'):
                with torch.no_grad():
                    # Same attention as baseline
                    f.write("Attention: Same as baseline (node-level)\n")
                    f.write("Edge attention: Same as baseline\n\n")
                    
                    # Calculate motif consistency loss for this graph
                    nodes_to_motifs_full = batch_data.nodes_to_motifs
                    
                    f.write("Motif Consistency Analysis:\n")
                    for motif_id in unique_motifs:
                        motif_mask = (nodes_to_motifs == motif_id)
                        motif_att_vals = graph_node_att[motif_mask]
                        if len(motif_att_vals) > 1:
                            variance = motif_att_vals.var().item()
                            f.write(f"  Motif {motif_id.item()}: "
                                   f"mean_att={motif_att_vals.mean().item():.4f}, "
                                   f"variance={variance:.6f}, "
                                   f"nodes={[f'{v.item():.4f}' for v in motif_att_vals]}\n")
                        else:
                            f.write(f"  Motif {motif_id.item()}: "
                                   f"att={motif_att_vals.mean().item():.4f} (single node)\n")
            else:
                f.write("Motif information not available - same as baseline.\n")
            
            # =====================================================================
            # SECTION 5: METHOD = 'readout' (Motif-level Pooling)
            # =====================================================================
            f.write("\n" + "-" * 80 + "\n")
            f.write("5. METHOD: 'readout' (MOTIF-LEVEL POOLING)\n")
            f.write("-" * 80 + "\n\n")
            f.write("Description: Pool node embeddings to motif level, score motifs,\n")
            f.write("             then lift attention back to nodes and edges.\n\n")
            
            if hasattr(batch_data, 'nodes_to_motifs'):
                with torch.no_grad():
                    # Step 1: Motif pooling
                    f.write("Step 1: Motif Mean Pooling\n")
                    motif_emb, motif_batch, inverse_indices, _ = motif_mean_pooling(
                        emb, batch_data.nodes_to_motifs, batch_data.batch
                    )
                    
                    # Get motifs for this graph
                    graph_batch_idx = batch_data.batch[node_mask][0].item()
                    motif_mask_for_graph = (motif_batch == graph_batch_idx)
                    graph_motif_emb = motif_emb[motif_mask_for_graph]
                    
                    f.write(f"  Total motif embeddings: {motif_emb.shape}\n")
                    f.write(f"  Motifs in this graph: {graph_motif_emb.shape[0]}\n")
                    f.write(f"  Motif embedding dim: {graph_motif_emb.shape[1]}\n")
                    f.write(f"\n  Motif embedding stats:\n")
                    for i in range(graph_motif_emb.shape[0]):
                        f.write(f"    Motif {i}: mean={graph_motif_emb[i].mean().item():.4f}, "
                               f"norm={graph_motif_emb[i].norm().item():.4f}\n")
                    
                    # Step 2: Motif attention
                    f.write("\nStep 2: Motif Attention Scoring\n")
                    motif_att_log_logits = extractor(motif_emb, None, motif_batch)
                    motif_att = torch.sigmoid(motif_att_log_logits)
                    graph_motif_att = motif_att[motif_mask_for_graph]
                    
                    f.write(f"  Motif attention values:\n")
                    for i in range(graph_motif_att.shape[0]):
                        f.write(f"    Motif {i}: {graph_motif_att[i].item():.4f}\n")
                    
                    # Step 3: Lift to node attention
                    f.write("\nStep 3: Lift Motif Attention to Nodes\n")
                    node_att_from_motif = lift_motif_att_to_node_att(motif_att, inverse_indices)
                    graph_node_att_readout = node_att_from_motif[node_mask]
                    
                    f.write(f"  Node attention (from motifs):\n")
                    for i in range( num_nodes):
                        f.write(f"    Node {i}: {graph_node_att_readout[i].item():.4f}\n")
                    
                    # Step 4: Lift to edge attention
                    f.write("\nStep 4: Lift to Edge Attention\n")
                    edge_att_readout = node_att_from_motif[src_full] * node_att_from_motif[dst_full]
                    graph_edge_att_readout = edge_att_readout[edge_mask]
                    
                    f.write(f"  Edge attention stats: mean={graph_edge_att_readout.mean().item():.4f}, "
                           f"std={graph_edge_att_readout.std().item():.4f}\n")
                    f.write(f"\n  All edges:\n")
                    for i in range(num_edges):
                        f.write(f"    Edge {local_src[i].item()}->{local_dst[i].item()}: "
                               f"{graph_edge_att_readout[i].item():.4f}\n")
            else:
                f.write("Motif information not available - cannot run readout method.\n")
            
            # =====================================================================
            # SECTION 6: METHOD = 'graph' (Motif Graph Construction)
            # =====================================================================
            f.write("\n" + "-" * 80 + "\n")
            f.write("6. METHOD: 'graph' (MOTIF GRAPH CONSTRUCTION)\n")
            f.write("-" * 80 + "\n\n")
            f.write("Description: Construct a coarsened motif-level graph,\n")
            f.write("             run GNN on it, then map attention back to original edges.\n\n")
            
            if hasattr(batch_data, 'nodes_to_motifs'):
                with torch.no_grad():
                    # Step 1: Construct motif graph
                    f.write("Step 1: Construct Motif Graph\n")
                    motif_x, motif_edge_index, motif_edge_attr, motif_batch_graph, unique_motifs_graph, inverse_indices_graph = \
                        construct_motif_graph(batch_data.x, batch_data.edge_index, edge_attr, 
                                            batch_data.nodes_to_motifs, batch_data.batch)
                    
                    # Get motif graph for this specific graph
                    motif_mask_graph = (motif_batch_graph == graph_batch_idx)
                    num_motif_nodes = motif_mask_graph.sum().item()
                    
                    f.write(f"  Original graph: {num_nodes} nodes, {num_edges} edges\n")
                    f.write(f"  Motif graph: {num_motif_nodes} nodes (motifs)\n")
                    
                    # Count motif edges for this graph
                    if motif_edge_index.size(1) > 0:
                        motif_src, motif_dst = motif_edge_index
                        motif_node_indices = torch.where(motif_mask_graph)[0]
                        # Create set for fast lookup
                        valid_motif_nodes = set(motif_node_indices.cpu().numpy())
                        motif_edge_mask = torch.tensor([
                            (s.item() in valid_motif_nodes and d.item() in valid_motif_nodes)
                            for s, d in zip(motif_src, motif_dst)
                        ], device=device)
                        num_motif_edges = motif_edge_mask.sum().item()
                    else:
                        num_motif_edges = 0
                    
                    f.write(f"  Motif edges: {num_motif_edges}\n")
                    f.write(f"  Compression ratio: {num_nodes}/{num_motif_nodes} = {num_nodes/max(1,num_motif_nodes):.2f}x\n")
                    
                    # Motif features
                    graph_motif_x = motif_x[motif_mask_graph]
                    f.write(f"\n  Motif node features:\n")
                    for i in range(num_motif_nodes):
                        f.write(f"    Motif node {i}: mean={graph_motif_x[i].mean().item():.4f}, "
                               f"norm={graph_motif_x[i].norm().item():.4f}\n")
                    
                    # Step 2: Get motif embeddings
                    f.write("\nStep 2: Motif Graph Embeddings (from GNN)\n")
                    motif_emb_graph = model.get_emb(motif_x, motif_edge_index, batch=motif_batch_graph, 
                                                   edge_attr=motif_edge_attr)
                    graph_motif_emb_g = motif_emb_graph[motif_mask_graph]
                    
                    f.write(f"  Motif embedding shape: {graph_motif_emb_g.shape}\n")
                    f.write(f"  Embedding stats: mean={graph_motif_emb_g.mean().item():.4f}, "
                           f"std={graph_motif_emb_g.std().item():.4f}\n")
                    
                    # Step 3: Motif attention
                    f.write("\nStep 3: Motif Attention Scoring\n")
                    motif_att_log_graph = extractor(motif_emb_graph, motif_edge_index, motif_batch_graph)
                    motif_att_graph = torch.sigmoid(motif_att_log_graph)
                    graph_motif_att_g = motif_att_graph[motif_mask_graph]
                    
                    f.write(f"  Motif attention values:\n")
                    for i in range(num_motif_nodes):
                        f.write(f"    Motif node {i}: {graph_motif_att_g[i].item():.4f}\n")
                    
                    # Step 4: Map to original edge attention
                    f.write("\nStep 4: Map to Original Edge Attention\n")
                    edge_att_graph = map_motif_att_to_edge_att(motif_att_graph, batch_data.edge_index, 
                                                              inverse_indices_graph)
                    graph_edge_att_graph = edge_att_graph[edge_mask]
                    
                    f.write(f"  Edge attention stats: mean={graph_edge_att_graph.mean().item():.4f}, "
                           f"std={graph_edge_att_graph.std().item():.4f}\n")
                    f.write(f"\n  All edges:\n")
                    for i in range(num_edges):
                        f.write(f"    Edge {local_src[i].item()}->{local_dst[i].item()}: "
                               f"{graph_edge_att_graph[i].item():.4f}\n")
                    
                    # Intra-motif vs inter-motif edge analysis
                    f.write("\nStep 5: Intra-motif vs Inter-motif Edge Analysis\n")
                    local_inverse = inverse_indices_graph[node_mask]
                    src_motif = local_inverse[local_src - local_src.min()]
                    dst_motif = local_inverse[local_dst - local_dst.min()]
                    
                    intra_mask = (src_motif == dst_motif)
                    inter_mask = ~intra_mask
                    
                    if intra_mask.any():
                        intra_att = graph_edge_att_graph[intra_mask]
                        f.write(f"  Intra-motif edges: {intra_mask.sum().item()}, "
                               f"mean_att={intra_att.mean().item():.4f}\n")
                    if inter_mask.any():
                        inter_att = graph_edge_att_graph[inter_mask]
                        f.write(f"  Inter-motif edges: {inter_mask.sum().item()}, "
                               f"mean_att={inter_att.mean().item():.4f}\n")
            else:
                f.write("Motif information not available - cannot run graph method.\n")
            
            f.write("\n")
    
    print(f"\n[INFO] Detailed pipeline output saved to: {output_file}")
    print(f"[INFO] Processed {len(test_graphs)} graphs through all 4 methods")
    return output_file


# def create_ordered_batch_iterator(dataset, batch_size=2):
#     """
#     Create ordered batch iterator from dataset to ensure data correspondence.
    
#     This function creates small batches (batch_size>=2) to solve InstanceNorm issues
#     while maintaining exact correspondence with the original dataset order.
    
#     Args:
#         dataset: Original dataset (PyG InMemoryDataset)
#         batch_size: Size of each batch (default=2, minimum for InstanceNorm)
        
#     Yields:
#         batch_data: PyG Batch object containing multiple molecules
#         batch_indices: List of original dataset indices  
#         original_samples: List of original Data objects
#         skip_first: Whether to skip the first sample in results (for odd-sized batches)
#     """
#     for i in range(0, len(dataset), batch_size):
#         batch_samples = []
#         batch_indices = []
        
#         # Collect consecutive samples
#         for j in range(i, min(i + batch_size, len(dataset))):
#             sample = dataset[j]
#             batch_samples.append(sample)
#             batch_indices.append(j)
        
#         # Handle last batch with only 1 sample
#         if len(batch_samples) == 1 and i > 0:
#             # Add previous sample to make batch_size=2 (to satisfy InstanceNorm)
#             prev_sample = dataset[i-1]
#             batch_samples = [prev_sample, batch_samples[0]]
#             batch_indices = [i-1, i]
#             skip_first = True  # Skip the duplicated first sample in results
#         else:
#             skip_first = False
        
#         # Create PyG batch
#         try:
#             batch_data = Batch.from_data_list(batch_samples)
#             yield batch_data, batch_indices, batch_samples, skip_first
#         except Exception as e:
#             print(f"Error creating batch at indices {batch_indices}: {e}")
#             # If batch creation fails, process individually with padding
#             for sample, idx in zip(batch_samples, batch_indices):
#                 if idx > 0:
#                     # Create batch with previous sample to satisfy InstanceNorm
#                     padded_batch = Batch.from_data_list([dataset[idx-1], sample])
#                     yield padded_batch, [idx-1, idx], [dataset[idx-1], sample], True
#                 else:
#                     # For first sample, duplicate it
#                     padded_batch = Batch.from_data_list([sample, sample])
#                     yield padded_batch, [idx, idx], [sample, sample], True


# def parse_batch_attention_to_samples(batch_att, batch_data, original_samples, batch_indices, skip_first, learn_edge_att):
#     """
#     Parse batch attention results into individual sample results.
    
#     This function takes the attention results from a batch and splits them back
#     into individual molecule attention scores, maintaining exact correspondence
#     with the original dataset samples.
    
#     Args:
#         batch_att: Attention tensor from batch processing
#         batch_data: PyG Batch object 
#         original_samples: List of original Data objects
#         batch_indices: List of dataset indices
#         skip_first: Whether to skip first sample (for duplicated samples)
#         learn_edge_att: Whether learning edge attention
        
#     Returns:
#         List of dicts containing individual sample results
#     """
#     # Convert batch attention to individual attention arrays
#     if learn_edge_att:
#         batch_edge_att = batch_att.detach().cpu().numpy()
#         batch_node_att = None
#     else:
#         batch_node_att = batch_att.detach().cpu().numpy()
#         batch_edge_att = None  # Will be calculated per sample to avoid batch edge_index issues
    
#     results = []
#     node_ptr = 0
#     edge_ptr = 0
    
#     # Skip first sample if needed (for duplicated samples)
#     start_idx = 1 if skip_first else 0
    
#     for i in range(start_idx, len(original_samples)):
#         sample = original_samples[i]
#         dataset_idx = batch_indices[i]
        
#         num_nodes = sample.x.shape[0]
#         num_edges = sample.edge_index.shape[1]
        
#         # Extract attention scores for current sample
#         if batch_node_att is not None:
#             sample_node_att = batch_node_att[node_ptr:node_ptr + num_nodes]
#         else:
#             sample_node_att = None
            
#         if batch_edge_att is not None:
#             sample_edge_att = batch_edge_att[edge_ptr:edge_ptr + num_edges]
#         else:
#             sample_edge_att = None
        
#         results.append({
#             'dataset_idx': dataset_idx,
#             'sample': sample,
#             'node_att': sample_node_att,
#             'edge_att': sample_edge_att
#         })
        
#         # Update pointers for next sample
#         node_ptr += num_nodes
#         edge_ptr += num_edges
    
#     return results


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, fold, task_type='classification', datasets=None, masked_data=None, motif_clf=None,
                 motif_list=None, intra_motif_pool=None, motif_prior_node_gate_module=None):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.intra_motif_pool = intra_motif_pool  # Intra-motif attention pool (readout + intra_att only)
        self.motif_prior_node_gate_module = motif_prior_node_gate_module  # readout + motif_prior_node_gate only
        self.motif_clf = motif_clf  # Separate model for motif graph (optional)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        
        self.fold = fold
        self.datasets = datasets
        self.masked_data_features = masked_data
        self.task_type = task_type
        self.motif_list = motif_list  # SMILES (or str) per global motif id; None for datasets without vocabulary

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']
        self.embedding_viz_every = int(shared_config.get('embedding_viz_every', 0))
        self.embedding_viz_max_points = int(shared_config.get('embedding_viz_max_points', 5000))
        self.embedding_viz_max_batches = int(shared_config.get('embedding_viz_max_batches', 12))
        self.embedding_viz_max_motif_annotations = int(shared_config.get('embedding_viz_max_motif_annotations', 200))
        self.embedding_viz_skip_epoch0 = bool(shared_config.get('embedding_viz_skip_epoch0', True))
        self.embedding_viz_dpi = int(shared_config.get('embedding_viz_dpi', 300))

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']
        self.motif_loss_coef = method_config['motif_loss_coef']
        self.between_motif_coef = method_config.get('between_motif_coef', 0.0)

        # Early stopping: stop when validation metric does not improve for this many epochs
        self.early_stopping_patience = method_config.get('early_stopping_patience', None)  # None = disabled
        self.early_stopping_min_epochs = method_config.get('early_stopping_min_epochs', 20)  # don't stop before this

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)
        self.model_name = method_config["model_name"]
        self.motif_readout = method_config["model_name"]

        self.multi_label = multi_label
        self.num_class = num_class
        self.criterion = Criterion(num_class, multi_label, task_type)
        
        # Motif incorporation method settings
        self.motif_method = method_config.get('motif_incorporation_method', None)
        self.train_motif_graph = method_config.get('train_motif_graph', False)
        self.separate_motif_model = method_config.get('separate_motif_model', False)
        self.motif_level_info_loss = method_config.get('motif_level_info_loss', False)
        self.motif_level_sampling = method_config.get('motif_level_sampling', False)
        self.use_raw_score_loss = method_config.get('use_raw_score_loss', False)
        self.motif_pooling_method = method_config.get('motif_pooling_method', 'mean')
        self.motif_prior_node_gate = bool(method_config.get('motif_prior_node_gate', False))
        self.motif_prior_detach_alpha = bool(method_config.get('motif_prior_detach_alpha', False))
        self.motif_prior_detach_z = bool(method_config.get('motif_prior_detach_z', False))
        if self.motif_prior_node_gate:
            if self.motif_method != 'readout':
                raise ValueError("motif_prior_node_gate=True requires motif_incorporation_method='readout'")
            if motif_prior_node_gate_module is None:
                raise ValueError("motif_prior_node_gate=True requires motif_prior_node_gate_module (train_gsat_one_seed)")
        if self.motif_pooling_method == 'intra_att':
            if self.motif_method != 'readout':
                raise ValueError("motif_pooling_method='intra_att' requires motif_incorporation_method='readout'")
            if intra_motif_pool is None:
                raise ValueError("motif_pooling_method='intra_att' requires intra_motif_pool module (train_gsat_one_seed)")

        # Info loss warmup: prediction-only phase before info loss kicks in
        self.info_warmup_epochs = method_config.get('info_warmup_epochs', 0)
        
        # Vanilla GNN mode: bypass attention entirely
        self.no_attention = method_config.get('no_attention', False)
        if self.no_attention:
            self.info_loss_coef = 0
            self.motif_loss_coef = 0
            self.between_motif_coef = 0

        # Node attention injection points (only for learn_edge_att=False)
        self.w_feat = method_config.get('w_feat', False)
        self.w_message = method_config.get('w_message', True)
        self.w_readout = method_config.get('w_readout', False)
        
        self.target_k = method_config.get('target_k', None)  # Graph-adaptive r: r_g = target_k / M_g
        # 'interpolate': r(t) = alpha(t)*init_r + (1-alpha(t))*score_r, alpha decays from 1→0
        # 'max': r(t) = max(score_r, get_r(t)), score kicks in once global r decays below it
        # None: fixed score-based r from epoch 0
        self.score_r_schedule = method_config.get('score_r_schedule', None)
        
        # # Score-based per-motif r: load pre-computed motif importance scores
        # motif_scores_path = method_config.get('motif_scores_path', None)
        # if motif_scores_path is not None and motif_list is not None:
        #     self.motif_r_values = load_motif_scores_as_r(motif_scores_path, fold, motif_list).to(device)
        # else:
        #     self.motif_r_values = None
        #     if motif_scores_path is not None and motif_list is None:
        #         print(f'[WARNING] motif_scores_path set but motif_list not provided — score-based r disabled')
        
        # If method is None (baseline), disable motif-structure losses automatically
        if self.motif_method is None:
            self.motif_loss_coef = 0.0
            self.between_motif_coef = 0.0
        # If method is 'readout' or 'graph', motif consistency loss is not applicable
        # (consistency is enforced structurally), so disable it
        elif self.motif_method in ['readout', 'graph']:
            # For these methods, motif_loss_coef is used for auxiliary motif graph loss
            # when train_motif_graph=True (only applicable for 'graph' method)
            pass
        
        print(f'[INFO] No attention (vanilla GNN): {self.no_attention}')
        print(f'[INFO] Motif incorporation method: {self.motif_method}')
        print(f'[INFO] Train motif graph: {self.train_motif_graph}')
        print(f'[INFO] Separate motif model: {self.separate_motif_model}')
        print(f'[INFO] Motif-level info loss: {self.motif_level_info_loss}')
        print(f'[INFO] Motif-level sampling: {self.motif_level_sampling}')
        print(f'[INFO] Motif pooling method: {self.motif_pooling_method}')
        if self.motif_prior_node_gate:
            print(
                f'[INFO] Motif prior node gate: ON (detach_alpha={self.motif_prior_detach_alpha}, '
                f'detach_z={self.motif_prior_detach_z})'
            )
        print(f'[INFO] Use raw score loss: {self.use_raw_score_loss}')
        if self.info_warmup_epochs > 0:
            print(f'[INFO] Info loss warmup: {self.info_warmup_epochs} epochs (prediction-only, deterministic gating)')
        if not self.learn_edge_att:
            print(f'[INFO] Node att injection: W_FEAT={self.w_feat} W_MESSAGE={self.w_message} W_READOUT={self.w_readout}')
        if self.target_k is not None:
            print(f'[INFO] Graph-adaptive r: target_k={self.target_k} (r_g = {self.target_k} / M_g)')
        if self.embedding_viz_every > 0:
            sk0 = 'skip epoch 0' if self.embedding_viz_skip_epoch0 else 'include epoch 0'
            print(f'[INFO] W&B valid embedding PCA viz every {self.embedding_viz_every} epochs '
                  f'(binary non-multilabel only; max_batches={self.embedding_viz_max_batches}; {sk0})')
            if self.task_type != 'classification' or self.multi_label or self.num_class != 2:
                print(
                    f'[WARNING] Embedding PCA viz will be skipped: need binary single-label classification '
                    f'(got task={self.task_type}, multi_label={self.multi_label}, num_class={self.num_class}).'
                )
        # if self.motif_r_values is not None:
        #     print(f'[INFO] Score-based per-motif r: loaded {len(self.motif_r_values)} values')
        
        # Create deterministic directory for saving scores (NO TIMESTAMP!)
        tuning_id = method_config.get('tuning_id', 'default')
        experiment_name = method_config.get('experiment_name', 'default_experiment')

        # Use environment variable if set (for HPC), otherwise use relative path
        results_base = os.environ.get('RESULTS_DIR', '../tuning_results')
        
        # Include motif method in path for proper experiment organization
        motif_method_str = str(self.motif_method) if self.motif_method else 'none'
        train_motif_str = 'trainmotif' if self.train_motif_graph else 'notrain'
        separate_model_str = 'separate' if self.separate_motif_model else 'shared'
        
        self.seed_dir = os.path.join(
            results_base,  # Base directory
            str(self.dataset_name),
            f'model_{self.model_name}',
            f'experiment_{experiment_name}',
            f'tuning_{tuning_id}',
            f'method_{motif_method_str}_{train_motif_str}_{separate_model_str}',
            f'pred{self.pred_loss_coef}_info{self.info_loss_coef}_motif{self.motif_loss_coef}_between{self.between_motif_coef}',
            f'init{self.init_r}_final{self.final_r}_decay{self.decay_r}',
            f'fold{self.fold}_seed{self.random_state}'  # NO TIMESTAMP!
        )
        os.makedirs(self.seed_dir, exist_ok=True)
        # Save configs
        with open(os.path.join(self.seed_dir, "method_config.yaml"), "w") as f:
            yaml.safe_dump(method_config, f, sort_keys=False)

        with open(os.path.join(self.seed_dir, "shared_config.yaml"), "w") as f:
            yaml.safe_dump(shared_config, f, sort_keys=False)

        # Save a summary of this experimental run
        summary = {
            'dataset': self.dataset_name,
            'model': self.model_name,
            'fold': self.fold,
            'seed': self.random_state,
            'task_type': self.task_type,
            'experiment_name': experiment_name,
            'tuning_id': tuning_id,
            'motif_incorporation': {
                'no_attention': self.no_attention,
                'method': self.motif_method,
                'motif_pooling_method': self.motif_pooling_method,
                'train_motif_graph': self.train_motif_graph,
                'separate_motif_model': self.separate_motif_model,
                'motif_level_info_loss': self.motif_level_info_loss,
                'motif_level_sampling': self.motif_level_sampling,
                'use_raw_score_loss': self.use_raw_score_loss,
                'info_warmup_epochs': self.info_warmup_epochs,
                'w_feat': self.w_feat,
                'w_message': self.w_message,
                'w_readout': self.w_readout,
                'target_k': self.target_k,
                'motif_scores_path': method_config.get('motif_scores_path', None),
            },
            'loss_coefficients': {
                'pred_loss_coef': self.pred_loss_coef,
                'info_loss_coef': self.info_loss_coef,
                'motif_loss_coef': self.motif_loss_coef,
                'between_motif_coef': self.between_motif_coef,
            },
            'weight_distribution_params': {
                'init_r': self.init_r,
                'final_r': self.final_r,
                'decay_r': self.decay_r,
                'decay_interval': self.decay_interval,
                'fix_r': self.fix_r
            }
        }

        with open(os.path.join(self.seed_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
    def save_epoch_metrics(self, epoch, phase, loss_dict, att_auroc, precision, clf_acc, clf_roc):
        """Save metrics for each epoch to track training progress."""
        metrics_path = os.path.join(self.seed_dir, 'epoch_metrics.jsonl')

        metric_record = {
            'epoch': epoch,
            'phase': phase,
            'loss': loss_dict['loss'],
            'pred_loss': loss_dict['pred'],
            'info_loss': loss_dict['info'],
            'motif_within': loss_dict.get('motif_within', 0),
            'motif_between': loss_dict.get('motif_between', 0),
            'motif_loss': loss_dict['motif_consistency'],
            'att_auroc': float(att_auroc) if att_auroc is not None else None,
            'precision_at_k': float(precision) if precision is not None else None,
            'clf_acc': float(clf_acc) if clf_acc is not None else None,
            'clf_roc': float(clf_roc) if clf_roc is not None else None,
            'learning_rate': get_lr(self.optimizer)
        }

        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metric_record) + '\n')


    def save_attention_distributions(self, epoch, phase, att):
        """Save attention weight distribution statistics."""
        dist_path = os.path.join(self.seed_dir, 'attention_distributions.jsonl')

        att_np = att.cpu().numpy() if torch.is_tensor(att) else att

        # Calculate distribution metrics
        dist_record = {
            'epoch': epoch,
            'phase': phase,
            'mean': float(np.mean(att_np)),
            'std': float(np.std(att_np)),
            'median': float(np.median(att_np)),
            'min': float(np.min(att_np)),
            'max': float(np.max(att_np)),
            'q25': float(np.percentile(att_np, 25)),
            'q75': float(np.percentile(att_np, 75)),
            # Measure how polarized weights are (near 0/1 vs 0.5)
            'pct_near_0': float(np.mean(att_np < 0.1)),  # Percentage < 0.1
            'pct_near_1': float(np.mean(att_np > 0.9)),  # Percentage > 0.9
            'pct_middle': float(np.mean((att_np >= 0.4) & (att_np <= 0.6))),  # Percentage in [0.4, 0.6]
            # Entropy as measure of uncertainty
            'entropy': float(self._calculate_entropy(att_np))
        }

        with open(dist_path, 'a') as f:
            f.write(json.dumps(dist_record) + '\n')

    def save_final_metrics(self, metric_dict):
        """Save final best metrics after training completes."""
        final_metrics_path = os.path.join(self.seed_dir, 'final_metrics.json')
        serializable = {k: v for k, v in metric_dict.items() if isinstance(v, (int, float, str, bool, type(None)))}
        with open(final_metrics_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def _motif_level_pool(self, emb, nodes_to_motifs, batch):
        """Pool node embeddings to motif rows (same indexing as readout)."""
        if self.motif_pooling_method == 'intra_att':
            inverse_indices, motif_batch, motif_ids = compute_motif_inverse_indices(nodes_to_motifs, batch)
            motif_emb = self.intra_motif_pool(emb, inverse_indices)
            return motif_emb, motif_batch, inverse_indices, motif_ids
        return motif_pooling(emb, nodes_to_motifs, batch, reduce=self.motif_pooling_method)
            
    @staticmethod
    def _calculate_entropy(weights, num_bins=20):
        """Calculate entropy of weight distribution."""
        hist, _ = np.histogram(weights, bins=num_bins, range=(0, 1), density=True)
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log(hist))

    def __loss__(self, att, clf_logits, clf_labels, epoch, nodes_to_motifs, batch,
                 aux_clf_logits=None, motif_batch=None, motif_ids=None,
                 raw_att_for_loss=None):
        """
        Compute the total loss for GSAT training.
        
        Args:
            att: Attention scores (node-level or motif-level depending on method)
            clf_logits: Classifier predictions on original graph
            clf_labels: Ground truth labels
            epoch: Current training epoch
            nodes_to_motifs: Node to motif mapping
            batch: Batch assignment for nodes
            aux_clf_logits: (Optional) Auxiliary classifier predictions from motif graph
            motif_batch: (Optional) [M] batch assignment for motifs, needed for graph-adaptive r
            motif_ids: (Optional) [M] original motif indices, needed for score-based r
            raw_att_for_loss: (Optional) sigmoid(logits) before sampling, used when use_raw_score_loss=True
        
        Returns:
            loss: Total loss value
            loss_dict: Dictionary of individual loss components
        """
        if not self.multi_label:
            clf_logits = clf_logits.squeeze(-1)
        
        pred_loss = self.criterion(clf_logits, clf_labels)

        # if self.motif_r_values is not None and motif_ids is not None:
        #     score_r = self.motif_r_values[motif_ids].unsqueeze(-1)
        #     if self.score_r_schedule == 'interpolate':
        #         # Blend from init_r → score_r using the same decay timing as standard GSAT
        #         global_r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        #         alpha = (global_r - self.final_r) / max(self.init_r - self.final_r, 1e-8)
        #         r = alpha * self.init_r + (1 - alpha) * score_r
        #     elif self.score_r_schedule == 'max':
        #         # Use standard decaying r until it drops below the motif's score
        #         global_r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        #         r = torch.clamp(score_r, min=global_r)
        #     else:
        #         r = score_r
        # elif self.target_k is not None and motif_batch is not None:
        #     # Graph-adaptive r: r_g = target_k / M_g for each graph g
        #     motifs_per_graph = scatter(torch.ones_like(motif_batch, dtype=att.dtype),
        #                                motif_batch, dim=0, reduce='sum')
        #     r_per_motif = self.target_k / motifs_per_graph[motif_batch]
        #     r = r_per_motif.clamp(min=0.01, max=0.99).unsqueeze(-1)
        # else:
        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        att_for_loss = raw_att_for_loss if raw_att_for_loss is not None else att
        info_loss = (att_for_loss * torch.log(att_for_loss/r + 1e-6) +
                     (1-att_for_loss) * torch.log((1-att_for_loss)/(1-r+1e-6) + 1e-6)).mean()

        # <LOSS> Tunable coefficients: pred_loss_coef, info_loss_coef, motif_loss_coef (within-motif), between_motif_coef (spread across motifs)
        pred_loss = pred_loss * self.pred_loss_coef
        if epoch < self.info_warmup_epochs:
            info_loss = att.new_tensor(0.0)
        else:
            info_loss = info_loss * self.info_loss_coef

        within_term = att.new_tensor(0.0)
        between_term = att.new_tensor(0.0)
        needs_motif_structure = (
            self.motif_method == 'loss'
            and nodes_to_motifs is not None
            and (self.motif_loss_coef > 0 or self.between_motif_coef > 0)
        )
        if needs_motif_structure:
            within_var, between_var = motif_consistency_loss(att_for_loss, nodes_to_motifs, batch)
            if self.motif_loss_coef > 0:
                within_term = within_var * self.motif_loss_coef
            if self.between_motif_coef > 0:
                # Maximize between-motif variance → minimize negative between_var
                between_term = -self.between_motif_coef * between_var

        motif_loss = within_term
        loss = pred_loss + info_loss + within_term + between_term

        loss_dict = {
            'loss': loss.item(),
            'pred': pred_loss.item(),
            'info': info_loss.item(),
            'motif_within': within_term.item(),
            'motif_between': between_term.item(),
            'motif_consistency': motif_loss.item(),
        }

        # # Auxiliary motif graph loss (commented out for simplification)
        # if self.motif_method == 'graph' and self.train_motif_graph and aux_clf_logits is not None:
        #     if not self.multi_label:
        #         aux_clf_logits = aux_clf_logits.squeeze(-1)
        #     aux_pred_loss = self.criterion(aux_clf_logits, clf_labels) * self.motif_loss_coef
        #     loss = loss + aux_pred_loss
        #     loss_dict['loss'] = loss.item()
        #     loss_dict['motif_graph_loss'] = aux_pred_loss.item()
        
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        """
        Forward pass through GSAT with different motif incorporation methods.
        
        Methods:
            None: Baseline GSAT (node-level attention, no motif loss)
            'loss': GSAT with motif consistency loss (node-level attention)
            'readout': Motif-level readout (pool embeddings, score motifs, broadcast to nodes)
            'graph': (commented out) Motif-level graph
        """
        # Vanilla GNN: bypass attention entirely
        if self.no_attention:
            clf_logits = self.clf(data.x, data.edge_index, data.batch,
                                 edge_attr=data.edge_attr, edge_atten=None)
            if self.learn_edge_att:
                edge_att = torch.ones(data.edge_index.size(1), 1, device=data.x.device)
            else:
                node_att = torch.ones(data.x.size(0), 1, device=data.x.device)
                edge_att = self.lift_node_att_to_edge_att(node_att, data.edge_index)
            pred_loss = self.criterion(clf_logits, data.y)
            loss = self.pred_loss_coef * pred_loss
            loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': 0.0,
                        'motif_within': 0.0, 'motif_between': 0.0, 'motif_consistency': 0.0}
            return edge_att, loss, loss_dict, clf_logits

        # Check for unmapped nodes if using motif incorporation methods
        needs_motifs = self.motif_method in ['loss', 'readout']
        if needs_motifs:
            nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
            if nodes_to_motifs is None:
                raise ValueError(f"motif_method='{self.motif_method}' requires nodes_to_motifs attribute in data. "
                                f"This dataset may not support motif incorporation methods.")
            check_no_unmapped_nodes(nodes_to_motifs)
        
        aux_clf_logits = None  # Auxiliary predictions from motif graph (if applicable)
        motif_batch = None    # Batch assignment for motifs (used for graph-adaptive r)
        motif_ids = None      # Original motif IDs per pooled motif (used for score-based r)
        raw_att_for_loss = None  # sigmoid(logits) before sampling, used when use_raw_score_loss=True

        if self.motif_method in [None, 'loss']:
            # =================================================================
            # BASELINE / LOSS METHOD: Node-level attention
            # =================================================================
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            att_log_logits = self.extractor(emb, data.edge_index, data.batch)
            att, raw_att_for_loss = sample_node_level_gsat(
                self.sampling, att_log_logits, epoch, training, self.use_raw_score_loss,
            )

            if self.learn_edge_att:
                # Edge-level attention (preserved for reference; not used in streamlined configs).
                # if is_undirected(data.edge_index):
                #     trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                #     trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                #     edge_att = (att + trans_val_perm) / 2
                # else:
                #     edge_att = att
                raise ValueError(
                    "learn_edge_att=True is disabled in the streamlined training path; "
                    "set shared_config['learn_edge_att'] to False."
                )
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

            clf_logits = forward_clf_with_node_attention_injection(
                self.clf, data, att, edge_att, self.learn_edge_att,
                self.w_feat, self.w_message, self.w_readout, edge_attr=data.edge_attr,
            )

        elif self.motif_method == 'readout':
            # =================================================================
            # READOUT METHOD: Pool node embeddings to motif level, score motifs
            # =================================================================
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            
            nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
            if nodes_to_motifs is None:
                raise ValueError("'readout' method requires nodes_to_motifs attribute in data")
            motif_emb, motif_batch, inverse_indices, motif_ids = self._motif_level_pool(
                emb, nodes_to_motifs, data.batch,
            )

            motif_att_log_logits = self.extractor(motif_emb, None, motif_batch)

            if self.motif_prior_node_gate:
                node_att, att, raw_att_for_loss = sample_motif_readout_with_prior_node_gate(
                    self.sampling,
                    motif_att_log_logits,
                    motif_emb,
                    emb,
                    inverse_indices,
                    data.batch,
                    epoch,
                    training,
                    self.use_raw_score_loss,
                    self.motif_prior_node_gate_module,
                    self.motif_prior_detach_alpha,
                    self.motif_prior_detach_z,
                )
            else:
                node_att, att, raw_att_for_loss = sample_motif_readout_branch(
                    self.sampling,
                    motif_att_log_logits,
                    inverse_indices,
                    self.motif_level_sampling,
                    self.motif_level_info_loss,
                    epoch,
                    training,
                    self.use_raw_score_loss,
                )

            edge_att = self.lift_node_att_to_edge_att(node_att, data.edge_index)

            if self.learn_edge_att:
                # clf_logits = self.clf(data.x, data.edge_index, data.batch,
                #                       edge_attr=data.edge_attr, edge_atten=edge_att)
                raise ValueError(
                    "learn_edge_att=True is disabled in the streamlined training path; "
                    "set shared_config['learn_edge_att'] to False."
                )
            clf_logits = forward_clf_with_node_attention_injection(
                self.clf, data, node_att, edge_att, self.learn_edge_att,
                self.w_feat, self.w_message, self.w_readout, edge_attr=data.edge_attr,
            )
            
        # # GRAPH METHOD (commented out for simplification)
        # elif self.motif_method == 'graph':
        #     nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
        #     if nodes_to_motifs is None:
        #         raise ValueError("'graph' method requires nodes_to_motifs attribute in data")
        #     motif_x, motif_edge_index, motif_edge_attr, motif_batch, motif_ids, inverse_indices = \
        #         construct_motif_graph(data.x, data.edge_index, data.edge_attr, nodes_to_motifs, data.batch)
        #     if self.separate_motif_model and self.motif_clf is not None:
        #         motif_emb = self.motif_clf.get_emb(motif_x, motif_edge_index, batch=motif_batch, edge_attr=motif_edge_attr)
        #     else:
        #         motif_emb = self.clf.get_emb(motif_x, motif_edge_index, batch=motif_batch, edge_attr=motif_edge_attr)
        #     motif_att_log_logits = self.extractor(motif_emb, motif_edge_index, motif_batch)
        #     motif_att = self.sampling(motif_att_log_logits, epoch, training)
        #     edge_att = map_motif_att_to_edge_att(motif_att, data.edge_index, inverse_indices)
        #     clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        #     if self.train_motif_graph:
        #         motif_src, motif_dst = motif_edge_index
        #         motif_edge_att = motif_att[motif_src] * motif_att[motif_dst]
        #         if self.separate_motif_model and self.motif_clf is not None:
        #             aux_clf_logits = self.motif_clf(motif_x, motif_edge_index, motif_batch,
        #                                             edge_attr=motif_edge_attr, edge_atten=motif_edge_att)
        #         else:
        #             aux_clf_logits = self.clf(motif_x, motif_edge_index, motif_batch,
        #                                      edge_attr=motif_edge_attr, edge_atten=motif_edge_att)
        #     att = motif_att
        
        else:
            raise ValueError(f"Unknown motif_incorporation_method: {self.motif_method}")

        # Get nodes_to_motifs if available (needed for motif_method='loss')
        nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch, nodes_to_motifs, data.batch,
                                        aux_clf_logits=aux_clf_logits, motif_batch=motif_batch,
                                        motif_ids=motif_ids, raw_att_for_loss=raw_att_for_loss)
        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def _embedding_snapshot_batch(self, data, epoch):
        """
        Return (emb, node_att, motif_emb, motif_imp, batch, motif_batch, motif_global_ids) on CPU for wandb PCA viz.
        motif_global_ids: [M] vocabulary index per pooled motif row (same order as motif_emb).
        motif_* / motif_global_ids are None when nodes_to_motifs is absent.
        """
        # Vanilla GNN: no GSAT attention, but still log node (and motif) embeddings with flat importance for PCA coloring.
        if self.no_attention:
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            node_att = torch.ones(emb.size(0), 1, device=emb.device, dtype=emb.dtype)
            nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
            if nodes_to_motifs is None:
                return emb.detach().cpu(), node_att.detach().cpu(), None, None, data.batch.detach().cpu(), None, None
            motif_emb, motif_batch_vec, inverse_indices, motif_global_ids = self._motif_level_pool(
                emb, nodes_to_motifs, data.batch,
            )
            motif_imp = torch.ones(motif_emb.size(0), 1, device=emb.device, dtype=emb.dtype)
            return (
                emb.detach().cpu(),
                node_att.detach().cpu(),
                motif_emb.detach().cpu(),
                motif_imp.detach().cpu(),
                data.batch.detach().cpu(),
                motif_batch_vec.detach().cpu(),
                motif_global_ids.detach().cpu(),
            )

        if self.motif_method in [None, 'loss']:
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            att_log_logits = self.extractor(emb, data.edge_index, data.batch)
            node_att, _ = sample_node_level_gsat(
                self.sampling, att_log_logits, epoch, False, self.use_raw_score_loss,
            )
            nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
            if nodes_to_motifs is None:
                return emb.detach().cpu(), node_att.detach().cpu(), None, None, data.batch.detach().cpu(), None, None
            motif_emb, motif_batch_vec, inverse_indices, motif_global_ids = self._motif_level_pool(
                emb, nodes_to_motifs, data.batch,
            )
            na = node_att.squeeze(-1)
            motif_imp = scatter(na, inverse_indices, dim=0, reduce='mean', dim_size=motif_emb.size(0))
            return (
                emb.detach().cpu(),
                node_att.detach().cpu(),
                motif_emb.detach().cpu(),
                motif_imp.detach().cpu(),
                data.batch.detach().cpu(),
                motif_batch_vec.detach().cpu(),
                motif_global_ids.detach().cpu(),
            )

        if self.motif_method == 'readout':
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
            if nodes_to_motifs is None:
                return None
            motif_emb, motif_batch_vec, inverse_indices, motif_global_ids = self._motif_level_pool(
                emb, nodes_to_motifs, data.batch,
            )
            motif_att_log_logits = self.extractor(motif_emb, None, motif_batch_vec)
            if self.motif_prior_node_gate:
                node_att, _, _ = sample_motif_readout_with_prior_node_gate(
                    self.sampling,
                    motif_att_log_logits,
                    motif_emb,
                    emb,
                    inverse_indices,
                    data.batch,
                    epoch,
                    False,
                    self.use_raw_score_loss,
                    self.motif_prior_node_gate_module,
                    self.motif_prior_detach_alpha,
                    self.motif_prior_detach_z,
                )
            else:
                node_att, _, _ = sample_motif_readout_branch(
                    self.sampling,
                    motif_att_log_logits,
                    inverse_indices,
                    self.motif_level_sampling,
                    self.motif_level_info_loss,
                    epoch,
                    False,
                    self.use_raw_score_loss,
                )
            na = node_att.squeeze(-1)
            motif_imp = scatter(na, inverse_indices, dim=0, reduce='mean', dim_size=motif_emb.size(0))
            return (
                emb.detach().cpu(),
                node_att.detach().cpu(),
                motif_emb.detach().cpu(),
                motif_imp.detach().cpu(),
                data.batch.detach().cpu(),
                motif_batch_vec.detach().cpu(),
                motif_global_ids.detach().cpu(),
            )

        return None

    @staticmethod
    def _graph_labels_to_bin01(y_graph):
        """[num_graphs] or [num_graphs,1] -> long {0,1}."""
        y = y_graph.view(-1)
        if y.dtype.is_floating_point:
            return (y >= 0.5).long()
        return y.long()

    def _subsample_xy(self, X, imp, rng):
        n = X.shape[0]
        m = self.embedding_viz_max_points
        if n <= m:
            return X, imp
        idx = rng.choice(n, size=m, replace=False)
        return X[idx], imp[idx]

    def _subsample_xy_motif(self, X, imp, motif_ids, rng):
        n = X.shape[0]
        m = self.embedding_viz_max_points
        if n <= m:
            return X, imp, motif_ids
        idx = rng.choice(n, size=m, replace=False)
        return X[idx], imp[idx], motif_ids[idx]

    def _motif_vocab_label(self, mid):
        mid = int(mid)
        if self.motif_list is None or mid < 0 or mid >= len(self.motif_list):
            return f'motif_id_{mid}'
        return str(self.motif_list[mid])

    def _pca_scatter_panel(self, ax, X, imp, title):
        if X.shape[0] < 2:
            ax.set_title(f'{title} (n<2)', fontsize=12)
            ax.axis('off')
            return
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(X)
        imp = np.asarray(imp).reshape(-1)
        imp = np.clip(imp, 0.0, 1.0)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=imp, cmap='viridis', s=12, alpha=0.65, vmin=0.0, vmax=1.0)
        ax.set_title(title + f' (n={X.shape[0]})', fontsize=12)
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='importance')

    def _pca_scatter_motif_panel(self, ax, X, imp, motif_ids_np, title):
        """PCA scatter for motif embeddings; annotate points with motif names (capped); full labels in wandb Table."""
        if X.shape[0] < 2:
            ax.set_title(f'{title} (n<2)', fontsize=12)
            ax.axis('off')
            return None
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(X)
        imp = np.clip(np.asarray(imp).reshape(-1), 0.0, 1.0)
        mids = np.asarray(motif_ids_np).reshape(-1).astype(np.int64)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=imp, cmap='viridis', s=18, alpha=0.65, vmin=0.0, vmax=1.0)
        n_ann = min(len(xy), self.embedding_viz_max_motif_annotations)
        # Readable when exported at embedding_viz_dpi (avoid 2.5–7pt caps that look fine on screen but blur when downloaded).
        fs = max(8.0, min(16.0, 2200.0 / max(n_ann, 1)))
        for i in range(n_ann):
            label = self._motif_vocab_label(mids[i])
            if len(label) > 36:
                label = label[:33] + '…'
            ax.annotate(
                label,
                (xy[i, 0], xy[i, 1]),
                fontsize=fs,
                alpha=0.85,
                ha='center',
                va='center',
            )
        if len(xy) > n_ann:
            ax.set_title(title + f' (n={len(xy)}, labeled {n_ann})', fontsize=12)
        else:
            ax.set_title(title + f' (n={len(xy)})', fontsize=12)
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='importance')
        table_rows = []
        for i in range(len(xy)):
            table_rows.append([
                float(xy[i, 0]),
                float(xy[i, 1]),
                float(imp[i]),
                self._motif_vocab_label(mids[i]),
                int(mids[i]),
            ])
        return table_rows

    @torch.no_grad()
    def log_valid_embedding_viz_wandb(self, valid_loader, epoch, use_edge_attr):
        if self.embedding_viz_every <= 0:
            return
        if self.task_type != 'classification' or self.multi_label or self.num_class != 2:
            return

        buckets = {
            0: {'node_emb': [], 'node_imp': [], 'motif_emb': [], 'motif_imp': [], 'motif_gid': []},
            1: {'node_emb': [], 'node_imp': [], 'motif_emb': [], 'motif_imp': [], 'motif_gid': []},
        }

        self.clf.eval()
        self.extractor.eval()
        n_batches = 0
        # Stop early if we already have enough rows to subsample (limits peak RAM before np.concatenate).
        max_accum = max(self.embedding_viz_max_points * 6, 20000)
        for data in valid_loader:
            if n_batches >= self.embedding_viz_max_batches:
                break
            n_batches += 1
            data = process_data(data, use_edge_attr).to(self.device)
            snap = self._embedding_snapshot_batch(data, epoch)
            if snap is None:
                del data
                continue
            emb, node_att, motif_emb, motif_imp, batch, motif_batch_vec, motif_global_ids = snap
            y_graph_cpu = self._graph_labels_to_bin01(data.y.cpu())
            if y_graph_cpu.numel() != data.num_graphs:
                continue
            node_y = y_graph_cpu[batch.long()]
            na = node_att.view(-1).numpy()
            ne = emb.numpy()
            for cls in (0, 1):
                sel = (node_y == cls).numpy()
                if sel.any():
                    buckets[cls]['node_emb'].append(ne[sel])
                    buckets[cls]['node_imp'].append(na[sel])
            if motif_emb is not None and motif_batch_vec is not None and motif_global_ids is not None:
                mb = motif_batch_vec.long()
                motif_y = y_graph_cpu[mb]
                me = motif_emb.numpy()
                mi = motif_imp.numpy()
                mg = motif_global_ids.numpy()
                for cls in (0, 1):
                    sel = (motif_y == cls).numpy()
                    if sel.any():
                        buckets[cls]['motif_emb'].append(me[sel])
                        buckets[cls]['motif_imp'].append(mi[sel])
                        buckets[cls]['motif_gid'].append(mg[sel])
            del data
            rows = sum(len(a) for c in (0, 1) for a in buckets[c]['node_emb'])
            if rows >= max_accum:
                break

        has_nodes = any(buckets[c]['node_emb'] for c in (0, 1))
        if not has_nodes:
            print(
                f'[WARNING] W&B embedding viz at epoch {epoch}: no node embeddings collected '
                f'(check valid batches: y vs num_graphs, or unsupported motif_method for snapshots).'
            )
            return

        rng = np.random.RandomState((self.random_state or 0) * 100000 + int(epoch))
        any_logged = False

        for cls in (0, 1):
            parts = buckets[cls]
            if not parts['node_emb']:
                continue
            Xn = np.concatenate(parts['node_emb'], axis=0)
            In = np.concatenate(parts['node_imp'], axis=0)
            Xn, In = self._subsample_xy(Xn, In, rng)

            Xm, Im, Gm = None, None, None
            if parts['motif_emb']:
                Xm = np.concatenate(parts['motif_emb'], axis=0)
                Im = np.concatenate(parts['motif_imp'], axis=0)
                Gm = np.concatenate(parts['motif_gid'], axis=0)
                Xm, Im, Gm = self._subsample_xy_motif(Xm, Im, Gm, rng)

            dual = Xm is not None
            fig_w, fig_h = (24, 9) if dual else (12, 9)
            fig, axes = plt.subplots(1, 2 if dual else 1, figsize=(fig_w, fig_h))
            if not dual:
                axes = [axes]
            self._pca_scatter_panel(axes[0], Xn, In, f'Nodes (y={cls})')
            motif_table_rows = None
            if dual:
                motif_table_rows = self._pca_scatter_motif_panel(
                    axes[1], Xm, Im, Gm, f'Motifs (y={cls})',
                )
            fig.suptitle(f'Valid PCA embeddings (epoch {epoch})', fontsize=14, y=1.02)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=self.embedding_viz_dpi,
                bbox_inches='tight',
                pad_inches=0.25,
            )
            plt.close(fig)
            # Materialize PNG bytes before wandb.Image — lazy uploads can read after the BytesIO is gone.
            png_bytes = buf.getvalue()
            try:
                pil_img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
                log_payload = {f'valid/embedding_viz_y{cls}': wandb.Image(pil_img)}
                if motif_table_rows:
                    log_payload[f'valid/motif_emb_table_y{cls}'] = wandb.Table(
                        columns=['pc1', 'pc2', 'importance', 'motif_name', 'motif_id'],
                        data=motif_table_rows,
                    )
                # One log per step avoids W&B merging quirks when image + table share the same step.
                wandb.log(log_payload, step=epoch)
            except Exception as e:
                print(f'[WARNING] wandb embedding viz log failed (epoch {epoch}, y={cls}): {e}')
            else:
                any_logged = True
        if any_logged:
            print(
                f'[INFO] W&B embedding PCA logged at epoch {epoch}: view plots under the run Media panel '
                f'(keys valid/embedding_viz_y0, valid/embedding_viz_y1; tables may also appear under Tables/Artifacts).'
            )
        del buckets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        if self.motif_clf is not None:
            self.motif_clf.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        if self.motif_clf is not None:
            self.motif_clf.train()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                                                                                        all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
                self.save_epoch_metrics(epoch, phase.strip(), all_loss_dict, att_auroc, precision, clf_acc, clf_roc)
                self.save_attention_distributions(epoch, phase.strip(), all_att)
            pbar.set_description(desc)
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        # viz_set = self.get_viz_idx(test_set, self.dataset_name)
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            do_embed_viz = (
                self.embedding_viz_every > 0
                and epoch % self.embedding_viz_every == 0
                and (not self.embedding_viz_skip_epoch0 or epoch > 0)
            )
            if do_embed_viz:
                try:
                    self.log_valid_embedding_viz_wandb(loaders['valid'], epoch, use_edge_attr)
                except Exception as e:
                    print(f'[WARNING] log_valid_embedding_viz_wandb failed: {e}')
            self.writer.add_scalar('gsat_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            # For regression tasks, smaller metrics (MAE/MSE) are better, so use < instead of >
            if self.task_type == 'regression':
                is_better = ((valid_res[main_metric_idx] < metric_dict['metric/best_clf_valid'] and valid_res[main_metric_idx] > 0)
                            or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                and valid_res[4] < metric_dict['metric/best_clf_valid_loss']))
            else:
                is_better = ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                             or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                             and valid_res[4] < metric_dict['metric/best_clf_valid_loss']))
            if (r == self.final_r or self.fix_r) and epoch > 10 and is_better:

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_clf_acc_train': train_res[2], 'metric/best_clf_acc_valid': valid_res[2], 'metric/best_clf_acc_test': test_res[2],
                               'metric/best_clf_roc_train': train_res[3], 'metric/best_clf_roc_valid': valid_res[3], 'metric/best_clf_roc_test': test_res[3],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.clf, Path(self.seed_dir), model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, Path(self.seed_dir), model_name='gsat_att_epoch_' + str(epoch))
                if self.intra_motif_pool is not None:
                    save_checkpoint(self.intra_motif_pool, Path(self.seed_dir), model_name='gsat_intra_motif_pool_epoch_' + str(epoch))
                if self.motif_prior_node_gate_module is not None:
                    save_checkpoint(
                        self.motif_prior_node_gate_module, Path(self.seed_dir),
                        model_name='gsat_motif_prior_node_gate_epoch_' + str(epoch),
                    )
                epochs_without_improvement = 0
            elif (r == self.final_r or self.fix_r) and epoch > 10:
                epochs_without_improvement += 1

            for metric, value in metric_dict.items():
                if not isinstance(value, (int, float)):
                    continue
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'gsat_best/{metric}', value, epoch)
            # Log best prediction performance to wandb for verification
            try:
                wandb.log({
                    'best/clf_roc_train': metric_dict['metric/best_clf_roc_train'],
                    'best/clf_roc_valid': metric_dict['metric/best_clf_roc_valid'],
                    'best/clf_roc_test': metric_dict['metric/best_clf_roc_test'],
                    'best/clf_acc_test': metric_dict['metric/best_clf_acc_test'],
                    'best/clf_valid_loss': metric_dict['metric/best_clf_valid_loss'],
                    'best/epoch': metric_dict['metric/best_clf_epoch'],
                }, step=epoch)
            except Exception:
                pass

            # Early stopping: stop when validation has not improved for patience epochs
            if (self.early_stopping_patience is not None and epoch >= self.early_stopping_min_epochs
                    and epochs_without_improvement >= self.early_stopping_patience):
                print(f'[INFO] Early stopping at epoch {epoch} (no valid improvement for {self.early_stopping_patience} epochs). Best epoch: {metric_dict["metric/best_clf_epoch"]}')
                break

            # Calculate explainer performance every 10 epochs (resource intensive)
            if epoch % 10 == 0 and epoch > 0:
                try:
                    print(f"[INFO] Calculating explainer performance at epoch {epoch}")
                    explainer_metrics = calculate_explainer_performance(
                        self.clf, self.extractor, loaders['valid'], self.device, epoch, self.learn_edge_att
                    )
                    wandb.log(explainer_metrics, step=epoch)
                    # Persist explainer metrics so they can be collected later without retraining
                    metric_dict.update(explainer_metrics)
                    print(f"[INFO] Fidelity-: {explainer_metrics['explainer/fidelity_minus']:.4f}, "
                          f"Fidelity+: {explainer_metrics['explainer/fidelity_plus']:.4f}, "
                          f"Sparsity: {explainer_metrics['explainer/sparsity']:.4f}")
                    print(f"[INFO] Edge Att - Mean: {explainer_metrics['edge_att/mean']:.4f}, "
                          f"Std: {explainer_metrics['edge_att/std']:.4f}, "
                          f"Low(<0.3): {explainer_metrics['edge_att/pct_below_0.3']:.2%}, "
                          f"High(>0.7): {explainer_metrics['edge_att/pct_above_0.7']:.2%}")
                    if 'motif/att_impact_correlation' in explainer_metrics:
                        print(f"[INFO] Motif Att-Impact Correlation: {explainer_metrics['motif/att_impact_correlation']:.4f} "
                              f"(n={explainer_metrics.get('motif/att_impact_n_samples', 0)})")
                except Exception as e:
                    print(f"[WARNING] Failed to calculate explainer performance: {e}")

            # if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
            #     if self.multi_label:
            #         raise NotImplementedError
            #     for idx, tag in viz_set:
            #         self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, Path(self.seed_dir), model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, Path(self.seed_dir), model_name='gsat_att_epoch_' + str(epoch))
                if self.intra_motif_pool is not None:
                    save_checkpoint(self.intra_motif_pool, Path(self.seed_dir), model_name='gsat_intra_motif_pool_epoch_' + str(epoch))
                if self.motif_prior_node_gate_module is not None:
                    save_checkpoint(
                        self.motif_prior_node_gate_module, Path(self.seed_dir),
                        model_name='gsat_motif_prior_node_gate_epoch_' + str(epoch),
                    )

            # ===== Save edge and node scores for the last epoch using small batches =====
            '''
            NOTE: CHECK LOGIC
            '''
            # Skip detailed score export if datasets are not provided (e.g., for paper dataset replication)
            if epoch == self.epochs - 1 and self.datasets is not None:
                
                print(f"[INFO] Computing attention scores using small batch processing")
                
                # Export node and edge scores to jsonl files using small batch processing
                node_jsonl_path = os.path.join(self.seed_dir, 'node_scores.jsonl')
                edge_jsonl_path = os.path.join(self.seed_dir, 'edge_scores.jsonl')
                
                with open(node_jsonl_path, 'w') as node_f, open(edge_jsonl_path, 'w') as edge_f:
                    for split_name in ['train', 'valid', 'test']:
                        dataset = self.datasets[split_name]
                        print(f"[INFO] Processing {split_name} split with {len(dataset)} samples")
                        
                        # Use ordered batch iterator to solve InstanceNorm issues
                        # for batch_data, batch_indices, original_samples, skip_first in create_ordered_batch_iterator(dataset, batch_size=2):
                        for di,data in enumerate(dataset):

                            data = data.to(self.device)
                            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
                            emb = self.clf.get_emb(data.x, data.edge_index, batch, edge_attr=data.edge_attr)

                            # Use the correct attention extraction path based on method
                            if self.motif_method == 'readout':
                                n2m = getattr(data, 'nodes_to_motifs', None)
                                if n2m is not None:
                                    motif_emb, motif_batch, inv_idx, _ = self._motif_level_pool(
                                        emb, n2m, batch)
                                    motif_logits = self.extractor(motif_emb, None, motif_batch)
                                    if self.motif_level_sampling:
                                        motif_att = self.sampling(motif_logits, epoch, training=False)
                                        att = lift_motif_att_to_node_att(motif_att, inv_idx)
                                    else:
                                        node_logits = lift_motif_att_to_node_att(motif_logits, inv_idx)
                                        att = self.sampling(node_logits, epoch, training=False)
                                else:
                                    att_log_logits = self.extractor(emb, data.edge_index, batch)
                                    att = self.sampling(att_log_logits, epoch, training=False)
                            else:
                                att_log_logits = self.extractor(emb, data.edge_index, batch)
                                att = self.sampling(att_log_logits, epoch, training=False)

                            sample_results = {}
                            # Handle edge attention conversion for batch processing
                            if not self.learn_edge_att:

                                sample_results['node_att'] = att.detach().cpu().numpy()
                                
                                sample_results['edge_att'] = self.lift_node_att_to_edge_att(
                                            att, 
                                            data.edge_index.to(self.device)
                                        ).detach().cpu().numpy()
                                sample_results['sample'] = data 

                                # ── Individual node/edge masking impact ──
                                old_prediction = self.clf.forward(data.x, data.edge_index, batch, edge_attr=data.edge_attr)
                                old_pred_val = float(old_prediction.squeeze().detach().cpu().item())
                                nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
                                smiles_val = getattr(data, 'smiles', None)

                                # 1) Individual_node: zero node features + remove edges touching that node
                                for node_idx in range(data.x.size(0)):
                                    masked_x = data.x.clone()
                                    masked_x[node_idx] = 0.0
                                    node_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=data.x.device)
                                    node_mask[node_idx] = True
                                    src, dst = data.edge_index
                                    keep = ~(node_mask[src] | node_mask[dst])
                                    filt_ei = data.edge_index[:, keep]
                                    filt_ea = data.edge_attr[keep] if data.edge_attr is not None else None

                                    new_pred = self.clf.forward(masked_x, filt_ei, batch, edge_attr=filt_ea)
                                    rec = {
                                        'split': split_name, 'graph_idx': di, 'smiles': smiles_val,
                                        'node_index': node_idx,
                                        'motif_index': int(nodes_to_motifs[node_idx].item()) if nodes_to_motifs is not None else -1,
                                        'score': float(att[node_idx].detach().cpu().item()),
                                        'old_prediction': old_pred_val,
                                        'new_prediction': float(new_pred.squeeze().detach().cpu().item()),
                                    }
                                    if not hasattr(self, '_indiv_node_f'):
                                        self._indiv_node_f = open(os.path.join(self.seed_dir, 'Individual_node_node_masking_impact.jsonl'), 'w')
                                    self._indiv_node_f.write(json.dumps(rec) + '\n')

                                # 2) Individual_edge: remove one undirected edge at a time
                                ei = data.edge_index
                                seen_edges = set()
                                for e_idx in range(ei.size(1)):
                                    u, v = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                                    canon = (min(u, v), max(u, v))
                                    if canon in seen_edges:
                                        continue
                                    seen_edges.add(canon)

                                    drop = ~((ei[0] == u) & (ei[1] == v)) & ~((ei[0] == v) & (ei[1] == u))
                                    filt_ei = ei[:, drop]
                                    filt_ea = data.edge_attr[drop] if data.edge_attr is not None else None

                                    new_pred_e = self.clf.forward(data.x, filt_ei, batch, edge_attr=filt_ea)
                                    rec_e = {
                                        'split': split_name, 'graph_idx': di, 'smiles': smiles_val,
                                        'edge_src': u, 'edge_dst': v,
                                        'old_prediction': old_pred_val,
                                        'new_prediction': float(new_pred_e.squeeze().detach().cpu().item()),
                                    }
                                    if not hasattr(self, '_indiv_edge_f'):
                                        self._indiv_edge_f = open(os.path.join(self.seed_dir, 'Individual_edge_node_and_edge_masking_impact.jsonl'), 'w')
                                    self._indiv_edge_f.write(json.dumps(rec_e) + '\n')

                                # ── Motif-level masking impact ──
                                if nodes_to_motifs is not None and self.masked_data_features is not None:
                                    for local_motif in nodes_to_motifs.unique():
                                        local_motif_key = int(local_motif.item()) if hasattr(local_motif, 'item') else int(local_motif)

                                        if local_motif_key not in self.masked_data_features[split_name]:
                                            if local_motif in self.masked_data_features[split_name]:
                                                local_motif_key = local_motif
                                            else:
                                                continue

                                        if di not in self.masked_data_features[split_name][local_motif_key]:
                                            continue

                                        masked_feature_graph = self.masked_data_features[split_name][local_motif_key][di]
                                        masked_feature_graph = masked_feature_graph.to(self.device)

                                        if masked_feature_graph.numel() == 0:
                                            continue

                                        old_prediction_motif = self.clf.forward(data.x, data.edge_index, batch, edge_attr=data.edge_attr)
                                        old_pv = float(old_prediction_motif.squeeze().detach().cpu().item())

                                        # 3) Motif_level_node_masking: zero motif features only
                                        new_prediction = self.clf.forward(masked_feature_graph, data.edge_index, batch, edge_attr=data.edge_attr)
                                        new_pv = float(new_prediction.squeeze().detach().cpu().item())
                                        motif_rec = {
                                            'split': split_name, 'graph_idx': di,
                                            'smiles': data.smiles, 'motif_idx': int(local_motif_key),
                                            'old_prediction': old_pv, 'new_prediction': new_pv,
                                        }
                                        if not hasattr(self, '_motif_node_f'):
                                            self._motif_node_f = open(os.path.join(self.seed_dir, 'Motif_level_node_masking_impact.jsonl'), 'w')
                                        self._motif_node_f.write(json.dumps(motif_rec) + '\n')

                                        # 4) Motif_level_node_and_edge_masking: zero features + remove edges between masked nodes
                                        masked_nodes = (masked_feature_graph.abs().sum(dim=1) == 0) & (data.x.abs().sum(dim=1) > 0)
                                        src, dst = data.edge_index
                                        keep_edge_mask = ~(masked_nodes[src] & masked_nodes[dst])
                                        filtered_edge_index = data.edge_index[:, keep_edge_mask]
                                        filtered_edge_attr = data.edge_attr[keep_edge_mask] if data.edge_attr is not None else None

                                        new_pred_edge = self.clf.forward(masked_feature_graph, filtered_edge_index, batch, edge_attr=filtered_edge_attr)
                                        new_pv_e = float(new_pred_edge.squeeze().detach().cpu().item())
                                        motif_edge_rec = {
                                            'split': split_name, 'graph_idx': di,
                                            'smiles': data.smiles, 'motif_idx': int(local_motif_key),
                                            'old_prediction': old_pv, 'new_prediction': new_pv_e,
                                        }
                                        if not hasattr(self, '_motif_edge_f'):
                                            self._motif_edge_f = open(os.path.join(self.seed_dir, 'Motif_level_node_and_edge_masking_impact.jsonl'), 'w')
                                        self._motif_edge_f.write(json.dumps(motif_edge_rec) + '\n')


                            else:
                                sample_results['edge_att'] = att.detach().cpu().numpy()
                                sample_results['sample'] = data
                                sample_results['node_att'] = None  # edge-level attention only

                            # Save scores for each sample in the batch (requires 'sample', 'node_att', 'edge_att')
                            if 'sample' in sample_results:
                                self.save_sample_scores(sample_results, split_name, di, node_f, edge_f)
                        
                print(f"[INFO] Successfully saved attention scores to {node_jsonl_path} and {edge_jsonl_path}")
                
                # Close all impact files
                for attr, label in [
                    ('_indiv_node_f',  'Individual_node_node_masking_impact.jsonl'),
                    ('_indiv_edge_f',  'Individual_edge_node_and_edge_masking_impact.jsonl'),
                    ('_motif_node_f',  'Motif_level_node_masking_impact.jsonl'),
                    ('_motif_edge_f',  'Motif_level_node_and_edge_masking_impact.jsonl'),
                ]:
                    if hasattr(self, attr):
                        getattr(self, attr).close()
                        print(f"[INFO] Saved {os.path.join(self.seed_dir, label)}")
            # ===== End of saving scores =====
            if self.task_type == 'regression':
                print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                      f'Best Val Pred MAE: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred MAE: {metric_dict["metric/best_clf_test"]:.3f}, '
                      f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            else:
                print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                      f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                      f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
        self.save_final_metrics(metric_dict)
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        
        # Log to wandb (only when not batch)
        if not batch:
            try:
                wandb_metrics = {
                    f'{phase}/loss': loss_dict['loss'],
                    f'{phase}/pred_loss': loss_dict['pred'],
                    f'{phase}/info_loss': loss_dict['info'],
                    f'{phase}/motif_within': loss_dict.get('motif_within', 0),
                    f'{phase}/motif_between': loss_dict.get('motif_between', 0),
                    f'{phase}/motif_loss': loss_dict.get('motif_consistency', 0),
                    f'{phase}/motif_graph_loss': loss_dict.get('motif_graph_loss', 0),
                    f'{phase}/att_auroc': att_auroc if att_auroc is not None else 0,
                    f'{phase}/precision_k': precision if precision is not None else 0,
                    'epoch': epoch,
                }
                
                if self.task_type == 'regression':
                    wandb_metrics[f'{phase}/mae'] = clf_acc if clf_acc is not None else 0
                    wandb_metrics[f'{phase}/mse'] = clf_roc if clf_roc is not None else 0
                else:
                    wandb_metrics[f'{phase}/clf_acc'] = clf_acc if clf_acc is not None else 0
                    wandb_metrics[f'{phase}/clf_roc'] = clf_roc if clf_roc is not None else 0
                
                if phase == 'train':
                    wandb_metrics['learning_rate'] = get_lr(self.optimizer)
                
                # Compute attention quality metrics
                att_quality = self._compute_attention_quality_metrics(att)
                wandb_metrics.update({f'{phase}/{k}': v for k, v in att_quality.items()})
                
                # Per-edge attention histogram (`att` is flattened edge_att from forward_pass)
                h_edge = _wandb_histogram_safe(att)
                if h_edge is not None:
                    wandb_metrics[f'{phase}/edge_att/distribution'] = h_edge
                
                # Log label distributions (true and predicted)
                if self.task_type != 'regression':
                    clf_preds = get_preds(clf_logits, self.multi_label)
                    clf_labels_np = clf_labels.numpy() if hasattr(clf_labels, 'numpy') else clf_labels
                    clf_preds_np = clf_preds.numpy() if hasattr(clf_preds, 'numpy') else clf_preds
                    
                    # Get prediction probabilities for confidence analysis
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    
                    if self.multi_label or self.num_class == 2:
                        # Binary/multi-label: sigmoid probabilities
                        clf_probs = torch.sigmoid(clf_logits_tensor).numpy()
                        # For binary, get probability of positive class
                        if self.num_class == 2:
                            clf_probs = clf_probs.flatten()
                    else:
                        # Multi-class: softmax probabilities, take max confidence
                        clf_probs = torch.softmax(clf_logits_tensor, dim=1).numpy()
                        clf_probs = clf_probs.max(axis=1)  # Max probability (confidence)
                    
                    # Flatten if needed
                    clf_labels_flat = clf_labels_np.flatten()
                    clf_preds_flat = clf_preds_np.flatten()
                    
                    # Compute class distributions
                    valid_mask = ~np.isnan(clf_labels_flat)
                    unique_true, counts_true = np.unique(clf_labels_flat[valid_mask], return_counts=True)
                    unique_pred, counts_pred = np.unique(clf_preds_flat, return_counts=True)
                    
                    # Log distribution statistics
                    total_true = counts_true.sum()
                    total_pred = counts_pred.sum()
                    
                    for cls, cnt in zip(unique_true, counts_true):
                        wandb_metrics[f'{phase}/true_label_class_{int(cls)}_ratio'] = cnt / total_true
                    for cls, cnt in zip(unique_pred, counts_pred):
                        wandb_metrics[f'{phase}/pred_label_class_{int(cls)}_ratio'] = cnt / total_pred
                    
                    # Log class imbalance ratio (max/min class ratio)
                    if len(counts_true) > 1:
                        wandb_metrics[f'{phase}/true_label_imbalance'] = counts_true.max() / (counts_true.min() + 1e-10)
                    if len(counts_pred) > 1:
                        wandb_metrics[f'{phase}/pred_label_imbalance'] = counts_pred.max() / (counts_pred.min() + 1e-10)
                    
                    # Log prediction confidence statistics
                    clf_probs_flat = clf_probs.flatten() if hasattr(clf_probs, 'flatten') else clf_probs
                    wandb_metrics[f'{phase}/pred_confidence_mean'] = float(np.mean(clf_probs_flat))
                    wandb_metrics[f'{phase}/pred_confidence_std'] = float(np.std(clf_probs_flat))
                    wandb_metrics[f'{phase}/pred_confidence_min'] = float(np.min(clf_probs_flat))
                    wandb_metrics[f'{phase}/pred_confidence_max'] = float(np.max(clf_probs_flat))
                    
                    for key, arr in (
                        (f'{phase}/true_label_distribution', clf_labels_flat[valid_mask]),
                        (f'{phase}/pred_label_distribution', clf_preds_flat),
                        (f'{phase}/pred_confidence_distribution', clf_probs_flat),
                    ):
                        h = _wandb_histogram_safe(arr, num_bins=50)
                        if h is not None:
                            wandb_metrics[key] = h
                
                wandb.log(wandb_metrics, step=epoch)
            except Exception:
                pass  # Silently fail if wandb is not initialized
        
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']
    
    def _compute_attention_quality_metrics(self, att):
        """
        Compute attention quality metrics.
        
        - Entropy: Lower = more decisive/peaked attention
        - Gini: Higher = more unequal/sparse attention
        - Sparsity: Fraction of attention values below threshold
        """
        att_np = att.numpy() if torch.is_tensor(att) else att
        att_np = att_np.flatten()
        
        # Clamp to avoid log(0)
        att_np = np.clip(att_np, 1e-10, 1 - 1e-10)
        
        metrics = {}
        
        # Entropy (normalized to [0, 1])
        eps = 1e-12
        att = att.clamp(eps, 1 - eps)

        entropy = -(att * att.log() + (1 - att) * (1 - att).log()).mean()
        max_entropy = np.log(2)  # Maximum entropy for binary
        metrics['att_entropy'] = entropy / max_entropy
        
        # Gini coefficient (measures inequality/sparsity)
        sorted_att = np.sort(att_np)
        n = len(sorted_att)
        if n > 0:
            cumulative = np.cumsum(sorted_att)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_att))) / (n * np.sum(sorted_att) + 1e-10) - (n + 1) / n
            metrics['att_gini'] = max(0, gini)  # Gini should be non-negative
        
        # Sparsity metrics
        metrics['att_sparsity_0.1'] = float(np.mean(att_np < 0.1))  # % below 0.1
        metrics['att_sparsity_0.5'] = float(np.mean(att_np < 0.5))  # % below 0.5
        
        # Polarization: % of attention near 0 or 1 (good for explainability)
        metrics['att_polarization'] = float(np.mean((att_np < 0.2) | (att_np > 0.8)))
        
        return metrics

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        # Keep original labels for OGB evaluator (needs [N, 1] shape)
        clf_labels_orig = clf_labels
        
        if self.task_type == 'regression':
            # For regression, calculate MAE as primary metric, keep MSE as auxiliary
            clf_preds = clf_logits.squeeze()
            clf_targets = clf_labels.squeeze().float()
            mae = F.l1_loss(clf_preds, clf_targets).item()
            mse = F.mse_loss(clf_preds, clf_targets).item()
            clf_acc = mae   # Use MAE as primary accuracy metric for regression
            clf_roc = mse   # Keep MSE as auxiliary metric
        else:
            clf_preds = get_preds(clf_logits, self.multi_label)
            # Squeeze clf_labels to handle [N, 1] vs [N] shape mismatch for accuracy
            clf_labels_squeezed = clf_labels.squeeze() if hasattr(clf_labels, 'squeeze') else clf_labels
            clf_acc = 0 if self.multi_label else (clf_preds == clf_labels_squeezed).sum().item() / clf_labels_squeezed.shape[0]

        if batch:
            if self.task_type == 'regression':
                return f'mae: {clf_acc:.3f}', None, None, None, None
            else:
                return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        
        # FIX: Calculate ROC-AUC properly for all datasets
        if self.task_type == 'regression':
            clf_roc = mse
        else:
            # Try to calculate ROC-AUC for binary/multi-class classification
            try:
                if 'ogb' in self.dataset_name:
                    # OGB evaluator expects [N, 1] shape for both y_pred and y_true
                    evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
                    # Ensure correct shapes for OGB evaluator
                    if isinstance(clf_logits, np.ndarray):
                        y_pred = torch.from_numpy(clf_logits)
                    else:
                        y_pred = clf_logits
                    if isinstance(clf_labels_orig, np.ndarray):
                        y_true = torch.from_numpy(clf_labels_orig)
                    else:
                        y_true = clf_labels_orig
                    # Ensure [N, 1] shape
                    if len(y_pred.shape) == 1:
                        y_pred = y_pred.unsqueeze(1)
                    if len(y_true.shape) == 1:
                        y_true = y_true.unsqueeze(1)
                    clf_roc = evaluator.eval({'y_pred': y_pred, 'y_true': y_true})['rocauc']
                elif self.multi_label:
                    # Multi-label classification
                    # Convert to tensor if needed, then apply sigmoid
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.sigmoid(clf_logits_tensor).numpy()
                    # Ensure clf_labels is numpy array (use original labels for multi-label)
                    clf_labels_np = clf_labels_orig.numpy() if hasattr(clf_labels_orig, 'numpy') else clf_labels_orig
                    # Handle NaN values in labels for multi-label
                    valid_mask = ~np.isnan(clf_labels_np)
                    if valid_mask.any():
                        clf_roc = roc_auc_score(clf_labels_np[valid_mask], clf_probs[valid_mask], average='micro')
                    else:
                        clf_roc = 0
                elif len(np.unique(clf_labels_squeezed)) == 2:
                    # Binary classification
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.sigmoid(clf_logits_tensor.squeeze()).numpy()
                    # Ensure both clf_labels and clf_probs are 1D for roc_auc_score
                    clf_labels_np = clf_labels_squeezed.numpy() if hasattr(clf_labels_squeezed, 'numpy') else clf_labels_squeezed
                    clf_probs = clf_probs.squeeze() if hasattr(clf_probs, 'squeeze') else clf_probs
                    clf_roc = roc_auc_score(clf_labels_np, clf_probs)
                else:
                    # Multi-class classification
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.softmax(clf_logits_tensor, dim=1).numpy()
                    # Ensure clf_labels is 1D numpy array for roc_auc_score
                    clf_labels_np = clf_labels_squeezed.numpy() if hasattr(clf_labels_squeezed, 'numpy') else clf_labels_squeezed
                    clf_roc = roc_auc_score(clf_labels_np, clf_probs, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"[WARNING] Could not calculate ROC-AUC for {phase}: {e}")
                clf_roc = 0

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.writer.add_histogram(f'gsat_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'gsat_{phase}/signal_att_weights', signal_att_weights, epoch)
        if self.task_type == 'regression':
            self.writer.add_scalar(f'gsat_{phase}/mae/', clf_acc, epoch)
            self.writer.add_scalar(f'gsat_{phase}/mse/', clf_roc, epoch)
        else:
            self.writer.add_scalar(f'gsat_{phase}/clf_acc/', clf_acc, epoch)
            self.writer.add_scalar(f'gsat_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, att, epoch)
        if self.task_type == 'regression':
            desc = f'mae: {clf_acc:.3f}, mse: {clf_roc:.3f}, ' + \
                   f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        else:
            desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
                   f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i.to(exp_labels.device)]
            mask_log_logits_for_graph_i = att[edges_for_graph_i.to(exp_labels.device)]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif 'ogbg' in self.dataset_name:
                # OGB node feature x[:, 0] is atomic number index (0-based); +1 for element symbol
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att.to(self.device))

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    
    '''
    NOTE: CHECK LOGIC
    '''
    def save_sample_scores(self, sample_result, split_name, graph_idx,node_f, edge_f):
        """
        Save individual sample attention scores to JSONL files.
        
        This function saves the attention scores for a single molecule sample,
        ensuring all data types are JSON-serializable and maintaining exact
        correspondence with the original dataset.
        
        Args:
            sample_result: Dict containing sample data and attention scores
            split_name: 'train', 'valid', or 'test'
            node_f: File handle for node scores
            edge_f: File handle for edge scores
        """
        import json
        
        sample = sample_result['sample']
        node_att = sample_result['node_att']
        edge_att = sample_result['edge_att']
        
        # Save node scores
        if node_att is not None:
            for local_node_idx in range(len(node_att)):
                node_record = {
                    'split': split_name,
                    'graph_idx': graph_idx,
                    'smiles': sample.smiles,
                    'node_index': local_node_idx,
                    'motif_index': int(sample.nodes_to_motifs[local_node_idx]),
                    'score': float(node_att[local_node_idx])
                }
                node_f.write(json.dumps(node_record) + '\n')
        
        # Save edge scores
        if edge_att is not None:
            for local_edge_idx in range(len(edge_att)):
                source = int(sample.edge_index[0, local_edge_idx])
                target = int(sample.edge_index[1, local_edge_idx])
                
                edge_record = {
                    'split': split_name,
                    'graph_idx': graph_idx,
                    'smiles': sample.smiles,
                    'edge_index': local_edge_idx,
                    'source': source,
                    'target': target,
                    'score': float(edge_att[local_edge_idx])
                }
                edge_f.write(json.dumps(edge_record) + '\n')

    def sampling(self, att_log_logits, epoch, training):
        if training and epoch < self.info_warmup_epochs:
            att = att_log_logits.sigmoid()
        else:
            att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config, input_dim=None):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']
        mult = shared_config.get('extractor_hidden_mult', 1)

        if input_dim is None:
            input_dim = hidden_size

        if self.learn_edge_att:
            self.feature_extractor = MLP([input_dim * 2, hidden_size * 4 * mult, hidden_size * mult, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([input_dim, hidden_size * 2 * mult, hidden_size * mult, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att and edge_index is not None:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def check_artifacts_exist(seed_dir):
    """
    Check if training artifacts already exist to skip retraining.
    Looks for key completion markers that indicate training finished successfully.
    
    Args:
        seed_dir: Path to the seed directory (string or Path)
    """
    seed_dir = Path(seed_dir)
    
    # Check for final checkpoint and metrics files
    checkpoint_patterns = [
        seed_dir / 'final_metrics.json',  # Final metrics saved at end of training
        seed_dir / 'node_scores.jsonl',   # Attention scores saved at last epoch
        seed_dir / 'edge_scores.jsonl',   # Edge scores saved at last epoch
    ]
    
    exists = all(p.exists() for p in checkpoint_patterns)
    if exists:
        print(f"[INFO] ✓ Training already completed")
        print(f"[INFO] ✓ Found artifacts in: {seed_dir}")
        print(f"[INFO] Skipping training. Delete artifacts to retrain.")
    return exists


def _wandb_histogram_safe(values, max_samples=100_000, num_bins=64):
    """
    Build wandb.Histogram from 1D float values. Handles large tensors and API quirks
    (some wandb versions reject certain num_bins or empty inputs).
    """
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    if a.size > max_samples:
        rng = np.random.default_rng(0)
        a = rng.choice(a, size=max_samples, replace=False)
    try:
        return wandb.Histogram(a, num_bins=num_bins)
    except Exception:
        pass
    try:
        return wandb.Histogram(a)
    except Exception:
        pass
    try:
        counts, bin_edges = np.histogram(a, bins=min(num_bins, max(1, a.size)))
        return wandb.Histogram(np_histogram=(counts, bin_edges))
    except Exception:
        return None


def calculate_explainer_performance(model, extractor, data_loader, device, epoch, learn_edge_att=False):
    """
    Calculate explainer performance using edge masking approach.
    
    Fidelity metrics measure how well attention scores identify important edges:
    - Fidelity- : Prediction drop when masking important edges (higher = better)
    - Fidelity+ : Prediction maintained when keeping only important edges (lower = better)
    
    Also collects edge attention distribution statistics for wandb logging.
    
    Args:
        model: The GNN classifier
        extractor: The attention extractor MLP
        data_loader: DataLoader for evaluation
        device: torch device
        epoch: Current epoch number
        learn_edge_att: Whether extractor outputs edge-level attention
    
    Returns:
        Dictionary of metrics for wandb logging
    """
    model.eval()
    extractor.eval()
    
    fidelity_minus = []  # Prediction drop when masking important edges
    fidelity_plus = []   # Prediction similarity when keeping only important edges
    sparsity = []        # Fraction of edges with low attention
    
    # Collect all attention values for distribution analysis
    all_att_values = []
    all_node_att_values = []  # For node-level attention (before edge conversion)
    
    # Motif consistency metrics
    within_motif_variances = []
    between_motif_variances = []
    
    # Per-motif edge weight distribution (min/max per motif, then aggregate)
    per_motif_edge_mins = []
    per_motif_edge_maxs = []
    
    # Motif attention vs impact correlation (sampled for efficiency)
    motif_att_scores = []
    motif_impact_scores = []
    max_batches_for_correlation = 3  # Only sample first N batches for correlation
    
    top_k_ratio = 0.2  # Consider top 20% as "important"
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)
            
            # Get edge attributes if available
            edge_attr = batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
            
            # Get node embeddings
            emb = model.get_emb(batch_data.x, batch_data.edge_index, batch_data.batch, edge_attr=edge_attr)
            
            # Get attention scores
            att_log_logits = extractor(emb, batch_data.edge_index, batch_data.batch)
            att = att_log_logits.sigmoid()
            
            # Store node attention for motif analysis
            node_att = att.squeeze()
            all_node_att_values.append(node_att.cpu().numpy())
            
            # Compute edge attention before motif block (needed for motif edge stats and impact)
            if not learn_edge_att:
                src, dst = batch_data.edge_index
                edge_att = (att[src] * att[dst]).squeeze()
            else:
                edge_att = att.squeeze()
            
            # Compute within-motif consistency, per-motif edge weight min/max, and correlation if motif information available
            if hasattr(batch_data, 'nodes_to_motifs'):
                try:
                    nodes_to_motifs = batch_data.nodes_to_motifs
                    graph_batch = batch_data.batch
                    src, dst = batch_data.edge_index
                    
                    # Create unique (graph_id, motif_id) pairs
                    max_motif_id = nodes_to_motifs.max().item() + 1
                    graph_motif_id = graph_batch * max_motif_id + nodes_to_motifs
                    unique_graph_motifs = graph_motif_id.unique()
                    
                    # Per-motif edge weight min/max (for wandb: distribution of min and max *per motif*)
                    for gm_id in unique_graph_motifs:
                        if gm_id.item() < 0:
                            continue  # skip unmapped nodes (e.g. nodes_to_motifs=-1)
                        node_mask = (graph_motif_id == gm_id)
                        src_in_motif = node_mask[src]
                        dst_in_motif = node_mask[dst]
                        motif_edge_mask = src_in_motif & dst_in_motif
                        if motif_edge_mask.sum() > 0:
                            motif_edges_att = edge_att[motif_edge_mask]
                            per_motif_edge_mins.append(motif_edges_att.min().item())
                            per_motif_edge_maxs.append(motif_edges_att.max().item())
                    
                    # Node-level metrics and correlation only when using node attention
                    if not learn_edge_att:
                        motif_means = []
                        for gm_id in unique_graph_motifs:
                            mask = (graph_motif_id == gm_id)
                            if mask.sum() > 1:
                                motif_att = node_att[mask]
                                within_motif_variances.append(motif_att.var().item())
                                motif_means.append(motif_att.mean().item())
                        if len(motif_means) > 1:
                            between_motif_variances.append(np.var(motif_means))
                        
                        # Motif attention vs impact correlation (sampled)
                        if batch_idx < max_batches_for_correlation:
                            orig_output = model(batch_data.x, batch_data.edge_index, graph_batch,
                                               edge_attr=edge_attr, edge_atten=None)
                            orig_probs = torch.sigmoid(orig_output).squeeze()
                            for gm_id in unique_graph_motifs:
                                node_mask = (graph_motif_id == gm_id)
                                if node_mask.sum() < 1:
                                    continue
                                graph_id = graph_batch[node_mask][0].item()
                                motif_mean_att = node_att[node_mask].mean().item()
                                src_in_motif = node_mask[src]
                                dst_in_motif = node_mask[dst]
                                motif_edge_mask = src_in_motif & dst_in_motif
                                if motif_edge_mask.sum() == 0:
                                    continue
                                masked_edge_att = edge_att.clone()
                                masked_edge_att[motif_edge_mask] = 0.0
                                masked_output = model(batch_data.x, batch_data.edge_index, graph_batch,
                                                     edge_attr=edge_attr,
                                                     edge_atten=masked_edge_att.unsqueeze(-1))
                                masked_prob = torch.sigmoid(masked_output[graph_id]).item()
                                impact = abs(orig_probs[graph_id].item() - masked_prob)
                                motif_att_scores.append(motif_mean_att)
                                motif_impact_scores.append(impact)
                except Exception as exc:
                    import traceback
                    print(f"[WARN] Motif analysis failed on batch {batch_idx}: {exc}")
                    traceback.print_exc()
            
            edge_att = edge_att.squeeze()
            all_att_values.append(edge_att.cpu().numpy())
            
            # Get original prediction (with full attention)
            orig_output = model(batch_data.x, batch_data.edge_index, batch_data.batch, 
                               edge_attr=edge_attr, edge_atten=None)
            
            # Process each graph in the batch
            num_graphs = batch_data.batch.max().item() + 1
            
            for graph_idx in range(num_graphs):
                try:
                    # Get edges for this graph
                    node_mask = batch_data.batch == graph_idx
                    edge_mask = node_mask[batch_data.edge_index[0]] & node_mask[batch_data.edge_index[1]]
                    
                    if edge_mask.sum() == 0:
                        continue
                    
                    graph_edge_att = edge_att[edge_mask]
                    num_edges = graph_edge_att.size(0)
                    
                    if num_edges < 2:
                        continue
                    
                    # Determine threshold for important edges (top k%)
                    k = max(1, int(top_k_ratio * num_edges))
                    threshold = torch.topk(graph_edge_att, k).values[-1]
                    
                    # Create masks for important vs unimportant edges
                    important_mask = graph_edge_att >= threshold
                    
                    # Calculate sparsity (fraction of low-attention edges)
                    sparsity.append((~important_mask).float().mean().item())
                    
                    # Get original prediction for this graph
                    orig_pred = orig_output[graph_idx]
                    if orig_pred.dim() == 0:
                        orig_pred = orig_pred.unsqueeze(0)
                    orig_prob = torch.sigmoid(orig_pred)
                    
                    # Create edge attention masks for the full batch
                    # Fidelity- : Set important edge attention to 0 (mask out important)
                    fid_minus_att = edge_att.clone()
                    fid_minus_att[edge_mask] = torch.where(
                        important_mask,
                        torch.zeros_like(graph_edge_att),
                        graph_edge_att
                    )
                    
                    # Fidelity+ : Set unimportant edge attention to 0 (keep only important)
                    fid_plus_att = edge_att.clone()
                    fid_plus_att[edge_mask] = torch.where(
                        important_mask,
                        graph_edge_att,
                        torch.zeros_like(graph_edge_att)
                    )
                    
                    # Get predictions with masked attention
                    output_minus = model(batch_data.x, batch_data.edge_index, batch_data.batch,
                                        edge_attr=edge_attr, edge_atten=fid_minus_att.unsqueeze(-1))
                    output_plus = model(batch_data.x, batch_data.edge_index, batch_data.batch,
                                       edge_attr=edge_attr, edge_atten=fid_plus_att.unsqueeze(-1))
                    
                    pred_minus = torch.sigmoid(output_minus[graph_idx])
                    pred_plus = torch.sigmoid(output_plus[graph_idx])
                    
                    # Fidelity- : How much does prediction drop when removing important edges?
                    # Higher is better (important edges are truly important)
                    fidelity_minus.append((orig_prob - pred_minus).abs().item())
                    
                    # Fidelity+ : How similar is prediction when keeping only important edges?
                    # Lower is better (important edges are sufficient)
                    fidelity_plus.append((orig_prob - pred_plus).abs().item())
                    
                except Exception as e:
                    continue
    
    # Compute attention distribution statistics
    if all_att_values:
        all_att = np.concatenate(all_att_values)
        att_mean = float(np.mean(all_att))
        att_std = float(np.std(all_att))
        att_median = float(np.median(all_att))
        att_min = float(np.min(all_att))
        att_max = float(np.max(all_att))
        pct_low = float(np.mean(all_att < 0.3))  # % of edges with att < 0.3
        pct_high = float(np.mean(all_att > 0.7))  # % of edges with att > 0.7
        
        att_histogram = _wandb_histogram_safe(all_att, num_bins=50)
    else:
        att_mean = att_std = att_median = att_min = att_max = pct_low = pct_high = 0.0
        att_histogram = None
    
    metrics = {
        'explainer/fidelity_minus': np.mean(fidelity_minus) if fidelity_minus else 0.0,
        'explainer/fidelity_plus': np.mean(fidelity_plus) if fidelity_plus else 0.0,
        'explainer/sparsity': np.mean(sparsity) if sparsity else 0.0,
        'explainer/epoch': epoch,
        # Edge attention distribution
        'edge_att/mean': att_mean,
        'edge_att/std': att_std,
        'edge_att/median': att_median,
        'edge_att/min': att_min,
        'edge_att/max': att_max,
        'edge_att/pct_below_0.3': pct_low,
        'edge_att/pct_above_0.7': pct_high,
    }
    
    if att_histogram is not None:
        metrics['edge_att/distribution'] = att_histogram
    
    # Add motif consistency metrics (critical for motif-based analysis)
    if within_motif_variances:
        metrics['motif/within_variance_mean'] = float(np.mean(within_motif_variances))
        metrics['motif/within_variance_std'] = float(np.std(within_motif_variances))
        # Lower within-motif variance = nodes in same motif have similar attention (good!)
    
    if between_motif_variances:
        metrics['motif/between_variance_mean'] = float(np.mean(between_motif_variances))
        # Higher between-motif variance = different motifs have different attention (good!)
    
    # Motif consistency ratio: high between / low within is ideal
    if within_motif_variances and between_motif_variances:
        within_mean = np.mean(within_motif_variances)
        between_mean = np.mean(between_motif_variances)
        if within_mean > 1e-10:
            metrics['motif/consistency_ratio'] = float(between_mean / within_mean)
        # Higher ratio = better motif-level attention consistency
    
    # Per-motif edge weight distribution (min and max per motif, then aggregate for wandb)
    if per_motif_edge_mins and per_motif_edge_maxs:
        metrics['motif_edge_att/min_mean'] = float(np.mean(per_motif_edge_mins))
        metrics['motif_edge_att/min_std'] = float(np.std(per_motif_edge_mins))
        metrics['motif_edge_att/max_mean'] = float(np.mean(per_motif_edge_maxs))
        metrics['motif_edge_att/max_std'] = float(np.std(per_motif_edge_maxs))
        metrics['motif_edge_att/n_motifs'] = len(per_motif_edge_mins)
    
    # Motif attention vs impact correlation
    # Higher correlation = attention correctly identifies impactful motifs
    if len(motif_att_scores) >= 5:  # Need enough samples for meaningful correlation
        try:
            correlation = np.corrcoef(motif_att_scores, motif_impact_scores)[0, 1]
            if not np.isnan(correlation):
                metrics['motif/att_impact_correlation'] = float(correlation)
                metrics['motif/att_impact_n_samples'] = len(motif_att_scores)
        except Exception:
            pass
    
    return metrics


def train_vanilla_gnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state, fold, task_type='classification'):
    """
    Train a clean vanilla GNN with only prediction loss. Bypasses the GSAT class
    entirely -- no extractor, no attention, no info loss. Only the GNN classifier
    parameters are optimized.
    """
    path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_CREATE"

    gsat_config = local_config.get('GSAT_config', {})
    tuning_id = gsat_config.get('tuning_id', 'default')
    experiment_name = gsat_config.get('experiment_name', 'default_experiment')

    results_base = os.environ.get('RESULTS_DIR', '../tuning_results')
    seed_dir = Path(os.path.join(
        results_base, str(dataset_name), f'model_{model_name}',
        f'experiment_{experiment_name}', f'tuning_{tuning_id}',
        'method_vanilla', 'vanilla',
        f'fold{fold}_seed{random_state}'
    ))

    if check_artifacts_exist(seed_dir):
        try:
            with open(Path(seed_dir) / 'experiment_summary.json', 'r') as f:
                summary = json.load(f)
            with open(Path(seed_dir) / 'final_metrics.json', 'r') as f:
                metric_dict = json.load(f)
            print(f"[INFO] Loaded existing results from {seed_dir}")
            return summary.get('hparams', {}), metric_dict
        except Exception as e:
            print(f"[WARNING] Failed to load existing results: {e}")

    print(f'[INFO] Training clean vanilla GNN: {model_name} on {dataset_name} fold={fold} seed={random_state}')
    set_seed(random_state)

    model_config = local_config['model_config']
    model_config['use_edge_attr'] = False
    data_config = local_config['data_config']
    method_config = local_config['GSAT_config']

    wandb_project = f"GSAT-{dataset_name}"
    wandb_name = f"{model_name}-fold{fold}-seed{random_state}-vanilla_clean"
    try:
        wandb_dir = os.environ.get('WANDB_DIR', '../wandb')
        wandb.init(project=wandb_project, name=wandb_name, dir=wandb_dir,
                   config={'dataset': dataset_name, 'model': model_name, 'fold': fold,
                           'seed': random_state, 'experiment': experiment_name, 'method': 'vanilla_clean'},
                   reinit=True)
    except Exception as e:
        print(f"[WARNING] wandb init failed: {e}")

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loader_result = get_data_loaders(
        data_dir, dataset_name, batch_size, splits, random_state,
        data_config.get('mutag_x', False), fold, path=path,
        dictionary_fold_variant=data_config.get('dictionary_fold_variant', 'nofilter'),
    )

    if len(loader_result) == 9:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features, motif_list = loader_result
    elif len(loader_result) == 8:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features = loader_result
    else:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = loader_result

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)

    use_edge_attr = model_config.get('use_edge_attr', True)
    multi_label = aux_info['multi_label']
    epochs = method_config.get('epochs', 200)
    lr = method_config.get('lr', 1e-3)
    wd = method_config.get('weight_decay', 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    criterion = Criterion(num_class, multi_label, task_type)

    os.makedirs(seed_dir, exist_ok=True)
    summary = {
        'experiment_name': experiment_name,
        'model_name': model_name,
        'dataset': dataset_name,
        'fold': fold,
        'seed': random_state,
        'method': 'vanilla_clean',
        'hparams': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in model_config.items()},
        'loss_coefficients': {'pred_loss_coef': 1.0, 'info_loss_coef': 0.0},
        'weight_distribution_params': {'fix_r': 1.0},
    }
    with open(os.path.join(seed_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    best_metric = {'metric/best_clf_epoch': 0, 'metric/best_clf_valid_loss': float('inf'),
                   'metric/best_clf_train': 0, 'metric/best_clf_valid': 0, 'metric/best_clf_test': 0,
                   'metric/best_clf_acc_train': 0, 'metric/best_clf_acc_valid': 0, 'metric/best_clf_acc_test': 0,
                   'metric/best_clf_roc_train': 0, 'metric/best_clf_roc_valid': 0, 'metric/best_clf_roc_test': 0}

    def run_epoch(loader, phase):
        is_train = (phase == 'train')
        model.train() if is_train else model.eval()
        all_logits, all_labels, total_loss = [], [], 0.0
        for data in loader:
            data = process_data(data, use_edge_attr)
            data = data.to(device)
            if is_train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(is_train):
                logits = model(data.x, data.edge_index, data.batch,
                               edge_attr=data.edge_attr, edge_atten=None)
                if not multi_label:
                    logits = logits.squeeze(-1)
                loss = criterion(logits, data.y)
                if is_train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(data.y.detach().cpu())
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / len(loader)
        preds = get_preds(all_logits, multi_label)
        labels_sq = all_labels.squeeze()
        acc = (preds == labels_sq).float().mean().item() if not multi_label else 0.0
        try:
            if num_class == 2 and not multi_label:
                probs = torch.sigmoid(all_logits).numpy()
                roc = roc_auc_score(all_labels.numpy(), probs)
            else:
                roc = 0.0
        except Exception:
            roc = 0.0
        return avg_loss, acc, roc

    for epoch in range(epochs):
        train_loss, train_acc, train_roc = run_epoch(loaders['train'], 'train')
        val_loss, val_acc, val_roc = run_epoch(loaders['valid'], 'valid')
        test_loss, test_acc, test_roc = run_epoch(loaders['test'], 'test')

        scheduler.step(val_roc)

        try:
            wandb.log({'train/loss': train_loss, 'train/roc': train_roc, 'train/acc': train_acc,
                       'valid/loss': val_loss, 'valid/roc': val_roc, 'valid/acc': val_acc,
                       'test/loss': test_loss, 'test/roc': test_roc, 'test/acc': test_acc}, step=epoch)
        except Exception:
            pass

        if epoch > 5 and val_roc > best_metric['metric/best_clf_roc_valid']:
            best_metric = {
                'metric/best_clf_epoch': epoch,
                'metric/best_clf_valid_loss': val_loss,
                'metric/best_clf_train': train_acc, 'metric/best_clf_valid': val_acc, 'metric/best_clf_test': test_acc,
                'metric/best_clf_acc_train': train_acc, 'metric/best_clf_acc_valid': val_acc, 'metric/best_clf_acc_test': test_acc,
                'metric/best_clf_roc_train': train_roc, 'metric/best_clf_roc_valid': val_roc, 'metric/best_clf_roc_test': test_roc,
            }
            save_checkpoint(model, seed_dir, model_name=f'best_model')

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f'  [Epoch {epoch:3d}] train_roc={train_roc:.4f} val_roc={val_roc:.4f} test_roc={test_roc:.4f} '
                  f'best_val_roc={best_metric["metric/best_clf_roc_valid"]:.4f} (ep{best_metric["metric/best_clf_epoch"]})')

    with open(os.path.join(seed_dir, 'final_metrics.json'), 'w') as f:
        json.dump(best_metric, f, indent=2)
    with open(os.path.join(seed_dir, 'node_scores.jsonl'), 'w') as f:
        pass
    with open(os.path.join(seed_dir, 'edge_scores.jsonl'), 'w') as f:
        pass

    print(f'[INFO] Vanilla GNN training complete. Best val ROC={best_metric["metric/best_clf_roc_valid"]:.4f} at epoch {best_metric["metric/best_clf_epoch"]}')
    print(f'[INFO] Artifacts saved to {seed_dir}')

    try:
        wandb.finish()
    except Exception:
        pass

    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in model_config.items()}
    return hparam_dict, best_metric


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state,  fold, task_type='classification'):
    
    # path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DICTIONARY"
    path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_CREATE"
    
    # Build the deterministic seed_dir path to check for artifacts
    gsat_config = local_config.get('GSAT_config', {})
    tuning_id = gsat_config.get('tuning_id', 'default')
    experiment_name = gsat_config.get('experiment_name', 'default_experiment')
    
    pred_coef = gsat_config.get('pred_loss_coef', 1.0)
    info_coef = gsat_config.get('info_loss_coef', 1.0)
    motif_coef = gsat_config.get('motif_loss_coef', 0.0)
    between_coef = gsat_config.get('between_motif_coef', 0.0)
    init_r = gsat_config.get('init_r', 0.9)
    final_r = gsat_config.get('final_r', 0.7)
    decay_r = gsat_config.get('decay_r', 0.1)
    
    # Get motif incorporation method settings
    motif_method = gsat_config.get('motif_incorporation_method', None)
    train_motif_graph = gsat_config.get('train_motif_graph', False)
    separate_motif_model = gsat_config.get('separate_motif_model', False)
    motif_method_str = str(motif_method) if motif_method else 'none'
    train_motif_str = 'trainmotif' if train_motif_graph else 'notrain'
    separate_model_str = 'separate' if separate_motif_model else 'shared'

    # Mirror the type coercions from GSAT.__init__ so the pre-check path
    # matches what the class actually creates on disk.
    # Order must match GSAT.__init__: no_attention (line 915) then motif_method (line 941).
    no_attention = gsat_config.get('no_attention', False)
    if no_attention:
        info_coef = 0
        motif_coef = 0
    # Must match GSAT.__init__: when motif_method is None, both motif and between are forced to 0.0
    # (float). Using int 0 for between_coef here produced ...between0 while GSAT wrote ...between0.0,
    # so check_artifacts_exist pointed at a non-existent directory and every rerun trained from scratch.
    if motif_method is None:
        motif_coef = 0.0
        between_coef = 0.0

    # Use environment variable if set (for HPC), otherwise use relative path
    results_base = os.environ.get('RESULTS_DIR', '../tuning_results')

    seed_dir = os.path.join(
        results_base,
        str(dataset_name),
        f'model_{model_name}',
        f'experiment_{experiment_name}',
        f'tuning_{tuning_id}',
        f'method_{motif_method_str}_{train_motif_str}_{separate_model_str}',
        f'pred{pred_coef}_info{info_coef}_motif{motif_coef}_between{between_coef}',
        f'init{init_r}_final{final_r}_decay{decay_r}',
        f'fold{fold}_seed{random_state}'
    )
    
    # Check if artifacts already exist
    if check_artifacts_exist(seed_dir):
        # Load and return existing results
        try:
            with open(Path(seed_dir) / 'experiment_summary.json', 'r') as f:
                summary = json.load(f)
            with open(Path(seed_dir) / 'final_metrics.json', 'r') as f:
                metric_dict = json.load(f)
            print(f"[INFO] Loaded existing results from {seed_dir}")
            return summary.get('hparams', {}), metric_dict
        except Exception as e:
            print(f"[WARNING] Failed to load existing results: {e}")
            print(f"[INFO] Proceeding with training...")
    
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')
    
    # Initialize wandb
    wandb_project = f"GSAT-{dataset_name}"
    wandb_name = f"{model_name}-fold{fold}-seed{random_state}-{tuning_id}"
    
    try:
        # Use environment variable if set (for HPC), otherwise use relative path
        wandb_dir = os.environ.get('WANDB_DIR', '../wandb')
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            dir=wandb_dir,  # Store wandb logs (uses env var on HPC)
            config={
                'dataset': dataset_name,
                'model': model_name,
                'fold': fold,
                'seed': random_state,
                'tuning_id': tuning_id,
                'experiment_name': experiment_name,
                **local_config.get('GSAT_config', {}),
                **local_config.get('shared_config', {}),
                **local_config.get('model_config', {}),
                **local_config.get('data_config', {}),
                **local_config.get(f'{method_name}_config', {})
            },
            reinit=True
        )
        print(f"[INFO] Initialized wandb: {wandb_project}/{wandb_name}")
    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb: {e}")
        print(f"[INFO] Continuing without wandb...")

    set_seed(random_state)

    model_config = local_config['model_config']
    model_config['use_edge_attr'] = False
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name
    
    method_config['model_name'] = model_name
    
    # if not use_motif_loss:
    #     method_config['motif_loss_coef'] = 0.0

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    
    # get_data_loaders returns different number of values depending on dataset:
    # - Paper datasets (ba_2motifs, mutag, etc.): 6 values
    # - Molecular datasets with motif info (BBBP, Mutagenicity, etc.): 9 values
    loader_result = get_data_loaders(
        data_dir, dataset_name, batch_size, splits, random_state,
        data_config.get('mutag_x', False), fold, path=path,
        dictionary_fold_variant=data_config.get('dictionary_fold_variant', 'nofilter'),
    )
    
    if len(loader_result) == 9:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features, motif_list = loader_result
    elif len(loader_result) == 8:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features = loader_result
        motif_list = None
    else:
        # Paper datasets don't have motif information
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = loader_result
        datasets = None
        masked_data_features = None
        motif_list = None

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    # Create separate motif model if configured (for 'graph' method)
    motif_clf = None
    # if separate_motif_model and motif_method == 'graph':
    #     print('[INFO] Creating separate GNN for motif graph processing...')
    #     motif_clf = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
        # Note: motif_clf is trained from scratch (no pretraining on motif graphs available)

    hidden_size = model_config['hidden_size']
    motif_pool = method_config.get('motif_pooling_method', 'mean')
    if motif_pool == 'multi':
        extractor_input_dim = hidden_size * 3
    elif motif_pool == 'intra_att':
        extractor_input_dim = hidden_size * 3 + 1
    else:
        extractor_input_dim = hidden_size
    extractor = ExtractorMLP(hidden_size, shared_config, input_dim=extractor_input_dim).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)

    intra_motif_pool = None
    if motif_pool == 'intra_att':
        intra_motif_pool = IntraMotifAttentionPool(hidden_size).to(device)

    motif_prior_node_gate_module = None
    if method_config.get('motif_prior_node_gate', False):
        if method_config.get('motif_incorporation_method') != 'readout':
            raise ValueError('motif_prior_node_gate requires motif_incorporation_method=readout')
        motif_prior_node_gate_module = MotifPriorNodeGateMLP(
            hidden_size, extractor_input_dim, shared_config,
        ).to(device)

    # Build parameter list for optimizer
    params_to_optimize = list(extractor.parameters()) + list(model.parameters())
    if intra_motif_pool is not None:
        params_to_optimize += list(intra_motif_pool.parameters())
    if motif_prior_node_gate_module is not None:
        params_to_optimize += list(motif_prior_node_gate_module.parameters())
        if method_config.get('motif_level_sampling'):
            print('[WARNING] motif_prior_node_gate: motif_level_sampling is ignored (always node-level gate).')
    # if motif_clf is not None:
    #     params_to_optimize += list(motif_clf.parameters())
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(
        model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class,
        aux_info['multi_label'], random_state, method_config, shared_config, fold, task_type,
        datasets, masked_data_features, motif_clf=motif_clf, motif_list=motif_list,
        intra_motif_pool=intra_motif_pool, motif_prior_node_gate_module=motif_prior_node_gate_module,
    )

    # Save model_config to seed_dir for post-hoc analysis (hidden_size, n_layers, etc.)
    model_config_save = {k: v for k, v in model_config.items()
                         if not isinstance(v, torch.Tensor)}
    with open(os.path.join(gsat.seed_dir, "model_config.yaml"), "w") as f:
        yaml.safe_dump(model_config_save, f, sort_keys=False)

    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', False))
    scalar_metric_dict = {k: v for k, v in metric_dict.items() if isinstance(v, (int, float))}
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=scalar_metric_dict)
    
    # Note: Artifacts are saved automatically by GSAT class to seed_dir
    # - experiment_summary.json (in GSAT.__init__)
    # - final_metrics.json (in GSAT.save_final_metrics)
    # - node_scores.jsonl and edge_scores.jsonl (at last epoch)
    print(f"[INFO] All artifacts saved to {gsat.seed_dir}")
    
    # Finish wandb
    try:
        wandb.finish()
    except:
        pass
    
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--fold', type=int, help='fold number to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides global config if provided)')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'], help='task type: classification or regression')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--learn_edge_att', action='store_true', default=None,
                        help='Use edge-level attention (extractor outputs per-edge attention). '
                             'If not set, uses config default (node-level attention).')
    parser.add_argument('--no_learn_edge_att', action='store_true', default=False,
                        help='Force node-level attention (overrides config).')
    parser.add_argument('--motif_incorporation_method', type=str, default=None,
                        choices=[None, 'loss', 'readout', 'graph'],
                        help='Method for incorporating motif information: '
                             'None=baseline GSAT (no motif), '
                             'loss=motif consistency loss, '
                             'readout=motif-level attention readout, '
                             'graph=motif-level graph construction')
    parser.add_argument('--train_motif_graph', action='store_true', default=False,
                        help='For graph method: also train classifier on motif graph (auxiliary loss)')
    parser.add_argument('--separate_motif_model', action='store_true', default=False,
                        help='For graph method: use separate GNN for motif graph processing (vs shared parameters)')
    parser.add_argument('--motif_level_info_loss', action='store_true', default=False,
                        help='Compute info_loss at motif level (1 term per motif) instead of node level (avoids size-weighting bias)')
    parser.add_argument('--motif_level_sampling', action='store_true', default=False,
                        help='For motif_method=readout: sample at motif level then broadcast to nodes. '
                             'Default (False): lift motif logits to nodes, sample per-node like base GSAT.')
    parser.add_argument('--motif_pooling_method', type=str, default=None,
                        choices=['mean', 'sum', 'multi', 'intra_att'],
                        help='Pool node embeddings to motif rows for readout (overrides config if set).')
    parser.add_argument('--motif_prior_node_gate', action='store_true', default=False,
                        help='Readout: node gate g_i = f([h_i || z_m || α_m]) with motif prior α_m = σ(extractor); '
                             'then α_i = σ(GSAT sampling(g_i)).')
    parser.add_argument('--motif_prior_detach_alpha', action='store_true', default=False,
                        help='With motif_prior_node_gate: detach α_m from gate input (no grad through prior).')
    parser.add_argument('--motif_prior_detach_z', action='store_true', default=False,
                        help='With motif_prior_node_gate: detach motif embedding z_m from gate input.')
    parser.add_argument('--use_raw_score_loss', action='store_true', default=False,
                        help='Use sigmoid(logits) before sampling for info_loss and motif_consistency_loss.')
    parser.add_argument('--w_feat', action='store_true', default=False,
                        help='Weight node features by attention before message passing (learn_edge_att=False only).')
    parser.add_argument('--w_message', action='store_true', default=True,
                        help='Weight messages by edge attention during message passing (default GSAT behavior).')
    parser.add_argument('--no_w_message', dest='w_message', action='store_false',
                        help='Disable message weighting by attention.')
    parser.add_argument('--w_readout', action='store_true', default=False,
                        help='Weight node embeddings by attention at readout before pooling (learn_edge_att=False only).')
    parser.add_argument('--no_attention', action='store_true', default=False,
                        help='Vanilla GNN mode: bypass attention entirely, run classifier with edge_atten=None.')
    parser.add_argument('--target_k', type=float, default=None,
                        help='Graph-adaptive r: r_g = target_k / M_g (expected number of important motifs per graph). '
                             'Overrides fix_r/final_r for motif-level methods.')
    parser.add_argument('--motif_scores_path', type=str, default=None,
                        help='Path to CSV with pre-computed motif importance scores to use as per-motif r values. '
                             'Overrides target_k and fix_r/final_r.')
    parser.add_argument('--score_r_schedule', type=str, default=None,
                        choices=[None, 'interpolate', 'max'],
                        help='Schedule for score-based r: interpolate=blend init_r→score_r, '
                             'max=use max(score_r, decaying_global_r), None=fixed from epoch 0')
    parser.add_argument('--run_test_graphs', type=int, default=0,
                        help='Run N graphs through the pipeline and save detailed outputs to file (0 = disabled)')
    parser.add_argument('--embedding_viz_every', type=int, default=None,
                        help='If set, log valid-set PCA embedding scatters (nodes + motifs) to wandb every N epochs; '
                             'binary non-multilabel classification only. Overrides shared_config value.')
    parser.add_argument('--config', type=str, default=None,
                   help='Path to tuning config file')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda
    fold = args.fold
    task_type = args.task

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    # Load tuning config if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            tuning_config = yaml.safe_load(f)
        
        # Merge into local_config['GSAT_config']
        for key, value in tuning_config.items():
            local_config['GSAT_config'][key] = value

    # Override with command-line arguments (take precedence over config file)
    if args.motif_incorporation_method is not None:
        local_config['GSAT_config']['motif_incorporation_method'] = args.motif_incorporation_method
    elif 'motif_incorporation_method' not in local_config['GSAT_config']:
        local_config['GSAT_config']['motif_incorporation_method'] = None
    
    if args.train_motif_graph:
        local_config['GSAT_config']['train_motif_graph'] = True
    elif 'train_motif_graph' not in local_config['GSAT_config']:
        local_config['GSAT_config']['train_motif_graph'] = False
    
    if args.separate_motif_model:
        local_config['GSAT_config']['separate_motif_model'] = True
    elif 'separate_motif_model' not in local_config['GSAT_config']:
        local_config['GSAT_config']['separate_motif_model'] = False
    
    if args.motif_level_info_loss:
        local_config['GSAT_config']['motif_level_info_loss'] = True
    elif 'motif_level_info_loss' not in local_config['GSAT_config']:
        local_config['GSAT_config']['motif_level_info_loss'] = False
    
    if args.motif_level_sampling:
        local_config['GSAT_config']['motif_level_sampling'] = True
    elif 'motif_level_sampling' not in local_config['GSAT_config']:
        local_config['GSAT_config']['motif_level_sampling'] = False

    if args.motif_pooling_method is not None:
        local_config['GSAT_config']['motif_pooling_method'] = args.motif_pooling_method

    if args.motif_prior_node_gate:
        local_config['GSAT_config']['motif_prior_node_gate'] = True
    if args.motif_prior_detach_alpha:
        local_config['GSAT_config']['motif_prior_detach_alpha'] = True
    if args.motif_prior_detach_z:
        local_config['GSAT_config']['motif_prior_detach_z'] = True

    if 'use_raw_score_loss' not in local_config['GSAT_config']:
        local_config['GSAT_config']['use_raw_score_loss'] = args.use_raw_score_loss

    if 'no_attention' not in local_config['GSAT_config']:
        local_config['GSAT_config']['no_attention'] = args.no_attention

    if 'w_feat' not in local_config['GSAT_config']:
        local_config['GSAT_config']['w_feat'] = args.w_feat
    if 'w_message' not in local_config['GSAT_config']:
        local_config['GSAT_config']['w_message'] = args.w_message
    if 'w_readout' not in local_config['GSAT_config']:
        local_config['GSAT_config']['w_readout'] = args.w_readout

    if args.target_k is not None:
        local_config['GSAT_config']['target_k'] = args.target_k
    elif 'target_k' not in local_config['GSAT_config']:
        local_config['GSAT_config']['target_k'] = None
    
    if args.motif_scores_path is not None:
        local_config['GSAT_config']['motif_scores_path'] = args.motif_scores_path
    elif 'motif_scores_path' not in local_config['GSAT_config']:
        local_config['GSAT_config']['motif_scores_path'] = None
    
    if args.score_r_schedule is not None:
        local_config['GSAT_config']['score_r_schedule'] = args.score_r_schedule
    elif 'score_r_schedule' not in local_config['GSAT_config']:
        local_config['GSAT_config']['score_r_schedule'] = None
    
    if args.learn_edge_att:
        local_config['shared_config']['learn_edge_att'] = True
    if args.no_learn_edge_att:
        local_config['shared_config']['learn_edge_att'] = False

    if args.embedding_viz_every is not None:
        local_config['shared_config']['embedding_viz_every'] = int(args.embedding_viz_every)

    print(f'[INFO] Motif incorporation method: {local_config["GSAT_config"].get("motif_incorporation_method", None)}')
    print(f'[INFO] Learn edge attention: {local_config["shared_config"].get("learn_edge_att", False)}')
    print(f'[INFO] Train motif graph: {local_config["GSAT_config"].get("train_motif_graph", False)}')
    print(f'[INFO] Separate motif model: {local_config["GSAT_config"].get("separate_motif_model", False)}')
    print(f'[INFO] Motif-level info loss: {local_config["GSAT_config"].get("motif_level_info_loss", False)}')
    print(f'[INFO] Motif-level sampling: {local_config["GSAT_config"].get("motif_level_sampling", False)}')
    print(f'[INFO] Motif pooling method: {local_config["GSAT_config"].get("motif_pooling_method", "mean")}')
    print(f'[INFO] Target k (graph-adaptive r): {local_config["GSAT_config"].get("target_k", None)}')
    print(f'[INFO] Motif scores path: {local_config["GSAT_config"].get("motif_scores_path", None)}')

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    # Handle --run_test_graphs mode (run N graphs through pipeline and exit)
    if args.run_test_graphs > 0:
        print(f"\n[INFO] Running test graph pipeline mode with {args.run_test_graphs} graphs")
        
        # Load data
        data_config = local_config['data_config']
        model_config = local_config['model_config']
        batch_size = data_config['batch_size']
        splits = data_config.get('splits', None)

        path = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MotifBreakdown/DICTIONARY_NOFILTER"
        
        loader_result = get_data_loaders(
            data_dir, dataset_name, batch_size, splits, 0,
            data_config.get('mutag_x', False), fold, path=path,
            dictionary_fold_variant=data_config.get('dictionary_fold_variant', 'nofilter'),
        )
        if len(loader_result) == 9:
            loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, _, _ = loader_result
        else:
            loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, _ = loader_result
        
        # Create model and extractor
        model_config['deg'] = aux_info['deg']
        model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
        _hs = model_config['hidden_size']
        _mp = local_config.get('GSAT_config', {}).get('motif_pooling_method', 'mean')
        if _mp == 'multi':
            _ext_in = _hs * 3
        elif _mp == 'intra_att':
            _ext_in = _hs * 3 + 1
        else:
            _ext_in = _hs
        extractor = ExtractorMLP(_hs, local_config['shared_config'], input_dim=_ext_in).to(device)
        
        # Create output file
        output_file = f'test_pipeline_output_{dataset_name}_{model_name}_fold{fold}.txt'
        
        # Run test pipeline
        run_test_graphs_pipeline(
            n_graphs=args.run_test_graphs,
            model=model,
            extractor=extractor,
            data_loader=loaders['valid'],  # Use validation set for testing
            device=device,
            output_file=output_file,
            model_config=model_config
        )
        
        print(f"\n[INFO] Test pipeline complete. Output saved to: {output_file}")
        return  # Exit after test mode

    # If seed is specified via command line, use it; otherwise use range from config
    if args.seed is not None:
        seeds_to_run = [args.seed]
    else:
        seeds_to_run = range(num_seeds)

    metric_dicts = []
    for random_state in seeds_to_run:
        print('=' * 80)
        print(f'STARTING SEED {random_state}')
        print('=' * 80)
        
        # log_dir is not used anymore since GSAT creates its own deterministic seed_dir
        # Keeping for backward compatibility with pretrain_clf
        log_dir = data_dir / f'{dataset_name}-fold{fold}' / 'logs' / f'{model_name}-seed{random_state}-{method_name}'
        
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, fold, task_type)
        metric_dicts.append(metric_dict)

    # Summary stats saved by analyze_tuning_results.py
    # Individual runs are in tuning_results/, no need for additional summary here
    if len(metric_dicts) > 1:
        log_dir = data_dir / f'{dataset_name}-fold{fold}' / 'logs' / f'{model_name}-seed_summary-{method_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = Writer(log_dir=log_dir)
        write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
