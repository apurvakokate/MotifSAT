import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import os
import pdb
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score, mean_squared_error
from rdkit import Chem
import wandb

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict

from torch_geometric.utils import scatter  # or from torch_scatter import scatter


def motif_consistency_loss(att, nodes_to_motifs):
    """
    att: [N, 1] or [N] node attentions, same device as nodes_to_motifs
    nodes_to_motifs: [N] LongTensor with motif id per node
    """
    # Flatten attentions to [N]
    att = att.view(-1)
    device = att.device

    # Make sure types/devices line up
    nodes_to_motifs = nodes_to_motifs.to(device=device, dtype=torch.long)

    unique_motifs = nodes_to_motifs.unique()
    total_loss = att.new_tensor(0.0)
    motif_count = 0

    for mid in unique_motifs:
        mask = (nodes_to_motifs == mid)
        num_nodes = mask.sum()
        if num_nodes <= 1:
            continue  # no variance within this motif

        vals = att[mask]               # [k]
        mean_val = vals.mean()         # scalar
        total_loss = total_loss + (vals - mean_val).pow(2).mean()
        motif_count += 1

    if motif_count == 0:
        return att.new_tensor(0.0)

    return total_loss / motif_count

def create_ordered_batch_iterator(dataset, batch_size=2):
    """
    Create ordered batch iterator from dataset to ensure data correspondence.
    
    This function creates small batches (batch_size>=2) to solve InstanceNorm issues
    while maintaining exact correspondence with the original dataset order.
    
    Args:
        dataset: Original dataset (PyG InMemoryDataset)
        batch_size: Size of each batch (default=2, minimum for InstanceNorm)
        
    Yields:
        batch_data: PyG Batch object containing multiple molecules
        batch_indices: List of original dataset indices  
        original_samples: List of original Data objects
        skip_first: Whether to skip the first sample in results (for odd-sized batches)
    """
    for i in range(0, len(dataset), batch_size):
        batch_samples = []
        batch_indices = []
        
        # Collect consecutive samples
        for j in range(i, min(i + batch_size, len(dataset))):
            sample = dataset[j]
            batch_samples.append(sample)
            batch_indices.append(j)
        
        # Handle last batch with only 1 sample
        if len(batch_samples) == 1 and i > 0:
            # Add previous sample to make batch_size=2 (to satisfy InstanceNorm)
            prev_sample = dataset[i-1]
            batch_samples = [prev_sample, batch_samples[0]]
            batch_indices = [i-1, i]
            skip_first = True  # Skip the duplicated first sample in results
        else:
            skip_first = False
        
        # Create PyG batch
        try:
            batch_data = Batch.from_data_list(batch_samples)
            yield batch_data, batch_indices, batch_samples, skip_first
        except Exception as e:
            print(f"Error creating batch at indices {batch_indices}: {e}")
            # If batch creation fails, process individually with padding
            for sample, idx in zip(batch_samples, batch_indices):
                if idx > 0:
                    # Create batch with previous sample to satisfy InstanceNorm
                    padded_batch = Batch.from_data_list([dataset[idx-1], sample])
                    yield padded_batch, [idx-1, idx], [dataset[idx-1], sample], True
                else:
                    # For first sample, duplicate it
                    padded_batch = Batch.from_data_list([sample, sample])
                    yield padded_batch, [idx, idx], [sample, sample], True


def parse_batch_attention_to_samples(batch_att, batch_data, original_samples, batch_indices, skip_first, learn_edge_att):
    """
    Parse batch attention results into individual sample results.
    
    This function takes the attention results from a batch and splits them back
    into individual molecule attention scores, maintaining exact correspondence
    with the original dataset samples.
    
    Args:
        batch_att: Attention tensor from batch processing
        batch_data: PyG Batch object 
        original_samples: List of original Data objects
        batch_indices: List of dataset indices
        skip_first: Whether to skip first sample (for duplicated samples)
        learn_edge_att: Whether learning edge attention
        
    Returns:
        List of dicts containing individual sample results
    """
    # Convert batch attention to individual attention arrays
    if learn_edge_att:
        batch_edge_att = batch_att.detach().cpu().numpy()
        batch_node_att = None
    else:
        batch_node_att = batch_att.detach().cpu().numpy()
        batch_edge_att = None  # Will be calculated per sample to avoid batch edge_index issues
    
    results = []
    node_ptr = 0
    edge_ptr = 0
    
    # Skip first sample if needed (for duplicated samples)
    start_idx = 1 if skip_first else 0
    
    for i in range(start_idx, len(original_samples)):
        sample = original_samples[i]
        dataset_idx = batch_indices[i]
        
        num_nodes = sample.x.shape[0]
        num_edges = sample.edge_index.shape[1]
        
        # Extract attention scores for current sample
        if batch_node_att is not None:
            sample_node_att = batch_node_att[node_ptr:node_ptr + num_nodes]
        else:
            sample_node_att = None
            
        if batch_edge_att is not None:
            sample_edge_att = batch_edge_att[edge_ptr:edge_ptr + num_edges]
        else:
            sample_edge_att = None
        
        results.append({
            'dataset_idx': dataset_idx,
            'sample': sample,
            'node_att': sample_node_att,
            'edge_att': sample_edge_att
        })
        
        # Update pointers for next sample
        node_ptr += num_nodes
        edge_ptr += num_edges
    
    return results


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, fold, task_type='classification', datasets=None, masked_data = None):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
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

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']
        self.motif_loss_coef = method_config['motif_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)
        self.model_name = method_config["model_name"]
        self.motif_readout = method_config["model_name"]

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label, task_type)
        
        # Create deterministic directory for saving scores (NO TIMESTAMP!)
        tuning_id = method_config.get('tuning_id', 'default')
        experiment_name = method_config.get('experiment_name', 'default_experiment')

        self.seed_dir = os.path.join(
            "tuning_results",  # Base directory
            str(self.dataset_name),
            f'model_{self.model_name}',
            f'experiment_{experiment_name}',
            f'tuning_{tuning_id}',
            f'pred{self.pred_loss_coef}_info{self.info_loss_coef}_motif{self.motif_loss_coef}',
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
            'loss_coefficients': {
                'pred_loss_coef': self.pred_loss_coef,
                'info_loss_coef': self.info_loss_coef,
                'motif_loss_coef': self.motif_loss_coef
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

        with open(final_metrics_path, 'w') as f:
            json.dump(metric_dict, f, indent=2)
            
    @staticmethod
    def _calculate_entropy(weights, num_bins=20):
        """Calculate entropy of weight distribution."""
        hist, _ = np.histogram(weights, bins=num_bins, range=(0, 1), density=True)
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log(hist))

    def __loss__(self, att, clf_logits, clf_labels, epoch, nodes_to_motifs):
        if not self.multi_label:
            clf_logits = clf_logits.squeeze(-1)
        # pdb.set_trace()
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        motif_loss = motif_consistency_loss(att, nodes_to_motifs) * self.motif_loss_coef
        loss = pred_loss + info_loss + motif_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'motif_consistency': motif_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
            
        # input(data)
        # pdb.set_trace()
        # if self.motif_readout is not None:
            #todo get edge_att after going a motif level readout if self.motif_readout = add do weighted add readout and if mean a weighted mean readout
            #from nodes to motif get node_attr and then use self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch, data.nodes_to_motifs)
        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()

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
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
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
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'gsat_best/{metric}', value, epoch)

            # Calculate explainer performance every 10 epochs (resource intensive)
            if epoch % 10 == 0 and epoch > 0:
                try:
                    print(f"[INFO] Calculating explainer performance at epoch {epoch}")
                    explainer_metrics = calculate_explainer_performance(
                        self.clf, self.extractor, loaders['valid'], self.device, epoch, self.learn_edge_att
                    )
                    wandb.log(explainer_metrics)
                    print(f"[INFO] Fidelity-: {explainer_metrics['explainer/fidelity_minus']:.4f}, "
                          f"Fidelity+: {explainer_metrics['explainer/fidelity_plus']:.4f}")
                except Exception as e:
                    print(f"[WARNING] Failed to calculate explainer performance: {e}")

            # if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
            #     if self.multi_label:
            #         raise NotImplementedError
            #     for idx, tag in viz_set:
            #         self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))
                
            # ===== Save edge and node scores for the last epoch using small batches =====
            '''
            NOTE: CHECK LOGIC
            '''
            if epoch == self.epochs - 1:
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
                            att_log_logits = self.extractor(emb,data.edge_index, batch)
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

                                # For each motif in the current graph mask it out 

                                for local_motif in data.nodes_to_motifs.unique():
                                    # Convert tensor to int/Python scalar for key comparison
                                    local_motif_key = int(local_motif.item()) if hasattr(local_motif, 'item') else int(local_motif)
                                    
                                    # Check if motif exists before accessing to avoid defaultdict auto-creation
                                    if local_motif_key not in self.masked_data_features[split_name]:
                                        # Also try with original tensor key
                                        if local_motif in self.masked_data_features[split_name]:
                                            local_motif_key = local_motif
                                        else:
                                            continue
                                    
                                    # Check if graph index exists in this motif
                                    if di not in self.masked_data_features[split_name][local_motif_key]:
                                        continue
                                    
                                    masked_feature_graph = self.masked_data_features[split_name][local_motif_key][di]
                                    
                                    # Ensure data is on correct device immediately after retrieval
                                    masked_feature_graph = masked_feature_graph.to(self.device)
                                    
                                    # Check if masked data is empty and skip if so
                                    if masked_feature_graph.numel() == 0:
                                        continue
                                    
                                    # Calculate predictions with masked and original data
                                    new_prediction = self.clf.forward(masked_feature_graph, data.edge_index, batch, edge_attr=data.edge_attr)
                                    
                                    # Calculate old_prediction (original data)
                                    old_prediction = self.clf.forward(data.x, data.edge_index, batch, edge_attr=data.edge_attr)
                                    
                                    # Save masked prediction results
                                    old_pred_value = float(old_prediction.squeeze().detach().cpu().item())
                                    new_pred_value = float(new_prediction.squeeze().detach().cpu().item())
                                    
                                    masked_result = {
                                        'split': split_name,
                                        'graph_idx': di,
                                        'smiles': data.smiles,
                                        'motif_idx': int(local_motif_key),
                                        'old_prediction': old_pred_value,
                                        'new_prediction': new_pred_value
                                    }
                                    
                                    # Write to masked-impact.jsonl file
                                    if not hasattr(self, 'masked_impact_file'):
                                        masked_impact_path = os.path.join(self.seed_dir, 'masked-impact.jsonl')
                                        self.masked_impact_file = open(masked_impact_path, 'w')
                                    
                                    self.masked_impact_file.write(json.dumps(masked_result) + '\n')
                                    
                                    # ===== Masked Edge Impact Calculation =====
                                    # Calculate prediction with masked features + filtered edges
                                    # Identify masked nodes by comparing with original features
                                    masked_nodes = (masked_feature_graph.abs().sum(dim=1) == 0) & (data.x.abs().sum(dim=1) > 0)
                                    
                                    # Filter edges: remove edges where both source and destination are masked nodes
                                    src, dst = data.edge_index
                                    keep_edge_mask = ~(masked_nodes[src] & masked_nodes[dst])
                                    filtered_edge_index = data.edge_index[:, keep_edge_mask]
                                    
                                    # Filter edge_attr accordingly
                                    if data.edge_attr is not None:
                                        filtered_edge_attr = data.edge_attr[keep_edge_mask]
                                    else:
                                        filtered_edge_attr = None
                                    
                                    # Calculate new prediction with masked features + filtered edges
                                    new_prediction_edge = self.clf.forward(masked_feature_graph, filtered_edge_index, batch, edge_attr=filtered_edge_attr)
                                    
                                    # Save masked edge results
                                    old_pred_value_edge = float(old_prediction.squeeze().detach().cpu().item())
                                    new_pred_value_edge = float(new_prediction_edge.squeeze().detach().cpu().item())
                                    
                                    masked_edge_result = {
                                        'split': split_name,
                                        'graph_idx': di,
                                        'smiles': data.smiles,
                                        'motif_idx': int(local_motif_key),
                                        'old_prediction': old_pred_value_edge,
                                        'new_prediction': new_pred_value_edge
                                    }
                                    
                                    # Write to masked-edge-impact.jsonl file
                                    if not hasattr(self, 'masked_edge_impact_file'):
                                        masked_edge_impact_path = os.path.join(self.seed_dir, 'masked-edge-impact.jsonl')
                                        self.masked_edge_impact_file = open(masked_edge_impact_path, 'w')
                                    
                                    self.masked_edge_impact_file.write(json.dumps(masked_edge_result) + '\n')


                            else:
                                sample_results['edge_att'] = att.detach().cpu().numpy()

                            # Save scores for each sample in the batch
                            self.save_sample_scores(sample_results, split_name, di, node_f, edge_f)
                        
                print(f"[INFO] Successfully saved attention scores to {node_jsonl_path} and {edge_jsonl_path}")
                
                # Close masked impact file if it was opened
                if hasattr(self, 'masked_impact_file'):
                    self.masked_impact_file.close()
                    masked_impact_path = os.path.join(self.seed_dir, 'masked-impact.jsonl')
                    print(f"[INFO] Successfully saved masked prediction results to {masked_impact_path}")
                
                # Close masked edge impact file if it was opened
                if hasattr(self, 'masked_edge_impact_file'):
                    self.masked_edge_impact_file.close()
                    masked_edge_impact_path = os.path.join(self.seed_dir, 'masked-edge-impact.jsonl')
                    print(f"[INFO] Successfully saved masked edge prediction results to {masked_edge_impact_path}")
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
                    f'{phase}/motif_loss': loss_dict.get('motif_consistency', 0),
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
                
                wandb.log(wandb_metrics)
            except Exception as e:
                pass  # Silently fail if wandb is not initialized
        
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
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
            clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

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
                    evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
                    clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']
                elif self.multi_label:
                    # Multi-label classification
                    # Convert to tensor if needed, then apply sigmoid
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.sigmoid(clf_logits_tensor).numpy()
                    # Handle NaN values in labels for multi-label
                    valid_mask = ~np.isnan(clf_labels)
                    if valid_mask.any():
                        clf_roc = roc_auc_score(clf_labels[valid_mask], clf_probs[valid_mask], average='micro')
                    else:
                        clf_roc = 0
                elif len(np.unique(clf_labels)) == 2:
                    # Binary classification
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.sigmoid(clf_logits_tensor).numpy()
                    if clf_probs.ndim > 1 and clf_probs.shape[1] > 1:
                        clf_probs = clf_probs[:, 1] if clf_probs.shape[1] == 2 else clf_probs.squeeze()
                    clf_roc = roc_auc_score(clf_labels, clf_probs)
                else:
                    # Multi-class classification
                    if isinstance(clf_logits, np.ndarray):
                        clf_logits_tensor = torch.from_numpy(clf_logits)
                    else:
                        clf_logits_tensor = clf_logits
                    clf_probs = torch.softmax(clf_logits_tensor, dim=1).numpy()
                    clf_roc = roc_auc_score(clf_labels, clf_probs, multi_class='ovr', average='macro')
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
            elif self.dataset_name == 'ogbg_molhiv':
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

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
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


def calculate_explainer_performance(model, extractor, data_loader, device, epoch, learn_edge_att=True):
    """
    Calculate explainer performance by comparing predictions:
    1. Original graph vs graph with important regions removed (fidelity-)
    2. Original graph vs only important regions kept (fidelity+)
    
    Returns metrics for wandb logging.
    """
    model.eval()
    extractor.eval()
    
    fidelity_minus = []  # Prediction drop when removing important parts
    fidelity_plus = []   # Prediction maintained with only important parts
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            
            # Process each graph in the batch individually to avoid shape mismatches
            num_graphs = batch_data.batch.max().item() + 1
            
            for graph_idx in range(num_graphs):
                try:
                    # Extract single graph from batch
                    node_mask = batch_data.batch == graph_idx
                    node_indices = torch.where(node_mask)[0]
                    
                    # Get edges for this graph
                    edge_mask = node_mask[batch_data.edge_index[0]] & node_mask[batch_data.edge_index[1]]
                    graph_edge_index = batch_data.edge_index[:, edge_mask]
                    
                    # Remap node indices to start from 0
                    node_mapping = torch.zeros(batch_data.x.size(0), dtype=torch.long, device=device)
                    node_mapping[node_indices] = torch.arange(len(node_indices), device=device)
                    graph_edge_index = node_mapping[graph_edge_index]
                    
                    # Extract features
                    graph_x = batch_data.x[node_mask]
                    graph_nodes_to_motifs = batch_data.nodes_to_motifs[node_mask] if hasattr(batch_data, 'nodes_to_motifs') else None
                    graph_batch = torch.zeros(graph_x.size(0), dtype=torch.long, device=device)
                    
                    # Get original prediction
                    orig_output, orig_emb = model(graph_x, graph_edge_index, graph_batch, graph_nodes_to_motifs)
                    orig_pred = torch.sigmoid(orig_output) if orig_output.shape[-1] == 1 else torch.softmax(orig_output, dim=1)
                    
                    # Get attention scores
                    att_log_logits = extractor(orig_emb, graph_edge_index, graph_batch)
                    att = att_log_logits.sigmoid()
                    
                    if learn_edge_att and graph_edge_index.size(1) > 0:
                        # Edge-level attention
                        num_edges = graph_edge_index.size(1)
                        k = max(1, int(0.2 * num_edges))  # Keep top 20%
                        
                        if att.numel() >= k:
                            top_k_indices = torch.topk(att.squeeze(), min(k, att.numel())).indices
                            
                            # Create mask for important edges
                            important_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
                            important_mask[top_k_indices] = True
                            
                            # Fidelity- : Remove important edges
                            remaining_edges = graph_edge_index[:, ~important_mask]
                            if remaining_edges.size(1) > 0:
                                output_minus, _ = model(graph_x, remaining_edges, graph_batch, graph_nodes_to_motifs)
                                pred_minus = torch.sigmoid(output_minus) if output_minus.shape[-1] == 1 else torch.softmax(output_minus, dim=1)
                                fidelity_minus.append((orig_pred - pred_minus).abs().mean().item())
                            
                            # Fidelity+ : Keep only important edges
                            important_edges = graph_edge_index[:, important_mask]
                            if important_edges.size(1) > 0:
                                output_plus, _ = model(graph_x, important_edges, graph_batch, graph_nodes_to_motifs)
                                pred_plus = torch.sigmoid(output_plus) if output_plus.shape[-1] == 1 else torch.softmax(output_plus, dim=1)
                                fidelity_plus.append((orig_pred - pred_plus).abs().mean().item())
                
                except Exception as e:
                    # Skip this graph if there's an error
                    continue
    
    metrics = {
        'explainer/fidelity_minus': np.mean(fidelity_minus) if fidelity_minus else 0.0,
        'explainer/fidelity_plus': np.mean(fidelity_plus) if fidelity_plus else 0.0,
        'explainer/epoch': epoch
    }
    
    return metrics


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state,  fold, task_type='classification', use_motif_loss = False):
    # Build the deterministic seed_dir path to check for artifacts
    gsat_config = local_config.get('GSAT_config', {})
    tuning_id = gsat_config.get('tuning_id', 'default')
    experiment_name = gsat_config.get('experiment_name', 'default_experiment')
    pred_coef = gsat_config.get('pred_loss_coef', 1.0)
    info_coef = gsat_config.get('info_loss_coef', 1.0)
    motif_coef = gsat_config.get('motif_loss_coef', 0.0)
    init_r = gsat_config.get('init_r', 0.9)
    final_r = gsat_config.get('final_r', 0.7)
    decay_r = gsat_config.get('decay_r', 0.1)
    
    seed_dir = os.path.join(
        "tuning_results",
        str(dataset_name),
        f'model_{model_name}',
        f'experiment_{experiment_name}',
        f'tuning_{tuning_id}',
        f'pred{pred_coef}_info{info_coef}_motif{motif_coef}',
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
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                'dataset': dataset_name,
                'model': model_name,
                'fold': fold,
                'seed': random_state,
                'tuning_id': tuning_id,
                'experiment_name': experiment_name,
                **local_config.get('GSAT_config', {}),
                **local_config.get('model_config', {})
            },
            reinit=True
        )
        print(f"[INFO] Initialized wandb: {wandb_project}/{wandb_name}")
    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb: {e}")
        print(f"[INFO] Continuing without wandb...")

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name
    
    method_config['model_name'] = model_name
    
    if not use_motif_loss:
        method_config['motif_loss_coef'] = 0.0

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info, datasets, masked_data_features = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False), fold)

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

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config, fold, task_type, datasets, masked_data_features)
    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    
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
    parser.add_argument(
        "--use_motif_loss",
        dest="use_motif_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If False uses sigmoid activation",
    )
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

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

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
        
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, fold, task_type, args.use_motif_loss)
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
