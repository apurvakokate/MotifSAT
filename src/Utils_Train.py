import os
from PIL import Image
import numpy as np
import shutil
import pathlib
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score
import pickle
import seaborn as sns
import numpy as np
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pdb
import pandas as pd

def compute_pos_weights(dataset):
    """Calculate pos_weight for each task based on valid (non-NaN) samples."""
    
    y = torch.tensor(dataset.y, dtype=torch.float)

    if y.ndim == 1:  # Single-label binary classification
        return torch.tensor([compute_pos_weights_one_binary_task(y)], dtype=torch.float32)

    pos_weights = []
    
    for task_idx in range(y.shape[1]):
        
        labels = torch.tensor(dataset.y[:, task_idx], dtype=torch.float)
        
        pos_weight = compute_pos_weights_one_task(labels)
        
        pos_weights.append(pos_weight)
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def compute_pos_weights_one_binary_task(labels):
    """Compute positive weight as N_neg / N_pos for BCEWithLogitsLoss."""
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    return neg / (pos + 1e-8)

def compute_pos_weights_one_task(labels):
    # Extract valid labels for this task
    valid_mask = ~torch.isnan(labels)
    valid_labels = labels[valid_mask]

    # Count positive and negative samples
    num_pos = valid_labels.sum().item()
    num_neg = len(valid_labels) - num_pos

    # Handle edge cases (no positives or negatives)
    if num_pos == 0:
        pos_weight = 0.0  # No positive samples to weight
    elif num_neg == 0:
        pos_weight = 1.0  # No negative samples (trivial balance)
    else:
        pos_weight = num_neg / num_pos
        
    return pos_weight

def remove_bad_mols(dataset):
    indices_to_remove = np.ones(len(dataset), dtype=bool)
    for i,data in enumerate(dataset):
        if data is None: 
            indices_to_remove[i] = False
        elif data.num_nodes == 0:
            print(f"Skipping molecule {data['smiles']} since it "
                      f"resulted in zero atoms")
            indices_to_remove[i] = False

    return dataset[indices_to_remove]

def calculate_fidelity(output, predictions, device):
    output = torch.cat(output, dim=0).to(device)
    output = torch.exp(output)
    fidelity = abs(output - predictions).float().mean()
    
    return fidelity
    
    
def mae(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _= model(data.x, data.edge_index, data.batch, data.nodes_to_motifs)
            # pred_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            all_preds.append(output.cpu())
            all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mae = mean_absolute_error(all_labels, all_preds)
    return mae   

def rmse(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _= model(data.x, data.edge_index, data.batch, data.nodes_to_motifs)
            # probs = torch.exp(output)  # Convert log-softmax output to probabilities
            # pred_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            # all_preds.append(probs.cpu())
            all_preds.append(output.cpu())
            all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    return rmse   



def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_pred_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _= model(data.x, data.edge_index, data.batch, data.nodes_to_motifs)
            all_preds.append(output.cpu())
            all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if model.task_type == 'MultiTask':
        # Step 1: Identify valid indices (exclude NaN and infinity values)
        valid_mask = ~np.isnan(all_labels) & ~np.isnan(all_preds) & np.isfinite(all_preds)
        valid_mask = valid_mask.to(torch.bool)

        # Step 2: Filter invalid entries
        filtered_labels = all_labels[valid_mask]
        filtered_preds = all_preds[valid_mask]

        roc_auc = roc_auc_score(filtered_labels, filtered_preds)
    elif model.task_type == 'MultiClass':
        input("Support pending")
        roc_auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
    else:
        roc_auc = roc_auc_score(all_labels, all_preds)
    return roc_auc

# def evaluate_model_prediction(model, loader, device, original_prediction_y =None):
#     '''
#     Return Class probability of predicted class
#     '''
#     model.eval()
#     preds_out = []
#     preds_y = []
    
#     with torch.no_grad():
#         for i,data in enumerate(loader):
#             data = data.to(device)
#             output, _ = model(data.x, data.edge_index, data.batch, data.smiles)
#             predicted_class = torch.argmax(output.data, dim = 1)  # Get the index of the max log-probability
#             if original_prediction_y is not None:
#                 output_at_predicted_class = output[range(len(predicted_class)), original_prediction_y[i]]
#             else:
#                 output_at_predicted_class = output[range(len(predicted_class)), predicted_class]
#                 preds_y.append(predicted_class)
            
#             preds_out.extend(output_at_predicted_class)
#     return preds_out,preds_y


# def get_model_prediction(model, loader, device):
#     '''
#     Return Class probability of predicted class
#     '''
#     model.eval()
#     preds_out = []
#     labels = []
    
#     with torch.no_grad():
#         for i,data in enumerate(loader):
#             data = data.to(device)
#             output, _ = model(data.x, data.edge_index, data.batch, data.smiles)
#             labels.extend(data.y)
#             preds_out.extend(output)
#     return preds_out, labels


def get_masked_graphs(loader, motif_idx, lookup_dict):
    '''
    Checks the meaningfulness of each motif importance.
    '''
    new_loader = [data.clone() for data in loader]  # Cloning each data instance to avoid in-place modifications
    
    for data in new_loader:
        
        device = data.x.device
        batch_offsets = torch.cumsum(
            torch.cat([torch.tensor([0], device=device), torch.bincount(data.batch, minlength=data.num_graphs)]), 
            dim=0
        )[:-1]

        mask = torch.ones(data.x.size(0), device=device, dtype=torch.float32)  # Initialize mask with float dtype

        for gid, smile in enumerate(data.smiles):
            if smile in lookup_dict:  # Ensure smile exists in the lookup_dict
                node_to_motif = lookup_dict[smile]
                graph_offset = batch_offsets[gid]

                # Identify all nodes corresponding to the given motif_idx
                indices_to_zero = [
                    graph_offset + node_index for node_index, (_, m_idx) in node_to_motif.items() if m_idx == motif_idx
                ]
                
                if indices_to_zero:  # If there are any indices to zero out
                    indices_to_zero = torch.stack(indices_to_zero)  # Stack list into a single tensor
                    mask[indices_to_zero] = 0.0  # Zero out the selected indices

        # Apply the mask to the entire graph
        data.x = data.x * mask.unsqueeze(1)  # Broadcasting mask over feature dimensions

    return new_loader

def get_masked_graphs_from_list(data_list, motif_idx, lookup_dict):
    '''
    Checks the meaningfulness of each motif importance.
    '''
    for data in data_list:
        
        device = data.x.device

        mask = torch.ones(data.x.size(0), device=device, dtype=torch.float32)  # Initialize mask with float dtype

        smile = data.smiles
        if smile in lookup_dict:  # Ensure smile exists in the lookup_dict
            node_to_motif = lookup_dict[smile]

            # Identify all nodes corresponding to the given motif_idx
            indices_to_zero = [node_index for node_index, (_, m_idx) in node_to_motif.items() 
                               if m_idx == motif_idx]
                
            if indices_to_zero:  # If there are any indices to zero out
                indices_to_zero = torch.tensor(indices_to_zero)  # Stack list into a single tensor
                mask[indices_to_zero] = 0.0  # Zero out the selected indices

        # Apply the mask to the entire graph
        data.x = data.x * mask.unsqueeze(1)  # Broadcasting mask over feature dimensions\

    return data_list

def nested_defaultdict():
    return defaultdict(torch.Tensor)

def get_masked_graphs_from_list_for_each_motif(data_list, lookup_dict, motif_lengths):
    """
    Checks the meaningfulness of each motif importance by masking out motifs.

    Args:
        data_list (list): List of PyTorch Geometric data objects.
        lookup_dict (dict): A dictionary mapping SMILES strings to node-to-motif mappings.
        motif_lengths (dict): A dictionary mapping motif names to their respective lengths.

    Returns:
        dict: A dictionary where keys are indices of data and values are masked node features for each motif.
    """
    collect_data = defaultdict(nested_defaultdict)
    
    for data_i, data in enumerate(data_list):
        device = data.x.device

        
        # Create a full mask of ones
        mask = torch.ones(data.x.size(0), device=device, dtype=torch.float32)

        smile = data.smiles
        if smile in lookup_dict:  # Ensure the SMILES exists in the lookup_dict
            node_to_motif = lookup_dict[smile]
            
            # Group nodes by motifs
            motif_groups = {-1:[]}
            for node_idx, (motif_name, motif_index) in node_to_motif.items():
                if motif_name in motif_lengths:
                    if motif_index not in motif_groups:
                        motif_groups[motif_index] = []
                    motif_groups[motif_index].append(node_idx)
                else:
                    motif_groups[-1].append(node_idx)
            
            # Sort indices for each motif group and handle cycles
            for motif_index, group_indices in motif_groups.items():
                group_indices = sorted(group_indices)
                if len(group_indices) > 1 and group_indices[-1] - group_indices[0] + 1 != len(group_indices):
                    # Handle wrapping cases (e.g., [20, 21, 0, 1])
                    group_indices = sorted(group_indices, key=lambda x: (x >= group_indices[0], x))

                # Apply masking for this group
                group_mask = mask.clone()
                group_indices_tensor = torch.tensor(group_indices, device=device)
                if len(group_indices_tensor) > 0:

                    group_mask[group_indices_tensor] = 0.0  # Mask out this motif

                    # Store the masked data in the collect_data dictionary
                    collect_data[motif_index][data_i] = data.x * group_mask.unsqueeze(1)

    return collect_data





# def get_masked_graphs_from_list_for_each_motif(data_list, lookup_dict, motif_lengths):
#     """
#     Checks the meaningfulness of each motif importance by masking out motifs.
    
#     Args:
#         data_list (list): List of PyTorch Geometric data objects.
#         lookup_dict (dict): A dictionary mapping SMILES strings to node-to-motif mappings.
#         motif_lengths (dict): A dictionary mapping motif names to their respective lengths.

#     Returns:
#         dict: A dictionary where keys are indices of data and values are masked node features for each motif.
#     """
#     collect_data = {}
    
#     for data_i, data in enumerate(data_list):
#         device = data.x.device

#         # Initialize the dictionary for the current data
#         collect_data[data_i] = {}
        
#         # Initialize a mask with all ones
#         mask = torch.ones(data.x.size(0), device=device, dtype=torch.float32)

#         smile = data.smiles
#         if smile in lookup_dict:  # Ensure the SMILES exists in the lookup_dict
#             node_to_motif = lookup_dict[smile]
            
#             # Iterate through each motif using motif_lengths
#             motif_start = 0
            
#             while motif_start < len(node_to_motif)-1:
#                 motif_name, m_idx = node_to_motif[motif_start]
#                 print(smile,motif_start,motif_name)
#                 if m_idx is None or motif_name not in motif_lengths:
#                     motif_start += 1
#                     continue
                
#                 # Determine the end index of the current motif
#                 motif_end = motif_start + motif_lengths[motif_name]
                
#                 print(smile,motif_start,motif_name,motif_lengths[motif_name])
                
#                 # Get the indices corresponding to the motif
#                 indices_to_zero = list(range(motif_start, motif_end))
#                 print(motif_end)
#                 if indices_to_zero:  # If there are any indices to zero out
#                     indices_to_zero = torch.tensor(indices_to_zero, device=device)  # Convert to tensor
#                     mask[indices_to_zero] = 0.0  # Zero out the selected indices

#                 # Store the masked data.x in the collect_data dictionary
#                 collect_data[data_i][m_idx] = data.x * mask.unsqueeze(1)
                
#                 # Move to the next motif
#                 motif_start = motif_end
#                 print("final",motif_start,motif_end)

#     return collect_data


def plot_distribution_displaced_topk(tensor, title, image_path, motif_list, colors=None):
    tensor = np.array(tensor)
    x, y = tensor[:, 0], tensor[:, 1]

    plt.figure(figsize=(6, 6))
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, tensor.shape[0]))

    plt.scatter(x, y, c=colors, alpha=0.5)
    # annotate_divergent_points(tensor, motif_list)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def plot_distribution(tensor, title, image_path, motif_list, colors=None):
    if tensor.shape[1] == 1:
        plt.figure(figsize=(6, 6))
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1,tensor.shape[0]))
            
#         # Convert tensors to scalars (assuming grayscale colors)
#         colors = [float(c.item()) for c in colors]  # Extract scalar values from tensors

#         # Normalize the grayscale values to a valid range for matplotlib (0-1)
#         colors = [max(0, min(c, 1)) for c in colors]  # Ensure values are within [0, 1]
        # print(len(colors), tensor.shape, len(motif_list))

        x = tensor
        plt.ylim(0,20)
        sns.kdeplot(data=x.detach().numpy())
        # Scatter plot on top of the density plot
        for m_i, value in enumerate(tensor.flatten()):
            if m_i not in colors:
                #Motif not observed
                plt.scatter(value, 0, c=0.0, label=f'Motif {motif_list[m_i]}' if motif_list else None)
            else:
                # Add a small random perturbation to the position
                perturbation = np.random.uniform(-0.02, 0.02, size=1)
                alpha_value = 0.7  # Adjust alpha for transparency

                # Updated scatter plot code
                plt.scatter(
                    value + perturbation,  # Apply perturbation to the x-coordinate
                    0,  # y-coordinate remains fixed at 0
                    c=float(colors[m_i]),  # Color based on `colors`
                    alpha=alpha_value,  # Add transparency
                    label=f'Motif {motif_list[m_i]}' if motif_list else None
                )

        plt.xlabel('Importance per Motif in Vocabulary')
        plt.ylabel('Density')
        plt.title(title)
        
    else:
        tensor = np.array(tensor)
        x, y = tensor[:, 0], tensor[:, 1]

        plt.figure(figsize=(6, 6))
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, tensor.shape[0]))

        plt.scatter(x, y, c=colors, alpha=0.5)
        annotate_divergent_points(tensor, motif_list)
        plt.title(title)
        plt.xlabel('Class 0 importance')
        plt.ylabel('Class 1 importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def annotate_divergent_points(tensor, motif_list):
    mean = np.mean(tensor, axis=0)
    cov_matrix = np.cov(tensor.T)
    if np.linalg.det(cov_matrix) != 0.0:
        
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        mahalanobis_distances = [
            distance.mahalanobis(point, mean, inv_cov_matrix) for point in tensor
        ]
        divergent_indices = np.argsort(mahalanobis_distances)[-10:]

        for i in divergent_indices:
            plt.annotate(motif_list[i], (tensor[i, 0], tensor[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center')

def save_training_artifacts(image_files, dataset_name, output_dir):
    images = [Image.open(image) for image in image_files]
    images[0].save(f'{output_dir}/{dataset_name}.gif', save_all=True, append_images=images[1:], duration=500, loop=0)

def train_and_evaluate_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device, config, output_dir, plot=False, motif_list=None, dataset_name='dataset', patience=10, delta=1e-4, ignore_unknowns = False, predictions = None, model_dir = None, train_mask_data = None, val_mask_data = None, test_mask_data = None, class_weights = None):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    image_files = []

    image_dir = f'./{output_dir}/epoch_artifacts_{dataset_name}/'
    os.makedirs(image_dir, exist_ok=True)
    
    #check if model directory exists or use output dir
    if model_dir is None:
        model_dir = output_dir

    model.to(device)

    best_val_loss = float('inf')
    lowest_fidelity = float('inf')
    epochs_without_loss_improvement = 0
    epochs_wihout_fidelity_improvement = 0
    fidelity_flag = False
    
    #check if output_dir/dataset_name.csv exists
    #if it exists: start epoch at csv column[Epoch] last row + 1
    # Initialize starting epoch
    start_epoch = 0
    model_checkpoint_path = os.path.join(output_dir, "model_checkpoint.pth")
    optimizer_checkpoint_path = os.path.join(output_dir, "optimizer_checkpoint.pth")
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")

    # Check if the dataset.csv exists
    if os.path.exists(csv_path):
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Check if the CSV has any rows
        if not df.empty:
            # Get the last epoch value
            start_epoch = int(df["Epoch"].iloc[-1])

        # Load model and optimizer states
        model_state = torch.load(model_checkpoint_path)
        model.load_state_dict(model_state)

        optimizer_state = torch.load(optimizer_checkpoint_path)
        optimizer.load_state_dict(optimizer_state)
        
    print(f"Starting at epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, mask_loss           = train_one_epoch(model, 
                                                      criterion, 
                                                      optimizer, 
                                                      train_loader, 
                                                      device, 
                                                      config, 
                                                      ignore_unknowns= ignore_unknowns,
                                                     class_weights = class_weights)
        # pdb.set_trace()
        train_losses.append(train_loss)
        val_loss, train_acc, val_acc, fidelity = validate_model(model, criterion, val_loader, device, train_loader, predictions)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        

        if plot and hasattr(model, 'motif_params'):
            masked_dataset = [val_mask_data]
            if model.motif_params.shape[1] == 1:
                logit_diff = plot_and_save_distribution(model, epoch, image_dir, motif_list, image_files, masked_dataset)
            else:
                logit_diff = plot_and_save_distribution_multi_class(model, epoch, image_dir, motif_list, image_files, masked_dataset)
            print_epoch_summary(epoch, num_epochs, train_loss, mask_loss, val_loss, train_acc, val_acc, output_dir, dataset_name, logit_diff = logit_diff, params = model.motif_params.detach().cpu())
            
        else:
            print_epoch_summary(epoch, num_epochs, train_loss, mask_loss, val_loss, train_acc, val_acc, output_dir, dataset_name)
            
        torch.save(model.state_dict(), f'{output_dir}/model_checkpoint.pth')
        torch.save(optimizer.state_dict(), f'{output_dir}/optimizer_checkpoint.pth')
            

        # Early stopping logic
        loss_flag, best_val_loss, epochs_without_loss_improvement = early_stopping(val_loss, 
                                                                              best_val_loss, 
                                                                              delta, 
                                                                              patience, 
                                                                              epochs_without_loss_improvement, 
                                                                              model,
                                                                              model_dir, 
                                                                              dataset_name)
        if predictions is not None:
            pdb.set_trace()
            fidelity_flag, lowest_fidelity, epochs_wihout_fidelity_improvement = early_stopping_explainer(fidelity, 
                                                                                       lowest_fidelity, 
                                                                                       delta, 
                                                                                       patience, 
                                                                                       epochs_wihout_fidelity_improvement, 
                                                                                       model,
                                                                                       model_dir, 
                                                                                       dataset_name)
            
        if loss_flag or fidelity_flag:
            break
            
        

    # if plot and hasattr(model, 'motif_params'):
    #     save_training_artifacts(image_files, dataset_name, output_dir)
    #     path = pathlib.Path(image_dir)
    #     shutil.rmtree(path)

    return train_losses, val_losses, train_accs, val_accs

def train_one_epoch(model, criterion, optimizer, train_loader, device, config, noise_factor = 0.0, ignore_unknowns = False, class_weights = None):
    '''
    Main training function
    '''
    running_loss = []
    running_loss_mask = []

    for i,data in enumerate(train_loader):
        data = data.to(device)
        model = model.to(device)
        optimizer.zero_grad()

        output, mask = model(data.x, data.edge_index, data.batch, data.nodes_to_motifs, ignore_unknowns = ignore_unknowns)
        
        is_regression = isinstance(criterion, nn.MSELoss) and model.task_type == 'Regression'
        is_binary_classification = isinstance(criterion, nn.BCEWithLogitsLoss) and model.task_type == 'BinaryClass'
        is_multi_task = isinstance(criterion, nn.BCEWithLogitsLoss) and model.task_type == 'MultiTask'

        if is_regression or is_binary_classification:
            '''
            Target size (torch.Size([64])) input size (torch.Size([64, 1])) for regression MSELoss
            hence we squeeze output as dim ([64])
            '''
            loss = criterion(output.squeeze(), data.y.float())
        elif is_multi_task:
            valid_mask = ~torch.isnan(data.y)
            data.y[~valid_mask] = -1.0
            per_element_loss = criterion(output, data.y)
            # pdb.set_trace()
            masked_loss = per_element_loss * valid_mask
            
            # Apply class weighting for imbalance
            loss = masked_loss * class_weights.to(data.y.device)
            
            # Fixed code
            masked_loss = per_element_loss * valid_mask
            task_losses = (masked_loss * class_weights.to(data.y.device))  # Weight per task
            loss = task_losses.sum() / valid_mask.sum()  # Scalar aggregation

            
        else:
            pdb.set_trace()
            '''
            MULTI CLASS
            '''
            loss = criterion(output, data.y.long().flatten())

        if mask is not None:
            reg_loss = calculate_mask_loss(mask, config)
            loss += reg_loss
            running_loss_mask.append(reg_loss.item())
        # pdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        running_loss.append(loss.item())

    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_mask_loss = sum(running_loss_mask) / len(running_loss_mask) if running_loss_mask else None

    return epoch_train_loss, epoch_mask_loss

def calculate_mask_loss(mask, config):
    EPS = 1e-15
    size_loss = torch.sum(mask) * config["size_reg"]
    mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
    mask_ent_loss = config["ent_reg"] * torch.mean(mask_ent_reg)
    # if mask.shape[1] != 1:
    #     class_disc = config["class_reg"] * torch.sum((mask[:,0] + mask[:,1] - 1)**2)
    #     return size_loss + mask_ent_loss + class_disc
    # else:
    return size_loss + mask_ent_loss

def validate_model(model, criterion, val_loader, device, train_loader, predictions = None):
    '''
    TODO : REFACTOR METHOD!!
    
    '''
    model.eval()
    val_loss = 0.0
    
    #for fidelity
    new_predictions = []
    fidelity = None
    
    is_regression = isinstance(criterion, nn.MSELoss) and model.task_type == 'Regression'
    is_binary_classification = isinstance(criterion, nn.BCEWithLogitsLoss) and model.task_type == 'BinaryClass'
    is_multi_task = isinstance(criterion, nn.BCEWithLogitsLoss) and model.task_type == 'MultiTask'
    
    with torch.no_grad():
        for val_data in val_loader:
            val_data = val_data.to(device)
            
            output, _ = model(val_data.x, val_data.edge_index, val_data.batch, val_data.nodes_to_motifs) 
            

            if is_regression or is_binary_classification:
                '''
                Target size (torch.Size([64])) input size (torch.Size([64, 1])) for regression MSELoss
                hence we squeeze output as dim ([64])
                '''
                val_loss += criterion(output.squeeze(), val_data.y.float()).item()
            elif is_multi_task:
                valid_mask = ~torch.isnan(val_data.y)
                val_data.y[~valid_mask] = 0.0
                per_element_loss = criterion(output, val_data.y)
                masked_loss = per_element_loss * valid_mask
                val_loss_epoch = masked_loss.sum() / valid_mask.sum()
                val_loss += val_loss_epoch.item()
                
            else:
                pdb.set_trace()
                val_loss += criterion(output, val_data.y.long().flatten()).item()
    
            new_predictions.append(output)
    
    if predictions is not None:
        pdb.set_trace()
        # For early stopping using fidelity. TODO
        fidelity = calculate_fidelity(new_predictions, predictions, device)

    # Evaluation
    epoch_val_loss = val_loss / len(val_loader)
    if is_regression:
        train_acc = mae(model, train_loader, device)
        val_acc = mae(model, val_loader, device)
    else:    
        train_acc = evaluate_model(model, train_loader, device, model.num_classes)
        val_acc = evaluate_model(model, val_loader, device, model.num_classes)

    return epoch_val_loss, train_acc, val_acc, fidelity

import os
import csv

def print_epoch_summary(epoch, num_epochs, train_loss, mask_loss, val_loss, train_auc, val_auc, output_dir, dataset_name, logit_diff = None, params = None):
    csv_file = os.path.join(output_dir, f"{dataset_name}.csv")
    
    # Delete the CSV file if it exists in the first epoch
    if epoch == 0 and os.path.isfile(csv_file):
        os.remove(csv_file)
    
    # Print the summary
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", end='')
    if mask_loss is not None:
        print(f", Mask Loss: {mask_loss:.4f}", end='')
    print(f", Val Loss: {val_loss:.4f}, Train ROC-AUC: {train_auc:.4f}, Val ROC-AUC: {val_auc:.4f}")
    
    # Save the summary to CSV
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if logit_diff is not None:
            if not file_exists:
                # Write the header only once
                writer.writerow(["Epoch", "Train Loss", "Mask Loss", "Val Loss", "Train ROC-AUC", "Val ROC-AUC", "Logit Diff", "Motif params"])
            writer.writerow([epoch+1, train_loss, mask_loss if mask_loss is not None else "N/A", val_loss, train_auc, val_auc, logit_diff, params.flatten().tolist()])
        else:
            if not file_exists:
                # Write the header only once
                writer.writerow(["Epoch", "Train Loss", "Mask Loss", "Val Loss", "Train ROC-AUC", "Val ROC-AUC"])
            writer.writerow([epoch+1, train_loss, mask_loss if mask_loss is not None else "N/A", val_loss, train_auc, val_auc])

# Helper function to compute motif colors (logit differences)
def compute_motif_colors(mask_data, model_device):
    motif_colors = {}
    for motif_idx in mask_data[0]:
        logit_diff = 0
        for graph_idx in mask_data[0][motif_idx]:
            data = mask_data[1][graph_idx].to(model_device)
            original_prediction, _ = model(data.x, data.edge_index, None, data.smiles)
            perturbed_data = mask_data[0][motif_idx][graph_idx].to(model_device)
            new_prediction, _ = model(perturbed_data, data.edge_index, None, data.smiles)
            logit_diff += original_prediction - new_prediction
        motif_colors[motif_idx] = logit_diff.item() / len(mask_data[0][motif_idx])
    return motif_colors

def plot_and_save_distribution_multi_class(model, epoch, image_dir, motif_list, image_files, masked_data):

    # Get the device from the model
    model_device = next(model.parameters()).device
    num_classes = model.motif_params.shape[1]  # Determine the number of columns dynamically
    motif_colors = {}

    # Iterate through each motif
    for motif_idx, _ in enumerate(motif_list):
        # Initialize logit_diff for all classes
        logit_diff = torch.zeros(1,num_classes, device=model_device)
        valid_counts = torch.zeros(1,num_classes, device=model_device)  # Track valid counts per class

        # Process each dataset in masked_data
        for dataset in masked_data:
            for graph_idx in dataset[0][motif_idx]:
                data = dataset[1][graph_idx].to(model_device)

                valid_mask = ~torch.isnan(data.y)

                # Skip graphs with all NaN labels
                if not valid_mask.any():
                    continue

                # Compute original and perturbed predictions
                original_prediction, _ = model(data.x, data.edge_index, None, data.smiles)
                new_prediction, _ = model(
                    dataset[0][motif_idx][graph_idx].to(model_device),
                    data.edge_index,
                    None,
                    data.smiles
                )
                

                # Accumulate logit differences for valid classes only
                logit_diff += (original_prediction - new_prediction) * valid_mask.float()
                valid_counts += valid_mask.float()  # Count valid entries for each class

        # Normalize logit_diff by valid_counts for each class (avoid division by zero)
        motif_colors[motif_idx] = (
            (logit_diff / valid_counts).cpu().tolist() if valid_counts.sum() > 0 else [0] * num_classes
        )

    return motif_colors

            
def plot_and_save_distribution(model, epoch, image_dir, motif_list, image_files, masked_data):
    # title = f'Epoch {epoch}'
    # Get the device from the model
    model_device = next(model.parameters()).device
    motif_weights = model.motif_params.detach().cpu()
    # image_path = os.path.join(image_dir, f'plot_{epoch}.png')
    motif_impact = {}
    
    # Process each dataset: train, val, test
    for dataset in masked_data:
        
        # Generalize train_mask_data to handle train, val, and test datasets
        for motif_idx in dataset[0]:
            logit_diff = torch.tensor([[0.0]], device = model_device)
            total_graphs = 0  # To track the total number of graphs across train, val, and test

            for graph_idx in dataset[0][motif_idx]:
                total_graphs += 1  # Count graphs
                data = dataset[1][graph_idx].to(model_device)

                # Original and perturbed predictions
                original_prediction, _ = model(data.x, data.edge_index, None, data.smiles)
                new_prediction, _ = model(
                    dataset[0][motif_idx][graph_idx].to(model_device), 
                    data.edge_index, 
                    None, 
                    data.smiles
                )
                # pdb.set_trace()
                logit_diff += original_prediction - new_prediction

        # Normalize the logit difference by the total number of graphs
        
        motif_impact[motif_idx] = logit_diff.item() / total_graphs if total_graphs!= 0 else 0
            
    return motif_impact

def early_stopping(val_loss, best_val_loss, delta, patience, epochs_without_improvement, model, output_dir, dataset_name):
    if val_loss < best_val_loss - delta:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f'{output_dir}/{dataset_name}_1weighted_best_model.pth')
    else:
        epochs_without_improvement += 1

    return epochs_without_improvement >= patience, best_val_loss, epochs_without_improvement

def early_stopping_explainer(fidelity, lowest_fidelity, delta, patience, epochs_without_improvement, model, output_dir, dataset_name):
    if fidelity < lowest_fidelity:
        lowest_fidelity = fidelity
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f'{output_dir}/{dataset_name}_1weighted_best_model.pth')
    else:
        epochs_without_improvement += 1

    return epochs_without_improvement >= patience, lowest_fidelity, epochs_without_improvement


# Calculate the correlation between the corresponding columns
def calculate_correlation(model_weight_tensor, dataset_name):
    with open(f'{dataset_name}_logistic_regression_weights.pickle', 'rb') as file:
        log_reg_weights = pickle.load(file)
    weights = log_reg_weights[0]
    N = len(weights)
    lostic_weight_tensor = torch.zeros((N, 2))

    # Populate the tensor
    lostic_weight_tensor[:, 0] = torch.tensor(np.abs(weights) * (weights < 0).astype(int))
    lostic_weight_tensor[:, 1] = torch.tensor(np.abs(weights) * (weights > 0).astype(int))
    correlation_matrix = torch.zeros(2, 2)
    for i in range(2):
        # Calculate the correlation between the i-th columns of t1 and t2
        column1 = model_weight_tensor[:, i]
        column2 = lostic_weight_tensor[:, i] -0.5
        
        # Compute mean
        mean1 = torch.mean(column1)
        mean2 = torch.mean(column2)
        
        # Compute the numerator (covariance) and denominators (std deviations)
        numerator = torch.sum((column1 - mean1) * (column2 - mean2))
        denominator = torch.sqrt(torch.sum((column1 - mean1) ** 2) * torch.sum((column2 - mean2) ** 2))
        
        # Compute correlation
        correlation_matrix[i, i] = numerator / denominator

    return correlation_matrix