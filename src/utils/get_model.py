import torch.nn as nn
import torch.nn.functional as F
from models import GIN, PNA, SPMotifNet, GCN, GAT, SAGE
from torch_geometric.nn import InstanceNorm


def get_model(x_dim, edge_attr_dim, num_class, multi_label, model_config, device):
    if model_config['model_name'] == 'GIN':
        model = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'GAT':
        model = GAT(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'GCN':
        model = GCN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'SAGE':
        model = SAGE(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'PNA':
        model = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'SPMotifNet':
        model = SPMotifNet(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    else:
        raise ValueError('[ERROR] Unknown model name!')
    return model.to(device)


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label, task_type='classification'):
        super(Criterion, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        self.task_type = task_type
        print(f'[INFO] Using multi_label: {self.multi_label}')
        print(f'[INFO] Task type: {self.task_type}')

    def forward(self, logits, targets):
        if self.task_type == 'regression':
            # For regression, use MSE loss for training
            # Ensure both tensors have the same shape by squeezing both
            loss = F.mse_loss(logits.squeeze(), targets.squeeze().float())
        elif self.num_class == 2 and not self.multi_label:
            # Squeeze both to ensure shapes match: [N] and [N]
            # Some datasets have targets shape [N, 1], others have [N]
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.squeeze().float())
        elif self.num_class > 2 and not self.multi_label:
            # For multi-class, targets should be [N] with class indices
            loss = F.cross_entropy(logits, targets.squeeze().long())
        else:
            # Multi-label classification
            # Ensure consistent shapes before masking
            logits_flat = logits.view(-1) if len(logits.shape) > 1 else logits
            targets_flat = targets.view(-1) if len(targets.shape) > 1 else targets
            is_labeled = targets_flat == targets_flat  # mask for labeled data (non-NaN)
            loss = F.binary_cross_entropy_with_logits(logits_flat[is_labeled], targets_flat[is_labeled].float())
        return loss


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif len(logits.shape) > 1 and logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary (handles both [N] and [N, 1] shapes)
        preds = (logits.squeeze().sigmoid() > 0.5).float()
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)
