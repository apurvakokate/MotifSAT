# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder

from .conv_layers import SAGEConvWithAtten


class SAGE(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.dropout_p = model_config['dropout_p']
        self.task_type = model_config.get('task_type', 'classification')
        
        self.aggr = model_config.get('sage_aggr', 'mean')
        self.skip_node_encoder = model_config.get('skip_node_encoder', False)

        self.use_atom_encoder = model_config.get('atom_encoder', False)
        if self.use_atom_encoder:
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            first_dim = hidden_size
        elif self.skip_node_encoder:
            self.node_encoder = None
            first_dim = x_dim
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            first_dim = hidden_size

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for i in range(self.n_layers):
            in_dim = first_dim if i == 0 else hidden_size
            self.convs.append(SAGEConvWithAtten(in_dim, hidden_size, aggr=self.aggr))

        if self.task_type == 'regression':
            self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1))
        else:
            self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        if self.use_atom_encoder:
            x = x.long()
            x = self.node_encoder(x)
        elif self.node_encoder is not None:
            x = self.node_encoder(x)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc_out(self.pool(x, batch))

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        if self.use_atom_encoder:
            x = x.long()
            x = self.node_encoder(x)
        elif self.node_encoder is not None:
            x = self.node_encoder(x)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
