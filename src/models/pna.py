# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/example.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import BatchNorm, global_mean_pool
from .conv_layers import PNAConvSimple


class PNA(torch.nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()
        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.edge_attr_dim = edge_attr_dim
        self.skip_node_encoder = model_config.get('skip_node_encoder', False)

        self.use_atom_encoder = model_config.get('atom_encoder', False)
        self.use_edge_attr = model_config.get('use_edge_attr', True)
        if self.use_atom_encoder:
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
            first_dim = hidden_size
        elif self.skip_node_encoder:
            self.node_encoder = None
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)
            first_dim = x_dim
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)
            first_dim = hidden_size

        # Residual projection for first layer when input dim != hidden_size
        if first_dim != hidden_size:
            self.residual_proj = Linear(first_dim, hidden_size)
        else:
            self.residual_proj = None

        aggregators = model_config['aggregators']
        scalers = ['identity', 'amplification', 'attenuation'] if model_config['scalers'] else ['identity']
        deg = model_config['deg']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for i in range(self.n_layers):
            node_dim = first_dim if i == 0 else hidden_size
            if self.use_edge_attr and edge_attr_dim != 0:
                in_channels = node_dim * 2 + hidden_size
            else:
                in_channels = node_dim * 2

            conv = PNAConvSimple(in_channels=in_channels, out_channels=hidden_size, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.pool = global_mean_pool
        self.fc_out = Sequential(Linear(hidden_size, hidden_size//2), ReLU(),
                                 Linear(hidden_size//2, hidden_size//4), ReLU(),
                                 Linear(hidden_size//4, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr, edge_atten=None):
        if self.use_atom_encoder:
            x = x.long()
            if edge_attr is not None and hasattr(self, 'edge_encoder'):
                edge_attr = self.edge_encoder(edge_attr.long())
        else:
            if edge_attr is not None and hasattr(self, 'edge_encoder'):
                edge_attr = self.edge_encoder(edge_attr.float())

        if self.node_encoder is not None:
            x = self.node_encoder(x)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            if i == 0 and self.residual_proj is not None:
                x = h + self.residual_proj(x)
            else:
                x = h + x
            x = F.dropout(x, self.dropout_p, training=self.training)

        x = self.pool(x, batch)
        return self.fc_out(x)

    def get_emb(self, x, edge_index, batch, edge_attr, edge_atten=None):
        if self.use_atom_encoder:
            x = x.long()
            if edge_attr is not None and hasattr(self, 'edge_encoder'):
                edge_attr = self.edge_encoder(edge_attr.long())
        else:
            if edge_attr is not None and hasattr(self, 'edge_encoder'):
                edge_attr = self.edge_encoder(edge_attr.float())

        if self.node_encoder is not None:
            x = self.node_encoder(x)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            if i == 0 and self.residual_proj is not None:
                x = h + self.residual_proj(x)
            else:
                x = h + x
            x = F.dropout(x, self.dropout_p, training=self.training)

        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
