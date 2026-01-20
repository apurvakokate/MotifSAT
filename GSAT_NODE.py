import torch.nn as nn
from torch_geometric.nn import global_add_pool, TopKPooling
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential, Dropout
from torch_geometric.nn.conv import GINConv
import torch
from torch_scatter import scatter_add
from Utils_Train import evaluate_model
from Utils_model import create_conv_layers 
import pdb
import scipy
from torch_geometric.nn import InstanceNorm
class GNNModel(nn.Module): 
    def __init__(self,input_dim, output_dim, hidden_channels, num_layers, layer_type, use_explainer=False,
                motif_params=None, lookup=None, task_type= 'BinaryClass', test_lookup=None):
        super().__init__()
        num_mp_layers  = num_layers
        hidden         = hidden_channels
        
        self.num_classes = output_dim
        self.task_type = task_type
        
        self.convs = create_conv_layers(input_dim, hidden_channels*2, num_layers, layer_type)
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, self.num_classes)
        
        self.extractor = ExtractorMLP(hidden_channels*2).to(device)
        
        if not use_explainer:
            print("No Explainer parameters will be used. Assuming all node weights are 1")
            self.use_ones = True
        else:
            self.use_ones = False
            
            self.num_params = motif_params.size(0)
            self.motif_params = nn.Parameter(motif_params, requires_grad=True)
            self.lookup = lookup
            self.test_lookup = test_lookup
            


    def forward(self, x, edge_index, batch=None, node_to_motifs = None,edge_weight = None, ignore_unknowns=False, return_logit=False):
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if edge_weight is not None:
            print("PostHoc support pending")
            exit()
            
        if self.use_ones:
            node_weights = None
            # x.to(edge_index.device)

            #Node Embeddings
            x = self.embedding(x, edge_index)
            #Graph Embedding
            x = global_add_pool(x, batch)
            self.readout_output = x
            x = self.classification(x)
            

        else:
            emb = self.embedding(x, edge_index)
            att_log_logits = self.extractor(emb, edge_index, batch)
            node_weights = self.sampling(att_log_logits, training)
            
            node_weights =  node_weights.to(edge_index.device)
            
            # Class 0 embedding
            x = self.get_graph_representation(x, edge_index, node_weights, batch)

        return x, node_weights
    
    
    def get_graph_representation(self, x, edge_index, node_weights, batch):
        node_weights =  node_weights.to(edge_index.device)
        x_cls = x * node_weights
        x_cls = self.embedding(x_cls, edge_index)

        # Readout phase: global mean pooling
        x_cls = global_add_pool(x_cls* node_weights, batch)
        
        self.readout_output = x_cls
        
        # Classification
        return self.classification(x_cls)
    
    def embedding(self, x, edge_index):
        for conv in self.convs:
            conv = conv.to(edge_index.device)
            x = conv(x, edge_index)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = F.relu(x)
        return x
    
    def classification(self, x):
        # self.lin1[str(class_id] = self.lin1[class_id].to(x.device)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
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
    
class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits
