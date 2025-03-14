import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GCNConv, GATConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import global_mean_pool


def GIN_kernel(dim):
    """ GIN kernel """
    kernel = nn.Sequential(
        nn.BatchNorm1d(3),
        nn.Linear(3, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )
    return kernel


def attn_kernel(dim):
    """ Attentional Aggregation kernel """
    kernel = AttentionalAggregation(gate_nn=nn.Linear(dim, 1))
    return kernel
    
    
class GNN(nn.Module):
    """ Pure data-driven GNN model """
    def __init__(self, input_dim, dim, output_dim, readout='attn', kernel='GIN'):
        super().__init__()
        if kernel == 'GIN':
            self.nn1 = GIN_kernel(dim)
            self.conv1 = GINConv(self.nn1)
        elif kernel == 'GCN':
            self.conv1 = GCNConv(input_dim, dim)
        elif kernel == 'GAT':
            self.conv1 = GATConv(input_dim, dim)
        self.fc1 = nn.Linear(dim, output_dim)
        if readout == "attn":
            self.readout = attn_kernel(dim)
        elif readout == "mean":
            self.readout = global_mean_pool
            

    def forward(self, data):
        x, edge_index, batch, base = data.x, data.edge_index, data.batch, data.base
        x = F.relu(self.conv1(x, edge_index))
        x = self.readout(x, batch)
        x = self.fc1(x)
        
        return x, None, None
    
    
class Baseline(nn.Module):
    """ Baseline GC model """
    def __init__(self, input_dim, dim, output_dim, readout='mean', kernel='GIN'):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.fc1 = nn.Linear(1, 1)

    def forward(self, data):
        _, _, _, base = data.x, data.edge_index, data.batch, data.base
        x = self.fc1(base)
        return x, None, None
    
    
class GNN_Guided_Baseline_Simple(nn.Module):
    """ GNN guided baseline model (GC-GNN)"""
    def __init__(self, input_dim, dim, output_dim, readout='attn', kernel='GIN'):
        super().__init__()
        if kernel == 'GIN':
            self.nn1 = GIN_kernel(dim)
            self.conv1 = GINConv(self.nn1)
        elif kernel == 'GCN':
            self.conv1 = GCNConv(input_dim, dim)
        elif kernel == 'GAT':
            self.conv1 = GATConv(input_dim, dim)
            
        self.fc1 = nn.Linear(dim, output_dim)
        self.fc2 = nn.Linear(dim, output_dim)
        
        if readout == "attn":
            self.readout = attn_kernel(dim)
        elif readout == "mean":
            self.readout = global_mean_pool

    def forward(self, data):
        x, edge_index, batch, base = data.x, data.edge_index, data.batch, data.base
        x = F.relu(self.conv1(x, edge_index))
        x = self.readout(x, batch)
        a = self.fc1(x)
        b = self.fc2(x)
        x = a * base + b
        return x, a, b
    