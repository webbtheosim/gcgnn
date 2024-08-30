import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn.aggr import AttentionalAggregation


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
    def __init__(self, input_dim, dim, output_dim):
        super().__init__()
        self.nn1 = GIN_kernel(dim)
        self.conv1 = GINConv(self.nn1)
        self.fc1 = nn.Linear(dim, output_dim)
        self.attention = attn_kernel(dim)
        

    def forward(self, data):
        x, edge_index, batch, base = data.x, data.edge_index, data.batch, data.base
        x = F.relu(self.conv1(x, edge_index))
        x = self.attention(x, batch)
        x = self.fc1(x)
        
        return x, None, None
    
    
class Baseline(nn.Module):
    """ Baseline GC model """
    def __init__(self, input_dim, dim, output_dim):
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
    def __init__(self, input_dim, dim, output_dim):
        super().__init__()
        self.nn1 = GIN_kernel(dim)
        self.conv1 = GINConv(self.nn1)
        self.fc1 = nn.Linear(dim, output_dim)
        self.fc2 = nn.Linear(dim, output_dim)
        self.attention = attn_kernel(dim)

    def forward(self, data):
        x, edge_index, batch, base = data.x, data.edge_index, data.batch, data.base
        x = F.relu(self.conv1(x, edge_index))
        x = self.attention(x, batch)
        a = self.fc1(x)
        b = self.fc2(x)
        x = a * base + b
        return x, a, b