import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import Sequential, GCNConv, SAGEConv

class DLANN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(DLANN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ANN layers
        self.ann = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, x, edge_index, load_bidx):

        # ANN layers
        x = self.ann(x)

        return x[load_bidx]
    
class DLSAGE(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(DLSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # SAGE layers
        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(self.input_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.output_dim), 'x, edge_index -> x')
        ])
        
    def forward(self, x, edge_index, load_bidx):

        # GNN layers
        x = self.gnn(x, edge_index)

        return x[load_bidx]
    
class DLGCN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(DLGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # SAGE layers
        self.gnn = Sequential('x, edge_index', [
            (GCNConv(self.input_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim, self.output_dim), 'x, edge_index -> x')
        ])
        
    def forward(self, x, edge_index, load_bidx):

        # GNN layers
        x = self.gnn(x, edge_index)

        return x[load_bidx]