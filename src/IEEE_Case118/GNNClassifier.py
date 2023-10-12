import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import MLP
from torch_geometric.nn import GCNConv, SAGEConv, Sequential

class UCANN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(UCANN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ANN layers
        self.ann = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):

        # ANN layers
        x = self.ann(x)

        return x

class UCSAGE(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(UCSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # SAGEConv
        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(self.input_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.output_dim), 'x, edge_index -> x'),
            nn.Sigmoid()
        ])
    
    def forward(self, x, edge_index):

        # SAGEConv
        x = self.gnn(x, edge_index)

        return x
    
class UCGCN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(UCGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GCNConv
        self.gnn = Sequential('x, edge_index', [
            (GCNConv(self.input_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim, self.output_dim), 'x, edge_index -> x'),
            nn.Sigmoid()
        ])
    
    def forward(self, x, edge_index):

        # GCNConv
        x = self.gnn(x, edge_index)

        return x