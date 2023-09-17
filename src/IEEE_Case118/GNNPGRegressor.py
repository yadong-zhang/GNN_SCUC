import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GCNConv, SAGEConv, Sequential

class PGANN(nn.Module):
    def __init__(self, input_dim1=32, input_dim2=44, hidden_dim1=32, hidden_dim2=32, output_dim1=32, output_dim2=12):
        super(PGANN, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.output_dim1),
            nn.ReLU()
        )

        # ANN layers
        self.ann = nn.Sequential(
            nn.Linear(self.input_dim2, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.output_dim2)
        )
    
    def forward(self, x, edge_index):
        # Encoder
        temp = self.encoder(x[:, :self.input_dim1])
        x = torch.cat([temp, x[:, self.input_dim1:]], dim=1)

        # ANN layers
        x = self.ann(x)

        return x
    

class PGSAGE(nn.Module):
    def __init__(self, input_dim1=32, input_dim2=44, hidden_dim1=32, hidden_dim2=32, output_dim1=32, output_dim2=12):
        super(PGSAGE, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.output_dim1),
            nn.ReLU()
        )

        # SAGE layers
        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(self.input_dim2, self.hidden_dim2), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim2, self.hidden_dim2), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim2, self.output_dim2), 'x, edge_index -> x')
        ])
    
    def forward(self, x, edge_index):
        # Encoder
        temp = self.encoder(x[:, :self.input_dim1])
        x = torch.cat([temp, x[:, self.input_dim1:]], dim=1)

        # GNN layers
        x = self.gnn(x, edge_index)

        return x
    

class PGGCN(nn.Module):
    def __init__(self, input_dim1=32, input_dim2=44, hidden_dim1=32, hidden_dim2=32, output_dim1=32, output_dim2=12):
        super(PGGCN, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.output_dim1),
            nn.ReLU()
        )

        # GCN layers
        self.gnn = Sequential('x, edge_index', [
            (GCNConv(self.input_dim2, self.hidden_dim2), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim2, self.hidden_dim2), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(self.hidden_dim2, self.output_dim2), 'x, edge_index -> x')
        ])
    
    def forward(self, x, edge_index):
        # Encoder
        temp = self.encoder(x[:, :self.input_dim1])
        x = torch.cat([temp, x[:, self.input_dim1:]], dim=1)

        # GNN layers
        x = self.gnn(x, edge_index)

        return x