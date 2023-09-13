import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GCNConv, GATConv, GINConv, MLP

class GNNLoadRegressor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(GNNLoadRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ANN layers
        self.ann = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )

        # PyG MLP layers
        self.mlp = MLP(in_channels=self.input_dim, hidden_channels=self.hidden_dim, out_channels=output_dim, num_layers=3, dropout=0.3)

        # PyG GAT layers
        self.gat = nn.ModuleList([
            GATConv(self.input_dim, self.hidden_dim, heads=10),
            GATConv(self.hidden_dim*10, self.hidden_dim, heads=5),
            GATConv(self.hidden_dim*5, self.hidden_dim, heads=2),
            GATConv(self.hidden_dim*2, self.hidden_dim),
            GATConv(self.hidden_dim, self.output_dim)
        ])

    
    def forward(self, x, edge_index, edge_attr):

        # ANN layers
        # x = self.ann(x)
        for layer in self.gat:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        return x