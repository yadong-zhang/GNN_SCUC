import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import Sequential, Linear, SAGEConv, ResGatedGraphConv, TAGConv, ARMAConv

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
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # PyG GAT layers
        self.gnn = Sequential('x, edge_index', [
            (TAGConv(self.input_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (TAGConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (TAGConv(self.hidden_dim, self.output_dim), 'x, edge_index -> x')
        ])
        

    def forward(self, x, edge_index, edge_attr):

        # ANN layers
        # x = self.ann(x)
        
        # GAT layers
        x = self.gnn(x, edge_index)

        return x