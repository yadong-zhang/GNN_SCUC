import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GCNConv

random.seed = 0

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, 
                 output_dim, num_gnn_layers, drop_rate):
        super(GNNClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers
        self.drop_rate = drop_rate

        self.dropout = nn.Dropout(self.drop_rate)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU()
        )

        # GNN layers
        self.gnn = nn.ModuleList([
            GCNConv(self.hidden_dim2, self.hidden_dim2) for _ in range(num_gnn_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.ReLU(),
            nn.Linear(self.hidden_dim3, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr):

        # Encoder
        x = self.encoder(x)

        # GNN layers
        for layer in self.gnn:
            x = layer(x, edge_index=edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Decoder
        x = self.decoder(x)

        return x