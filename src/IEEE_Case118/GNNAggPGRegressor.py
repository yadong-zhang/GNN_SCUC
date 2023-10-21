import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import Sequential, SAGEConv, global_mean_pool, global_max_pool


class AggPGSAGE(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=24, output_dim=12):
        super(AggPGSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim),
            nn.ReLU()
        )

        # SAGE layers
        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU()
        ])

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 48)
        )

        
    def forward(self, x, edge_index, batch):

        x = self.encoder(x)
        x = self.gnn(x, edge_index)
        x = self.decoder(x)
        x = global_max_pool(x, batch)

        return x.view(-1, 12)