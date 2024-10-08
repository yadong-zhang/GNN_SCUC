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


class CorrNN(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=12):
        super(CorrNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim),
            nn.ReLU()
        )

        self.corr =Sequential('x, edge_index', [
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(self.hidden_dim, self.hidden_dim), 'x, edge_index -> x'),
            nn.ReLU()
        ])

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim,  self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.output_dim)
        )
    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.corr(x, edge_index)
        x = self.decoder(x)
        return x
