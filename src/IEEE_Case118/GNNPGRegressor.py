import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GCNConv, GINConv, MLP

class GNNPGRegressor(nn.Module):
    def __init__(self, input_dim1=32, input_dim2=44, hidden_dim1=32, hidden_dim2=32, output_dim1=32, output_dim2=12):
        super(GNNPGRegressor, self).__init__()

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
            nn.Linear(self.hidden_dim2, self.output_dim2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # Encoder
        temp = self.encoder(x[:, :self.input_dim1])
        x = torch.cat([temp, x[:, self.input_dim1:]], dim=1)

        # ANN layers
        x = self.ann(x)

        return x