import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F

class GNNLossFunc(nn.Module):
    def __init__(self, omega=1.):
        super(GNNLossFunc, self).__init__()
        self.omega = omega
    
    def forward(self, y_true, y_pred):
        nt = int(y_true.shape[1] / 2)
        loss1 = F.mse_loss(y_true, y_pred)
        loss2 = F.mse_loss(y_true[:, -nt:], y_pred[:, -nt:])

        loss = loss1 + self.omega*loss2

        return loss
    
    