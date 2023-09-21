import torch
import torch.nn as nn
from torch.functional import F

class CustomLoss(nn.Module):
    def __init__(self, threshold=25):
        super(CustomLoss, self).__init__()

        self.threshold = threshold

    def forward(self, y_pred, y_true):
        # MES loss for all data
        loss1 = F.mse_loss(y_pred, y_true)

        # Penalty for PG predictions in [0, 30]
        indices = (y_pred>0) & (y_pred<30)
        loss2 = F.mse_loss(y_pred[indices], y_true[indices])

        # Final laoss 
        loss = loss1 + loss2

        return loss