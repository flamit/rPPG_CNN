import torch
import torch.nn as nn


class PearsonLoss(nn.Module):

    def __init__(self, T):
        super(PearsonLoss, self).__init__()
        self.T = T

    def forward(self, logits, target):
        num = (self.T * torch.mul(logits, target)).sum(dim=1) - (logits.sum(dim=1) * target.sum(dim=1))
        denom = (self.T * torch.pow(logits, 2).sum(dim=1) - torch.pow(logits.sum(dim=1), 2)) * (
                    self.T * torch.pow(target, 2).sum(dim=1) - torch.pow(target.sum(dim=1), 2))
        loss = num.mean() / torch.sqrt(denom).mean()

        return 1.0 - loss