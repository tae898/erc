import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return self.loss(output, target)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, beta, **kwargs):
        if isinstance(beta, (float, int)):
            self.beta = torch.Tensor([beta])
        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)
        super().__init__(pos_weight=self.beta, **kwargs)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)
