import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import getter


class MixedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.loss_fns = nn.ModuleList([getter.get_instance(loss)
                                       for loss in losses])
        if weights is None:
            weights = torch.ones(len(losses))
        elif isinstance(weights, list):
            weights = torch.FloatTensor(weights)
        weights /= weights.sum()
        self.register_buffer('weights', weights)

    def forward(self, output, target):
        total = 0.0
        for w, loss_fn in zip(self.weights, self.loss_fns):
            total += w * loss_fn(output, target)
        return total
