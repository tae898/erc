import torch.nn as nn


class LossTemplate(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, output, target):
        raise NotImplementedError
