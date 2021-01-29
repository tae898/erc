import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelTemplate(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
