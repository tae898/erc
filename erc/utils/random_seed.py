import torch
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
