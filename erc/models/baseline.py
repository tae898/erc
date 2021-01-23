import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import getter


class BaselineClassifier(nn.Module):
    def __init__(self, extractor_cfg, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.classifier = nn.Linear(self.feature_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor(x)
        return self.classifier(x)
