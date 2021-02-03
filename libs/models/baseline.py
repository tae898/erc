import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

assert timm.__version__ == "0.3.2"
from utils import getter
from .extractors import EfficientNetExtractor


class BaseModel(nn.Module):
    def __init__(self, num_classes=5, version=3, freeze_backbone=False, from_pretrained=True):
        super().__init__()
        self.extractor = EfficientNetExtractor(version, from_pretrained=from_pretrained)

        self.feature_dim = self.extractor.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x

class BaseTimmModel(nn.Module):
    def __init__(self, num_classes, name, from_pretrained=True, freeze_backbone=False):
        super().__init__()
        self.model = timm.create_model(name, pretrained=from_pretrained)
        try:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        except:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x