import torch.nn as nn
from torchvision import models

from .extractor_network import ExtractorNetwork


class ResNetExtractor(ExtractorNetwork):
    arch = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, version):
        super().__init__()
        assert version in ResNetExtractor.arch, \
            f'{version} is not implemented.'
        cnn = ResNetExtractor.arch[version](pretrained=True)
        self.extractor = nn.Sequential(*list(cnn.children())[:-1])
        self.feature_dim = cnn.fc.in_features

    def forward(self, x):
        return self.extractor(x).view(x.size(0), -1)
