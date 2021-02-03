import torch.nn as nn

from .extractor_network import ExtractorNetwork
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor(ExtractorNetwork):
	def __init__(self, version, from_pretrained=True):
		super().__init__()
		assert version in range(8)
		self.extractor = EfficientNet.from_name(f"efficientnet-b{version}")
		if from_pretrained:
			self.extractor = EfficientNet.from_pretrained(f"efficientnet-b{version}")
		self.feature_dim = self.extractor._fc.in_features

	def forward(self, x):
		x = self.extractor.extract_features(x)
		x = self.extractor._avg_pooling(x)
		x = x.view(x.size(0), -1)
		return x
