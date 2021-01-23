import torch.nn as nn


class ExtractorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def get_embedding(self, x):
        return self.forward(x)
