import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import *
from .former import *

class ContextAwareModel(nn.Module):
    def __init__(self, num_classes, AUDIO_FEAT_DIM=1280, TEXT_FEAT_DIM=1024, EMB_DIM=512, use_concat=False):
        super().__init__()
        self.AUDIO_FEAT_DIM = AUDIO_FEAT_DIM
        self.TEXT_FEAT_DIM = TEXT_FEAT_DIM
        self.EMB_DIM = EMB_DIM
        self.use_concat = use_concat
        self.fusion_model = FusionModel(num_classes=num_classes, AUDIO_FEAT_DIM=AUDIO_FEAT_DIM, TEXT_FEAT_DIM=TEXT_FEAT_DIM, EMB_DIM=EMB_DIM, use_concat=use_concat, single_utt=False)
        self.transformer = CTransformer(emb=EMB_DIM, heads=2, depth=10, seq_length=128, num_tokens=10000, num_classes=7)
        self.classifier = nn.Sequential(
            nn.Linear(EMB_DIM, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        self.fc = nn.Linear(EMB_DIM, num_classes)

    def forward(self, inp):
        fused = self.fusion_model(inp)
        encoded = self.transformer(fused)
        x = self.classifier(encoded)
        # x = self.fc(encoded)
        return x