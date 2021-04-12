import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import getter

class FusionModel(nn.Module):
    def __init__(self, num_classes, AUDIO_FEAT_DIM=1280, TEXT_FEAT_DIM=1024):
        super().__init__()
        self.AUDIO_FEAT_DIM = AUDIO_FEAT_DIM
        self.TEXT_FEAT_DIM = TEXT_FEAT_DIM

        self.classifier = nn.Sequential(
            nn.Linear(self.AUDIO_FEAT_DIM + self.TEXT_FEAT_DIM, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, inp):
        audio_feat, text_feat = inp
        audio_feat = torch.squeeze(audio_feat)
        text_feat = torch.squeeze(text_feat)
        
        concat = torch.cat([audio_feat, text_feat], dim=1)
        x = self.classifier(concat)
        return x
