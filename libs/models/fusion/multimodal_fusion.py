import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import getter

class FusionModel(nn.Module):
    def __init__(self, num_classes, AUDIO_FEAT_DIM=1280, TEXT_FEAT_DIM=1024, use_concat=False, single_utt=True):
        super().__init__()
        self.AUDIO_FEAT_DIM = AUDIO_FEAT_DIM
        self.TEXT_FEAT_DIM = TEXT_FEAT_DIM
        self.mbp = MBP(AUDIO_FEAT_DIM, TEXT_FEAT_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(AUDIO_FEAT_DIM + TEXT_FEAT_DIM, 512) if use_concat else nn.Linear(1000, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        self.fc = nn.Linear(1000, num_classes)
        dev_id = (
            "cuda:0"
            if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(dev_id)
        self.use_concat = use_concat
        self.single_utt = single_utt

    def forward(self, inp):
        if self.single_utt:
            audio_feat, text_feat = inp
            audio_feat = torch.squeeze(audio_feat)
            text_feat = torch.squeeze(text_feat)
            fused = torch.cat([audio_feat, text_feat], dim=1) if self.use_concat else self.mbp(audio_feat, text_feat)
            x = self.classifier(fused)
            return x
        else:
            audio_feats, text_feats = inp
            batch_size, num_utt, _ = audio_feats.shape
            res = torch.zeros(size=(batch_size, num_utt, 1000), device=self.device)
            for i in range(batch_size):
                audio_feat = audio_feats[i].float()
                text_feat = text_feats[i].float()
                fused = self.mbp(audio_feat, text_feat)
                res[i] = fused
            return res


class MBP(nn.Module):
    """
        Multi-modal Factorized Bilinear Pooling - https://arxiv.org/pdf/1708.01471.pdf
    """
    def __init__(self, AUDIO_FEAT_DIM, TEXT_FEAT_DIM, SUM_POOLING_WINDOW=3, OUTPUT_DIM=1000):
        super().__init__()
        self.AUDIO_FEAT_DIM = AUDIO_FEAT_DIM
        self.TEXT_FEAT_DIM = TEXT_FEAT_DIM
        self.SUM_POOLING_WINDOW = SUM_POOLING_WINDOW
        self.OUTPUT_DIM = OUTPUT_DIM
        self.FUSED_DIM = SUM_POOLING_WINDOW * OUTPUT_DIM

        self.audio_linear_projection = nn.Linear(AUDIO_FEAT_DIM, self.FUSED_DIM)
        self.text_linear_projection = nn.Linear(TEXT_FEAT_DIM, self.FUSED_DIM)
        self.dropout = nn.Dropout(0.2)

    def forward(self, audio, text):
        x = self.audio_linear_projection(audio)
        y = self.text_linear_projection(text)
        z = torch.mul(x, y)
        z = self.dropout(z)
        z = z.view(-1, 1, self.OUTPUT_DIM, self.SUM_POOLING_WINDOW)
        z = torch.sum(z, dim=3)
        z = torch.squeeze(z)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z)
        return z