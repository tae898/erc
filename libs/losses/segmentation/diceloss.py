import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    r'''
    Dice Loss
    Ref: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/8
    '''

    def __init__(self, weights=None, ignore_index=None, size_average=True, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        if weights is None:
            self.weights = 1
        if isinstance(weights, list):
            self.weights = torch.FloatTensor(weights)
        self.size_average = size_average
        self.eps = eps

    def __call__(self, output, target):
        encoded_target = torch.zeros(output.size()).to(output.device)
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        output = F.softmax(output, dim=1)
        intersection = output * encoded_target  # [B, C, H, W]
        numerator = 2 * intersection.sum((-1, -2))  # [B, C]
        denominator = output + encoded_target  # [B, C, H, W]
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum((-1, -2)) + self.eps

        loss_per_channel = self.weights * (1 - (numerator / denominator))

        loss = loss_per_channel.mean(1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
