import torch

from utils.segmentation import multi_class_prediction, binary_prediction


class PixelAccuracy():
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.pred_fn = multi_class_prediction
        if nclasses == 1:
            self.nclasses += 1
            self.pred_fn = binary_prediction
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        prediction = self.pred_fn(output)

        image_size = target.size(1) * target.size(2)

        ignore_mask = torch.zeros(target.size()).bool().to(target.device)
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index).bool()
        ignore_size = ignore_mask.sum((1, 2))

        correct = ((prediction == target) | ignore_mask).sum((1, 2))
        acc = (correct - ignore_size + 1e-6) / \
            (image_size - ignore_size + 1e-6)
        return acc.cpu()

    def update(self, value):
        self.total_correct += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        return (self.total_correct / self.sample_size).item()

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0

    def summary(self):
        print(f'Pixel Accuracy: {self.value():.6f}')
