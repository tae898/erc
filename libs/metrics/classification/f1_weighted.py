import torch
import numpy as np
from sklearn.metrics import f1_score

class F1Weighted():
    def __init__(self, *args, **kwargs):
        self.pred = []
        self.true = []

    def calculate(self, output, target):
        batch_pred = torch.argmax(output, dim=1)
        batch_pred = batch_pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return batch_pred, target

    def update(self, value):
        self.pred.append(value[0])
        self.true.append(value[1])

    def reset(self):
        self.pred = []
        self.true = []

    def value(self):
        self.pred = np.reshape(self.pred, newshape=(-1,))
        self.true = np.reshape(self.true, newshape=(-1,))
        return f1_score(self.pred, self.true, average='weighted')

    def summary(self):
        print(f'F1 weighted: {self.value()}')
