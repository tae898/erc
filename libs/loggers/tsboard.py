
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger():
    def __init__(self, path):
        assert path != None, "path is None"
        self.writer = SummaryWriter(log_dir=path)

    def update_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def update_loss(self, phase, value, step):
        self.update_scalar(f'{phase}/loss', value, step)

    def update_metric(self, phase, metric, value, step):
        self.update_scalar(f'{phase}/{metric}', value, step)

    def update_lr(self, gid, value, step):
        self.update_scalar(f'lr/group_{gid}', value, step)
