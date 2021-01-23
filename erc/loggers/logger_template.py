class LoggerTemplate():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update_scalar(self, tag, value, step):
        raise NotImplementedError

    def update_loss(self, phase, value, step):
        raise NotImplementedError

    def update_metric(self, phase, metric, value, step):
        raise NotImplementedError

    def update_lr(self, gid, value, step):
        raise NotImplementedError
