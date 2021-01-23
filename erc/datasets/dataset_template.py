from torch.utils import data


class DatasetTemplate(data.Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
