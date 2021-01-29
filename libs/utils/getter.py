from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split

from losses import *
from datasets import *
from models import *
from metrics import *
from dataloaders import *

from .random_seed import set_seed


def get_function(name):
    return globals()[name]


def get_instance(config, **kwargs):
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)


def get_dataloader(cfg, dataset):
    collate_fn = None
    if cfg.get('collate_fn', False):
        collate_fn = get_function(cfg['collate_fn'])

    dataloader = get_instance(cfg,
                              dataset=dataset,
                              collate_fn=collate_fn)
    return dataloader


def get_single_data(cfg, with_dataset=True):
    dataset = get_instance(cfg)
    dataloader = get_dataloader(cfg['loader'], dataset)
    output = dataloader
    if with_dataset:
        output = [output, dataset]
    return output


def get_data(cfg, seed, with_dataset=False):
    if cfg.get('train', False) and cfg.get('val', False):
        # If dataset already has trainval split
        # Get (seeded) dataloader for train
        set_seed(seed)
        train_dataloader, train_dataset = get_single_data(cfg['train'],
                                                          with_dataset=True)
        # Get (seeded) dataloader for test
        set_seed(seed)
        val_dataloader, val_dataset = get_single_data(cfg['val'],
                                                      with_dataset=True)
    elif cfg.get('trainval', False):
        # If dataset does not have any trainval split
        trainval_cfg = cfg['trainval']
        # Split dataset train:val = ratio:(1-ratio)
        set_seed(seed)
        ratio = trainval_cfg['ratio']
        dataset = get_instance(trainval_cfg)
        train_sz = max(1, int(ratio * len(dataset)))
        val_sz = len(dataset) - train_sz
        train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])
        # Get dataloader
        train_dataloader = get_dataloader(trainval_cfg['loader']['train'],
                                          train_dataset)
        val_dataloader = get_dataloader(trainval_cfg['loader']['val'],
                                        val_dataset)
    else:
        raise Exception('Dataset config is not correctly formatted.')

    output = [train_dataloader, val_dataloader]
    if with_dataset:
        output += [train_dataset, val_dataset]
    return output
