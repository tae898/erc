import yaml
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils import data
from tqdm import tqdm
from torchnet import meter

from workers.trainer import Trainer
from utils.random_seed import set_seed, set_determinism
from utils.getter import get_instance, get_data

import argparse
import pprint


def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    # Get device
    dev_id = 'cuda:{}'.format(config['gpus']) \
        if torch.cuda.is_available() and config.get('gpus', None) is not None \
        else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]

    pretrained = None
    if (pretrained_path != None):
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model"]:
            config[item] = pretrained["config"][item]

    # 1: Load datasets
    train_dataloader, val_dataloader = \
        get_data(config['dataset'], config['seed'])

    # 2: Define network
    set_seed(config['seed'])
    model = get_instance(config['model']).to(device)

    # Train from pretrained if it is not None
    if pretrained is not None:
        model.load_state_dict(pretrained['model_state_dict'])

    # 3: Define loss
    set_seed(config['seed'])
    criterion = get_instance(config['loss']).to(device)

    # 4: Define Optimizer
    set_seed(config['seed'])
    optimizer = get_instance(config['optimizer'],
                             params=model.parameters())
    if pretrained is not None:
        optimizer.load_state_dict(pretrained['optimizer_state_dict'])

    # 5: Define Scheduler
    set_seed(config['seed'])
    scheduler = get_instance(config['scheduler'],
                             optimizer=optimizer)

    # 6: Define metrics
    set_seed(config['seed'])
    metric = {mcfg['name']: get_instance(mcfg)
              for mcfg in config['metric']}

    # 6: Create trainer
    set_seed(config['seed'])
    trainer = Trainer(device=device,
                      config=config,
                      model=model,
                      criterion=criterion,
                      optimier=optimizer,
                      scheduler=scheduler,
                      metric=metric)

    # 7: Start to train
    set_seed(config['seed'])
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['debug'] = args.debug

    set_determinism()
    train(config)
