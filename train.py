from workers.trainer import Trainer
from utils.random_seed import set_seed
from utils.getter import get_instance
from tqdm import tqdm
from torch.utils.data import DataLoadier
from torch.utils import data
import yaml
import torch.nn as nn
import torch
import os
import pprint
import argparse
import sys
sys.path.append("./libs/")


def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    dev_id = (
        "cuda:{}".format(config["gpus"])
        if torch.cuda.is_available() and config.get("gpus", None) is not None
        else "cpu"
    )
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]

    pretrained = None
    if pretrained_path != None:
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model"]:
            config[item] = pretrained["config"][item]

    # 1: Load datasets
    set_seed()
    train_dataset = get_instance(config["dataset"]["train"])
    train_dataloader = get_instance(
        config["dataset"]["train"]["loader"], dataset=train_dataset
    )

    set_seed()
    val_dataset = get_instance(config["dataset"]["val"])
    val_dataloader = get_instance(
        config["dataset"]["val"]["loader"], dataset=val_dataset
    )

    # 2: Define network
    set_seed()
    model = get_instance(config["model"]).to(device)

    # Train from pretrained if it is not None
    if pretrained is not None:
        model.load_state_dict(pretrained["model_state_dict"])

    # 3: Define loss
    set_seed()
    criterion = get_instance(config["loss"]).to(device)

    # 4: Define Optimizer
    set_seed()
    optimizer = get_instance(config["optimizer"], params=model.parameters())
    if pretrained is not None:
        optimizer.load_state_dict(pretrained["optimizer_state_dict"])

    # 5: Define Scheduler
    set_seed()
    scheduler = get_instance(config["scheduler"], optimizer=optimizer)

    # 6: Define metrics
    set_seed()
    metric = {mcfg["name"]: get_instance(mcfg) for mcfg in config["metric"]}

    # 7: Create trainer
    set_seed()
    trainer = Trainer(
        device=device,
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
    )

    # 7: Start to train
    set_seed()
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)

    # 8: Delete unused variable
    del (
        optimizer,
        model,
        train_dataloader,
        val_dataloader,
        train_dataset,
        val_dataset,
        trainer,
        metric,
        pretrained,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config_path = args.config
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    config["gpus"] = args.gpus
    config["debug"] = args.debug

    train(config)
