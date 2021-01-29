import warnings
import logging
import sys

sys.path.append("./libs/")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import argparse
import os
import pdb
import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer import seed_everything
from torch.utils import data
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.getter import get_instance
from utils.random_seed import SEED


class pipeline(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_instance(self.config["model"])
        self.loss = get_instance(self.config["loss"])

    def prepare_data(self):
        self.train_dataset = get_instance(self.config["dataset"]["train"])
        self.val_dataset = get_instance(self.config["dataset"]["val"])

    def forward(self, batch):
        logits = self.model(batch)
        return logits

    def training_step(self, batch, batch_idx):
        inps, lbls = batch
        logits = self.forward(inps)
        loss = self.loss(logits, lbls).mean()
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        inps, lbls = batch
        logits = self.forward(inps)
        loss = self.loss(logits, lbls)
        acc = (logits.argmax(-1) == lbls).float()
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        acc = torch.mean(torch.cat([o["acc"] for o in outputs], dim=0))
        out = {"val_loss": loss, "val_acc": acc}
        return {**out, "log": out}

    def train_dataloader(self):

        train_dataloader = get_instance(
            self.config["dataset"]["train"]["loader"],
            dataset=self.train_dataset,
            num_workers=4,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = get_instance(
            self.config["dataset"]["val"]["loader"],
            dataset=self.val_dataset,
            num_workers=4,
        )
        return val_dataloader

    def configure_optimizers(self):
        optimizer = get_instance(
            self.config["optimizer"], params=self.model.parameters()
        )
        lr_scheduler = get_instance(config["scheduler"], optimizer=optimizer)
        return [optimizer], [lr_scheduler]


def train(config):
    classify_pipeline = pipeline(config)
    cp_dir = config["save_dir"]
    cp_dir = os.path.join(cp_dir, config["model"]["name"]) + str(
        config.get("id", "None")
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=cp_dir + "/{epoch}-{train_loss:.3f}-{val_acc:.2f}",
        monitor="val_acc",
        verbose=config["verbose"],
        save_top_k=7,
    )
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["nepochs"],
        tpu_cores=(8 if config["use_tpu"] else 0),
        progress_bar_refresh_rate=2,
        gpus=(1 if torch.cuda.is_available() else 0),
        check_val_every_n_epoch=config["trainer"]["val_step"],
        checkpoint_callback=checkpoint_callback,
        default_root_dir="runs",
        logger=pl.loggers.TensorBoardLogger(
            "runs/", name=config["model"]["name"], version=config.get("id", "None")
        ),
    )
    trainer.fit(classify_pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit.yaml")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--use_tpu", action="store_true", default=False)
    parser.add_argument("--debug", default=False)
    parser.add_argument("--save_dir", default="./runs_lightning")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()
    seed_everything(seed=args.seed)
    config_path = args.config
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    config["debug"] = args.debug
    config["use_tpu"] = args.use_tpu
    config["save_dir"] = args.save_dir
    config["verbose"] = args.verbose
    train(config)
