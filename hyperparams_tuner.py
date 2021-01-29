import sys

sys.path.append("./libs/")

import argparse
import pprint

import torch
import torch.nn as nn
import yaml
from torch.utils import data
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from utils.getter import get_instance
from utils.random_seed import set_seed
from workers.trainer import Trainer

import optuna
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from losses import *
from datasets import *
from models import *
from metrics import *
from dataloaders import *
from schedulers import *

def train(trial, config):
	assert config is not None, "Do not have config file!"

	# pprint.PrettyPrinter(indent=2).pprint(config)

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
	# criterion_name = trial.suggest_categorical("criterion", ["FocalLoss", "CrossEntropyLoss"])
	# criterion = globals()[criterion_name]()

	# 4: Define Optimizer
	set_seed()
	# optimizer_name = "Adam"
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_float("lr", low=0.0001, high=0.01, log=True)
	weight_decay = trial.suggest_float("weight_decay", low=0, high=0.00001)
	optimizer = getattr(torch.optim, optimizer_name)(lr=lr, weight_decay=weight_decay, params=model.parameters()) 
	if pretrained is not None:
		optimizer.load_state_dict(pretrained["optimizer_state_dict"])

	# 5: Define Scheduler
	set_seed()
	scheduler_name = "StepLR"
	step_size = trial.suggest_int("step_size", low=1, high=5)
	gamma = trial.suggest_float("gamma", low=0.1, high=0.5)
	scheduler = globals()[scheduler_name](optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)

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
		optimier=optimizer,
		scheduler=scheduler,
		metric=metric,
	)

	# 8: Start to train
	set_seed()
	trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)
	return trainer.best_metric[config["metric"][0]["name"]].item()

class Objective(object):
	def __init__(self, config):
		self.config = config
	
	def __call__(self, trial):
		val_acc = train(trial, self.config)
		sys.stdout.flush()
		return val_acc


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config")
	parser.add_argument("--gpus", default=None)
	parser.add_argument("--fp16", action="store_true", default=False)
	parser.add_argument("--fp16_opt_level", default="O2")
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--optuna_n_trials", default=1)
	parser.add_argument("--verbose", action="store_true", default=False)
	args = parser.parse_args()
	config_path = args.config
	config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
	config["gpus"] = args.gpus
	config["debug"] = args.debug
	config["fp16"] = args.fp16
	config["fp16_opt_level"] = args.fp16_opt_level
	config["verbose"] = args.verbose

	# Create a study
	study = optuna.create_study(direction="maximize", study_name="optuna_v0")
	study.optimize(Objective(config), n_trials=int(args.optuna_n_trials))
	print("Best trial:")
	trial = study.best_trial
	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
	
	# Save study
	import joblib
	joblib.dump(study, 'vit_optuna.pkl')
