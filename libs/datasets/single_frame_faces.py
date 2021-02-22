from torch.utils import data
from PIL import Image
from transforms import *

import torchvision.transforms as tf
import pandas as pd
import cv2
import os

class DatasetAdvance(data.Dataset):
	def __init__(self, csv_path, data_dir, IMG_SIZE, is_train):
		train_aug = Compose(
			[
				Resize(IMG_SIZE, IMG_SIZE),
				Transpose(p=0.5),
				HorizontalFlip(p=0.5),
				VerticalFlip(p=0.5),
				ShiftScaleRotate(p=0.5),
				HueSaturationValue(
					hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
				),
				RandomBrightnessContrast(
					brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
				),
				Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
					max_pixel_value=255.0,
					p=1.0,
				),
				CoarseDropout(p=0.5),
				Cutout(p=0.5),
				ToTensorV2(p=1.0),
			],
			p=1.0
		)

		val_aug = Compose(
			[
				Resize(IMG_SIZE, IMG_SIZE),
				Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
					max_pixel_value=255.0,
					p=1.0,
				),
				ToTensorV2(p=1.0),
			],
			p=1.0
		)

		self.data_dir = data_dir
		self.data = pd.read_csv(csv_path)
		self.IMG_SIZE = IMG_SIZE
		self.tf = train_aug if is_train else val_aug
		self.is_train = is_train

	def __getitem__(self, index):
		path, lbl = self.data.image_id.values[index], self.data.label.values[index]
		path = os.path.join(self.data_dir, path)
		img = cv2.imread(path)[:, :, ::-1]
		img = self.tf(image=img)["image"]
		return img, lbl

	def __len__(self):
		return len(self.data)