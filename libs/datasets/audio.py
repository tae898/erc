from torch.utils import data
from PIL import Image

import torchvision.transforms as tf
import librosa as lb
import pandas as pd
import numpy as np
import cv2
import os

class AudioDataset(data.Dataset):
	def __init__(self, csv_path, data_dir, format, is_train, sr, duration, n_mels):
		self.csv = pd.read_csv(csv_path, dtype={'audio_id': 'string'})
		self.data_dir = data_dir
		self.format = format
		self.is_train = is_train

		self.sr = sr
		self.n_mels = n_mels
		self.duration = duration
		self.audio_length = duration * sr

	def get_audio(self, audio_path):
		offset = 0
		y, _ = lb.load(audio_path, sr=self.sr, duration=self.duration, offset=offset)

		# Resize the audio sequence to a fixed size audio_length
		if len(y) < self.audio_length:
			y = np.concatenate([y, np.zeros(self.audio_length - len(y))])
		elif len(y) > self.audio_length:
			start = 0
			y = y[start:start + self.audio_length]

		y = y.astype(np.float32, copy=False)
		return y
	
	def audio2melspec(self, y):
		melspec = lb.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels)
		melspec = lb.power_to_db(melspec).astype(np.float32)
		return melspec

	def melspec2img(self, melspec):
		# Convert 1 channel to 3 channels
		img = np.stack([melspec, melspec, melspec], axis=-1)

		# Standardization
		img = (img - img.mean()) / (img.std() + 1E-6)

		# Normalization
		min_val, max_val = img.min(), img.max()
		if (max_val - min_val) > 1E-6:
			img = np.clip(img, min_val, max_val)
			img = 255 * (img - min_val) / (max_val - min_val)
			img = img.astype(np.uint8)
		else:
			img = np.zeros_like(img, dtype=np.uint8)

		img = img / 255.0
		return np.moveaxis(img, 2, 0).astype(np.float32)

	def __getitem__(self, index):
		df_row = self.csv.iloc[index]
		audio_id, label = df_row['audio_id'], df_row['label']
		audio_path = os.path.join(self.data_dir, audio_id)
		audio_path_w_format = audio_path + self.format
		y = self.get_audio(audio_path_w_format)
		melspec = self.audio2melspec(y)
		image = self.melspec2img(melspec)
		return image, label		

	def __len__(self):
		return len(self.csv)