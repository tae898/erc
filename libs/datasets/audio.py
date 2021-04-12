from torch.utils import data
from PIL import Image

import torchvision.transforms as tf
import librosa as lb
import pandas as pd
import numpy as np
import cv2
import os

from transforms import *

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
  
		# Audio augmentation
		self.audio_transform = AudioTfCompose(
      		[
            	OneOf(
                 	[
                    	GaussianNoiseSNR(min_snr=10), 
                     	PinkNoiseSNR(min_snr=10)
                    ]), 
                TimeShift(sr=self.sr), 
                VolumeControl(p=0.5)
            ])
  
		# Image transformation
		self.image_transform = Compose(
      		[
				Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
					max_pixel_value=255.0,
					p=1.0,
				),
				ToTensorV2(p=1.0),
           	])

	def get_audio(self, audio_path, dur):
		offset = 0
		if self.is_train and self.duration < dur + 1:
			offset = np.random.uniform(0, int(dur - self.duration))

		# Assume the default sample rate = 22050
		y, _ = lb.load(audio_path, sr=None, duration=self.duration, offset=offset)

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

		# Convert to pixel range [0..255]
		img = (img - img.mean()) / (img.std() + 1E-6)
		min_val, max_val = img.min(), img.max()
		if (max_val - min_val) > 1E-6:
			img = np.clip(img, min_val, max_val)
			img = 255 * (img - min_val) / (max_val - min_val)
			img = img.astype(np.uint8)
		else:
			img = np.zeros_like(img, dtype=np.uint8)

		return img.astype(np.float32)

	def __getitem__(self, index):
		df_row = self.csv.iloc[index]
		audio_id, label, dur = df_row['audio_id'], df_row['label'], df_row['duration']
		audio_path = os.path.join(self.data_dir, audio_id)
		audio_path_w_format = audio_path + self.format
  
		y = self.get_audio(audio_path_w_format, dur)
		if self.is_train:
			y = self.audio_transform(y)
		melspec = self.audio2melspec(y)
		image = self.melspec2img(melspec)
		image = self.image_transform(image=image)['image']
		return image, label, audio_id

	def __len__(self):
		return len(self.csv)