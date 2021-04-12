from torch.utils import data
from PIL import Image

import torchvision.transforms as tf
import librosa as lb
import pandas as pd
import numpy as np
import cv2
import os

from transforms import *

class AudioTextFeatureVectorDataset(data.Dataset):
	def __init__(self, csv_path, audio_feat_dir, text_feat_dir):
		self.csv = pd.read_csv(csv_path, dtype={'audio_id': 'string'})
		self.audio_feat_dir = audio_feat_dir
		self.text_feat_dir = text_feat_dir

	def __getitem__(self, index):
		df_row = self.csv.iloc[index]
		audio_id, label, dur = df_row['audio_id'], df_row['label'], df_row['duration']
		audio_feat = np.load(os.path.join(self.audio_feat_dir, audio_id + '.npy'), allow_pickle=True)
		text_feat = np.load(os.path.join(self.text_feat_dir, audio_id + '.npy'), allow_pickle=True).item()['features']
		return audio_feat, text_feat, label

	def __len__(self):
		return len(self.csv)