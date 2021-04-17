from torch.utils import data
from PIL import Image
import pandas as pd
import numpy as np
import os
import json

class ContextAwareDataset(data.Dataset):
	def __init__(self, csv_path, audio_feat_dir, text_feat_dir, ordered_json_list, num_utt=2, dataset='train'):
		self.csv = pd.read_csv(csv_path, dtype={'audio_id': 'string'})
		self.audio_feat_dir = audio_feat_dir
		self.text_feat_dir = text_feat_dir
		self.json_data = None
		with open(ordered_json_list) as f:
			self.json_data = json.load(f)
		self.num_utt = num_utt
		self.dataset = dataset

	def __getitem__(self, index):
		df_row = self.csv.iloc[index]
		audio_id, label, dur = df_row['audio_id'], df_row['label'], df_row['duration']
		dialog_id = audio_id.split('_')[0]
		utter_id = audio_id.split('_')[1]
		utter_list = self.json_data[self.dataset][dialog_id]
		utter_pos = utter_list.index(audio_id)
		audio_feats = []
		text_feats = []
		for _ in range(self.num_utt):
			if utter_pos < 0:
				audio_feats.append(np.zeros(shape=(1, 1280)))
				text_feats.append(np.zeros(shape=(1, 1024)))
			else:
				audio_feat = np.load(os.path.join(self.audio_feat_dir, utter_list[utter_pos] + '.npy'), allow_pickle=True)
				text_feat = np.load(os.path.join(self.text_feat_dir, utter_list[utter_pos] + '.npy'), allow_pickle=True).item()['features']
				audio_feats.append(audio_feat)
				text_feats.append(text_feat)
			utter_pos -= 1
		audio_feats = np.squeeze(np.array(audio_feats[::-1]))
		text_feats = np.squeeze(np.array(text_feats[::-1]))
		return (audio_feats, text_feats), label

	def __len__(self):
		return len(self.csv)