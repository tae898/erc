from torch.utils import data
from PIL import Image
import pickle

import torchvision.transforms as tf
import pandas as pd
import cv2
import os
from utils.video import loadvideo
from glob import glob
import json


class IEMOCAP(data.Dataset):
    def __init__(self, data_dir, faces_dir, label_path, every_N, num_frames,
                 image_size, normalize, datatype):
        self.emotion2num = {'anger': 0,
                            'disgust': 1,
                            'excited': 2,
                            'fear': 3,
                            'frustration': 4,
                            'happiness': 5,
                            'neutral': 6,
                            'other': 7,
                            'sadness': 8,
                            'surprise': 9,
                            'undecided': 10
                            }
        self.num2emotion = {val: key for key, val in self.emotion2num.items()}

        with open(label_path, 'r') as stream:
            labels = json.load(stream)[datatype]

        uttids = [uttid for uttid, _ in labels.items()]
        self.labels = [self.emotion2num[labels[uttid]] for uttid in uttids]
        self.videos_path = [os.path.join(
            data_dir, uttid + '.mp4') for uttid in uttids]
        self.faces_path = [os.path.join(
            faces_dir, uttid + '.pkl') for uttid in uttids]

        assert len(self.labels) == len(
            self.videos_path) == len(self.faces_path)

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):
        vidpath, lbl = self.videos_path[index], self.labels[index]
        with open(self.faces_path[index], 'rb') as stream:
            face = pickle.load(stream)
        video = loadvideo(path=vidpath, every_N=self.every_N,
                          num_frames=self.num_frames,
                          frame_width=self.image_size,
                          frame_height=self.image_size,
                          face=face)

        return video, lbl

    def __len__(self):
        return len(self.labels)
