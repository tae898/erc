from torch.utils import data
from PIL import Image

import os


class ImageFolderDataset(data.Dataset):
    def __init__(self, img_dir, transforms):
        self.dir = img_dir
        self.fns = os.listdir(img_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        fn = self.fns[index]
        img_path = os.path.join(self.dir, fn)
        im = Image.open(img_path).convert('RGB')
        im = self.transforms(im)
        return im, fn

    def __len__(self):
        return len(self.fns)
