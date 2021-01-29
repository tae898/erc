from torch.utils import data
import av
import numpy as np

import torch


class MeldDataset(data.Dataset):
    def __init__(self, list_vid_paths, labels, choose_frames, num_frames):
        """Initialize.

        Attributes
        ----------
        list_vid_paths: list
            A list of full video paths.

        labels: list
            A list of string labels. They will be converted to numbers here.

        """
        self.emotion2num = {'neutral': 0,
                            'surprise': 1,
                            'fear': 2,
                            'sadness': 3,
                            'joy': 4,
                            'disgust': 5,
                            'anger': 6}

        self.sentiment2num = {'neutral': 0,
                              'positive': 1,
                              'negative': 2}

        self.choose_frames = choose_frames
        self.num_frames = num_frames

        self.labels = [self.emotion2num[lbl] for lbl in labels]
        self.list_vid_paths = list_vid_paths

        assert len(self.labels) == len(self.list_vid_paths)

    def __len__(self):
        """Denote the total number of samples."""
        return len(self.list_vid_paths)

    def __getitem__(self, index):
        """Generate ONE sample of data, not batch."""
        # Select sample
        vidpath = self.list_vid_paths[index]

        # Load data and get label
        container = av.open(vidpath)

        X = []
        for frame in container.decode(video=0):
            numpy_RGB = np.array(frame.to_image())
            X.append(numpy_RGB)
        X = np.stack(X)
        X = torch.tensor(X)

        X = X.permute(0, 3, 1, 2)  # from THWC to TCHW

        if self.choose_frames == 'random_consecutive':
            frame_indexes = [i for i in range(X.shape[0])]
            if X.shape[0] < self.num_frames:
                indexes = upsample_indexes(
                    [idx for idx in range(X.shape[0])], self.num_frames)
                X = torch.stack([X[idx] for idx in indexes])
            elif X.shape[0] == self.num_frames:
                pass
            else:
                idx_start = np.random.randint(
                    0, X.shape[0] - self.num_frames + 1)
                X = X[idx_start:idx_start+self.num_frames]

        if X.shape[0] == 1:
            X = torch.squeeze(X)

        y = self.labels[index]

        return X, y


def upsample_indexes(indexes, len_desired):
    """Upsample linearly."""
    assert len(indexes) < len_desired, f"The number of indexes should be " \
        f"smaller than the desired length"

    duplicates = np.linspace(0, len(indexes) - 1, len_desired).round()

    indexes = [indexes[int(dup)] for dup in duplicates]

    assert len(indexes) == len_desired, f"Sampling gone wrong."

    return indexes
