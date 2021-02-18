from torch.utils import data
from PIL import Image
import pickle
import pandas as pd
import cv2
import os
from glob import glob
import json
import av
import math
from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers
from PIL import Image
import torch
from tqdm import tqdm
import pprint
from collections import Counter


def loadN(path, every_N=1):
    """Load every Nth frame from the original video."""
    container = av.open(path)
    frames = []
    for idx, frame in enumerate(container.decode(video=0)):
        if idx % every_N != 0:
            continue
        numpy_RGB = np.array(frame.to_image())
        frames.append(numpy_RGB)
    container.close()

    return frames


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class LetterBoxPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def loadvideo(path, every_N, num_frames, frame_width=224, frame_height=224,
              face=None):
    """Return RGB numpy arrays.

    This function uses the python av package, which uses ffmpeg in the backend.
    AFAIK, this is faster than the opencv counterpart.

    Parameters
    ----------
    path: string
        Path to the video.
    every_N: int
        The original video will first be resampled by sampling every Nth frame
        of the original video. Choose this value considering the fps of the
        original video.
    num_frames: int
        The total number of frames you wish to retrieve.

        if num_frames < length of video:
            randomly crop in time.
        if num_frames > length of video:
            pad zero-valued frames at the end (post padding)

    frame_width, frame_height: int
        Every frame will be resized to this value. If the aspect ratio cannot be
        preserved, then it'll be zero-padded.
    face: dict
        This contains face information in every frame.
        # TODO: Find a meaningful way to incorporate face information.

    Returns
    -------
    frames: np.ndarray
        The shape is (num_frames, channels, height, width)

    """
    assert isinstance(every_N, int)
    frames = loadN(path, every_N)

    if len(frames) < num_frames:
        for i in range(num_frames - len(frames)):
            frames.append(
                np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
    elif len(frames) > num_frames:
        start_idx = np.random.randint(0, len(frames) - num_frames+1)
        frames = frames[start_idx:start_idx+num_frames]

    assert len(frames) == num_frames

    frames = [Image.fromarray(frame) for frame in frames]
    tf = transforms.Compose([
        LetterBoxPad(fill=0, padding_mode='constant'),
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frames = [tf(frame) for frame in frames]
    frames = torch.stack(frames)

    return frames


def get_existing_paths(dir, uttids, extensions):
    """Get existing paths."""
    existing_paths = {}
    for uttid in uttids:
        candidates = [os.path.join(dir, uttid + extension)
                      for extension in extensions]
        for candidate in candidates:
            if os.path.isfile(candidate):
                existing_paths[uttid] = candidate

    return existing_paths


def find_common_uttids(*args):
    uttids_all = [[uttid for uttid, _ in arg.items()]
                  for arg in args if arg is not None]
    return set(uttids_all[0]).intersection(*uttids_all)


class Label(data.Dataset):
    def __init__(self, label_path, emotions, datatype):
        self.emotion2num = {emo: num for num, emo in enumerate(emotions)}
        self.num2emotion = {num: emo for num, emo in enumerate(emotions)}

        with open(label_path, 'r') as stream:
            self.uttid2emotion = json.load(stream)[datatype]

        self.uttid2emotion = {uttid: emotion for uttid,
                              emotion in self.uttid2emotion.items()
                              if emotion in emotions}

        self.uttid2num = {uttid: self.emotion2num[emotion]
                          for uttid, emotion in self.uttid2emotion.items()}

        self.uttids = None
        self.labels = None

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)


class VideoModality(data.Dataset):
    def __init__(self, videos_dir, uttids, every_N, num_frames,
                 frame_height, frame_width, faces_dir=None):
        self.uttid2videopath = get_existing_paths(
            videos_dir, uttids, ['.avi', '.mp4'])

        if faces_dir is not None:
            self.uttid2facepath = get_existing_paths(
                faces_dir, uttids, ['.pkl'])
            uttids = find_common_uttids(
                self.uttid2videopath, self.uttid2facepath)
            self.uttid2facepath = {
                uttid: self.uttid2facepath[uttid] for uttid in uttids}
            assert len(self.uttid2videopath) == len(self.uttid2facepath)

        # This means that the videos are pre-cropped face videos.
        if len(self.uttid2videopath) == 0:
            assert faces_dir is None
            self.is_video_cropped_face = True
            self.uttid2videopath = {uttid: glob(os.path.join(
                videos_dir, uttid, '*.mp4')) for uttid in uttids}
            self.uttid2videopath = {
                uttid: list_of_paths for uttid, list_of_paths
                in self.uttid2videopath.items() if len(list_of_paths) != 0}
            uttids = [uttid for uttid, _ in self.uttid2videopath.items()]

        else:
            self.is_video_cropped_face = False

        self.videopaths = None
        self.facepaths = None

        self.every_N = every_N
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __getitem__(self, index):
        if self.facepaths is not None:
            with open(self.facepaths[index], 'rb') as stream:
                face = pickle.load(stream)
        else:
            face = None

        if self.is_video_cropped_face:
            path = self.select_emotional_person(self.videopaths[index])
        else:
            path = self.videopaths[index]

        video = loadvideo(path=path, every_N=self.every_N,
                          num_frames=self.num_frames,
                          frame_width=self.image_size,
                          frame_height=self.image_size,
                          face=face)
        return video

    def __len__(self):
        return len(self.videopaths)

    def select_emotional_person(self, list_of_paths):
        persons = [loadN(path, every_N=1) for path in list_of_paths]
        persons = [sum([1 if np.sum(frame) < 1 else 0 for frame in person])
                   for person in persons]
        person_id = np.argmin(persons)

        return list_of_paths[person_id]


class AudioModality(data.Dataset):
    def __init__(self, audios_dir, uttids):
        self.uttid2audiopath = get_existing_paths(
            audios_dir, uttids, ['.wav', '.mp3'])
        self.audiopaths = None

    def __getitem__(self, index):
        # TODO: load audio with librosa instead of just path and convert it to
        # a spectrogram.
        return self.audiopaths[index]

    def __len__(self):
        return len(self.audiopaths)


class TextModality(data.Dataset):
    def __init__(self, texts_dir, uttids):
        self.uttid2textpath = get_existing_paths(texts_dir, uttids, ['.json'])
        self.textpaths = None

    def __getitem__(self, index):
        # TODO: load text utterance (json), tokenize it, and use BERT or
        # something.
        return self.textpaths[index]

    def __len__(self):
        return len(self.textpaths)


class MultiModalDataset(Label, VideoModality, TextModality, AudioModality):
    def __init__(self, label_path, every_N, num_frames, image_size, datatype,
                 videos_dir=None, faces_dir=None, audios_dir=None,
                 texts_dir=None, balancing=None):
        if all(dir is None for dir in [videos_dir, audios_dir, texts_dir]):
            raise ValueError("You should provide at least one modality!!!")

        self.modalities = [foo for foo, bar
                           in zip(['video', 'audio', 'text'],
                                  [videos_dir, audios_dir, texts_dir])
                           if bar is not None]
        Label.__init__(self, label_path, self.emotions, datatype)

        if videos_dir is not None:
            VideoModality.__init__(self, videos_dir,
                                   list(self.uttid2num.keys()), every_N,
                                   num_frames, image_size, image_size,
                                   faces_dir)
        else:
            self.uttid2videopath = None

        if audios_dir is not None:
            AudioModality.__init__(
                self, audios_dir, list(self.uttid2num.keys()))
        else:
            self.uttid2audiopath = None

        if texts_dir is not None:
            TextModality.__init__(self, texts_dir, list(self.uttid2num.keys()))
        else:
            self.uttid2textpath = None

        print(f"finding common utterances in {self.modalities} modalities, "
              f"in the {datatype} dataset ...")
        self.uttids = find_common_uttids(self.uttid2videopath,
                                         self.uttid2audiopath,
                                         self.uttid2textpath)

        self.labels = [self.uttid2num[uttid] for uttid in self.uttids]

        self.videopaths, self.facepaths, self.audiopaths, self.textpaths = \
            None, None, None, None
        if self.uttid2videopath is not None:
            self.videopaths = [self.uttid2videopath[uttid]
                               for uttid in self.uttids]
        if faces_dir is not None:
            self.facepaths = [self.uttid2facepath[uttid]
                              for uttid in self.uttids]
        if self.uttid2audiopath is not None:
            self.audiopaths = [self.uttid2audiopath[uttid]
                               for uttid in self.uttids]
        if self.uttid2textpath is not None:
            self.textpaths = [self.uttid2textpath[uttid]
                              for uttid in self.uttids]

        print(f"In total there are {len(self.labels)} utterances in common, "
              f"across the {self.modalities} modalities, in the {datatype} "
              f"dataset")

        if datatype == 'train':
            self.balance_class(balancing)

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):

        label = Label.__getitem__(self, index)

        if self.videopaths is not None:
            video = VideoModality.__getitem__(self, index)
        if self.audiopaths is not None:
            audio = AudioModality.__getitem__(self, index)
        if self.textpaths is not None:
            text = TextModality.__getitem__(self, index)

        return video, label

    def __len__(self):
        return len(self.labels)

    def count_classes(self):
        self.counts = {num: count
                       for num, count in dict(Counter(self.labels)).items()}

    def balance_class(self, balancing):
        self.count_classes()
        print(f"class distributions before balancing:")
        pprint.PrettyPrinter(indent=2).pprint(
            {self.num2emotion[num]: count
                for num, count in self.counts.items()})

        if balancing is None:
            print(f"Training will be done without class balancing")

        elif 'under' in balancing or 'down' in balancing:
            print(f"Training will be done with downsampling classes")

            min_class = min(self.counts, key=self.counts.get)
            count_to_fix = self.counts[min_class]
            indexes = []

            for num in list(self.counts.keys()):
                candidates = np.where(np.array(self.labels) == num)[0]
                candidates = np.random.permutation(candidates)[:count_to_fix]
                for cand in candidates:
                    indexes.append(cand)

            self.labels = [self.labels[i] for i in indexes]
            new_length = len(self.labels)
            if self.videopaths is not None:
                self.videopaths = [self.videopaths[i] for i in indexes]
                assert len(self.videopaths) == new_length
            if self.facepaths is not None:
                self.facepaths = [self.facepaths[i] for i in indexes]
                assert len(self.facepaths) == new_length
            if self.audiopaths is not None:
                self.audiopaths = [self.audiopaths[i] for i in indexes]
                assert len(self.audiopaths) == new_length
            if self.textpaths is not None:
                self.textpaths = [self.textpaths[i] for i in indexes]
                assert len(self.textpaths) == new_length

        elif 'over' in balancing:
            print(f"Training will be done with oversampling classes")

            max_class = max(self.counts, key=self.counts.get)
            count_to_fix = self.counts[max_class]
            indexes = []

            for num in list(self.counts.keys()):
                candidates = np.where(np.array(self.labels) == num)[0]
                candidates = candidates.tolist()

                for cand in np.random.choice(candidates,
                                             size=(count_to_fix - len(candidates))):
                    candidates.append(cand)
                assert len(candidates) == count_to_fix
                assert len(np.unique(candidates)) == (
                    np.array(self.labels) == num).sum()

                for cand in candidates:
                    indexes.append(cand)

            self.labels = [self.labels[i] for i in indexes]
            new_length = len(self.labels)
            if self.videopaths is not None:
                self.videopaths = [self.videopaths[i] for i in indexes]
                assert len(self.videopaths) == new_length
            if self.facepaths is not None:
                self.facepaths = [self.facepaths[i] for i in indexes]
                assert len(self.facepaths) == new_length
            if self.audiopaths is not None:
                self.audiopaths = [self.audiopaths[i] for i in indexes]
                assert len(self.audiopaths) == new_length
            if self.textpaths is not None:
                self.textpaths = [self.textpaths[i] for i in indexes]
                assert len(self.textpaths) == new_length

        else:
            raise ValueError

        self.count_classes()

        print(f"class distributions after balancing:")
        pprint.PrettyPrinter(indent=2).pprint(
            {self.num2emotion[num]: count
                for num, count in self.counts.items()})


class MELD(MultiModalDataset):
    def __init__(self, label_path, every_N, num_frames, image_size, datatype,
                 videos_dir=None, faces_dir=None, audios_dir=None,
                 texts_dir=None, balancing=None):
        self.emotions = ['anger',
                         'disgust',
                         'fear',
                         'joy',
                         'neutral',
                         'sadness',
                         'surprise']
        super().__init__(label_path, every_N, num_frames, image_size, datatype,
                         videos_dir, faces_dir, audios_dir,
                         texts_dir, balancing)


class AFEW(MultiModalDataset):
    def __init__(self, label_path, every_N, num_frames, image_size, datatype,
                 videos_dir=None, faces_dir=None, audios_dir=None,
                 texts_dir=None, balancing=None):
        self.emotions = ['angry',
                         'disgust',
                         'fear',
                         'happy',
                         'neutral',
                         'sad',
                         'surprise']
        super().__init__(label_path, every_N, num_frames, image_size, datatype,
                         videos_dir, faces_dir, audios_dir,
                         texts_dir, balancing)


class CAER(MultiModalDataset):
    def __init__(self, label_path, every_N, num_frames, image_size, datatype,
                 videos_dir=None, faces_dir=None, audios_dir=None,
                 texts_dir=None, balancing=None):
        self.emotions = ['anger',
                         'disgust',
                         'fear',
                         'happy',
                         'neutral',
                         'sad',
                         'surprise']
        super().__init__(label_path, every_N, num_frames, image_size, datatype,
                         videos_dir, faces_dir, audios_dir,
                         texts_dir, balancing)


class IEMOCAP(MultiModalDataset):
    def __init__(self, label_path, every_N, num_frames, image_size, datatype,
                 videos_dir=None, faces_dir=None, audios_dir=None,
                 texts_dir=None, balancing=None):
        self.emotions = ['anger',
                         'disgust',
                         'excited',
                         'fear',
                         'frustration',
                         'happiness',
                         'neutral',
                         'other',
                         'sadness',
                         'surprise',
                         'undecided']
        super().__init__(label_path, every_N, num_frames, image_size, datatype,
                         videos_dir, faces_dir, audios_dir,
                         texts_dir, balancing)
