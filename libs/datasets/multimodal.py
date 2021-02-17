from torch.utils import data
from PIL import Image
import pickle
import pandas as pd
import cv2
import os
from glob import glob
import json
import av
from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers
from PIL import Image
import torch


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
    uttids_all = [[uttid for uttid, _ in arg.items()] for arg in args]
    return set(uttids_all[0]).intersection(*uttids_all)


class Label(data.Dataset):
    def __init__(self, label_path, emotions, datatype):
        self.emotion2num = {emo: idx for idx, emo in enumerate(emotions)}
        self.num2emotion = {idx: emo for idx, emo in self.emotion2num.items()}

        with open(label_path, 'r') as stream:
            self.uttid2emotion = json.load(stream)[datatype]

        self.uttid2num = {uttid: self.emotion2num[emotion]
                          for uttid, emotion in self.uttid2emotion.items()}
        self.uttids = None
        self.labels = None

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)


class VideoModality(data.Dataset):
    def __init__(self, videos_dir, faces_dir, uttids, every_N, num_frames,
                 frame_height, frame_width):
        self.uttid2videopath = get_existing_paths(
            videos_dir, uttids, ['.avi', '.mp4'])
        self.uttid2facepath = get_existing_paths(faces_dir, uttids, ['.pkl'])

        uttids = find_common_uttids(self.uttid2videopath, self.uttid2facepath)

        self.uttid2videopath = {
            uttid: self.uttid2videopath[uttid] for uttid in uttids}
        self.uttid2facepath = {
            uttid: self.uttid2facepath[uttid] for uttid in uttids}

        assert len(self.uttid2videopath) == len(self.uttid2facepath)

        self.videopaths = None
        self.facepaths = None

        self.every_N = every_N
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width

    def __getitem__(self, index):
        with open(self.facepaths[index], 'rb') as stream:
            face = pickle.load(stream)

        # TODO: use face in the video.
        video = loadvideo(path=self.videopaths[index], every_N=self.every_N,
                          num_frames=self.num_frames,
                          frame_width=self.image_size,
                          frame_height=self.image_size,
                          face=face)
        return video

    def __len__(self):
        return len(self.videopaths)


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


class MELD(Label, VideoModality, TextModality, AudioModality):
    def __init__(self, videos_dir, faces_dir, audios_dir, texts_dir,
                 label_path, every_N, num_frames, image_size, datatype):
        self.emotions = ['anger',
                         'disgust',
                         'fear',
                         'joy',
                         'neutral',
                         'sadness',
                         'surprise']

        Label.__init__(self, label_path, self.emotions, datatype)
        VideoModality.__init__(self, videos_dir, faces_dir,
                               list(self.uttid2num.keys()), every_N,
                               num_frames, image_size, image_size)

        AudioModality.__init__(self, audios_dir, list(self.uttid2num.keys()))
        TextModality.__init__(self, texts_dir, list(self.uttid2num.keys()))

        print(f"finding common utterances in video, face, audio, and text ...")
        self.uttids = find_common_uttids(self.uttid2videopath,
                                         self.uttid2audiopath,
                                         self.uttid2textpath)

        self.labels = [self.uttid2num[uttid] for uttid in self.uttids]
        self.videopaths = [self.uttid2videopath[uttid]
                           for uttid in self.uttids]
        self.facepaths = [self.uttid2facepath[uttid] for uttid in self.uttids]
        self.audiopaths = [self.uttid2audiopath[uttid]
                           for uttid in self.uttids]
        self.textpaths = [self.uttid2textpath[uttid] for uttid in self.uttids]

        print(f"In total there are {len(self.labels)} utterances in common, "
              f"across the three modalities")

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):
        audio = AudioModality.__getitem__(self, index)
        text = TextModality.__getitem__(self, index)
        video = VideoModality.__getitem__(self, index)
        label = Label.__getitem__(self, index)

        return video, label

    def __len__(self):
        return len(self.labels)


class IEMOCAP(Label, VideoModality, TextModality, AudioModality):
    def __init__(self, videos_dir, faces_dir, audios_dir, texts_dir,
                 label_path, every_N, num_frames, image_size, datatype):
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
        Label.__init__(self, label_path, self.emotions, datatype)
        VideoModality.__init__(self, videos_dir, faces_dir,
                               list(self.uttid2num.keys()), every_N,
                               num_frames, image_size, image_size)

        AudioModality.__init__(self, audios_dir, list(self.uttid2num.keys()))
        TextModality.__init__(self, texts_dir, list(self.uttid2num.keys()))

        print(f"finding common utterances in video, face, audio, and text ...")
        self.uttids = find_common_uttids(self.uttid2videopath,
                                         self.uttid2audiopath,
                                         self.uttid2textpath)

        self.labels = [self.uttid2num[uttid] for uttid in self.uttids]
        self.videopaths = [self.uttid2videopath[uttid]
                           for uttid in self.uttids]
        self.facepaths = [self.uttid2facepath[uttid] for uttid in self.uttids]
        self.audiopaths = [self.uttid2audiopath[uttid]
                           for uttid in self.uttids]
        self.textpaths = [self.uttid2textpath[uttid] for uttid in self.uttids]

        print(f"In total there are {len(self.labels)} utterances in common, "
              f"across the three modalities")

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):
        audio = AudioModality.__getitem__(self, index)
        text = TextModality.__getitem__(self, index)
        video = VideoModality.__getitem__(self, index)
        label = Label.__getitem__(self, index)

        return video, label

    def __len__(self):
        return len(self.labels)


class AFEW(Label, VideoModality, AudioModality):
    def __init__(self, videos_dir, faces_dir, audios_dir, label_path, every_N,
                 num_frames, image_size, datatype):
        self.emotions = ['angry',
                         'disgust',
                         'fear',
                         'happy',
                         'neutral',
                         'sad',
                         'surprise']

        Label.__init__(self, label_path, self.emotions, datatype)
        VideoModality.__init__(self, videos_dir, faces_dir,
                               list(self.uttid2num.keys()), every_N,
                               num_frames, image_size, image_size)

        AudioModality.__init__(self, audios_dir, list(self.uttid2num.keys()))

        print(f"finding common utterances in video, face, and audio ...")
        self.uttids = find_common_uttids(self.uttid2videopath,
                                         self.uttid2audiopath)

        self.labels = [self.uttid2num[uttid] for uttid in self.uttids]
        self.videopaths = [self.uttid2videopath[uttid]
                           for uttid in self.uttids]
        self.facepaths = [self.uttid2facepath[uttid] for uttid in self.uttids]
        self.audiopaths = [self.uttid2audiopath[uttid]
                           for uttid in self.uttids]

        print(f"In total there are {len(self.labels)} utterances in common, "
              f"across the two modalities")

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):
        audio = AudioModality.__getitem__(self, index)
        video = VideoModality.__getitem__(self, index)
        label = Label.__getitem__(self, index)

        return video, label

    def __len__(self):
        return len(self.labels)


class CAER(Label, VideoModality, AudioModality):
    def __init__(self, videos_dir, faces_dir, audios_dir, label_path, every_N,
                 num_frames, image_size, datatype):
        self.emotions = ['anger',
                         'disgust',
                         'fear',
                         'happy',
                         'neutral',
                         'sad',
                         'surprise']

        Label.__init__(self, label_path, self.emotions, datatype)
        VideoModality.__init__(self, videos_dir, faces_dir,
                               list(self.uttid2num.keys()), every_N,
                               num_frames, image_size, image_size)

        AudioModality.__init__(self, audios_dir, list(self.uttid2num.keys()))

        print(f"finding common utterances in video, face, and audio ...")
        self.uttids = find_common_uttids(self.uttid2videopath,
                                         self.uttid2audiopath)

        self.labels = [self.uttid2num[uttid] for uttid in self.uttids]
        self.videopaths = [self.uttid2videopath[uttid]
                           for uttid in self.uttids]
        self.facepaths = [self.uttid2facepath[uttid] for uttid in self.uttids]
        self.audiopaths = [self.uttid2audiopath[uttid]
                           for uttid in self.uttids]

        print(f"In total there are {len(self.labels)} utterances in common, "
              f"across the two modalities")

        self.every_N = every_N
        self.num_frames = num_frames
        self.image_size = image_size

    def __getitem__(self, index):
        audio = AudioModality.__getitem__(self, index)
        video = VideoModality.__getitem__(self, index)
        label = Label.__getitem__(self, index)

        return video, label

    def __len__(self):
        return len(self.labels)
