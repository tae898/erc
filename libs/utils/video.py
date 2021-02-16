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
