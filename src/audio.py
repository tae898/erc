import librosa
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def load_audio(DATASET, SPLIT, uttid, sr=22050):
    path = f"multimodal-datasets/{DATASET}/raw-audios/{SPLIT}/{uttid}.wav"
    y, sr = librosa.load(path, sr)

    return y


def audio2spectrogram(y, sr=22050, n_fft=2048*8, hop_length=512*8, win_length=None,
                      power=2.0, n_mels=128*8):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length,
                                          win_length=win_length,
                                          power=power,
                                          n_mels=n_mels)

    return mels
