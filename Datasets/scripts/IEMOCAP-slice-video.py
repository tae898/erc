import librosa
from glob import glob
import os
from tqdm import tqdm
import av
import numpy as np
import math
import json

import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


with open('IEMOCAP/utterance-ordered.json', 'r') as stream:
    diautt_ordered = json.load(stream)

dias = {DATASET: sorted(os.listdir(f'IEMOCAP/raw-videos/{DATASET}/'))
        for DATASET in ['train', 'val', 'test']}

dias = {DATASET: [foo.split('.avi')[0] for foo in dias[DATASET]]
        for DATASET in ['train', 'val', 'test']}

# audios = {DATASET: {dia: sorted(glob(f'IEMOCAP/raw-audios/{DATASET}/{dia}/*.wav')) for dia in dias[DATASET]}
#           for DATASET in ['train', 'val', 'test']}

SAMPLING_RATE = 2**12

# This magic value was emperically chosen.
OFFSET_FRAMES = -5


def video2numpy(path):
    container = av.open(path)
    frames = []
    for idx, frame in enumerate(container.decode(video=0)):
        numpy_RGB = np.array(frame.to_image())
        frames.append(numpy_RGB)
    return frames


def find_start_end(audio_dia, audio_utt, start_from):
    mses = []
    for start in range(start_from, len(audio_dia) - len(audio_utt)):
        candidate = audio_dia[start:start+len(audio_utt)]
        mse = ((audio_utt - candidate)**2).sum()
        mses.append((start, mse))

    start = sorted(mses, key=lambda x: x[1])[0][0]
    end = start + len(audio_utt)

    return start, end


def get_start_end_sec(audio_utt_path):
    text_utt_path = audio_utt_path.replace(
        'raw-audios', 'raw-texts').replace('.wav', '.json')
    text_utt_path = text_utt_path

    with open(text_utt_path, 'r') as stream:
        text_utt = json.load(stream)

    start = text_utt['StartTime']
    end = text_utt['EndTime']

    return start, end


def write_video(video_utt, savepath, fps):
    container = av.open(savepath, mode='w')
    stream = container.add_stream('mpeg4', rate=round(fps))
    stream.width = video_utt[0].shape[1]
    stream.height = video_utt[0].shape[0]

    for frame in video_utt:
        frame_ = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame_):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


def process_dia(DATASET, dia):
    video_dia = video2numpy(f"IEMOCAP/raw-videos/{DATASET}/{dia}.avi")
    audio_dia = librosa.core.load(
        f"IEMOCAP/raw-videos/{DATASET}/{dia}.avi", sr=SAMPLING_RATE)[0]

    # start_from = 0
    for utt in diautt_ordered[DATASET][dia]:
        uttwav = os.path.join(f"IEMOCAP/raw-audios/{DATASET}/{dia}/{utt}.wav")
        audio_utt = librosa.core.load(uttwav, sr=SAMPLING_RATE)[0]
        # start, end = find_start_end(audio_dia, audio_utt, start_from)
        # start_frame = math.floor(len(video_dia) * start / len(audio_dia))
        # end_frame = math.ceil(len(video_dia) * end / len(audio_dia))

        start, end = get_start_end_sec(uttwav)
        start_frame = math.floor(
            len(video_dia) * start / (len(audio_dia) / SAMPLING_RATE))
        end_frame = math.ceil(len(video_dia) * end /
                              (len(audio_dia) / SAMPLING_RATE))

        start_frame += OFFSET_FRAMES
        end_frame += OFFSET_FRAMES

        # number of frames divided by the duration in seconds
        fps = (end_frame - start_frame) / (len(audio_utt) / SAMPLING_RATE)
        video_utt = video_dia[start_frame:end_frame]

        os.makedirs(
            f"IEMOCAP/raw-videos/{DATASET}/{dia}", exist_ok=True)

        uttmp4 = os.path.basename(uttwav).replace('.wav', '.mp4')
        save_path = f"IEMOCAP/raw-videos/{DATASET}/{dia}/{uttmp4}"

        write_video(video_utt, save_path, fps)

        # if start >= start_from:
        #     start_from = start


for DATASET in tqdm(['train', 'val', 'test']):
    for dia in tqdm(dias[DATASET]):
        process_dia(DATASET, dia)
    # Parallel(n_jobs=2)(delayed(process_dia)(DATASET, dia)
    #                    for dia in tqdm(dias[DATASET]))
