# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tqdm import tqdm
from glob import glob
import os
import random
import av
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import multiprocessing
from joblib import Parallel, delayed
from insightface.app.face_analysis import FaceAnalysis
# https://github.com/deepinsight/insightface
# THIS REPO IS AMAZING!!
# Face = collections.namedtuple('Face', [
#     'bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age',
#     'embedding_norm', 'normed_embedding'
# ])


def video2numpy(path):
    container = av.open(path)
    frames = {}
    for idx, frame in enumerate(container.decode(video=0)):
        numpy_RGB = np.array(frame.to_image())
        frames[idx] = numpy_RGB
    container.close()

    return frames


def create_dirs(all_vids_paths):
    for bar in set([os.path.dirname(foo) for foo in all_vids_paths]):
        os.makedirs(bar.replace('raw-videos', 'faces'), exist_ok=True)


def get_paths():
    all_vids_paths = glob(f'./*/raw-videos/*/*')
    create_dirs(all_vids_paths)
    random.shuffle(all_vids_paths)

    return all_vids_paths


def batch_paths(all_vids_paths, n_jobs):
    batch_size = len(all_vids_paths) // n_jobs
    batched = [
        all_vids_paths[i*batch_size: (i+1)*batch_size] for i in range(n_jobs)]

    batched[-1] += all_vids_paths[batch_size*n_jobs:]

    assert sorted(all_vids_paths) == sorted(
        [bar for foo in batched for bar in foo])

    return batched


def process_paths(all_vids_path, gpu_id):

    fa = FaceAnalysis(det_name='retinaface_r50_v1',
                      rec_name='arcface_r100_v1',
                      ga_name='genderage_v1')

    fa.prepare(ctx_id=gpu_id)
    for videopath in tqdm(all_vids_path):
        try:
            if '.mp4' in videopath:
                savepath = videopath.replace(
                    'raw-videos', 'faces').replace('.mp4', '.pkl')
            elif '.avi' in videopath:
                savepath = videopath.replace(
                    'raw-videos', 'faces').replace('.avi', '.pkl')
            else:
                raise FileNotFoundError(f"{videopath} not a legit video")

            if os.path.isfile(savepath) and os.path.getsize(savepath) > 256:
                continue

                frames = video2numpy(videopath)

            detections = {}
            for idx, frame in frames.items():
                results = fa.get(frame)
                detections[idx] = results

            pickle.dump(detections, open(savepath, 'wb'))
        except Exception as e:
            print(f"{e}, {videopath}")
            pass


# %%
parser = argparse.ArgumentParser(description='extract faces')
# general
parser.add_argument('--num-jobs', default=1, help='number of jobs')
parser.add_argument('--gpu-id', default=-1, type=int,
                    help='gpu id. -1 means CPU')
args = parser.parse_args()

n_jobs = int(args.num_jobs)
gpu_id = int(args.gpu_id)

print(f"n_jobs: {n_jobs}, gpu_id: {gpu_id}")


all_vids_paths = get_paths()
print(all_vids_paths)
batched = batch_paths(all_vids_paths, n_jobs)

Parallel(n_jobs=n_jobs)(delayed(process_paths)(batch, gpu_id)
                        for batch in tqdm(batched))
