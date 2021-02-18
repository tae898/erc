from insightface.utils import face_align
from joblib import Parallel, delayed
import shutil
from sklearn.cluster import AgglomerativeClustering
import random
import pickle
import argparse
import numpy as np
import av
from tqdm import tqdm
from glob import glob
import os
CURRENT_DIR = './'
DET_THRESHOLD = 0.90
COS_THRESHOLD = 0.80
IMAGE_SIZE = 112
DATABASES = ['MELD', 'IEMOCAP', 'CAER', 'AFEW']


def video2numpy(path):
    container = av.open(path)
    frames = {}
    fps = round(float(container.streams.video[0].average_rate))

    for idx, frame in enumerate(container.decode(video=0)):
        numpy_RGB = np.array(frame.to_image())
        frames[idx] = numpy_RGB
    container.close()

    return frames, fps


def get_facepath(vidpath):
    facepath = None
    if '.mp4' in vidpath:
        facepath = vidpath.replace(
            'raw-videos', 'faces').replace('.mp4', '.pkl')
    elif '.avi' in vidpath:
        facepath = vidpath.replace(
            'raw-videos', 'faces').replace('.avi', '.pkl')
    return facepath


def load_face(facepath):
    with open(facepath, 'rb') as stream:
        face_dict = pickle.load(stream)
    return face_dict


def align_and_crop_faces(frame, face):
    faces_aligned = []
    for fc in face:
        img_aligned = face_align.norm_crop(
            frame, landmark=fc['landmark'], image_size=IMAGE_SIZE)
        faces_aligned.append(img_aligned)
    return faces_aligned


def get_unique_faces(embeddings_all, faces_aligned_all, blank_frames=True):

    X = [emb for embs in embeddings_all.values() for emb in embs]

    if len(X) == 1:
        labels_clustered = np.array([0])

    elif len(X) == 0:
        return None

    else:
        ac = AgglomerativeClustering(n_clusters=None,
                                     affinity='cosine',
                                     linkage='average',
                                     distance_threshold=COS_THRESHOLD)

        clustering = ac.fit(X)
        labels_clustered = clustering.labels_

    labels_unique = np.unique(labels_clustered)

    vids = {lbl: np.zeros((len(embeddings_all), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for lbl in labels_unique}

    count = 0
    for frame_idx, faces_aligned in faces_aligned_all.items():
        for face_aligned in faces_aligned:
            lbl = labels_clustered[count]
            vids[lbl][frame_idx] = face_aligned
            count += 1

    return vids


def save_face_videos(vids, vidpath, fps):

    dir_path = vidpath.replace(
        'raw-videos', 'face-videos').split('.mp4')[0].split('.avi')[0]
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)

    for speaker_id in range(len(vids)):
        save_full_path = os.path.join(dir_path, os.path.basename(
            dir_path)) + f"_{speaker_id:03d}.mp4"

        container = av.open(save_full_path, mode='w')

        stream = container.add_stream('mpeg4', rate=fps)
        stream.width = IMAGE_SIZE
        stream.height = IMAGE_SIZE

        for face in vids[speaker_id]:
            frame = av.VideoFrame.from_ndarray(face, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()


def batch_paths(all_vids_paths, n_jobs):
    batch_size = len(all_vids_paths) // n_jobs
    batched = [
        all_vids_paths[i*batch_size: (i+1)*batch_size] for i in range(n_jobs)]

    batched[-1] += all_vids_paths[batch_size*n_jobs:]

    assert sorted(all_vids_paths) == sorted(
        [bar for foo in batched for bar in foo])

    return batched


def run(videopaths):

    for vidpath in tqdm(videopaths):
        try:
            frames, fps = video2numpy(vidpath)
            facepath = get_facepath(vidpath)
            faces = load_face(facepath)

        except Exception as e:
            print(e)
            continue

        assert len(frames) == len(faces)

        indexes = list(frames.keys())

        faces_aligned_all = {}
        embeddings_all = {}
        for idx in indexes:
            frame = frames[idx]
            face = faces[idx]
            face = [fc for fc in face if fc['det_score'] > DET_THRESHOLD]
            faces_aligned = align_and_crop_faces(frame, face)
            faces_aligned_all[idx] = faces_aligned
            embeddings_all[idx] = [fc['normed_embedding'] for fc in face]

        assert len(frames) == len(faces) == len(
            faces_aligned_all) == len(embeddings_all)

        vids = get_unique_faces(embeddings_all, faces_aligned_all)

        if vids is None:
            continue
        
        for speaker_id, vid in vids.items():
            assert len(vid) == len(frames)

        save_face_videos(vids, vidpath, fps)


parser = argparse.ArgumentParser(description='crop faces')
parser.add_argument('--num-jobs', default=1, help='number of jobs')
args = parser.parse_args()

n_jobs = int(args.num_jobs)

print(f"n_jobs: {n_jobs}")


for DATABASE in tqdm(DATABASES):
    for DATASET in tqdm(['train', 'val', 'test']):
        videopaths = glob(os.path.join(
            CURRENT_DIR, DATABASE, 'raw-videos', DATASET, '*'))

        random.shuffle(videopaths)

        batched = batch_paths(videopaths, n_jobs)

        Parallel(n_jobs=n_jobs)(delayed(run)(batch) for batch in tqdm(batched))
