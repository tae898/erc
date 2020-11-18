from tqdm import tqdm
import random
import json
import numpy as np
from glob import glob
import os
import cv2
from cltl_face_all.face_alignment import FaceDetection
from cltl_face_all.arcface import ArcFace
from cltl_face_all.arcface import calc_angle_distance
from cltl_face_all.agegender import AgeGender


HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = os.getcwd()

ag = AgeGender(device='cpu')
af = ArcFace(device='cpu')
fd = FaceDetection(device='cpu', face_detector='dlib')

with open('/home/tk/datasets/MELD/MELD.Raw/train/small_dataset.json', 'r') as stream:
    small_dataset = json.load(stream)

vid_root_path = '/home/tk/datasets/MELD/MELD.Raw/train/train_splits/'

random_dia = random.choice(small_dataset['train'])
print(f"random dialogue : {random_dia}")
print()

utts = glob(vid_root_path + random_dia + '*.mp4')
utts.sort()
print(f"number of utterance videos is {len(utts)}")
print(utts)
print()

utt = random.choice(utts)
print(f"random utt: {utt}")
print()

cap = cv2.VideoCapture(utt)
frames = []
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

print(f"number of frames in this utt is {len(frames)}")



batch_size = 64
num_iter = len(frames) // batch_size
remainders = len(frames) % batch_size
landmarks = []
bboxes = []
faces = []

for i in tqdm(range(num_iter)):
    batch = np.stack(frames[i*batch_size: (i+1)*batch_size])
    print(batch.shape)

    bboxes_ = fd.detect_faces(batch)
    landmarks_ = fd.detect_landmarks(batch, bboxes_)
    faces_ = fd.crop_and_align(batch, bboxes_, landmarks_)
    for bb, lm, fa in zip(bboxes_, landmarks_, faces_):
        landmarks.append(lm)
        bboxes.append(bb)
        faces.append(fa)

if remainders != 0:
    batch = np.stack(frames[num_iter*batch_size:])
    print(batch.shape)

    bboxes_ = fd.detect_faces(batch)
    landmarks_ = fd.detect_landmarks(batch, bboxes_)
    faces_ = fd.crop_and_align(batch, bboxes_, landmarks_)

    for bb, lm, fa in zip(bboxes_, landmarks_, faces_):
        landmarks.append(lm)
        bboxes.append(bb)
        faces.append(fa)

num_real_bboxes = [len(bb) for bb in bboxes]
num_real_landmarks = [len(lm) for lm in landmarks]
num_real_faces = [len(fc) for fc in faces]


assert num_real_bboxes == num_real_landmarks == num_real_faces

num_real = num_real_bboxes

bboxes = np.concatenate(bboxes)
landmarks = np.concatenate(landmarks)
faces = np.concatenate(faces)

print(bboxes.shape, landmarks.shape, faces.shape)

age, gender = ag.predict(faces)
print(age, gender)


embeddings = af.predict(faces)
emb_matrix = calc_angle_distance(embeddings, embeddings)
print(emb_matrix)