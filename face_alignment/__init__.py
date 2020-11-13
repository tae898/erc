# -*- coding: utf-8 -*-

__author__ = """Adrian Bulat"""
__email__ = 'adrian.bulat@nottingham.ac.uk'
__version__ = '1.1.1'

import torch
import numpy as np

from .api import FaceAlignment, LandmarksType, NetworkSize
from.align_trans import get_reference_facial_points, warp_and_crop_face


class FaceDetection():
    def __init__(self, device='cpu', face_detector='sfd', landmarks_type='2D'):
        """Instantiate a FaceDetection object.

        Parameters
        ----------

        device: str
            'cpu', 'cuda', etc.

        face_detector: str
            Currently either 'sfd' or 'blazeface' supported. 'sfd'is slower
            than blazeface but more accurate.

        landmarks_type: str
            Either '2D' or '3D'

        """
        self.device = device
        if landmarks_type == '2D':
            landmarks_type = LandmarksType._2D
        else:
            landmarks_type = LandmarksType._3D

        self.model = FaceAlignment(landmarks_type=LandmarksType._2D,
                                   device=self.device,
                                   face_detector=face_detector)

    def detect_faces(self, images):
        """Detect the faces and their probs in a given image.

        Inference can be batched.

        Parameters
        ----------
        images: a numpy array whose shape is (B, H, W, C). Mind the batch axis.
        The dtype should be unit8. RGB channel expected.

        Returns
        -------
        bboxes: a list of np.ndarrays
            A list of list of np.ndarrays, where each np.ndarray has the shape
            of (num_faces, 5), where the last axis is [x1, y1, x2, y2, prob]

        """
        if len(images.shape) == 3:
            face = images[np.newaxis, ...]

        assert images.dtype == np.dtype('uint8'), "dtype should be unit8!"

        images = images.transpose(0, 3, 1, 2)
        images = torch.Tensor(images)
        images = images.to(self.device)
        bboxes = self.model.face_detector.detect_from_batch(images)

        return bboxes

    def detect_landmarks(self, images, bboxes):
        """Detect landmarks from the images, given the bounding boxes.

        Parameters
        ----------
        images: a numpy array whose shape is (B, H, W, C). Mind the batch axis.
        The dtype should be unit8. RGB channel expected.

        bboxes: a list of np.ndarrays
            A list of list of np.ndarrays, where each np.ndarray has the shape
            of (num_faces, 5), where the last axis is [x1, y1, x2, y2, prob]

        Returns
        -------
        landmarks: a list of np.ndarrays
            Each np.ndarray has shape of (num_faces, 68, 2)

        """
        if len(images.shape) == 3:
            face = images[np.newaxis, ...]

        assert images.dtype == np.dtype('uint8'), "dtype should be unit8!"

        images = images.transpose(0, 3, 1, 2)
        images = torch.Tensor(images)
        images = images.to(self.device)

        landmarks = self.model.get_landmarks_from_batch(images, bboxes)

        return landmarks

    def crop_and_align(self, images, bboxes, landmarks, crop_size=112):
        """Crop and align the faces.

        Parameters
        ----------
        images: a numpy array whose shape is (B, H, W, C). Mind the batch axis.
            The dtype should be unit8. RGB channel expected.

        bboxes: a list of np.ndarrays
            A list of list of np.ndarrays, where each np.ndarray has the shape
            of (num_faces, 5), where the last axis is [x1, y1, x2, y2, prob]

        landmarks: a list of np.ndarrays
            Each np.ndarray has shape of (num_faces, 68, 2)


        Returns
        -------
        faces: a list of np.ndarrays
            Each np.ndarry has the shape of (B, num_faces, H, W, C), where
            H and W are crop_size and C is the number of channels (RGB). 

        """
        if len(images.shape) == 3:
            face = images[np.newaxis, ...]

        assert images.dtype == np.dtype('uint8'), "dtype should be unit8!"

        assert len(images) == len(bboxes) == len(landmarks)

        faces = []

        for img, bbox, landmark in zip(images, bboxes, landmarks):
            assert len(bbox) == len(landmark)

            if len(bbox) == len(landmark) == 0:
                faces_ = np.array([]).reshape(
                    0, crop_size, crop_size, 3).astype(np.uint8)

            else:
                faces_ = []

                for bbox_, landmark_ in zip(bbox, landmark):
                    face_warped = _crop_and_align_per_box(
                        img, bbox_, landmark_, crop_size)
                    faces_.append(face_warped)
                faces_ = np.stack(faces_, axis=0)

            faces.append(faces_)

        return faces


def _crop_and_align_per_box(img, bbox, landmark, crop_size=112):
    x1, y1, x2, y2, prob = bbox.astype(int).tolist()

    # cropped = img[y1:y2, x1:x2, :]
    # lm_translated = landmark - np.array([x1, y1])
    # eye_left_shifted = [lm_translated[36:42, 0].mean(), lm_translated[36:42, 1].mean()]
    # eye_right_shifted = [lm_translated[42:48, 0].mean(), lm_translated[42:48, 1].mean()]
    # nose_shifted = [lm_translated[30, 0], lm_translated[30, 1]]
    # mouth_left_shifted = [lm_translated[48, 0], lm_translated[48, 1]]
    # mouth_right_shifted = [lm_translated[54, 0], lm_translated[54, 1]]
    eye_left = [landmark[36:42, 0].mean(), landmark[36:42, 1].mean()]
    eye_right = [landmark[42:48, 0].mean(), landmark[42:48, 1].mean()]
    nose = [landmark[30, 0], landmark[30, 1]]
    mouth_left = [landmark[48, 0], landmark[48, 1]]
    mouth_right = [landmark[54, 0], landmark[54, 1]]

    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    facial5points = [eye_left, eye_right, nose, mouth_left, mouth_right]
    warped_face = warp_and_crop_face(
        img, facial5points, reference, crop_size=(crop_size, crop_size))

    return warped_face
    