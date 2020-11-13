from arcface.modules.models import ArcFaceModel
from arcface.modules.utils import set_memory_growth, load_yaml, l2_norm

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import numpy as np
import requests
import cv2
import zipfile
import shutil
from sklearn.metrics.pairwise import euclidean_distances

HERE = os.path.dirname(os.path.abspath(__file__))


class ArcFace():
    def __init__(self):
        logger = tf.get_logger()
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

        cfg = load_yaml(os.path.join(HERE, './configs/arc_res50.yaml'))

        self.model = ArcFaceModel(size=cfg['input_size'],
                                  backbone_type=cfg['backbone_type'],
                                  training=False)

        if not (os.path.isdir(os.path.join(HERE, './pretrained_models')) and
                os.path.isdir(os.path.join(HERE, './pretrained_models/arc_res50')) and
                os.path.isfile(os.path.join(HERE, './pretrained_models/arc_res50/checkpoint')) and
                os.path.isfile(os.path.join(HERE, './pretrained_models/arc_res50/e_8_b_40000.ckpt.data-00000-of-00002')) and
                os.path.isfile(os.path.join(HERE, './pretrained_models/arc_res50/e_8_b_40000.ckpt.data-00001-of-00002')) and
                os.path.isfile(os.path.join(HERE, './pretrained_models/arc_res50/e_8_b_40000.ckpt.index'))):

            toextract = "https://github.com/leolani/cltl-face-all/releases/download/v0.0/arc_res50.zip"

            print(f"Downloading the model from {toextract}...")
            shutil.rmtree(os.path.join(
                HERE, 'pretrained_models'), ignore_errors=True)
            os.makedirs(os.path.join(
                HERE, './pretrained_models/arc_res50'), exist_ok=True)

            r = requests.get(toextract)  # create HTTP response object
            with open(os.path.join(HERE, "arc_res50.zip"), 'wb') as f:
                f.write(r.content)

            with zipfile.ZipFile(os.path.join(HERE, "arc_res50.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(HERE, 'pretrained_models'))
            os.remove(os.path.join(HERE, 'arc_res50.zip'))

        ckpt_path = tf.train.latest_checkpoint(
            os.path.join(HERE, './pretrained_models/') + cfg['sub_name'])

        try:
            print("[*] load ckpt from {}".format(ckpt_path))
            self.model.load_weights(ckpt_path)
        except:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path))
            raise

    def predict(self, faces):
        """Predict face embeddings.

        Inference can be batched.

        Parameters
        ----------
        faces: a numpy array whose shape is (B, H, W, C). Mind the batch axis.
        The dtype should be unit8. faces should be cropped and aligned with
        the shape of 112 by 112 and RGB

        Returns
        -------
        embeddings: np.ndarray, float32, shape=(B, 512)

        """
        if len(faces.shape) == 3:
            face = faces[np.newaxis, ...]

        assert faces.dtype == np.dtype('uint8'), "dtype should be unit8!"

        assert (faces.shape[1], faces.shape[2],
                faces.shape[3]) == (112, 112, 3), "Faces should be cropped "
        "and aligned with the shape of 112 by 112 and RGB!"

        faces = faces.astype(np.float32) / 255.
        embeddings = self.model(faces)
        embeddings = l2_norm(embeddings)
        embeddings = embeddings.numpy()

        return embeddings

    def calc_angle_distance(self, emb1, emb2):
        return np.arccos(np.clip((emb1 @ emb2.T), -1, 1))

    def calc_euclidean_distance(self, emb1, emb2):

        if len(emb1.shape) == len(emb2.shape) == 2:
            return euclidean_distances(emb1, emb2)
        elif len(emb1.shape) == len(emb2.shape) == 1:
            return np.linalg.norm((emb1 - emb2), axis=0)
        else:
            return np.linalg.norm((emb1 - emb2), axis=1)
