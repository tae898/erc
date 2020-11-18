from .modules.models import ArcFaceModel
from .modules.utils import set_memory_growth, load_yaml, l2_norm
from .modules.utils import calc_angle_distance, calc_euclidean_distance

from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import numpy as np
import requests
import cv2
import zipfile
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


class ArcFace():
    def __init__(self, device='cpu'):
        logger = tf.get_logger()
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

        # cfg = load_yaml(os.path.join(HERE, './configs/arc_res50.yaml'))

        self.model = ArcFaceModel(size=112,
                                  backbone_type='ResNet50',
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
            os.path.join(HERE, './pretrained_models/') + 'arc_res50')

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
