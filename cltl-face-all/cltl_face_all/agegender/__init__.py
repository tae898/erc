from pathlib import Path
import cv2
import numpy as np
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
import os
from .factory import get_model


HERE = os.path.dirname(os.path.abspath(__file__))


class AgeGender():
    def __init__(self, device='cpu'):
        pretrained_model = "https://github.com/leolani/cltl-face-all/releases/download/v0.0/EfficientNetB3_224_weights.11-3.44.hdf5"
        modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5",
                               pretrained_model,
                               cache_subdir="pretrained_models",
                               file_hash=modhash,
                               cache_dir=str(Path(__file__).resolve().parent))
        model_name, img_size = Path(weight_file).stem.split("_")[:2]
        self.img_size = int(img_size)
        cfg = OmegaConf.from_dotlist(
            [f"model.model_name={model_name}", f"model.img_size={img_size}"])
        self.model = get_model(cfg)
        self.model.load_weights(weight_file)

    def predict(self, faces):
        """Predict age and gender.

        Inference can be batched.

        Parameters
        ----------
        faces: a numpy array whose shape is (B, H, W, C). Mind the batch axis.
        The dtype should be unit8. faces should be cropped and aligned with 
        the shape of 112 by 112 and RGB, unit8

        Returns
        -------
        predicted_ages: np.ndarray, float32, shape=(B,)
            Expectation values of the ages.
        predicted_genders: np.ndarray, float32, shape=(B,)
            Femaleness of the faces, where 0 is the most male and 1 is the most
            female.

        """
        if len(faces.shape) == 3:
            face = faces[np.newaxis, ...]

        assert faces.dtype == np.dtype('uint8'), "dtype should be unit8!"

        assert (faces.shape[1], faces.shape[2],
                faces.shape[3]) == (112, 112, 3), "Faces should be cropped "\
            "and aligned with the shape of 112 by 112 and RGB"

        faces = [cv2.resize(face, (self.img_size, self.img_size))
                 for face in faces]
        faces = [cv2.cvtColor(face, cv2.COLOR_RGB2BGR) for face in faces]
        faces = np.stack(faces)

        results = self.model.predict(faces)
        predicted_genders = np.array([res[0] for res in results[0]])
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        return predicted_ages, predicted_genders
