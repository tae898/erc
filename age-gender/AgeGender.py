from pathlib import Path
import cv2
import numpy as np
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from factory import get_model
from mlsocket import MLSocket
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class AgeGender():
    def __init__(self):
        data_path = os.path.join(HERE, 'pretrained_models')
        tflite_path = os.path.join(data_path, "model.tflite")
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        if not os.path.exists(tflite_path):
            self._download_model_file(tflite_path)


pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'


weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                       file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist(
    [f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights(weight_file)


with MLSocket() as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"waiting for the client ...")
    conn, address = s.accept()
    print(f"connection estbalished")

    with conn:
        while True:
            try:
                face = conn.recv(1024)
                face = cv2.resize(face, (img_size, img_size))

                if len(face.shape) == 3:
                    face = np.expand_dims(face, axis=0)

                results = model.predict(face)
                gender, age = results

                # gender is how "female" it is.
                gender = np.array([gender[0][0]])

                # get the expectation of the age
                age = age.dot(np.arange(0, 101).reshape(101, 1)).flatten()
                age = np.array([round(age[0])]).astype(int)

                conn.send(age)
                conn.send(gender) 

            except Exception as e:
                print(e)
                conn.send(np.array([]))
                conn.send(np.array([]))
                pass

    print("disconnected")
