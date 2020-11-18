import yaml
import numpy as np
import tensorflow as tf
from absl import logging
from sklearn.metrics.pairwise import euclidean_distances


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, steps + 1


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def calc_angle_distance(emb1, emb2):
    """Calculate the angle (radian) distance between the embeddings."""
    return np.arccos(np.clip((emb1 @ emb2.T), -1, 1))


def calc_euclidean_distance(emb1, emb2):
    """Calculate the euclidean distance between the embeddings."""
    if len(emb1.shape) == len(emb2.shape) == 2:
        return euclidean_distances(emb1, emb2)
    elif len(emb1.shape) == len(emb2.shape) == 1:
        return np.linalg.norm((emb1 - emb2), axis=0)
    else:
        return np.linalg.norm((emb1 - emb2), axis=1)
