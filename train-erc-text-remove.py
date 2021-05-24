import argparse
import logging
from glob import glob
import json
import yaml
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def read_json(path):
    with open(path, 'r') as stream:
        foo = json.load(stream)
    return foo


def read_yaml(path):
    with open(path, 'r') as stream:
        foo = yaml.load(stream)
    return foo


def find_best_seed(OUTPUT_DIR, metric='test_f1_weighted', higher_is_better=True):
    logging.info(f"finding the best seed ...")
    paths = glob(os.path.join(OUTPUT_DIR, '*/test-results.json'))
    paths = {path.split('/')[-2]: read_json(path) for path in paths}
    paths = {key: val[metric] for key, val in paths.items()}

    if higher_is_better:
        seed = max(paths, key=paths.get)
    else:
        seed = min(paths, key=paths.get)

    score = paths[seed]

    logging.info(f"best seed is {seed} whose {metric} scored {score}")
    return seed


def remove_non_best_seeds(OUTPUT_DIR, seed):
    logging.info(f"removing optimizer.pt and pytorch_model.bin that are not good enough ...")
    to_remove = glob(os.path.join(OUTPUT_DIR, "*/*/optimizer.pt")) + \
        glob(os.path.join(OUTPUT_DIR, "*/*/pytorch_model.bin"))

    for path in to_remove:
        if path.split('/')[-3] != seed:
            logging.info(f"removing {path}")
            os.remove(path)

def find_best_checkpoint(OUTPUT_DIR, seed):
    logging.info(f"finding the best checkpoint in seed {seed} ...")
    paths = glob(os.path.join(OUTPUT_DIR, f"{seed}/*/trainer_state.json"))
    paths = [(path, os.path.getmtime(path)) for path in paths]
    latest_path = sorted(paths, key=lambda x:-x[1])[0][0]
    best_checkpoint_path = read_json(latest_path)['best_model_checkpoint']

    logging.info(f"best checkpoint found at {best_checkpoint_path}")

    return best_checkpoint_path

def remove_non_best_checkpoints(OUTPUT_DIR, best_checkpoint_path):
    logging.info(f"removing optimizer.pt and pytorch_model.bin that are not good enough ...")
    to_remove = glob(os.path.join(OUTPUT_DIR, "*/*/optimizer.pt")) + \
        glob(os.path.join(OUTPUT_DIR, "*/*/pytorch_model.bin"))

    for path in to_remove:
        if best_checkpoint_path not in path:
            logging.info(f"removing {path}")
            os.remove(path)

def main(OUTPUT_DIR):
    seed = find_best_seed(OUTPUT_DIR)
    remove_non_best_seeds(OUTPUT_DIR, seed)
    best_checkpoint_path = find_best_checkpoint(OUTPUT_DIR, seed)
    remove_non_best_checkpoints(OUTPUT_DIR, best_checkpoint_path)

    logging.info(f"DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove the models that are not the best.')
    parser.add_argument('--OUTPUT-DIR', type=str)

    args = parser.parse_args()
    args = vars(args)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
