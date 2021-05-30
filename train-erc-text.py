import logging
import datetime
import yaml
import subprocess
from utils import save_special_tokenzier
import os
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main(**kwargs):

    logging.info(f"automatic hyperparameter tuning , "
                 f"num_past_utterances: {kwargs['num_past_utterances']}, "
                 f"num_future_utterances: {kwargs['num_future_utterances']}")
    CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    OUTPUT_DIR = f"results/{kwargs['DATASET']}/{kwargs['model_checkpoint']}/{CURRENT_TIME}"

    os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'kwargs.yaml'), 'w') as stream:
        yaml.dump(kwargs, stream)

    save_special_tokenzier(DATASET=kwargs['DATASET'], ADD_BOU=kwargs['ADD_BOU'],
                           ADD_EOU=kwargs['ADD_EOU'],
                           ADD_SPEAKER_TOKENS=kwargs['ADD_SPEAKER_TOKENS'],
                           SPLITS=kwargs['SPEAKER_SPLITS'],
                           base_tokenizer=kwargs['model_checkpoint'],
                           save_at=os.path.join(OUTPUT_DIR, 'tokenizer'))

    subprocess.call(
        ["python3", "train-erc-text-hp.py", "--OUTPUT-DIR", OUTPUT_DIR,
         "--training-config", kwargs['training_config']])

    for SEED in kwargs['SEEDS']:
        subprocess.call(["python3", "train-erc-text-full.py",
                         "--OUTPUT-DIR", OUTPUT_DIR, "--SEED", str(SEED),
                         "--training-config", kwargs['training_config']])

    subprocess.call(
        ["python3", "train-erc-text-remove.py", "--OUTPUT-DIR", OUTPUT_DIR])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    parser.add_argument('--training-config', type=str)

    args = parser.parse_args()
    args = vars(args)

    with open(args['training_config'], 'r') as stream:
        args_ = yaml.load(stream)

    for key, val in args_.items():
        args[key] = val

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
