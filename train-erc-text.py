"""Main training script"""
import logging
import datetime
import yaml
import subprocess
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main(DATASET: str, BATCH_SIZE: int, model_checkpoint: str, speaker_mode: str,
         num_past_utterances: int, num_future_utterances: int, SEEDS: list, **kwargs):
    """Call `train-erc-text-hp.py and `train-erc-text-full.py`"""

    logging.info(f"automatic hyperparameter tuning with speaker_mode: {speaker_mode}, "
                 f"num_past_utterances: {num_past_utterances}, "
                 f"num_future_utterances: {num_future_utterances}")
    CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    SEED = 42   # This seed is only for the hyperparameter tuning.
    OUTPUT_DIR = f"results/{DATASET}/{model_checkpoint}/SEEDS/{CURRENT_TIME}-"\
        f"speaker_mode-{speaker_mode}-num_past_utterances-{num_past_utterances}-"\
        f"num_future_utterances-{num_future_utterances}-batch_size-{BATCH_SIZE}-seed-{SEED}"

    subprocess.call(
        ["python3", "train-erc-text-hp.py", "--OUTPUT-DIR", OUTPUT_DIR])

    for SEED in tqdm(SEEDS):
        subprocess.call(["python3", "train-erc-text-full.py",
                         "--OUTPUT-DIR", OUTPUT_DIR, "--SEED", str(SEED)])


if __name__ == "__main__":
    with open('./train-erc-text.yaml', 'r') as stream:
        args = yaml.load(stream)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
