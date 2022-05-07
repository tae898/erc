"""Main training script"""
import datetime
import logging
import subprocess

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    DATASET: str,
    BATCH_SIZE: int,
    roberta: str,
    speaker_mode: str,
    num_past_utterances: int,
    num_future_utterances: int,
    SEEDS: list,
    **kwargs,
):
    """Call `train-erc-text-hp.py and `train-erc-text-full.py`

    Args
    ----
    DATASET: MELD, IEMOCAP, or MELD_IEMOCAP
    BATCH_SIZE: number of data samples per batch
    roberta: either `base` or `large`
    speaker_mode: upper, title, or None
    num_past_utterances: number of past utterances to consider.
    num_future_utterances: number of future utterances to consider.
    SEEDS: list of random seeds.

    """
    logging.info(
        f"automatic hyperparameter tuning with speaker_mode: {speaker_mode}, "
        f"num_past_utterances: {num_past_utterances}, "
        f"num_future_utterances: {num_future_utterances}"
    )
    CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    SEED = 42  # This seed is only for the hyperparameter tuning.

    OUTPUT_DIR = (
        f"results/{DATASET}/roberta-{roberta}/SEEDS/{CURRENT_TIME}-"
        f"speaker_mode-{speaker_mode}-num_past_utterances-{num_past_utterances}-"
        f"num_future_utterances-{num_future_utterances}-batch_size-{BATCH_SIZE}-seed-{SEED}"
    )

    subprocess.call(
        [
            "python3",
            "train-erc-text-hp.py",
            "--OUTPUT-DIR",
            OUTPUT_DIR,
            "--SEED",
            str(SEED),
        ]
    )

    for SEED in tqdm(SEEDS):
        subprocess.call(
            [
                "python3",
                "train-erc-text-full.py",
                "--OUTPUT-DIR",
                OUTPUT_DIR,
                "--SEED",
                str(SEED),
            ]
        )


if __name__ == "__main__":
    with open("./train-erc-text.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
