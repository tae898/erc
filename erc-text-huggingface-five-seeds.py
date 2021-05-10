import logging
import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main(DATASET, BATCH_SIZE, model_checkpoint, speaker_mode, num_past_utterances, num_future_utterances, **kwargs):

    logging.info(f"automatic hyperparameter tuning with speaker_mode: {speaker_mode}, "
                 f"num_past_utterances: {num_past_utterances}, "
                 f"num_future_utterances: {num_future_utterances}")
    CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    SEED = 42
    OUTPUT_DIR = f"huggingface-results/{DATASET}/{model_checkpoint}/SEEDS/{CURRENT_TIME}-speaker_mode-{speaker_mode}-num_past_utterances-{num_past_utterances}-num_future_utterances-{num_future_utterances}-batch_size-{BATCH_SIZE}-seed-{SEED}"

    import subprocess

    subprocess.call(
        ["python3", "erc-text-huggingface-five-seeds-hp.py", "--OUTPUT-DIR", OUTPUT_DIR])

    for SEED in [0, 1, 2, 3, 4]:
        subprocess.call(["python3", "erc-text-huggingface-five-seeds-full.py",
                        "--OUTPUT-DIR", OUTPUT_DIR, "--SEED", str(SEED)])


if __name__ == "__main__":
    with open('./erc-text-huggingface-five-seeds.yaml', 'r') as stream:
        args = yaml.load(stream)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)


# max batch size for tesla v100

# <roberta-base>

# MELD: 16
# IEMOCAP: 16
# EmoryNLP: 16
# DailyDialog: 16

# <roberta-large>

# MELD: 4
# IEMOCAP: 4
# EmoryNLP: 4
# DailyDialog: 4
