import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from utils import ErcTextDataset, get_num_classes
import os
import argparse
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main(WEIGHT_DECAY, WARMUP_RATIO, NUM_TRAIN_EPOCHS, HP_ONLY_UPTO, OUTPUT_DIR, DATASET,
         BATCH_SIZE, model_checkpoint, speaker_mode, num_past_utterances, num_future_utterances,
         HP_N_TRIALS, **kwargs):

    logging.info(f"automatic hyperparameter tuning with speaker_mode: {speaker_mode}, "
                 f"num_past_utterances: {num_past_utterances}, "
                 f"num_future_utterances: {num_future_utterances}")
    EVALUATION_STRATEGY = 'epoch'
    LOGGING_STRATEGY = 'epoch'
    SAVE_STRATEGY = 'no'

    ROOT_DIR = './multimodal-datasets/'

    PER_DEVICE_TRAIN_BATCH_SIZE = BATCH_SIZE
    PER_DEVICE_EVAL_BATCH_SIZE = BATCH_SIZE*2
    LOAD_BEST_MODEL_AT_END = False
    SEED = 42
    FP16 = True

    NUM_CLASSES = get_num_classes(DATASET)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy=EVALUATION_STRATEGY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        logging_strategy=LOGGING_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        seed=SEED,
        fp16=FP16,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_CLASSES)

    ds_train = ErcTextDataset(DATASET=DATASET, SPLIT='train', speaker_mode=speaker_mode,
                              num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                              model_checkpoint=model_checkpoint, ONLY_UPTO=HP_ONLY_UPTO,
                              ROOT_DIR=ROOT_DIR, SEED=SEED)

    ds_val = ErcTextDataset(DATASET=DATASET, SPLIT='val', speaker_mode=speaker_mode,
                            num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                            model_checkpoint=model_checkpoint, ONLY_UPTO=HP_ONLY_UPTO,
                            ROOT_DIR=ROOT_DIR, SEED=SEED)


    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
    )

    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        }

    best_run = trainer.hyperparameter_search(
        direction="minimize", hp_space=my_hp_space, n_trials=HP_N_TRIALS)

    logging.info(f"best hyper parameters found at {best_run}")

    with open(os.path.join(OUTPUT_DIR, 'hp.json'), 'w') as stream:
        json.dump(best_run.hyperparameters, stream, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    parser.add_argument('--OUTPUT-DIR', type=str)

    args = parser.parse_args()
    args = vars(args)

    with open('./train-erc-text.yaml', 'r') as stream:
        args_ = yaml.load(stream)

    for key, val in args_.items():
        args[key] = val

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
