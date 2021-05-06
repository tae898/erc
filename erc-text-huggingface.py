import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import json
import os
import logging
from glob import glob
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import f1_score
import numpy as np
import math
from collections import Counter
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id(DATASET):

    emotions = {}
    # MELD has 7 classes
    emotions['MELD'] = ['neutral',
                        'joy',
                        'surprise',
                        'anger',
                        'sadness',
                        'disgust',
                        'fear']

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions['IEMOCAP'] = ['neutral',
                           'frustration',
                           'sadness',
                           'anger',
                           'excited',
                           'happiness']

    # EmoryNLP has 7 classes
    emotions['EmoryNLP'] = ['neutral',
                            'joyful',
                            'scared',
                            'mad',
                            'peaceful',
                            'powerful',
                            'sad']

    # DailyDialog originally has 7 classes, but be sure not to include the
    # neutral class, which accounts for 80% of the data, in calculating
    # the micro f1_score.
    emotions['DailyDialog'] = ['neutral',
                               'happiness',
                               'surprise',
                               'sadness',
                               'anger',
                               'disgust',
                               'fear']

    emotion2id = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2id[DATASET]


class ErcTextDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET='MELD', SPLIT='train', speaker_mode='upper',
                 num_past_utterances=0, num_future_utterances=0, num_jobs=1,
                 model_checkpoint='roberta-base',
                 ROOT_DIR='multimodal-datasets/',
                 ONLY_UPTO=False, SEED=0):

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.num_jobs = num_jobs
        self.model_checkpoint = model_checkpoint
        self.emotion2id = get_emotion2id(self.DATASET)
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED

        self._load_emotions()
        self._load_utterance_ordered()
        self._string2tokens()
        self.id2emotion = {val: key for key, val in self.emotion2id.items()}

    def _load_emotions(self):
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'emotions.json'), 'r') as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def _load_utterance_ordered(self):
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'utterance-ordered.json'), 'r') as stream:
            utterance_ordered = json.load(stream)[self.SPLIT]

        logging.debug(f"sanity check on if the text files exist ...")
        count = 0
        self.utterance_ordered = {}
        for diaid, uttids in utterance_ordered.items():
            self.utterance_ordered[diaid] = []
            for uttid in uttids:
                try:
                    with open(os.path.join(self.ROOT_DIR, self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json'), 'r') as stream:
                        foo = json.load(stream)
                    self.utterance_ordered[diaid].append(uttid)
                except Exception as e:
                    count += 1
        if count != 0:
            logging.warning(f"number of not existing text files: {count}")
        else:
            logging.info(f"every text file exists fine!")

    def __len__(self):
        return len(self.inputs_)

    def _load_utterance_speaker_emotion(self, uttid, speaker_mode):
        text_path = os.path.join(
            self.ROOT_DIR, self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json')

        with open(text_path, 'r') as stream:
            text = json.load(stream)

        utterance = text['Utterance'].strip()
        emotion = text['Emotion']

        if self.DATASET in ['MELD', 'EmoryNLP']:
            speaker = text['Speaker']
        elif self.DATASET == 'IEMOCAP':
            # speaker = {'Female': 'Alice', 'Male': 'Bob'}[text['Speaker']]
            sessid = text['SessionID']
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                       'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                       'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                       'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                       'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}[sessid][text['Speaker']]

        elif self.DATASET == 'DailyDialog':
            speaker = {'A': 'Alex', 'B': 'Blake'}[text['Speaker']]
        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        if speaker_mode is not None and speaker_mode.lower() == 'upper':
            utterance = speaker.upper() + ': ' + utterance
        elif speaker_mode is not None and speaker_mode.lower() == 'title':
            utterance = speaker.title() + ': ' + utterance

        return {'Utterance': utterance, 'Emotion': emotion}

    def _create_input(self, diaids, speaker_mode, num_past_utterances, num_future_utterances):

        args = {'diaids': diaids,
                'speaker_mode': speaker_mode,
                'num_past_utterances': num_past_utterances,
                'num_future_utterances': num_future_utterances}

#         logging.debug(f"arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0

        inputs = []
        for diaid in tqdm(diaids):
            ues = [self._load_utterance_speaker_emotion(uttid, speaker_mode)
                   for uttid in self.utterance_ordered[diaid]]

            num_tokens = [len(tokenizer(ue['Utterance'])['input_ids'])
                          for ue in ues]

            for idx, ue in enumerate(ues):
                if ue['Emotion'] not in list(self.emotion2id.keys()):
                    continue

                label = self.emotion2id[ue['Emotion']]

                indexes = [idx]
                indexes_past = [i for i in range(
                    idx-1, idx-num_past_utterances-1, -1)]
                indexes_future = [i for i in range(
                    idx+1, idx+num_future_utterances+1, 1)]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future)-len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]['Utterance'] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + utterances[-1]
                    else:
                        final_utterance = ' '.join(
                            utterances[:-1]) + '</s></s>' + utterances[-1]

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + '</s></s>'
                    else:
                        final_utterance = utterances[0] + \
                            '</s></s>' + ' '.join(utterances[1:])

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + \
                            utterances[0] + '</s></s>'
                    else:
                        final_utterance = ' '.join(utterances[:offset]) + '</s></s>' + \
                            utterances[offset] + '</s></s>' + \
                            ' '.join(utterances[offset+1:])
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask['input_ids']
                attention_mask = input_ids_attention_mask['attention_mask']

                input_ = {'input_ids': input_ids,
                          'attention_mask': attention_mask, 'label': label}

                inputs.append(input_)

        logging.info(f"number of truncated utterances: {num_truncated}")
        return inputs

    def _string2tokens(self):
        logging.info(f"converting utterances into tokens ...")
        logging.debug(f"batching dialogues ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        random.shuffle(diaids)

        if self.ONLY_UPTO:
            logging.info(
                f"Using only the first {self.ONLY_UPTO} dialogues ...")
            diaids = diaids[:self.ONLY_UPTO]

        self.num_jobs = min(len(diaids), self.num_jobs)

        BATCH_SIZE = len(diaids) // self.num_jobs

        diaids_batch = [diaids[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                        for i in range(self.num_jobs)]

        diaids_batch[-1] = diaids_batch[-1] + diaids[self.num_jobs*BATCH_SIZE:]
        assert set(diaids) == set([bar for foo in diaids_batch for bar in foo])

        logging.info(f"batching done. batches have "
                     f"{[len(diaids) for diaids in diaids_batch]} dialogues, respectively")

        logging.info(
            f"creating input utterance data with {self.num_jobs} jobs ... ")
        inputs_batch = Parallel(n_jobs=self.num_jobs)(
            delayed(self._create_input)(
                diaids=diaids,
                speaker_mode=self.speaker_mode,
                num_past_utterances=self.num_past_utterances,
                num_future_utterances=self.num_future_utterances)
            for diaids in diaids_batch)

        self.inputs_ = [utt for batch in inputs_batch for utt in batch]

    def __getitem__(self, index):

        return self.inputs_[index]

    def create_utterance():
        raise NotImplementedError


def main(model_checkpoint):

    for DATASET in ['IEMOCAP']:

        if DATASET == 'DailyDialog':
            LABELS_FOR_EVAL = [1, 2, 3, 4, 5, 6]
        else:
            LABELS_FOR_EVAL = None

        def compute_metrics(eval_predictions):
            predictions, label_ids = eval_predictions
            preds = np.argmax(predictions, axis=1)

            f1_weighted = f1_score(
                label_ids, preds, labels=LABELS_FOR_EVAL, average='weighted')
            f1_micro = f1_score(
                label_ids, preds, labels=LABELS_FOR_EVAL, average='micro')
            f1_macro = f1_score(
                label_ids, preds, labels=LABELS_FOR_EVAL, average='macro')

            return {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro}

        sm_pu_fu = [
            ('upper', 8, 8),
            ('upper', 4, 4)]

        for speaker_mode, num_past_utterances, num_future_utterances in sm_pu_fu:

            logging.info(f"automatic hyperparameter tuning with speaker_mode: {speaker_mode}, "
                         f"num_past_utterances: {num_past_utterances}, "
                         f"num_future_utterances: {num_future_utterances}")
            CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            OUTPUT_DIR = f"huggingface-results/{DATASET}/{model_checkpoint}/{CURRENT_TIME}-speaker_mode-{speaker_mode}-num_past_utterances-{num_past_utterances}-num_future_utterances-{num_future_utterances}"

            EVALUATION_STRATEGY = 'epoch'
            LOGGING_STRATEGY = 'epoch'
            SAVE_STRATEGY = 'no'

            ONLY_UPTO = 100

            if model_checkpoint == 'roberta-base':
                BATCH_SIZE = 16
            elif model_checkpoint == 'roberta-large':
                BATCH_SIZE = 16
            else:
                raise ValueError

            ROOT_DIR = './multimodal-datasets/'

            PER_DEVICE_TRAIN_BATCH_SIZE = BATCH_SIZE
            PER_DEVICE_EVAL_BATCH_SIZE = BATCH_SIZE*2
            LOAD_BEST_MODEL_AT_END = False
            SEED = 0
            FP16 = True

            if DATASET in ['MELD', 'EmoryNLP', 'DailyDialog']:
                NUM_CLASSES = 7
            elif DATASET == 'IEMOCAP':
                NUM_CLASSES = 6
            else:
                raise ValueError

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
            )

            def model_init():
                return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_CLASSES)

            ds_train = ErcTextDataset(DATASET=DATASET, SPLIT='train', speaker_mode=speaker_mode,
                                      num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                                      model_checkpoint=model_checkpoint,
                                      ROOT_DIR=ROOT_DIR, ONLY_UPTO=ONLY_UPTO, SEED=SEED)

            ds_val = ErcTextDataset(DATASET=DATASET, SPLIT='val', speaker_mode=speaker_mode,
                                    num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                                    model_checkpoint=model_checkpoint,
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
                    "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.5),
                    "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.5),
                    "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8)
                }

            best_run = trainer.hyperparameter_search(
                direction="minimize", hp_space=my_hp_space, n_trials=10)

            logging.info(f"best hyperparameters found at {best_run}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint, num_labels=NUM_CLASSES)

            logging.info(
                f"training a model with the given hyper parameters ...")

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_train,
                eval_dataset=ds_val,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            with open(os.path.join(OUTPUT_DIR, 'hp.json'), 'w') as stream:
                json.dump(best_run.hyperparameters, stream, indent=4)
            for n, v in best_run.hyperparameters.items():
                logging.info(f"{n}: {v}")
                setattr(trainer.args, n, v)

            trainer.train()

            val_results = trainer.evaluate()
            with open(os.path.join(OUTPUT_DIR, 'val-results.json'), 'w') as stream:
                json.dump(val_results, stream, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    # parser.add_argument('--DATASET', type=str)
    parser.add_argument('--model-checkpoint', type=str)

    args = parser.parse_args()
    args = vars(args)

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
