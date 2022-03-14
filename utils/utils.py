"""utility and helper functions / classes."""
import json
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def make_MELD_IEMOCAP():

    SEED = 42
    ratios = {"train": 0.9, "val": 0.1, "test": 0}

    assert sum(list(ratios.values())) == 1

    utterance_ordered = {}

    with open(f"./multimodal-datasets/MELD/utterance-ordered.json", "r") as stream:
        utterance_ordered["MELD"] = json.load(stream)

    with open(f"./multimodal-datasets/IEMOCAP/utterance-ordered.json", "r") as stream:
        utterance_ordered["IEMOCAP"] = json.load(stream)

    diaids_merged = []

    for DATASET in ["MELD", "IEMOCAP"]:
        for SPLIT in ["train", "val", "test"]:
            diaids = list(utterance_ordered[DATASET][SPLIT].keys())
            for diaid in diaids:
                diaids_merged.append(f"{DATASET}/{SPLIT}/{diaid}")

    random.seed(SEED)
    random.shuffle(diaids_merged)

    train_idx = int(len(diaids_merged) * ratios["train"])
    val_idx = int(len(diaids_merged) * (ratios["train"] + ratios["val"]))

    diaids_train = diaids_merged[:train_idx]
    diaids_val = diaids_merged[train_idx:val_idx]
    diaids_test = diaids_merged[val_idx:]

    assert len(diaids_merged) == (
        len(diaids_train) + len(diaids_val) + len(diaids_test)
    )

    diaids_merged = {"train": diaids_train, "val": diaids_val, "test": diaids_test}

    utterance_ordered_merged = {}

    for SPLIT in ["train", "val", "test"]:
        utterance_ordered_merged[SPLIT] = {}

        for diaid in tqdm(diaids_merged[SPLIT]):

            d_, s_, d__ = diaid.split("/")
            utterance_ordered_merged[SPLIT][diaid] = [
                f"{d_}/{s_}/{d__}/{uttid}" for uttid in utterance_ordered[d_][s_][d__]
            ]

    assert len(
        [
            val___
            for key, val in utterance_ordered.items()
            for key_, val_ in val.items()
            for key__, val__ in val_.items()
            for val___ in val__
        ]
    ) == len(
        [
            val__
            for key, val in utterance_ordered_merged.items()
            for key_, val_ in val.items()
            for val__ in val_
        ]
    )

    with open("./utterance-ordered-MELD_IEMOCAP.json", "w") as stream:
        json.dump(utterance_ordered_merged, stream, indent=4)


def get_num_classes(DATASET: str) -> int:
    """Get the number of classes to be classified by dataset."""
    if DATASET == "MELD":
        NUM_CLASSES = 7
    elif DATASET == "IEMOCAP":
        NUM_CLASSES = 6
    elif DATASET == "MELD_IEMOCAP":
        NUM_CLASSES = 7
    else:
        raise ValueError

    return NUM_CLASSES


def compute_metrics(eval_predictions) -> dict:
    """Return f1_weighted, f1_micro, and f1_macro scores."""
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average="weighted")
    f1_micro = f1_score(label_ids, preds, average="micro")
    f1_macro = f1_score(label_ids, preds, average="macro")

    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}


def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.

    Set everything to be deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id(DATASET: str) -> Tuple[dict, dict]:
    """Get a dict that converts string class to numbers."""

    if DATASET == "MELD":
        # MELD has 7 classes
        emotions = [
            "neutral",
            "joy",
            "surprise",
            "anger",
            "sadness",
            "disgust",
            "fear",
        ]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    elif DATASET == "IEMOCAP":
        # IEMOCAP originally has 11 classes but we'll only use 6 of them.
        emotions = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    elif DATASET == "MELD_IEMOCAP":
        # IEMOCAP emotions are mapped to MELD emotions, 7 classes.
        emotions = [
            "neutral",
            "joy",
            "surprise",
            "anger",
            "sadness",
            "disgust",
            "fear",
        ]

        emotion2id = {
            "neutral": 0,
            "joy": 1,
            "happiness": 1,
            "excited": 1,
            "surprise": 2,
            "anger": 3,
            "frustration": 3,
            "sadness": 4,
            "disgust": 5,
            "fear": 6,
        }
        id2emotion = {idx: emotion for idx, emotion in enumerate(emotions)}

    return emotion2id, id2emotion


class ErcTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        DATASET="MELD",
        SPLIT="train",
        speaker_mode="upper",
        num_past_utterances=0,
        num_future_utterances=0,
        model_checkpoint="roberta-base",
        ROOT_DIR="multimodal-datasets/",
        ONLY_UPTO=False,
        SEED=0,
    ):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.speaker_mode = speaker_mode
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.emotion2id, self.id2emotion = get_emotion2id(self.DATASET)
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED

        self._load_emotions()
        self._load_utterance_ordered()
        self._string2tokens()

    def _load_emotions(self):
        """Load the supervised labels"""
        if self.DATASET in ["MELD", "IEMOCAP"]:
            with open(
                os.path.join(self.ROOT_DIR, self.DATASET, "emotions.json"), "r"
            ) as stream:
                self.emotions = json.load(stream)[self.SPLIT]

    def _load_utterance_ordered(self):
        """Load the ids of the utterances in order."""
        if self.DATASET in ["MELD", "IEMOCAP"]:
            path = os.path.join(self.ROOT_DIR, self.DATASET, "utterance-ordered.json")
        elif self.DATASET == "MELD_IEMOCAP":
            path = "./utterance-ordered-MELD_IEMOCAP.json"

        with open(path, "r") as stream:
            self.utterance_ordered = json.load(stream)[self.SPLIT]

    def __len__(self):
        return len(self.inputs_)

    def _load_utterance_speaker_emotion(self, uttid, speaker_mode) -> dict:
        """Load an speaker-name prepended utterance and emotion label"""

        if self.DATASET in ["MELD", "IEMOCAP"]:
            text_path = os.path.join(
                self.ROOT_DIR, self.DATASET, "raw-texts", self.SPLIT, uttid + ".json"
            )
        elif self.DATASET == "MELD_IEMOCAP":
            assert len(uttid.split("/")) == 4
            d_, s_, d__, u_ = uttid.split("/")
            text_path = os.path.join(self.ROOT_DIR, d_, "raw-texts", s_, u_ + ".json")

        with open(text_path, "r") as stream:
            text = json.load(stream)

        utterance = text["Utterance"].strip()
        emotion = text["Emotion"]

        if self.DATASET == "MELD":
            speaker = text["Speaker"]
        elif self.DATASET == "IEMOCAP":
            sessid = text["SessionID"]
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {
                "Ses01": {"Female": "Mary", "Male": "James"},
                "Ses02": {"Female": "Patricia", "Male": "John"},
                "Ses03": {"Female": "Jennifer", "Male": "Robert"},
                "Ses04": {"Female": "Linda", "Male": "Michael"},
                "Ses05": {"Female": "Elizabeth", "Male": "William"},
            }[sessid][text["Speaker"]]
        elif self.DATASET == "MELD_IEMOCAP":
            speaker = ""
        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        if speaker_mode is not None and speaker_mode.lower() == "upper":
            utterance = speaker.upper() + ": " + utterance
        elif speaker_mode is not None and speaker_mode.lower() == "title":
            utterance = speaker.title() + ": " + utterance

        return {"Utterance": utterance, "Emotion": emotion}

    def _create_input(
        self, diaids, speaker_mode, num_past_utterances, num_future_utterances
    ):
        """Create an input which will be an input to RoBERTa."""

        args = {
            "diaids": diaids,
            "speaker_mode": speaker_mode,
            "num_past_utterances": num_past_utterances,
            "num_future_utterances": num_future_utterances,
        }

        logging.debug(f"arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0

        inputs = []
        for diaid in tqdm(diaids):
            ues = [
                self._load_utterance_speaker_emotion(uttid, speaker_mode)
                for uttid in self.utterance_ordered[diaid]
            ]

            num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues]

            for idx, ue in enumerate(ues):
                if ue["Emotion"] not in list(self.emotion2id.keys()):
                    continue

                label = self.emotion2id[ue["Emotion"]]

                indexes = [idx]
                indexes_past = [
                    i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
                ]
                indexes_future = [
                    i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
                ]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future) - len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if (
                            sum([num_tokens[idx_] for idx_ in indexes])
                            > max_model_input_size
                        ):
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[-1]
                    else:
                        final_utterance = (
                            " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                        )

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            utterances[0] + "</s></s>" + " ".join(utterances[1:])
                        )

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            " ".join(utterances[:offset])
                            + "</s></s>"
                            + utterances[offset]
                            + "</s></s>"
                            + " ".join(utterances[offset + 1 :])
                        )
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]

                input_ = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": label,
                }

                inputs.append(input_)

        logging.info(f"number of truncated utterances: {num_truncated}")
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting utterances into tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        random.shuffle(diaids)

        if self.ONLY_UPTO:
            logging.info(f"Using only the first {self.ONLY_UPTO} dialogues ...")
            diaids = diaids[: self.ONLY_UPTO]

        logging.info(f"creating input utterance data ... ")
        self.inputs_ = self._create_input(
            diaids=diaids,
            speaker_mode=self.speaker_mode,
            num_past_utterances=self.num_past_utterances,
            num_future_utterances=self.num_future_utterances,
        )

    def __getitem__(self, index):

        return self.inputs_[index]
