from sklearn.metrics import f1_score
import random
import os
import torch
import numpy as np
from transformers import RobertaTokenizerFast
import logging
from glob import glob
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def get_num_classes(DATASET):
    if DATASET == 'MELD':
        NUM_CLASSES = 7
    elif DATASET == 'IEMOCAP':
        NUM_CLASSES = 6
    else:
        raise ValueError

    return NUM_CLASSES


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    f1_weighted = f1_score(label_ids, preds, average='weighted')
    f1_micro = f1_score(label_ids, preds, average='micro')
    f1_macro = f1_score(label_ids, preds, average='macro')

    return {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


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

    emotion2id = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2id[DATASET]


def get_IEMOCAP_names():

    # https: // www.ssa.gov/oact/babynames/decades/century.html
    names_dict = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                  'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                  'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                  'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                  'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}

    return names_dict


def save_special_tokenzier(DATASET='MELD', ROOT_DIR='multimodal-datasets/',
                           ADD_BOU_EOU=False, ADD_SPEAKER_TOKENS=False, SPLITS=['train'],
                           base_tokenizer='roberta-base', save_at='./'):

    tokenizer = RobertaTokenizerFast.from_pretrained(base_tokenizer)

    special_tokens_dict = {'additional_special_tokens': []}
    if ADD_BOU_EOU:
        special_tokens_dict['additional_special_tokens'].append('<u>')
        special_tokens_dict['additional_special_tokens'].append('</u>')

        logging.info(f"BOU: <u> and EOU: </u> added.")

    if ADD_SPEAKER_TOKENS:
        # special_tokens_dict['additional_special_tokens'].append('<Stranger>')
        # logging.info(f"stranger: <Stranger> added.")

        # special_tokens_dict['additional_special_tokens'].append('<Stranger1>')
        # logging.info(f"stranger: <Stranger1> added.")

        speakers = []

        for SPLIT in SPLITS:
            for text_path in glob(os.path.join(ROOT_DIR, DATASET, 'raw-texts', SPLIT, '*.json')):
                with open(text_path, 'r') as stream:
                    text = json.load(stream)

                if DATASET == 'MELD':
                    speaker = text['Speaker']

                elif DATASET == 'IEMOCAP':
                    sessid = text['SessionID']
                    speaker = get_IEMOCAP_names()[sessid][text['Speaker']]

                else:
                    raise ValueError(f"{DATASET} is not supported!!!")

                speaker = speaker.strip()
                speaker = speaker.title()

                speakers.append(speaker)

        speakers = sorted(list(set(speakers)))

        speakers = [f"<{speaker}>"for speaker in speakers]
        logging.info(f"{len(speakers)} speaker-specific tokens added.")

        for speaker in speakers:
            special_tokens_dict['additional_special_tokens'].append(speaker)

    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    logging.info(f"In total of {num_added_tokens} special tokens added.")

    tokenizer.save_pretrained(save_at)
