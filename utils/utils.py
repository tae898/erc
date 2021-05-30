import torch
import json
import os
import logging
from torch._C import Value
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import random
from transformers import RobertaTokenizerFast
from glob import glob


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


class ErcTextDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET='MELD', SPLIT='train',
                 num_past_utterances=0, num_future_utterances=0,
                 model_checkpoint='roberta-base',
                 ROOT_DIR='multimodal-datasets/', ADD_BOU=False,
                 ADD_EOU=False,
                 ADD_SPEAKER_TOKENS=False, REPLACE_NAMES_IN_UTTERANCES=False,
                 ONLY_UPTO=False, SEED=0):

        self.DATASET = DATASET
        self.ROOT_DIR = ROOT_DIR
        self.SPLIT = SPLIT
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.emotion2id = get_emotion2id(self.DATASET)
        self.id2emotion = {val: key for key, val in self.emotion2id.items()}
        self.ONLY_UPTO = ONLY_UPTO
        self.SEED = SEED
        self.ADD_BOU = ADD_BOU
        self.ADD_EOU = ADD_EOU
        self.ADD_SPEAKER_TOKENS = ADD_SPEAKER_TOKENS
        self.REPLACE_NAMES_IN_UTTERANCES = REPLACE_NAMES_IN_UTTERANCES

        if self.ADD_BOU or self.ADD_EOU or self.ADD_SPEAKER_TOKENS:
            with open(os.path.join(self.model_checkpoint, 'added_tokens.json'), 'r') as stream:
                self.added_tokens = json.load(stream)
        else:
            self.added_tokens = []

        self._load_emotions()
        self._load_utterance_ordered()
        self._string2tokens()

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

    def _replace_names_to_tokens(self, utterance):
        for token, token_id in self.added_tokens.items():
            if (self.ADD_BOU or self.ADD_EOU) and token in ['<u>', '</u>']:
                continue
            token_stripped = token.split('<')[-1].split('>')[0]

            utterance = utterance.replace(token_stripped, token)

        return utterance

    def _load_utterance_speaker_emotion(self, uttid):
        text_path = os.path.join(
            self.ROOT_DIR, self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json')

        with open(text_path, 'r') as stream:
            text = json.load(stream)

        utterance = text['Utterance'].strip()

        if self.REPLACE_NAMES_IN_UTTERANCES:
            assert self.ADD_SPEAKER_TOKENS, f"ADD_SPEAKER_TOKENS has to be true!"
            utterance = self._replace_names_to_tokens(utterance)

        emotion = text['Emotion']

        if self.DATASET == 'MELD':
            speaker = text['Speaker']
        elif self.DATASET == 'IEMOCAP':
            sessid = text['SessionID']
            speaker = get_IEMOCAP_names()[sessid][text['Speaker']]

        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        speaker = speaker.strip()
        speaker = speaker.title()

        if self.ADD_SPEAKER_TOKENS:
            if f"<{speaker}>" not in list(self.added_tokens.keys()):
                raise ValueError(f"{speaker} not found!!")
            utterance = f"<{speaker}>" + utterance

        return {'Utterance': utterance, 'Emotion': emotion}

    def _augment_utterance(self, utterance):
        if self.ADD_BOU:
            to_prepend = '<u>'
        else:
            to_prepend = ''

        if self.ADD_EOU:
            to_append = '</u>'
        else:
            to_append = ''

        return to_prepend + utterance + to_append

    def _create_input(self, diaids, num_past_utterances, num_future_utterances):

        args = {'diaids': diaids,
                'num_past_utterances': num_past_utterances,
                'num_future_utterances': num_future_utterances}

        logging.debug(f"arguments given: {args}")
        tokenizer = RobertaTokenizerFast.from_pretrained(
            self.model_checkpoint, use_fast=True)
        max_model_input_size = 512
        num_truncated = 0

        inputs = []
        self.uttids = []
        for diaid in tqdm(diaids):
            ues = [self._load_utterance_speaker_emotion(uttid)
                   for uttid in self.utterance_ordered[diaid]]
            uttids_ = [uttid for uttid in self.utterance_ordered[diaid]]

            assert len(ues) == len(uttids_)

            pad_BOU_EOU = 0
            if self.ADD_BOU:
                pad_BOU_EOU += 1
            if self.ADD_EOU:
                pad_BOU_EOU += 1

            num_tokens = [len(tokenizer(ue['Utterance'])['input_ids']) + pad_BOU_EOU
                          for ue in ues]

            for idx, (ue, uttid) in enumerate(zip(ues, uttids_)):
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
                    final_utterance = self._augment_utterance(utterances[0])

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + \
                            self._augment_utterance(utterances[-1])
                    else:
                        final_utterance = ''.join([self._augment_utterance(utt)
                                                   for utt in utterances[:-1]]) + '</s></s>' + self._augment_utterance(utterances[-1])

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = self._augment_utterance(
                            utterances[0]) + '</s></s>'
                    else:
                        final_utterance = self._augment_utterance(utterances[0]) + \
                            '</s></s>' + \
                            ''.join([self._augment_utterance(utt)
                                     for utt in utterances[1:]])

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + \
                            self._augment_utterance(utterances[0]) + '</s></s>'
                    else:
                        final_utterance = ''.join([self._augment_utterance(utt) for utt in utterances[:offset]]) + '</s></s>' + \
                            utterances[offset] + '</s></s>' + \
                            ''.join([self._augment_utterance(utt)
                                     for utt in utterances[offset+1:]])
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask['input_ids']
                attention_mask = input_ids_attention_mask['attention_mask']

                input_ = {'input_ids': input_ids,
                          'attention_mask': attention_mask, 'label': label}

                inputs.append(input_)
                self.uttids.append(uttid)

        logging.info(f"number of truncated utterances: {num_truncated}")
        return inputs

    def _string2tokens(self):
        logging.info(f"converting utterances into tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(self.SEED)
        random.shuffle(diaids)

        if self.ONLY_UPTO:
            logging.info(
                f"Using only the first {self.ONLY_UPTO} dialogues ...")
            diaids = diaids[:self.ONLY_UPTO]

        logging.info(f"creating input utterance data ... ")
        self.inputs_ = self._create_input(diaids=diaids,
                                          num_past_utterances=self.num_past_utterances,
                                          num_future_utterances=self.num_future_utterances)

        assert len(self.inputs_) == len(self.uttids)

    def __getitem__(self, index):

        return self.inputs_[index]


def save_special_tokenzier(DATASET='MELD', ROOT_DIR='multimodal-datasets/',
                           ADD_BOU=False, ADD_EOU=False,
                           ADD_SPEAKER_TOKENS=False, SPLITS=['train'],
                           base_tokenizer='roberta-base', save_at='./'):

    tokenizer = RobertaTokenizerFast.from_pretrained(base_tokenizer)

    special_tokens_dict = {'additional_special_tokens': []}
    if ADD_BOU:
        special_tokens_dict['additional_special_tokens'].append('<u>')
        logging.info(f"BOU: <u> added")
    if ADD_EOU:
        special_tokens_dict['additional_special_tokens'].append('</u>')
        logging.info(f"EOU: </u> added.")

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
