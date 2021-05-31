import torch
import json
import os
import logging
from tqdm import tqdm
import random
from transformers import RobertaTokenizerFast
from glob import glob
from .common import get_emotion2id, set_seed, get_IEMOCAP_names
from .audio import load_audio, audio2spectrogram


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class ErcTextDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET='MELD', SPLIT='train',
                 num_past_utterances=0, num_future_utterances=0,
                 model_checkpoint='roberta-base',
                 ROOT_DIR='multimodal-datasets/', ADD_BOU=False,
                 ADD_EOU=False,
                 ADD_SPEAKER_TOKENS=False, REPLACE_NAMES_IN_UTTERANCES=False,
                 ONLY_UPTO=False, SEED=0, **kwargs):

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

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_checkpoint)

        self._load_emotions()
        self._load_utterance_ordered()
        self._string2tokens()

    def __len__(self):
        return len(self.inputs_text)

    def __getitem__(self, index):
        return self.inputs_text[index]

    def _load_emotions(self):
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'emotions.json'), 'r') as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def _load_utterance_ordered(self):
        with open(os.path.join(self.ROOT_DIR, self.DATASET, 'utterance-ordered.json'), 'r') as stream:
            utterance_ordered = json.load(stream)[self.SPLIT]

        logging.debug(f"sanity check on if the text files exist ...")
        count = 0
        self.utterance_ordered = {}
        for diaid, uttids in tqdm(utterance_ordered.items()):
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
        self._create_tokens(diaids=diaids,
                            num_past_utterances=self.num_past_utterances,
                            num_future_utterances=self.num_future_utterances)

    def _replace_names_to_tokens(self, utterance):
        for token in self.tokenizer.additional_special_tokens:
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
            if f"<{speaker}>" not in self.tokenizer.additional_special_tokens:
                raise ValueError(f"{speaker} not found!!")
            utterance = f"<{speaker}>" + utterance

        if self.ADD_BOU:
            to_prepend = '<u>'
        else:
            to_prepend = ''

        if self.ADD_EOU:
            to_append = '</u>'
        else:
            to_append = ''

        utterance = to_prepend + utterance + to_append

        return {'Utterance': utterance, 'Emotion': emotion}

    def _create_tokens(self, diaids, num_past_utterances, num_future_utterances):

        args = {'diaids': diaids,
                'num_past_utterances': num_past_utterances,
                'num_future_utterances': num_future_utterances}

        logging.debug(f"arguments given: {args}")
        max_model_input_size = 512
        if num_past_utterances == 0 and num_future_utterances == 0:
            max_model_input_size -= 2
        elif num_past_utterances > 0 and num_future_utterances == 0:
            max_model_input_size -= 2
            max_model_input_size -= 2

        elif num_past_utterances == 0 and num_future_utterances > 0:
            max_model_input_size -= 2
            max_model_input_size -= 2

        elif num_past_utterances > 0 and num_future_utterances > 0:
            max_model_input_size -= 2
            max_model_input_size -= 2
            max_model_input_size -= 2
        else:
            raise ValueError

        num_truncated = 0

        self.inputs_text, self.tokens_length, self.uttids, self.uttids_used = [], [], [], []
        for diaid in tqdm(diaids):
            uses = [self._load_utterance_speaker_emotion(uttid)
                    for uttid in self.utterance_ordered[diaid]]
            uttids_ = [uttid for uttid in self.utterance_ordered[diaid]]

            assert len(uses) == len(uttids_)

            num_tokens = [len(self.tokenizer.encode(use['Utterance'], add_special_tokens=False))
                          for use in uses]

            for idx, (use, uttid) in enumerate(zip(uses, uttids_)):
                if use['Emotion'] not in list(self.emotion2id.keys()):
                    continue

                label = self.emotion2id[use['Emotion']]

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
                    if j is not None and j < len(uses):
                        indexes.append(j)
                        if sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size:
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [uses[idx_]['Utterance'] for idx_ in indexes]
                uttids_used = [uttids_[idx_] for idx_ in indexes]

                assert len(utterances) == len(uttids_used)

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + utterances[-1]
                    else:
                        final_utterance = ''.join(
                            [utt for utt in utterances[:-1]]) + '</s></s>' + utterances[-1]

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + '</s></s>'
                    else:
                        final_utterance = utterances[0] + \
                            '</s></s>' + \
                            ''.join([utt for utt in utterances[1:]])

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = '</s></s>' + \
                            utterances[0] + '</s></s>'
                    else:
                        final_utterance = ''.join([utt for utt in utterances[:offset]]) + '</s></s>' + \
                            utterances[offset] + '</s></s>' + \
                            ''.join([utt for utt in utterances[offset+1:]])

                input_ids_attention_mask = self.tokenizer(final_utterance)
                input_ids = input_ids_attention_mask['input_ids']

                attention_mask = input_ids_attention_mask['attention_mask']

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(input_ids) == sum(
                        [num_tokens[idx_] for idx_ in indexes]) + 2

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    assert len(input_ids) == sum(
                        [num_tokens[idx_] for idx_ in indexes]) + 4

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    assert len(input_ids) == sum(
                        [num_tokens[idx_] for idx_ in indexes]) + 4

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    assert len(input_ids) == sum(
                        [num_tokens[idx_] for idx_ in indexes]) + 6

                input_t = {'input_ids': input_ids,
                           'attention_mask': attention_mask, 'label': label}

                assert len([num_tokens[idx_]
                            for idx_ in indexes]) == len(uttids_used)
                self.tokens_length.append(
                    [num_tokens[idx_] for idx_ in indexes])
                self.inputs_text.append(input_t)
                self.uttids.append(uttid)
                self.uttids_used.append(uttids_used)

        logging.info(f"number of truncated utterances: {num_truncated}")
        assert len(self.inputs_text) == len(
            self.tokens_length) == len(self.uttids)


def save_special_tokenzier(DATASET='MELD', ROOT_DIR='multimodal-datasets/',
                           ADD_BOU=False, ADD_EOU=False,
                           ADD_SPEAKER_TOKENS=True, SPLITS=['train', 'val', 'test'],
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

    if save_at is None:
        return special_tokens_dict['additional_special_tokens']

    tokenizer.save_pretrained(save_at)
