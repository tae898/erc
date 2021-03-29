import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import os
from glob import glob
import sys
import nltk
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
DATASET_DIR = "Datasets/"
DATASETS_SUPPORTED = ['MELD', 'IEMOCAP', 'EmoryNLP', 'DailyDialog']


def clean_utterance(utterance):
    """Clean utterance.

    I define a clean utterance as an utterane without more than one white space
    between characters. Furthermore, every utterance should end with a proper
    punctuation (e.g. '.'). If the given utterance does not end with a proper
    puncutation, period (.) is appended at the end.

    """
    assert isinstance(utterance, str)

    specials_1 = ["!", "%",  ")", ",", ".", ":", ";", "?",
                  "’", "”", "′", "。"]

    specials_2 = ["\"", "#", "(", "@",
                  "°", "‘", "“", "′", "’"]

    utterance = utterance.strip()

    for special in specials_1:
        utterance = utterance.replace(' ' + special, special)

    for special in specials_2:
        utterance = utterance.replace(special + ' ', special)

    num_whitespaces = 10
    last_punctuations = \
        ['!', '"', '%', '&', "'", ')', '*', ',',
            '-', '.', '/', ':', ';', '?', ']', '_',
            '—', '’', '”', '…', '。']
    utterance = utterance.strip()
    for i in range(num_whitespaces, 1, -1):
        utterance = utterance.replace(' '*i, ' ')
    if utterance[-1] not in last_punctuations:
        utterance += '.'

    return utterance


def get_emotion2num(DATASET):

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

    # DailyDialog originally has 7 classes, but we'll use 6 of them
    # we remove neutral
    emotions['DailyDialog'] = ['happiness',
                               'surprise',
                               'sadness',
                               'anger',
                               'disgust',
                               'fear']

    emotion2num = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2num[DATASET]


def make_utterance(utterance, speaker, speaker_mode='title'):
    if speaker_mode == 'title':
        utterance = speaker.title() + ': ' + utterance
    elif speaker_mode == 'upper':
        utterance = speaker.upper() + ': ' + utterance
    elif speaker_mode == 'lower':
        utterance = speaker.lower() + ': ' + utterance

    return utterance


def load_labels_utt_ordered(DATASET):
    with open(os.path.join(DATASET_DIR, DATASET, 'labels.json'), 'r') as stream:
        labels = json.load(stream)

    with open(os.path.join(DATASET_DIR, DATASET, 'utterance-ordered.json'), 'r') as stream:
        utterance_ordered = json.load(stream)

    return labels, utterance_ordered


def get_uttid_speaker_utterance_emotion(DATASET, labels, SPLIT, json_path,
                                        speaker_mode=None):

    with open(json_path, 'r') as stream:
        text = json.load(stream)
    uttid = os.path.basename(json_path).split('.json')[0]
    if DATASET in ['MELD', 'EmoryNLP']:
        speaker = text['Speaker']
    elif DATASET == 'IEMOCAP':
        speaker = {'Female': 'Alice', 'Male': 'Bob'}[text['Speaker']]
        # sessid = text['SessionID']
        # https: // www.ssa.gov/oact/babynames/decades/century.html
        # speaker = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
        #            'Ses02': {'Female': 'Patricia', 'Male': 'John'},
        #            'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
        #            'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
        #            'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}[sessid][speaker]

    elif DATASET == 'DailyDialog':
        speaker = {'A': 'Alice', 'B': 'Bob'}[text['Speaker']]
        # assert text['Speaker'] in ['A', 'B']
        # speaker = 'Person' + ' ' + text['Speaker']
        # random two gender neutral names
        # speaker = {'A': 'Alex',
        #            'B': 'Charlie'}[text['Speaker']]
    else:
        raise ValueError(f"{DATASET} not supported!!!!!!")

    utterance = text['Utterance']
    emotion = labels[SPLIT][uttid]

    # very important here.
    utterance = make_utterance(utterance, speaker, speaker_mode)

    return uttid, speaker, utterance, emotion


def write_input_label(DATASET, SPLIT, labels, num_utts,
                      utterance_ordered, emotion2num, speaker_mode=None,
                      tokens_per_sample=512, clean_utterances=True):
    NUM_TOTAL_TRUNCATIONS = 0
    max_tokens_input0 = 0
    max_tokens_input1 = 0

    diaids = list(utterance_ordered[SPLIT].keys())
    assert num_utts > 1

    input1 = []
    input0 = []
    labelnums = []

    for diaid in tqdm(diaids):
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            DATASET, labels, SPLIT, json_path, speaker_mode)
            for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        assert len(utterances) == len(emotions)

        for idx, (utterance, emotion) in enumerate(zip(utterances, emotions)):
            if emotion not in list(emotion2num.keys()):
                continue

            if clean_utterances:
                utterance = clean_utterance(utterance)

            num_tokens1 = len(roberta.encode(utterance).tolist())

            # -2 is for <CLS> and <SEP>
            JUST_IN_CASE = 2
            if num_tokens1 > tokens_per_sample - 2 - JUST_IN_CASE:
                raise ValueError(f"{utterance} is too long!!")

            max_tokens_input1 = max(max_tokens_input1, num_tokens1)

            history = []
            start = idx - num_utts + 1
            end = idx
            for i in range(start, end):
                if i >= 0:
                    if clean_utterances:
                        history.append(clean_utterance(utterances[i]))
                    else:
                        history.append(utterances[i])

            is_truncated = 0
            while True:
                num_tokens0 = len(roberta.encode(' '.join(history)).tolist())
                # -4 is for <CLS>, <SEP><SEP>, and <SEP>
                if num_tokens0 + num_tokens1 <= tokens_per_sample - 4 - JUST_IN_CASE:
                    break
                else:
                    # remove the oldest history utterance
                    history.pop(0)
                    is_truncated = 1

            max_tokens_input0 = max(max_tokens_input0, num_tokens0)

            NUM_TOTAL_TRUNCATIONS += is_truncated

            history = ' '.join(history)

            input0.append(history)
            input1.append(utterance)
            labelnums.append(emotion2num[emotion])

    assert len(input0) == len(input1) == len(labelnums)

    f_input0 = open(os.path.join(DATASET_DIR, DATASET,
                                 'roberta', SPLIT + f'.input0'), 'w')

    f_input1 = open(os.path.join(DATASET_DIR, DATASET,
                                 'roberta', SPLIT + f'.input1'), 'w')

    f_label = open(os.path.join(DATASET_DIR, DATASET,
                                'roberta', SPLIT + '.label'), 'w')

    for i0, i1, ln in zip(input0, input1, labelnums):
        f_input0.write(i0 + '\n')
        f_input1.write(i1 + '\n')
        f_label.write(str(ln) + '\n')

    f_input0.close()
    f_input1.close()
    f_label.close()

    print(f"{DATASET}, {SPLIT} has {NUM_TOTAL_TRUNCATIONS} truncations")

    return max_tokens_input0, max_tokens_input1


def write_input_label_simple(DATASET, SPLIT, labels, utterance_ordered,
                             emotion2num, speaker_mode=None,
                             clean_utterances=True):
    max_tokens_input0 = 0
    samples = []
    diaids = list(utterance_ordered[SPLIT].keys())

    for diaid in tqdm(diaids):
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            DATASET, labels, SPLIT, json_path, speaker_mode) for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        assert len(utterances) == len(emotions)

        for utterance, emotion in zip(utterances, emotions):

            if emotion not in list(emotion2num.keys()):
                continue

            if clean_utterances:
                utterance = clean_utterance(utterance)

            num_tokens0 = len(roberta.encode(utterance).tolist())
            max_tokens_input0 = max(max_tokens_input0, num_tokens0)
            samples.append((utterance, emotion2num[emotion]))

    f_input0 = open(os.path.join(DATASET_DIR, DATASET,
                                 'roberta', SPLIT + f'.input0'), 'w')

    f_label = open(os.path.join(DATASET_DIR, DATASET,
                                'roberta', SPLIT + '.label'), 'w')

    for sample in samples:
        f_input0.write(sample[0] + '\n')
        f_label.write(str(sample[1]) + '\n')
    f_input0.close()
    f_label.close()

    return max_tokens_input0


def format_classification(DATASET, num_utts=1, speaker_mode=None,
                          tokens_per_sample=512, clean_utterances=True):
    assert num_utts > 0

    if speaker_mode == 'none':
        speaker_mode = None
    emotion2num = get_emotion2num(DATASET)
    labels, utterance_ordered = load_labels_utt_ordered(DATASET)

    for SPLIT in tqdm(['train', 'val', 'test']):
        if num_utts == 1:
            max_tokens_input0 = write_input_label_simple(
                DATASET, SPLIT, labels, utterance_ordered,
                emotion2num, speaker_mode, clean_utterances)
            max_tokens_input1 = None
        else:
            max_tokens_input0, max_tokens_input1 = write_input_label(
                DATASET, SPLIT, labels, num_utts,
                utterance_ordered, emotion2num, speaker_mode, tokens_per_sample,
                clean_utterances)

        print(f"{DATASET}, {SPLIT}, input0 has max tokens of {max_tokens_input0}")

        if max_tokens_input1 is not None:
            print(f"{DATASET}, {SPLIT}, input1 has max tokens of {max_tokens_input1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--DATASET', help=f"e.g. {DATASETS_SUPPORTED}")
    parser.add_argument('--num-utts', default=1, type=int,
                        help='e.g. 1, 2, 3, etc.')
    parser.add_argument('--speaker-mode', default=None,
                        help='e.g. title, upper, lower, none')
    parser.add_argument('--tokens-per-sample', default=512, type=int,
                        help='e.g. 512, 1024, etc.')
    parser.add_argument('--clean-utterances', help="clean utterances or not.")

    args = parser.parse_args()
    args = vars(args)

    if args['clean_utterances'] == 'true':
        args['clean_utterances'] = True
    else:
        args['clean_utterances'] = False

    print(f"arguments given to {__file__}: {args}")

    if args['DATASET'] not in DATASETS_SUPPORTED:
        sys.exit(f"{args['DATASET']} is not supported!")

    format_classification(**args)
