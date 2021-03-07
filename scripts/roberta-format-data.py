import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import os
from glob import glob
import sys
DATASET_DIR = "Datasets/"


def get_emotion2num(DATASET):

    emotions = {}
    emotions['MELD'] = ['anger',
                        'disgust',
                        'fear',
                        'joy',
                        'neutral',
                        'sadness',
                        'surprise']

    emotions['IEMOCAP'] = ['anger',
                           'disgust',
                           'excited',
                           'fear',
                           'frustration',
                           'happiness',
                           'neutral',
                           'other',
                           'sadness',
                           'surprise',
                           'undecided']

    emotions['CAER'] = ['anger',
                        'disgust',
                        'fear',
                        'happy',
                        'neutral',
                        'sad',
                        'surprise']

    emotion2num = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}


    return emotion2num[DATASET]


def load_labels_utt_ordered(DATASET):
    with open(os.path.join(DATASET_DIR, DATASET, 'labels.json'), 'r') as stream:
        labels = json.load(stream)

    with open(os.path.join(DATASET_DIR, DATASET, 'utterance-ordered.json'), 'r') as stream:
        utterance_ordered = json.load(stream)

    return labels, utterance_ordered


def get_uttid_speaker_utterance_emotion(labels, SPLIT, json_path):
    with open(json_path, 'r') as stream:
        text = json.load(stream)
    uttid = os.path.basename(json_path).split('.json')[0]
    speaker = text['Speaker']
    utterance = text['Utterance']
    emotion = labels[SPLIT][uttid]

    # very important here.
    utterance = speaker.title() + ': ' + utterance

    return uttid, speaker, utterance, emotion


def write_input_label(DATASET, SPLIT, input_order, num_utt, labels,
                      utterance_ordered, emotion2num):
    history = num_utt - input_order - 1
    samples = []
    diaids = list(utterance_ordered[SPLIT].keys())

    for diaid in diaids:
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            labels, SPLIT, json_path) for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        for _ in range(history):
            utterances.insert(0, '')

        for _ in range(history):
            utterances.pop()

        assert len(utterances) == len(emotions)

        for utterance, emotion in zip(utterances, emotions):
            samples.append((utterance, emotion2num[emotion]))

    f1 = open(os.path.join(DATASET_DIR, DATASET,
                           'roberta', SPLIT + f'.input{input_order}'), 'w')

    f2 = open(os.path.join(DATASET_DIR, DATASET,
                           'roberta', SPLIT + '.label'), 'w')

    for sample in samples:
        f1.write(sample[0] + '\n')
        f2.write(str(sample[1]) + '\n')
    f1.close()
    f2.close()


def format_classification(DATASET, num_utt=1):
    if DATASET not in ['MELD', 'IEMOCAP', 'AFEW', 'CAER']:
        sys.exit(f"{DATASET} is not supported!")

    emotion2num = get_emotion2num(DATASET)
    labels, utterance_ordered = load_labels_utt_ordered(DATASET)

    for SPLIT in tqdm(['train', 'val', 'test']):
        for input_order in range(num_utt):
            write_input_label(DATASET, SPLIT, input_order, num_utt, labels,
                              utterance_ordered, emotion2num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--DATASET', help='e.g. IEMOCAP, MELD, AFEW, CAER')
    parser.add_argument('--num-utt', default=1, type=int,
                        help='e.g. 1, 2, 3, etc.')
    parser.add_argument('--pretrain-nsp', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    print(f"arguments given to {__file__}: {args}")

    if args['pretrain_nsp']:
        args.pop('pretrain_nsp')
        format_nsp(**args)
    else:
        args.pop('pretrain_nsp')
        format_classification(**args)
