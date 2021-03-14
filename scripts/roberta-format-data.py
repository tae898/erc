import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import os
from glob import glob
import sys
DATASET_DIR = "Datasets/"
DATASETS_SUPPORTED = ['MELD', 'IEMOCAP', 'EmoryNLP', 'DailyDialog']


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

    emotion2num = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2num[DATASET]


def make_utterance(utterance, speaker, mode='title'):
    if mode == 'title':
        utterance = speaker.title() + ': ' + utterance
    elif mode == 'upper':
        utterance = speaker.upper() + ': ' + utterance
    elif mode == 'lower':
        utterance = speaker.lower() + ': ' + utterance

    return utterance


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
    utterance = make_utterance(utterance, speaker, 'title')

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


def format_mlm(DATASET):
    print(f"formatting {DATASET} data for pretraining with mlm ...")
    _, utterance_ordered = load_labels_utt_ordered(DATASET)
    os.makedirs(os.path.join(DATASET_DIR, DATASET, 'roberta/mlm'), exist_ok=True)
    for SPLIT in tqdm(['train', 'val', 'test']):
        rawtext = []
        for diaid, uttids in tqdm(utterance_ordered[SPLIT].items()):
            diatext = []
            for uttid in uttids:
                with open(f"Datasets/{DATASET}/raw-texts/{SPLIT}/{uttid}.json") as stream:
                    utt = json.load(stream)
                utterance = utt['Utterance']
                speaker = utt['Speaker']

                utterance = make_utterance(utterance, speaker, 'title')

                diatext.append(utterance)
            rawtext.append(diatext)

        f1 = open(os.path.join(DATASET_DIR, DATASET, 'roberta/mlm',
                               f"{SPLIT}.raw"), 'w')

        for diatext in rawtext:
            for utterance in diatext:
                f1.write(utterance + ' ')
            f1.write('\n\n')
        f1.close()


def format_nsp(DATASETS, num_utt):
    raise NotImplementedError


def format_classification(DATASET, num_utt=1):
    emotion2num = get_emotion2num(DATASET)
    labels, utterance_ordered = load_labels_utt_ordered(DATASET)

    for SPLIT in tqdm(['train', 'val', 'test']):
        for input_order in range(num_utt):
            write_input_label(DATASET, SPLIT, input_order, num_utt, labels,
                              utterance_ordered, emotion2num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--DATASET', help=f"e.g. {DATASETS_SUPPORTED}")
    parser.add_argument('--num-utt', default=1, type=int,
                        help='e.g. 1, 2, 3, etc.')
    parser.add_argument('--pretrain-mlm', action='store_true')
    parser.add_argument('--pretrain-nsp', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    print(f"arguments given to {__file__}: {args}")

    if args['DATASET'] not in DATASETS_SUPPORTED:
        sys.exit(f"{args['DATASET']} is not supported!")

    if args['pretrain_nsp']:
        args.pop('pretrain_nsp')
        format_nsp(**args)
    elif args['pretrain_mlm']:
        format_mlm(args['DATASET'])
    else:
        args.pop('pretrain_nsp')
        format_classification(**args)
