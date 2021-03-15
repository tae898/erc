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


def get_uttid_speaker_utterance_emotion(labels, SPLIT, json_path,
                                        speaker_mode=None):
    with open(json_path, 'r') as stream:
        text = json.load(stream)
    uttid = os.path.basename(json_path).split('.json')[0]
    speaker = text['Speaker']
    utterance = text['Utterance']
    emotion = labels[SPLIT][uttid]

    # very important here.
    utterance = make_utterance(utterance, speaker, speaker_mode)

    return uttid, speaker, utterance, emotion


def write_input_label(DATASET, SPLIT, labels, num_utts,
                      utterance_ordered, emotion2num, speaker_mode=None):
    diaids = list(utterance_ordered[SPLIT].keys())
    assert num_utts > 0

    input1 = []
    input0 = []
    labelnums = []

    for diaid in diaids:
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            labels, SPLIT, json_path, speaker_mode) for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        assert len(utterances) == len(emotions)

        for idx, (utterance, emotion) in enumerate(zip(utterances, emotions)):
            labelnums.append(emotion2num[emotion])
            input1.append(utterance)

            history = []
            start = idx - num_utts + 1
            end = idx
            for i in range(start, end):
                if i < 0:
                    history.append(' ')
                else:
                    history.append(utterances[i])
            history = ' '.join(history)
            input0.append(history)

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


def write_input_label_simple(DATASET, SPLIT, labels, utterance_ordered,
                             emotion2num, speaker_mode=None):
    samples = []
    diaids = list(utterance_ordered[SPLIT].keys())

    for diaid in diaids:
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            labels, SPLIT, json_path, speaker_mode) for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        assert len(utterances) == len(emotions)

        for utterance, emotion in zip(utterances, emotions):
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


def format_classification(DATASET, num_utts=1, speaker_mode=None):
    assert num_utts > 0

    if speaker_mode == 'none':
        speaker_mode = None
    emotion2num = get_emotion2num(DATASET)
    labels, utterance_ordered = load_labels_utt_ordered(DATASET)

    for SPLIT in tqdm(['train', 'val', 'test']):
        if num_utts == 1:
            write_input_label_simple(DATASET, SPLIT, labels, utterance_ordered,
                                     emotion2num, speaker_mode)
        else:
            write_input_label(DATASET, SPLIT, labels, num_utts,
                              utterance_ordered, emotion2num, speaker_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--DATASET', help=f"e.g. {DATASETS_SUPPORTED}")
    parser.add_argument('--num-utts', default=1, type=int,
                        help='e.g. 1, 2, 3, etc.')
    parser.add_argument('--speaker-mode', default=None,
                        help='e.g. title, upper, lower, none')

    args = parser.parse_args()
    args = vars(args)

    print(f"arguments given to {__file__}: {args}")

    if args['DATASET'] not in DATASETS_SUPPORTED:
        sys.exit(f"{args['DATASET']} is not supported!")

    format_classification(**args)
