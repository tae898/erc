import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import os
from glob import glob
import sys
DATASET_DIR = "Datasets/"
np.random.seed(0)
random.seed(0)


def get_emotion2num(DATASET):
    emotions_MELD = ['anger',
                     'disgust',
                     'fear',
                     'joy',
                     'neutral',
                     'sadness',
                     'surprise']

    emotions_IEMOCAP = ['anger',
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

    emotion2num_MELD = {emotion: idx for idx,
                        emotion in enumerate(emotions_MELD)}
    emotion2num_IEMOCAP = {emotion: idx for idx,
                           emotion in enumerate(emotions_IEMOCAP)}

    emotion2num = {}
    emotion2num['MELD'] = emotion2num_MELD
    emotion2num['IEMOCAP'] = emotion2num_IEMOCAP

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

    # This line is a game changer. I got the inspiration from the movie
    # scripts where the character names are always upper-case.
    utterance = speaker.upper() + ': ' + utterance
    
    emotion = labels[SPLIT][uttid]

    return uttid, speaker, utterance, emotion


def concatenate_utterances(utterances, emotions, num_past_utterances,
                           predict_next):

    before = len(utterances)
    assert before == len(emotions)

    utterances_concat = []
    emotions_concat = []

    for i in range(len(utterances)):
        if predict_next:
            next = i+1
            if next == len(utterances):
                continue
            emotions_concat.append(emotions[next])
        else:
            emotions_concat.append(emotions[i])

        utt_prevs = [utterances[j] for j in range(i-num_past_utterances, i+1)]
        utt_prevs = ' '.join(utt_prevs)
        utterances_concat.append(utt_prevs)

    if predict_next:
        after = before - 1
    else:
        after = len(utterances_concat)
    assert after == len(emotions_concat)

    return utterances_concat, emotions_concat


def main(DATASET, num_past_utterances=0, predict_next=False):
    if DATASET not in ['MELD', 'IEMOCAP', 'AFEW', 'CAER']:
        sys.exit(f"{DATASET} is not supported!")
    num_past_utterances = int(num_past_utterances)
    emotion2num = get_emotion2num(DATASET)

    labels, utterance_ordered = load_labels_utt_ordered(DATASET)

    for SPLIT in tqdm(['train', 'val', 'test']):
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

            utterances, emotions = concatenate_utterances(
                utterances, emotions, num_past_utterances, predict_next)

            for utterance, emotion in zip(utterances, emotions):
                samples.append((utterance, emotion2num[emotion]))

        f1 = open(os.path.join(DATASET_DIR, DATASET,
                               'roberta', SPLIT + '.input0'), 'w')
        f2 = open(os.path.join(DATASET_DIR, DATASET,
                               'roberta', SPLIT + '.label'), 'w')

        # random.shuffle(samples)

        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--dataset', help='e.g. IEMOCAP, MELD, AFEW, CAER')
    parser.add_argument('--num-past-utterances',
                        default=0, help='e.g. 0, 1, 2')
    parser.add_argument('--predict-next', action='store_true')

    args = parser.parse_args()

    print(f"arguments given: {args}")

    main(args.dataset, args.num_past_utterances, args.predict_next)
