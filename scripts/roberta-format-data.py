import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import os
from glob import glob
import sys
DATASET_DIR = "Datasets/"


def main(DATASET):
    if DATASET not in ['MELD', 'IEMOCAP', 'AFEW', 'CAER']:
        sys.exit(f"{DATASET} is not supported!")

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

    np.random.seed(0)
    random.seed(0)

    with open(os.path.join(DATASET_DIR, DATASET, 'labels.json'), 'r') as stream:
        labels = json.load(stream)

    for SPLIT in tqdm(['train', 'val', 'test']):
        paths = glob(os.path.join(DATASET_DIR, DATASET,
                                  'raw-texts', SPLIT, '*.json'))
        random.shuffle(paths)
        samples = []
        for json_path in paths:
            with open(json_path, 'r') as stream:
                text = json.load(stream)
            uttid = os.path.basename(json_path).split('.json')[0]
            speaker = text['Speaker']
            utterance = text['Utterance']
            emotion = labels[SPLIT][uttid]

            utterance = speaker.upper() + ': ' + utterance

            samples.append((utterance, emotion2num[DATASET][emotion]))

        f1 = open(os.path.join(DATASET_DIR, DATASET,
                               'roberta', SPLIT + '.input0'), 'w')
        f2 = open(os.path.join(DATASET_DIR, DATASET,
                               'roberta', SPLIT + '.label'), 'w')

        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--dataset', help='e.g. IEMOCAP, MELD, AFEW, CAER')
    args = parser.parse_args()

    main(args.dataset)
