from sklearn.metrics import (
    f1_score,
    accuracy_score,
)
from tqdm import tqdm
import os
from glob import glob
import argparse
from fairseq.models.roberta import RobertaModel
import sys
import pprint
import json
import numpy as np
DATASET_DIR = "Datasets/"


def evaluate_model(DATASET, model_path, num_utt, use_cuda):
    if DATASET not in ['MELD', 'IEMOCAP', 'AFEW', 'CAER']:
        sys.exit(f"{DATASET} is not supported!")

    roberta = RobertaModel.from_pretrained(
        os.path.dirname(model_path),
        checkpoint_file=os.path.basename(model_path),
        data_name_or_path=os.path.join(DATASET_DIR, DATASET, 'roberta/bin')
    )

    def label_fn(label): return roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    roberta.eval()  # disable dropout
    if use_cuda:
        roberta.cuda()

    y_true = {}
    y_pred = {}

    for SPLIT in tqdm(['train', 'val', 'test']):

        y_true[SPLIT] = []
        y_pred[SPLIT] = []

        X = {}
        for i in range(num_utt):
            X[i] = os.path.join(DATASET_DIR, DATASET,
                                'roberta', SPLIT + f'.input{i}')

            with open(X[i], 'r') as stream:
                X[i] = [line.strip() for line in stream.readlines()]

        Y = os.path.join(DATASET_DIR, DATASET, 'roberta', SPLIT + '.label')
        with open(Y, 'r') as stream:
            Y = [line.strip() for line in stream.readlines()]

        for idx, label in enumerate(tqdm(Y[:100])):
            to_encode = [X[i][idx] for i in range(num_utt)]
            # print(to_encode)
            tokens = roberta.encode(*to_encode)
            # print(tokens)
            pred = label_fn(roberta.predict(
                DATASET + '_head', tokens).argmax().item())

            y_true[SPLIT].append(label)
            y_pred[SPLIT].append(pred)

    scores_all = {}
    for SPLIT in ['train', 'val', 'test']:
        scores_all[SPLIT] = {}
        scores_all[SPLIT]['f1_weighted'] = f1_score(
            y_true[SPLIT], y_pred[SPLIT], average='weighted')
        scores_all[SPLIT]['accuracy'] = accuracy_score(
            y_true[SPLIT], y_pred[SPLIT])

    pprint.PrettyPrinter(indent=4).pprint(scores_all)

    with open(model_path.replace('.pt', '.json'), 'w') as stream:
        json.dump(scores_all, stream, indent=4, ensure_ascii=False)


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def evaluate_all_seeds(model_path):
    DIR_NAME = os.path.dirname(model_path)
    jsonpaths = [path for path in glob(os.path.join(DIR_NAME, '*.json'))
                 if hasNumbers(os.path.basename(path))]

    scores_all = {SPLIT: {metric: [] for metric in ['f1_weighted', 'accuracy']}
                  for SPLIT in ['train', 'val', 'test']}

    for jsonpath in jsonpaths:
        with open(jsonpath, 'r') as stream:
            scores = json.load(stream)

        for SPLIT in ['train', 'val', 'test']:
            for metric in ['f1_weighted', 'accuracy']:
                scores_all[SPLIT][metric].append(scores[SPLIT][metric])

    for SPLIT in ['train', 'val', 'test']:
        for metric in ['f1_weighted', 'accuracy']:
            scores_all[SPLIT][metric] = {
                'mean': np.mean(np.array(scores_all[SPLIT][metric])),
                'std': np.std(np.array(scores_all[SPLIT][metric]))}

    pprint.PrettyPrinter(indent=4).pprint(scores_all)

    with open(os.path.join(DIR_NAME, 'results.json'), 'w') as stream:
        json.dump(scores_all, stream, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--DATASET', help='e.g. IEMOCAP, MELD, AFEW, CAER')
    parser.add_argument('--model-path', help='e.g. model path')
    parser.add_argument('--num-utt', default=0, type=int, help='e.g. 0, 1')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--evaluate-seeds', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    print(f"arguments given to {__file__}: {args}")

    if args['evaluate_seeds']:
        evaluate_all_seeds(args['model_path'])
    else:
        args.pop('evaluate_seeds')
        evaluate_model(**args)
