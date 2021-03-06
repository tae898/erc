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
MODEL_DIR = 'models/'


def make_markdown_table(array):
    """ Input: Python list with rows of table as lists
               First element as header. 
        Output: String to put into a .md file 

    Ex Input: 
        [["Name", "Age", "Height"],
         ["Jake", 20, 5'10],
         ["Mary", 21, 5'7]] 
    """

    markdown = "\n" + str("| ")

    for e in array[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += '|'
    for i in range(len(array[0])):
        markdown += str("-------------- | ")
    markdown += "\n"

    for entry in array[1:]:
        markdown += str("| ")
        for e in entry:
            to_add = str(e) + str(" | ")
            markdown += to_add
        markdown += "\n"

    return markdown + "\n"


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

        for idx, label in enumerate(tqdm(Y)):
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
        scores_all[SPLIT]['f1_micro'] = f1_score(
            y_true[SPLIT], y_pred[SPLIT], average='micro')
        scores_all[SPLIT]['f1_macro'] = f1_score(
            y_true[SPLIT], y_pred[SPLIT], average='macro')
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

    metrics = ['f1_weighted', 'f1_micro', 'f1_macro', 'accuracy']
    scores_all = {SPLIT: {metric: [] for metric in metrics}
                  for SPLIT in ['train', 'val', 'test']}

    for jsonpath in jsonpaths:
        with open(jsonpath, 'r') as stream:
            scores = json.load(stream)

        for SPLIT in ['train', 'val', 'test']:
            for metric in metrics:
                if metric in scores[SPLIT].keys():
                    scores_all[SPLIT][metric].append(scores[SPLIT][metric])

    for SPLIT in ['train', 'val', 'test']:
        for metric in metrics:
            if metric in scores[SPLIT].keys():
                scores_all[SPLIT][metric] = {
                    'mean': np.mean(np.array(scores_all[SPLIT][metric])),
                    'std': np.std(np.array(scores_all[SPLIT][metric]))}

    pprint.PrettyPrinter(indent=4).pprint(scores_all)

    with open(os.path.join(DIR_NAME, 'results.json'), 'w') as stream:
        json.dump(scores_all, stream, indent=4, ensure_ascii=False)


def leaderboard():
    results_paths = glob(os.path.join(MODEL_DIR, '*/*/*/results.json'))
    print(results_paths)

    DATASETS = sorted(list(set([path.split('/')[2]
                                for path in results_paths])))
    leaderboard = {DATASET: [] for DATASET in DATASETS}
    for path in results_paths:
        BASE_MODEL = path.split('/')[1]
        DATASET = path.split('/')[2]
        METHOD = path.split('/')[3]

        with open(path, 'r') as stream:
            results = json.load(stream)

        if DATASET == 'DailyDialog':
            metric = 'f1_micro'
        else:
            metric = 'f1_weighted'

        leaderboard[DATASET].append([BASE_MODEL, METHOD,
                                     round(results['train']
                                           [metric]['mean']*100, 3),
                                     round(results['val'][metric]
                                           ['mean']*100, 3),
                                     round(results['test'][metric]['mean']*100, 3)])

    with open('leaderboard.md', 'w') as stream:
        stream.write('# Leaderboard\n')

    for DATASET in DATASETS:
        leaderboard[DATASET].sort(key=lambda x: x[-1])
        table = leaderboard[DATASET]
        table.insert(0, ["base model", "method", "train", "val", "test"])

        table = make_markdown_table(table)
        with open('leaderboard.md', 'a') as stream:
            stream.write(f"## Dataset: {DATASET} \n")
            stream.write(f"The metric is {metric} (%)")
            stream.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the model on f1 and acc')
    parser.add_argument('--DATASET', default=None,
                        help='e.g. IEMOCAP, MELD, AFEW, CAER')
    parser.add_argument('--model-path', default=None, help='e.g. model path')
    parser.add_argument('--num-utt', default=0, type=int, help='e.g. 0, 1')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--evaluate-seeds', action='store_true')
    parser.add_argument('--leaderboard', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    print(f"arguments given to {__file__}: {args}")

    if args['evaluate_seeds']:
        evaluate_all_seeds(args['model_path'])
    elif args['leaderboard']:
        leaderboard()
    else:
        [args.pop(not_needed) for not_needed in
         ['evaluate_seeds', 'leaderboard']]
        evaluate_model(**args)
