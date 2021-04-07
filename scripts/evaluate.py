from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from glob import glob
import argparse
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import sys
import pprint
import json
import numpy as np
import random
DATASET_DIR = "Datasets/"
MODEL_DIR = 'models/'
DATASETS_SUPPORTED = ['MELD', 'IEMOCAP', 'EmoryNLP', 'DailyDialog']


def add_markdown_sota(sota_values):
    markdown = str("| ")
    markdown += str(sota_values['model']) + " | "
    markdown += "SOTA" + " | "
    markdown += " " + " | "
    markdown += " " + " | "
    markdown += str(sota_values['value']) + " |"
    markdown += "\n"

    return markdown


def make_markdown_table(array):
    """ Input: Python list with rows of table as lists
               First element as header.
        Output: String to put into a .md file

    Ex Input:
        [["Name", "Age", "Height"],
         ["Jake", 20, 5'10],
         ["Mary", 21, 5'7]]
    """

    markdown = "\n" + "| "

    for e in array[0]:
        to_add = " " + str(e) + " |"
        markdown += to_add
    markdown += "\n"

    markdown += '|'
    for i in range(len(array[0])):
        markdown += "-------------- | "
    markdown += "\n"

    for entry in array[1:]:
        markdown += "| "
        for e in entry:
            to_add = str(e) + " | "
            markdown += to_add
        markdown += "\n"

    # if len(array) > 1:
    #     markdown += "| "
    #     for e in array[-1]:
    #         to_add = "**" + str(e) + "**" + " |"
    #         markdown += to_add
    #     markdown += "\n"

    return markdown


def list_duplicates_of(list_of_items):
    groups = []
    groups.append(0)

    for idx, (prev, next) in enumerate(zip(list_of_items[:-1], list_of_items[1:])):
        if prev != next:
            groups.append(idx+1)
    groups.append(len(list_of_items))

    groups = [[k for k in range(i, j)]
              for i, j in zip(groups[:-1], groups[1:])]

    return groups


def get_scores_all(DATASET, X, Y, y_true, y_pred, cross_entropy_loss_all,
                   probs_all, num_utts, score_pooling):

    if DATASET == 'DailyDialog':
        LABELS_FOR_EVAL = ['1', '2', '3', '4', '5', '6']
    else:
        LABELS_FOR_EVAL = None

    scores_all = {}

    if num_utts == -1:
        X_1_unrolled = [bar for foo in X[1] for bar in foo]
        assert len(X_1_unrolled) == len(y_true) == len(probs_all)

        groups = list_duplicates_of(X_1_unrolled)

        y_true_pooled = []
        y_pred_pooled = []
        probs_all_pooled = []
        cross_entropy_loss_all_pooled = []

        for group in groups:
            assert len(set([y_true[idx] for idx in group])) == 1
            probs = [probs_all[idx] for idx in group]

            if score_pooling == 'max':
                idx = np.argmax(np.max(np.array(probs), axis=1))
                idx = group[idx]
                y_true_pooled.append(y_true[idx])
                y_pred_pooled.append(y_pred[idx])
                probs_all_pooled.append(probs_all[idx])
                cross_entropy_loss_all_pooled.append(
                    cross_entropy_loss_all[idx])
            else:
                prob = np.mean(np.array(probs), axis=0)
                y_true_pooled.append(int(y_true[group[0]]))
                y_pred_pooled.append(int(np.argmax(prob)))
                probs_all_pooled.append(prob)
                cross_entropy_loss_all_pooled.append(
                    np.mean([cross_entropy_loss_all[idx] for idx in group]))

        scores_all['f1_weighted'] = f1_score(
            y_true_pooled, y_pred_pooled, labels=LABELS_FOR_EVAL,
            average='weighted')
        scores_all['f1_micro'] = f1_score(
            y_true_pooled, y_pred_pooled, labels=LABELS_FOR_EVAL,
            average='micro')
        scores_all['f1_macro'] = f1_score(
            y_true_pooled, y_pred_pooled, labels=LABELS_FOR_EVAL,
            average='macro')
        scores_all['cross_entropy_loss'] = np.mean(
            cross_entropy_loss_all_pooled)

    else:
        scores_all['f1_weighted'] = f1_score(
            y_true, y_pred, labels=LABELS_FOR_EVAL, average='weighted')
        scores_all['f1_micro'] = f1_score(
            y_true, y_pred, labels=LABELS_FOR_EVAL, average='micro')
        scores_all['f1_macro'] = f1_score(
            y_true, y_pred, labels=LABELS_FOR_EVAL, average='macro')
        scores_all['cross_entropy_loss'] = np.mean(cross_entropy_loss_all)

    scores_all = {key: float(val) for key, val in scores_all.items()}

    return scores_all


def evalute_SPLIT(roberta, DATASET, batch_size, num_utts, score_pooling, SPLIT):
    def label_fn(label):
        return roberta.task.label_dictionary.string(
            [label + roberta.task.label_dictionary.nspecial])

    y_true = []
    y_pred = []

    X = {}
    num_inputs = len(glob(os.path.join(DATASET_DIR, DATASET, 'roberta',
                                       f"{SPLIT}.input*.bpe")))
    for i in range(num_inputs):
        X[i] = os.path.join(DATASET_DIR, DATASET,
                            'roberta', SPLIT + f'.input{i}')

        with open(X[i], 'r') as stream:
            X[i] = [line.strip() for line in stream.readlines()]

    Y = os.path.join(DATASET_DIR, DATASET, 'roberta', SPLIT + '.label')
    with open(Y, 'r') as stream:
        Y = [line.strip() for line in stream.readlines()]

    # to avoid OOM
    if num_inputs == 1:
        XY = list(zip(X[0], Y))
        random.shuffle(XY)

        X[0], Y = zip(*XY)

    elif num_inputs == 2:
        XY = list(zip(X[0], X[1], Y))
        random.shuffle(XY)

        X[0], X[1], Y = zip(*XY)
    else:
        raise ValueError("something is wrong")

    for i in range(num_inputs):
        assert len(X[i]) == len(Y)

    original_length = len(Y)
    num_batches = original_length // batch_size

    for i in range(num_inputs):
        X[i] = [X[i][j*batch_size:(j+1)*batch_size] for j in range(num_batches)] + \
            [[X[i][j]] for j in range(batch_size*num_batches, original_length)]

    Y = [Y[j*batch_size:(j+1)*batch_size] for j in range(num_batches)] + \
        [[Y[j]] for j in range(batch_size*num_batches, original_length)]

    for i in range(num_inputs):
        assert len(X[i]) == len(Y)

    cross_entropy_loss_all = []
    probs_all = []
    for idx in tqdm(range(len(Y))):
        batch = [X[i][idx] for i in range(num_inputs)]
        batch = list(map(list, zip(*batch)))

        batch = collate_tokens(
            [roberta.encode(*[sequence for sequence in sequences])
             for sequences in batch], pad_idx=1
        )

        logprobs = roberta.predict(DATASET + '_head', batch)
        probs = np.exp(logprobs.detach().cpu().numpy())
        pred = logprobs.argmax(dim=1)
        label = Y[idx]

        assert len(logprobs) == len(label) == len(probs)
        for lp, l, p in zip(logprobs, label, probs):
            cel = -lp[int(l)]
            cross_entropy_loss_all.append(cel.detach().cpu().numpy())
            probs_all.append(p)

        assert len(pred) == len(label)
        for p, l in zip(pred, label):
            y_true.append(l)
            y_pred.append(label_fn(p))

    assert original_length == len(y_true) == len(y_pred) == \
        len(cross_entropy_loss_all) == len(probs_all), \
        f"{original_length}, {len(y_true)}, {len(y_pred)}, " \
        f"{len(cross_entropy_loss_all)}, {len(probs_all)}"

    scores_all = get_scores_all(DATASET, X, Y, y_true, y_pred, cross_entropy_loss_all,
                                probs_all, num_utts, score_pooling)

    return scores_all


def evaluate_model(DATASET, seed, checkpoint_dir, base_dir, metric,
                   batch_size, use_cuda, num_utts, score_pooling, keep_the_best,
                   **kwargs):
    if DATASET not in DATASETS_SUPPORTED:
        sys.exit(f"{DATASET} is not supported!")

    if metric.lower() not in ['f1_weighted', 'f1_micro', 'f1_macro',
                              'cross_entropy_loss']:
        raise ValueError(f"{metric} not supported!!")

    if metric.lower() == 'cross_entropy_loss' and num_utts != -1:
        model_paths = glob(os.path.join(checkpoint_dir, 'checkpoint_best.pt'))
    else:
        model_paths = glob(os.path.join(checkpoint_dir, '*.pt'))
        model_paths = [path for path in model_paths if os.path.basename(
            path) not in ['checkpoint_last.pt', 'checkpoint_best.pt']]

    stats = {}
    for model_path in tqdm(model_paths):
        checkpoint_file = os.path.basename(model_path)
        print(checkpoint_file)

        roberta = RobertaModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=os.path.join(DATASET_DIR, DATASET, 'roberta/bin')
        )

        roberta.eval()  # disable dropout
        if use_cuda:
            roberta.cuda()
        SPLIT = 'val'
        print(f"evaluating {DATASET}, {model_path}, {SPLIT} ...")
        scores = evalute_SPLIT(roberta, DATASET,
                               batch_size, num_utts, score_pooling, SPLIT=SPLIT)
        print(model_path)
        pprint.PrettyPrinter(indent=4).pprint(scores)
        stats[model_path] = scores

        del roberta

    pprint.PrettyPrinter(indent=4).pprint(stats)

    stats = {key: val[metric] for key, val in stats.items()}

    if metric.lower() == 'cross_entropy_loss':
        best_model_path = min(stats, key=stats.get)
    else:
        best_model_path = max(stats, key=stats.get)

    print(f"{best_model_path} has the best {metric} performance on val")

    checkpoint_file = os.path.basename(best_model_path)
    roberta = RobertaModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=os.path.join(DATASET_DIR, DATASET, 'roberta/bin')
    )

    roberta.eval()  # disable dropout
    if use_cuda:
        roberta.cuda()

    stats = {}
    for SPLIT in tqdm(['train', 'val', 'test']):
        print(f"evaluating {DATASET}, {best_model_path}, {SPLIT} ...")
        scores = evalute_SPLIT(roberta, DATASET,
                               batch_size, num_utts, score_pooling, SPLIT=SPLIT)

        stats[SPLIT] = scores
    pprint.PrettyPrinter(indent=4).pprint(stats)

    del roberta

    with open(os.path.join(base_dir, f"{seed}.json"),  'w') as stream:
        json.dump(stats, stream, indent=4, ensure_ascii=False)

    for model_path in glob(os.path.join(checkpoint_dir, '*.pt')):
        if os.path.basename(best_model_path) != os.path.basename(model_path):
            os.remove(model_path)

    if keep_the_best:
        jsons_saved = [path for path in glob(os.path.join(base_dir, '*.json'))
                       if os.path.basename(path) != 'results.json']
        seeds_all = {}
        for path in jsons_saved:
            with open(path, 'r') as stream:
                results_seed = json.load(stream)
            seeds_all[os.path.basename(path)] = results_seed

        if DATASET == 'DailyDialog':
            metric = 'f1_micro'
        else:
            metric = 'f1_weighted'

        seeds_all = {key: val['test'][metric]
                     for key, val in seeds_all.items()}

        best_seed = max(seeds_all, key=seeds_all.get)

        if best_seed == f"{seed}.json":
            os.rename(best_model_path, os.path.join(
                base_dir, os.path.basename(best_model_path)))
    else:
        os.remove(best_model_path)


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def evaluate_all_seeds(base_dir):
    DIR_NAME = base_dir
    jsonpaths = [path for path in glob(os.path.join(DIR_NAME, '*.json'))
                 if hasNumbers(os.path.basename(path))]

    metrics = ['f1_weighted', 'f1_micro', 'f1_macro', 'cross_entropy_loss']
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
    with open('scripts/sota.json', 'r') as stream:
        sota = json.load(stream)
    results_paths = glob(os.path.join(MODEL_DIR, '*/*/*/results.json'))
    print(results_paths)

    leaderboard = {DATASET: [] for DATASET in DATASETS_SUPPORTED}
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

    with open('LEADERBOARD.md', 'w') as stream:
        stream.write('# Leaderboard\n')
        stream.write("Note that only DailyDialog uses a different metric "
                     "(f1_micro) from others (f1_weighted). f1_micro is the "
                     "same as accuracy when every data point is assigned only "
                     "one class.\n\nThe reported performance of my models are "
                     "the mean values of the 5 random seed runs. I expect the "
                     "other authors have done the same thing or something "
                     "similar, since the numbers are stochastic in nature.\n\n"
                     "Since the distribution of classes is different for every "
                     "dataset and train / val / tests splits, and also not all "
                     "datasets have the same performance metric, the optimization "
                     "is done to minimize the validation cross entropy loss, "
                     "since its the most generic metric, "
                     "with backpropagation on training data split.\n\n")

        stream.write("As for DailyDialog, the neutral class, which accounts "
                     "for 80% of the data, is not included in the f1_score "
                     "calcuation. Note that they are still used in training.\n\n")

    for DATASET in DATASETS_SUPPORTED:

        if DATASET == 'DailyDialog':
            metric = 'f1_micro'
        else:
            metric = 'f1_weighted'

        leaderboard[DATASET].sort(key=lambda x: x[1])
        leaderboard[DATASET].sort(key=lambda x: x[0])

        table = leaderboard[DATASET]
        table.insert(0, ["base model", "method", "train", "val", "test"])

        with open('LEADERBOARD.md', 'a') as stream:
            table = make_markdown_table(table)

            stream.write(f"## {DATASET} \n")
            stream.write(f"The metric is {metric} (%)")
            stream.write(table)
            table = add_markdown_sota(sota[DATASET])
            stream.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the model on f1 and acc')
    parser.add_argument('--DATASET', default=None,
                        help='e.g. IEMOCAP, MELD, EmoryNLP, DailyDialog')
    parser.add_argument('--seed', type=int, default=None, help='e.g. SEED num')
    parser.add_argument('--model-path', default=None, help='e.g. model path')
    parser.add_argument('--batch-size', default=1,
                        type=int, help='e.g. 1, 2, 4')
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--metric', default='f1_weighted')
    parser.add_argument('--evaluate-seeds', action='store_true')
    parser.add_argument('--leaderboard', action='store_true')
    parser.add_argument('--num-utts', type=int, help='e.g. 1, 2, 4')
    parser.add_argument('--score-pooling', help='mean or max')
    parser.add_argument('--keep-the-best', default='false',
                        help='true or false')
    parser.add_argument('--num-gpus', default=0, help='e.g. 0, 1, 2 ...')

    args = parser.parse_args()
    args = vars(args)

    args['keep_the_best'] = {'true': True,
                             'false': False}[args['keep_the_best']]

    if int(args['num_gpus']) > 0:
        args['use_cuda'] = True
    else:
        args['use_cuda'] = False
    del args['num_gpus']

    print(f"arguments given to {__file__}: {args}")

    if args['evaluate_seeds']:
        evaluate_all_seeds(args['base_dir'])
    elif args['leaderboard']:
        leaderboard()
    else:
        evaluate_model(**args)
