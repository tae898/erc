from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from glob import glob
import argparse
from fairseq.models.roberta import RobertaModel
import sys
DATASET_DIR = "Datasets/"


def main(DATASET, use_cuda):
    if DATASET not in ['MELD', 'IEMOCAP', 'AFEW', 'CAER']:
        sys.exit(f"{DATASET} is not supported!")

    roberta = RobertaModel.from_pretrained(
        'models/roberta.base',
        checkpoint_file=f'{DATASET}.pt',
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

        X = os.path.join(DATASET_DIR, DATASET, 'roberta', SPLIT + '.input0')
        with open(X, 'r') as stream:
            X = [line.strip() for line in stream.readlines()]

        Y = os.path.join(DATASET_DIR, DATASET, 'roberta', SPLIT + '.label')
        with open(Y, 'r') as stream:
            Y = [line.strip() for line in stream.readlines()]

        for line, label in tqdm(zip(X, Y)):
            tokens = roberta.encode(line)
            pred = label_fn(roberta.predict(
                DATASET + '_head', tokens).argmax().item())

            y_true[SPLIT].append(label)
            y_pred[SPLIT].append(pred)

    for SPLIT in ['train', 'val', 'test']:
        score = f1_score(y_true[SPLIT], y_pred[SPLIT], average='weighted')
        print(f"Weighed f1 score on {DATASET}, {SPLIT}: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data for roberta')
    parser.add_argument('--dataset', help='e.g. IEMOCAP, MELD, AFEW, CAER')
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.use_cuda)
