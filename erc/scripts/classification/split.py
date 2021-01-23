import csv
import argparse
import random

TRAIN = 'train_train'
VAL = 'train_val'

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type=str,
                    help='path to csv file')
parser.add_argument('-ratio', type=float, default=0.8,
                    help='ratio of the train set (default: 0.8)')
parser.add_argument('-seed', type=int, default=3698,
                    help='random seed (default: 3698)')
parser.add_argument('-out', type=str, default='.',
                    help='directory to save the splits (default: .)')

args = parser.parse_args()

# Seed the random processes
random.seed(args.seed)

# Load CSV
lines = csv.reader(open(args.csv))
next(lines)
data = list(lines)

# Build class to image_fns dictionary
d = dict()
for fn, cl in data:
    d.setdefault(cl, [])
    d[cl].append(fn)

# Stratified split
splits = {
    TRAIN: dict(),
    VAL: dict(),
}
for cls_id, cls_list in d.items():
    train_sz = max(int(len(cls_list) * args.ratio), 1)
    shuffled = random.sample(cls_list, k=len(cls_list))
    splits[TRAIN][cls_id] = shuffled[:train_sz]
    splits[VAL][cls_id] = shuffled[train_sz:]

# Save split
for split, classes in splits.items():
    out = [['filename', 'category']]
    out.extend([
        [fn, cl]
        for cl, fns in classes.items()
        for fn in fns
    ])
    csv.writer(open(f'{args.out}/{split}.csv', 'w')).writerows(out)
