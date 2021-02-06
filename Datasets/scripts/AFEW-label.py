import json
from glob import glob
import os

datasets = {}
datasets['train'] = sorted(glob('./DEBUG/AFEW/Train/*/*.avi'))
datasets['val'] = sorted(glob('./DEBUG/AFEW/Val/*/*.avi'))
datasets['test'] = sorted(glob('./DEBUG/AFEW/Test/Test_vid_Distribute/*.avi'))
labels = {}

for DATASET in ['train', 'val', 'test']:
    labels[DATASET] = {}
    for filename in datasets[DATASET]:
        if DATASET == 'test':
            continue

        lbl = filename.split('/')[-2]
        basename = os.path.basename(filename).split('.avi')[0]

        labels[DATASET][str(basename)] = lbl.lower()

    print(
        f"{len(set([foo for foo in labels[DATASET]]))} labeled videos in {DATASET}")
    assert len(set([foo for foo in labels[DATASET]])) == len(
        [foo for foo in labels[DATASET]])

with open(f"./AFEW/labels.json", 'w') as stream:
    json.dump(labels, stream, indent=4, sort_keys=True, ensure_ascii=False)

README = \
    f"This dataset doesn't have text modality. Maybe I'll do some ASR to get the texts.\n"\
    f"What's more annoying is that the test dataset is not labeled ...\n\n"\
    f"This README is written by Taewoon Kim (https://tae898.github.io/)"

with open(f"./AFEW/README.txt", 'w') as stream:
    stream.write(README)
