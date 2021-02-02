import json
from glob import glob
import os

datasets = {}
datasets['train'] = sorted(glob('./CAER/raw-videos/train/*.avi'))
datasets['val'] = sorted(glob('./CAER/raw-videos/val/*.avi'))
datasets['test'] = sorted(glob('./CAER/raw-videos/test/*.avi'))
labels = {}

for DATASET in ['train', 'val', 'test']:
    labels[DATASET] = {}
    for filename in datasets[DATASET]:

        basename = os.path.basename(filename).split('.avi')[0]

        labels[DATASET][str(basename)] = basename.split('-')[0].lower()

    print(
        f"{len(set([foo for foo in labels[DATASET]]))} labeled videos in {DATASET}")
    assert len(set([foo for foo in labels[DATASET]])) == len(
        [foo for foo in labels[DATASET]])

with open(f"./CAER/labels.json", 'w') as stream:
    json.dump(labels, stream, indent=4, sort_keys=True, ensure_ascii=False)

README = \
    f"This dataset doesn't have text modality. Maybe I'll do some ASR to get the texts.\n"
    
with open(f"./CAER/README.txt", 'w') as stream:
    stream.write(README)
