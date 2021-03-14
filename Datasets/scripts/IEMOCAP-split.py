import json
import os
from glob import glob

dialogxl_videoids = {}
for SPLIT in ['train', 'val', 'test']:
    jsonpath = f'scripts/IEMOCAP-DialogXL/{SPLIT}_data.json'
    if SPLIT == 'val':
        jsonpath = f'scripts/IEMOCAP-DialogXL/dev_data.json'
    with open(jsonpath, 'r') as stream:
        dialogxl_videoids[SPLIT] = json.load(stream)

for SPLIT in ['train', 'val', 'test']:
    dialogxl_videoids[SPLIT] = \
        [f['speaker'] for fo in dialogxl_videoids[SPLIT] for f in fo]

for SPLIT in ['train', 'val', 'test']:
    dialogxl_videoids[SPLIT] = \
        sorted(list(set(['_'.join(foo.split('_')[:-1])
                         for foo in dialogxl_videoids[SPLIT]])))

os.chdir('IEMOCAP')
os.chdir('raw-texts')
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)

for foo in glob('*.txt'):

    belongsto = None
    for bar in ['train', 'val', 'test']:
        if os.path.basename(foo).split('.txt')[0] in dialogxl_videoids[bar]:
            belongsto = bar

    assert belongsto is not None

    os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))

os.chdir('../raw-videos')
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)

for foo in glob('*.avi'):

    belongsto = None
    for bar in ['train', 'val', 'test']:
        if os.path.basename(foo).split('.avi')[0] in dialogxl_videoids[bar]:
            belongsto = bar

    assert belongsto is not None

    os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))

os.chdir('../raw-audios')
for foo in glob('*'):
    os.makedirs('train', exist_ok=True)
    os.makedirs('val', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    belongsto = None
    for bar in ['train', 'val', 'test']:
        if os.path.basename(foo) in dialogxl_videoids[bar]:
            belongsto = bar

    assert belongsto is not None, f"{foo}, {bar}"

    os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))
