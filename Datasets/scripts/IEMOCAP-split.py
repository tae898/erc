import json
import os
from glob import glob

with open('scripts/iemocap-DeCLaRe.json', 'r') as stream:
    declare = json.load(stream)

os.chdir('IEMOCAP')
os.chdir('raw-texts')
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)

for foo in glob('*.txt'):

    belongsto = None
    for bar in ['train', 'val', 'test']:
        if os.path.basename(foo).split('.txt')[0] in declare[bar]['video_id']:
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
        if os.path.basename(foo).split('.avi')[0] in declare[bar]['video_id']:
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
        if os.path.basename(foo) in declare[bar]['video_id']:
            belongsto = bar

    assert belongsto is not None, f"{foo}, {bar}"
        
    os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))