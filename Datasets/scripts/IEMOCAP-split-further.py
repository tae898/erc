import json
import os
from glob import glob

os.chdir('IEMOCAP/raw-texts')
weird = 0
durations = {'train': {}, 'val': {}, 'test': {}}
diautt_ordered = {'train': {}, 'val': {}, 'test': {}}
for foo in glob('*/*.txt'):
    DATASET = foo.split('/')[0]
    dia = foo.split('/')[1].split('.txt')[0]
    os.makedirs(os.path.join(DATASET, dia), exist_ok=True)
    durations[DATASET][dia] = {}
    diautt_ordered[DATASET][dia] = []

    with open(foo, 'r') as stream:
        lines = [line.strip() for line in stream.readlines()]
    for line in lines:
        tokens = line.split(' ')

        try:
            utterance_id = tokens[0]
            assert '_' in utterance_id
            assert 'XX' not in utterance_id
            speaker = utterance_id.split('_')[-1]
            speaker = ''.join([i for i in speaker if not i.isdigit()])
            dur = tokens[1]
            dur = dur.split(':')[0]
            dur = dur.split('-')
            start = float(dur[0].split('[')[1])
            end = float(dur[1].split(']')[0])

            durations[DATASET][dia][utterance_id] = (start, end)
            utterance = ' '.join(tokens[2:])
            print(utterance)
            diautt_ordered[DATASET][dia].append(utterance_id)

            to_dump = {'Utterance': utterance,
                       'StartTime': start, 
                       'EndTime': end,
                       'Speaker': {'M': 'Male', 'F':'Female'}[speaker]}
            with open(os.path.join(DATASET, dia, utterance_id + '.json'), 'w') as stream:
                json.dump(to_dump, stream, indent=4)

        except:
            weird += 1
            pass

    os.remove(foo)
os.chdir('../')

diautt_ordered_ = {}

for DATASET in ['train', 'val', 'test']:
    diautt_ordered_[DATASET] = {}
    for dia, utts in diautt_ordered[DATASET].items():
        diautt_ordered_[DATASET][dia] = []

        for utt in utts:
            path_to_check = os.path.join(
                f'raw-audios/{DATASET}/{dia}/{utt}.wav')
            print(path_to_check)
            if os.path.isfile(path_to_check):
                diautt_ordered_[DATASET][dia].append(utt)


for DATASET in ['train', 'val', 'test']:
    for dia, utts in diautt_ordered_[DATASET].items():
        assert len(utts) == len(glob(f"raw-audios/{DATASET}/{dia}/*.wav"))

with open('../IEMOCAP/utterance-ordered.json', 'w') as stream:
    json.dump(diautt_ordered_, stream, indent=4, sort_keys=True)

print(f"There are in total of {weird} weird utterances. It's fine I went "
      f"through them. Nothing serious")
