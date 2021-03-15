from glob import glob
import os
import json


def parse_emotion_dialogue(textpath):
    diaid = os.path.basename(textpath).split('.txt')[0]
    parsed = {}
    with open(textpath, 'r') as stream:
        to_parse = [line.strip() for line in stream.readlines()]

    indexes = []
    for idx, line in enumerate(to_parse):
        if diaid not in line:
            continue
        indexes.append(idx)

    # include the last line for computataion convenience
    indexes.append(len(to_parse))

    for idx_prev, idx_next in zip(indexes[:-1], indexes[1:]):
        for i in range(idx_prev, idx_next):
            line = to_parse[i]
            if diaid in line:
                uttid, voted = line.split('\t')[1], line.split('\t')[2]
                parsed[uttid] = {'voted': voted}
            if 'C-' in line:
                cx, label = line.split('\t')[0], line.split('\t')[1]
                cx = cx.split(':')[0]
                label = [lbl for lbl in label.split(';')]
                label = [lbl for lbl in label if len(lbl) != 0]
                label = [lbl.replace(" ", "") for lbl in label]
                parsed[uttid][cx] = label

    return diaid, parsed


text_paths = sorted(glob(
    'DEBUG/IEMOCAP_full_release/Session*/dialog/EmoEvaluation/Ses*.txt'))

print(f"{len(text_paths)} labeled dialogs found.")

labels = [parse_emotion_dialogue(textpath) for textpath in text_paths]
labels = {lbl[0]: lbl[1] for lbl in labels}

with open('IEMOCAP/utterance-ordered.json', 'r') as stream:
    utterance_ordered = json.load(stream)

labels = {SPLIT: {diaid: foo for diaid, foo in labels.items() if diaid in list(utterance_ordered[SPLIT].keys())}
          for SPLIT in ['train', 'val', 'test']}

for SPLIT in ['train', 'val', 'test']:
    num_utts = len([val_ for key, val in labels[SPLIT].items()
                    for val_ in val])
                    
    assert num_utts == len(
        [uttid for diaid, list_of_utts in utterance_ordered[SPLIT].items() for uttid in list_of_utts])

    print(
        f"{SPLIT} has {len(labels[SPLIT])} dialogues and "
        f"{len([val_ for key, val in labels[SPLIT].items() for val_ in val] )} utterances")


voted_labels = [bar['voted'] for SPLIT in ['train', 'val', 'test']
                for diaid, foo in labels[SPLIT].items() for _, bar in foo.items()]

print(f"There are {len(set(voted_labels))} unique labels.")

label_3 = {'ang', 'dis', 'exc', 'fea', 'fru',
           'hap', 'neu', 'oth', 'sad', 'sur', 'xxx'}

label_fullname = {'Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
                  'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise'}
label_fullname.add('Undecided')

label_map = {'ang': 'Anger',
             'dis': 'Disgust',
             'exc': 'Excited',
             'fea': 'Fear',
             'fru': 'Frustration',
             'hap': 'Happiness',
             'neu': 'Neutral',
             'oth': 'Other',
             'sad': 'Sadness',
             'sur': 'Surprise',
             'xxx': 'Undecided'}

assert set(list(label_map.keys())) == label_3
assert set(list(label_map.values())) == label_fullname


undecided = {SPLIT: {diaid: {uttid: {baz: qux for baz, qux in bar.items() if baz != 'voted'}
                             for uttid, bar in foo.items() if bar['voted'] == 'xxx'}
                     for diaid, foo in labels[SPLIT].items()}
             for SPLIT in ['train', 'val', 'test']}


# remove diaid.
undecided = {SPLIT: {uttid: bar for diaid, foo in undecided[SPLIT].items()
                     for uttid, bar in foo.items()}
             for SPLIT in ['train', 'val', 'test']}

with open('IEMOCAP/undecided.json', 'w') as stream:
    json.dump(undecided, stream, indent=4)

labels = {SPLIT: {diaid: {uttid: label_map[bar['voted']].lower()
                          for uttid, bar in foo.items()}
                  for diaid, foo in labels[SPLIT].items()}
          for SPLIT in ['train', 'val', 'test']}

# remove diaid.
labels = {SPLIT: {uttid: bar for diaid, foo in labels[SPLIT].items()
                  for uttid, bar in foo.items()}
          for SPLIT in ['train', 'val', 'test']}


with open('IEMOCAP/labels.json', 'w') as stream:
    json.dump(labels, stream, indent=4)

foo = []
for jsonpath in glob(f"IEMOCAP/raw-texts/*/*.json"):
    with open(jsonpath, 'r') as stream:
        text = json.load(stream)
    SPLIT = jsonpath.split('/')[2]
    uttid = os.path.basename(jsonpath).split('.json')[0]
    emotion = labels[SPLIT][uttid]
    text['Emotion'] = emotion
    foo.append(text)

    with open(jsonpath, 'w') as stream:
        json.dump(text, stream, indent=4, ensure_ascii=False)

with open('foo.json', 'w') as stream:
    json.dump(foo, stream, indent=4, ensure_ascii=False)


README = f"This dataset has all three modalities!\n"\
    f"Every utterance is part of a dialogue. If you also want to take the dialogue\n"\
    f"into consideration, see utterance-ordered.json\n"\
    f"One thing annoying about this dataset is that there are a lot of 'xxx' labels.\n"\
    f", which means they are 'Undecided' due to the labelers not agreeing on one thing.\n"\
    f"If you want to see the votes, see 'undecided.json'\n\n"\
    f"This README is written by Taewoon Kim (https://tae898.github.io/)"

with open(f"./IEMOCAP/README.txt", 'w') as stream:
    stream.write(README)
