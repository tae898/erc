from glob import glob
import os
import json

data = {SPLIT: {} for SPLIT in ['train', 'val', 'test']}

mapSPLIT = {'train': 'trn',
            'val': 'dev',
            'test': 'tst'}
for SPLIT in ['train', 'val', 'test']:
    filename = f"DEBUG/emotion-detection-emotion-detection-1.0/json/emotion-detection-{mapSPLIT[SPLIT]}.json"
    with open(filename, 'r') as stream:
        data[SPLIT] = json.load(stream)

diautts = {}
diautts_ordered = {}
for SPLIT in ['train', 'val', 'test']:
    diautts[SPLIT] = {}
    diautts_ordered[SPLIT] = {}
    episodes = data[SPLIT]['episodes']
    for episode in episodes:
        episode_id = episode['episode_id']
        scenes = episode['scenes']
        for scene in scenes:
            scene_id = scene['scene_id']
            diaid = scene_id
            diautts[SPLIT][diaid] = {}
            diautts_ordered[SPLIT][diaid] = []
            utterances = scene['utterances']
            for utterance in utterances:
                utterance_id = utterance['utterance_id']
                uttid = utterance_id
                speakers = utterance['speakers']
                speaker = speakers[0]
                transcript = utterance['transcript']
                tokens = utterance['tokens']
                emotion = utterance['emotion']

                diautts_ordered[SPLIT][diaid].append(uttid)

                diautts[SPLIT][diaid][uttid] = {}
                diautts[SPLIT][diaid][uttid]['episode_id'] = episode_id
                diautts[SPLIT][diaid][uttid]['scene_id'] = scene_id
                diautts[SPLIT][diaid][uttid]['diaid'] = diaid
                diautts[SPLIT][diaid][uttid]['uttid'] = uttid
                diautts[SPLIT][diaid][uttid]['Utterance'] = transcript
                diautts[SPLIT][diaid][uttid]['Speaker'] = speaker
                diautts[SPLIT][diaid][uttid]['Emotion'] = emotion.lower()

                # There are very few cases (0.56%) where there is more than
                # one speaker in an utterance. I'll just take the first in the
                # list as the speaker.

labels = {}
num_utts_total = 0
for SPLIT in ['train', 'val', 'test']:
    labels[SPLIT] = {}
    os.makedirs(f"EmoryNLP/raw-texts/{SPLIT}", exist_ok=True)
    diaids = list(diautts[SPLIT].keys())
    for diaid in diaids:
        uttids = list(diautts[SPLIT][diaid].keys())
        for uttid in uttids:
            utt = diautts[SPLIT][diaid][uttid]
            with open(f"EmoryNLP/raw-texts/{SPLIT}/{uttid}.json", 'w') as stream:
                json.dump(utt, stream, indent=4, ensure_ascii=False)
            labels[SPLIT][uttid] = utt['Emotion']
            num_utts_total += 1

assert num_utts_total == sum([1 for SPLIT in ['train', 'val', 'test']
                              for diaid, uttids 
                              in diautts_ordered[SPLIT].items() 
                              for uttid in uttids])
with open('EmoryNLP/labels.json', 'w') as stream:
    json.dump(labels, stream, indent=4, ensure_ascii=False)

with open('EmoryNLP/utterance-ordered.json', 'w') as stream:
    json.dump(diautts_ordered, stream, indent=4, ensure_ascii=False)


README = \
    f"There are very few cases (0.56%) where there is more than one speaker\n"\
    f"in an utterance. I'll just take the first in the list as the speaker.\n"\
    f"Every utterance is part of a dialogue. If you also want to take the dialogue\n"\
    f"into consideration, see utterance-ordered.json\n"\
    f"This README is written by Taewoon Kim (https://tae898.github.io/)"

with open(f"./EmoryNLP/README.txt", 'w') as stream:
    stream.write(README)
