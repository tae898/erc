from glob import glob
import os
import json
import re

specials_1 = ["!", "%",  ")", ",", ".", ":", ";", "?",
              "’", "”", "′"]

specials_2 = ["\"", "#", "(", "@",
              "°", "‘", "“", "′", "’"]


def clean_utt(utt):
    utt = utt.strip()
    utt.replace('。', '.')

    for special in specials_1:
        utt = utt.replace(' ' + special, special)

    for special in specials_2:
        utt = utt.replace(special + ' ', special)

    return utt


def load_json(path):
    with open(path, 'r') as stream:
        foo = json.load(stream)

    return foo
# Here are some explanations about the files:

# 1) dialogues_text.txt: The DailyDialog dataset which contains 11,318 transcribed dialogues.
# 2) dialogues_topic.txt: Each line in dialogues_topic.txt corresponds to the topic of that in dialogues_text.txt.
#                         The topic number represents: {1: Ordinary Life, 2: School Life, 3: Culture & Education,
#                         4: Attitude & Emotion, 5: Relationship, 6: Tourism , 7: Health, 8: Work, 9: Politics, 10: Finance}
# 3) dialogues_act.txt: Each line in dialogues_act.txt corresponds to the dialog act annotations in dialogues_text.txt.
#                       The dialog act number represents: { 1: inform，2: question, 3: directive, 4: commissive }
# 4) dialogues_emotion.txt: Each line in dialogues_emotion.txt corresponds to the emotion annotations in dialogues_text.txt.
#                           The emotion number represents: { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
# 5) train.zip, validation.zip and test.zip are two different segmentations of the whole dataset.


dias = {SPLIT: {} for SPLIT in ['train', 'val', 'test']}
acts = {SPLIT: {} for SPLIT in ['train', 'val', 'test']}
emotions = {SPLIT: {} for SPLIT in ['train', 'val', 'test']}

mapSPLIT = {'train': 'train',
            'val': 'validation',
            'test': 'test'}

num2emotion = {0: 'neutral',
               1: 'anger',
               2: 'disgust',
               3: 'fear',
               4: 'happiness',
               5: 'sadness',
               6: 'surprise'}

num2act = {1: 'inform',
           2: 'question',
           3: 'directive',
           4: 'commissive'}


for SPLIT in ['train', 'val', 'test']:
    filename = f"DEBUG/ijcnlp_dailydialog/{mapSPLIT[SPLIT]}/dialogues_{mapSPLIT[SPLIT]}.txt"
    with open(filename, 'r') as stream:
        dias[SPLIT] = [line.strip() for line in stream.readlines()]

    dias[SPLIT] = [dia.split('__eou__')[:-1] for dia in dias[SPLIT]]

    filename = f"DEBUG/ijcnlp_dailydialog/{mapSPLIT[SPLIT]}/dialogues_act_{mapSPLIT[SPLIT]}.txt"
    with open(filename, 'r') as stream:
        acts[SPLIT] = [line.strip() for line in stream.readlines()]

    acts[SPLIT] = [[num2act[int(a)] for a in act.split()]
                   for act in acts[SPLIT]]

    filename = f"DEBUG/ijcnlp_dailydialog/{mapSPLIT[SPLIT]}/dialogues_emotion_{mapSPLIT[SPLIT]}.txt"
    with open(filename, 'r') as stream:
        emotions[SPLIT] = [line.strip() for line in stream.readlines()]

    emotions[SPLIT] = [[num2emotion[int(e)] for e in emotion.split()]
                       for emotion in emotions[SPLIT]]

diautts_ordered = {}
diacount = 0
uttcount = 0
labels = {}
for SPLIT in ['train', 'val', 'test']:
    labels[SPLIT] = {}
    os.makedirs(f"DailyDialog/raw-texts/{SPLIT}", exist_ok=True)
    diautts_ordered[SPLIT] = {}
    assert len(dias[SPLIT]) == len(acts[SPLIT]) == len(emotions[SPLIT])
    for dia, act, emotion in zip(dias[SPLIT], acts[SPLIT], emotions[SPLIT]):
        diaid = 'dia_' + str(diacount).zfill(5)
        diautts_ordered[SPLIT][diaid] = []
        assert len(dia) == len(act) == len(emotion)
        for idx, (utt, ac, em) in enumerate(zip(dia, act, emotion)):
            if idx % 2 == 0:
                speaker = 'A'
            else:
                speaker = 'B'
            utt = clean_utt(utt)
            uttid = 'utt_' + str(uttcount).zfill(6)
            with open(f"DailyDialog/raw-texts/{SPLIT}/{uttid}.json", 'w') as stream:
                json.dump({'Utterance': utt, 'Act': ac, 'Speaker': speaker,
                           'Emotion': em}, stream, indent=4, ensure_ascii=False)
            diautts_ordered[SPLIT][diaid].append(uttid)
            labels[SPLIT][uttid] = em
            uttcount += 1
        diacount += 1

assert sum([len(labels[SPLIT])
            for SPLIT in ['train', 'val', 'test']]) == uttcount

with open('DailyDialog/utterance-ordered.json', 'w') as stream:
    json.dump(diautts_ordered, stream, indent=4, ensure_ascii=False)

with open('DailyDialog/labels.json', 'w') as stream:
    json.dump(labels, stream, indent=4, ensure_ascii=False)

README = \
    f"There are no speaker label. I assume that the two speakers are taking turns.\n"\
    f"That is, every utterance spoken by speaker A is followed by another utterance\n"\
    f"followed by speaker B.\n\n"\
    f"This data has too many redundant white spaces before and after some special characters\n"\
    f", such as {specials_1} and {specials_2}. Many of them were removed.\n"\
    f"This README is written by Taewoon Kim (https://tae898.github.io/)"

with open(f"./DailyDialog/README.txt", 'w') as stream:
    stream.write(README)
