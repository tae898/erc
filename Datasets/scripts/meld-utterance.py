import csv
import json

annotations = {}
for DATASET in ['train', 'dev', 'test']:
    with open(f'./DEBUG/MELD.Raw/{DATASET}_sent_emo_converted.csv') as f:
        reader = csv.reader(f)
        if DATASET == 'dev':
            DATASET = 'val'
        annotations[DATASET] = list(reader)

datasets = {}
labels = {}
for DATASET in ['train', 'val', 'test']:
    datasets[DATASET] = {}
    labels[DATASET] = {}
    for row in annotations[DATASET][1:]:
        SrNo, Utterance, Speaker, Emotion, Sentiment, Dialogue_ID,\
            Utterance_ID, Season, Episode, StartTime, EndTime = row

        to_save = ['SrNo', 'Utterance', 'Speaker', 'Emotion', 'Sentiment', 'Dialogue_ID',
                   'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime']

        to_dump = {'SrNo': SrNo,
                   'Utterance': Utterance,
                   'Speaker': Speaker,
                   'Emotion': Emotion,
                   'Sentiment': Sentiment,
                   'Dialogue_ID': Dialogue_ID,
                   'Utterance_ID': Utterance_ID,
                   'Season': Season,
                   'Episode': Episode,
                   'StartTime': StartTime,
                   'EndTime': EndTime}

        labels[DATASET][f"dia{Dialogue_ID}_utt{Utterance_ID}"] = Emotion.lower()

        with open(f"./MELD/raw-texts/{DATASET}/"
                  f"dia{Dialogue_ID}_utt{Utterance_ID}.json", 'w') as stream:
            json.dump(to_dump, stream, indent=4,
                      sort_keys=True, ensure_ascii=False)

    assert len(labels[DATASET]) == len(set(labels[DATASET]))

with open(f"./MELD/labels.json", 'w') as stream:
    json.dump(labels, stream, indent=4, sort_keys=True, ensure_ascii=False)

