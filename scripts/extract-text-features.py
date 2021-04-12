from fairseq.models.roberta import RobertaModel
import numpy as np
import json
from tqdm import tqdm
import os

DATASET_DIR = "Datasets/"
MODEL_DIR = "models/roberta.large/"


def load_labels_utt_ordered(DATASET):
    with open(os.path.join(DATASET_DIR, DATASET, 'labels.json'), 'r') as stream:
        labels = json.load(stream)

    with open(os.path.join(DATASET_DIR, DATASET, 'utterance-ordered.json'), 'r') as stream:
        utterance_ordered = json.load(stream)

    return labels, utterance_ordered


def get_emotion2num(DATASET):
    emotions = {}
    # MELD has 7 classes
    emotions['MELD'] = ['neutral',
                        'joy',
                        'surprise',
                        'anger',
                        'sadness',
                        'disgust',
                        'fear']

    # IEMOCAP originally has 11 classes but we'll only use 6 of them.
    emotions['IEMOCAP'] = ['neutral',
                           'frustration',
                           'sadness',
                           'anger',
                           'excited',
                           'happiness']

    # EmoryNLP has 7 classes
    emotions['EmoryNLP'] = ['neutral',
                            'joyful',
                            'scared',
                            'mad',
                            'peaceful',
                            'powerful',
                            'sad']

    # DailyDialog originally has 7 classes, but be sure not to include the
    # neutral class, which accounts for 80% of the data, in calculating
    # the micro f1_score.
    emotions['DailyDialog'] = ['neutral',
                               'happiness',
                               'surprise',
                               'sadness',
                               'anger',
                               'disgust',
                               'fear']

    emotion2num = {DATASET: {emotion: idx for idx, emotion in enumerate(
        emotions_)} for DATASET, emotions_ in emotions.items()}

    return emotion2num[DATASET]


def make_utterance(utterance, speaker, speaker_mode='title'):
    if speaker_mode == 'title':
        utterance = speaker.title() + ': ' + utterance
    elif speaker_mode == 'upper':
        utterance = speaker.upper() + ': ' + utterance
    elif speaker_mode == 'lower':
        utterance = speaker.lower() + ': ' + utterance

    return utterance


def get_uttid_speaker_utterance_emotion(DATASET, labels, SPLIT, json_path,
                                        speaker_mode=None):

    with open(json_path, 'r') as stream:
        text = json.load(stream)
    uttid = os.path.basename(json_path).split('.json')[0]
    if DATASET in ['MELD', 'EmoryNLP']:
        speaker = text['Speaker']
    elif DATASET == 'IEMOCAP':
        # speaker = {'Female': 'Alice', 'Male': 'Bob'}[text['Speaker']]
        sessid = text['SessionID']
        # https: // www.ssa.gov/oact/babynames/decades/century.html
        speaker = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                   'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                   'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                   'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                   'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}[sessid][text['Speaker']]

    elif DATASET == 'DailyDialog':
        speaker = {'A': 'Alex', 'B': 'Blake'}[text['Speaker']]
    else:
        raise ValueError(f"{DATASET} not supported!!!!!!")

    utterance = text['Utterance']
    emotion = labels[SPLIT][uttid]

    # very important here.
    utterance = make_utterance(utterance, speaker, speaker_mode)

    return uttid, speaker, utterance, emotion


def get_input_label_simple(DATASET, SPLIT, roberta, labels, utterance_ordered,
                           emotion2num, speaker_mode=None):
    max_tokens_input0 = 0
    samples = []
    diaids = list(utterance_ordered[SPLIT].keys())
    uttids_all = []

    for diaid in tqdm(diaids):
        uttids = utterance_ordered[SPLIT][diaid]
        json_paths = [os.path.join(
            DATASET_DIR, DATASET, 'raw-texts', SPLIT, uttid + '.json')
            for uttid in uttids]
        usue = [get_uttid_speaker_utterance_emotion(
            DATASET, labels, SPLIT, json_path, speaker_mode) for json_path in json_paths]

        utterances = [usue_[2] for usue_ in usue]
        emotions = [usue_[3] for usue_ in usue]

        assert len(utterances) == len(emotions) == len(uttids)

        for utterance, emotion, uttid in zip(utterances, emotions, uttids):

            if emotion not in list(emotion2num.keys()):
                continue

            uttids_all.append(uttid)

            utterance = utterance.strip()

            num_tokens0 = len(roberta.encode(utterance).tolist())
            max_tokens_input0 = max(max_tokens_input0, num_tokens0)
            samples.append((utterance, emotion2num[emotion]))

    print(f"max tokens in {DATASET}, {SPLIT} is {max_tokens_input0}")

    X = [sample[0] for sample in samples]
    Y = [sample[1] for sample in samples]

    return X, Y, uttids_all


def load_model(DATASET):
    if DATASET == 'MELD':
        method = '01-utt-SPEAKER'
    elif DATASET == 'IEMOCAP':
        method = '01-utt-SPEAKER-10-names'
    else:
        raise ValueError(f"{DATASET} not supported.")

    roberta = RobertaModel.from_pretrained(
        os.path.join(MODEL_DIR, f'{DATASET}/{method}/'),
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=os.path.join(MODEL_DIR, f'{DATASET}/{method}/bin')
    )
    roberta.eval()

    return roberta


class RobertaFeatures():
    def __init__(self, DATASET, num_utts, speaker_mode='upper', tokens_per_sample=512):

        self.DATASET = DATASET
        self.num_utts = num_utts
        self.speaker_mode = speaker_mode
        self.tokens_per_sample = tokens_per_sample
        self.emotion2num = get_emotion2num(DATASET)
        self.labels, self.utterance_ordered = load_labels_utt_ordered(DATASET)
        self.roberta = load_model(DATASET)

        self.X, self.Y, self.uttids = {}, {}, {}
        for SPLIT in ['train', 'val', 'test']:
            self.X[SPLIT], self.Y[SPLIT], self.uttids[SPLIT] = \
                get_input_label_simple(DATASET, SPLIT, self.roberta, self.labels, self.utterance_ordered,
                                       self.emotion2num, self.speaker_mode)

            assert len(self.X[SPLIT]) == len(
                self.Y[SPLIT]) == len(self.uttids[SPLIT])

    def get_features(self, SPLIT, idx):
        self.utterance = self.X[SPLIT][idx]
        self.label = int(self.Y[SPLIT][idx])
        self.uttid = self.uttids[SPLIT][idx]
        self.tokens = self.roberta.encode(self.utterance)
        self.logprobs = self.roberta.predict(
            self.DATASET + '_head', self.tokens)
        self.probs = np.exp(self.logprobs.detach().cpu().numpy())
        self.pred = int(self.logprobs.argmax(dim=1).detach().cpu().numpy()[0])
        self.logprobs = self.roberta.predict(
            self.DATASET + '_head', self.tokens).detach().cpu().numpy()

        self.features = self.roberta.extract_features(self.tokens)
        self.features = self.features[:, 0, :]  # <CLS> token feature

        #  dropout doesn't have an effect (i.e. p=0) but I'll still preserve it.
        self.features = \
            self.roberta.model.classification_heads[f'{DATASET}_head'].dropout(
                self.roberta.model.classification_heads[f'{DATASET}_head'].activation_fn(
                    self.roberta.model.classification_heads[f'{DATASET}_head'].dense(
                        self.roberta.model.classification_heads[f'{DATASET}_head'].dropout(self.features))))

        self.features = self.features.detach().cpu().numpy()


for DATASET in tqdm(['MELD', 'IEMOCAP']):
    rf = RobertaFeatures(DATASET=DATASET, num_utts=1,
                         speaker_mode='upper', tokens_per_sample=512)
    for SPLIT in tqdm(['train', 'val', 'test']):
        os.makedirs(os.path.join(DATASET_DIR, DATASET,
                                 'text-features', SPLIT), exist_ok=True)
        for idx in tqdm(range(len(rf.X[SPLIT]))):
            rf.get_features(SPLIT, idx)
            savepath = os.path.join(
                DATASET_DIR, DATASET, 'text-features', SPLIT, f"{rf.uttid}.npy")
            to_save = {'utterance': rf.utterance,
                       'label': rf.label,
                       'uttid': rf.uttid,
                       'tokens': rf.tokens,
                       'logprobs': rf.logprobs,
                       'probs': rf.probs,
                       'pred': rf.pred,
                       'features': rf.features,
                       'emotion2num': rf.emotion2num}

            np.save(savepath, to_save)
