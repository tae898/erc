import json
import yaml
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from .common import get_num_classes
from .dataset import ErcTextDataset
import numpy as np
from glob import glob
from pprint import pprint
import os
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def read_json(path):
    with open(path, 'r') as stream:
        foo = json.load(stream)
    return foo


def read_yaml(path):
    with open(path, 'r') as stream:
        foo = yaml.load(stream)
    return foo


def gets_DATASET_kwargs(model_checkpoint):
    DATASET = model_checkpoint.split('/')[1]

    kwargs_path = f"{'/'.join(model_checkpoint.split('/')[:-3])}/kwargs.yaml"
    kwargs = read_yaml(kwargs_path)
    kwargs

    return DATASET, kwargs


def get_tokenizer_model_ds(DATASET, kwargs, model_checkpoint, SPLIT='train'):
    NUM_CLASSES = get_num_classes(DATASET)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        model_checkpoint)

    model = RobertaForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=NUM_CLASSES)

    model.eval()
    model.cpu()

    SEED = int(model_checkpoint.split('/')[-3])

    if kwargs['ADD_BOU_EOU']:
        kwargs['ADD_BOU'] = True
        kwargs['ADD_EOU'] = True
    else:
        kwargs['ADD_BOU'] = False
        kwargs['ADD_EOU'] = False

    ds = ErcTextDataset(DATASET=DATASET, SPLIT=SPLIT,
                        num_past_utterances=kwargs['num_past_utterances'], num_future_utterances=kwargs['num_future_utterances'],
                        model_checkpoint=f"{'/'.join(model_checkpoint.split('/')[:-3])}/tokenizer",
                        ADD_BOU_EOU=kwargs[
                            'ADD_BOU_EOU'], ADD_SPEAKER_TOKENS=f"{'/'.join(model_checkpoint.split('/')[:-3])}/tokenizer/added_tokens.json",
                        REPLACE_NAMES_IN_UTTERANCES=kwargs['REPLACE_NAMES_IN_UTTERANCES'],
                        ROOT_DIR='multimodal-datasets/', SEED=SEED)

    return tokenizer, model, ds


def get_diaid_uttid_utts(ds, tokenizer, idx=None):
    if idx is None:
        idx = np.random.randint(0, len(ds))
    uttid = ds.uttids[idx]
    diaid = None

    for diaid_, uttids_ in ds.utterance_ordered.items():
        for uttid_ in uttids_:
            if uttid_ == uttid:
                diaid = diaid_

    utts = {foo.split('/')[-1].split('.json')[0]: f"({read_json(foo)['Emotion']}) " +
            f"<{read_json(foo)['Speaker']}>" + read_json(foo)['Utterance']
            for foo in glob(f'./multimodal-datasets/{ds.DATASET}/raw-texts/{ds.SPLIT}/{diaid}_*.json')}
    # utts_keys = [foo[0] for foo in sorted([(foo, int(foo.split('utt')[1])) for foo in list(utts.keys())], key=lambda x:x[1])]
    utts = {key: utts[key] for key in ds.utterance_ordered[diaid]}
    utt = utts[uttid]
    input_ids = ds[idx]['input_ids']
    attention_mask = ds[idx]['attention_mask']
    labelid = ds[idx]['label']

    print(f'idx: {idx}')
    print()
    print(f"diaid: {diaid} \nuttid: {uttid}")
    print()
    print(input_ids)
    print()
    print(tokenizer.decode(input_ids))
    print()
    print(utt)
    print()
    for key, val in utts.items():
        pprint(f"{key}, {val}")

    decoded = tokenizer.decode(input_ids)

    input_ids = torch.tensor(input_ids).view(-1, len(input_ids))
    attention_mask = torch.tensor(attention_mask).view(-1, len(attention_mask))
    labelid = torch.tensor(labelid).view(-1, 1)

    return idx, input_ids, attention_mask, labelid, decoded, diaid, uttid, utt, utts


def parse_path(path):

    try:
        kwargs = read_yaml(path)

        DIR = '/'.join(path.split('/')[:-1])
        splits = path.split('/')
        timestamp = splits[2]

        hp = read_json(os.path.join(DIR, 'hp.json'))

        kwargs['timestamp'] = timestamp
        kwargs['peak_learning_rate'] = hp['learning_rate']
        kwargs['path'] = path

        for SPLIT in ['val', 'test']:
            kwargs[f"{SPLIT}_results"] = {path.split(
                '/')[-2]: read_json(path) for path in glob(os.path.join(DIR, f'*/{SPLIT}-results.json'))}

            metrics = set([key_ for key, val in kwargs[f"{SPLIT}_results"].items(
            ) for key_, val_ in val.items()])
            metrics = {metric: [] for metric in metrics}

            for key, val in kwargs[f"{SPLIT}_results"].items():
                for key_, val_ in val.items():
                    metrics[key_].append(val_)

            kwargs[f"{SPLIT}_results_mean_std"] = {metric: {'mean': np.mean(numbers), 'std': np.std(
                numbers), 'num_samples': len(numbers)} for metric, numbers in metrics.items()}
    except:
        return None

    return kwargs


def print_what_you_want(seed_results, what_you_want=['NUM_TRAIN_EPOCHS', 'ADD_BOU_EOU', 'ADD_SPEAKER_TOKENS', 'REPLACE_NAMES_IN_UTTERANCES', 'SPEAKER_SPLITS',
                                                     'model_checkpoint', 'num_past_utterances',
                                                     'num_future_utterances', 'path'],
                        metric='f1_weighted'):

    seed_results = sorted(seed_results, key=lambda x: -
                          x['test_results_mean_std'][f'test_{metric}']['mean'])
    for results in seed_results:
        for key, val in results.items():
            if key in what_you_want:
                print(f"{key}: {val}")
            elif key in ['test_results_mean_std', 'val_results_mean_std']:
                try:
                    print(f"{key}: {val[f'eval_{metric}']}")
                except Exception as e:
                    print(f"{key}: {val[f'test_{metric}']}")

        print()


def return_coeffs(tokenizer, input_ids, attentions, BATCH_IDX=0, LAYER=-1, QUERY_TOKEN_IDX=0, annoying_char='Ġ'):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[BATCH_IDX].tolist())
    QUERY_TOKEN = tokens[QUERY_TOKEN_IDX].split(annoying_char)[-1]

    coeffs = attentions[LAYER][BATCH_IDX].cpu().detach().numpy().sum(axis=0)[
        QUERY_TOKEN_IDX]
    coeffs /= coeffs.sum()

    idx_token_coeffs = [(idx, token.split(annoying_char)[-1], coeffs[idx])
                        for idx, token in enumerate(tokens)]

    assert len(coeffs) == len(tokens) == len(idx_token_coeffs)

    return QUERY_TOKEN, coeffs, tokens, idx_token_coeffs
