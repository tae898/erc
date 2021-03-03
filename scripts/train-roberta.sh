#!/usr/bin/env bash
if [[ "$@" =~ ^(MELD|IEMOCAP|AFEW|CAER)$ ]]; then
    DATASET=$@
    echo "Processing $DATASET ..."
else
    echo "$@ dataset is not supported"
    echo "Only MELD, IEMOCAP, AFEW, and CAER are supported!!!"
    exit 1
fi

rm -rf Datasets/$DATASET/roberta/
mkdir -p Datasets/$DATASET/roberta/

mkdir -p models/gpt2-bpe/
# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' -P 'models/gpt2-bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' -P 'models/gpt2-bpe'

# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P 'models/gpt2-bpe'

mkdir -p models/roberta.base

FILE='models/roberta.base/model.pt'
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    # Download RoBERTa-base
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz' -P 'models/'
    tar zxvf models/roberta.base.tar.gz --directory models/
fi

FILE='models/roberta.base.tar.gz'
if test -f "$FILE"; then
    rm $FILE
fi

# format data for roberta
python3 scripts/roberta-format-data.py --dataset $DATASET --num-past-utterances 1

# BPE encode for roberta
for SPLIT in train val test; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json models/gpt2-bpe/encoder.json \
        --vocab-bpe models/gpt2-bpe/vocab.bpe \
        --inputs "Datasets/$DATASET/roberta/$SPLIT.input0" \
        --outputs "Datasets/$DATASET/roberta/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done

# Preprocess data into binary format for roberta
fairseq-preprocess \
    --only-source \
    --trainpref "Datasets/$DATASET/roberta/train.input0.bpe" \
    --validpref "Datasets/$DATASET/roberta/val.input0.bpe" \
    --destdir "Datasets/$DATASET/roberta/bin/input0" \
    --workers 60 \
    --srcdict models/gpt2-bpe/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "Datasets/$DATASET/roberta/train.label" \
    --validpref "Datasets/$DATASET/roberta/val.label" \
    --destdir "Datasets/$DATASET/roberta/bin/label" \
    --workers 60

TOTAL_NUM_UPDATES=10000                    # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=600                         # 6 percent of the number of updates
LR=1e-05                                   # Peak LR for polynomial LR scheduler.
HEAD_NAME="${DATASET}_head"                # Custom name for the classification head.
MAX_SENTENCES=8                            # Batch size.
ROBERTA_PATH=models/roberta.base/model.pt # pre-trained roberta
SEED=0
GPU_IDS=0,1
MAX_EPOCH=10

if [ $DATASET = MELD ]; then
    NUM_CLASSES=7
else
    # TODO
    NUM_CLASSES=7
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS fairseq-train Datasets/$DATASET/roberta/bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch $MAX_EPOCH \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4 \
    --seed $SEED

# remove every trained model except the best one (val).
cd checkpoints
find . ! -name 'checkpoint_best.pt' -type f -exec rm -f {} +
cd ..

FILE='models/roberta.base/$DATASET.pt'
if test -f "$FILE"; then
    echo "$FILE exists."
    rm $FILE
fi
mkdir -p models/roberta.base/
mv checkpoints/checkpoint_best.pt models/roberta.base/$DATASET.pt

# evaluate with f1 scores
python3 scripts/evaluate.py --dataset $DATASET --use_cuda