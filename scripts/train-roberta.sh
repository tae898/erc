#!/usr/bin/env bash

# load config
. scripts/train-roberta.config.sh

CURRENT_TIME=$(date +%Y%m%d_%H%M%s)
CHECKPOINT_DIR="checkpoints_${CURRENT_TIME}"

rm -rf Datasets/$DATASET/roberta/
mkdir -p Datasets/$DATASET/roberta/

mkdir -p models/gpt2-bpe/
# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' -P 'models/gpt2-bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' -P 'models/gpt2-bpe'

# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P 'models/gpt2-bpe'

BASE_DIR="models/roberta.${ROBERTA_SIZE}/${DATASET}/${CURRENT_TIME}"
rm -rf $BASE_DIR
mkdir -p $BASE_DIR
cp scripts/train-roberta.config.sh $BASE_DIR

if test -f "${ROBERTA_PATH}"; then
    echo "${ROBERTA_PATH} exists."
else
    echo "${ROBERTA_PATH} does not exist."

    wget -N "https://dl.fbaipublicfiles.com/fairseq/models/roberta.${ROBERTA_SIZE}.tar.gz" -P 'models/'
    tar zxvf "models/roberta.${ROBERTA_SIZE}.tar.gz" --directory models/
    rm "models/roberta.${ROBERTA_SIZE}.tar.gz"
fi

echo "Training will be done over the SEEDS ${SEEDS}"

if [ "$PRETRAIN_NSP" = true ]; then
    echo 'pretraining with nsp ...'

    # format data for roberta
    python3 scripts/roberta-format-data.py --DATASET $DATASET --num-utt $NUM_UTT --pretrain-nsp

    # BPE encode for roberta
    for INPUT_ORDER in $(seq 0 $(expr $NUM_UTT - 1)); do
        for SPLIT in train val test; do
            python -m examples.roberta.multiprocessing_bpe_encoder \
                --encoder-json models/gpt2-bpe/encoder.json \
                --vocab-bpe models/gpt2-bpe/vocab.bpe \
                --inputs "Datasets/${DATASET}/roberta/$SPLIT.input${INPUT_ORDER}" \
                --outputs "Datasets/${DATASET}/roberta/$SPLIT.input${INPUT_ORDER}.bpe" \
                --workers $WORKERS \
                --keep-empty
        done
    done

    # Preprocess data into binary format for roberta
    for INPUT_ORDER in $(seq 0 $(expr $NUM_UTT - 1)); do
        fairseq-preprocess \
            --only-source \
            --trainpref "Datasets/${DATASET}/roberta/train.input${INPUT_ORDER}.bpe" \
            --validpref "Datasets/${DATASET}/roberta/val.input${INPUT_ORDER}.bpe" \
            --destdir "Datasets/${DATASET}/roberta/bin/input${INPUT_ORDER}" \
            --workers $WORKERS \
            --srcdict models/gpt2-bpe/dict.txt
    done

    fairseq-preprocess \
        --only-source \
        --trainpref "Datasets/${DATASET}/roberta/train.label" \
        --validpref "Datasets/${DATASET}/roberta/val.label" \
        --destdir "Datasets/${DATASET}/roberta/bin/label" \
        --workers $WORKERS
fi

# format data for roberta
python3 scripts/roberta-format-data.py --DATASET $DATASET --num-utt $NUM_UTT

# BPE encode for roberta
for INPUT_ORDER in $(seq 0 $(expr $NUM_UTT - 1)); do
    for SPLIT in train val test; do
        python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json models/gpt2-bpe/encoder.json \
            --vocab-bpe models/gpt2-bpe/vocab.bpe \
            --inputs "Datasets/${DATASET}/roberta/$SPLIT.input${INPUT_ORDER}" \
            --outputs "Datasets/${DATASET}/roberta/$SPLIT.input${INPUT_ORDER}.bpe" \
            --workers $WORKERS \
            --keep-empty
    done
done

# Preprocess data into binary format for roberta
for INPUT_ORDER in $(seq 0 $(expr $NUM_UTT - 1)); do
    fairseq-preprocess \
        --only-source \
        --trainpref "Datasets/${DATASET}/roberta/train.input${INPUT_ORDER}.bpe" \
        --validpref "Datasets/${DATASET}/roberta/val.input${INPUT_ORDER}.bpe" \
        --destdir "Datasets/${DATASET}/roberta/bin/input${INPUT_ORDER}" \
        --workers $WORKERS \
        --srcdict models/gpt2-bpe/dict.txt
done

fairseq-preprocess \
    --only-source \
    --trainpref "Datasets/${DATASET}/roberta/train.label" \
    --validpref "Datasets/${DATASET}/roberta/val.label" \
    --destdir "Datasets/${DATASET}/roberta/bin/label" \
    --workers $WORKERS

HEAD_NAME="${DATASET}_head" # Custom name for the classification head.

mkdir -p $CHECKPOINT_DIR

for SEED in ${SEEDS//,/ }; do
    echo "SEED number: ${SEED}"

    CUDA_VISIBLE_DEVICES=$GPU_IDS fairseq-train Datasets/$DATASET/roberta/bin/ \
        --save-dir $CHECKPOINT_DIR \
        --restore-file $ROBERTA_PATH \
        --max-positions $MAX_POSITIONS \
        --batch-size $MAX_SENTENCES \
        --max-tokens $MAX_TOKENS \
        --task sentence_prediction \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --init-token 0 --separator-token 2 \
        --arch "roberta_${ROBERTA_SIZE}" \
        --criterion sentence_prediction \
        --classification-head-name $HEAD_NAME \
        --num-classes $NUM_CLASSES \
        --dropout $DROP_OUT --attention-dropout $ATTENTION_DROP_OUT \
        --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --max-epoch $MAX_EPOCH \
        --save-interval $SAVE_INTERVAL \
        --shorten-method "truncate" \
        --find-unused-parameters \
        --update-freq $UPDATE_FREQ \
        --patience $PATIENCE \
        --seed $SEED

    # remove every trained model except the best one (val).
    cd $CHECKPOINT_DIR
    find . ! -name 'checkpoint_best.pt' -type f -exec rm -f {} +
    cd ..

    mv "${CHECKPOINT_DIR}/checkpoint_best.pt" "${BASE_DIR}/${SEED}.pt"

    # evaluate with weighted f1 scores and accuracy
    python3 scripts/evaluate.py --DATASET $DATASET --model-path "${BASE_DIR}/${SEED}.pt" --num-utt $NUM_UTT --use-cuda
    # rm "${BASE_DIR}/${SEED}.pt"
done

python3 scripts/evaluate.py --DATASET $DATASET --model-path "${BASE_DIR}/${SEED}.pt" --evaluate-seeds

rm -rf $CHECKPOINT_DIR

python3 scripts/evaluate.py --leaderboard

echo 'DONE!'
