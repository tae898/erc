#!/usr/bin/env bash

pip install scikit-learn requests nltk importlib_metadata

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

# format data for roberta
python3 scripts/roberta-format-data.py --DATASET $DATASET --num-utts $NUM_UTTS \
    --speaker-mode $SPEAKER_MODE --tokens-per-sample $TOKENS_PER_SAMPLE --clean-utterances $CLEAN_UTTERANCES

if ((NUM_UTTS > 1)); then
    NUM_INPUTS=2
else
    NUM_INPUTS=1
fi
echo "number of inputs: ${NUM_INPUTS}"

# BPE encode for roberta
for INPUT_ORDER in $(seq 0 $(expr $NUM_INPUTS - 1)); do
    echo "input order: ${INPUT_ORDER}"
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
for INPUT_ORDER in $(seq 0 $(expr $NUM_INPUTS - 1)); do
    echo "input order: ${INPUT_ORDER}"
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

NUM_TRAINING_SAMPLES=$(cat Datasets/${DATASET}/roberta/train.label | wc -l)
NUM_GPUS=$(($(echo ${GPU_IDS} | tr -cd , | wc -c) + 1))
TOTAL_NUM_UPDATES=$((NUM_EPOCHS * NUM_TRAINING_SAMPLES / NUM_GPUS / BATCH_SIZE / UPDATE_FREQ))
WARMUP_UPDATES=$((NUM_WARMUP_EPOCHS * NUM_TRAINING_SAMPLES / NUM_GPUS / BATCH_SIZE / UPDATE_FREQ))
MAX_TOKENS=$(($BATCH_SIZE * TOKENS_PER_SAMPLE))

echo "NUM_TRAINING_SAMPLES=${NUM_TRAINING_SAMPLES}, NUM_GPUS=${NUM_GPUS}, \
TOTAL_NUM_UPDATES=${TOTAL_NUM_UPDATES}, WARMUP_UPDATES=${WARMUP_UPDATES}, MAX_TOKENS=${MAX_TOKENS}"

HEAD_NAME="${DATASET}_head" # Custom name for the classification head.

mkdir -p $CHECKPOINT_DIR

for SEED in ${SEEDS//,/ }; do
    echo "SEED number: ${SEED}"

    CUDA_VISIBLE_DEVICES=$GPU_IDS fairseq-train Datasets/$DATASET/roberta/bin/ \
        --save-dir $CHECKPOINT_DIR \
        --restore-file $ROBERTA_PATH \
        --max-positions $TOKENS_PER_SAMPLE \
        --batch-size $BATCH_SIZE \
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
        --max-epoch $NUM_EPOCHS \
        --save-interval $SAVE_INTERVAL \
        --shorten-method "truncate" \
        --find-unused-parameters \
        --update-freq $UPDATE_FREQ \
        --patience $PATIENCE \
        --seed $SEED

    # remove every trained model except the best one (val).

    # evaluate with the metric
    python3 scripts/evaluate.py --DATASET $DATASET --seed $SEED \
        --checkpoint-dir $CHECKPOINT_DIR --base-dir $BASE_DIR \
        --batch-size $BATCH_SIZE \
        --metric $METRIC --use-cuda

done

python3 scripts/evaluate.py --DATASET $DATASET --base-dir $BASE_DIR \
    --evaluate-seeds

rm -rf $CHECKPOINT_DIR

python3 scripts/evaluate.py --leaderboard

echo 'DONE!'
