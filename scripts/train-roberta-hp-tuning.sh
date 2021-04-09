#!/usr/bin/env bash
# pip install scikit-learn requests nltk importlib_metadata

for DATASET in IEMOCAP EmoryNLP; do
    if [ "${DATASET}" = MELD ]; then
        NUM_CLASSES=7
    elif [ "${DATASET}" = IEMOCAP ]; then
        NUM_CLASSES=6
    elif [ "${DATASET}" = EmoryNLP ]; then
        NUM_CLASSES=7
    elif [ "${DATASET}" = DailyDialog ]; then
        NUM_CLASSES=7
    else
        echo "${DATASET} is not supported"
        exit 1
    fi

    METRIC=cross_entropy_loss                              # should be one of f1_weighted, f1_micro, f1_macro, or cross_entropy_loss
    SPEAKER_MODE=upper                                     # should be one of title, upper, lower, none
    SCORE_POOLING=max                                      # this is only used when NUM_UTTS is -1 (should be max or mean)
    KEEP_THE_BEST=false                                    # keep the best model instead of deleting. (true or false)
    WORKERS=60                                             # number of workers for preprocessing RoBERTa
    LR=1e-05                                               # Peak LR for polynomial LR scheduler.
    ROBERTA_SIZE=large                                     # either "base" or "large"
    ROBERTA_PATH="models/roberta.${ROBERTA_SIZE}/model.pt" # pre-trained
    PATIENCE=5                                             # early stopping in number of training epochs
    TOKENS_PER_SAMPLE=512                                  # I think this should be fixed to 512.
    UPDATE_FREQ=4                                          # update parameters every N_i batches, when in epoch i
    NUM_EPOCHS=100                                         # force stop training at specified epoch
    NUM_WARMUP_EPOCHS=2                                    # number of warmup epochs
    SAVE_INTERVAL=1                                        # save a checkpoint every N epochs
    GPU_IDS=0                                              # The GPU ids of your machine. use `nvidia-smi` to check them out.
    WEIGHT_DECAY=0.1                                       # I haven't tune this yet.
    DROP_OUT=0.1                                           # I haven't tune this yet.
    ATTENTION_DROP_OUT=0.1                                 # I haven't tune this yet.

    SEEDS=0 # random seeds
    # for NUM_UTTS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 1000; do
    for NUM_UTTS in 1 2 4 8 16 32; do
        if [ "${DATASET}" = MELD ]; then
            if ((NUM_UTTS == 1)); then
                BATCH_SIZE=8
            elif ((NUM_UTTS > 1 && NUM_UTTS <= 2)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 2 && NUM_UTTS <= 4)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 4 && NUM_UTTS <= 8)); then
                BATCH_SIZE=2
            elif ((NUM_UTTS > 8 && NUM_UTTS <= 16)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 16 && NUM_UTTS <= 32)); then
                BATCH_SIZE=1
            else
                BATCH_SIZE=1
            fi
        elif [ "${DATASET}" = IEMOCAP ]; then
            if ((NUM_UTTS == 1)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 1 && NUM_UTTS <= 2)); then
                BATCH_SIZE=2
            elif ((NUM_UTTS > 2 && NUM_UTTS <= 4)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 4 && NUM_UTTS <= 8)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 8 && NUM_UTTS <= 16)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 16 && NUM_UTTS <= 32)); then
                BATCH_SIZE=1
            else
                BATCH_SIZE=1
            fi
        elif [ "${DATASET}" = EmoryNLP ]; then
            if ((NUM_UTTS == 1)); then
                BATCH_SIZE=2
            elif ((NUM_UTTS > 1 && NUM_UTTS <= 2)); then
                BATCH_SIZE=2
            elif ((NUM_UTTS > 2 && NUM_UTTS <= 4)); then
                BATCH_SIZE=2
            elif ((NUM_UTTS > 4 && NUM_UTTS <= 8)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 8 && NUM_UTTS <= 16)); then
                BATCH_SIZE=1
            elif ((NUM_UTTS > 16 && NUM_UTTS <= 32)); then
                BATCH_SIZE=1
            else
                BATCH_SIZE=1
            fi
        elif [ "${DATASET}" = DailyDialog ]; then
            if ((NUM_UTTS == 1)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 1 && NUM_UTTS <= 2)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 2 && NUM_UTTS <= 4)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 4 && NUM_UTTS <= 8)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 8 && NUM_UTTS <= 16)); then
                BATCH_SIZE=4
            elif ((NUM_UTTS > 16 && NUM_UTTS <= 32)); then
                BATCH_SIZE=4
            else
                BATCH_SIZE=4
            fi
        fi
        echo $DATASET $NUM_UTTS $BATCH_SIZE
        CURRENT_TIME=$(date +%Y%m%d_%H%M%s)
        CHECKPOINT_DIR="checkpoints-${CURRENT_TIME}-${NUM_UTTS}-${BATCH_SIZE}-${SPEAKER_MODE}"
        echo $CHECKPOINT_DIR

        rm -rf Datasets/$DATASET/roberta/
        mkdir -p Datasets/$DATASET/roberta/

        mkdir -p models/gpt2-bpe/
        # Download encoder.json and vocab.bpe
        wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' -P 'models/gpt2-bpe'
        wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' -P 'models/gpt2-bpe'

        # Download fairseq dictionary.
        wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P 'models/gpt2-bpe'

        BASE_DIR="models/roberta.${ROBERTA_SIZE}/${DATASET}/${CURRENT_TIME}-${NUM_UTTS}-${BATCH_SIZE}-${SPEAKER_MODE}"
        rm -rf $BASE_DIR
        mkdir -p $BASE_DIR

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
            --speaker-mode $SPEAKER_MODE --tokens-per-sample $TOKENS_PER_SAMPLE

        if ((NUM_UTTS == 1)); then
            NUM_INPUTS=1
        else
            NUM_INPUTS=2
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
                --num-utts $NUM_UTTS \
                --score-pooling $SCORE_POOLING \
                --keep-the-best $KEEP_THE_BEST \
                --metric $METRIC \
                --num-gpus $NUM_GPUS

        done

        python3 scripts/evaluate.py --DATASET $DATASET --base-dir $BASE_DIR \
            --evaluate-seeds

        rm -rf $CHECKPOINT_DIR

        # python3 scripts/evaluate.py --leaderboard

        echo 'DONE!'

    done
done
