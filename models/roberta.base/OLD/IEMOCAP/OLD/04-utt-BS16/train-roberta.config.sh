DATASET="IEMOCAP"

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
SPEAKER_MODE=none                                     # should be one of title, upper, lower, none
SEEDS=0,1,2,3,4                                                # random seeds
NUM_UTTS=4                                             # number of utterances in one sequence
SCORE_POOLING=max                                      # this is only used when NUM_UTTS is -1 (should be max or mean)
KEEP_THE_BEST=true                                     # keep the best model instead of deleting. (true or false)
WORKERS=60                                             # number of workers for preprocessing RoBERTa
LR=1e-05                                               # Peak LR for polynomial LR scheduler.
BATCH_SIZE=16                                          # Batch size, per GPU
ROBERTA_SIZE=base                                      # either "base" or "large"
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
