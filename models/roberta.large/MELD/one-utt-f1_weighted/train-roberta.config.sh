DATASET="MELD"

if [ "${DATASET}" = MELD ]; then
    NUM_CLASSES=7
elif [ "${DATASET}" = IEMOCAP ]; then
    NUM_CLASSES=7
elif [ "${DATASET}" = CAER ]; then
    NUM_CLASSES=7
elif [ "${DATASET}" = EmoryNLP ]; then
    NUM_CLASSES=7
elif [ "${DATASET}" = DailyDialog ]; then
    NUM_CLASSES=7
else
    echo "${DATASET} is not supported"
    exit 1
fi

METRIC=f1_weighted                                     # should be one of cross_entropy_loss, f1_weighted, f1_micro, or f1_macro
PRETRAIN_NSP=false                                     # pretrain next sentence prediction
SEEDS=0,1,2,3,4                                        # random seeds
NUM_UTT=1                                              # number of utterances in one sequence
WORKERS=60                                             # number of workers for preprocessing RoBERTa
TOTAL_NUM_UPDATES=1600                                 # one epoch is around 160 updates for MELD, when MAX_SENTENCES=8 and UPDATE_FREQ=4
WARMUP_UPDATES=320                                     # 20 percent of the number of updates
LR=1e-05                                               # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8                                        # Batch size, per GPU
MAX_TOKENS=4400                                        # maximum number of tokens in a batch, per GPU
ROBERTA_SIZE=large                                     # either "base" or "large"
ROBERTA_PATH="models/roberta.${ROBERTA_SIZE}/model.pt" # pre-trained
PATIENCE=10                                            # early stopping in number of training epochs
MAX_POSITIONS=512                                      # I think this should be fixed to 512.
UPDATE_FREQ=4                                          # update parameters every N_i batches, when in epoch i
MAX_EPOCH=10                                           # force stop training at specified epoch
SAVE_INTERVAL=1                                        # save a checkpoint every N epochs
GPU_IDS=0,1                                            # The GPU ids of your machine. use `nvidia-smi` to check them out.
WEIGHT_DECAY=0.1                                       # I haven't tune this yet.
DROP_OUT=0.1                                           # I haven't tune this yet.
ATTENTION_DROP_OUT=0.1                                 # I haven't tune this yet.
