DATASET="MELD"
NUM_CLASSES=7
SEEDS=0,1,2,3,4 # random seeds
NUM_UTTS=1       # number of utterances in one sequence
WORKERS=60

TOTAL_NUM_UPDATES=1600                    # one epoch is around 160 updates for MELD, when BATCH_SIZE=8 and UPDATE_FREQ=4
WARMUP_UPDATES=320                        # 20 percent of the number of updates
LR=1e-05                                  # Peak LR for polynomial LR scheduler.
BATCH_SIZE=8                           # Batch size.
MAX_TOKENS=4400                           # maximum number of tokens in a batch
ROBERTA_PATH=models/roberta.base/model.pt # pre-trained roberta-base TODO: use roberta-large
PATIENCE=10                               # early stopping in number of training epochs
ROBERTA_SIZE=roberta_base                 # either roberta_base or roberta_large
TOKENS_PER_SAMPLE=512                         # I think this should be fixed to 512.
UPDATE_FREQ=4                             # update parameters every N_i batches, when in epoch i
NUM_EPOCHS=10                              # force stop training at specified epoch
SAVE_INTERVAL=1                           # save a checkpoint every N epochs
GPU_IDS=0,1
WEIGHT_DECAY=0.1
DROP_OUT=0.1
ATTENTION_DROP_OUT=0.1
