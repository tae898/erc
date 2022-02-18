# qualitative analysis

## MELD

model_checkpoint= `2021-05-08-19-57-31-speaker_mode-upper-num_past_utterances-1000-num_future_utterances-1000-batch_size-4-seed-4/checkpoint-9992/`

### true samples

idx,last `<s>` focus on the speaker, first speaker attends interlocutors

946,T,T

1684,T,T

1995,T,T

421,T,T

1150,T,T

1588,T,T

1024,T,T

2581,T,T

1073,T,T

927,T,T

**100%**

### false samples

idx,last `<s>` focus on the speaker, first speaker attends interlocutors

1034,F,T

1411,T,T

183,T,T

149,T,T

1751,T,T

1912,F,T

1399,T,T

2601,F,T

259,T,T

151,F,T

**60%**

## IEMOCAP

model_checkpoint = `2021-05-09-12-19-54-speaker_mode-upper-num_past_utterances-1000-num_future_utterances-0-batch_size-4-seed-4/checkpoint-5975/`

### true samples

idx,last `<s>` focus on the speaker, first speaker attends interlocutors

1150,T,T

1146,F,T

174,F,T

439,F,T

369,F,T

456,F,T

818,T,T

393,F,T

109,F,T

951,F,T

**20%**

### false samples

idx,last `<s>` focus on the speaker, first speaker attends interlocutors

1232,F,T

709,F,T

1082,F,T

823,F,T

1337,F,T

946,F,T

1318,T,T

422,F,T

954,F,T

343,F,T

**10%**
