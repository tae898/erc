# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | one-utt | 76.691 | 62.604 | 63.978 | 
| roberta.large | one-utt-SPEAKER | 74.727 | 62.168 | 64.546 | 
| roberta.large | five-utt | 75.217 | 63.467 | 64.753 | 
| roberta.large | eight-utt | 75.556 | 63.964 | 64.774 | 
| roberta.large | three-utt | 78.206 | 63.613 | 64.844 | 
| roberta.large | two-utt | 79.01 | 63.806 | 65.034 | 
| roberta.large | four-utt | 79.499 | 63.847 | 65.242 | 
| roberta.large | six-utt | 77.312 | 64.862 | 65.363 | 
| roberta.large | seven-utt | 75.592 | 64.159 | 65.409 | 
| roberta.large | five-utt-SPEAKER | 77.048 | 64.606 | 65.458 | 
| roberta.large | two-utt-SPEAKER | 78.852 | 63.909 | 65.472 | 
| roberta.large | four-utt-SPEAKER | 78.215 | 64.014 | 66.061 | 
| **roberta.large** |**three-utt-SPEAKER** |**76.488** |**64.138** |**66.115** |
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | one-utt-SPEAKER | 72.516 | 54.961 | 53.412 | 
| roberta.large | one-utt | 72.317 | 54.899 | 54.002 | 
| roberta.large | two-utt-SPEAKER | 72.873 | 58.523 | 57.154 | 
| roberta.large | two-utt | 71.724 | 56.913 | 57.488 | 
| roberta.large | four-utt | 72.844 | 59.195 | 61.468 | 
| roberta.large | three-utt | 74.452 | 59.434 | 61.966 | 
| roberta.large | nine-utt | 74.224 | 62.085 | 62.311 | 
| roberta.large | seven-utt | 73.767 | 61.824 | 62.578 | 
| roberta.large | five-utt | 73.535 | 60.196 | 62.972 | 
| roberta.large | six-utt | 73.682 | 61.035 | 63.045 | 
| roberta.large | ten-utt | 75.523 | 61.903 | 63.178 | 
| **roberta.large** |**eight-utt** |**76.076** |**61.909** |**64.342** |
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CESTa | SOTA |   |   | 63.12 |
