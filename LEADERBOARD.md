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
| roberta.large | one-utt-Speaker | 75.723 | 62.306 | 64.636 | 
| roberta.large | five-utt | 75.217 | 63.467 | 64.753 | 
| roberta.large | three-utt | 78.206 | 63.613 | 64.844 | 
| roberta.large | two-utt | 79.01 | 63.806 | 65.034 | 
| roberta.large | two-utt-Speaker | 78.747 | 63.807 | 65.166 | 
| roberta.large | four-utt | 79.499 | 63.847 | 65.242 | 
| roberta.large | five-utt-SPEAKER | 77.048 | 64.606 | 65.458 | 
| roberta.large | two-utt-SPEAKER | 78.852 | 63.909 | 65.472 | 
| roberta.large | five-utt-Speaker | 78.211 | 64.444 | 65.706 | 
| roberta.large | four-utt-Speaker | 77.216 | 64.384 | 65.822 | 
| roberta.large | three-utt-Speaker | 76.811 | 64.562 | 65.998 | 
| roberta.large | four-utt-SPEAKER | 78.215 | 64.014 | 66.061 | 
| **roberta.large** |**three-utt-SPEAKER** |**76.488** |**64.138** |**66.115** |
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | one-utt | 72.317 | 54.899 | 54.002 | 
| roberta.large | two-utt | 71.724 | 56.913 | 57.488 | 
| roberta.large | four-utt | 72.844 | 59.195 | 61.468 | 
| roberta.large | three-utt | 74.452 | 59.434 | 61.966 | 
| roberta.large | five-utt | 73.535 | 60.196 | 62.972 | 
| **roberta.large** |**six-utt** |**73.682** |**61.035** |**63.045** |
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
