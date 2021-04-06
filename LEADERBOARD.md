# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

As for DailyDialog, the neutral class, which accounts for 80% of the data, is not included in the f1_score calcuation. Note that they are still used in training.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt-BS64 | 68.081 | 58.539 | 62.638 | 
| roberta.large | 01-utt-SPEAKER-BS32 | 75.828 | 62.51 | 65.292 | 
| roberta.base | 01-utt-SPEAKER-BS64 | 68.811 | 58.708 | 63.089 | 
| roberta.base | 02-utt-BS32 | 70.685 | 60.114 | 63.288 | 
| roberta.base | 02-utt-SPEAKER-BS32 | 71.501 | 60.119 | 63.563 | 
| roberta.base | 03-utt-BS32 | 70.183 | 60.369 | 63.957 | 
| roberta.base | 03-utt-SPEAKER-BS32 | 71.218 | 60.24 | 64.02 | 
| roberta.base | 04-utt-BS32 | 73.168 | 61.085 | 64.302 | 
| roberta.base | 04-utt-SPEAKER-BS32 | 73.056 | 61.222 | 64.211 | 
| roberta.base | 05-utt-BS32 | 72.049 | 61.105 | 64.078 | 
| roberta.base | 05-utt-SPEAKER-BS32 | 71.921 | 60.972 | 64.297 | 
| roberta.base | all-utt-BS8 | 75.709 | 60.905 | 63.506 | 
| roberta.base | all-utt-SPEAKER-BS8 | 73.063 | 61.471 | 63.535 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt-BS64 | 72.38 | 54.621 | 52.716 | 
| roberta.base | 01-utt-SPEAKER-02-names-BS64 | 72.495 | 54.211 | 51.987 | 
| roberta.large | 01-utt-SPEAKER-10-names-BS16 | 73.992 | 55.639 | 53.051 | 
| roberta.base | 01-utt-SPEAKER-10-names-BS64 | 74.114 | 54.301 | 51.643 | 
| roberta.base | 02-utt-BS32 | 70.612 | 56.113 | 55.088 | 
| roberta.base | 02-utt-SPEAKER-02-names-BS32 | 71.4 | 56.94 | 54.731 | 
| roberta.base | 02-utt-SPEAKER-10-names-BS32 | 74.658 | 58.572 | 55.504 | 
| roberta.base | 03-utt-BS32 | 71.118 | 57.543 | 58.645 | 
| roberta.base | 03-utt-SPEAKER-02-names-BS32 | 66.71 | 58.423 | 56.688 | 
| roberta.base | 03-utt-SPEAKER-10-names-BS32 | 74.296 | 59.521 | 58.048 | 
| roberta.base | all-utt-BS8 | 75.114 | 61.039 | 62.785 | 
| roberta.base | all-utt-SPEAKER-02-names-BS8 | 69.636 | 60.741 | 61.362 | 
| roberta.base | all-utt-SPEAKER-10-names-BS8 | 70.966 | 62.245 | 61.894 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt-BS32 | 43.255 | 37.534 | 34.414 | 
| roberta.base | 01-utt-SPEAKER-BS32 | 44.276 | 37.093 | 34.318 | 
| roberta.base | 02-utt-BS32 | 44.09 | 37.44 | 34.398 | 
| roberta.base | 02-utt-SPEAKER-BS32 | 45.283 | 37.661 | 35.275 | 
| roberta.base | 03-utt-BS32 | 44.754 | 37.701 | 34.589 | 
| roberta.base | 03-utt-SPEAKER-BS32 | 44.104 | 37.509 | 34.732 | 
| roberta.base | all-utt-BS8 | 45.351 | 37.175 | 33.945 | 
| roberta.base | all-utt-SPEAKER-BS8 | 48.431 | 37.935 | 34.148 | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | all-utt-SPEAKER-BS4 | 69.538 | 58.016 | 55.522 | 
| CESTa | SOTA |   |   | 63.12 |
