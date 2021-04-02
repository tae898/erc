# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt | 68.081 | 58.539 | 62.638 | 
| roberta.base | 01-utt-SPEAKER | 68.811 | 58.708 | 63.089 | 
| roberta.base | 02-utt | 70.685 | 60.114 | 63.288 | 
| roberta.base | all-utt | 75.709 | 60.905 | 63.506 | 
| roberta.base | all-utt-SPEAKER | 73.063 | 61.471 | 63.535 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt | 72.38 | 54.621 | 52.716 | 
| roberta.base | 01-utt-SPEAKER-02-names | 72.495 | 54.211 | 51.987 | 
| roberta.base | 01-utt-SPEAKER-10-names | 74.114 | 54.301 | 51.643 | 
| roberta.base | all-utt | 75.114 | 61.039 | 62.785 | 
| roberta.base | all-utt-SPEAKER-02-names | 69.636 | 60.741 | 61.362 | 
| roberta.base | all-utt-SPEAKER-10-names | 70.966 | 62.245 | 61.894 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt | 43.255 | 37.534 | 34.414 | 
| roberta.base | 01-utt-SPEAKER | 44.276 | 37.093 | 34.318 | 
| roberta.base | all-utt | 45.351 | 37.175 | 33.945 | 
| roberta.base | all-utt-SPEAKER | 48.431 | 37.935 | 34.148 | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | 01-utt | 93.679 | 90.219 | 87.005 | 
| roberta.base | 01-utt-SPEAKER | 92.492 | 90.239 | 86.371 | 
| roberta.base | all-utt | 94.965 | 91.176 | 89.147 | 
| roberta.base | all-utt-SPEAKER | 94.808 | 91.093 | 88.696 | 
| CESTa | SOTA |   |   | 63.12 |
