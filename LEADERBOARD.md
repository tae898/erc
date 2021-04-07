# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

As for DailyDialog, the neutral class, which accounts for 80% of the data, is not included in the f1_score calcuation. Note that they are still used in training.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt-SPEAKER-BS32 | 75.828 | 62.51 | 65.292 | 
| roberta.large | all-utt-SPEAKER-BS4 | 80.067 | 64.755 | 64.731 | 
| roberta.large | all-utt-SPEAKERS-BS1 | 77.66 | 63.005 | 64.865 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt-SPEAKER-10-names-BS16 | 73.992 | 55.639 | 53.051 | 
| roberta.large | all-utt-SPEAKER-10-names-BS1 | 81.225 | 65.412 | 67.421 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | all-utt-SPEAKERS-BS1 | 51.055 | 38.016 | 35.008 | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | all-utt-SPEAKER-BS4 | 69.538 | 58.016 | 55.522 | 
| CESTa | SOTA |   |   | 63.12 |
