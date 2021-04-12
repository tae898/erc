# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

As for DailyDialog, the neutral class, which accounts for 80% of the data, is not included in the f1_score calcuation. Note that they are still used in training.

## MELD 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt-SPEAKER-BS32 | 75.828 | 62.51 | 65.292 | 0.8262 | 1.12646 | 1.0548 | 
| roberta.large | all-utt-SPEAKER-BS1 | 77.66 | 63.005 | 64.865 | 0.80001 | 1.13455 | 1.07002 | 
| roberta.large | all-utt-SPEAKER-BS4 | 80.067 | 64.755 | 64.731 | 0.75938 | 1.09908 | 1.07431 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt-SPEAKER-BS16 | 73.992 | 55.639 | 53.051 | 0.71408 | 1.16846 | 1.24647 | 
| roberta.large | all-utt-SPEAKER-BS1 | 81.225 | 65.412 | 67.421 | 0.50056 | 0.89793 | 0.85201 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | all-utt-SPEAKER-BS1 | 51.055 | 38.016 | 35.008 | 1.28932 | 1.58423 | 1.62383 | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 

|  base model | method | train (f1_micro) | val (f1_micro) | test (f1_micro) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | all-utt-SPEAKER-BS4 | 69.538 | 58.016 | 55.522 | 0.23034 | 0.23084 | 0.37066 | 
| CESTa | SOTA |   |   | 63.12 |
