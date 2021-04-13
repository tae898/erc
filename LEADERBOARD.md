# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

As for DailyDialog, the neutral class, which accounts for 80% of the data, is not included in the f1_score calcuation. Note that they are still used in training.

## MELD 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 3-4-upper | 77.455 | 64.525 | 65.84 | 0.80606 | 1.10779 | 1.04602 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 32-1-upper | 82.853 | 64.472 | 68.044 | 0.45922 | 0.93426 | 0.85083 | 
| roberta.large | 32-1-upper-10-SPEAKERS | 78.64 | 64.857 | 66.465 | 0.55932 | 0.89895 | 0.87319 | 
| roberta.large | 8-1-upper | 76.901 | 64.094 | 66.138 | 0.60966 | 0.9247 | 0.8906 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 

|  base model | method | train (f1_weighted) | val (f1_weighted) | test (f1_weighted) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 

|  base model | method | train (f1_micro) | val (f1_micro) | test (f1_micro) | train (cse) | val (cse) | test (cse) |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 3-4-upper | 60.335 | 57.029 | 53.362 | 0.29179 | 0.23114 | 0.36888 | 
| CESTa | SOTA |   |   | 63.12 |
