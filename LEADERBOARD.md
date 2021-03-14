# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

The rows are ordred by the validation performance, not the test performance, because otherwise it's cheating.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | Speaker-one-utt-cross_entropy_loss | 70.333 | 58.835 | 63.258 | 
| roberta.base | SPEAKER-one-utt-cross_entropy_loss | 70.956 | 58.873 | 63.039 | 
| roberta.base | one-utt-cross_entropy_loss | 72.248 | 59.371 | 63.294 | 
| roberta.base | two-utt-cross_entropy_loss | 74.237 | 60.161 | 63.416 | 
| roberta.base | Speaker-two-utt-cross_entropy_loss | 75.14 | 60.516 | 63.273 | 
| roberta.base | SPEAKER-two-utt-cross_entropy_loss | 75.561 | 60.519 | 64.0 | 
| roberta.large | SPEAKER-one-utt-cross_entropy_loss | 76.522 | 62.034 | 64.877 | 
| roberta.large | Speaker-one-utt-cross_entropy_loss | 75.693 | 62.309 | 64.854 | 
| roberta.large | one-utt-cross_entropy_loss | 77.344 | 62.49 | 64.396 | 
| roberta.large | SPEAKER-one-utt-f1_weighted | 84.438 | 62.701 | 64.582 | 
| roberta.large | Speaker-two-utt-cross_entropy_loss | 76.15 | 62.75 | 64.728 | 
| roberta.large | Speaker-one-utt-f1_weighted | 83.441 | 63.072 | 64.775 | 
| roberta.large | one-utt-f1_weighted | 86.893 | 63.276 | 63.614 | 
| roberta.large | SPEAKER-two-utt-cross_entropy_loss | 77.408 | 63.338 | 65.108 | 
| roberta.large | two-utt-cross_entropy_loss | 77.919 | 63.354 | 64.811 | 
| roberta.large | SPEAKER-two-utt-f1_weighted | 87.518 | 64.772 | 64.645 | 
| roberta.large | two-utt-f1_weighted | 85.133 | 64.82 | 65.021 | 
| **roberta.large** |**Speaker-two-utt-f1_weighted** |**90.19** |**64.869** |**64.887** |
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
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
