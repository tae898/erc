# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

The rows are ordred by the validation performance, not the test performance, because otherwise it's cheating.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | SPEAKER-one-utt-f1_weighted | 84.438 | 62.701 | 64.582 | 
| roberta.large | Speaker-one-utt-f1_weighted | 83.441 | 63.072 | 64.775 | 
| roberta.large | one-utt-f1_weighted | 86.893 | 63.276 | 63.614 | 
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
