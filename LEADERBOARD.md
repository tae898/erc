# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed trainings. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

## Dataset: MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | SPEAKER-one-utt | 70.956 | 58.873 | 63.039 | 
| roberta.base | Speaker-one-utt | 70.333 | 58.835 | 63.258 | 
| roberta.base | Speaker-utt-two | 75.14 | 60.516 | 63.273 | 
| roberta.base | one-utt | 72.248 | 59.371 | 63.294 | 
| roberta.base | two-utt | 74.237 | 60.161 | 63.416 | 
| **roberta.base** |**SPEAKER-two-utt** |**75.561** |**60.519** |**64.0** |
| COSMIC | SOTA |   |   | 65.21 |
## Dataset: IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CESTa | SOTA |   |   | 67.1 |
## Dataset: CAER 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CAER-Net | SOTA |   |   | 77.04 |
## Dataset: EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| COSMIC | SOTA |   |   | 38.11 |
## Dataset: DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CESTa | SOTA |   |   | 63.12 |
