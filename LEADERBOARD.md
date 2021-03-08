# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

`CEL` stands for cross entropy loss. The models with CEL stopped training when their validation cross entropy loss hit the minimum.
## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.base | SPEAKER-one-utt-CEL | 70.956 | 58.873 | 63.039 | 
| roberta.base | Speaker-one-utt-CEL | 70.333 | 58.835 | 63.258 | 
| roberta.base | Speaker-two-utt-CEL | 75.14 | 60.516 | 63.273 | 
| roberta.base | one-utt-CEL | 72.248 | 59.371 | 63.294 | 
| roberta.base | two-utt-CEL | 74.237 | 60.161 | 63.416 | 
| roberta.base | SPEAKER-two-utt-CEL | 75.561 | 60.519 | 64.0 | 
| roberta.large | one-utt-CEL | 77.344 | 62.49 | 64.396 | 
| roberta.large | Speaker-two-utt-CEL | 76.15 | 62.75 | 64.728 | 
| roberta.large | two-utt-CEL | 77.919 | 63.354 | 64.811 | 
| roberta.large | Speaker-one-utt-CEL | 75.693 | 62.309 | 64.854 | 
| roberta.large | SPEAKER-one-utt-CEL | 76.522 | 62.034 | 64.877 | 
| **roberta.large** |**SPEAKER-two-utt-CEL** |**77.408** |**63.338** |**65.108** |
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CESTa | SOTA |   |   | 67.1 |
## CAER 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CAER-Net | SOTA |   |   | 77.04 |
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
