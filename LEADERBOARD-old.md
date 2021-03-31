# Leaderboard
Note that only DailyDialog uses a different metric (f1_micro) from others (f1_weighted). f1_micro is the same as accuracy when every data point is assigned only one class.

The reported performance of my models are the mean values of the 5 random seed runs. I expect the other authors have done the same thing or something similar, since the numbers are stochastic in nature.

Since the distribution of classes is different for every dataset and train / val / tests splits, and also not all datasets have the same performance metric, the optimization is done to minimize the validation cross entropy loss, since its the most generic metric, with backpropagation on training data split.

## MELD 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt | 76.691 | 62.604 | 63.978 | 
| roberta.large | 01-utt-SPEAKER | 74.727 | 62.168 | 64.546 | 
| roberta.large | 02-utt | 79.01 | 63.806 | 65.034 | 
| roberta.large | 02-utt-SPEAKER | 78.852 | 63.909 | 65.472 | 
| roberta.large | 03-utt | 78.206 | 63.613 | 64.844 | 
| roberta.large | 03-utt-SPEAKER | 76.488 | 64.138 | 66.115 | 
| roberta.large | 04-utt | 79.499 | 63.847 | 65.242 | 
| roberta.large | 04-utt-SPEAKER | 78.215 | 64.014 | 66.061 | 
| roberta.large | 05-utt | 75.217 | 63.467 | 64.753 | 
| roberta.large | 05-utt-SPEAKER | 77.048 | 64.606 | 65.458 | 
| roberta.large | 06-utt | 77.312 | 64.862 | 65.363 | 
| roberta.large | 06-utt-SPEAKER | 77.261 | 64.078 | 65.598 | 
| roberta.large | 07-utt | 75.592 | 64.159 | 65.409 | 
| roberta.large | 07-utt-SPEAKER | 78.052 | 64.721 | 65.78 | 
| roberta.large | 08-utt | 75.556 | 63.964 | 64.774 | 
| roberta.large | 08-utt-SPEAKER | 78.114 | 64.485 | 65.912 | 
| roberta.large | 09-utt | 75.646 | 63.892 | 65.219 | 
| roberta.large | 09-utt-SPEAKER | 77.086 | 65.068 | 65.398 | 
| roberta.large | 10-utt | 81.141 | 64.486 | 65.031 | 
| roberta.large | 10-utt-SPEAKER | 77.552 | 64.326 | 64.98 | 
| roberta.large | all-utt | 82.618 | 63.565 | 64.204 | 
| roberta.large | all-utt-SPEAKER | 79.89 | 62.857 | 65.014 | 
| COSMIC | SOTA |   |   | 65.21 |
## IEMOCAP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt | 72.317 | 54.899 | 54.002 | 
| roberta.large | 01-utt-SPEAKER | 72.516 | 54.961 | 53.412 | 
| roberta.large | 02-utt | 71.724 | 56.913 | 57.488 | 
| roberta.large | 02-utt-SPEAKER | 72.873 | 58.523 | 57.154 | 
| roberta.large | 03-utt | 74.452 | 59.434 | 61.966 | 
| roberta.large | 03-utt-SPEAKER | 76.49 | 61.363 | 61.932 | 
| roberta.large | 04-utt | 72.844 | 59.195 | 61.468 | 
| roberta.large | 04-utt-SPEAKER | 81.734 | 63.315 | 64.543 | 
| roberta.large | 05-utt | 73.535 | 60.196 | 62.972 | 
| roberta.large | 05-utt-SPEAKER | 75.027 | 60.374 | 64.181 | 
| roberta.large | 06-utt | 73.682 | 61.035 | 63.045 | 
| roberta.large | 06-utt-SPEAKER | 78.209 | 62.602 | 64.803 | 
| roberta.large | 07-utt | 73.767 | 61.824 | 62.578 | 
| roberta.large | 07-utt-SPEAKER | 83.187 | 65.375 | 65.594 | 
| roberta.large | 08-utt | 76.076 | 61.909 | 64.342 | 
| roberta.large | 08-utt-SPEAKER | 82.548 | 64.626 | 65.722 | 
| roberta.large | 09-utt | 74.224 | 62.085 | 62.311 | 
| roberta.large | 09-utt-SPEAKER | 77.454 | 62.898 | 66.073 | 
| roberta.large | 10-utt | 75.523 | 61.903 | 63.178 | 
| roberta.large | 10-utt-SPEAKER | 75.871 | 63.369 | 65.564 | 
| roberta.large | 11-utt | 68.497 | 60.844 | 62.733 | 
| roberta.large | 11-utt-SPEAKER | 77.85 | 64.148 | 64.84 | 
| roberta.large | 12-utt | 67.705 | 59.859 | 61.833 | 
| roberta.large | 12-utt-SPEAKER | 78.399 | 63.655 | 66.396 | 
| roberta.large | 13-utt | 70.184 | 59.181 | 63.344 | 
| roberta.large | 13-utt-SPEAKER | 76.937 | 64.32 | 66.865 | 
| roberta.large | 14-utt | 76.032 | 61.894 | 63.448 | 
| roberta.large | 14-utt-SPEAKER | 79.036 | 64.357 | 66.927 | 
| roberta.large | 15-utt | 73.801 | 60.999 | 63.19 | 
| roberta.large | 15-utt-SPEAKER | 76.906 | 65.249 | 66.946 | 
| roberta.large | 16-utt | 75.222 | 61.697 | 63.639 | 
| roberta.large | 16-utt-SPEAKER | 80.687 | 64.307 | 67.422 | 
| roberta.large | 17-utt | 76.789 | 62.168 | 64.315 | 
| roberta.large | 17-utt-SPEAKER | 79.467 | 65.29 | 67.518 | 
<<<<<<< HEAD:LEADERBOARD-old.md
| roberta.large | 18-utt | 77.293 | 62.676 | 64.036 | 
=======
>>>>>>> WTF:LEADERBOARD.md
| roberta.large | 18-utt-SPEAKER | 77.54 | 62.912 | 65.715 | 
| roberta.large | 19-utt-SPEAKER | 74.662 | 62.546 | 65.684 | 
| roberta.large | 20-utt-SPEAKER | 77.392 | 63.376 | 66.846 | 
| roberta.large | 21-utt-SPEAKER | 78.315 | 64.098 | 67.498 | 
| roberta.large | 22-utt-SPEAKER | 78.834 | 63.724 | 65.483 | 
<<<<<<< HEAD:LEADERBOARD-old.md
=======
| roberta.large | 23-utt-SPEAKER | 81.636 | 64.044 | 67.84 | 
| roberta.large | 24-utt-SPEAKER | 79.419 | 64.132 | 66.684 | 
| roberta.large | 25-utt-SPEAKER | 80.068 | 64.985 | 66.384 | 
| roberta.large | 26-utt-SPEAKER | 78.272 | 62.664 | 65.72 | 
| roberta.large | 27-utt-SPEAKER | 81.699 | 63.923 | 66.568 | 
>>>>>>> WTF:LEADERBOARD.md
| roberta.large | all-utt | 76.718 | 60.17 | 63.898 | 
| roberta.large | all-utt-SPEAKER | 81.804 | 64.98 | 67.763 | 
| roberta.large | all-utt-SPEAKER? | 79.985 | 63.525 | 66.276 | 
| CESTa | SOTA |   |   | 67.1 |
## EmoryNLP 
The metric is f1_weighted (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| roberta.large | 01-utt | 48.941 | 38.139 | 35.518 | 
| roberta.large | 01-utt-SPEAKER | 49.031 | 38.499 | 35.387 | 
| roberta.large | 02-utt-SPEAKER | 51.053 | 38.886 | 35.809 | 
| roberta.large | 03-utt-SPEAKER | 48.519 | 38.292 | 35.935 | 
| roberta.large | 04-utt-SPEAKER | 49.094 | 38.974 | 36.169 | 
| roberta.large | 05-utt-SPEAKER | 55.997 | 39.847 | 36.674 | 
| roberta.large | 06-utt-SPEAKER | 51.707 | 38.539 | 35.484 | 
| roberta.large | 07-utt-SPEAKER | 49.742 | 38.711 | 35.22 | 
| roberta.large | 08-utt-SPEAKER | 52.071 | 38.341 | 35.686 | 
| roberta.large | all-utt | 47.401 | 36.838 | 34.192 | 
| roberta.large | all-utt-SPEAKER | 45.824 | 36.324 | 33.817 | 
| COSMIC | SOTA |   |   | 38.11 |
## DailyDialog 
The metric is f1_micro (%)
|  base model | method | train | val | test |
|-------------- | -------------- | -------------- | -------------- | -------------- | 
| CESTa | SOTA |   |   | 63.12 |
