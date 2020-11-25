# examples
This directory includes a lot of examples that you can run either locally (`./local`) or on google colab (`./colab`).
You have to have `requirements.txt` installed to run examples on your local machine, whereas you don't have requirements for colab.

## MELD Datasets

Download the full datasets from here https://affective-meld.github.io/.

I've also created smaller datasets from the original train dataset. 

`dataset-large.json` has 64, 8, and 8 dialogues in its train, dev, and test dataset, respectively.

`dataset-medium.json` has 32, 4, and 4 dialogues in its train, dev, and test dataset, respectively.

`dataset-small.json` has 16, 2, and 2 dialogues in its train, dev, and test dataset, respectively.

The smaller dataset is part of the bigger dataset's train data.

Run below to download and unzip the `smaller-dataset.zip`

```
!pip install gdown
!gdown --id 16ck7plW9v9eSHGCs5wuB2AhhufPRt3Wi
!unzip smaller-dataset.zip
!rm smaller-dataset.zip
```

## MELD Annotations

Full annotations can be found here https://github.com/declare-lab/MELD/tree/master/data/MELD

Dyadic annotations can be found here https://github.com/declare-lab/MELD/tree/master/data/MELD_Dyadic

## Visual feature extraction from the videos

Refer to this python package https://github.com/leolani/cltl-face-all. It gives you every visual feature from a given image.

Run below to download and unzip the already extracted visual features of the smaller-datset.

```
!pip install gdown
!gdown --id 1-2LeHC_5Cm2gWWT6vBrVhp8jorbjkN1_
!unzip visual-features.zip
!rm visual-features.zip
```

## Signal time-alignment

I’m thinking about how this can be done.

Let’s say that we are given an utterance, say, 3 seconds. Let’s assume that the fps of this video is 24. Then the number of images in this utterance is 24 fps * 3s = 72 frames.  In other words, there are 72 consecutive images in this utterance. 

The utterance comes with an audio file too. Let’s assume that the audio is mono and the sampling rate of the audio is 16000 Hz. Then the number of points is 16000 Hz * 3s = 48000.

The utterance also comes with an annotated text as well. Let’s assume that for the 3 seconds what was spoken was “Challenges are what make life interesting and overcoming them is what makes life meaningful.” If we tokenize the sentence, then it’ll give us 14 tokens. 

The three different modalities all give us different time length. Of course we can just resample them so that they can have the same time length, but there’s gotta be a better and smarter way.

## `./local` (run this locally)
* `smaller-datasets.ipynb` to reproduce the smaller datasets.
* `visual-features-extraction.ipynb` to extract visual features from the entire datasets.
* `visual-features-check.ipynb` to check the extracted visual features.
* ...

## `./colab` (run this on colab)
* `visual-features-extraction-colab.ipynb` to extract visual features from the smaller datasets.
* `visual-features-check-colab.ipynb` to check the extracted visual features.
* `find-relevant-faces-colab.ipynb` to see how to cluster face embeddings.
