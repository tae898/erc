# MELD

This directory includes jupyter notebooks. They are pretty much self-explanatory.

## Jupyter notebooks.

- `1.smaller-datasets.ipynb` is to generate smaller datasets from the original datasets. Just ignore this. We don't do this anymore.
-  `2.visual-features-extraction.ipynb` is to extract all of the visual features that are deemed necessary. This takes a lot of of time. 

    Pre computed visual features can be downloaded from [here](https://drive.google.com/file/d/1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt/view?usp=sharing), or using `gdown`,
    > `gdown --id 1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt`
- `3.visual-features-check.ipynb` is to qualitatively see if the extracted visual features are correct or not.
- `4.find-relevant-faces.ipynb` is to run clustering (unsupervised) on the face embedding vectors to find the faces we want. 

    The face embedding vectors of the main six actors are saved as `main-actors`, in this directory.
- `5.dataset-stats.ipynb` is to see the stats of the datasets.

    This will give you a good overview.
- `6.extract-face-videos.ipynb` is to extract the video of the faces, if the face matches the speaker. I do face recognition here. Currently I only matched the main 6 chracters (i.e. ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']), which account for the majority.

    The generated videos are saved along with the pre-computed visual features from `2.visual-features-extraction.ipynb`. 

    The generated face videos can be downloaded from [here](https://drive.google.com/file/d/1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt/view?usp=sharing), or using `gdown`,
    > `gdown --id 1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt`
