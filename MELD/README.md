# MELD

This directory includes jupyter notebooks and python scripts to train a model. They are pretty much self-explanatory. You must install some python packages. If you haven't, go back to `README.md` of the root repo.

## meld2gmrc

You probably don't have to care about this. This is actually [part of the GMRC repo](https://github.com/cltl/GMRCAnnotation). I just didn't really know where to put this, and thus it is here.

## configs

Pleae check this first before running the jupyter notebooks. This directory includes yaml files. They help you write configs and set the hyper parameter values. Once they are set right, you don't have to set paths every time using command arguments, which can be very tedious.

## Jupyter notebooks.

- `1.smaller-datasets.ipynb` is to generate smaller datasets from the original datasets. Just ignore this. We don't do this anymore.
- `2.visual-features-extraction.ipynb` is to extract all of the visual features that are deemed necessary. This takes a lot of of time. 

    Pre computed visual features can be downloaded from [here](https://drive.google.com/file/d/1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt/view?usp=sharing), or using `gdown`,
    ```
    gdown --id 1jS2ufbIovxg8umkZM5UKzsvtSp4UJJyt
    ```
- `3.visual-features-check.ipynb` is to qualitatively see if the extracted visual features are correct or not.
- `4.find-relevant-faces.ipynb` is to run clustering (unsupervised) on the face embedding vectors to find the faces we want. 

    The face embedding vectors of the main six actors are saved as `main-actors`, in this directory.
- `5.extract-face-videos.ipynb` is to extract the video of the faces, if the face matches the speaker. 

    I do face recognition here. Currently I only matched the main 6 chracters (i.e. ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']), which account for the majority.

    The generated face videos can be downloaded from [here](https://drive.google.com/file/d/18zbjkuHJTU1fUGKAO6pCwCDSVxN2fK06/view?usp=sharing), or using `gdown`,
    ```
    gdown --id 18zbjkuHJTU1fUGKAO6pCwCDSVxN2fK06
    ```

- `6.dataset-stats.ipynb` is to see the stats of the datasets.

    This will give you a good overview.

- `7.train-simple.ipynb` aims to train a simple model. This uses pytorch dataset and data generator classes.


