# Datasets

## Supported datasets
There are three datasets:

1. [MELD](https://affective-meld.github.io/)

1. [IEMOCAP](https://sail.usc.edu/iemocap/)

1. [AFEW](https://cs.anu.edu.au/few/AFEW.html)

1. [CAER](https://caer-dataset.github.io/)

Out of the three modalities (text, audio, and visual), the above datasets have at least two of them, if not all three.

## Instructions

### Download the original datasets

You have to download the original datasets yourselves from the original authors. If you click on the links, the authors tell you how to download them. Some of them can be downloaded directly while some ask you to write them emails. Anyways, if you get them, they'll look like this:

```
├── CAER.zip
├── EmotiW_2018.zip
├── IEMOCAP_full_release.tar.gz
├── MELD.Raw.tar.gz
```
Btw, AFEW is `EmotiW_2018.zip`. Have them here in this directory (i.e. `root/of/the/repo/Datasets`). Don't change the archive names. 

### Clean the datasets

Run the bash script by

```
bash clean.sh
```
This might take a while ...

You need to have some Unix programs installed (i.e. ffmpeg, unrar, untar, unzip, and python3). They are probably already installed if you are an average Unix user.