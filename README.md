# ERC (Emotion Recognition in Coversation)

This repo is is to reach the SOTA of multimodal ERC challenges. The authors aim to publish a paper in 2021. 

The datasets to work on include

1. [MELD](https://affective-meld.github.io/) (ongoing)

    Go to the directory `MELD` to see the ongoing results.

2. [IEMOCAP](https://sail.usc.edu/iemocap/) (not yet started)

    I do have the datasets in my storage.

3. [AFEW](https://cs.anu.edu.au/few/AFEW.html) (not yet started)

    I sent a mail to the authors if I can get the dataset. I haven't got any responses yet.


## Prerequisites

* An x86-64 Unix or Unix-like machines 
* Python 3.6x, 3.7x, or 3.8x

## Installing the necessary python packages

The functions and classes, mostly pytorch stuff, you need to for train, test, and inference are bundled into a package `erc`. In addition, since we deal with multimodal data, there are three different packages to install. Each modality is a separate repo. They are all submodules of this repo `erc`. Use the `git submodule` commands properly.


1. erc

    In the root directory of this repo, where this `README.md` is located in, run

    ```
    pip install .
    ```
    I highly recommend you to run everything in a virtual environment.
1. Vision

    Go to the directory `cltl-face-all` and follow the instructions.

1. Text

    TBD

1. Audio

    TBD

## templates

This directory includes some templates that I follow. They are submodules and linked to the original repos. You don't have to care about them. They are just for references.


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors
* Taewoon Kim (t.kim@vu.nl)
* Khanh Nguyen Vu (k2.vu@student.vu.nl)

## License
[MIT](https://choosealicense.com/licenses/mit/)
