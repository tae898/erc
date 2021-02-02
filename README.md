# ERC (Emotion Recognition in Coversation)

This repo is is to reach the SOTA of multimodal ERC challenges. The authors aim to publish a paper in 2021. 

## Prerequisites

* An x86-64 Unix or Unix-like machines 
* Python 3.6x, 3.7x, or 3.8x


## Datasets

There are multiple datasets that we experimented on. You check check them out in the `Datasets` directory.

## Installing the necessary python packages

Since we deal with multimodal data, there are three different packages to install. Each modality is a separate repo. They are all submodules of this repo `erc`. Use the `git submodule` commands properly. If you just want to train / test erc models, you probably dont' have to install them.

1. Vision

    Go to the directory `cltl-face-all` and follow the instructions.

1. Text

    TBD

1. Audio

    TBD

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
