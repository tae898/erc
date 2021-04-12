# Multimodal Emotion Recognition in Coversation (ERC)

This repo is is to reach the SOTA of multimodal ERC challenges. The authors aim to publish a paper in 2021. 

## Prerequisites

* An x86-64 Unix or Unix-like machine
* Python 3.6x, 3.7x, or 3.8x
* [fairseq](https://github.com/pytorch/fairseq#requirements-and-installation). 
    * I hope that I can replace this soon.
* [apex](https://github.com/pytorch/fairseq#requirements-and-installation)
    * This can be "easily" replaced by the builtin functions and packages that the new pytorch versions can offer.

## Datasets

There are multiple datasets that we experimented on. You check check them out in the `Datasets/` directory. We only include the datasets where the speaker emotion is labeled. Our goal is to recognize speaker emotion in conversation. **At the moment we find that only MELD, IEMOCAP, EmoryNLP, and DailyDialog are only relevant to us.** MELD and IEMOCAP have all of the three modalities (i.e. text, audio, and vision), whereas the other two datasets only have the text modality. It's surprisingly difficult to find quality datasets that have all of the three modalities. 


## RoBERTa Training
In this directory run training by
```
bash scripts/train-roberta.sh <DATASET>
```
This requires `fairseq` and `apex`. I hope I can replace them soon.

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
