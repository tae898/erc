# Multimodal Emotion Recognition in Coversation (ERC)

This branch (EmoBERTa) only uses the text modality to correctly classify the emotion of the utterances.The experiments were carried out on two datasets (i.e. MELD and IEMOCAP) 

## Prerequisites

* An x86-64 Unix or Unix-like machine
* Python 3.7 or higher
* [`multimodal-datasets` repo](https://github.com/tae898/multimodal-datasets) (submodule)

## RoBERTa Training

First configure the hyper parameters and the dataset in `train-erc-text.yaml` and then,
In this directory run training by

```bash
python3 train-erc-text.py
```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors
* Taewoon Kim (t.kim@vu.nl)

## License
[MIT](https://choosealicense.com/licenses/mit/)
