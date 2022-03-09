# Emotion Recognition in Coversation (ERC)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emoberta-speaker-aware-emotion-recognition-in/emotion-recognition-in-conversation-on)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on?p=emoberta-speaker-aware-emotion-recognition-in)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emoberta-speaker-aware-emotion-recognition-in/emotion-recognition-in-conversation-on-meld)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld?p=emoberta-speaker-aware-emotion-recognition-in)<br>

At the moment, we only use the text modality to correctly classify the emotion of the utterances.The experiments were carried out on two datasets (i.e. MELD and IEMOCAP)

## Prerequisites

- An x86-64 Unix or Unix-like machine
- Python 3.7 or higher
- [`multimodal-datasets` repo](https://github.com/tae898/multimodal-datasets) (submodule)

## RoBERTa training

First configure the hyper parameters and the dataset in `train-erc-text.yaml` and then,
In this directory run the below commands. I recommend you to run this in a virtualenv.

```bash
pip install -r requirements.txt
python train-erc-text.py
```

This will subsequently call `train-erc-text-hp.py` and `train-erc-text-full.py`.

## Results on the test split (weighted f1 scores)

| Model    |                                          |      MELD      |     IEMOCAP    |
|----------|------------------------------------------|:--------------:|:--------------:|
| EmoBERTa | No past and future utterances            |      63.46     |      56.09     |
|          | Only past utterances                     |      64.55     |    **68.57**   |
|          | Only future utterances                   |      64.23     |      66.56     |
|          | Both past and future utterances          |    **65.61**   |      67.42     |
|          | → *without speaker names*            |      65.07     |      64.02     |

Above numbers are the mean values of five random seed runs.

If you want to see more training test details, check out `./results/`

If you want to download the trained checkpoints and stuff, then [here](https://surfdrive.surf.nl/files/index.php/s/khREwk4MUI7MSnO/download) is where you can download them. It's a pretty big zip file.

## Troubleshooting

The best way to find and solve your problems is to see in the github issue tab. If you can't find what you want, feel free to raise an issue. We are pretty responsive.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Cite our work

Check out the [paper](https://arxiv.org/abs/2108.12009).

```bibtex
@misc{kim2021emoberta,
      title={EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa}, 
      author={Taewoon Kim and Piek Vossen},
      year={2021},
      eprint={2108.12009},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
[![DOI](https://zenodo.org/badge/328375452.svg)](https://zenodo.org/badge/latestdoi/328375452)<br>

## Authors

- [Taewoon Kim](https://taewoonkim.com/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
