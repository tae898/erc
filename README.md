# Emotion Recognition in Coversation (ERC)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emoberta-speaker-aware-emotion-recognition-in/emotion-recognition-in-conversation-on)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on?p=emoberta-speaker-aware-emotion-recognition-in)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emoberta-speaker-aware-emotion-recognition-in/emotion-recognition-in-conversation-on-meld)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld?p=emoberta-speaker-aware-emotion-recognition-in)<br>

At the moment, we only use the text modality to correctly classify the emotion of the utterances.The experiments were carried out on two datasets (i.e. MELD and IEMOCAP)

## Prerequisites

1. An x86-64 Unix or Unix-like machine
1. Python 3.8 or higher
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. [`multimodal-datasets` repo](https://github.com/tae898/multimodal-datasets) (submodule)
1. pip install -r requirements.txt

## EmoBERTa training

First configure the hyper parameters and the dataset in `train-erc-text.yaml` and then,
In this directory run the below commands. I recommend you to run this in a virtualenv.

```sh
python train-erc-text.py
```

This will subsequently call `train-erc-text-hp.py` and `train-erc-text-full.py`.

## Results on the test split (weighted f1 scores)

| Model    |                                 |   MELD    |  IEMOCAP  |
| -------- | ------------------------------- | :-------: | :-------: |
| EmoBERTa | No past and future utterances   |   63.46   |   56.09   |
|          | Only past utterances            |   64.55   | **68.57** |
|          | Only future utterances          |   64.23   |   66.56   |
|          | Both past and future utterances | **65.61** |   67.42   |
|          | → *without speaker names*       |   65.07   |   64.02   |

Above numbers are the mean values of five random seed runs.

If you want to see more training test details, check out `./results/`

If you want to download the trained checkpoints and stuff, then [here](https://surfdrive.surf.nl/files/index.php/s/khREwk4MUI7MSnO/download) is where you can download them. It's a pretty big zip file.

## Deployment

[Watch a demo video!](https://youtu.be/qbr7fNd6J28)

### Huggingface

We have released our models on huggingface:

- [emoberta-base](https://huggingface.co/tae898/emoberta-base)
- [emoberta-large](https://huggingface.co/tae898/emoberta-large)

They are based on [RoBERTa-base](https://huggingface.co/roberta-base) and [RoBERTa-large](https://huggingface.co/roberta-large), respectively. They were trained on [both MELD and IEMOCAP datasets](utterance-ordered-MELD_IEMOCAP.json). Our deployed models are neither speaker-aware nor take previous utterances into account, meaning that it only classifies one utterance at a time without the speaker information (e.g., "I love you").

### Flask app

You can either run the Flask RESTful server app as a docker container or just as a python script.

1. Running the app as a docker container **(recommended)**.

   There are four images. Take what you need:

   - `docker run -it --rm -p 10006:10006 tae898/emoberta-base`
   - `docker run -it --rm -p 10006:10006 --gpus all tae898/emoberta-base-cuda`
   - `docker run -it --rm -p 10006:10006 tae898/emoberta-large`
   - `docker run -it --rm -p 10006:10006 --gpus all tae898/emoberta-large-cuda`

1. Running the app in your python environment:

   This method is less recommended than the docker one.

   Run `pip install -r requirements-deploy.txt` first.<br>
   The [`app.py`](app.py) is a flask RESTful server. The usage is below:

   ```console
   app.py [-h] [--host HOST] [--port PORT] [--device DEVICE] [--model-type MODEL_TYPE]
   ```

   For example:

   ```sh
   python app.py --host 0.0.0.0 --port 10006 --device cpu --model-type emoberta-base
   ```

### Client

Once the app is running, you can send a text to the server. First install the necessary packages: `pip install -r requirements-client.txt`, and the run the [client.py](client.py). The usage is as below:

```console
client.py [-h] [--url-emoberta URL_EMOBERTA] --text TEXT
```

For example:

```sh
python client.py --text "Emotion recognition is so cool\!"
```

will give you:

```json
{
    "neutral": 0.0049800905,
    "joy": 0.96399665,
    "surprise": 0.018937444,
    "anger": 0.0071516023,
    "sadness": 0.002021492,
    "disgust": 0.001495996,
    "fear": 0.0014167271
}
```

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
