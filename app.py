"""Emoberta app"""
import argparse
import logging
import os

import jsonpickle
import torch
from flask import Flask, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------- GLOBAL VARIABLES ---------------------- #
emotions = [
    "neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear",
]
id2emotion = {idx: emotion for idx, emotion in enumerate(emotions)}

tokenizer = None
model = None
device = None

app = Flask(__name__)
# --------------------------------------------------------------- #


def load_tokenizer_model(model_type: str, device_: str) -> None:
    """Load tokenizer and model.

    Args
    ----
    model_type: Should be either "emoberta-base" or "emoberta-large"
    device_: "cpu" or "cuda"

    """
    if "large" in model_type.lower():
        model_type = "emoberta-large"
    elif "base" in model_type.lower():
        model_type = "emoberta-base"
    else:
        raise ValueError(
            f"{model_type} is not a valid model type! Should be 'base' or 'large'."
        )

    if not os.path.isdir(model_type):
        model_type = f"tae898/{model_type}"

    global device
    device = device_
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    global model
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    model.eval()
    model.to(device)


@app.route("/", methods=["POST"])
def run_emoberta():
    """Receive everything in json!!!"""
    app.logger.debug("Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    text = data["text"]

    app.logger.info(f"raw text received: {text}")

    tokens = tokenizer(text, truncation=True)

    tokens["input_ids"] = torch.tensor(tokens["input_ids"]).view(1, -1).to(device)
    tokens["attention_mask"] = (
        torch.tensor(tokens["attention_mask"]).view(1, -1).to(device)
    )

    outputs = model(**tokens)
    outputs = torch.softmax(outputs["logits"].detach().cpu(), dim=1).squeeze().numpy()
    outputs = {id2emotion[idx]: prob.item() for idx, prob in enumerate(outputs)}
    app.logger.info(f"prediction: {outputs}")

    response = jsonpickle.encode(outputs)
    app.logger.info("json-pickle is done.")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="emoberta app.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host ip address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10006,
        help="port number",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="emoberta-base",
        help="should be either emoberta-base or emoberta-large",
    )

    args = parser.parse_args()
    load_tokenizer_model(args.model_type, args.device)

    app.run(host=args.host, port=args.port)
