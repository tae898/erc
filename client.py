"""
This is just a simple client example. Hack it as much as you want. 
"""
import argparse
import logging

import jsonpickle
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_text(text: str, url_emoberta: str) -> None:
    """Send data to the flask server.

    Args
    ----
    text: raw text
    url_emoberta: e.g., http://127.0.0.1:10006/

    """
    data = {"text": text}

    logging.debug("sending text to server...")
    data = jsonpickle.encode(data)
    response = requests.post(url_emoberta, json=data)
    logging.info(f"got {response} from server!...")
    print(response.text)
    response = jsonpickle.decode(response.text)

    logging.info(f"emoberta results: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify room type")
    parser.add_argument("--url-emoberta", type=str, default="http://127.0.0.1:10006/")
    parser.add_argument("--text", type=str, required=True)

    args = vars(parser.parse_args())

    run_text(**args)
