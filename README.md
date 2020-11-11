# cltl-facedetection
A face detection (micro) service.
This services expects a numpy array as input (an RGB image) and outputs bounding boxes, probs, and five face landmarks. Currenty the model is implemented in [pytorch-based mtcnn](https://github.com/timesler/facenet-pytorch). The service acts as a server, and you'd have to implement a small client to talk to it. I used [mlsocket](https://github.com/k2sebeom/mlsocket) for this.

## Prerequisites

[Docker Engine](https://docs.docker.com/engine/install/)

I recommend x86 Unix or Unix-like machines. 

## Installation

1. Clone this repo

    ```
    git clone https://github.com/leolani/cltl-facedetection.git
    ```
2. At the root directory of the repo (e.g. `cltl-facedetection`), build a docker image by running
    ```
    docker build -t cltl-facedetection .
    ```
## Usage

For most of the time, CPU might be enough.

### CPU

```
docker run -p 27004:27004 -it --rm cltl-facedetection
```

### GPU

```
docker run -p 27004:27004 -it --rm --gpus all cltl-facedetection
```

If you want to use a GPU, your system has to have a CUDA GPU with nvidia-driver installed. This might not work so well. Follow [Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit). 

## Server-Client Example

1. Start the server in one terminal.

    ```
    docker run -p 27004:27004 -it --rm cltl-facedetection
    ```
2. Open up another terminal. Preferably install a new python virtualenv for this client.

3. Install the necessary packages for this client example.
    ```
    pip install -r requirements_example.txt  
    ```
4. Run the `client.py`
    ```
    python client.py
    ```

[See this video](https://youtu.be/0zYOsTlfPFY), for a step by step guide.
[See this video](https://youtu.be/EIDBSBH1avU), to see how to use what the model outputs.


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors
* Taewoon Kim (t.kim@vu.nl)

## License
[MIT](https://choosealicense.com/licenses/mit/)
