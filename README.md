# cltl-face-all

This python package was initially made to improve robot-human communication, but the use cases can vary. Given an image with a face, useful features are extracted with which robots / agents can better engage in a conversation with humans. 

This python package contains four models to get visual features from human faces:

1. **face detection** gives you bounding boxes around the faces and their probabilities.
2. **face landmark detection** gives you 68 face landmarks. This depends on (1)
3. **age/gender detection** gives you estimated gender and age. This depends on (1) and (2).
4. **face recognition** gives you 512-D face embedding vectors. This depends on (1) and (2).


## Prerequisites

* A x86 Unix or Unix-like machines 
* Python 3.7.9 environment
* (Optional) [Docker Engine](https://docs.docker.com/engine/install/)

## Python package installation

1. Clone this repo

    ```
    git clone https://github.com/leolani/cltl-face-all
    ```

2. Install the requirements (virtual python environment is highly recommended)
    ```
    pip install -r requirements.txt
    ```

3. Go to the directory where this `README.md` is located. Install the `cltl-face-all` repo by running
    ```
    pip install .
    ```


## (Optional) Building and running the docker image 


1. Clone this repo

    ```
    git clone https://github.com/leolani/cltl-face-all
    ```

2. Go to the directory where this `README.md` is located.
    ```
    docker build -t cltl-face-all .
    ```

3. Run the docker container.
    ```
    docker run -p 27004:27004 -it --rm cltl-face-all /bin/bash
    ```

## Usage

In your python environment, import the module `cltl-face-all` to use the classes and functions. Below is a code snippet.

```
from cltl_face_all.face_alignment import FaceDetection
from cltl_face_all.arcface import ArcFace
from cltl_face_all.agegender import AgeGender


ag = AgeGender(device='cpu')
af = ArcFace(device='cpu')
fd = FaceDetection(device='cpu', face_detector='blazeface')
```

## Examples

Go to the `examples` folder to and take a look at some of the jupyter notebooks. 

* Watch [this video](https://youtu.be/4i0s_dnylZ0) for registering your face and webcam demo.
* Watch [this video](https://youtu.be/asJtjhDJ5ZM) for obtaining face features from videos and inference on images.

## Disclaimer

None of the models used were trained by me. I copied the codes and the binary files from already existing repos. The original sources are mentioned in my code.

## TODOs

1. Create a test dataset to set a baseline.
2. Currently both tensorflow and pytorch are used. Stick to one (preferably pytorch) and make it compatible.
3. Better organize the binary weights file downloading. They are stored everywhere at the moment.
4. Find a better face detector. This package supports some face detectors (e.g. sfd, blazeface, and dlib).
5. Clean and readable code.
6. Better docstring. 
7. GPU support.
8. Create a server in docker.
9. Decouple face detection (bounding box) and face landmark detection. They are technically two separate things.
10. Think about extending the visual features from faces to full-sized humans (e.g. human poses)


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
