# cltl-face-all

This python package was initially made to improve robot-human communication, but the use cases can vary. Given an image with a face, useful features are extracted with which robots / agents can better engage in a conversation with humans. 

This python package contains four models to get visual features from human faces:

1. **face detection** gives you bounding boxes around the faces and their probabilities.
1. **face landmark detection** gives you 68 face landmarks. This depends on (1)
1. **age/gender detection** gives you estimated gender and age. This depends on (1) and (2).
1. **face recognition** gives you 512-D face embedding vectors. This depends on (1) and (2).


## Prerequisites

* A x86 Unix or Unix-like machines 
* Python 3.6x, 3.7x, or 3.8x

## Python package installation


1. Install the requirements (virtual python environment is highly recommended)
    ```
    pip install -r requirements.txt
    ```

1. Go to the directory where this `README.md` is located. Install the `cltl-face-all` repo by running
    ```
    pip install .
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

* Watch [this video](https://youtu.be/4i0s_dnylZ0) for registering your face and webcam demo.
* Watch [this video](https://youtu.be/asJtjhDJ5ZM) for obtaining face features from videos and inference on images.

## Disclaimer

None of the models used were trained by me. I copied the codes and the binary files from already existing repos. The original sources are mentioned in my code.

## TODOs

1. Create a test dataset to set a baseline.
1. Currently both tensorflow and pytorch are used. Stick to one (preferably pytorch) and make it compatible.
1. Better organize the binary weights file downloading. They are stored everywhere at the moment.
1. Find a better face detector. This package supports some face detectors (e.g. sfd, blazeface, and dlib), but there's gotta be a better one.
1. Include facial emotion detection.
1. Clean and readable code.
1. Better docstring. 
1. GPU support.
1. Create a server in docker.
1. Decouple face detection (bounding box) and face landmark detection. They are technically two separate things.
1. Think about extending the visual features from faces to full-sized humans (e.g. human poses)


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
