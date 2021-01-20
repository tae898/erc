# meld2gmrc

This directory includes jupyter notebook with which you can save MELD friends videos into GMRC annotation format. Running `meld2gmrc.ipynb` will give you the GMRC annotations, which you can read from the GUI app. You can run it yourself, or download from
```
gdown --id 1c13rqBWRoJ0boJCY2qvLv7vQ0VfqE7-8
```

Below is an example.

![meld2gmrc example](meld2gmrc.png)
> Age (a floating point number from 0 to 100), gender (femaleness on a scale from 0 to 1 where 1 being "absolute" female), and the face recognition) are predicted by machine, not human! See https://github.com/leolani/cltl-face-all for more details.

## Things to note

- The unit of the unix time stamps is milliseconds.
- Probably this is not perfect (work in progress).

## Requirements

1. You have to run this on your local machine. This is intended since you'll have to run the webapp locally to test it anyways. I only tested it on Python3.7.9, x86-64 Ubuntu machine. It'll probably work fine on a Mac too. On your local machine, install the requriements by
    ```
    pip install -r requirements.txt
    ```
    I highly recommend you to run above command in your virtual python environment.


## jupyter notebooks

- `meld2gmrc.ipynb`

  Run this locally to do everything at one go (You stil have to manually load the GUI webapp later though.)

## Authors

- Taewoon Kim (t.kim@vu.nl)
