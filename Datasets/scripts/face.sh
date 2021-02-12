#!/usr/bin/env bash

# https://github.com/deepinsight/insightface
# THIS REPO IS AMAZING!!

if [ $1 = "download" ]; then
    echo downloading ...
    python3 -m pip install gdown

    rm -f MELD-faces.zip
    rm -f IEMOCAP-faces.zip
    rm -f AFEW-faces.zip
    rm -f CAER-faces.zip

    # MELD
    gdown --id 1xJ65HyNHmo-HUxRNIRLYzN0KL0RmOLsP
    # IEMOCAP
    gdown --id 1ov83wtza25M2h3QFuRXpswCzYivswktL
    # AFEW
    gdown --id 1e2INqhdSXyG0fBEKj5yjr0Jwy4p-cgi5
    # CAER
    gdown --id 1GMmMdrIMqacrUY5HC9DS3Na8HLoy9OD0

    mkdir -p MELD
    mkdir -p IEMOCAP
    mkdir -p AFEW
    mkdir -p CAER

    unzip -o MELD-faces.zip -d MELD/
    unzip -o IEMOCAP-faces.zip -d IEMOCAP/
    unzip -o AFEW-faces.zip -d AFEW/
    unzip -o CAER-faces.zip -d CAER/

    rm -f MELD-faces.zip
    rm -f IEMOCAP-faces.zip
    rm -f AFEW-faces.zip
    rm -f CAER-faces.zip

fi

if [ $1 = "compute" ]; then
    echo computing ...
    python3 scripts/extract-faces.py --num-jobs=$2 --gpu-id=$3
fi
