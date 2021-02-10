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
    gdown --id 1DVBuVLTQuq8hi2EyRjDD-smBMINKmJ5Q
    # IEMOCAP
    gdown --id 12UET7i4GBGQYo6fjFbgqBSed2lQJ4087
    # AFEW
    gdown --id 1hF2kMcM5XmxjSrJZmH8DEqnt9xBgdqkb
    # CAER
    gdown --id 1jbA69tQhT0ftnMzLcEyA2P07wFC9rQ3o

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
