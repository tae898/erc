#!/usr/bin/env bash

# https://github.com/deepinsight/insightface
# THIS REPO IS AMAZING!!

if [ $1 = "download" ]; then
    echo downloading ...

    rm -f MELD-faces.zip
    rm -f IEMOCAP-faces.zip

    wget -O MELD-faces.zip https://surfdrive.surf.nl/files/index.php/s/1CsIG1PM76uIbfh/download
    wget -O IEMOCAP-faces.zip https://surfdrive.surf.nl/files/index.php/s/pGMV6yrZLQS66yu/download

    mkdir -p MELD 
    mkdir -p IEMOCAP

    unzip -o MELD-faces.zip -d MELD/
    unzip -o IEMOCAP-faces.zip -d IEMOCAP/

    rm -f MELD-faces.zip
    rm -f IEMOCAP-faces.zip

fi

if [ $1 = "compute" ]; then
    echo computing ...
    python3 scripts/extract-faces.py --num-jobs=$2 --gpu-id=$3
    rm -rf */face-videos
    python3 scripts/crop-faces.py --num-jobs=$2
fi
