#!/usr/bin/env bash

# https://github.com/deepinsight/insightface
# THIS REPO IS AMAZING!!

if [ $1 = "download" ]; then
    echo downloading ...

    rm -f MELD-faces.zip
    rm -f IEMOCAP-faces.zip
    rm -f AFEW-faces.zip
    rm -f CAER-faces.zip

    wget -O MELD-faces.zip https://surfdrive.surf.nl/files/index.php/s/1CsIG1PM76uIbfh/download
    wget -O IEMOCAP-faces.zip https://surfdrive.surf.nl/files/index.php/s/pGMV6yrZLQS66yu/download
    wget -O AFEW-faces.zip https://surfdrive.surf.nl/files/index.php/s/9Q4ADaEH4SSeKkq/download
    wget -O CAER-faces.zip https://surfdrive.surf.nl/files/index.php/s/ivUyfZGQc7gfqFm/download

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
    rm -rf */face-videos
    python3 scripts/crop-faces.py --num-jobs=$2
fi
