#!/usr/bin/env bash

if [ $1 = "download" ]; then
    echo downloading ...
fi

if [ $1 = "compute" ]; then
    echo computing ...
    python3 scripts/extract-faces.py --num-jobs=$2 --gpu-id=$3
fi
