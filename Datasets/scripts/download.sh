#!/usr/bin/env bash

for arg in "$@"; do
    echo $arg
    if [ $arg = "IEMOCAP" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/EcIkP4iRzoJpBzR/download"
        FILENAME="IEMOCAP.zip"
    elif [ $arg = "MELD" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/nnjxH1oboRN3996/download"
        FILENAME="MELD.zip"
    elif [ $arg = "AFEW" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/Y90LClcdqqrRAhz/download"
        FILENAME="AFEW.zip"
    elif [ $arg = "CAER" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/TrgLYrGFwmBawBi/download"
        FILENAME="CAER.zip"
    else
        echo "Currently only IEMOCAP, MELD, AFEW, and CAER datasets are supported"
        continue
    fi
    wget -O $FILENAME $LINK
done

for FILE in "IEMOCAP.zip" "MELD.zip" "AFEW.zip" "CAER.zip"
do
    if test -f "$FILE"; then
        echo $FILE exists
        DS=$(basename -- "$FILE")
        extension="${DS##*.}"
        DS="${DS%.*}"
        unzip -o $FILE -d ./Datasets/
        rm $FILE
    fi
done