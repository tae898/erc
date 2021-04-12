#!/usr/bin/env bash

for arg in "$@"; do
    echo $arg
    if [ $arg = "IEMOCAP" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/EcIkP4iRzoJpBzR/download"
        FILENAME="IEMOCAP.zip"
    elif [ $arg = "MELD" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/nnjxH1oboRN3996/download"
        FILENAME="MELD.zip"
    elif [ $arg = "EmoryNLP" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/PqcSGwnWViOKv1o/download"
        FILENAME="EmoryNLP.zip"
    elif [ $arg = "DailyDialog" ]; then
        LINK="https://surfdrive.surf.nl/files/index.php/s/xdv2Lmwo2H32rTN/download"
        FILENAME="DailyDialog.zip"
    else
        echo "Currently only IEMOCAP, MELD, EmoryNLP, and DailyDialog datasets are supported"
        continue
    fi
    wget -O $FILENAME $LINK
done

for FILE in "IEMOCAP.zip" "MELD.zip" "EmoryNLP.zip" "DailyDialog.zip"
do
    if test -f "$FILE"; then
        echo $FILE exists
        DS=$(basename -- "$FILE")
        extension="${DS##*.}"
        DS="${DS%.*}"
        unzip -o $FILE -d ./
        rm $FILE
    fi
done