#!/usr/bin/env bash

mkdir -p DEBUG

FILE=MELD.Raw.tar.gz
if test -f "$FILE"; then
    echo "$FILE exists."
    tar -zxvf $FILE
    rm $FILE

    cd MELD.Raw
    FILE=train.tar.gz
    tar -zxvf $FILE
    rm $FILE

    FILE=dev.tar.gz
    tar -zxvf $FILE
    rm $FILE

    FILE=test.tar.gz
    tar -zxvf $FILE
    rm $FILE

    cd ..

    rm -rf DEBUG/MELD.Raw
    mv -f MELD.Raw DEBUG/

    mkdir -p MELD/raw-videos/train
    for filename in DEBUG/MELD.Raw/train_splits/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename MELD/raw-videos/train/$f
    done

    mkdir -p MELD/raw-videos/val
    for filename in DEBUG/MELD.Raw/dev_splits_complete/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename MELD/raw-videos/val/$f
    done

    mkdir -p MELD/raw-videos/test
    for filename in DEBUG/MELD.Raw/output_repeated_splits_test/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename MELD/raw-videos/test/$f
    done

    mkdir -p MELD/raw-audios/train
    for filename in DEBUG/MELD.Raw/train_splits/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .mp4).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a MELD/raw-audios/train/$f
    done

    mkdir -p MELD/raw-audios/val
    for filename in DEBUG/MELD.Raw/dev_splits_complete/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .mp4).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a MELD/raw-audios/val/$f
    done

    mkdir -p MELD/raw-audios/test
    for filename in DEBUG/MELD.Raw/output_repeated_splits_test/*.mp4; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .mp4).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a MELD/raw-audios/test/$f
    done

    python3 scripts/convert-cp1252-to-utf8.py
    mkdir -p MELD/raw-texts/train
    mkdir -p MELD/raw-texts/val
    mkdir -p MELD/raw-texts/test

    python3 scripts/meld-utterance.py
fi

FILE=IEMOCAP_full_release.tar.gz
if test -f "$FILE"; then
    echo "$FILE exists."
    tar -zxvf $FILE
    rm $FILE

    rm -rf DEBUG/IEMOCAP_full_release
    mv -f IEMOCAP_full_release DEBUG/
fi

FILE=EmotiW_2018.zip
if test -f "$FILE"; then
    echo "$FILE exists."
    mkdir -p AFEW
    unzip -o $FILE -d AFEW
    rm $FILE

    cd AFEW
    mkdir -p Train
    FILE=Train_AFEW.zip
    unzip -o $FILE -d Train
    rm $FILE

    mkdir -p Val
    FILE=Val_AFEW.zip
    unzip -o $FILE -d Val
    rm $FILE

    cd Test
    FILE=OneDrive-2018-06-22.zip
    unzip -o $FILE
    rm $FILE

    FILE=LBPTOP.zip
    unzip -o $FILE
    rm $FILE

    FILE=Test_2017_Faces_Distribute.zip
    unzip -o $FILE
    rm $FILE

    FILE=Test_2017_points_distribute.zip
    unzip -o $FILE
    rm $FILE

    FILE=Test_vid_Distribute.rar
    unrar x -o+ $FILE
    rm $FILE

    cd ../..

    rm -rf DEBUG/AFEW
    mv -f AFEW DEBUG/

    mkdir -p AFEW/raw-videos/train
    for filename in DEBUG/AFEW/Train/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename AFEW/raw-videos/train/$f
    done

    mkdir -p AFEW/raw-videos/val
    for filename in DEBUG/AFEW/Val/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename AFEW/raw-videos/val/$f
    done

    mkdir -p AFEW/raw-videos/test
    for filename in DEBUG/AFEW/Test/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename AFEW/raw-videos/test/$f
    done

    mkdir -p AFEW/raw-audios/train
    for filename in DEBUG/AFEW/Train/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .avi).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/train/$f
    done

    mkdir -p AFEW/raw-audios/val
    for filename in DEBUG/AFEW/Val/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .avi).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/val/$f
    done

    mkdir -p AFEW/raw-audios/test
    for filename in DEBUG/AFEW/Test/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- "$filename" .avi).mp3"
        ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/test/$f
    done

    python3 scripts/AFEW-label.py
fi

FILE=CAER.zip
if test -f "$FILE"; then
    echo "$FILE exists."
    unzip $FILE
    rm $FILE

    rm -rf DEBUG/CAER
    mv -f CAER DEBUG/

    mkdir -p CAER/raw-videos/train
    for filename in DEBUG/CAER/train/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename

        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        echo $f
        ln -sf $filename CAER/raw-videos/train/$f
    done

    mkdir -p CAER/raw-videos/val
    for filename in DEBUG/CAER/validation/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename

        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        echo $f
        ln -sf $filename CAER/raw-videos/val/$f
    done

    mkdir -p CAER/raw-videos/test
    for filename in DEBUG/CAER/test/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename

        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        echo $f
        ln -sf $filename CAER/raw-videos/test/$f
    done

    mkdir -p CAER/raw-audios/train
    for filename in DEBUG/CAER/train/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        f="$(basename -- "$f" .avi).mp3"
        echo $f
        ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/train/$f
    done

    mkdir -p CAER/raw-audios/val
    for filename in DEBUG/CAER/validation/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        f="$(basename -- "$f" .avi).mp3"
        echo $f
        ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/val/$f
    done

    mkdir -p CAER/raw-audios/test
    for filename in DEBUG/CAER/test/*/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
        emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
        echo $emotion
        f="$(basename -- $filename)"
        f="${emotion}-${f}"
        f="$(basename -- "$f" .avi).mp3"
        echo $f
        ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/test/$f
    done

    python3 scripts/CAER-label.py
fi
