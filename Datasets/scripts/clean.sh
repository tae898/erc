#!/usr/bin/env bash

# sudo apt-get install libsndfile1-dev
pip install av librosa
mkdir -p DEBUG

FILE=MELD.Raw.tar.gz
if test -f "$FILE"; then
    rm -rf MELD
    echo "$FILE exists."
    tar -zxvf $FILE

    cd MELD.Raw
    FILE_=train.tar.gz
    tar -zxvf $FILE_
    rm $FILE_

    FILE_=dev.tar.gz
    tar -zxvf $FILE_
    rm $FILE_

    FILE_=test.tar.gz
    tar -zxvf $FILE_
    rm $FILE_

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

    python3 scripts/MELD-utterance.py

    rm $FILE

fi

FILE=IEMOCAP_full_release.tar.gz
if test -f "$FILE"; then
    rm -rf IEMOCAP
    echo "$FILE exists."
    tar -zxvf $FILE

    rm -rf DEBUG/IEMOCAP_full_release
    mv -f IEMOCAP_full_release DEBUG/

    rm -f ._IEMOCAP_full_release

    mkdir -p IEMOCAP/raw-videos
    for filename in DEBUG/IEMOCAP_full_release/*/dialog/avi/DivX/*.avi; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename IEMOCAP/raw-videos/$f
    done

    mkdir -p IEMOCAP/raw-texts
    for filename in DEBUG/IEMOCAP_full_release/*/dialog/transcriptions/*.txt; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename IEMOCAP/raw-texts/$f
    done

    mkdir -p IEMOCAP/raw-audios
    for filename in DEBUG/IEMOCAP_full_release/*/sentences/wav/*/*.wav; do
        filename=$(readlink -f $filename)
        echo $filename
        f="$(basename -- $filename)"
        ln -sf $filename IEMOCAP/raw-audios/$f
    done
    python3 scripts/IEMOCAP-sort-audio.py

    python3 scripts/IEMOCAP-split.py

    python3 scripts/IEMOCAP-split-further.py

    python3 scripts/IEMOCAP-slice-video.py

    rm -f IEMOCAP/raw-videos/*/*.avi

    for filename in IEMOCAP/raw-videos/*/*/*.mp4; do
        videopath=$(readlink -f $filename)
        echo $videopath
        audiopath=${filename/raw-videos/raw-audios}
        audiopath=${audiopath/.mp4/.wav}
        ffmpeg -y -i $videopath -i $audiopath -c:v copy -c:a aac $videopath.mp4
        rm $videopath
        mv $videopath.mp4 $videopath

    done

    mv -f IEMOCAP/raw-audios/train/*/*.wav IEMOCAP/raw-audios/train/
    mv -f IEMOCAP/raw-audios/val/*/*.wav IEMOCAP/raw-audios/val/
    mv -f IEMOCAP/raw-audios/test/*/*.wav IEMOCAP/raw-audios/test/
    find IEMOCAP/raw-audios/ -type d -empty -delete

    mv -f IEMOCAP/raw-videos/train/*/*.mp4 IEMOCAP/raw-videos/train/
    mv -f IEMOCAP/raw-videos/val/*/*.mp4 IEMOCAP/raw-videos/val/
    mv -f IEMOCAP/raw-videos/test/*/*.mp4 IEMOCAP/raw-videos/test/
    find IEMOCAP/raw-videos/ -type d -empty -delete

    mv -f IEMOCAP/raw-texts/train/*/*.json IEMOCAP/raw-texts/train/
    mv -f IEMOCAP/raw-texts/val/*/*.json IEMOCAP/raw-texts/val/
    mv -f IEMOCAP/raw-texts/test/*/*.json IEMOCAP/raw-texts/test/

    find IEMOCAP/raw-texts/ -type d -empty -delete

    rm $FILE
fi

FILE=EmotiW_2018.zip
if test -f "$FILE"; then
    rm -rf AFEW
    echo "$FILE exists."
    mkdir -p AFEW
    unzip -o $FILE -d AFEW

    cd AFEW
    mkdir -p Train
    FILE_=Train_AFEW.zip
    unzip -o $FILE_ -d Train
    rm $FILE_

    mkdir -p Val
    FILE_=Val_AFEW.zip
    unzip -o $FILE_ -d Val
    rm $FILE_

    cd Test
    FILE_=OneDrive-2018-06-22.zip
    unzip -o $FILE_
    rm $FILE_

    FILE_=LBPTOP.zip
    unzip -o $FILE_
    rm $FILE_

    FILE_=Test_2017_Faces_Distribute.zip
    unzip -o $FILE_
    rm $FILE_

    FILE_=Test_2017_points_distribute.zip
    unzip -o $FILE_
    rm $FILE_

    FILE_=Test_vid_Distribute.rar
    unrar x -o+ $FILE_
    rm $FILE_

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

    rm $FILE

fi

FILE=CAER.zip
if test -f "$FILE"; then
    rm -rf CAER
    echo "$FILE exists."
    unzip $FILE

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
        datatype=train
        f="${datatype}-${f}"
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
        datatype=train
        f="${datatype}-${f}"
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
        datatype=train
        f="${datatype}-${f}"
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
        datatype=train
        f="${datatype}-${f}"
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
        datatype=train
        f="${datatype}-${f}"
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
        datatype=train
        f="${datatype}-${f}"
        f="$(basename -- "$f" .avi).mp3"
        echo $f
        ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/test/$f
    done

    python3 scripts/CAER-label.py
    rm $FILE

fi

echo DONE
