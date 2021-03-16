#!/usr/bin/env bash

# sudo apt-get install libsndfile1-dev
pip install av librosa
mkdir -p DEBUG

if test -f "MELD.Raw.tar.gz"; then
    rm -rf MELD
    echo "Processing the MELD dataset ..."
    tar -zxvf MELD.Raw.tar.gz

    cd MELD.Raw
    for FILE in "train.tar.gz" "dev.tar.gz" "test.tar.gz"; do
        tar -xzvf $FILE
        rm $FILE
    done
    cd ..

    rm -rf DEBUG/MELD.Raw
    mv -f MELD.Raw DEBUG/

    mv DEBUG/MELD.Raw/train_splits DEBUG/MELD.Raw/train
    mv DEBUG/MELD.Raw/dev_splits_complete DEBUG/MELD.Raw/val
    mv DEBUG/MELD.Raw/output_repeated_splits_test DEBUG/MELD.Raw/test

    for SPLIT in "train" "val" "test"; do
        mkdir -p "MELD/raw-videos/${SPLIT}"
        for filename in DEBUG/MELD.Raw/$SPLIT/*.mp4; do
            filename=$(readlink -f $filename)
            echo "Processing ${filename} ..."
            f="$(basename -- $filename)"
            echo $f
            ln -sf "${filename}" "MELD/raw-videos/${SPLIT}/${f}"
        done
    done

    for SPLIT in "train" "val" "test"; do
        mkdir -p "MELD/raw-audios/${SPLIT}"
        for filename in DEBUG/MELD.Raw/$SPLIT/*.mp4; do
            filename=$(readlink -f $filename)
            echo "Processing ${filename} ..."
            f="$(basename -- $filename .mp4).wav"
            echo $f
            # ffmpeg -y -i $filename -q:a 0 -map a MELD/raw-audios/$SPLIT/$f
            ffmpeg -i $filename -acodec pcm_s16le -ac 1 -ar 22050 MELD/raw-audios/$SPLIT/$f
        done
    done

    python3 scripts/convert-cp1252-to-utf8.py
    mkdir -p MELD/raw-texts/train
    mkdir -p MELD/raw-texts/val
    mkdir -p MELD/raw-texts/test

    python3 scripts/MELD-utterance.py

    rm "MELD.Raw.tar.gz"
fi

if test -f "IEMOCAP_full_release.tar.gz"; then
    rm -rf IEMOCAP
    echo "Processing the IEMOCAP dataset ..."
    tar -zxvf IEMOCAP_full_release.tar.gz

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
    
    python3 scripts/IEMOCAP-label.py

    rm "IEMOCAP_full_release.tar.gz"
fi


if test -f "emotion-detection-emotion-detection-1.0.tar.gz"; then
    rm -rf EmoryNLP
    echo "Processing the EmoryNLP dataset ..."
    tar -zxvf emotion-detection-emotion-detection-1.0.tar.gz

    rm -rf DEBUG/emotion-detection-emotion-detection-1.0
    mv -f emotion-detection-emotion-detection-1.0 DEBUG/
    mkdir -p EmoryNLP/raw-texts

    python3 scripts/process-EmoryNLP.py

    rm "emotion-detection-emotion-detection-1.0.tar.gz"
fi

if test -f "ijcnlp_dailydialog.zip"; then
    rm -rf DailyDialog
    echo "Processing the DailyDialog dataset ..."
    unzip ijcnlp_dailydialog.zip

    cd ijcnlp_dailydialog
    unzip train.zip
    unzip validation.zip
    unzip test.zip
    rm train.zip
    rm validation.zip
    rm test.zip
    cd ..

    rm -rf DEBUG/ijcnlp_dailydialog
    mv -f ijcnlp_dailydialog DEBUG/ijcnlp_dailydialog

    mkdir -p DailyDialog/raw-texts

    python3 scripts/process-DailyDialog.py

    rm "ijcnlp_dailydialog.zip"
fi
# FILE=EmotiW_2018.zip
# if test -f "$FILE"; then
#     rm -rf AFEW
#     echo "$FILE exists."
#     mkdir -p AFEW
#     unzip -o $FILE -d AFEW

#     cd AFEW
#     mkdir -p Train
#     FILE_=Train_AFEW.zip
#     unzip -o $FILE_ -d Train
#     rm $FILE_

#     mkdir -p Val
#     FILE_=Val_AFEW.zip
#     unzip -o $FILE_ -d Val
#     rm $FILE_

#     cd Test
#     FILE_=OneDrive-2018-06-22.zip
#     unzip -o $FILE_
#     rm $FILE_

#     FILE_=LBPTOP.zip
#     unzip -o $FILE_
#     rm $FILE_

#     FILE_=Test_2017_Faces_Distribute.zip
#     unzip -o $FILE_
#     rm $FILE_

#     FILE_=Test_2017_points_distribute.zip
#     unzip -o $FILE_
#     rm $FILE_

#     FILE_=Test_vid_Distribute.rar
#     unrar x -o+ $FILE_
#     rm $FILE_

#     cd ../..

#     rm -rf DEBUG/AFEW
#     mv -f AFEW DEBUG/

#     mkdir -p AFEW/raw-videos/train
#     for filename in DEBUG/AFEW/Train/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- $filename)"
#         ln -sf $filename AFEW/raw-videos/train/$f
#     done

#     mkdir -p AFEW/raw-videos/val
#     for filename in DEBUG/AFEW/Val/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- $filename)"
#         ln -sf $filename AFEW/raw-videos/val/$f
#     done

#     mkdir -p AFEW/raw-videos/test
#     for filename in DEBUG/AFEW/Test/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- $filename)"
#         ln -sf $filename AFEW/raw-videos/test/$f
#     done

#     mkdir -p AFEW/raw-audios/train
#     for filename in DEBUG/AFEW/Train/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- "$filename" .avi).mp3"
#         ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/train/$f
#     done

#     mkdir -p AFEW/raw-audios/val
#     for filename in DEBUG/AFEW/Val/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- "$filename" .avi).mp3"
#         ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/val/$f
#     done

#     mkdir -p AFEW/raw-audios/test
#     for filename in DEBUG/AFEW/Test/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         f="$(basename -- "$filename" .avi).mp3"
#         ffmpeg -y -i $filename -q:a 0 -map a AFEW/raw-audios/test/$f
#     done

#     python3 scripts/AFEW-label.py

#     rm $FILE

# fi

# FILE=CAER.zip
# if test -f "$FILE"; then
#     rm -rf CAER
#     echo "$FILE exists."
#     unzip $FILE

#     rm -rf DEBUG/CAER
#     mv -f CAER DEBUG/

#     mkdir -p CAER/raw-videos/train
#     for filename in DEBUG/CAER/train/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename

#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=train
#         f="${datatype}-${f}"
#         echo $f
#         ln -sf $filename CAER/raw-videos/train/$f
#     done

#     mkdir -p CAER/raw-videos/val
#     for filename in DEBUG/CAER/validation/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename

#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=val
#         f="${datatype}-${f}"
#         echo $f
#         ln -sf $filename CAER/raw-videos/val/$f
#     done

#     mkdir -p CAER/raw-videos/test
#     for filename in DEBUG/CAER/test/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename

#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=test
#         f="${datatype}-${f}"
#         echo $f
#         ln -sf $filename CAER/raw-videos/test/$f
#     done

#     mkdir -p CAER/raw-audios/train
#     for filename in DEBUG/CAER/train/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=train
#         f="${datatype}-${f}"
#         f="$(basename -- "$f" .avi).mp3"
#         echo $f
#         ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/train/$f
#     done

#     mkdir -p CAER/raw-audios/val
#     for filename in DEBUG/CAER/validation/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=val
#         f="${datatype}-${f}"
#         f="$(basename -- "$f" .avi).mp3"
#         echo $f
#         ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/val/$f
#     done

#     mkdir -p CAER/raw-audios/test
#     for filename in DEBUG/CAER/test/*/*.avi; do
#         filename=$(readlink -f $filename)
#         echo $filename
#         emotion="$(echo "$filename" | rev | cut -d '/' -f 2 | rev)"
#         emotion="$(echo "$emotion" | tr '[:upper:]' '[:lower:]')"
#         echo $emotion
#         f="$(basename -- $filename)"
#         f="${emotion}-${f}"
#         datatype=test
#         f="${datatype}-${f}"
#         f="$(basename -- "$f" .avi).mp3"
#         echo $f
#         ffmpeg -y -i $filename -q:a 0 -map a CAER/raw-audios/test/$f
#     done

#     python3 scripts/CAER-label.py
#     rm $FILE

# fi

# echo DONE
