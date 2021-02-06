import os


def get_unique_dias(list_of_diautts):
    """Get unique dialogues."""
    return sorted(list(set([diautt.split('_')[0] for diautt in list_of_diautts])))


def remove_non_existent(dia_diautts, vid_dir):
    """Remove annotations that don't have videos."""
    dia_diautts_clean = {}

    for dia, diautts in dia_diautts.items():
        foo = []
        for diautt in diautts:
            if os.path.isfile(os.path.join(vid_dir, diautt + '.mp4')):
                foo.append(diautt)
        if len(foo) != 0:
            dia_diautts_clean[dia] = foo

    return dia_diautts_clean


def get_face_dia_diauttts(datasets, PATHS):
    """Get the annotations in {DIALOGUE#: {UTTERANCE#}}"""
    dia_diautts = {}
    for DATASET in ['train', 'dev', 'test']:
        dia_diautts[DATASET] = list(datasets[DATASET].keys())
        dia_diautts[DATASET] = {dia: [diautt for diautt in dia_diautts[DATASET] if dia + '_' in diautt]
                                for dia in get_unique_dias(dia_diautts[DATASET])}

        dia_diautts[DATASET] = remove_non_existent(
            dia_diautts[DATASET], PATHS['FACE_VIDEOS'][DATASET])

    return dia_diautts


def get_face_utts(datasets, PATHS):
    """Get utterances.

    Which utterance belongs to which dialogue is not considered here.

    """
    dia_diautts = get_face_dia_diauttts(datasets, PATHS)

    utts = {DATASET: [diautt for dia, diautts in dia_diautts[DATASET].items()
                      for diautt in diautts] for DATASET in ['train', 'dev', 'test']}

    return utts
