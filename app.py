import librosa
import numpy as np

def extract_feature(file_path):

    y, sr = librosa.load(
        file_path,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    feature = np.mean(
        mfcc.T,
        axis=0
    )

    return feature
