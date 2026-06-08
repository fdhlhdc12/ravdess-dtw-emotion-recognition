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

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    mfcc_mean = np.mean(
        mfcc.T,
        axis=0
    )

    delta_mean = np.mean(
        delta.T,
        axis=0
    )

    delta2_mean = np.mean(
        delta2.T,
        axis=0
    )

    feature = np.hstack([
        mfcc_mean,
        delta_mean,
        delta2_mean
    ])

    return feature
