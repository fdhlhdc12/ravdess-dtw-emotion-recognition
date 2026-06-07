import librosa
import numpy as np
import os
import pandas as pd

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_label(filepath):
    filename = os.path.basename(filepath)
    parts = filename.replace('.wav', '').split('-')
    return EMOTION_MAP[parts[2]]

def extract_features(filepath, n_mfcc=13):
    """Ekstrak MFCC sebagai time series untuk DTW, + statistik untuk klasifikasi biasa"""
    y, sr = librosa.load(filepath, duration=3.0, offset=0.5)
    
    # MFCC time series (untuk DTW)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape: (13, T)
    
    # Fitur statistik dari MFCC (untuk fitur vektor)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    
    # Fitur tambahan
    chroma     = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel        = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    contrast   = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    zcr        = np.mean(librosa.feature.zero_crossing_rate(y))
    
    feat_vector = np.concatenate([mfcc_mean, mfcc_std, chroma, mel, contrast, [zcr]])
    
    return feat_vector, mfcc  # kembalikan keduanya

def build_dataset(data_dir):
    X, y, mfcc_series = [], [], []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.wav'):
                path = os.path.join(root, f)
                try:
                    feat, mfcc = extract_features(path)
                    label = extract_label(path)
                    X.append(feat)
                    y.append(label)
                    mfcc_series.append(mfcc)  # simpan untuk DTW
                except Exception as e:
                    print(f"Error {f}: {e}")
    return np.array(X), np.array(y), mfcc_series
