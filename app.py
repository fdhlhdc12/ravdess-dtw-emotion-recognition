import streamlit as st
import librosa
import pickle
import tempfile

from dtaidistance import dtw
from collections import Counter

# =========================
# LOAD DATA TRAINING
# =========================

with open("train_features.pkl", "rb") as f:
    train_features = pickle.load(f)

with open("train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)

# =========================
# MFCC FEATURE EXTRACTION
# =========================

def extract_mfcc(file_path):

    signal, sr = librosa.load(
        file_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    return mfcc.T

# =========================
# DTW DISTANCE
# =========================

def dtw_distance(a, b):

    return dtw.distance(
        a.flatten(),
        b.flatten()
    )

# =========================
# KNN-DTW CLASSIFIER
# =========================

def predict_knn_dtw(
    test_mfcc,
    train_features,
    train_labels,
    k=3
):

    distances = []

    for feature, label in zip(
        train_features,
        train_labels
    ):

        dist = dtw_distance(
            test_mfcc,
            feature
        )

        distances.append(
            (dist, label)
        )

    distances.sort()

    nearest = distances[:k]

    nearest_labels = [
        label for _, label in nearest
    ]

    prediction = Counter(
        nearest_labels
    ).most_common(1)[0][0]

    return prediction

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="centered"
)

st.title("🎤 Speech Emotion Recognition")
st.write(
    "Klasifikasi emosi suara menggunakan MFCC, DTW, dan KNN pada dataset RAVDESS."
)

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    with st.spinner(
        "Melakukan klasifikasi..."
    ):

        mfcc_test = extract_mfcc(
            temp_path
        )

        prediction = predict_knn_dtw(
            mfcc_test,
            train_features,
            train_labels,
            k=3
        )

    st.success(
        f"Prediksi Emosi: {prediction.upper()}"
    )
