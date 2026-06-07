import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib

from feature_extraction import extract_feature

# ======================
# Load Models
# ======================

knn_model = joblib.load("model_knn.pkl")
svm_model = joblib.load("model_svm.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Page Config
# ======================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Speech Emotion Recognition")
st.markdown("MFCC + KNN vs MFCC + SVM")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=[
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac"
    ]
)

if uploaded_file is not None:

    st.audio(uploaded_file)
    st.write(
    "File Type:",
    uploaded_file.type
)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())

        audio_path = tmp.name

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # ======================
    # Waveform
    # ======================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Waveform")

        fig, ax = plt.subplots()

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        st.pyplot(fig)

    # ======================
    # Spectrogram
    # ======================

    with col2:

        st.subheader("Spectrogram")

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)),
            ref=np.max
        )

        fig2, ax2 = plt.subplots()

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax2
        )

        plt.colorbar(img)

        st.pyplot(fig2)

    # ======================
    # Audio Information
    # ======================

    st.subheader("Audio Information")

    st.write(f"Duration : {duration:.2f} sec")
    st.write(f"Sample Rate : {sr}")

    # ======================
    # Feature Extraction
    # ======================

    feature = extract_feature(audio_path)

    feature = feature.reshape(1,-1)

    feature = scaler.transform(feature)

    # ======================
    # KNN Prediction
    # ======================

    pred_knn = knn_model.predict(feature)

    emotion_knn = encoder.inverse_transform(
        pred_knn
    )[0]

    # ======================
    # SVM Prediction
    # ======================

    pred_svm = svm_model.predict(feature)

    emotion_svm = encoder.inverse_transform(
        pred_svm
    )[0]

    probs = svm_model.predict_proba(
    feature
)[0]

confidence = np.max(probs)

if confidence < 0.50:
    st.warning(
        "Low confidence prediction. Audio may differ from training data."
    )

# ======================
# Result
# ======================

st.subheader("Prediction Result")

col3, col4 = st.columns(2)

with col3:

    st.success(
        f"KNN : {emotion_knn.upper()}"
    )

with col4:

    st.success(
        f"SVM : {emotion_svm.upper()} ({confidence:.2%})"
    )
    # ======================
    # Probability
    # ======================

    st.subheader("Emotion Probability")

    prob_df = pd.DataFrame({

        "Emotion": encoder.classes_,
        "Probability": probs

    })

    st.bar_chart(
        prob_df.set_index(
            "Emotion"
        )
    )
