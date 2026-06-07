```python
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib

from feature_extraction import extract_feature

# =====================================================
# LOAD MODEL
# =====================================================

knn_model = joblib.load("model_knn.pkl")
svm_model = joblib.load("model_svm.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Speech Emotion Recognition")

st.markdown(
    """
    Klasifikasi emosi suara menggunakan:

    - MFCC Feature Extraction
    - K-Nearest Neighbor (KNN)
    - Support Vector Machine (SVM)
    """
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

# =====================================================
# MAIN PROCESS
# =====================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    st.write(
        "File Type:",
        uploaded_file.type
    )

    # -----------------------------------------
    # SAVE TEMP AUDIO
    # -----------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())

        audio_path = tmp.name

    # -----------------------------------------
    # LOAD AUDIO
    # -----------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # -----------------------------------------
    # WAVEFORM & SPECTROGRAM
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Waveform")

        fig_wave, ax_wave = plt.subplots()

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        st.pyplot(fig_wave)

    with col2:

        st.subheader("Spectrogram")

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)),
            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots()

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        plt.colorbar(img)

        st.pyplot(fig_spec)

    # -----------------------------------------
    # AUDIO INFO
    # -----------------------------------------

    st.subheader("Audio Information")

    st.write(
        f"Duration : {duration:.2f} seconds"
    )

    st.write(
        f"Sample Rate : {sr}"
    )

    # -----------------------------------------
    # FEATURE EXTRACTION
    # -----------------------------------------

    feature = extract_feature(
        audio_path
    )

    feature = feature.reshape(1, -1)

    feature = scaler.transform(
        feature
    )

    # -----------------------------------------
    # KNN
    # -----------------------------------------

    pred_knn = knn_model.predict(
        feature
    )

    emotion_knn = encoder.inverse_transform(
        pred_knn
    )[0]

    # -----------------------------------------
    # SVM
    # -----------------------------------------

    pred_svm = svm_model.predict(
        feature
    )

    emotion_svm = encoder.inverse_transform(
        pred_svm
    )[0]

    probs = svm_model.predict_proba(
        feature
    )[0]

    confidence = np.max(
        probs
    )

    # -----------------------------------------
    # WARNING
    # -----------------------------------------

    if confidence < 0.50:

        st.warning(
            "Low confidence prediction. The uploaded audio may differ from training data."
        )

    # -----------------------------------------
    # RESULT
    # -----------------------------------------

    st.subheader("Prediction Result")

    col3, col4 = st.columns(2)

    with col3:

        st.success(
            f"KNN Prediction : {emotion_knn.upper()}"
        )

    with col4:

        st.success(
            f"SVM Prediction : {emotion_svm.upper()} ({confidence:.2%})"
        )

    # -----------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------

    if emotion_knn == emotion_svm:

        st.success(
            "✅ Both models agree"
        )

    else:

        st.warning(
            "⚠ KNN and SVM produce different predictions"
        )

    # -----------------------------------------
    # FINAL PREDICTION
    # -----------------------------------------

    st.subheader("Final Prediction")

    st.info(
        emotion_svm.upper()
    )

    # -----------------------------------------
    # PROBABILITY CHART
    # -----------------------------------------

    st.subheader(
        "Emotion Probability"
    )

    prob_df = pd.DataFrame({
        "Emotion": encoder.classes_,
        "Probability": probs
    })

    st.bar_chart(
        prob_df.set_index(
            "Emotion"
        )
    )

    # -----------------------------------------
    # DISCLAIMER
    # -----------------------------------------

    st.caption(
        """
        Model trained using the RAVDESS dataset.
        Predictions on external recordings may have lower accuracy.
        """
    )
```
