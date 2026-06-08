```python
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib
import tensorflow as tf

from feature_extraction_ml import extract_feature_ml
from feature_extraction_dl import extract_feature_dl

# =====================================================
# LOAD MODELS
# =====================================================

knn_model = joblib.load(
    "model_knn.pkl"
)

svm_model = joblib.load(
    "model_svm.pkl"
)

encoder = joblib.load(
    "label_encoder.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

cnn_lstm_model = tf.keras.models.load_model(
    "cnn_lstm_model.keras"
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

st.title(
    "🎤 Speech Emotion Recognition"
)

st.markdown(
"""
Comparison of:

• KNN + GridSearchCV

• SVM + GridSearchCV

• CNN-LSTM
"""
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Dataset")

    st.info(
    """
    RAVDESS Dataset

    2880 Audio Files

    8 Emotions
    """
    )

# =====================================================
# FILE UPLOAD
# =====================================================

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

# =====================================================
# PROCESS
# =====================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        audio_path = tmp.name

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # ===================================
    # WAVEFORM
    # ===================================

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Waveform")

        fig, ax = plt.subplots()

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        st.pyplot(fig)

    # ===================================
    # SPECTROGRAM
    # ===================================

    with col2:

        st.subheader("Spectrogram")

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(y)
            ),
            ref=np.max
        )

        fig2, ax2 = plt.subplots()

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax2
        )

        plt.colorbar(img)

        st.pyplot(fig2)

    # ===================================
    # AUDIO INFO
    # ===================================

    st.subheader(
        "Audio Information"
    )

    c1,c2 = st.columns(2)

    with c1:

        st.metric(
            "Duration",
            f"{duration:.2f} sec"
        )

    with c2:

        st.metric(
            "Sample Rate",
            sr
        )

    # ===================================
    # MACHINE LEARNING FEATURE
    # ===================================

    feature_ml = extract_feature_ml(
        audio_path
    )

    feature_ml = feature_ml.reshape(
        1,
        -1
    )

    feature_ml = scaler.transform(
        feature_ml
    )

    # ===================================
    # KNN
    # ===================================

    pred_knn = knn_model.predict(
        feature_ml
    )

    emotion_knn = encoder.inverse_transform(
        pred_knn
    )[0]

    # ===================================
    # SVM
    # ===================================

    pred_svm = svm_model.predict(
        feature_ml
    )

    emotion_svm = encoder.inverse_transform(
        pred_svm
    )[0]

    probs = svm_model.predict_proba(
        feature_ml
    )[0]

    confidence = np.max(
        probs
    )

    # ===================================
    # CNN LSTM
    # ===================================

    feature_dl = extract_feature_dl(
        audio_path
    )

    feature_dl = np.expand_dims(
        feature_dl,
        axis=0
    )

    pred_cnn = cnn_lstm_model.predict(
        feature_dl,
        verbose=0
    )

    pred_cnn = np.argmax(
        pred_cnn,
        axis=1
    )

    emotion_cnn = encoder.inverse_transform(
        pred_cnn
    )[0]

    # ===================================
    # RESULTS
    # ===================================

    st.subheader(
        "Prediction Results"
    )

    r1,r2,r3 = st.columns(3)

    with r1:

        st.success(
            f"KNN\n\n{emotion_knn.upper()}"
        )

    with r2:

        st.success(
            f"SVM\n\n{emotion_svm.upper()}"
        )

    with r3:

        st.success(
            f"CNN-LSTM\n\n{emotion_cnn.upper()}"
        )

    # ===================================
    # FINAL PREDICTION
    # ===================================

    st.subheader(
        "Final Prediction"
    )

    st.info(
        f"{emotion_svm.upper()} ({confidence:.2%})"
    )

    # ===================================
    # PROBABILITY
    # ===================================

    st.subheader(
        "SVM Probability"
    )

    prob_df = pd.DataFrame({

        "Emotion":
        encoder.classes_,

        "Probability":
        probs
    })

    st.bar_chart(
        prob_df.set_index(
            "Emotion"
        )
    )
```
