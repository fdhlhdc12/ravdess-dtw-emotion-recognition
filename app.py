import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib

from feature_extraction_ml import extract_feature_ml

knn_model = joblib.load("model_knn.pkl")
svm_model = joblib.load("model_svm.pkl")

encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

with st.sidebar:

    st.image(
        "assets/logo_kampus.png",
        width=180
    )

    st.markdown("""
    ### 📊 Dataset

    RAVDESS

    🎭 24 Actors

    🎤 2880 Audio Files

    😊 8 Emotions
    """)

    st.markdown("---")

    st.markdown("""
    ### 🤖 Models

    ✅ KNN + GridSearchCV

    ✅ SVM + GridSearchCV
    """)

st.image(
    "assets/banner.png",
    width="stretch"
)

st.markdown("""
### 🎤 Speech Emotion Recognition

Detect human emotions from speech using
MFCC + Delta + Delta² Features.

KNN Accuracy : 87.15%

SVM Accuracy : 90.28%
""")

uploaded_file = st.file_uploader(
    "Upload Audio File",
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

    feature = extract_feature_ml(
        audio_path
    )

    feature = feature.reshape(
        1,
        -1
    )

    feature = scaler.transform(
        feature
    )

    pred_knn = knn_model.predict(
        feature
    )

    pred_svm = svm_model.predict(
        feature
    )

    emotion_knn = encoder.inverse_transform(
        pred_knn
    )[0]

    emotion_svm = encoder.inverse_transform(
        pred_svm
    )[0]

    probs = svm_model.predict_proba(
        feature
    )[0]

    confidence = np.max(
        probs
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

    c2.metric(
        "Sample Rate",
        sr
    )

    c3.metric(
        "Confidence",
        f"{confidence:.2%}"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_wave, ax_wave = plt.subplots(
            figsize=(6,3)
        )

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        st.pyplot(
            fig_wave
        )

    with col2:

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(y)
            ),
            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots(
            figsize=(6,3)
        )

        librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        st.pyplot(
            fig_spec
        )

    r1, r2 = st.columns(2)

    with r1:
        st.success(
            f"KNN : {emotion_knn.upper()}"
        )

    with r2:
        st.success(
            f"SVM : {emotion_svm.upper()}"
        )

    st.subheader(
        "🏆 Final Prediction"
    )

    st.info(
        emotion_svm.upper()
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

    fig_pie, ax_pie = plt.subplots(
        figsize=(4,4)
    )

    ax_pie.pie(
        probs,
        labels=encoder.classes_,
        autopct="%1.1f%%"
    )

    st.pyplot(
        fig_pie
    )
