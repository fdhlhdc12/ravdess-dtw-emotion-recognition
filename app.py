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

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

h1 {
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================

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

# ==================================================
# EMOJI
# ==================================================

emotion_icon = {

    "happy":"😊",
    "sad":"😢",
    "angry":"😠",
    "fearful":"😨",
    "calm":"😌",
    "neutral":"😐",
    "disgust":"🤢",
    "surprised":"😲"
}

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📊 Dataset Information")

    st.info("""
    Dataset : RAVDESS

    🎭 24 Actors

    🎤 2880 Audio Files

    😊 8 Emotions
    """)

    st.header("🤖 Models")

    st.success("KNN + GridSearchCV")

    st.success("SVM + GridSearchCV")

    st.success("CNN-LSTM")

# ==================================================
# HEADER
# ==================================================

st.title(
    "🎤 Speech Emotion Recognition System"
)

st.markdown("""
### Human Emotion Detection from Speech

Compare performance of:

- 🤖 KNN + GridSearchCV
- 🤖 SVM + GridSearchCV
- 🧠 CNN-LSTM

Dataset: RAVDESS
""")

# ==================================================
# FILE UPLOADER
# ==================================================

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

# ==================================================
# PROCESS
# ==================================================

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

    # ==============================================
    # LOAD AUDIO
    # ==============================================

    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # ==============================================
    # AUDIO INFO
    # ==============================================

    st.subheader(
        "📈 Audio Information"
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

    # ==============================================
    # WAVEFORM + SPECTROGRAM
    # ==============================================

    col1,col2 = st.columns(2)

    with col1:

        st.subheader(
            "🎵 Waveform"
        )

        fig_wave, ax_wave = plt.subplots()

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        st.pyplot(fig_wave)

    with col2:

        st.subheader(
            "📊 Spectrogram"
        )

        D = librosa.amplitude_to_db(

            np.abs(
                librosa.stft(y)
            ),

            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots()

        img = librosa.display.specshow(

            D,

            sr=sr,

            x_axis='time',

            y_axis='hz',

            ax=ax_spec
        )

        plt.colorbar(img)

        st.pyplot(fig_spec)

    # ==============================================
    # FEATURE ML
    # ==============================================

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

    # ==============================================
    # KNN
    # ==============================================

    pred_knn = knn_model.predict(
        feature_ml
    )

    emotion_knn = encoder.inverse_transform(
        pred_knn
    )[0]

    # ==============================================
    # SVM
    # ==============================================

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

    # ==============================================
    # CNN LSTM
    # ==============================================

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

    # ==============================================
    # CONFIDENCE
    # ==============================================

    st.metric(
        "Confidence Score",
        f"{confidence:.2%}"
    )

    # ==============================================
    # PREDICTIONS
    # ==============================================

    st.subheader(
        "🤖 Prediction Results"
    )

    r1,r2,r3 = st.columns(3)

    with r1:

        st.markdown(f"""
        ### 🤖 KNN

        # {emotion_icon[emotion_knn]}

        ### {emotion_knn.upper()}
        """)

    with r2:

        st.markdown(f"""
        ### 🤖 SVM

        # {emotion_icon[emotion_svm]}

        ### {emotion_svm.upper()}
        """)

    with r3:

        st.markdown(f"""
        ### 🧠 CNN-LSTM

        # {emotion_icon[emotion_cnn]}

        ### {emotion_cnn.upper()}
        """)

    # ==============================================
    # AGREEMENT
    # ==============================================

    if emotion_knn == emotion_svm:

        st.success(
            "✅ KNN and SVM agree"
        )

    else:

        st.warning(
            "⚠ KNN and SVM produce different predictions"
        )

    # ==============================================
    # FINAL PREDICTION
    # ==============================================

    st.markdown("---")

    st.subheader(
        "🏆 Final Prediction"
    )

    st.success(
        f"{emotion_icon[emotion_svm]} {emotion_svm.upper()} ({confidence:.2%})"
    )

    # ==============================================
    # BAR CHART
    # ==============================================

    st.subheader(
        "📊 Emotion Probability"
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

    # ==============================================
    # PIE CHART
    # ==============================================

    st.subheader(
        "🥧 Emotion Distribution"
    )

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie(

        probs,

        labels=encoder.classes_,

        autopct="%1.1f%%"
    )

    st.pyplot(
        fig_pie
    )

    # ==============================================
    # FOOTER
    # ==============================================

    st.caption(
        """
        Speech Emotion Recognition using
        MFCC + Delta + Delta² Features,
        KNN, SVM and CNN-LSTM Models.
        """
    )
