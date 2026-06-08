import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib

from feature_extraction_ml import extract_feature_ml

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

.stApp {
    background-color: #0f172a;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.final-card {
    background-color: #2563eb;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
}

h1,h2,h3,h4,p {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_models():
    knn = joblib.load("model_knn.pkl")
    svm = joblib.load("model_svm.pkl")
    encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    return knn, svm, encoder, scaler


knn_model, svm_model, encoder, scaler = load_models()

# ==================================================
# EMOTION ICON
# ==================================================

emotion_icon = {
    "angry": "😠",
    "calm": "😌",
    "disgust": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😲"
}

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🎤 SER App")

    st.markdown("---")

    st.markdown("""
### Dataset

RAVDESS Dataset

- 24 Actors
- 2880 Audio Files
- 8 Emotions
""")

    st.markdown("---")

    st.markdown("""
### Models

- KNN + GridSearchCV
- SVM + GridSearchCV
""")

# ==================================================
# HEADER
# ==================================================

st.title("🎤 Speech Emotion Recognition")

st.markdown("""
Upload audio suara dan sistem akan memprediksi emosi menggunakan
model Machine Learning.
""")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "ogg", "m4a", "flac"]
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

        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("Analyzing audio..."):

        # ==========================================
        # LOAD AUDIO
        # ==========================================

        y, sr = librosa.load(
            audio_path,
            sr=None
        )

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        # ==========================================
        # FEATURE EXTRACTION
        # ==========================================

        feature = extract_feature_ml(audio_path)

        feature = feature.reshape(1, -1)

        feature = scaler.transform(feature)

        # ==========================================
        # PREDICTION
        # ==========================================

        pred_knn = knn_model.predict(feature)

        pred_svm = svm_model.predict(feature)

        emotion_knn = encoder.inverse_transform(pred_knn)[0]
        emotion_svm = encoder.inverse_transform(pred_svm)[0]

        probs = svm_model.predict_proba(feature)[0]

        confidence = np.max(probs)

    # ==================================================
    # METRICS
    # ==================================================

    st.subheader("📈 Audio Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Duration",
            f"{duration:.2f} sec"
        )

    with col2:
        st.metric(
            "Sample Rate",
            sr
        )

    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    # ==================================================
    # VISUALIZATION
    # ==================================================

    st.subheader("📊 Audio Visualization")

    col1, col2 = st.columns(2)

    with col1:

        fig_wave, ax_wave = plt.subplots(
            figsize=(8, 3)
        )

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        ax_wave.set_title("Waveform")

        st.pyplot(fig_wave)

    with col2:

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)),
            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots(
            figsize=(8, 3)
        )

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        plt.colorbar(img)

        ax_spec.set_title("Spectrogram")

        st.pyplot(fig_spec)

    # ==================================================
    # MODEL PREDICTION
    # ==================================================

    st.subheader("🤖 Model Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"KNN Prediction : "
            f"{emotion_icon.get(emotion_knn,'🎤')} "
            f"{emotion_knn.upper()}"
        )

    with col2:

        st.success(
            f"SVM Prediction : "
            f"{emotion_icon.get(emotion_svm,'🎤')} "
            f"{emotion_svm.upper()}"
        )

    # ==================================================
    # FINAL RESULT
    # ==================================================

    st.markdown(
        f"""
<div class="final-card">

<h2>Final Prediction</h2>

<h1>
{emotion_icon.get(emotion_svm,'🎤')}
</h1>

<h1>
{emotion_svm.upper()}
</h1>

<h3>
Confidence: {confidence:.2%}
</h3>

</div>
        """,
        unsafe_allow_html=True
    )

    # ==================================================
    # PROBABILITY TABLE
    # ==================================================

    st.subheader("📋 Emotion Probability")

    prob_df = pd.DataFrame({
        "Emotion": encoder.classes_,
        "Probability": probs
    })

    prob_df = prob_df.sort_values(
        by="Probability",
        ascending=False
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )

    # ==================================================
    # BAR CHART
    # ==================================================

    st.subheader("📊 Probability Chart")

    st.bar_chart(
        prob_df.set_index("Emotion")
    )

    # ==================================================
    # PIE CHART
    # ==================================================

    st.subheader("🥧 Emotion Distribution")

    fig_pie, ax_pie = plt.subplots(
        figsize=(5, 5)
    )

    ax_pie.pie(
        probs,
        labels=encoder.classes_,
        autopct="%1.1f%%"
    )

    ax_pie.set_title(
        "Emotion Distribution"
    )

    st.pyplot(fig_pie)
