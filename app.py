```python
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
# DARK THEME CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}

.pred-card {
    background: #1e293b;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #3b82f6;
}

.final-card {
    background: #2563eb;
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    color: white;
}

.metric-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}

h1,h2,h3,h4,p {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================

knn_model = joblib.load("model_knn.pkl")
svm_model = joblib.load("model_svm.pkl")

encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# ==================================================
# EMOTION ICON
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

    try:
        st.image(
            "assets/logo_kampus.png",
            width=180
        )
    except:
        pass

    st.markdown("---")

    st.markdown("""
### 📊 Dataset

**RAVDESS**

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

# ==================================================
# BANNER
# ==================================================

try:
    st.image(
        "assets/banner.png",
        use_container_width=True
    )
except:
    pass

# ==================================================
# UPLOAD SECTION
# ==================================================

st.markdown("""
<div class="card">

<h2>🎙 Upload Audio File</h2>

<p>
Supported formats:
WAV, MP3, M4A, OGG, FLAC
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["wav","mp3","m4a","ogg","flac"]
)

# ==================================================
# MAIN PROCESS
# ==================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

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

    # =====================================
    # METRICS
    # =====================================

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Duration",
            f"{duration:.2f}s"
        )

    with c2:
        st.metric(
            "Sample Rate",
            sr
        )

    with c3:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    # =====================================
    # WAVEFORM & SPECTROGRAM
    # =====================================

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("🎵 Waveform")

        fig_wave, ax_wave = plt.subplots()

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        st.pyplot(fig_wave)

    with col2:

        st.subheader("📊 Spectrogram")

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
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        plt.colorbar(img)

        st.pyplot(fig_spec)

    # =====================================
    # PREDICTIONS
    # =====================================

    st.subheader("🤖 Model Predictions")

    r1,r2 = st.columns(2)

    with r1:

        st.markdown(f"""
<div class="pred-card">

<h3>🤖 KNN</h3>

<h1>{emotion_icon.get(emotion_knn,'🎤')}</h1>

<h2>{emotion_knn.upper()}</h2>

</div>
""", unsafe_allow_html=True)

    with r2:

        st.markdown(f"""
<div class="pred-card">

<h3>🤖 SVM</h3>

<h1>{emotion_icon.get(emotion_svm,'🎤')}</h1>

<h2>{emotion_svm.upper()}</h2>

</div>
""", unsafe_allow_html=True)

    # =====================================
    # FINAL PREDICTION
    # =====================================

    st.markdown("---")

    st.markdown(f"""
<div class="final-card">

<h2>🏆 FINAL PREDICTION</h2>

<h1>{emotion_icon.get(emotion_svm,'🎤')}</h1>

<h1>{emotion_svm.upper()}</h1>

<h3>Confidence: {confidence:.2%}</h3>

</div>
""", unsafe_allow_html=True)

    # =====================================
    # BAR CHART
    # =====================================

    st.subheader("📊 Emotion Probability")

    prob_df = pd.DataFrame({
        "Emotion": encoder.classes_,
        "Probability": probs
    })

    st.bar_chart(
        prob_df.set_index("Emotion")
    )

    # =====================================
    # PIE CHART
    # =====================================

    st.subheader("🥧 Emotion Distribution")

    fig_pie, ax_pie = plt.subplots(
        figsize=(6,6)
    )

    ax_pie.pie(
        probs,
        labels=encoder.classes_,
        autopct="%1.1f%%"
    )

    st.pyplot(fig_pie)
```
