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
# DARK PREMIUM CSS
# ==================================================

st.markdown("""
<style>

.stApp{
    background-color:#0f172a;
}

/* sidebar */

section[data-testid="stSidebar"]{
    background-color:#111827;
}

/* remove white containers */

[data-testid="stMetric"]{
    background-color:transparent;
    border:none;
}

/* cards */

.card{
    background:#1e293b;
    border-radius:20px;
    padding:20px;
    margin-bottom:20px;
    border:1px solid #334155;
}

.metric-card{
    background:#1e293b;
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid #334155;
}

.pred-card{
    background:#1e293b;
    border-radius:20px;
    padding:25px;
    text-align:center;
    border:1px solid #3b82f6;
}

.final-card{
    background:#2563eb;
    border-radius:25px;
    padding:30px;
    text-align:center;
}

/* text */

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

/* uploader */

[data-testid="stFileUploader"]{
    background:#1e293b;
    border-radius:20px;
    padding:20px;
}

/* audio player */

[data-testid="stAudio"]{
    background:#1e293b;
    border-radius:15px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
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

    st.image(
        "assets/logo_kampus.png",
        width=180
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    <h3>📊 Dataset</h3>

    <p>
    RAVDESS
    <br><br>
    🎭 24 Actors
    <br>
    🎤 2880 Audio Files
    <br>
    😊 8 Emotions
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    <h3>🤖 Models</h3>

    <p>
    ✅ KNN + GridSearchCV
    <br>
    ✅ SVM + GridSearchCV
    </p>

    </div>
    """, unsafe_allow_html=True)

# ==================================================
# BANNER
# ==================================================

st.image(
    "assets/banner.png",
    use_container_width=True
)

# ==================================================
# HERO CARD
# ==================================================

st.markdown("""
<div class='card'>

<h2>🎤 Speech Emotion Recognition System</h2>

<p>
Detect human emotions from speech using
MFCC + Delta + Delta² features.
</p>

<hr>

<p>

🤖 KNN Accuracy : <b>87.15%</b>

<br><br>

🤖 SVM Accuracy : <b>90.28%</b>

</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# DATASET SUMMARY
# ==================================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <h4>Dataset</h4>
    <h2>RAVDESS</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'>
    <h4>Emotions</h4>
    <h2>8</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <h4>KNN</h4>
    <h2>87.15%</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='metric-card'>
    <h4>SVM</h4>
    <h2>90.28%</h2>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# UPLOAD SECTION
# ==================================================

st.markdown("""
<div class='card'>

<h2>🎙 Upload Audio</h2>

<p>
Supported formats:
WAV, MP3, M4A, OGG, FLAC
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=[
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac"
    ]
)
