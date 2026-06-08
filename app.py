import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import joblib
import os

from feature_extraction_ml import extract_feature_ml

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Speech Emotion Recognition AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PREMIUM CSS
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp{
    background:
    radial-gradient(
        circle at top left,
        #1e3a8a,
        #0f172a 45%,
        #020617 100%
    );
}

/* SIDEBAR */

section[data-testid="stSidebar"]{
    background:
    linear-gradient(
        180deg,
        #020617,
        #0f172a
    );

    border-right:
    1px solid rgba(255,255,255,.08);
}

/* HERO */

.hero-card{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );

    border-radius:30px;

    padding:40px;

    margin-top:15px;

    margin-bottom:25px;

    box-shadow:
    0px 20px 50px rgba(79,70,229,.35);
}

/* METRIC */

.metric-card{

    background:
    rgba(30,41,59,.90);

    backdrop-filter: blur(12px);

    border-radius:24px;

    padding:20px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,.08);

    margin-bottom:15px;
}

/* PREDICTION */

.prediction-card{

    background:
    rgba(30,41,59,.90);

    border-radius:24px;

    padding:25px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,.08);

    min-height:240px;
}

/* FINAL */

.final-card{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    border-radius:30px;

    padding:40px;

    text-align:center;

    margin-top:20px;

    margin-bottom:20px;

    box-shadow:
    0px 20px 60px rgba(37,99,235,.35);
}

/* GLASS */

.glass-card{

    background:
    rgba(30,41,59,.85);

    backdrop-filter: blur(12px);

    border-radius:25px;

    padding:25px;

    margin-bottom:20px;

    border:
    1px solid rgba(255,255,255,.08);
}

h1,h2,h3,h4,h5,h6{
    color:white !important;
}

p{
    color:#cbd5e1 !important;
}

label{
    color:white !important;
}

span{
    color:white !important;
}

[data-testid="stFileUploader"]{
    background:
    rgba(30,41,59,.7);

    border-radius:20px;

    padding:20px;

    border:
    1px dashed #60a5fa;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_models():

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

    return (
        knn_model,
        svm_model,
        encoder,
        scaler
    )


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

    if os.path.exists(
        "assets/logo_kampus.png"
    ):
        st.image(
            "assets/logo_kampus.png",
            width=180
        )

    st.markdown("---")

    st.markdown("""
### 🎭 Dataset

**RAVDESS**

- 24 Actors
- 2880 Audio Files
- 8 Emotions
""")

    st.markdown("---")

    st.markdown("""
### 🤖 Models

- KNN + GridSearchCV
- SVM + GridSearchCV
""")

    st.markdown("---")

    st.markdown("""
### 🎓 Project

Speech Emotion Recognition

Machine Learning Based

Dashboard Version
""")

# ==================================================
# BANNER
# ==================================================

if os.path.exists(
    "assets/banner.png"
):
    st.image(
        "assets/banner.png",
        use_container_width=True
    )

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero-card">

<h1 style="font-size:55px;">
🎤 Speech Emotion Recognition AI
</h1>

<p style="font-size:18px;">

Analyze speech emotion using machine learning
models trained on the RAVDESS dataset.

</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# UPLOAD SECTION
# ==================================================

st.markdown("""
<div class="glass-card">

<h2>🎙 Upload Audio File</h2>

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

# ==================================================
# MAIN PROCESS
# ==================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        audio_path = tmp.name

    with st.spinner(
        "🤖 AI is analyzing emotion..."
    ):

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

        confidence = np.max(probs)

    # ==================================================
    # METRIC CARDS
    # ==================================================

    st.markdown("## 📈 Audio Analysis")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱ Duration</h3>
            <h1>{duration:.2f}s</h1>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎚 Sample Rate</h3>
            <h1>{sr}</h1>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Confidence</h3>
            <h1>{confidence:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)

    # ==================================================
    # VISUALIZATION
    # ==================================================

    st.markdown("## 📊 Audio Visualization")

    col1, col2 = st.columns(2)

    # ==========================================
    # WAVEFORM
    # ==========================================

    with col1:

        st.markdown("""
        <div class="glass-card">
        <h3>🎵 Waveform</h3>
        </div>
        """, unsafe_allow_html=True)

        fig_wave, ax_wave = plt.subplots(
            figsize=(8, 3)
        )

        fig_wave.patch.set_facecolor(
            "#0f172a"
        )

        ax_wave.set_facecolor(
            "#0f172a"
        )

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax_wave
        )

        ax_wave.tick_params(
            colors="white"
        )

        ax_wave.title.set_color(
            "white"
        )

        ax_wave.set_xlabel(
            "Time",
            color="white"
        )

        ax_wave.set_ylabel(
            "Amplitude",
            color="white"
        )

        for spine in ax_wave.spines.values():
            spine.set_color("white")

        st.pyplot(
            fig_wave,
            use_container_width=True
        )

    # ==========================================
    # SPECTROGRAM
    # ==========================================

    with col2:

        st.markdown("""
        <div class="glass-card">
        <h3>📡 Spectrogram</h3>
        </div>
        """, unsafe_allow_html=True)

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(y)
            ),
            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots(
            figsize=(8, 3)
        )

        fig_spec.patch.set_facecolor(
            "#0f172a"
        )

        ax_spec.set_facecolor(
            "#0f172a"
        )

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        ax_spec.tick_params(
            colors="white"
        )

        ax_spec.title.set_color(
            "white"
        )

        for spine in ax_spec.spines.values():
            spine.set_color("white")

        cbar = plt.colorbar(img)
        cbar.ax.yaxis.set_tick_params(
            color="white"
        )

        st.pyplot(
            fig_spec,
            use_container_width=True
        )

    # ==================================================
    # MODEL PREDICTIONS
    # ==================================================

    st.markdown("## 🤖 Model Predictions")

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(f"""
        <div class="prediction-card">

        <h3>KNN MODEL</h3>

        <h1 style="
        font-size:80px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_knn,'🎤')}
        </h1>

        <h2>
        {emotion_knn.upper()}
        </h2>

        </div>
        """, unsafe_allow_html=True)

    with p2:

        st.markdown(f"""
        <div class="prediction-card">

        <h3>SVM MODEL</h3>

        <h1 style="
        font-size:80px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_svm,'🎤')}
        </h1>

        <h2>
        {emotion_svm.upper()}
        </h2>

        </div>
        """, unsafe_allow_html=True)

    # ==================================================
    # FINAL PREDICTION
    # ==================================================

    st.markdown(f"""
    <div class="final-card">

    <h3>
    🏆 FINAL PREDICTION
    </h3>

    <h1 style="
    font-size:120px;
    margin-bottom:0px;
    ">
    {emotion_icon.get(emotion_svm,'🎤')}
    </h1>

    <h1 style="
    font-size:48px;
    ">
    {emotion_svm.upper()}
    </h1>

    <h3>
    Confidence Score:
    {confidence:.2%}
    </h3>

    </div>
    """, unsafe_allow_html=True)

    # ==================================================
    # PROBABILITY DATAFRAME
    # ==================================================

    prob_df = pd.DataFrame({
        "Emotion": encoder.classes_,
        "Probability": probs
    })

    prob_df = prob_df.sort_values(
        by="Probability",
        ascending=False
    )

    # ==================================================
    # RANKING + CHART
    # ==================================================

    st.markdown("## 📊 Emotion Analytics")

    a1, a2 = st.columns([1, 2])

    with a1:

        st.markdown("""
        <div class="glass-card">
        <h3>🏅 Top Emotion Ranking</h3>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            prob_df,
            use_container_width=True,
            hide_index=True
        )

    with a2:

        fig_bar, ax_bar = plt.subplots(
            figsize=(8, 4)
        )

        fig_bar.patch.set_facecolor(
            "#0f172a"
        )

        ax_bar.set_facecolor(
            "#0f172a"
        )

        ax_bar.bar(
            prob_df["Emotion"],
            prob_df["Probability"]
        )

        ax_bar.tick_params(
            colors="white"
        )

        ax_bar.set_title(
            "Emotion Probability",
            color="white"
        )

        for spine in ax_bar.spines.values():
            spine.set_color("white")

        st.pyplot(
            fig_bar,
            use_container_width=True
        )

    # ==================================================
    # PIE CHART
    # ==================================================

    st.markdown("## 🥧 Emotion Distribution")

    fig_pie, ax_pie = plt.subplots(
        figsize=(6, 6)
    )

    fig_pie.patch.set_facecolor(
        "#0f172a"
    )

    ax_pie.set_facecolor(
        "#0f172a"
    )

    ax_pie.pie(
        probs,
        labels=encoder.classes_,
        autopct="%1.1f%%"
    )

    ax_pie.set_title(
        "Emotion Distribution",
        color="white"
    )

    st.pyplot(
        fig_pie,
        use_container_width=True
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="
text-align:center;
padding:20px;
color:#94a3b8;
">

Speech Emotion Recognition Dashboard<br>

Built with Streamlit, Librosa, Scikit-Learn & RAVDESS Dataset

</div>
""", unsafe_allow_html=True)
