import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import joblib
import os

import plotly.graph_objects as go
import plotly.express as px

from feature_extraction_ml import extract_feature_ml

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PREMIUM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins', sans-serif;
}

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* ===================================================== */
/* APP */
/* ===================================================== */

.stApp{

    background:
    radial-gradient(
        circle at top left,
        #1e3a8a 0%,
        #0f172a 40%,
        #020617 100%
    );
}

/* ===================================================== */
/* SIDEBAR */
/* ===================================================== */

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

/* ===================================================== */
/* GLASS CARD */
/* ===================================================== */

.glass-card{

    background:
    rgba(15,23,42,.75);

    backdrop-filter:blur(15px);

    border-radius:28px;

    padding:25px;

    border:
    1px solid rgba(255,255,255,.08);

    margin-bottom:20px;
}

/* ===================================================== */
/* HERO */
/* ===================================================== */

.hero-card{

    background:
    linear-gradient(
        135deg,
        rgba(37,99,235,.9),
        rgba(124,58,237,.9)
    );

    border-radius:30px;

    padding:35px;

    margin-bottom:25px;

    box-shadow:
    0 20px 50px rgba(37,99,235,.25);
}

/* ===================================================== */
/* KPI */
/* ===================================================== */

.kpi-card{

    background:
    rgba(15,23,42,.85);

    border-radius:25px;

    padding:22px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,.08);

    min-height:170px;
}

/* ===================================================== */
/* PREDICTION */
/* ===================================================== */

.pred-card{

    background:
    rgba(15,23,42,.85);

    border-radius:25px;

    padding:25px;

    border:
    1px solid rgba(255,255,255,.08);

    text-align:center;
}

/* ===================================================== */
/* FINAL */
/* ===================================================== */

.final-card{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    border-radius:35px;

    padding:40px;

    text-align:center;

    box-shadow:
    0px 25px 60px rgba(37,99,235,.35);
}

/* ===================================================== */
/* FILE UPLOADER */
/* ===================================================== */

[data-testid="stFileUploader"]{

    background:
    rgba(15,23,42,.75);

    border-radius:20px;

    padding:15px;

    border:
    1px dashed #60a5fa;
}

/* ===================================================== */
/* TEXT */
/* ===================================================== */

h1,h2,h3,h4,h5,h6{

    color:white !important;
}

p,span,label{

    color:#e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================

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

# =====================================================
# EMOTION ICONS
# =====================================================

emotion_icon = {
    "angry":"😠",
    "calm":"😌",
    "disgust":"🤢",
    "fearful":"😨",
    "happy":"😊",
    "neutral":"😐",
    "sad":"😢",
    "surprised":"😲"
}

# =====================================================
# SIDEBAR
# =====================================================

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

• 24 Actors

• 2880 Audio Files

• 8 Emotion Classes
""")

    st.markdown("---")

    st.markdown("""
### 🤖 Models

• KNN + GridSearchCV

• SVM + GridSearchCV
""")

    st.markdown("---")

    st.markdown("""
### 🧠 AI Engine

• MFCC

• Delta

• Delta²

• StandardScaler
""")

    st.markdown("---")

    st.markdown("""
### 🎓 About

Speech Emotion Recognition Dashboard

Built using Machine Learning
""")

# =====================================================
# BANNER
# =====================================================

if os.path.exists(
    "assets/banner.png"
):
    st.image(
        "assets/banner.png",
        use_container_width=True
    )

# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown("""
<div class="glass-card">

<h2>🎙 Upload Audio</h2>

<p>
Upload a speech recording and let the AI
predict the speaker's emotion.
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

# =====================================================
# HELPER FUNCTION
# =====================================================

def create_dark_layout():

    return dict(

        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",

        font=dict(
            color="white",
            family="Poppins"
        ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

# =====================================================
# MAIN PROCESS
# =====================================================

if uploaded_file is not None:

    # ==========================================
    # AUDIO PLAYER CARD
    # ==========================================

    st.markdown("""
    <div class="glass-card">

    <h2>🎧 Audio Preview</h2>

    </div>
    """, unsafe_allow_html=True)

    st.audio(uploaded_file)

    # ==========================================
    # SAVE TEMP FILE
    # ==========================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        audio_path = tmp.name

    # ==========================================
    # AI ANALYSIS
    # ==========================================

    with st.spinner(
        "🧠 AI is analyzing the emotion..."
    ):

        y, sr = librosa.load(
            audio_path,
            sr=None
        )

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        # ======================================
        # FEATURE EXTRACTION
        # ======================================

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

        # ======================================
        # PREDICTION
        # ======================================

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

        confidence = float(
            np.max(probs)
        )

    # ==========================================
    # TOP EMOTION
    # ==========================================

    top_emotion = emotion_svm

    # ==========================================
    # KPI SECTION
    # ==========================================

    st.markdown(
        "## 📊 Audio Analytics"
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>⏱ Duration</h3>

        <h1>{duration:.2f}</h1>

        <p>Seconds</p>

        </div>
        """,
        unsafe_allow_html=True)

    with k2:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🎚 Sample Rate</h3>

        <h1>{sr}</h1>

        <p>Hz</p>

        </div>
        """,
        unsafe_allow_html=True)

    with k3:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🎯 Confidence</h3>

        <h1>{confidence:.1%}</h1>

        <p>Prediction Score</p>

        </div>
        """,
        unsafe_allow_html=True)

    with k4:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🏆 Top Emotion</h3>

        <h1>
        {emotion_icon.get(top_emotion,'🎤')}
        </h1>

        <p>
        {top_emotion.upper()}
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    # ==========================================
    # WAVEFORM + SPECTROGRAM
    # ==========================================

    st.markdown(
        "## 🎵 Audio Visualization"
    )

    col1, col2 = st.columns(2)

    # ======================================
    # WAVEFORM
    # ======================================

    with col1:

        time_axis = np.linspace(
            0,
            duration,
            len(y)
        )

        fig_wave = go.Figure()

        fig_wave.add_trace(

            go.Scatter(
                x=time_axis,
                y=y,
                mode="lines",
                line=dict(
                    color="#60a5fa",
                    width=1
                ),
                name="Waveform"
            )
        )

        fig_wave.update_layout(

            title="Waveform",

            xaxis_title="Time (sec)",

            yaxis_title="Amplitude",

            height=350,

            **create_dark_layout()
        )

        st.plotly_chart(
            fig_wave,
            use_container_width=True
        )

    # ======================================
    # SPECTROGRAM
    # ======================================

    with col2:

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(y)
            ),
            ref=np.max
        )

        fig_spec = go.Figure(

            data=go.Heatmap(
                z=D,
                colorscale="Turbo"
            )
        )

        fig_spec.update_layout(

            title="Spectrogram",

            height=350,

            xaxis_title="Frames",

            yaxis_title="Frequency",

            **create_dark_layout()
        )

        st.plotly_chart(
            fig_spec,
            use_container_width=True
        )

    # ==========================================
    # PROBABILITY DATAFRAME
    # ==========================================

    prob_df = pd.DataFrame({

        "Emotion":
        encoder.classes_,

        "Probability":
        probs
    })

    prob_df = prob_df.sort_values(
        by="Probability",
        ascending=False
    )

    # ==========================================
    # PASS DATA TO PART 3
    # ==========================================

    prediction_data = {
        "emotion_knn": emotion_knn,
        "emotion_svm": emotion_svm,
        "confidence": confidence,
        "prob_df": prob_df
    }

    # =====================================================
    # MODEL PREDICTIONS
    # =====================================================

    st.markdown(
        "## 🤖 AI Model Predictions"
    )

    p1, p2 = st.columns(2)

    # ==========================================
    # KNN CARD
    # ==========================================

    with p1:

        st.markdown(f"""
        <div class="pred-card">

        <h3>🔵 KNN MODEL</h3>

        <h1 style="
        font-size:90px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_knn,'🎤')}
        </h1>

        <h2>
        {emotion_knn.upper()}
        </h2>

        <p>
        K-Nearest Neighbor Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    # ==========================================
    # SVM CARD
    # ==========================================

    with p2:

        st.markdown(f"""
        <div class="pred-card">

        <h3>🟣 SVM MODEL</h3>

        <h1 style="
        font-size:90px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_svm,'🎤')}
        </h1>

        <h2>
        {emotion_svm.upper()}
        </h2>

        <p>
        Support Vector Machine Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    # =====================================================
    # FINAL PREDICTION HERO
    # =====================================================

    st.markdown(
        "## 🏆 Final Prediction"
    )

    st.markdown(f"""
    <div class="final-card">

    <h3>
    AI FINAL DECISION
    </h3>

    <h1 style="
    font-size:140px;
    margin-bottom:0px;
    ">
    {emotion_icon.get(emotion_svm,'🎤')}
    </h1>

    <h1 style="
    font-size:54px;
    ">
    {emotion_svm.upper()}
    </h1>

    <h3>
    Confidence Score:
    {confidence:.2%}
    </h3>

    </div>
    """,
    unsafe_allow_html=True)

    # =====================================================
    # EMOTION ANALYTICS
    # =====================================================

    st.markdown(
        "## 📈 Emotion Analytics"
    )

    a1, a2 = st.columns(
        [1, 2]
    )

    # ==========================================
    # RANKING PROGRESS BAR
    # ==========================================

with a1:

    st.markdown("### 🏅 Emotion Ranking")

    ranking_df = prob_df.copy()

    ranking_df["Probability"] = (
        ranking_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # BAR CHART
    # ==========================================

    with a2:

        fig_bar = px.bar(

            prob_df,

            x="Emotion",

            y="Probability",

            color="Probability",

            color_continuous_scale="Blues"
        )

        fig_bar.update_layout(

            title="Emotion Probability",

            height=420,

            coloraxis_showscale=False,

            **create_dark_layout()
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # =====================================================
    # DONUT CHART
    # =====================================================

    st.markdown(
        "## 🎯 Emotion Distribution"
    )

    top_probability = float(
        prob_df.iloc[0]["Probability"]
    )

    top_emotion_name = str(
        prob_df.iloc[0]["Emotion"]
    ).upper()

    fig_donut = go.Figure(

        data=[
            go.Pie(

                labels=prob_df["Emotion"],

                values=prob_df["Probability"],

                hole=0.65,

                textinfo="label+percent"
            )
        ]
    )

    fig_donut.update_layout(

        annotations=[
            dict(

                text=
                f"<b>{top_probability:.1%}</b><br>{top_emotion_name}",

                showarrow=False,

                font=dict(
                    size=22,
                    color="white"
                )
            )
        ],

        height=520,

        **create_dark_layout()
    )

    st.plotly_chart(

        fig_donut,

        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<hr style="
border:1px solid rgba(255,255,255,.08);
">
""",
unsafe_allow_html=True)

st.markdown("""
<div style="
text-align:center;
padding:20px;
color:#94a3b8;
font-size:14px;
">

🎤 Speech Emotion Recognition Dashboard

<br><br>

Built with Streamlit • Plotly • Librosa • Scikit-Learn

<br>

Powered by RAVDESS Dataset

</div>
""",
unsafe_allow_html=True)
