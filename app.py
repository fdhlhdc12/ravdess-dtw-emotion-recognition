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
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

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

.stApp{

    background:
    linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827
    );
}

/* SIDEBAR */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #020617,
        #111827
    );
}

section[data-testid="stSidebar"] *{

    color:white !important;
}

/* KPI CARD */

.kpi-card{

    background:
    rgba(30,41,59,.9);

    border-radius:24px;

    padding:25px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,.08);

    min-height:150px;
}

/* MODEL CARD */

.model-card{

    background:
    rgba(30,41,59,.9);

    border-radius:24px;

    padding:25px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,.08);
}

/* FINAL CARD */

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

    color:white;
}

/* GLASS */

.glass{

    background:
    rgba(30,41,59,.85);

    border-radius:24px;

    padding:20px;

    border:
    1px solid rgba(255,255,255,.08);
}

h1,h2,h3,h4,h5,h6{

    color:white !important;
}

p{

    color:#cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
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
# EMOTION ICON
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

RAVDESS

• 24 Actors

• 2880 Audio Files

• 8 Emotions
""")

    st.markdown("---")

    st.markdown("""
### 🤖 Models

• KNN

• SVM

• GridSearchCV
""")

    st.markdown("---")

    st.markdown("""
### 🧠 Features

• MFCC

• Delta

• Delta²
""")

# =====================================================
# BANNER
# =====================================================

if os.path.exists(
    "assets/banner.png"
):

    st.image(
        "assets/banner.png",
        width="stretch"
    )

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="glass">

<h2>🎙 Upload Audio</h2>

<p>
Upload speech audio and let AI predict
the emotion automatically.
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=[
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac"
    ],
    label_visibility="collapsed"
)

# =====================================================
# PLOTLY THEME
# =====================================================

def dark_layout():

    return dict(

        paper_bgcolor="#0f172a",

        plot_bgcolor="#0f172a",

        font=dict(
            color="white"
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
    # AUDIO PLAYER
    # ==========================================

    st.markdown("## 🎧 Audio Preview")

    st.audio(
        uploaded_file
    )

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
    # ANALYSIS
    # ==========================================

    with st.spinner(
        "🧠 Analyzing Emotion..."
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

        # ============================
        # PREDICTION
        # ============================

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

        top_emotion = emotion_svm

    # ==========================================
    # KPI CARDS
    # ==========================================

    st.markdown(
        "## 📊 Audio Analytics"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>⏱ Duration</h3>

        <h1>{duration:.2f}</h1>

        <p>Seconds</p>

        </div>
        """,
        unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🎚 Sample Rate</h3>

        <h1>{sr}</h1>

        <p>Hz</p>

        </div>
        """,
        unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🎯 Confidence</h3>

        <h1>{confidence:.1%}</h1>

        <p>AI Score</p>

        </div>
        """,
        unsafe_allow_html=True)

    with c4:

        st.markdown(f"""
        <div class="kpi-card">

        <h3>🏆 Emotion</h3>

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

    v1, v2 = st.columns(2)

    # ======================================
    # WAVEFORM
    # ======================================

    with v1:

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
                    width=2
                ),

                name="Waveform"
            )
        )

        fig_wave.update_layout(

            title="Waveform",

            xaxis_title="Time (sec)",

            yaxis_title="Amplitude",

            height=350,

            **dark_layout()
        )

        st.plotly_chart(
            fig_wave,
            width="stretch"
        )

    # ======================================
    # SPECTROGRAM
    # ======================================

    with v2:

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

            xaxis_title="Frames",

            yaxis_title="Frequency",

            height=350,

            **dark_layout()
        )

        st.plotly_chart(
            fig_spec,
            width="stretch"
        )

    # ==========================================
    # PROBABILITY DATA
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
    # SAVE FOR PART 3
    # ==========================================

    final_emotion = emotion_svm

    # =====================================================
    # MODEL PREDICTIONS
    # =====================================================

    st.markdown(
        "## 🤖 AI Model Predictions"
    )

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(f"""
        <div class="model-card">

        <h3>🔵 KNN MODEL</h3>

        <h1 style="
        font-size:80px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_knn,'🎤')}
        </h1>

        <h2>
        {emotion_knn.upper()}
        </h2>

        <p>
        K-Nearest Neighbor
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    with p2:

        st.markdown(f"""
        <div class="model-card">

        <h3>🟣 SVM MODEL</h3>

        <h1 style="
        font-size:80px;
        margin-bottom:0px;
        ">
        {emotion_icon.get(emotion_svm,'🎤')}
        </h1>

        <h2>
        {emotion_svm.upper()}
        </h2>

        <p>
        Support Vector Machine
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    # =====================================================
    # FINAL HERO
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
    font-size:120px;
    margin-bottom:0px;
    ">
    {emotion_icon.get(final_emotion,'🎤')}
    </h1>

    <h1>
    {final_emotion.upper()}
    </h1>

    <h3>
    Confidence Score :
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
    # RANKING
    # ==========================================

    with a1:

        st.markdown(
            "### 🏅 Emotion Ranking"
        )

        for _, row in prob_df.iterrows():

            emotion = row["Emotion"]

            probability = float(
                row["Probability"]
            )

            st.markdown(
                f"""
                **{emotion_icon.get(emotion,'🎤')}
                {emotion.upper()}**
                """
            )

            st.progress(
                probability
            )

            st.caption(
                f"{probability:.2%}"
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

            **dark_layout()
        )

        st.plotly_chart(
            fig_bar,
            width="stretch"
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

                hole=0.70,

                textinfo="none"
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
                    size=24,
                    color="white"
                )
            )
        ],

        height=380,

        **dark_layout()
    )

    st.plotly_chart(
        fig_donut,
        width="stretch"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:20px;
color:#94a3b8;
">

🎤 Speech Emotion Recognition Dashboard

<br><br>

Built with Streamlit • Plotly • Librosa • Scikit-Learn

<br>

Powered by RAVDESS Dataset

</div>
""",
unsafe_allow_html=True)
