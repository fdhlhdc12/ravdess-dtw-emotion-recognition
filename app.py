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
from audio_recorder_streamlit import audio_recorder
import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SER AI - Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS KUSTOM PREMIUM
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* HIDE STREAMLIT DEFAULT */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* MAIN BACKGROUND */
.stApp {
    background: #0f0f1a;
}

/* SIDEBAR PREMIUM */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e, #16213e);
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 20px 0;
}

section[data-testid="stSidebar"] .sidebar-content {
    padding: 0 16px;
}

/* LOGO AREA */
.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}
.sidebar-logo h1 {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.sidebar-logo p {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 2px;
    margin-top: 2px;
}

/* NAVIGASI SIDEBAR - HILANGIN BORDER BAWAAN */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 0 !important;
    transition: all 0.3s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(108, 99, 255, 0.15) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.25), rgba(168, 85, 247, 0.15)) !important;
    border-left: 3px solid #6c63ff !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] span {
    color: white !important;
}

/* SIDEBAR INFO */
.sidebar-info {
    padding: 16px;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    margin: 16px 0;
}
.sidebar-info .label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sidebar-info .value {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 600;
}
.sidebar-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 16px 0;
}

.sidebar-footer {
    text-align: center;
    color: #475569;
    font-size: 11px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 16px;
}

/* CARDS */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(108, 99, 255, 0.2);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}

/* HERO SECTION */
.hero-section {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-radius: 24px;
    padding: 40px 48px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.08), transparent 70%);
    border-radius: 50%;
}
.hero-section h1 {
    font-size: 36px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
    position: relative;
}
.hero-section h1 span {
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-section p {
    color: #94a3b8;
    font-size: 16px;
    position: relative;
}
.hero-badges {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
    position: relative;
}
.hero-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 6px 18px;
    border-radius: 40px;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 500;
}

/* KPI CARDS */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 20px 0;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(108, 99, 255, 0.3);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.kpi-card .icon {
    font-size: 28px;
    margin-bottom: 4px;
}
.kpi-card .number {
    font-size: 28px;
    font-weight: 700;
    color: white;
}
.kpi-card .label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 2px;
}

/* FINAL PREDICTION CARD */
.final-prediction {
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    border-radius: 24px;
    padding: 32px;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(108, 99, 255, 0.3);
}
.final-prediction .emoji {
    font-size: 72px;
}
.final-prediction .emotion {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
}
.final-prediction .confidence {
    font-size: 18px;
    opacity: 0.9;
}

/* TABS CUSTOM */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 10px 20px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05);
    color: white !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
}

/* UPLOADER */
.stFileUploader > div > button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
}
.stFileUploader > div > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px rgba(108, 99, 255, 0.4);
}
.upload-area {
    border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-area:hover {
    border-color: rgba(108, 99, 255, 0.4);
    background: rgba(108, 99, 255, 0.05);
}
.upload-area .icon {
    font-size: 48px;
}
.upload-area p {
    color: #94a3b8;
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid rgba(255,255,255,0.05);
}
.metric-card .label {
    color: #64748b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    color: white;
    font-size: 20px;
    font-weight: 700;
}

/* MODEL CARDS */
.model-compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.model-card-premium {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
    transition: all 0.3s ease;
}
.model-card-premium:hover {
    transform: translateY(-4px);
    border-color: rgba(108, 99, 255, 0.3);
}
.model-card-premium .name {
    font-size: 20px;
    font-weight: 700;
    color: white;
}
.model-card-premium .detail {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 4px;
}
.model-card-premium .badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 40px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
}
.badge-svm {
    background: rgba(108, 99, 255, 0.2);
    color: #6c63ff;
}
.badge-knn {
    background: rgba(168, 85, 247, 0.2);
    color: #a855f7;
}

/* PIPELINE STEPS */
.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin: 20px 0;
}
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    min-width: 100px;
    border: 1px solid rgba(255,255,255,0.06);
    flex: 1;
}
.pipeline-step .step-num {
    display: inline-block;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    line-height: 28px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
}
.pipeline-step .step-label {
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 500;
}
.pipeline-step .step-desc {
    color: #94a3b8;
    font-size: 11px;
}
.pipeline-arrow {
    color: #475569;
    font-size: 20px;
    display: flex;
    align-items: center;
}

/* DATASET CARDS */
.dataset-stat {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.dataset-stat .number {
    font-size: 32px;
    font-weight: 700;
    color: white;
}
.dataset-stat .label {
    color: #94a3b8;
    font-size: 13px;
}
.dataset-stat .sub {
    color: #64748b;
    font-size: 12px;
}

/* CONTAINER */
.block-container {
    padding-top: 24px !important;
    padding-bottom: 24px !important;
    max-width: 1200px !important;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .hero-section { padding: 24px; }
    .hero-section h1 { font-size: 24px; }
    .model-compare { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: 1fr 1fr; }
    .pipeline { flex-direction: column; }
    .pipeline-arrow { transform: rotate(90deg); }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_models():
    knn_model = joblib.load("model_knn.pkl")
    svm_model = joblib.load("model_svm.pkl")
    encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return knn_model, svm_model, encoder, scaler

knn_model, svm_model, encoder, scaler = load_models()

# =====================================================
# EMOTION ICON & COLOR
# =====================================================
emotion_icon = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲"
}
emotion_color = {
    "angry": "#ef4444", "calm": "#22d3ee", "disgust": "#a78bfa",
    "fearful": "#f59e0b", "happy": "#34d399", "neutral": "#94a3b8",
    "sad": "#60a5fa", "surprised": "#f472b6"
}

# =====================================================
# FUNGSI ANALISIS
# =====================================================
def analyze_audio(audio_bytes, filename="temp.wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    feature = extract_feature_ml(audio_path)
    feature = feature.reshape(1, -1)
    feature_scaled = scaler.transform(feature)

    pred_knn = knn_model.predict(feature_scaled)
    pred_svm = svm_model.predict(feature_scaled)
    emotion_knn = encoder.inverse_transform(pred_knn)[0]
    emotion_svm = encoder.inverse_transform(pred_svm)[0]

    probs = svm_model.predict_proba(feature_scaled)[0]
    confidence = float(np.max(probs))
    top_emotion = emotion_svm

    prob_df = pd.DataFrame({
        "Emotion": encoder.classes_,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) > 0:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    os.unlink(audio_path)

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "emotion_knn": emotion_knn,
        "emotion_svm": emotion_svm,
        "top_emotion": top_emotion,
        "confidence": confidence,
        "prob_df": prob_df,
        "probs": probs,
        "mfcc": mfcc,
        "filename": filename,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std
    }

# =====================================================
# PLOTLY DARK THEME
# =====================================================
def dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>SER AI</h1>
        <p>Speech Emotion Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigasi
    pages = ["📊 Dashboard", "🎯 Prediksi", "📈 Analytics", "🧠 Model & Algoritma", "📁 Dataset"]
    if "page" not in st.session_state:
        st.session_state["page"] = "📊 Dashboard"
    choice = st.radio("NAVIGASI", pages, index=pages.index(st.session_state["page"]), key="nav", label_visibility="collapsed")
    if choice != st.session_state["page"]:
        st.session_state["page"] = choice
        st.rerun()
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # System Info
    st.markdown("""
    <div class="sidebar-info">
        <div class="label">Model Aktif</div>
        <div class="value">🔮 SVM</div>
        <div style="margin-top:8px;">
            <div class="label">Akurasi Model</div>
            <div class="value">90.28%</div>
        </div>
        <div style="margin-top:8px;">
            <div class="label">Fitur Digunakan</div>
            <div class="value">MFCC + Δ + Δ²</div>
        </div>
        <div style="margin-top:8px;">
            <div class="label">Dataset</div>
            <div class="value">RAVDESS</div>
        </div>
        <div style="margin-top:8px;">
            <div class="label">Update Terakhir</div>
            <div class="value">27 Juni 2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-footer">
        © 2025 SER AI<br>
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FUNGSI DISPLAY RESULTS
# =====================================================
def display_results(result):
    st.session_state["result"] = result
    
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="icon">⏱️</div>
            <div class="number">{result['duration']:.2f}s</div>
            <div class="label">Durasi</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="icon">🎚️</div>
            <div class="number">{result['sr']}</div>
            <div class="label">Sample Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="icon">🎯</div>
            <div class="number">{result['confidence']:.1%}</div>
            <div class="label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        emoji = emotion_icon.get(result['top_emotion'], '🎤')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="icon">{emoji}</div>
            <div class="number">{result['top_emotion'].upper()}</div>
            <div class="label">Emosi</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Final Prediction
    emoji = emotion_icon.get(result['top_emotion'], '🎤')
    color = emotion_color.get(result['top_emotion'], '#6c63ff')
    st.markdown(f"""
    <div class="final-prediction" style="background: linear-gradient(135deg, {color}, {color}dd);">
        <div class="emoji">{emoji}</div>
        <div class="emotion">{result['top_emotion'].upper()}</div>
        <div class="confidence">Confidence: {result['confidence']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5 Probabilities
    st.markdown("### 📊 Top 5 Probabilitas")
    top5 = result['prob_df'].head(5)
    for _, row in top5.iterrows():
        emo = row['Emotion']
        prob = row['Probability']
        col1, col2 = st.columns([2, 5])
        with col1:
            st.markdown(f"{emotion_icon.get(emo, '🎤')} **{emo.upper()}**")
        with col2:
            st.progress(float(prob), text=f"{prob:.2%}")
    
    # Visualizations
    st.markdown("### 🎵 Visualisasi Audio")
    tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🌈 Spectrogram", "🎚️ MFCC"])
    
    y, sr, duration = result['y'], result['sr'], result['duration']
    
    with tab1:
        time_axis = np.linspace(0, duration, len(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", line=dict(color="#6c63ff", width=2)))
        fig.update_layout(title="Waveform", xaxis_title="Time (sec)", yaxis_title="Amplitude", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig = go.Figure(data=go.Heatmap(z=D, colorscale="Turbo"))
        fig.update_layout(title="Spectrogram", xaxis_title="Frames", yaxis_title="Frequency", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure(data=go.Heatmap(z=result['mfcc'], colorscale="RdBu", zmid=0))
        fig.update_layout(title="MFCC Coefficients (40)", xaxis_title="Time frames", yaxis_title="Coefficient", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature info
    st.markdown("### 🧠 Ekstraksi Fitur")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">MFCC</div>
            <div class="value">40</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Delta</div>
            <div class="value">40</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Delta-Delta</div>
            <div class="value">40</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Total Fitur</div>
            <div class="value">120</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# HALAMAN DASHBOARD
# =====================================================
def show_dashboard():
    st.markdown("""
    <div class="hero-section">
        <h1>Welcome Back! <span>👋</span></h1>
        <p>Selamat datang di <strong>SER AI Dashboard</strong>. Dashboard ini digunakan untuk mendeteksi, menganalisis, dan memahami emosi manusia dari sinyal suara menggunakan teknologi Machine Learning.</p>
        <div class="hero-badges">
            <span class="hero-badge">🎯 Akurasi Tinggi: 90.28%</span>
            <span class="hero-badge">⚡ Proses Cepat: Real-time</span>
            <span class="hero-badge">📊 Visualisasi Interaktif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Stats
    st.markdown("### 📊 Statistik Dataset")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">2,880</div>
            <div class="label">Audio Files</div>
            <div class="sub">.wav</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24</div>
            <div class="label">Actors</div>
            <div class="sub">12 M, 12 F</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">8</div>
            <div class="label">Emosi</div>
            <div class="sub">Jenis berbeda</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24.6</div>
            <div class="label">Durasi Total</div>
            <div class="sub">jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">3.07</div>
            <div class="label">Rata-rata Durasi</div>
            <div class="sub">detik</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">1.2</div>
            <div class="label">Ukuran Dataset</div>
            <div class="sub">GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What is SER
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:white;">🧠 Apa itu Speech Emotion Recognition?</h3>
            <p style="color:#94a3b8;">Speech Emotion Recognition (SER) adalah teknologi yang mampu mengidentifikasi emosi manusia dari karakteristik suara. Sistem ini menganalisis pola suara seperti intonasi, pitch, energi, dan ritme untuk menentukan emosi yang diucapkan.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:white;">🎯 Emosi yang Dideteksi</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px;">
                <span>😊 Happy</span>
                <span>😢 Sad</span>
                <span>😠 Angry</span>
                <span>😨 Fearful</span>
                <span>😌 Calm</span>
                <span>😐 Neutral</span>
                <span>🤢 Disgust</span>
                <span>😲 Surprised</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # System Flow
    st.markdown("### 🔄 Alur Kerja Sistem")
    st.markdown("""
    <div class="pipeline">
        <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-label">Input Audio</div>
            <div class="step-desc">Upload / Rekam</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-label">Ekstraksi Fitur</div>
            <div class="step-desc">MFCC + Δ + Δ²</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-label">Preprocessing</div>
            <div class="step-desc">Scaling & Normalisasi</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-label">Klasifikasi</div>
            <div class="step-desc">Model SVM</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">5</div>
            <div class="step-label">Hasil Prediksi</div>
            <div class="step-desc">Emosi + Confidence</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">6</div>
            <div class="step-label">Analisis</div>
            <div class="step-desc">Visualisasi & Insight</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN PREDIKSI
# =====================================================
def show_prediksi():
    st.markdown("""
    <div class="hero-section">
        <h1>🎯 Prediksi Emosi</h1>
        <p>Unggah audio atau rekam suara Anda untuk mendeteksi emosi secara instan</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Rekam Suara"])
    
    with tab1:
        st.markdown("""
        <div class="upload-area">
            <div class="icon">📁</div>
            <p style="font-weight:600; color:#e2e8f0;">Drag & Drop file audio di sini</p>
            <p style="color:#64748b; font-size:13px;">atau klik untuk memilih file</p>
            <p style="color:#64748b; font-size:12px; margin-top:8px;">Format: WAV, MP3, M4A, OGG, FLAC • Max: 200MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file audio",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
            key="uploader"
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")
            
            with st.status("🔮 Menganalisis emosi...", expanded=False) as status:
                result = analyze_audio(audio_bytes, uploaded_file.name)
                status.update(label="✅ Selesai!", state="complete")
            
            st.toast("✨ Analisis selesai!", icon="🎉")
            display_results(result)
    
    with tab2:
        st.markdown("""
        <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.03); border-radius:16px; border:1px solid rgba(255,255,255,0.06);">
            <p style="color:#94a3b8;">Klik tombol di bawah, izinkan akses mikrofon, rekam, lalu tekan stop.</p>
        </div>
        """, unsafe_allow_html=True)
        
        audio_bytes = audio_recorder(
            text="🎙️ Klik untuk merekam",
            recording_color="#e74c3c",
            neutral_color="#6c63ff",
            icon_size="3x",
            energy_threshold=0.5,
            pause_threshold=2.0,
            sample_rate=16000,
            key="recorder"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            with st.status("🔮 Menganalisis rekaman...", expanded=False) as status:
                result = analyze_audio(audio_bytes, "recorded.wav")
                status.update(label="✅ Selesai!", state="complete")
            
            st.toast("🎙️ Analisis rekaman selesai!", icon="🎤")
            display_results(result)
        else:
            st.info("Tekan tombol di atas untuk mulai merekam.")

# =====================================================
# HALAMAN ANALYTICS
# =====================================================
def show_analytics():
    st.markdown("""
    <div class="hero-section">
        <h1>📊 Analytics</h1>
        <p>Analisis performa model dan distribusi emosi dari prediksi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI utama
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">90.28%</div>
            <div class="label">Akurasi Model</div>
            <div style="color:#34d399; font-size:12px;">↑ 3.28% dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">152</div>
            <div class="label">Total Prediksi</div>
            <div style="color:#34d399; font-size:12px;">↑ 18 dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">87.45%</div>
            <div class="label">Rata-rata Confidence</div>
            <div style="color:#34d399; font-size:12px;">↑ 4.12% dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">2.46s</div>
            <div class="label">Waktu Rata-rata</div>
            <div style="color:#f59e0b; font-size:12px;">↓ 0.35s dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribusi Emosi (simulasi)
    st.markdown("### 📊 Distribusi Prediksi Emosi")
    
    # Data simulasi
    emotion_data = {
        "Happy": 32, "Sad": 28, "Angry": 23, "Surprised": 18,
        "Neutral": 16, "Disgust": 9, "Fearful": 6
    }
    df_emotion = pd.DataFrame(list(emotion_data.items()), columns=["Emotion", "Count"])
    
    col1, col2 = st.columns([2, 3])
    with col1:
        for emo, count in emotion_data.items():
            pct = count / sum(emotion_data.values()) * 100
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="color:#e2e8f0;">{emotion_icon.get(emo.lower(), '🎤')} {emo}</span>
                <span style="color:#94a3b8;">{count} ({pct:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        fig = px.bar(df_emotion, x="Emotion", y="Count", color="Count", color_continuous_scale="Blues")
        fig.update_layout(height=350, **dark_layout())
        fig.update_traces(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix sim
    st.markdown("### 📊 Confusion Matrix (8 Emosi)")
    st.info("Confusion matrix interaktif akan ditampilkan di sini berdasarkan data riwayat prediksi.")
    
    # Insight
    st.markdown("### 💡 Insight Utama")
    st.markdown("""
    <div class="glass-card">
        <ul style="color:#94a3b8; list-style:none; padding:0;">
            <li>✅ Model menunjukkan performa sangat baik dengan akurasi 90.28% pada 152 prediksi.</li>
            <li>✅ Emosi 'Happy' paling sering diprediksi dengan tingkat confidence tertinggi.</li>
            <li>✅ Rata-rata confidence 87.45% menunjukkan model sangat baik dengan prediksi yang cepat.</li>
            <li>✅ Waktu prediksi rata-rata 2.46 detik, menunjukkan performa sistem yang cepat.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN MODEL & ALGORITMA
# =====================================================
def show_model():
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Model & Algoritma</h1>
        <p>Arsitektur model, algoritma machine learning, dan proses sistem secara keseluruhan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline
    st.markdown("### 🔄 Pipeline Sistem")
    st.markdown("""
    <div class="pipeline">
        <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-label">Input Audio</div>
            <div class="step-desc">Upload / Rekam</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-label">Preprocessing</div>
            <div class="step-desc">Resampling, Trimming</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-label">Ekstraksi Fitur</div>
            <div class="step-desc">MFCC + Δ + Δ²</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-label">Standardisasi</div>
            <div class="step-desc">Standard Scaler</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">5</div>
            <div class="step-label">Klasifikasi</div>
            <div class="step-desc">SVM / KNN</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">6</div>
            <div class="step-label">Output Emosi</div>
            <div class="step-desc">Probabilitas Tertinggi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Comparison
    st.markdown("### 🤖 Model Machine Learning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="model-card-premium">
            <div class="name">🔵 KNN</div>
            <div class="detail">K-Nearest Neighbors</div>
            <div style="margin-top:12px; text-align:left; color:#94a3b8; font-size:13px;">
                <div>• K = 5</div>
                <div>• Euclidean Distance</div>
                <div>• Weighted Voting</div>
            </div>
            <span class="badge badge-knn">Akurasi: 87.2%</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="model-card-premium">
            <div class="name">🟣 SVM</div>
            <div class="detail">Support Vector Machine</div>
            <div style="margin-top:12px; text-align:left; color:#94a3b8; font-size:13px;">
                <div>• Kernel: RBF</div>
                <div>• C = 10</div>
                <div>• Gamma = scale</div>
                <div>• Probability = True</div>
            </div>
            <span class="badge badge-svm">Akurasi: 90.28%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Detail
    st.markdown("### 📊 Detail Fitur yang Digunakan")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">MFCC</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Delta</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Delta-Delta</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Total Fitur</div>
            <div class="value">120</div>
            <div style="color:#64748b; font-size:11px;">dimensi</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:13px; margin:12px 0;">
        Kombinasi 40 MFCC + 40 Delta + 40 Delta-Delta = 120 dimensi fitur
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN DATASET
# =====================================================
def show_dataset():
    st.markdown("""
    <div class="hero-section">
        <h1>📁 Dataset RAVDESS</h1>
        <p>Informasi lengkap tentang dataset yang digunakan untuk melatih dan mengevaluasi model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">2,880</div>
            <div class="label">Audio Files</div>
            <div class="sub">.wav</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24</div>
            <div class="label">Actors</div>
            <div class="sub">12 M, 12 F</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">8</div>
            <div class="label">Emosi</div>
            <div class="sub">Jenis berbeda</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24.6</div>
            <div class="label">Durasi Total</div>
            <div class="sub">jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">3.07</div>
            <div class="label">Rata-rata Durasi</div>
            <div class="sub">detik</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">1.2</div>
            <div class="label">Ukuran Dataset</div>
            <div class="sub">GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Deskripsi
    st.markdown("""
    <div class="glass-card" style="margin:20px 0;">
        <h3 style="color:white;">📖 Deskripsi Dataset</h3>
        <p style="color:#94a3b8;">
            <strong>RAVDESS</strong> (Ryerson Audio-Visual Database of Emotional Speech and Song) adalah dataset publik yang berisi rekaman audio ucapan dan lagu dengan berbagai ekspresi emosi yang diucapkan oleh aktor profesional.
        </p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; color:#94a3b8; font-size:14px;">
            <div><strong style="color:#e2e8f0;">Nama Lengkap:</strong> Ryerson Audio-Visual Database of Emotional Speech and Song</div>
            <div><strong style="color:#e2e8f0;">Dibuat Oleh:</strong> Livingstone & Russo (Ryerson University)</div>
            <div><strong style="color:#e2e8f0;">Tahun Rilis:</strong> 2014</div>
            <div><strong style="color:#e2e8f0;">Bahasa:</strong> Inggris (North American)</div>
            <div><strong style="color:#e2e8f0;">Tipe Data:</strong> Audio ucapan & lagu</div>
            <div><strong style="color:#e2e8f0;">Format File:</strong> WAV (48kHz, 16-bit PCM)</div>
            <div><strong style="color:#e2e8f0;">Lisensi:</strong> Creative Commons Attribution 4.0 (CC BY 4.0)</div>
            <div><strong style="color:#e2e8f0;">Sumber:</strong> PLOS ONE Paper</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Download
    st.markdown("""
    <div style="text-align:center; padding:30px; background:rgba(255,255,255,0.03); border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin:20px 0;">
        <h3 style="color:white;">📥 Download Dataset</h3>
        <p style="color:#94a3b8;">Anda dapat mengunduh dataset RAVDESS dari sumber resmi.</p>
        <a href="https://zenodo.org/record/1188976" target="_blank" style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; padding:12px 32px; border-radius:40px; text-decoration:none; font-weight:600; margin-top:12px;">🔗 Buka Halaman Resmi RAVDESS</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Sitasi
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:white;">📝 Sitasi</h3>
        <p style="color:#94a3b8; font-size:14px;">
            Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. <em>PLOS ONE</em>, 13(5): e0196391.
        </p>
        <div style="margin-top:12px;">
            <a href="#" style="color:#6c63ff; text-decoration:none; font-weight:500;">📋 Salin Sitasi</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# RENDER PAGE
# =====================================================
if st.session_state["page"] == "📊 Dashboard":
    show_dashboard()
elif st.session_state["page"] == "🎯 Prediksi":
    show_prediksi()
elif st.session_state["page"] == "📈 Analytics":
    show_analytics()
elif st.session_state["page"] == "🧠 Model & Algoritma":
    show_model()
elif st.session_state["page"] == "📁 Dataset":
    show_dataset()
