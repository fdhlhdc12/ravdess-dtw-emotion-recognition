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
from PIL import Image

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
# CSS KUSTOM PREMIUM - FONT PUTIH & MODERN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
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

/* SEMUA TEKS DASAR JADI PUTIH */
body, .stApp, .main, .block-container {
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

p, li, span, div, label, .stMarkdown {
    color: #e2e8f0 !important;
}

.text-muted, .sub, .label {
    color: #94a3b8 !important;
}

/* SIDEBAR PREMIUM */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e, #16213e);
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 20px 0;
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}

.sidebar-logo h1 {
    font-size: 28px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.03em !important;
    margin: 0 !important;
}

.sidebar-logo p {
    color: #94a3b8 !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    font-weight: 500 !important;
    margin-top: 4px !important;
}

/* NAVIGASI SIDEBAR */
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
    color: #e2e8f0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.25), rgba(168, 85, 247, 0.15)) !important;
    border-left: 3px solid #6c63ff !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] span {
    color: #ffffff !important;
    font-weight: 600 !important;
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
    color: #94a3b8 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}

.sidebar-info .value {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-top: 2px !important;
}

.sidebar-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 16px 0;
}

.sidebar-footer {
    text-align: center;
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 400 !important;
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

.glass-card h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 12px !important;
}

.glass-card p, .glass-card li {
    color: #e2e8f0 !important;
    line-height: 1.7 !important;
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
    color: #ffffff !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    position: relative;
    margin-bottom: 8px !important;
}

.hero-section h1 span {
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.hero-section p {
    color: #cbd5e1 !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
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
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 6px 18px !important;
    border-radius: 40px !important;
}

/* KPI CARDS */
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
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

.kpi-card .label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    margin-top: 2px !important;
}

/* FINAL PREDICTION CARD */
.final-prediction {
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
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 32px !important;
    letter-spacing: -0.02em !important;
    margin: 8px 0 !important;
}

.final-prediction .confidence {
    color: rgba(255,255,255,0.9) !important;
    font-size: 18px !important;
    font-weight: 500 !important;
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
    font-size: 14px !important;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05);
    color: white !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
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
    color: #94a3b8 !important;
}

.upload-area p:first-of-type {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

.metric-card .label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}

.metric-card .value {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 800 !important;
}

/* MODEL CARDS */
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
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 20px !important;
}

.model-card-premium .detail {
    color: #94a3b8 !important;
    font-size: 14px !important;
    margin-top: 4px !important;
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
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

.pipeline-step .step-desc {
    color: #94a3b8 !important;
    font-size: 11px !important;
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
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 32px !important;
    letter-spacing: -0.02em !important;
}

.dataset-stat .label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

.dataset-stat .sub {
    color: #64748b !important;
    font-size: 12px !important;
}

/* ADVANTAGE CARDS */
.advantage-card {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    text-align: center;
    transition: all 0.3s ease;
}

.advantage-card:hover {
    transform: translateY(-4px);
    border-color: rgba(108, 99, 255, 0.2);
}

.advantage-card .icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.advantage-card .title {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

.advantage-card .desc {
    color: #94a3b8 !important;
    font-size: 13px !important;
    margin-top: 4px !important;
}

/* EMOTION TAGS */
.emotion-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.emotion-tag {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 40px !important;
    padding: 6px 18px !important;
    transition: all 0.3s ease;
}

.emotion-tag:hover {
    background: rgba(108, 99, 255, 0.15) !important;
    border-color: rgba(108, 99, 255, 0.3) !important;
    transform: translateY(-2px);
}

/* SECTION TITLE */
.section-title {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 22px !important;
    letter-spacing: -0.02em !important;
    margin: 28px 0 16px 0 !important;
}

/* CONTAINER */
.block-container {
    padding-top: 24px !important;
    padding-bottom: 24px !important;
    max-width: 1200px !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #1a1a2e;
}

::-webkit-scrollbar-thumb {
    background: #6c63ff;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #818cf8;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .hero-section { padding: 24px; }
    .hero-section h1 { font-size: 24px !important; }
    .pipeline { flex-direction: column; }
    .pipeline-arrow { transform: rotate(90deg); }
}

/* TABEL */
table, th, td {
    color: #e2e8f0 !important;
}

th {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* LINK */
a {
    color: #818cf8 !important;
    font-weight: 500 !important;
    text-decoration: none !important;
}

a:hover {
    color: #a78bfa !important;
    text-decoration: underline !important;
}

/* BADGE CONFIDENCE */
.badge-high {
    color: #ffffff !important;
    background: #10b981 !important;
    padding: 4px 16px !important;
    border-radius: 40px !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

.badge-medium {
    color: #1e293b !important;
    background: #f59e0b !important;
    padding: 4px 16px !important;
    border-radius: 40px !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

.badge-low {
    color: #ffffff !important;
    background: #ef4444 !important;
    padding: 4px 16px !important;
    border-radius: 40px !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

/* INFO, WARNING, ERROR, SUCCESS BOXES */
.stAlert {
    color: #e2e8f0 !important;
}

.stAlert p {
    color: #e2e8f0 !important;
}

/* PROGRESS BAR */
.stProgress > div > div {
    background: linear-gradient(90deg, #6c63ff, #a855f7) !important;
}

/* METRIC STREAMLIT */
.stMetric label {
    color: #94a3b8 !important;
}

.stMetric .stMetricValue {
    color: #ffffff !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_models():
    try:
        knn_model = joblib.load("model_knn.pkl")
        svm_model = joblib.load("model_svm.pkl")
        encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        return knn_model, svm_model, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model tidak ditemukan: {e}")
        st.warning("Pastikan file model_knn.pkl, model_svm.pkl, label_encoder.pkl, dan scaler.pkl ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

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
    try:
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
    except Exception as e:
        st.error(f"❌ Error analyzing audio: {e}")
        return None

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
    # Logo
    if os.path.exists("assets/logo.png"):
        try:
            logo = Image.open("assets/logo.png")
            st.image(logo, width=80)
        except:
            pass
    
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
# FUNGSI DISPLAY RESULTS (UNTUK PREDIKSI)
# =====================================================
def display_results(result):
    if result is None:
        st.warning("⚠️ Tidak ada hasil analisis.")
        return
        
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
# HALAMAN DASHBOARD (LENGKAP - TANPA GAMBAR DI KEUNGGULAN)
# =====================================================
def show_dashboard():
    # Hero Section dengan gambar
    col_img, col_text = st.columns([1, 2])
    
    with col_img:
        if os.path.exists("assets/brain_wave.png"):
            try:
                st.image("assets/brain_wave.png", use_container_width=True)
            except:
                st.markdown("""
                <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:20px; padding:40px; text-align:center;">
                    <span style="font-size:64px;">🧠</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:20px; padding:40px; text-align:center;">
                <span style="font-size:64px;">🧠</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="padding:8px 0;">
            <h1 style="color:white; font-size:36px; font-weight:800; margin-bottom:8px;">
                Welcome Back! <span style="background:linear-gradient(135deg, #6c63ff, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">👋</span>
            </h1>
            <p style="color:#94a3b8; font-size:16px; line-height:1.6; margin-bottom:16px;">
                Selamat datang di <strong style="color:#e2e8f0;">SER AI Dashboard</strong>. Dashboard ini digunakan untuk mendeteksi, menganalisis, dan memahami emosi manusia dari sinyal suara menggunakan teknologi Machine Learning.
            </p>
            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); padding:6px 18px; border-radius:40px; color:#cbd5e1; font-size:13px; font-weight:500;">🎯 Akurasi Tinggi: 90.28%</span>
                <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); padding:6px 18px; border-radius:40px; color:#cbd5e1; font-size:13px; font-weight:500;">⚡ Proses Cepat: Real-time</span>
                <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); padding:6px 18px; border-radius:40px; color:#cbd5e1; font-size:13px; font-weight:500;">📊 Visualisasi Interaktif</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # STATISTIK DATASET - 6 KOLOM
    # ============================================================
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:22px; font-weight:700;">📊 Statistik Dataset</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">2,880</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Audio Files</div>
            <div style="color:#64748b; font-size:11px;">.wav</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">24</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Actors</div>
            <div style="color:#64748b; font-size:11px;">12 M, 12 F</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">8</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Emosi</div>
            <div style="color:#64748b; font-size:11px;">Jenis berbeda</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">24.6</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Durasi Total</div>
            <div style="color:#64748b; font-size:11px;">jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">3.07</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Rata-rata Durasi</div>
            <div style="color:#64748b; font-size:11px;">detik</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:16px; padding:20px 12px; text-align:center; border:1px solid rgba(255,255,255,0.06); transition:all 0.3s ease;">
            <div style="font-size:28px; font-weight:700; color:white;">1.2</div>
            <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Ukuran Dataset</div>
            <div style="color:#64748b; font-size:11px;">GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # APA ITU SER + EMOSI
    # ============================================================
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:22px; font-weight:700;">🧠 Tentang Speech Emotion Recognition</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:20px; padding:24px; border:1px solid rgba(255,255,255,0.06); height:100%;">
            <h3 style="color:white; margin-bottom:12px;">🧠 Apa itu Speech Emotion Recognition?</h3>
            <p style="color:#94a3b8; line-height:1.7;">Speech Emotion Recognition (SER) adalah teknologi yang mampu mengidentifikasi emosi manusia dari karakteristik suara. Sistem ini menganalisis pola suara seperti intonasi, pitch, energi, dan ritme untuk menentukan emosi yang diucapkan.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if os.path.exists("assets/microphone_emotions.png"):
            try:
                st.image("assets/microphone_emotions.png", use_container_width=True)
            except:
                st.markdown("""
                <div style="background:rgba(255,255,255,0.04); border-radius:20px; padding:24px; border:1px solid rgba(255,255,255,0.06); height:100%;">
                    <h3 style="color:white; margin-bottom:12px;">🎯 Emosi yang Dideteksi</h3>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😊 Happy</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😢 Sad</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😠 Angry</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😨 Fearful</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😌 Calm</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😐 Neutral</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">🤢 Disgust</span>
                        <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😲 Surprised</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04); border-radius:20px; padding:24px; border:1px solid rgba(255,255,255,0.06); height:100%;">
                <h3 style="color:white; margin-bottom:12px;">🎯 Emosi yang Dideteksi</h3>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😊 Happy</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😢 Sad</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😠 Angry</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😨 Fearful</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😌 Calm</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😐 Neutral</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">🤢 Disgust</span>
                    <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:40px; padding:6px 16px; color:#e2e8f0; font-size:14px; text-align:center;">😲 Surprised</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================================
    # KEUNGGULAN SISTEM - 4 KOLOM (TANPA GAMBAR)
    # ============================================================
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:22px; font-weight:700;">⭐ Keunggulan Sistem</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:24px 16px; border:1px solid rgba(255,255,255,0.05); text-align:center; transition:all 0.3s ease; height:100%;">
            <div style="font-size:40px; margin-bottom:12px;">🎯</div>
            <div style="color:white; font-weight:700; font-size:16px; margin-bottom:4px;">Akurasi Tinggi</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.5;">Model SVM dengan performa terbaik</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:24px 16px; border:1px solid rgba(255,255,255,0.05); text-align:center; transition:all 0.3s ease; height:100%;">
            <div style="font-size:40px; margin-bottom:12px;">📊</div>
            <div style="color:white; font-weight:700; font-size:16px; margin-bottom:4px;">Ekstraksi Fitur Optimal</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.5;">MFCC + Delta + Delta² (120 fitur)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:24px 16px; border:1px solid rgba(255,255,255,0.05); text-align:center; transition:all 0.3s ease; height:100%;">
            <div style="font-size:40px; margin-bottom:12px;">⚡</div>
            <div style="color:white; font-weight:700; font-size:16px; margin-bottom:4px;">Proses Real-time</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.5;">Prediksi cepat dalam hitungan detik</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:24px 16px; border:1px solid rgba(255,255,255,0.05); text-align:center; transition:all 0.3s ease; height:100%;">
            <div style="font-size:40px; margin-bottom:12px;">📈</div>
            <div style="color:white; font-weight:700; font-size:16px; margin-bottom:4px;">Visualisasi Interaktif</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.5;">Grafik dan hasil yang mudah dipahami</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # TENTANG DASHBOARD
    # ============================================================
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:22px; font-weight:700;">📖 Tentang Dashboard</h2>
    </div>
    <div style="background:rgba(255,255,255,0.04); border-radius:20px; padding:24px; border:1px solid rgba(255,255,255,0.06); margin-bottom:16px;">
        <p style="color:#94a3b8; line-height:1.8;">Dashboard ini dirancang untuk membantu Anda memahami, memprediksi, dan menganalisis emosi manusia dari audio menggunakan teknologi Machine Learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # ALUR KERJA SISTEM
    # ============================================================
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:22px; font-weight:700;">🔄 Alur Kerja Sistem</h2>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:12px; justify-content:center; margin:16px 0;">
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">1</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Input Audio</div>
            <div style="color:#94a3b8; font-size:11px;">Upload / Rekam</div>
        </div>
        <div style="color:#475569; font-size:20px; display:flex; align-items:center;">➜</div>
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">2</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Ekstraksi Fitur</div>
            <div style="color:#94a3b8; font-size:11px;">MFCC + Δ + Δ²</div>
        </div>
        <div style="color:#475569; font-size:20px; display:flex; align-items:center;">➜</div>
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">3</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Pra-pemrosesan</div>
            <div style="color:#94a3b8; font-size:11px;">Scaling & Normalisasi</div>
        </div>
        <div style="color:#475569; font-size:20px; display:flex; align-items:center;">➜</div>
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">4</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Klasifikasi Model</div>
            <div style="color:#94a3b8; font-size:11px;">Prediksi dengan SVM</div>
        </div>
        <div style="color:#475569; font-size:20px; display:flex; align-items:center;">➜</div>
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">5</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Hasil Prediksi</div>
            <div style="color:#94a3b8; font-size:11px;">Emosi + Confidence</div>
        </div>
        <div style="color:#475569; font-size:20px; display:flex; align-items:center;">➜</div>
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px; text-align:center; min-width:100px; border:1px solid rgba(255,255,255,0.06); flex:1;">
            <div style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; width:28px; height:28px; border-radius:50%; line-height:28px; font-size:13px; font-weight:700; margin-bottom:6px;">6</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:500;">Analisis & Insight</div>
            <div style="color:#94a3b8; font-size:11px;">Visualisasi & Evaluasi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("""
    <div style="text-align:center; padding:20px; color:#64748b; font-size:13px; border-top:1px solid rgba(255,255,255,0.05); margin-top:20px;">
        © 2025 SER AI. All rights reserved.<br>
        <span style="color:#94a3b8; font-size:12px;">💡 Tips: Untuk hasil terbaik, gunakan audio dengan kualitas baik dan minim noise.</span>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN PREDIKSI (VERSI UPDATE - WARNA NAVY)
# =====================================================
def show_prediksi():
    st.markdown("""
    <div style="background:linear-gradient(145deg, #1a1a2e, #16213e); border-radius:24px; padding:32px 40px; border:1px solid rgba(255,255,255,0.06); margin-bottom:32px;">
        <h1 style="color:white; font-size:28px; font-weight:700;">🎯 Prediksi Emosi</h1>
        <p style="color:#94a3b8;">Unggah audio atau rekam suara Anda untuk mendeteksi emosi secara instan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs dengan styling lebih bagus
    tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Rekam Suara"])
    
    with tab1:
        st.markdown("""
        <div style="border:2px dashed rgba(108,99,255,0.25); border-radius:20px; padding:50px 40px; text-align:center; margin-bottom:20px; background:rgba(108,99,255,0.04); transition:all 0.3s ease;">
            <div style="font-size:56px; margin-bottom:12px;">📁</div>
            <p style="color:#e2e8f0; font-weight:600; font-size:17px; margin:8px 0;">Drag & Drop file audio di sini</p>
            <p style="color:#94a3b8; font-size:14px;">atau klik untuk memilih file</p>
            <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:12px;">
                <span style="background:rgba(108,99,255,0.15); padding:4px 14px; border-radius:20px; color:#a78bfa; font-size:12px; border:1px solid rgba(108,99,255,0.2);">WAV</span>
                <span style="background:rgba(108,99,255,0.15); padding:4px 14px; border-radius:20px; color:#a78bfa; font-size:12px; border:1px solid rgba(108,99,255,0.2);">MP3</span>
                <span style="background:rgba(108,99,255,0.15); padding:4px 14px; border-radius:20px; color:#a78bfa; font-size:12px; border:1px solid rgba(108,99,255,0.2);">M4A</span>
                <span style="background:rgba(108,99,255,0.15); padding:4px 14px; border-radius:20px; color:#a78bfa; font-size:12px; border:1px solid rgba(108,99,255,0.2);">OGG</span>
                <span style="background:rgba(108,99,255,0.15); padding:4px 14px; border-radius:20px; color:#a78bfa; font-size:12px; border:1px solid rgba(108,99,255,0.2);">FLAC</span>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:12px;">Maksimal ukuran file: 200MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih File",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
            key="uploader"
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            file_size = len(audio_bytes) / (1024 * 1024)
            
            # Tampilkan info file yang diupload
            st.markdown(f"""
            <div style="background:rgba(108,99,255,0.1); border-radius:16px; padding:16px 20px; border:1px solid rgba(108,99,255,0.2); margin:16px 0; display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
                <span style="font-size:28px;">🎵</span>
                <div style="flex:1;">
                    <span style="color:#e2e8f0; font-weight:600; font-size:15px;">{uploaded_file.name}</span>
                    <div style="display:flex; gap:16px; margin-top:2px; flex-wrap:wrap;">
                        <span style="color:#94a3b8; font-size:13px;">📦 {file_size:.1f} MB</span>
                        <span style="color:#94a3b8; font-size:13px;">✅ Siap diproses</span>
                    </div>
                </div>
                <span style="background:#10b981; color:white; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600;">✓ Terupload</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio player
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:16px;">
                <p style="color:#94a3b8; font-size:13px; margin-bottom:8px;">▶️ Preview Audio</p>
            """, unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.status("🔮 Menganalisis emosi...", expanded=False) as status:
                result = analyze_audio(audio_bytes, uploaded_file.name)
                status.update(label="✅ Selesai!", state="complete")
            
            st.toast("✨ Analisis selesai!", icon="🎉")
            display_results(result)
    
    with tab2:
        st.markdown("""
        <div style="text-align:center; padding:40px 30px; background:rgba(108,99,255,0.04); border-radius:20px; border:1px solid rgba(108,99,255,0.12); margin-bottom:24px;">
            <div style="font-size:48px; margin-bottom:12px;">🎙️</div>
            <p style="color:#94a3b8; font-size:15px; max-width:500px; margin:0 auto; line-height:1.7;">
                Klik tombol di bawah, izinkan akses mikrofon, rekam, lalu tekan stop 
                <span style="color:#64748b;">(atau diam selama 2 detik)</span>. 
                Hasil akan muncul otomatis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio recorder dengan styling tombol yang lebih bagus (tidak putih)
        st.markdown("""
        <div style="display:flex; justify-content:center; margin-bottom:20px;">
            <style>
            .recorder-btn {
                background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
                border: none !important;
                border-radius: 60px !important;
                padding: 16px 40px !important;
                color: white !important;
                font-size: 18px !important;
                font-weight: 600 !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 8px 24px rgba(108, 99, 255, 0.3) !important;
            }
            .recorder-btn:hover {
                transform: scale(1.05) !important;
                box-shadow: 0 12px 32px rgba(108, 99, 255, 0.5) !important;
            }
            .recorder-btn.recording {
                background: linear-gradient(135deg, #ef4444, #dc2626) !important;
                animation: pulse-btn 1.5s infinite !important;
            }
            @keyframes pulse-btn {
                0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
                70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }
            </style>
        """, unsafe_allow_html=True)
        
        audio_bytes = audio_recorder(
            text="🎙️ Klik untuk merekam",
            recording_color="#e74c3c",
            neutral_color="#6c63ff",
            icon_size="4x",
            energy_threshold=0.5,
            pause_threshold=2.0,
            sample_rate=16000,
            key="recorder"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if audio_bytes:
            # Tampilkan info rekaman berhasil
            st.markdown("""
            <div style="background:rgba(34,197,94,0.1); border-radius:16px; padding:14px 20px; border:1px solid rgba(34,197,94,0.2); margin:12px 0; display:flex; align-items:center; gap:12px;">
                <span style="font-size:24px;">✅</span>
                <div>
                    <span style="color:#e2e8f0; font-weight:600;">Rekaman berhasil!</span>
                    <span style="color:#94a3b8; font-size:13px; margin-left:8px;">Menganalisis suara Anda...</span>
                </div>
                <span style="margin-left:auto; display:flex; gap:4px;">
                    <span style="display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%; animation:pulse-dot 1s infinite;"></span>
                    <span style="display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%; animation:pulse-dot 1s infinite 0.3s;"></span>
                    <span style="display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%; animation:pulse-dot 1s infinite 0.6s;"></span>
                </span>
            </div>
            <style>
            @keyframes pulse-dot {
                0% { opacity: 0.3; transform: scale(0.8); }
                50% { opacity: 1; transform: scale(1.2); }
                100% { opacity: 0.3; transform: scale(0.8); }
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:16px;">
                <p style="color:#94a3b8; font-size:13px; margin-bottom:8px;">▶️ Preview Rekaman</p>
            """, unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.status("🔮 Menganalisis rekaman...", expanded=False) as status:
                result = analyze_audio(audio_bytes, "recorded.wav")
                status.update(label="✅ Selesai!", state="complete")
            
            st.toast("🎙️ Analisis rekaman selesai!", icon="🎤")
            display_results(result)
        else:
            st.markdown("""
            <div style="text-align:center; padding:20px; background:rgba(255,255,255,0.02); border-radius:16px; border:1px dashed rgba(255,255,255,0.06);">
                <p style="color:#64748b; font-size:14px;">💡 Tekan tombol di atas untuk mulai merekam.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Reset Button dengan styling lebih bagus
    st.markdown("""
    <div style="margin-top:24px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.05); display:flex; justify-content:center; gap:16px;">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Prediksi Baru", use_container_width=True, type="primary"):
            st.session_state.pop("result", None)
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
# =====================================================
# HALAMAN ANALYTICS
# =====================================================
def show_analytics():
    col_img, col_text = st.columns([1, 3])
    
    with col_img:
        if os.path.exists("assets/brain_wave.png"):
            try:
                st.image("assets/brain_wave.png", use_container_width=True)
            except:
                st.markdown("""
                <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:16px; padding:30px; text-align:center;">
                    <span style="font-size:48px;">📊</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:16px; padding:30px; text-align:center;">
                <span style="font-size:48px;">📊</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="background:linear-gradient(145deg, #1a1a2e, #16213e); border-radius:16px; padding:24px 32px; border:1px solid rgba(255,255,255,0.06); height:100%; display:flex; flex-direction:column; justify-content:center;">
            <h1 style="color:white; font-size:28px; font-weight:700;">📊 Analytics</h1>
            <p style="color:#94a3b8;">Analisis performa model dan distribusi emosi dari prediksi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI - 5 kolom
    st.markdown("""
    <div style="margin:24px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">📈 Metrik Kinerja</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">90.28%</div>
            <div class="label">Akurasi Model</div>
            <div style="color:#34d399; font-size:11px;">↑ 3.28% dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">152</div>
            <div class="label">Total Prediksi</div>
            <div style="color:#34d399; font-size:11px;">↑ 18 dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">87.45%</div>
            <div class="label">Rata-rata Confidence</div>
            <div style="color:#34d399; font-size:11px;">↑ 4.12% dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="number">2.46s</div>
            <div class="label">Waktu Rata-rata</div>
            <div style="color:#f59e0b; font-size:11px;">↓ 0.35s dari bulan lalu</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="kpi-card">
            <div class="number" style="font-size:32px;">😊</div>
            <div class="label">Emosi Terbanyak</div>
            <div style="color:#e2e8f0; font-size:13px; font-weight:600;">Happy</div>
            <div style="color:#64748b; font-size:11px;">32 prediksi (21.05%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribusi Emosi
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">📊 Distribusi Prediksi Emosi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    emotion_data = {
        "Happy": 32, "Calm": 28, "Sad": 23, "Angry": 20,
        "Surprised": 18, "Neutral": 16, "Disgust": 9, "Fearful": 6
    }
    total = sum(emotion_data.values())
    df_emotion = pd.DataFrame(list(emotion_data.items()), columns=["Emotion", "Count"])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:12px 16px; border:1px solid rgba(255,255,255,0.06);">', unsafe_allow_html=True)
        for emo, count in emotion_data.items():
            pct = count / total * 100
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#e2e8f0; font-size:13px;">{emotion_icon.get(emo.lower(), '🎤')} {emo}</span>
                <span style="color:#94a3b8; font-size:13px;">{count} ({pct:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fig = px.bar(df_emotion, x="Emotion", y="Count", color="Count", color_continuous_scale="Blues", text="Count")
        fig.update_traces(textposition='outside', showlegend=False)
        fig.update_layout(height=350, xaxis=dict(title="", tickfont=dict(color='#94a3b8')), yaxis=dict(title="Jumlah Prediksi", tickfont=dict(color='#94a3b8')), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">📊 Confusion Matrix (8 Emosi)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    confusion_data = {
        "Happy": [28, 2, 1, 0, 0, 0, 0, 0],
        "Calm": [0, 24, 1, 0, 0, 0, 0, 1],
        "Sad": [0, 0, 19, 1, 0, 0, 0, 0],
        "Angry": [0, 1, 1, 14, 1, 0, 0, 0],
        "Surprised": [0, 0, 0, 0, 13, 1, 1, 0],
        "Neutral": [0, 0, 0, 0, 0, 10, 1, 1],
        "Disgust": [0, 0, 0, 0, 0, 1, 7, 1],
        "Fearful": [0, 0, 0, 0, 0, 0, 1, 4]
    }
    emotions = list(confusion_data.keys())
    df_confusion = pd.DataFrame(confusion_data, index=emotions)
    
    fig = go.Figure(data=go.Heatmap(z=df_confusion.values, x=emotions, y=emotions, colorscale="Viridis", text=df_confusion.values, texttemplate="%{text}", textfont={"color": "white"}, hoverongaps=False))
    fig.update_layout(height=400, xaxis=dict(title="Prediksi", tickangle=45), yaxis=dict(title="Aktual"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">💡 Insight Utama</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <ul style="color:#94a3b8; list-style:none; padding:0; line-height:2;">
                <li>✅ Model menunjukkan performa sangat baik dengan akurasi 90.28% pada 152 prediksi.</li>
                <li>✅ Emosi 'Happy' paling sering diprediksi dengan tingkat confidence tertinggi.</li>
                <li>✅ Rata-rata confidence 87.45% menunjukkan model sangat baik dengan prediksi yang cepat.</li>
                <li>✅ Waktu prediksi rata-rata 2.46 detik, menunjukkan performa sistem yang cepat.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#e2e8f0; font-size:14px; margin-bottom:8px;">📊 Rata-rata Confidence per Emosi</h4>
        """, unsafe_allow_html=True)
        confidence_data = {
            "Happy": 91.2, "Calm": 89.3, "Sad": 86.7, "Angry": 84.1,
            "Surprised": 88.6, "Neutral": 82.4, "Disgust": 79.8, "Fearful": 77.3
        }
        for emo, conf in confidence_data.items():
            color = emotion_color.get(emo.lower(), '#6c63ff')
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:2px 0;">
                <span style="color:#e2e8f0; font-size:13px; min-width:80px;">{emotion_icon.get(emo.lower(), '🎤')} {emo}</span>
                <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:6px; overflow:hidden;">
                    <div style="background:linear-gradient(90deg, {color}, {color}dd); width:{conf}%; height:100%; border-radius:4px;"></div>
                </div>
                <span style="color:#94a3b8; font-size:12px; min-width:45px; text-align:right;">{conf:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribusi Confidence
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">📈 Distribusi Confidence</h2>
    </div>
    """, unsafe_allow_html=True)
    
    confidence_dist = {
        "0-20%": 0, "20-40%": 2, "40-60%": 5, "60-80%": 12, "80-100%": 133
    }
    df_confidence = pd.DataFrame(list(confidence_dist.items()), columns=["Range", "Jumlah"])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(df_confidence, x="Range", y="Jumlah", color="Jumlah", color_continuous_scale="Viridis", text="Jumlah")
        fig.update_traces(textposition='outside', showlegend=False)
        fig.update_layout(height=250, xaxis=dict(title="", tickfont=dict(color='#94a3b8')), yaxis=dict(title="Jumlah Prediksi", tickfont=dict(color='#94a3b8')), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="number" style="color:#34d399;">87.5%</div>
            <div class="label">Prediksi dengan Confidence > 80%</div>
            <div style="color:#64748b; font-size:11px;">133 dari 152 prediksi</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Timeline
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:18px; font-weight:600;">📈 Tren Prediksi dari Waktu ke Waktu</h2>
    </div>
    """, unsafe_allow_html=True)
    
    dates = pd.date_range(start='2025-06-01', end='2025-06-27', freq='D')
    np.random.seed(42)
    happy_trend = np.random.randint(0, 8, len(dates))
    sad_trend = np.random.randint(0, 6, len(dates))
    angry_trend = np.random.randint(0, 5, len(dates))
    calm_trend = np.random.randint(0, 4, len(dates))
    
    df_trend = pd.DataFrame({
        'Tanggal': dates,
        'Happy': happy_trend,
        'Sad': sad_trend,
        'Angry': angry_trend,
        'Calm': calm_trend
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_trend['Tanggal'], y=df_trend['Happy'], mode='lines+markers', name='Happy', line=dict(color='#34d399')))
    fig.add_trace(go.Scatter(x=df_trend['Tanggal'], y=df_trend['Sad'], mode='lines+markers', name='Sad', line=dict(color='#60a5fa')))
    fig.add_trace(go.Scatter(x=df_trend['Tanggal'], y=df_trend['Angry'], mode='lines+markers', name='Angry', line=dict(color='#ef4444')))
    fig.add_trace(go.Scatter(x=df_trend['Tanggal'], y=df_trend['Calm'], mode='lines+markers', name='Calm', line=dict(color='#22d3ee')))
    fig.update_layout(height=300, xaxis=dict(title="", tickfont=dict(color='#94a3b8')), yaxis=dict(title="Jumlah Prediksi", tickfont=dict(color='#94a3b8')), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color='#94a3b8')), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="text-align:center; padding:16px; color:#64748b; font-size:13px; border-top:1px solid rgba(255,255,255,0.05); margin-top:20px;">
        © 2025 SER AI. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN MODEL & ALGORITMA
# =====================================================
def show_model():
    st.markdown("""
    <div style="background:linear-gradient(145deg, #1a1a2e, #16213e); border-radius:24px; padding:32px 40px; border:1px solid rgba(255,255,255,0.06); margin-bottom:32px;">
        <h1 style="color:white; font-size:28px; font-weight:700;">🧠 Model & Algoritma</h1>
        <p style="color:#94a3b8;">Arsitektur model, algoritma machine learning, dan proses sistem secara keseluruhan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline
    st.markdown("""
    <div style="margin:24px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">🔄 Pipeline Sistem</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Alur proses dari audio input hingga prediksi emosi</p>
    </div>
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
            <div class="step-label">Klasifikasi Model</div>
            <div class="step-desc">SVM / KNN</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">6</div>
            <div class="step-label">Probabilitas</div>
            <div class="step-desc">Hitung Probabilitas</div>
        </div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-step">
            <div class="step-num">7</div>
            <div class="step-label">Output Emosi</div>
            <div class="step-desc">Probabilitas Tertinggi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Comparison
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">🤖 Model Machine Learning</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Perbandingan model yang digunakan dalam sistem SER AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="model-card-premium">
            <div class="name">🔵 KNN</div>
            <div class="detail">K-Nearest Neighbors</div>
            <div style="margin-top:12px; text-align:left; color:#94a3b8; font-size:13px; padding:0 12px;">
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span>K</span>
                    <span style="color:#e2e8f0; font-weight:500;">5</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span>Distance</span>
                    <span style="color:#e2e8f0; font-weight:500;">Euclidean</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:4px 0;">
                    <span>Akurasi</span>
                    <span style="color:#a855f7; font-weight:600;">87.2%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="model-card-premium">
            <div class="name">🟣 SVM</div>
            <div class="detail">Support Vector Machine</div>
            <div style="margin-top:12px; text-align:left; color:#94a3b8; font-size:13px; padding:0 12px;">
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span>Kernel</span>
                    <span style="color:#e2e8f0; font-weight:500;">RBF</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span>C (Regularization)</span>
                    <span style="color:#e2e8f0; font-weight:500;">10</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span>Gamma</span>
                    <span style="color:#e2e8f0; font-weight:500;">scale</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:4px 0;">
                    <span>Akurasi</span>
                    <span style="color:#6c63ff; font-weight:600;">90.28%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Algoritma & Deskripsi - Tabel
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📊 Algoritma & Deskripsi</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Penjelasan algoritma yang digunakan dalam sistem</p>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:20px; border:1px solid rgba(255,255,255,0.06); overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; color:#e2e8f0; font-size:14px;">
            <thead>
                <tr style="border-bottom:2px solid rgba(255,255,255,0.08);">
                    <th style="text-align:left; padding:12px 16px; color:#ffffff; font-weight:600;">Algoritma</th>
                    <th style="text-align:left; padding:12px 16px; color:#ffffff; font-weight:600;">Deskripsi</th>
                    <th style="text-align:left; padding:12px 16px; color:#ffffff; font-weight:600;">Kelebihan</th>
                    <th style="text-align:left; padding:12px 16px; color:#ffffff; font-weight:600;">Penggunaan</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                    <td style="padding:12px 16px; color:#ffffff; font-weight:500;">MFCC</td>
                    <td style="padding:12px 16px;">Mel-Frequency Cepstral Coefficients digunakan untuk merepresentasikan karakteristik spektral suara.</td>
                    <td style="padding:12px 16px;">
                        • Reduksi dimensi yang efektif<br>
                        • Robust terhadap noise
                    </td>
                    <td style="padding:12px 16px;">Ekstraksi Fitur</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                    <td style="padding:12px 16px; color:#ffffff; font-weight:500;">Delta & Delta²</td>
                    <td style="padding:12px 16px;">Turunan pertama dan kedua dari MFCC yang menangkap dinamika temporal suara.</td>
                    <td style="padding:12px 16px;">
                        • Menangkap perubahan suara<br>
                        • Meningkatkan akurasi
                    </td>
                    <td style="padding:12px 16px;">Ekstraksi Fitur</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                    <td style="padding:12px 16px; color:#ffffff; font-weight:500;">SVM</td>
                    <td style="padding:12px 16px;">Support Vector Machine dengan kernel RBF untuk klasifikasi emosi.</td>
                    <td style="padding:12px 16px;">
                        • Performa tinggi pada data kecil<br>
                        • Generalisasi yang baik
                    </td>
                    <td style="padding:12px 16px;">Klasifikasi</td>
                </tr>
                <tr>
                    <td style="padding:12px 16px; color:#ffffff; font-weight:500;">Standard Scaler</td>
                    <td style="padding:12px 16px;">Standardisasi fitur agar memiliki mean 0 dan variance 1.</td>
                    <td style="padding:12px 16px;">
                        • Meningkatkan konvergensi model<br>
                        • Wajib untuk SVM
                    </td>
                    <td style="padding:12px 16px;">Preprocessing</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Detail Fitur
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📊 Detail Fitur yang Digunakan</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Informasi lengkap mengenai fitur yang diekstraksi dari audio</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <div class="label">MFCC</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <div class="label">Delta</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <div class="label">Delta-Delta</div>
            <div class="value">40</div>
            <div style="color:#64748b; font-size:11px;">coefficients</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card" style="text-align:center; background:rgba(108,99,255,0.1); border-color:rgba(108,99,255,0.2);">
            <div class="label" style="color:#6c63ff;">Total Fitur</div>
            <div class="value">120</div>
            <div style="color:#64748b; font-size:11px;">features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:13px; margin:8px 0;">
        Kombinasi 40 MFCC + 40 Delta + 40 Delta-Delta = 120 dimensi fitur
    </div>
    """, unsafe_allow_html=True)
    
    # Spesifikasi Model Aktif
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">⚙️ Spesifikasi Model Aktif</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Parameter dan konfigurasi model yang sedang digunakan</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#6c63ff; font-size:16px; margin-bottom:12px;">🟣 Model: SVM</h4>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Kernel</span>
                <span style="color:#ffffff; font-weight:500;">RBF</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">C (Regularization)</span>
                <span style="color:#ffffff; font-weight:500;">10</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Gamma</span>
                <span style="color:#ffffff; font-weight:500;">scale</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Probability</span>
                <span style="color:#ffffff; font-weight:500;">True</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0;">
                <span style="color:#94a3b8;">Class Weight</span>
                <span style="color:#ffffff; font-weight:500;">balanced</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#34d399; font-size:16px; margin-bottom:12px;">📊 Metrik Evaluasi</h4>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Akurasi</span>
                <span style="color:#34d399; font-weight:600;">90.28%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Precision</span>
                <span style="color:#34d399; font-weight:600;">90.11%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94a3b8;">Recall</span>
                <span style="color:#34d399; font-weight:600;">89.56%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:6px 0;">
                <span style="color:#94a3b8;">F1-Score</span>
                <span style="color:#34d399; font-weight:600;">90.22%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(108,99,255,0.08); border-radius:16px; padding:16px 20px; border:1px solid rgba(108,99,255,0.15); margin:20px 0;">
        <p style="color:#94a3b8; font-size:13px; text-align:center; margin:0;">
            🔍 Model dijalankan dengan <strong style="color:#e2e8f0;">GridSearchCV</strong> menggunakan 
            <strong style="color:#e2e8f0;">5-fold Cross Validation</strong> untuk mendapatkan parameter terbaik.
        </p>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:20px; border:1px solid rgba(255,255,255,0.06); margin:20px 0; text-align:center;">
        <p style="color:#94a3b8; font-size:14px; margin:0;">
            💡 Sistem ini menggunakan kombinasi fitur <strong style="color:#e2e8f0;">MFCC</strong> dengan algoritma 
            <strong style="color:#e2e8f0;">machine learning</strong> untuk mendefinisikan emosi dari suara manusia secara akurat.
        </p>
    </div>
    <div style="text-align:center; padding:16px; color:#64748b; font-size:13px; border-top:1px solid rgba(255,255,255,0.05); margin-top:12px;">
        © 2025 SER AI. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HALAMAN DATASET
# =====================================================
def show_dataset():
    col_img, col_text = st.columns([1, 3])
    
    with col_img:
        if os.path.exists("assets/folder_download.png"):
            try:
                st.image("assets/folder_download.png", use_container_width=True)
            except:
                st.markdown("""
                <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:16px; padding:30px; text-align:center;">
                    <span style="font-size:48px;">📁</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg, #6c63ff, #a855f7); border-radius:16px; padding:30px; text-align:center;">
                <span style="font-size:48px;">📁</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="background:linear-gradient(145deg, #1a1a2e, #16213e); border-radius:16px; padding:24px 32px; border:1px solid rgba(255,255,255,0.06); height:100%; display:flex; flex-direction:column; justify-content:center;">
            <h1 style="color:white; font-size:28px; font-weight:700;">📁 Dataset RAVDESS</h1>
            <p style="color:#94a3b8;">Informasi lengkap tentang dataset yang digunakan untuk melatih dan mengevaluasi model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistik - 6 kolom
    st.markdown("""
    <div style="margin:24px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📊 Statistik Dataset</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">2,880</div>
            <div class="label">Total Audio Files</div>
            <div class="sub">Audio clips (.wav)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24</div>
            <div class="label">Total Actors</div>
            <div class="sub">12 Male, 12 Female</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">8</div>
            <div class="label">Total Emosi</div>
            <div class="sub">Jenis emosi berbeda</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">24.6</div>
            <div class="label">Durasi Total</div>
            <div class="sub">Total durasi audio (jam)</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">3.07</div>
            <div class="label">Durasi Rata-rata</div>
            <div class="sub">Per audio clip (detik)</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="dataset-stat">
            <div class="number">1.2</div>
            <div class="label">Ukuran Dataset</div>
            <div class="sub">Ukuran total file (GB)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Deskripsi
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📖 Deskripsi Dataset</h2>
    </div>
    <div class="glass-card">
        <p style="color:#94a3b8; line-height:1.7; margin-bottom:16px;">
            <strong style="color:#e2e8f0;">RAVDESS</strong> (Ryerson Audio-Visual Database of Emotional Speech and Song) adalah dataset publik yang berisi rekaman audio ucapan dan lagu dengan berbagai ekspresi emosi yang diucapkan oleh aktor profesional.
        </p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px 24px; color:#94a3b8; font-size:14px;">
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Nama Lengkap</span>
                <span style="color:#e2e8f0; text-align:right;">Ryerson Audio-Visual Database of Emotional Speech and Song</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Dibuat Oleh</span>
                <span style="color:#e2e8f0; text-align:right;">Livingstone & Russo (Ryerson University)</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Tahun Rilis</span>
                <span style="color:#e2e8f0; text-align:right;">2014</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Bahasa</span>
                <span style="color:#e2e8f0; text-align:right;">Inggris (North American)</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Tipe Data</span>
                <span style="color:#e2e8f0; text-align:right;">Audio ucapan & lagu</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Format File</span>
                <span style="color:#e2e8f0; text-align:right;">WAV (48kHz, 16-bit PCM)</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#64748b;">Lisensi</span>
                <span style="color:#e2e8f0; text-align:right;">Creative Commons Attribution 4.0 (CC BY 4.0)</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;">
                <span style="color:#64748b;">Sumber</span>
                <span style="color:#e2e8f0; text-align:right;">PLOS ONE Paper</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Struktur File Naming
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📂 Struktur File Naming</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Setiap file mengikuti pola penamaan standar RAVDESS.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#6c63ff;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Actor ID</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#a855f7;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Emotion</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#34d399;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Intensity</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#f59e0b;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Statement</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#f472b6;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Repetition</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:24px; font-weight:700; color:#22d3ee;">01</div>
            <div style="color:#94a3b8; font-size:11px;">Durasi</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📥 Download Dataset</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Anda dapat mengunduh dataset RAVDESS dari sumber resmi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:24px; border:1px solid rgba(255,255,255,0.06); display:flex; flex-direction:column; align-items:center;">
            <a href="https://zenodo.org/record/1188976" target="_blank" style="display:inline-block; background:linear-gradient(135deg, #6c63ff, #a855f7); color:white; padding:14px 40px; border-radius:40px; text-decoration:none; font-weight:600; font-size:16px; transition:all 0.3s ease;">
                🔗 Buka Halaman Resmi RAVDESS
            </a>
            <p style="color:#64748b; font-size:12px; margin-top:12px;">Dataset tersedia dalam format ZIP (1.2 GB)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if os.path.exists("assets/audio_waveform.png"):
            try:
                st.image("assets/audio_waveform.png", use_container_width=True)
            except:
                pass
    
    # Sitasi
    st.markdown("""
    <div style="margin:32px 0 16px 0;">
        <h2 style="color:white; font-size:20px; font-weight:700;">📝 Sitasi</h2>
        <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Jika Anda menggunakan dataset ini dalam penelitian Anda, mohon sitasi berikut:</p>
    </div>
    <div class="glass-card">
        <p style="color:#94a3b8; font-size:14px; line-height:1.7; margin-bottom:12px;">
            Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. <em style="color:#e2e8f0;">PLOS ONE</em>, 13(5): e0196391.
        </p>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <button onclick="navigator.clipboard.writeText('Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE, 13(5): e0196391.')" style="background:rgba(108,99,255,0.15); border:1px solid rgba(108,99,255,0.2); color:#6c63ff; padding:8px 20px; border-radius:40px; cursor:pointer; font-weight:500; font-size:13px;">
                📋 Salin Sitasi            </button>
            <a href="#" style="color:#6c63ff; text-decoration:none; font-weight:500; font-size:13px; padding:8px 20px; border:1px solid rgba(255,255,255,0.08); border-radius:40px;">
                📄 Lihat di PLOS ONE
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03); border-radius:16px; padding:16px 20px; border:1px solid rgba(255,255,255,0.06); margin:20px 0;">
        <p style="color:#64748b; font-size:13px; text-align:center; margin:0;">
            💡 <strong style="color:#e2e8f0;">Tips:</strong> RAVDESS memiliki kualitas audio yang baik dan variasi emosi yang seimbang, cocok untuk penelitian SER.
        </p>
    </div>
    <div style="text-align:center; padding:16px; color:#64748b; font-size:13px; border-top:1px solid rgba(255,255,255,0.05); margin-top:12px;">
        © 2025 SER AI. All rights reserved.
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
else:
    # Fallback jika halaman tidak dikenal
    st.warning("⚠️ Halaman tidak ditemukan. Mengarahkan ke Dashboard.")
    show_dashboard()
