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
# HALAMAN DASHBOARD
# =====================================================
def show_dashboard():
    # Hero Section dengan gambar
    col_img, col_text = st.columns([1, 2])
    
    with col_img:
        # Tampilkan gambar brain_wave.png
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
    
    # Statistik Dataset - 6 kolom
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
        """, unsafe
