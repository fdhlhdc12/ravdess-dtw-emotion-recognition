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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import datetime

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
# CSS KUSTOM
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {
    background: linear-gradient(145deg, #0b1120 0%, #111827 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120, #1e293b);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}
.glass {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 28px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.5);
}
.kpi-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(8px);
    border-radius: 24px;
    padding: 24px 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-6px);
    border-color: rgba(96, 165, 250, 0.4);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.model-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(8px);
    border-radius: 24px;
    padding: 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
}
.model-card:hover {
    transform: translateY(-6px);
    border-color: rgba(139, 92, 246, 0.4);
}
.final-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 32px;
    padding: 40px 20px;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
}
.hero {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 3rem 2rem;
    border-radius: 40px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.cta-button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    padding: 14px 42px;
    border-radius: 60px;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
    text-decoration: none;
    display: inline-block;
}
.cta-button:hover {
    transform: scale(1.04);
    box-shadow: 0 12px 32px rgba(37, 99, 235, 0.5);
}
.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
p, li, .stMarkdown { color: #cbd5e1 !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
}
.js-plotly-plot .plotly .main-svg {
    border-radius: 16px !important;
}
.badge-high {
    background: #10b981;
    color: white;
    padding: 4px 16px;
    border-radius: 40px;
    font-weight: 600;
    display: inline-block;
}
.badge-medium {
    background: #f59e0b;
    color: #1e293b;
    padding: 4px 16px;
    border-radius: 40px;
    font-weight: 600;
    display: inline-block;
}
.badge-low {
    background: #ef4444;
    color: white;
    padding: 4px 16px;
    border-radius: 40px;
    font-weight: 600;
    display: inline-block;
}
.insight-box {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #2563eb;
    padding: 16px 20px;
    border-radius: 12px;
    margin: 12px 0;
}
.workflow-step {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin: 10px 0;
}
.workflow-arrow {
    color: #60a5fa;
    font-size: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL (cached)
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
# EMOTION ICON
# =====================================================
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

# =====================================================
# FUNGSI ANALISIS UTAMA (dengan pitch)
# =====================================================
def analyze_audio(audio_bytes, filename="temp.wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Ekstraksi fitur untuk model
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

    # MFCC untuk heatmap
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pitch analysis (mean, std)
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
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

# =====================================================
# BADGE CONFIDENCE
# =====================================================
def confidence_badge(conf):
    if conf >= 0.8:
        return f'<span class="badge-high">🟢 High Confidence ({conf:.1%})</span>'
    elif conf >= 0.6:
        return f'<span class="badge-medium">🟡 Medium Confidence ({conf:.1%})</span>'
    else:
        return f'<span class="badge-low">🔴 Low Confidence ({conf:.1%})</span>'

# =====================================================
# AI INSIGHT (dengan Top 3 candidates)
# =====================================================
def generate_insight(result):
    top3 = result['prob_df'].head(3)
    top_emo = result['top_emotion']
    conf = result['confidence']
    
    base = f"The primary emotion detected is **{top_emo}** with confidence {conf:.1%}."
    if conf > 0.8:
        base += " The model is highly confident."
    elif conf > 0.6:
        base += " The model is moderately confident."
    else:
        base += " The model has low confidence; consider checking audio quality."

    # Tambahan berdasarkan emosi
    emotion_insights = {
        "happy": "Voice exhibits high pitch variation and energy, typical of happiness.",
        "sad": "Voice shows lower pitch and slower tempo, characteristic of sadness.",
        "angry": "Voice has sharp intensity and high volume, indicative of anger.",
        "calm": "Voice is steady with minimal pitch fluctuation, showing calmness.",
        "fearful": "Voice may have trembling and higher pitch, associated with fear.",
        "disgust": "Voice may have harsh or nasal quality, typical of disgust.",
        "surprised": "Voice shows sudden pitch rise and high energy, indicating surprise.",
        "neutral": "Voice is relatively flat and monotone, typical of neutral emotion."
    }
    extra = emotion_insights.get(top_emo, "")
    if extra:
        base += " " + extra

    # Top 3 candidates
    candidates = " | ".join([f"{row['Emotion']} ({row['Probability']:.1%})" for _, row in top3.iterrows()])
    base += f"\n\n**Top 3 Candidates:** {candidates}"
    
    return base

# =====================================================
# FUNGSI DISPLAY HASIL
# =====================================================
def display_results(result):
    st.session_state["result"] = result

    # KPI Cards
    st.markdown("## 📊 Audio Analytics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>⏱ Duration</h3>
            <h1>{result['duration']:.2f}</h1>
            <p>detik</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>🎚 Sample Rate</h3>
            <h1>{result['sr']}</h1>
            <p>Hz</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>🎯 Confidence</h3>
            <h1>{result['confidence']:.1%}</h1>
            <p>Skor AI</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>🏆 Emotion</h3>
            <h1>{emotion_icon.get(result['top_emotion'], '🎤')}</h1>
            <p>{result['top_emotion'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)

    # Confidence Badge
    st.markdown(confidence_badge(result['confidence']), unsafe_allow_html=True)

    # AI Insight
    st.markdown("## 🤖 AI Insight")
    insight_text = generate_insight(result)
    st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)

    # Waveform + Spectrogram + MFCC Heatmap
    st.markdown("## 🎵 Audio Visualization")
    col_wave, col_spec, col_mfcc = st.columns(3)
    y = result['y']
    sr = result['sr']
    duration = result['duration']

    with col_wave:
        time_axis = np.linspace(0, duration, len(y))
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", line=dict(color="#60a5fa", width=2), name="Waveform"))
        fig_wave.update_layout(title="Waveform", xaxis_title="Time (sec)", yaxis_title="Amplitude", height=300, **dark_layout())
        st.plotly_chart(fig_wave, use_container_width=True)

    with col_spec:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig_spec = go.Figure(data=go.Heatmap(z=D, colorscale="Turbo"))
        fig_spec.update_layout(title="Spectrogram", xaxis_title="Frames", yaxis_title="Frequency", height=300, **dark_layout())
        st.plotly_chart(fig_spec, use_container_width=True)

    with col_mfcc:
        mfcc = result['mfcc']
        fig_mfcc = go.Figure(data=go.Heatmap(z=mfcc, colorscale="RdBu", zmid=0))
        fig_mfcc.update_layout(title="MFCC (40 coefficients)", xaxis_title="Time frames", yaxis_title="Coefficient", height=300, **dark_layout())
        st.plotly_chart(fig_mfcc, use_container_width=True)

    # Model Predictions
    st.markdown("## 🤖 Model Predictions")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
        <div class="model-card">
            <h3>🔵 KNN</h3>
            <h1 style="font-size:80px; margin:0;">{emotion_icon.get(result['emotion_knn'], '🎤')}</h1>
            <h2>{result['emotion_knn'].upper()}</h2>
            <p>K-Nearest Neighbors</p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="model-card">
            <h3>🟣 SVM</h3>
            <h1 style="font-size:80px; margin:0;">{emotion_icon.get(result['emotion_svm'], '🎤')}</h1>
            <h2>{result['emotion_svm'].upper()}</h2>
            <p>Support Vector Machine</p>
        </div>
        """, unsafe_allow_html=True)

    # Final Prediction
    st.markdown("## 🏆 Final Prediction")
    st.markdown(f"""
    <div class="final-card">
        <h3>AI DECISION</h3>
        <h1 style="font-size:120px; margin:0;">{emotion_icon.get(result['top_emotion'], '🎤')}</h1>
        <h1>{result['top_emotion'].upper()}</h1>
        <h3>Confidence: {result['confidence']:.2%}</h3>
        {confidence_badge(result['confidence'])}
    </div>
    """, unsafe_allow_html=True)

    # Emotion Analytics (ranking + bar + donut)
    st.markdown("## 📈 Emotion Analytics")
    a1, a2 = st.columns([1, 2])
    prob_df = result['prob_df']
    with a1:
        st.markdown("### 🏅 Ranking Emosi")
        for _, row in prob_df.iterrows():
            emo = row["Emotion"]
            prob = row["Probability"]
            st.markdown(f"**{emotion_icon.get(emo, '🎤')} {emo.upper()}**")
            st.progress(float(prob))
            st.caption(f"{prob:.2%}")

    with a2:
        fig_bar = px.bar(prob_df, x="Emotion", y="Probability", color="Probability", color_continuous_scale="Blues")
        fig_bar.update_layout(title="Probabilitas per Emosi", height=420, coloraxis_showscale=False, **dark_layout())
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("## 🎯 Emotion Distribution")
    top_prob = float(prob_df.iloc[0]["Probability"])
    top_name = str(prob_df.iloc[0]["Emotion"]).upper()
    fig_donut = go.Figure(data=[go.Pie(labels=prob_df["Emotion"], values=prob_df["Probability"], hole=0.70, textinfo="none")])
    fig_donut.update_layout(
        annotations=[dict(text=f"<b>{top_prob:.1%}</b><br>{top_name}", showarrow=False, font=dict(size=24, color="white"))],
        height=380,
        **dark_layout()
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# =====================================================
# HALAMAN HOME
# =====================================================
def show_home():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 4.2rem; font-weight: 800; letter-spacing: -1px; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🎤 Speech Emotion AI
        </h1>
        <p style="font-size: 1.4rem; max-width: 700px; margin: 0.5rem auto; color: #94a3b8;">
            Deteksi emosi dari suara secara instan dengan Machine Learning
        </p>
        <br>
    </div>
    """, unsafe_allow_html=True)

    # Tombol navigasi ke Upload (fix)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Mulai Analisis", use_container_width=True):
            st.session_state["page"] = "📤 Upload Audio"
            st.rerun()

    st.markdown("---")

    # About Dashboard
    st.markdown("## 📖 About Dashboard")
    st.markdown("""
    Dashboard ini mengidentifikasi emosi manusia dari rekaman suara menggunakan ekstraksi fitur **MFCC** 
    dan model **Machine Learning** (KNN & SVM) yang dilatih pada dataset **RAVDESS**.
    """)

    # Workflow
    st.markdown("## 🔄 Workflow")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="text-align:center; padding:20px; background:rgba(30,41,59,0.5); border-radius:16px;">
            <h1 style="font-size:3rem;">🎤</h1>
            <p><strong>Upload Audio</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align:center; font-size:2rem; color:#60a5fa;'>→</p>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align:center; padding:20px; background:rgba(30,41,59,0.5); border-radius:16px;">
            <h1 style="font-size:3rem;">🎼</h1>
            <p><strong>Ekstraksi MFCC</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("<p style='text-align:center; font-size:2rem; color:#60a5fa;'>→</p>", unsafe_allow_html=True)
    with col1:
        st.markdown("""
        <div style="text-align:center; padding:20px; background:rgba(30,41,59,0.5); border-radius:16px;">
            <h1 style="font-size:3rem;">🤖</h1>
            <p><strong>Prediksi Emosi</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align:center; font-size:2rem; color:#60a5fa;'>→</p>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align:center; padding:20px; background:rgba(30,41,59,0.5); border-radius:16px;">
            <h1 style="font-size:3rem;">📄</h1>
            <p><strong>Laporan & Visualisasi</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Dataset Statistics
    st.markdown("---")
    st.markdown("## 📊 Dataset RAVDESS")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎬 Aktor", "24")
    with col2:
        st.metric("🎵 Audio", "2,880")
    with col3:
        st.metric("😃 Emosi", "8")
    with col4:
        st.metric("📁 Format", "WAV, 16kHz")

    st.markdown("---")
    # Feature Cards
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="glass" style="text-align:center; height:100%;">
            <h1 style="font-size:3rem;">🎯</h1>
            <h3 style="color:white;">Akurasi Tinggi</h3>
            <p>Model SVM & KNN dengan tuning optimal</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="glass" style="text-align:center; height:100%;">
            <h1 style="font-size:3rem;">⚡</h1>
            <h3 style="color:white;">Proses Cepat</h3>
            <p>Ekstraksi fitur dan prediksi dalam hitungan detik</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="glass" style="text-align:center; height:100%;">
            <h1 style="font-size:3rem;">📊</h1>
            <h3 style="color:white;">Visualisasi Interaktif</h3>
            <p>Waveform, spektrogram, MFCC, dan grafik probabilitas</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN UPLOAD
# =====================================================
def show_upload():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <h2>📤 Upload Audio</h2>
        <p>Upload file suara (WAV, MP3, M4A, OGG, FLAC) untuk dianalisis emosinya.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

        with st.status("🧠 Menganalisis emosi...", expanded=False) as status:
            result = analyze_audio(audio_bytes, uploaded_file.name)
            status.update(label="✅ Selesai!", state="complete")
        st.toast("✨ Analisis selesai!", icon="🎉")

        display_results(result)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN LIVE RECORD
# =====================================================
def show_live_record():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <h2>🎙️ Live Recording</h2>
        <p>Rekam suara langsung dari mikrofon Anda, lalu AI akan mendeteksi emosi secara real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    audio_bytes = audio_recorder(
        text="Klik untuk mulai merekam",
        recording_color="#e74c3c",
        neutral_color="#2563eb",
        icon_size="3x",
        energy_threshold=0.5,
        pause_threshold=2.0,
        sample_rate=16000
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.status("🧠 Menganalisis rekaman...", expanded=False) as status:
            result = analyze_audio(audio_bytes, "recorded.wav")
            status.update(label="✅ Selesai!", state="complete")
        st.toast("🎙️ Analisis rekaman selesai!", icon="🎤")
        display_results(result)
    else:
        st.info("Tekan tombol di atas untuk mulai merekam.")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN ANALYTICS
# =====================================================
def show_analytics():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <h2>📊 Analytics Dashboard</h2>
        <p>Analisis mendalam dari hasil prediksi terakhir.</p>
    </div>
    """, unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Belum ada hasil analisis. Silakan upload atau rekam audio terlebih dahulu.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    result = st.session_state["result"]
    prob_df = result['prob_df']

    # Top 3 Emotion Candidates
    st.markdown("### 🏅 Top 3 Emotion Candidates")
    top3 = prob_df.head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center; background:rgba(30,41,59,0.6); border-radius:16px; padding:20px;">
                <h1 style="font-size:3rem;">{emotion_icon.get(row['Emotion'], '🎤')}</h1>
                <h3>{row['Emotion'].upper()}</h3>
                <p style="font-size:1.5rem; font-weight:bold; color:#60a5fa;">{row['Probability']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Pitch Analysis
    st.markdown("### 🎵 Pitch Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🎵 Average Pitch", f"{result['pitch_mean']:.1f} Hz")
    with c2:
        st.metric("📊 Pitch Std Dev", f"{result['pitch_std']:.1f} Hz")

    # MFCC Heatmap full
    st.markdown("### 🎼 MFCC Heatmap (Full)")
    fig_mfcc_full = go.Figure(data=go.Heatmap(z=result['mfcc'], colorscale="RdBu", zmid=0))
    fig_mfcc_full.update_layout(
        title="MFCC Coefficients (40)",
        xaxis_title="Time frames",
        yaxis_title="Coefficient index",
        height=500,
        **dark_layout()
    )
    st.plotly_chart(fig_mfcc_full, use_container_width=True)

    # Bar chart
    st.markdown("### 📊 Distribusi Probabilitas")
    fig_bar_full = px.bar(prob_df, x="Emotion", y="Probability", color="Probability", color_continuous_scale="Viridis")
    fig_bar_full.update_layout(height=400, **dark_layout())
    st.plotly_chart(fig_bar_full, use_container_width=True)

    # Radar chart
    st.markdown("### 🕸️ Radar Emosi")
    categories = prob_df["Emotion"].tolist()
    values = prob_df["Probability"].tolist()
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Probabilitas',
        line_color='#60a5fa'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        height=450,
        **dark_layout()
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Insight
    st.markdown("### 🤖 AI Insight")
    insight_text = generate_insight(result)
    st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
    st.markdown(confidence_badge(result['confidence']), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN REPORT (dengan PDF)
# =====================================================
def show_report():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <h2>📄 Report</h2>
        <p>Ringkasan hasil analisis dan opsi unduh.</p>
    </div>
    """, unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Belum ada hasil analisis. Silakan upload atau rekam audio terlebih dahulu.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    result = st.session_state["result"]

    # Ringkasan
    st.markdown("### 📋 Ringkasan")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **File**: {result['filename']}
        - **Durasi**: {result['duration']:.2f} detik
        - **Sample Rate**: {result['sr']} Hz
        - **Pitch Mean**: {result['pitch_mean']:.1f} Hz
        """)
    with col2:
        st.markdown(f"""
        - **Emosi Terdeteksi**: {result['top_emotion'].upper()} {emotion_icon.get(result['top_emotion'], '')}
        - **Confidence**: {result['confidence']:.2%}
        - **Model**: SVM (final)
        - **Confidence Level**: {'High' if result['confidence']>=0.8 else 'Medium' if result['confidence']>=0.6 else 'Low'}
        """)

    # Tabel probabilitas
    st.markdown("### 📊 Tabel Probabilitas")
    st.dataframe(result['prob_df'].style.format({'Probability': '{:.2%}'}), use_container_width=True)

    # Download CSV
    csv = result['prob_df'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Probabilitas (CSV)",
        data=csv,
        file_name=f"emotion_probs_{result['filename']}.csv",
        mime="text/csv"
    )

    # Download TXT
    report_text = f"""
    SPEECH EMOTION RECOGNITION REPORT
    ==================================
    File       : {result['filename']}
    Durasi     : {result['duration']:.2f} detik
    Sample Rate: {result['sr']} Hz
    Pitch Mean : {result['pitch_mean']:.1f} Hz

    Prediksi Emosi:
    - KNN   : {result['emotion_knn'].upper()}
    - SVM   : {result['emotion_svm'].upper()}
    - Final : {result['top_emotion'].upper()} (Confidence: {result['confidence']:.2%})

    Probabilitas per Emosi:
    {result['prob_df'].to_string(index=False)}
    """
    st.download_button(
        label="📄 Download Laporan (TXT)",
        data=report_text,
        file_name=f"report_{result['filename']}.txt",
        mime="text/plain"
    )

    # Download PDF (menggunakan reportlab)
    def generate_pdf(result):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(100, height - 100, "Speech Emotion Recognition Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 130, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Details
        y = height - 170
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Audio Details")
        y -= 25
        c.setFont("Helvetica", 12)
        c.drawString(100, y, f"File: {result['filename']}")
        y -= 20
        c.drawString(100, y, f"Duration: {result['duration']:.2f} sec")
        y -= 20
        c.drawString(100, y, f"Sample Rate: {result['sr']} Hz")
        y -= 20
        c.drawString(100, y, f"Pitch Mean: {result['pitch_mean']:.1f} Hz")
        y -= 30

        # Prediction
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Prediction")
        y -= 25
        c.setFont("Helvetica", 12)
        c.drawString(100, y, f"Primary Emotion: {result['top_emotion'].upper()} {emotion_icon.get(result['top_emotion'], '')}")
        y -= 20
        c.drawString(100, y, f"Confidence: {result['confidence']:.2%}")
        y -= 20
        c.drawString(100, y, f"KNN: {result['emotion_knn'].upper()}")
        y -= 20
        c.drawString(100, y, f"SVM: {result['emotion_svm'].upper()}")
        y -= 30

        # Probabilities table (top 5)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Top 5 Emotion Probabilities")
        y -= 25
        c.setFont("Helvetica", 12)
        top5 = result['prob_df'].head(5)
        for _, row in top5.iterrows():
            c.drawString(100, y, f"{row['Emotion']}: {row['Probability']:.2%}")
            y -= 20

        c.save()
        buffer.seek(0)
        return buffer

    pdf_bytes = generate_pdf(result)
    st.download_button(
        label="📄 Download Laporan (PDF)",
        data=pdf_bytes,
        file_name=f"report_{result['filename']}.pdf",
        mime="application/pdf"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN ABOUT
# =====================================================
def show_about():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <h2>📖 Tentang Aplikasi</h2>
        <p>
            Aplikasi ini menggunakan <strong>Machine Learning</strong> untuk mengenali emosi dari sinyal suara.
            Model dilatih dengan dataset <strong>RAVDESS</strong> yang terdiri dari 2.880 file audio dari 24 aktor,
            mencakup 8 emosi: <em>marah, tenang, jijik, takut, bahagia, netral, sedih, terkejut</em>.
        </p>
        <p>
            <strong>Fitur yang diekstrak:</strong> MFCC, Delta, dan Delta².
            Dua model digunakan: <strong>KNN</strong> dan <strong>SVM</strong> (dengan GridSearchCV).
            Hasil akhir diambil dari prediksi SVM.
        </p>
        <p>
            Dibangun dengan <strong>Streamlit</strong>, <strong>Librosa</strong>, <strong>Plotly</strong>, dan <strong>Scikit-learn</strong>.
        </p>
        <p style="margin-top:1rem;">
            <strong>👨‍💻 Pengembang:</strong> Tim Speech Emotion Recognition
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# NAVIGASI SIDEBAR
# =====================================================
with st.sidebar:
    if os.path.exists("assets/logo_kampus.png"):
        st.image("assets/logo_kampus.png", width=180)
    else:
        st.markdown("### 🎤 SER")

    st.markdown("---")
    pages = ["🏠 Home", "📤 Upload Audio", "🎙️ Live Record", "📊 Analytics", "📄 Report", "📖 About"]
    # Gunakan session state untuk menyimpan pilihan
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 Home"
    choice = st.radio("Navigasi", pages, index=pages.index(st.session_state["page"]))
    st.session_state["page"] = choice

    st.markdown("---")
    with st.expander("🤖 Model"):
        st.write("KNN, SVM (GridSearchCV)")
    with st.expander("🧠 Fitur"):
        st.write("MFCC, Delta, Delta²")
    with st.expander("📊 Dataset"):
        st.write("RAVDESS (24 aktor, 8 emosi)")
    st.markdown("---")
    st.caption("v2.1 • Dibangun dengan Streamlit")

# =====================================================
# RENDER HALAMAN
# =====================================================
if choice == "🏠 Home":
    show_home()
elif choice == "📤 Upload Audio":
    show_upload()
elif choice == "🎙️ Live Record":
    show_live_record()
elif choice == "📊 Analytics":
    show_analytics()
elif choice == "📄 Report":
    show_report()
elif choice == "📖 About":
    show_about()

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#94a3b8; font-size:0.9rem;">
    🎤 Speech Emotion Recognition Dashboard • Built with Streamlit & Plotly • RAVDESS Dataset
</div>
""", unsafe_allow_html=True)
