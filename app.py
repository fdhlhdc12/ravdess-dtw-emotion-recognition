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
# CSS KUSTOM (dengan pemisah navigasi)
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

/* Pemisah navigasi: garis bawah tiap item radio */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 10px 0;
    margin: 0;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:last-child {
    border-bottom: none;
}

.glass {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(12px);
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
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(96,165,250,0.05), transparent 70%);
    animation: rotate 20s linear infinite;
}
@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.hero h1, .hero p { position: relative; z-index: 1; }
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
}
.cta-button:hover {
    transform: scale(1.04);
    box-shadow: 0 12px 32px rgba(37, 99, 235, 0.5);
}
.fade-in { animation: fadeIn 0.8s ease-in-out; }
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
    background: #10b981; color: white; padding: 4px 16px; border-radius: 40px; font-weight: 600; display: inline-block;
}
.badge-medium {
    background: #f59e0b; color: #1e293b; padding: 4px 16px; border-radius: 40px; font-weight: 600; display: inline-block;
}
.badge-low {
    background: #ef4444; color: white; padding: 4px 16px; border-radius: 40px; font-weight: 600; display: inline-block;
}
.insight-box {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #2563eb;
    padding: 16px 20px;
    border-radius: 12px;
    margin: 12px 0;
}
.workflow-item {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}
.workflow-item:hover {
    transform: translateY(-4px);
    border-color: rgba(96,165,250,0.3);
}
.workflow-arrow { text-align: center; font-size: 2rem; color: #60a5fa; }
.recording-indicator {
    display: inline-block;
    width: 20px; height: 20px;
    background-color: #ef4444;
    border-radius: 50%;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.2); }
    100% { opacity: 0.4; transform: scale(0.9); }
}
.report-card {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 16px;
}
.report-card h4 { color: #94a3b8; font-weight: 400; font-size: 0.9rem; margin-bottom: 4px; }
.report-card .value { font-size: 1.8rem; font-weight: 700; color: white; }
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
# EMOTION ICON
# =====================================================
emotion_icon = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲"
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
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

# =====================================================
# BADGE & INSIGHT
# =====================================================
def confidence_badge(conf):
    if conf >= 0.8:
        return f'<span class="badge-high">🟢 High ({conf:.1%})</span>'
    elif conf >= 0.6:
        return f'<span class="badge-medium">🟡 Medium ({conf:.1%})</span>'
    else:
        return f'<span class="badge-low">🔴 Low ({conf:.1%})</span>'

def generate_insight(result):
    top3 = result['prob_df'].head(3)
    top_emo = result['top_emotion']
    conf = result['confidence']
    base = f"Primary emotion: **{top_emo}** with confidence {conf:.1%}."
    if conf > 0.8:
        base += " Very confident."
    elif conf > 0.6:
        base += " Moderately confident."
    else:
        base += " Low confidence – check audio quality."

    emotion_insights = {
        "happy": "High pitch variation & energy.",
        "sad": "Lower pitch, slower tempo.",
        "angry": "Sharp intensity, high volume.",
        "calm": "Steady, minimal fluctuation.",
        "fearful": "Trembling, higher pitch.",
        "disgust": "Harsh or nasal quality.",
        "surprised": "Sudden pitch rise, high energy.",
        "neutral": "Flat and monotone."
    }
    extra = emotion_insights.get(top_emo, "")
    if extra:
        base += " " + extra
    candidates = " | ".join([f"{row['Emotion']} ({row['Probability']:.1%})" for _, row in top3.iterrows()])
    base += f"\n\n**Top 3:** {candidates}"
    return base

# =====================================================
# DISPLAY RESULTS
# =====================================================
def display_results(result):
    st.session_state["result"] = result

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

    st.markdown(confidence_badge(result['confidence']), unsafe_allow_html=True)

    st.markdown("## 🤖 AI Insight")
    st.markdown(f'<div class="insight-box">{generate_insight(result)}</div>', unsafe_allow_html=True)

    st.markdown("## 🎵 Audio Visualization")
    col_wave, col_spec, col_mfcc = st.columns(3)
    y, sr, duration = result['y'], result['sr'], result['duration']

    with col_wave:
        time_axis = np.linspace(0, duration, len(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", line=dict(color="#60a5fa", width=2)))
        fig.update_layout(title="Waveform", xaxis_title="Time (sec)", yaxis_title="Amplitude", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_spec:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig = go.Figure(data=go.Heatmap(z=D, colorscale="Turbo"))
        fig.update_layout(title="Spectrogram", xaxis_title="Frames", yaxis_title="Frequency", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_mfcc:
        fig = go.Figure(data=go.Heatmap(z=result['mfcc'], colorscale="RdBu", zmid=0))
        fig.update_layout(title="MFCC (40 coeff)", xaxis_title="Time frames", yaxis_title="Coefficient", height=300, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

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

    st.markdown("## 📈 Emotion Analytics")
    a1, a2 = st.columns([1, 2])
    prob_df = result['prob_df']
    with a1:
        st.markdown("### 🏅 Ranking")
        for _, row in prob_df.iterrows():
            emo, prob = row["Emotion"], row["Probability"]
            st.markdown(f"**{emotion_icon.get(emo, '🎤')} {emo.upper()}**")
            st.progress(float(prob))
            st.caption(f"{prob:.2%}")

    with a2:
        fig = px.bar(prob_df, x="Emotion", y="Probability", color="Probability", color_continuous_scale="Blues")
        fig.update_layout(height=420, coloraxis_showscale=False, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🎯 Emotion Distribution")
    top_prob = float(prob_df.iloc[0]["Probability"])
    top_name = str(prob_df.iloc[0]["Emotion"]).upper()
    fig = go.Figure(data=[go.Pie(labels=prob_df["Emotion"], values=prob_df["Probability"], hole=0.70, textinfo="none")])
    fig.update_layout(
        annotations=[dict(text=f"<b>{top_prob:.1%}</b><br>{top_name}", showarrow=False, font=dict(size=24, color="white"))],
        height=380, **dark_layout()
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HALAMAN ABOUT
# =====================================================
def show_about():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1e293b, #0f172a); padding: 3rem 2rem; border-radius: 40px; text-align: center; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📖 About This Project
        </h1>
        <p style="font-size: 1.2rem; color: #94a3b8; max-width: 700px; margin: 1rem auto;">
            Mengenal lebih dalam tentang teknologi di balik Speech Emotion Recognition.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h2>🎯 Tujuan</h2>
            <p>Mengidentifikasi emosi manusia dari sinyal suara menggunakan ekstraksi fitur MFCC dan model Machine Learning.</p>
            <h2>🧠 Model</h2>
            <p><strong>KNN</strong> dan <strong>SVM</strong> (dengan GridSearchCV) dilatih pada dataset RAVDESS.</p>
            <h2>📊 Dataset</h2>
            <p>RAVDESS: 2.880 file audio, 24 aktor, 8 emosi (marah, tenang, jijik, takut, bahagia, netral, sedih, terkejut).</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h2>🔧 Teknologi</h2>
            <ul>
                <li>Streamlit – Dashboard interaktif</li>
                <li>Librosa – Ekstraksi fitur audio</li>
                <li>Plotly – Visualisasi interaktif</li>
                <li>Scikit-learn – Model ML</li>
                <li>Reportlab – Generate PDF</li>
            </ul>
            <h2>📥 Dataset RAVDESS</h2>
            <p>Dataset ini dapat diakses dan diunduh melalui tautan berikut:</p>
            <a href="https://drive.google.com/drive/folders/1w8B4k8n9z3w6X5y4z3v2c1b" target="_blank" style="color:#60a5fa;">🔗 Klik untuk mengakses dataset</a>
            <p style="margin-top:1rem;"><strong>Catatan:</strong> Pastikan Anda memiliki izin untuk mengunduh dataset ini untuk keperluan akademik.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8;">
        <p>Versi 2.5 • © 2026 Speech Emotion Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN HOME (Upload + Live Record)
# =====================================================
def show_home():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 4.5rem; font-weight: 800; letter-spacing: -1px; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🎤 Speech Emotion AI
        </h1>
        <p style="font-size: 1.5rem; max-width: 700px; margin: 0.5rem auto; color: #94a3b8;">
            Deteksi emosi dari suara secara instan dengan Machine Learning
        </p>
        <div style="margin-top: 1.5rem; display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <span style="background: rgba(30,41,59,0.8); padding: 8px 20px; border-radius: 40px; color: #94a3b8; border: 1px solid rgba(255,255,255,0.05);">🎯 Akurasi Tinggi</span>
            <span style="background: rgba(30,41,59,0.8); padding: 8px 20px; border-radius: 40px; color: #94a3b8; border: 1px solid rgba(255,255,255,0.05);">⚡ Proses Cepat</span>
            <span style="background: rgba(30,41,59,0.8); padding: 8px 20px; border-radius: 40px; color: #94a3b8; border: 1px solid rgba(255,255,255,0.05);">📊 Visualisasi Interaktif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Live Record"])

    with tab1:
        st.markdown("""
        <div class="glass" style="margin-bottom: 20px;">
            <h3>Upload File Audio</h3>
            <p>Support: WAV, MP3, M4A, OGG, FLAC</p>
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
            with st.status("🧠 Menganalisis emosi...", expanded=False) as status:
                result = analyze_audio(audio_bytes, uploaded_file.name)
                status.update(label="✅ Selesai!", state="complete")
            st.toast("✨ Analisis selesai!", icon="🎉")
            display_results(result)

    with tab2:
        st.markdown("""
        <div class="glass" style="margin-bottom: 20px;">
            <h3>Rekam Suara Langsung</h3>
            <p>Klik tombol di bawah, izinkan akses mikrofon, rekam, lalu tekan stop (atau diam selama 2 detik). Hasil akan muncul otomatis.</p>
        </div>
        """, unsafe_allow_html=True)

        audio_bytes = audio_recorder(
            text="🎙️ Klik untuk merekam",
            recording_color="#e74c3c",
            neutral_color="#2563eb",
            icon_size="3x",
            energy_threshold=0.5,
            pause_threshold=2.0,
            sample_rate=16000,
            key="recorder"
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with st.status("🧠 Menganalisis rekaman...", expanded=False) as status:
                result = analyze_audio(audio_bytes, "recorded.wav")
                status.update(label="✅ Selesai!", state="complete")
            st.toast("🎙️ Analisis rekaman selesai!", icon="🎤")
            display_results(result)
        else:
            st.info("Tekan tombol di atas, rekam, dan hasil akan muncul otomatis.")

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
        st.warning("Belum ada hasil analisis. Silakan upload atau rekam audio terlebih dahulu di halaman Home.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    result = st.session_state["result"]
    prob_df = result['prob_df']

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
    st.markdown("### 🎵 Pitch Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🎵 Average Pitch", f"{result['pitch_mean']:.1f} Hz")
    with c2:
        st.metric("📊 Pitch Std Dev", f"{result['pitch_std']:.1f} Hz")

    st.markdown("---")
    st.markdown("### 🎼 Audio Visualizations")
    tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🌈 Spectrogram", "🎚️ MFCC Heatmap"])

    y, sr, duration = result['y'], result['sr'], result['duration']

    with tab1:
        time_axis = np.linspace(0, duration, len(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", line=dict(color="#60a5fa", width=2)))
        fig.update_layout(title="Waveform", xaxis_title="Time (sec)", yaxis_title="Amplitude", height=400, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig = go.Figure(data=go.Heatmap(z=D, colorscale="Turbo"))
        fig.update_layout(title="Spectrogram", xaxis_title="Frames", yaxis_title="Frequency", height=400, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure(data=go.Heatmap(z=result['mfcc'], colorscale="RdBu", zmid=0))
        fig.update_layout(title="MFCC Coefficients (40)", xaxis_title="Time frames", yaxis_title="Coefficient index", height=400, **dark_layout())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Distribusi Probabilitas")
    fig = px.bar(prob_df, x="Emotion", y="Probability", color="Probability", color_continuous_scale="Viridis")
    fig.update_layout(height=400, **dark_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🕸️ Radar Emosi")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=prob_df["Probability"], theta=prob_df["Emotion"],
        fill='toself', name='Probabilitas', line_color='#60a5fa'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450, **dark_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🤖 AI Insight")
    st.markdown(f'<div class="insight-box">{generate_insight(result)}</div>', unsafe_allow_html=True)
    st.markdown(confidence_badge(result['confidence']), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN REPORT
# =====================================================
def show_report():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1e293b, #0f172a); padding: 2rem; border-radius: 40px; text-align: center; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
        <h1 style="font-size: 2.8rem; font-weight: 700; color: white;">📄 Laporan Analisis</h1>
        <p style="color: #94a3b8;">Ringkasan lengkap hasil prediksi emosi dari audio</p>
    </div>
    """, unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Belum ada hasil analisis. Silakan upload atau rekam audio terlebih dahulu di halaman Home.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    result = st.session_state["result"]
    prob_df = result['prob_df']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="report-card"><h4>📁 Nama File</h4><div class="value">{result['filename']}</div></div>
        <div class="report-card"><h4>⏱ Durasi</h4><div class="value">{result['duration']:.2f} detik</div></div>
        <div class="report-card"><h4>🎚 Sample Rate</h4><div class="value">{result['sr']} Hz</div></div>
        <div class="report-card"><h4>🎵 Pitch Rata-rata</h4><div class="value">{result['pitch_mean']:.1f} Hz</div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="report-card"><h4>🏆 Emosi Terdeteksi</h4><div class="value" style="display:flex; align-items:center; gap:10px;">{emotion_icon.get(result['top_emotion'], '🎤')} {result['top_emotion'].upper()}</div></div>
        <div class="report-card"><h4>🎯 Confidence</h4><div class="value">{result['confidence']:.2%}</div></div>
        <div class="report-card"><h4>📊 Tingkat Keyakinan</h4><div>{confidence_badge(result['confidence'])}</div></div>
        <div class="report-card"><h4>🤖 Model Final</h4><div class="value">SVM</div></div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Probabilitas per Emosi")
    for _, row in prob_df.iterrows():
        emo, prob = row["Emotion"], row["Probability"]
        col1, col2, col3 = st.columns([2, 4, 1])
        with col1: st.markdown(f"**{emotion_icon.get(emo, '🎤')} {emo.upper()}**")
        with col2: st.progress(float(prob))
        with col3: st.markdown(f"<p style='text-align:right;'>{prob:.2%}</p>", unsafe_allow_html=True)

    st.markdown("### 💾 Unduh Laporan")
    col1, col2, col3 = st.columns(3)
    csv = prob_df.to_csv(index=False).encode('utf-8')
    with col1:
        st.download_button("📥 CSV (Probabilitas)", csv, f"emotion_probs_{result['filename']}.csv", "text/csv", use_container_width=True)

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
    {prob_df.to_string(index=False)}
    """
    with col2:
        st.download_button("📄 TXT (Laporan)", report_text, f"report_{result['filename']}.txt", "text/plain", use_container_width=True)

    def generate_pdf(result):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 20)
        c.drawString(100, height - 100, "Speech Emotion Recognition Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 130, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Top 5 Emotion Probabilities")
        y -= 25
        c.setFont("Helvetica", 12)
        top5 = prob_df.head(5)
        for _, row in top5.iterrows():
            c.drawString(100, y, f"{row['Emotion']}: {row['Probability']:.2%}")
            y -= 20
        c.save()
        buffer.seek(0)
        return buffer

    pdf_bytes = generate_pdf(result)
    with col3:
        st.download_button("📄 PDF (Laporan)", pdf_bytes, f"report_{result['filename']}.pdf", "application/pdf", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# HALAMAN MODEL (Edukasi)
# =====================================================
def show_model():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1e293b, #0f172a); padding: 2rem; border-radius: 40px; text-align: center; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
        <h1 style="font-size: 2.8rem; font-weight: 700; color: white;">🧠 Model & Algoritma</h1>
        <p style="color: #94a3b8;">Bagaimana AI mengenali emosi dari suara</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h2>🎯 Proses Pengenalan Emosi</h2>
        <p>Secara garis besar, pipeline sistem ini terdiri dari tiga tahap utama:</p>
        <ol>
            <li><strong>Ekstraksi Fitur</strong> – Mengubah sinyal audio menjadi representasi numerik (MFCC, Delta, Delta²).</li>
            <li><strong>Klasifikasi</strong> – Menggunakan model Machine Learning (KNN & SVM) untuk memprediksi emosi.</li>
            <li><strong>Kesimpulan</strong> – Menggabungkan prediksi dan memberikan hasil akhir beserta tingkat keyakinan.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h2>🎼 Ekstraksi Fitur</h2>
            <p><strong>MFCC</strong> (Mel-Frequency Cepstral Coefficients) menangkap karakteristik spektral suara yang berkorelasi dengan persepsi pendengaran manusia.</p>
            <p><strong>Delta</strong> dan <strong>Delta²</strong> adalah turunan pertama dan kedua dari MFCC, yang menangkap dinamika temporal (perubahan seiring waktu).</p>
            <p>Kombinasi ini memberikan informasi yang kaya untuk membedakan emosi.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h2>📊 Dataset RAVDESS</h2>
            <p>RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) berisi 2.880 file suara dari 24 aktor (12 pria, 12 wanita).</p>
            <p>Setiap aktor merekam 8 emosi dengan dua intensitas (normal dan kuat).</p>
            <p>Dataset ini banyak digunakan sebagai benchmark untuk SER.</p>
            <p><a href="https://drive.google.com/drive/folders/1w8B4k8n9z3w6X5y4z3v2c1b" target="_blank" style="color:#60a5fa;">🔗 Download Dataset</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="glass">
        <h2>🤖 Algoritma Klasifikasi</h2>
        <p>Dua model yang digunakan:</p>
        <ul>
            <li><strong>KNN (K-Nearest Neighbors)</strong> – Mengklasifikasikan berdasarkan mayoritas emosi dari k tetangga terdekat di ruang fitur.</li>
            <li><strong>SVM (Support Vector Machine)</strong> – Mencari hyperplane optimal yang memisahkan kelas emosi dengan margin maksimum.</li>
        </ul>
        <p>Kedua model dilatih dengan <strong>GridSearchCV</strong> untuk menemukan hyperparameter terbaik.</p>
        <p>Hasil akhir diambil dari prediksi SVM karena umumnya lebih akurat untuk data berdimensi tinggi.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="glass">
        <h2>📈 Evaluasi Model</h2>
        <p>Setelah pelatihan, model dievaluasi dengan metrik akurasi, precision, recall, dan F1-score.</p>
        <p>Confusion matrix menunjukkan performa per kelas emosi.</p>
        <p>Dengan tuning hyperparameter, SVM mencapai akurasi terbaik pada dataset RAVDESS.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# NAVIGASI SIDEBAR (dengan pemisah garis)
# =====================================================
with st.sidebar:
    if os.path.exists("assets/logo_kampus.png"):
        st.image("assets/logo_kampus.png", width=180)
    else:
        st.markdown("### 🎤 SER")

    st.markdown("---")

    # Menu utama dengan radio yang sudah diberi CSS border-bottom
    pages = ["📖 About", "🏠 Home", "📊 Analytics", "📄 Report", "🧠 Model"]
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 Home"
    choice = st.radio("Navigasi", pages, index=pages.index(st.session_state["page"]), key="nav")
    if choice != st.session_state["page"]:
        st.session_state["page"] = choice
        st.rerun()

    st.markdown("---")
    with st.expander("🤖 Model"):
        st.write("KNN, SVM (GridSearchCV)")
    with st.expander("🧠 Fitur"):
        st.write("MFCC, Delta, Delta²")
    with st.expander("📊 Dataset"):
        st.write("RAVDESS (24 aktor, 8 emosi)")
    st.markdown("---")
    st.caption("v2.5 • Dibangun dengan Streamlit")

# =====================================================
# RENDER HALAMAN
# =====================================================
if st.session_state["page"] == "📖 About":
    show_about()
elif st.session_state["page"] == "🏠 Home":
    show_home()
elif st.session_state["page"] == "📊 Analytics":
    show_analytics()
elif st.session_state["page"] == "📄 Report":
    show_report()
elif st.session_state["page"] == "🧠 Model":
    show_model()

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#94a3b8; font-size:0.9rem;">
    🎤 Speech Emotion Recognition Dashboard • Built with Streamlit & Plotly • RAVDESS Dataset
</div>
""", unsafe_allow_html=True)
