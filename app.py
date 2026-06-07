import streamlit as st
import librosa, numpy as np, joblib
from src.features import extract_features
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("**Dataset:** RAVDESS | **Method:** MFCC Features + DTW/Random Forest")

model = joblib.load('models/emotion_classifier.pkl')

EMOTIONS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}

uploaded = st.file_uploader("Upload file audio (.wav)", type=['wav'])

if uploaded:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format='audio/wav')
    
    # Load dan ekstrak fitur
    y, sr = librosa.load(io.BytesIO(audio_bytes), duration=3.0, offset=0.5)
    feat, mfcc = extract_features(io.BytesIO(audio_bytes))
    
    # Prediksi
    pred = model.predict([feat])[0]
    prob = model.predict_proba([feat])[0]
    
    st.markdown(f"## Emosi Terdeteksi: {EMOTIONS[pred]} **{pred.upper()}**")
    
    # Bar chart probabilitas
    classes = model.classes_
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(classes, prob, color=['#ff6b6b' if c==pred else '#74b9ff' for c in classes])
    ax.set_xlabel('Probabilitas')
    ax.set_title('Distribusi Prediksi Emosi')
    st.pyplot(fig)
    
    # Visualisasi MFCC (DTW input)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(mfcc, x_axis='time', ax=ax2)
    ax2.set_title('MFCC (Time Series — input untuk DTW)')
    plt.colorbar(ax2.collections[0], ax=ax2)
    st.pyplot(fig2)
    
    st.caption("MFCC time series di atas adalah representasi yang dibandingkan dengan DTW antar sampel suara")
