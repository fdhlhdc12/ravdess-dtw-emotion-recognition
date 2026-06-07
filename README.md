# 🎤 Speech Emotion Recognition using RAVDESS

## Overview

This project performs speech emotion classification using the RAVDESS dataset.

The workflow includes:

* Audio preprocessing
* MFCC feature extraction
* K-Nearest Neighbor (KNN)
* Support Vector Machine (SVM)
* Interactive Streamlit dashboard

## Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Supported emotions:

* Angry
* Calm
* Disgust
* Fearful
* Happy
* Neutral
* Sad
* Surprised

## Dashboard Features

* Upload WAV audio
* Audio playback
* Waveform visualization
* Spectrogram visualization
* KNN prediction
* SVM prediction
* Emotion probability chart

## Run Locally

pip install -r requirements.txt

streamlit run app.py

## Models

* model_knn.pkl
* model_svm.pkl
* label_encoder.pkl

## Author

Speech Emotion Recognition Project using MFCC, KNN, and SVM.
