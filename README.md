# Speech Emotion Recognition using DTW
## Deskripsi
Proyek ini merupakan implementasi klasifikasi emosi suara menggunakan metode Dynamic Time Warping (DTW) dan K-Nearest Neighbor (KNN) pada dataset RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).Penelitian ini dilakukan sebagai tugas Mata Kuliah Analisis Data Lanjut Program Magister Statistika.
---

## Tujuan
Mengklasifikasikan emosi pada data suara berdasarkan kemiripan pola temporal menggunakan Dynamic Time Warping (DTW).
---

## Dataset
Dataset yang digunakan adalah RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

Emosi yang digunakan:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Karena ukuran dataset cukup besar, dataset tidak disimpan dalam repository ini dan diakses melalui Google Drive saat proses training pada Google Colab.
---

## Metodologi
Tahapan analisis yang dilakukan:

1. Data Collection
2. Audio Preprocessing
3. Feature Extraction menggunakan MFCC
4. Similarity Measurement menggunakan DTW
5. Klasifikasi menggunakan KNN-DTW
6. Evaluasi Model
7. Deployment menggunakan Streamlit

## Author
Nama: Caesariansyah Dwi Fadhilah
Program Studi Magister Statistika
Institut Teknologi Sepuluh Nopember
Tahun 2026
