from src.features import build_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib, numpy as np

# Build dataset
X, y, mfcc_series = build_dataset('data/')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: SVM dengan fitur vektor
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(X_train, y_train)
print("SVM:", classification_report(y_test, svm.predict(X_test)))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("RF:", classification_report(y_test, rf.predict(X_test)))

# Simpan model terbaik
joblib.dump(rf, 'models/emotion_classifier.pkl')
joblib.dump({'mean': X_train.mean(0), 'std': X_train.std(0)}, 'models/scaler.pkl')
