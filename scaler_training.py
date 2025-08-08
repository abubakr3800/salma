import os
import numpy as np
import joblib
import librosa
from classify_old import extract_features  # replace with your actual import

SAMPLE_RATE = 16000
MAX_FRAMES = 48
FEATURE_SIZE = 143

AUDIO_DIR = "training_samples"  # folder of .wav or .wmv files
feature_list = []

for filename in os.listdir(AUDIO_DIR):
    if not filename.lower().endswith((".wav", ".wmv")):
        continue
    path = os.path.join(AUDIO_DIR, filename)
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    features = extract_features(y, sr)  # shape: (48, 143)
    features_flat = features.reshape(-1)  # shape: (6864,)
    feature_list.append(features_flat)

X = np.array(feature_list)
print(f"Shape of feature matrix: {X.shape}")  # Should be (N, 6864)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, "model/feature_scaler.pkl")
print("✅ New feature_scaler.pkl saved!")