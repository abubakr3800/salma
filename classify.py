import numpy as np
import tensorflow as tf
import librosa
import joblib
import os
from scipy.signal import butter, lfilter

# === Configuration ===
SAMPLE_RATE = 16000
DURATION = 2  # in seconds
N_MFCC = 40
MAX_FRAMES = 48
FEATURE_SIZE = 143
CONFIDENCE_THRESHOLD = 0.90

# === Load Label Encoder and Scaler ===
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/feature_scaler.pkl")
CLASSES = label_encoder.classes_.tolist()

# === Bandpass Filter ===
def bandpass_filter(data, sr, lowcut=50, highcut=7900, order=5):
    nyq = 0.5 * sr
    if highcut >= nyq:
        highcut = nyq - 1
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

# === Feature Extraction ===
def extract_features(y, sr):
    y = bandpass_filter(y, sr)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    min_len = min(f.shape[1] for f in [mfcc, delta, delta2, rms, contrast, zcr, rolloff, chroma, centroid])
    features = np.vstack([f[:, :min_len] for f in [mfcc, delta, delta2, rms, contrast, zcr, rolloff, chroma, centroid]]).T

    if features.shape[0] < MAX_FRAMES:
        features = np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)))
    else:
        features = features[:MAX_FRAMES]

    return features.astype(np.float32)

# === Load TFLite Model ===
interpreter = tf.lite.Interpreter(model_path="model/rain_car_noise_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Prediction Function ===
def predict_audio_class(file_path):
    try:
        # Convert to mono float32
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, int(SAMPLE_RATE * DURATION - len(y))))

        features = extract_features(y, sr)
        features_flat = features.reshape(1, -1)
        features_scaled = scaler.transform(features_flat).reshape(1, MAX_FRAMES, FEATURE_SIZE, 1)

        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_index = np.argmax(output)
        confidence = output[pred_index]
        predicted_label = CLASSES[pred_index]

        return {
            "label": predicted_label,
            "confidence": float(confidence),
            "raw_scores": output.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
