# 🎵 Audio Environment Classifier

A sophisticated web-based audio classification system that uses machine learning to identify and categorize environmental sounds including rain, car sounds, and background noise. Built with Flask and TensorFlow Lite for efficient edge deployment.

## 🚀 Project Overview

This project implements a lightweight yet powerful audio classification system that can accurately distinguish between three primary environmental sound categories:
- **Rain and Thunder** 🌧️
- **Car Sounds** 🚗
- **Background Noise** 🔊

The system leverages advanced audio processing techniques and a pre-trained TensorFlow Lite model optimized for real-time inference, making it suitable for both web applications and edge computing scenarios.

## ✨ Key Features

- **Real-time Audio Classification**: Upload audio files and get instant predictions
- **High Accuracy**: Achieves over 90% confidence threshold for reliable classifications
- **Web-based Interface**: Clean, responsive UI built with Bootstrap 5
- **Mobile Optimized**: Works seamlessly on desktop and mobile devices
- **Audio Playback**: Built-in audio controls for testing sample sounds
- **Visual Feedback**: Dynamic image changes based on classification results
- **Production Ready**: Configured for deployment on Heroku with Gunicorn

## 🛠️ Technology Stack

### Backend
- **Flask**: Lightweight Python web framework
- **TensorFlow Lite**: Optimized machine learning inference
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Data preprocessing and encoding
- **NumPy**: Numerical computations

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **JavaScript**: Dynamic client-side interactions
- **HTML5/CSS3**: Modern web standards

### Audio Processing
- **MFCC Features**: 40-dimensional Mel-frequency cepstral coefficients
- **Bandpass Filtering**: 50Hz - 7900Hz frequency range
- **Spectral Analysis**: RMS, spectral contrast, zero-crossing rate
- **Chroma Features**: Chromagram for harmonic content analysis

## 📊 Model Architecture

The system uses a TensorFlow Lite model (`rain_car_noise_model.tflite`) trained on:
- **Input Features**: 143-dimensional feature vectors
- **MFCC**: 40 coefficients with delta and delta-delta features
- **Spectral Features**: RMS, spectral contrast, rolloff, centroid
- **Temporal Features**: Zero-crossing rate, chroma features

### Preprocessing Pipeline
1. **Audio Loading**: 16kHz mono conversion
2. **Duration Normalization**: 2-second clips
3. **Bandpass Filtering**: Noise reduction and frequency focus
4. **Feature Extraction**: MFCC and spectral features
5. **Normalization**: Standard scaling for model input

## 🎯 Classification Categories

| Category | Description | Use Cases |
|----------|-------------|-----------|
| **Rain & Thunder** | Natural precipitation sounds | Weather monitoring, ambient sound generation |
| **Car Sounds** | Vehicle engine and movement noises | Traffic monitoring, automotive applications |
| **Background Noise** | General ambient environmental sounds | Noise pollution analysis, audio quality assessment |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/audio-environment-classifier.git
cd audio-environment-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the web interface**
Open your browser and navigate to `http://localhost:5000`

### Production Deployment

For Heroku deployment:
```bash
# Create Heroku app
heroku create your-audio-classifier

# Deploy
git push heroku main
```

## 📁 Project Structure

```
audio-environment-classifier/
├── app.py                 # Flask application entry point
├── classify.py           # Audio classification logic
├── model/                # Pre-trained ML models
│   ├── rain_car_noise_model.tflite
│   ├── feature_scaler.pkl
│   └── label_encoder.pkl
├── static/               # Static web assets
│   ├── css/
│   ├── js/
│   ├── img/
│   └── music/           # Sample audio files
├── templates/           # HTML templates
├── training_samples/    # Training audio samples
├── uploaded/           # User uploaded files
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku deployment config
└── README.md          # Project documentation
```

## 🔧 API Endpoints

### POST /predict
Classify an uploaded audio file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (.wav, .mp3, .wmv)

**Response:**
```json
{
  "result": {
    "label": "Rain and thunder",
    "confidence": 0.95,
    "raw_scores": [0.02, 0.95, 0.03]
  }
}
```

## 🎵 Supported Audio Formats

- **WAV**: Uncompressed audio
- **MP3**: Compressed audio
- **WMV**: Windows Media Audio

**Recommended Specifications:**
- Sample Rate: 16kHz
- Channels: Mono
- Duration: 2+ seconds
- Format: WAV for best results

## 🧪 Testing

The application includes built-in testing capabilities:
- **Sample Audio Files**: Pre-loaded sounds for each category
- **Visual Feedback**: Images change based on classification
- **Audio Playback**: Test sounds directly in the browser
- **Confidence Display**: Shows prediction certainty

## 📈 Performance Metrics

- **Model Size**: < 1MB (TensorFlow Lite)
- **Inference Time**: < 100ms on modern hardware
- **Accuracy**: > 90% on test dataset
- **Memory Usage**: < 100MB RAM

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow Lite team for the optimized inference engine
- Librosa team for excellent audio processing tools
- Bootstrap team for the responsive UI framework
- Contributors and testers who helped improve the system

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation wiki

---
