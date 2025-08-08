from flask import Flask, request, jsonify, render_template
# from classify_old import predict_audio_class
from classify import predict_audio_class
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')  # Make sure index.html exists in 'templates/'

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_path)

        # Double-check file was saved
        if not os.path.exists(upload_path):
            return jsonify({"error": f"Uploaded file not found: {upload_path}"}), 500

        # Predict
        result = predict_audio_class(upload_path)

        # Clean up
        os.remove(upload_path)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # get port from env or fallback to 5000
    app.run(host="0.0.0.0", port=port)