from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)


model = tf.keras.models.load_model('model/mnist_model.keras')

@app.route('/')
def serve_index():
    return send_from_directory('./', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Terima file gambar dari request
        file = request.files['image']
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # 2. Buka dan praproses gambar
        img = Image.open(file).convert('L')  # Konversi ke grayscale
        img = img.resize((28, 28))          # Resize ke 28x28
        img_array = np.array(img) / 255.0    # Normalisasi ke [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Bentuk input model

        # 3. Prediksi
        pred = model.predict(img_array)
        predicted_class = int(np.argmax(pred))

        # 4. Kembalikan hasil
        print(predicted_class)
        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)