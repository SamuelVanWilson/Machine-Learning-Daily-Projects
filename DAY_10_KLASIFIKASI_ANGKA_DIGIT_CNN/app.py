from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('mnist_cnn.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert ke grayscale
    img = img.resize((28, 28))  # Resize ke 28x28
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Tidak ada file!"
    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong!"
    
    # Simpan file & praproses
    file_path = f"static/uploads/{file.filename}"
    file.save(file_path)
    img_array = preprocess_image(file_path)
    
    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    return render_template('index.html', prediction=predicted_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)