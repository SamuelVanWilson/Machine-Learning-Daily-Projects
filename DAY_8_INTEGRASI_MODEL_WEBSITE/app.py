import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

app = Flask(__name__)

# Memuat model yang telah dilatih
model = tf.keras.models.load_model('model.h5')

# Daftar nama kelas CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    """
    Fungsi untuk melakukan preprocessing gambar agar sesuai dengan input model:
    - Mengubah ukuran gambar menjadi 32x32 pixel (ukuran CIFAR-10)
    - Mengubah gambar menjadi array NumPy dan menormalisasi nilai pixel ke rentang 0-1
    - Menambahkan dimensi batch
    """
    # Ubah ukuran gambar menjadi 32x32
    image = image.resize((32, 32))
    # Ubah gambar ke array dan normalisasi
    image = np.array(image) / 255.0
    # Jika gambar grayscale, ubah menjadi 3 channel
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    # Jika gambar memiliki 4 channel (RGBA), buang channel alpha
    elif image.shape[-1] == 4:
        image = image[..., :3]
    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                # Buka gambar yang diunggah
                image = Image.open(file)
                # Preprocessing gambar
                processed_image = preprocess_image(image)
                # Melakukan prediksi menggunakan model
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions[0])
                result = class_names[predicted_class]
            except Exception as e:
                result = "Terjadi kesalahan: " + str(e)
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
