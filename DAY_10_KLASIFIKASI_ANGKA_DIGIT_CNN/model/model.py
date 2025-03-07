import tensorflow as tf
from tensorflow.keras import layers, models

# Muat dataset MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalisasi & reshape data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Bangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 kelas (0-9)
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Simpan model
model.save('mnist_cnn.keras')

# Evaluasi
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Akurasi Test: {test_acc}')