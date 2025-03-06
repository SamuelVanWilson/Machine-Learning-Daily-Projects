from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Muat model dan vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Transformasi teks
    text_vector = tfidf.transform([text])
    # Prediksi
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)