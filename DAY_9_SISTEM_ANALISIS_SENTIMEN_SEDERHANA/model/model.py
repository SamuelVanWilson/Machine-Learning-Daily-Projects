import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('./model/IMDB-Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

tfidf = TfidfVectorizer(max_features=5000)
x = tfidf.fit_transform(df['review'])
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')