import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib


data = pd.read_csv(r"C:\Users\mynam\Downloads\Clause-Level-Sentiment-Analysis\Final_Annotation.csv")


data.columns = data.columns.str.strip()
print("Columns in CSV:", data.columns)

data = data.dropna(subset=['Clause', 'Final Sentiment'])

X = data['Clause']
y = data['Final Sentiment']

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X, y)
print("Model trained successfully!")

joblib.dump(pipeline, r"C:\Users\mynam\Downloads\Clause-Level-Sentiment-Analysis\models\taglish_sentiment_model.pkl")
print("Model saved to 'models/taglish_sentiment_model.pkl'")
