import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load CSV
data = pd.read_csv(r"C:\Users\mynam\Downloads\Clause-Level-Sentiment-Analysis\Final_Annotation.csv")

# Strip whitespace from column names
data.columns = data.columns.str.strip()
print("Columns in CSV:", data.columns)

# Drop any rows with missing values in relevant columns
data = data.dropna(subset=['Clause', 'Final Sentiment'])

# Features and labels
X = data['Clause']             # text column
y = data['Final Sentiment']    # label column

# Build pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X, y)
print("Model trained successfully!")

# Save the model
joblib.dump(pipeline, r"C:\Users\mynam\Downloads\Clause-Level-Sentiment-Analysis\models\taglish_sentiment_model.pkl")
print("Model saved to 'models/taglish_sentiment_model.pkl'")
