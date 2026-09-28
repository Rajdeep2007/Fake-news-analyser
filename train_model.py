"""
train_model.py

Trains the fake news classifier and saves the vectorizer + model to disk
so the FastAPI backend can load them without retraining every time.

Run this ONCE (locally, with internet access) before starting the backend:
    python train_model.py

Requires: pandas, scikit-learn, kagglehub, joblib
"""

import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import kagglehub


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

    fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(path, "True.csv"))

    fake_df['label'] = 0  # FAKE
    true_df['label'] = 1  # REAL

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning text...")
    df['clean_content'] = df['content'].apply(clean_text)

    X = df['clean_content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Tuning Logistic Regression with GridSearchCV...")
    param_grid = {'C': [0.1, 1, 10, 50]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train_tfidf, y_train)

    model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))

    # Save both the vectorizer and the trained model so the API can load them
    os.makedirs("backend/artifacts", exist_ok=True)
    joblib.dump(vectorizer, "backend/artifacts/vectorizer.pkl")
    joblib.dump(model, "backend/artifacts/model.pkl")
    print("\nSaved vectorizer.pkl and model.pkl to backend/artifacts/")


if __name__ == "__main__":
    main()
