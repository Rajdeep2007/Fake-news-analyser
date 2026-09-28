"""
backend/main.py

FastAPI backend that loads the trained model + vectorizer and exposes
a /predict endpoint. Run with:

    uvicorn backend.main:app --reload --port 8000

(run this command from the project root, i.e. the folder containing
both `backend/` and `frontend/`)
"""

import os
import re
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Source Credibility Analyzer API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "vectorizer.pkl"))
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ArticleRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str          # "REAL" or "FAKE"
    confidence: float    # 0.0 to 1.0
    top_words: list[str]  # words in this article with the strongest influence


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Source Credibility Analyzer API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ArticleRequest):
    cleaned = clean_text(request.text)
    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(proba[pred])

    # Find which words present in THIS article had the strongest weights
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    article_word_indices = vec.nonzero()[1]

    # Sort this article's words by absolute coefficient strength
    article_words_sorted = sorted(
        article_word_indices,
        key=lambda idx: abs(coefs[idx]),
        reverse=True
    )
    top_words = [feature_names[idx] for idx in article_words_sorted[:8]]

    return PredictionResponse(label=label, confidence=confidence, top_words=top_words)
