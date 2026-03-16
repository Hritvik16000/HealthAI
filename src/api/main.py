from pathlib import Path
from typing import Optional
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.sentiment_engine import HybridSentimentEngine

BASE_DIR = Path(__file__).resolve().parents[2]

TAB_DIR = BASE_DIR / "artifacts" / "tabular"
NLP_DIR = BASE_DIR / "artifacts" / "nlp"
CLUSTER_DIR = BASE_DIR / "artifacts" / "cluster"
ASSOC_DIR = BASE_DIR / "artifacts" / "association"

risk_model = joblib.load(TAB_DIR / "risk_classifier.pkl")
risk_encoder = joblib.load(TAB_DIR / "risk_label_encoder.pkl")
los_model = joblib.load(TAB_DIR / "los_regressor.pkl")
cluster_model = joblib.load(CLUSTER_DIR / "kmeans_pipeline.pkl")
sentiment_engine = HybridSentimentEngine()

cluster_names = {}
cluster_names_path = CLUSTER_DIR / "cluster_names.json"
if cluster_names_path.exists():
    with open(cluster_names_path, "r") as f:
        cluster_names = json.load(f)

app = FastAPI(title="HealthAI API", version="2.0.0")


class PatientInput(BaseModel):
    age: int
    gender: int
    bmi: float
    blood_pressure: float
    glucose: float
    cholesterol: float
    heart_rate: float
    smoker: int
    diabetes_history: int


class TextInput(BaseModel):
    review: str


@app.get("/")
def root():
    return {"message": "HealthAI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/risk")
def predict_risk(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    pred = risk_model.predict(df)[0]
    label = risk_encoder.inverse_transform([pred])[0]
    return {"risk_prediction": str(label)}


@app.post("/predict/los")
def predict_los(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    pred = los_model.predict(df)[0]
    return {"predicted_length_of_stay": round(float(pred), 2)}


@app.post("/cluster/patient")
def cluster_patient(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    cluster = int(cluster_model.predict(df)[0])
    cluster_name = cluster_names.get(str(cluster), cluster_names.get(cluster, f"Cluster {cluster}"))
    return {
        "cluster_id": cluster,
        "cluster_name": cluster_name
    }


@app.post("/predict/sentiment")
def predict_sentiment(data: TextInput):
    result = sentiment_engine.predict(data.review)
    return result


@app.get("/association/rules")
def association_rules(limit: Optional[int] = 10):
    csv_path = ASSOC_DIR / "association_rules.csv"
    df = pd.read_csv(csv_path)

    keep_cols = [c for c in ["antecedents", "consequents", "support", "confidence", "lift"] if c in df.columns]
    df = df[keep_cols].head(limit)

    for col in ["support", "confidence", "lift"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    return {"rules": df.to_dict(orient="records")}
