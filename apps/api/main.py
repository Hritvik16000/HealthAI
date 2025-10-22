import os

import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
app = FastAPI(title="HealthAI API", version="0.1.0")
origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))


class RiskInput(BaseModel):
    age: int
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float


class SentimentInput(BaseModel):
    text: str


class TranslateInput(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/risk")
def predict_risk(x: RiskInput):
    score = 0.3 + 0.01 * (x.bmi - 25) + 0.001 * (x.systolic_bp - 120) + 0.001 * (x.heart_rate - 70)
    return {"risk_probability": max(0, min(1, score))}


@app.post("/analyze/sentiment")
def analyze_sentiment(x: SentimentInput):
    return {"label": "neutral", "score": 0.5}


@app.post("/translate")
def translate(x: TranslateInput):
    return {"translation": x.text}


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
