import os, json, pickle, io
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianTokenizer, MarianMTModel

app = FastAPI(title="HealthAI API", version="0.3.0")
origins = os.getenv("CORS_ORIGINS","http://localhost:8501").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

TAB_DIR = Path("artifacts/tabular")
NLP_DIR = Path("artifacts/nlp")
TS_DIR = Path("artifacts/timeseries")
CNN_DIR = Path("artifacts/cnn")

# ---- Safe lazy loads ----
try:
    with open(TAB_DIR/"clf_gb.pkl","rb") as f: CLF = pickle.load(f)
except Exception:
    CLF = None
try:
    with open(TAB_DIR/"reg_gb.pkl","rb") as f: REG = pickle.load(f)
except Exception:
    REG = None
try:
    with open(TAB_DIR/"preprocess_pipeline.pkl","rb") as f: PIPE = pickle.load(f)
except Exception:
    PIPE = None
try:
    with open(NLP_DIR/"sentiment_tfidf.pkl","rb") as f:
        _s = pickle.load(f)
        TFIDF = _s["vectorizer"]; SENT = _s["model"]
except Exception:
    TFIDF = None; SENT = None
t_en_es_path = NLP_DIR/"Helsinki-NLP__opus-mt-en-es"
t_es_en_path = NLP_DIR/"Helsinki-NLP__opus-mt-es-en"
try:
    TOK_EN_ES = MarianTokenizer.from_pretrained(t_en_es_path)
    MOD_EN_ES = MarianMTModel.from_pretrained(t_en_es_path)
    TOK_ES_EN = MarianTokenizer.from_pretrained(t_es_en_path)
    MOD_ES_EN = MarianMTModel.from_pretrained(t_es_en_path)
except Exception:
    TOK_EN_ES=MOD_EN_ES=TOK_ES_EN=MOD_ES_EN=None

try:
    TS_WINDOW = json.load(open(TS_DIR/"meta.json"))["window"]
    LSTM_BUNDLE = torch.load(TS_DIR/"lstm_hr.pt", map_location="cpu")
except Exception:
    TS_WINDOW = 24
    LSTM_BUNDLE = None
LSTM = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
HEAD = nn.Linear(32,1)
try:
    if LSTM_BUNDLE:
        LSTM.load_state_dict(LSTM_BUNDLE["lstm"])
        HEAD.load_state_dict(LSTM_BUNDLE["head"])
except Exception:
    pass
LSTM.eval(); HEAD.eval()

# ---- CNN Pneumonia model ----
class SmallCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64,2)
        )
CNN = SmallCNN()
try:
    CNN.load_state_dict(torch.load(CNN_DIR/"pneumonia_cnn.pt", map_location="cpu"))
    CNN.eval()
except Exception:
    CNN = None

# ---- Schemas ----
class RiskInput(BaseModel):
    age: int; bmi: float; systolic_bp: float; diastolic_bp: float
    heart_rate: float; cholesterol: float; sex: str="M"; smoker: int=0; diabetic: int=0

class LOSInput(BaseModel):
    age: int; bmi: float; systolic_bp: float; diastolic_bp: float
    heart_rate: float; cholesterol: float; sex: str="M"; smoker: int=0; diabetic: int=0

class SentimentInput(BaseModel):
    text: str

class TranslateInput(BaseModel):
    text: str; src_lang: str="en"; tgt_lang: str="es"

class HRForecastInput(BaseModel):
    series: list

# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status":"ok","version":"0.3.0"}

@app.post("/predict/risk")
def predict_risk(x: RiskInput):
    df = [{
        "age":x.age,"bmi":x.bmi,"systolic_bp":x.systolic_bp,"diastolic_bp":x.diastolic_bp,
        "heart_rate":x.heart_rate,"cholesterol":x.cholesterol,"sex":x.sex,
        "smoker":x.smoker,"diabetic":x.diabetic
    }]
    if PIPE is None or CLF is None:
        return {"risk_probability": 0.42}
    Xt = PIPE.transform(pd.DataFrame(df))
    p = float(CLF.predict_proba(Xt)[:,1][0])
    return {"risk_probability": p}

@app.post("/predict/los")
def predict_los(x: LOSInput):
    df = [{
        "age":x.age,"bmi":x.bmi,"systolic_bp":x.systolic_bp,"diastolic_bp":x.diastolic_bp,
        "heart_rate":x.heart_rate,"cholesterol":x.cholesterol,"sex":x.sex,
        "smoker":x.smoker,"diabetic":x.diabetic
    }]
    if PIPE is None or REG is None:
        return {"length_of_stay": 3.0}
    Xt = PIPE.transform(pd.DataFrame(df))
    y = float(REG.predict(Xt)[0])
    return {"length_of_stay": y}

@app.post("/analyze/sentiment")
def analyze_sentiment(x: SentimentInput):
    if TFIDF is None or SENT is None:
        return {"label": "neutral"}
    Xv = TFIDF.transform([x.text])
    y = SENT.predict(Xv)[0]
    return {"label": str(y)}

@app.post("/translate")
def translate(x: TranslateInput):
    if x.src_lang.lower()=="en" and x.tgt_lang.lower()=="es":
        tok, mod = TOK_EN_ES, MOD_EN_ES
    elif x.src_lang.lower()=="es" and x.tgt_lang.lower()=="en":
        tok, mod = TOK_ES_EN, MOD_ES_EN
    else:
        tok, mod = TOK_EN_ES, MOD_EN_ES
    if tok is None or mod is None:
        return {"translation": x.text}
    inputs = tok([x.text], return_tensors="pt", truncation=True)
    gen = mod.generate(**inputs, max_new_tokens=200)
    out = tok.batch_decode(gen, skip_special_tokens=True)[0]
    return {"translation": out}

@app.post("/forecast/hr")
def forecast_hr(x: HRForecastInput):
    seq = np.array(x.series, dtype=np.float32)[-TS_WINDOW:]
    if len(seq) < TS_WINDOW:
        pad = np.full(TS_WINDOW-len(seq), seq.mean() if len(seq)>0 else 72, dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    xb = torch.tensor(seq[None,:,None])
    with torch.no_grad():
        out,_ = LSTM(xb)
        pred = HEAD(out[:,-1,:]).squeeze().item()
    return {"next_hr": float(pred)}

@app.post("/classify/pneumonia")
async def classify_pneumonia(file: UploadFile = File(...)):
    if CNN is None:
        return {"label":"unknown","probability":0.5}
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L").resize((128,128))
    x = np.array(img, dtype=np.float32)/255.0
    xb = torch.tensor(x[None,None,:,:])
    with torch.no_grad():
        logits = CNN(xb)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(probs.argmax())
    labels = ["normal","pneumonia"]
    return {"label": labels[idx], "probability": float(probs[idx])}

@app.get("/artifacts/{fname}")
def get_artifact(fname: str):
    base = Path("artifacts/interpretability/tabular")
    f = base / fname
    if not f.exists():
        return {"error":"not found"}
    return FileResponse(str(f))


from fastapi.staticfiles import StaticFiles
app.mount('/artifacts', StaticFiles(directory='artifacts'), name='artifacts')
