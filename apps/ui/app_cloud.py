import json, io, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="HealthAI (Cloud)", layout="wide")

import os, requests
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN",""))

def translate_hf(text: str, src: str, tgt: str) -> str:
    if not text.strip():
        return ""
    if not HF_API_TOKEN:
        return text
    model = f"Helsinki-NLP/opus-mt-{src.lower()}-{tgt.lower()}"
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        r = requests.post(url, headers=headers, json={"inputs": text}, timeout=20)
        if r.status_code == 200:
            out = r.json()
            if isinstance(out, list) and out and "translation_text" in out[0]:
                return out[0]["translation_text"]
        return text
    except Exception:
        return text


TAB_DIR = Path("artifacts/tabular")
NLP_DIR = Path("artifacts/nlp")

PIPE = None; CLF = None; REG = None; TFIDF=None; SENT=None

try:
    with open(TAB_DIR/"preprocess_pipeline.pkl","rb") as f: PIPE = pickle.load(f)
    with open(TAB_DIR/"clf_gb.pkl","rb") as f: CLF = pickle.load(f)
    with open(TAB_DIR/"reg_gb.pkl","rb") as f: REG = pickle.load(f)
except Exception:
    pass

try:
    with open(NLP_DIR/"sentiment_tfidf.pkl","rb") as f:
        s = pickle.load(f)
        TFIDF=s["vectorizer"]; SENT=s["model"]
except Exception:
    pass

dashboard_tab, risk_tab, los_tab, sentiment_tab, translate_tab, forecast_tab, system_tab = st.tabs(
    ["Dashboard","Risk","LOS","Sentiment","Translate","Forecast","System"]
)

with dashboard_tab:
    st.header("HealthAI Dashboard")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Tabular Models", "OK" if (PIPE and CLF and REG) else "Fallback")
    with c2: st.metric("NLP Sentiment", "OK" if (TFIDF and SENT) else "Fallback")
    with c3: st.metric("Translator", "Echo")

with risk_tab:
    st.subheader("Risk Prediction (Cloud, local inference)")
    c1,c2,c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 0, 120, 45)
        sex = st.selectbox("Sex", ["M","F"])
        smoker = st.selectbox("Smoker", [0,1])
        diabetic = st.selectbox("Diabetic", [0,1])
    with c2:
        bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
        sbp = st.number_input("Systolic BP", 60.0, 220.0, 122.0)
        dbp = st.number_input("Diastolic BP", 40.0, 140.0, 80.0)
    with c3:
        hr = st.number_input("Heart Rate", 30.0, 220.0, 72.0)
        chol = st.number_input("Cholesterol", 90.0, 400.0, 180.0)
    if st.button("Predict Risk"):
        df = pd.DataFrame([{
            "age":age,"bmi":bmi,"systolic_bp":sbp,"diastolic_bp":dbp,
            "heart_rate":hr,"cholesterol":chol,"sex":sex,"smoker":smoker,"diabetic":diabetic
        }])
        if PIPE and CLF:
            Xt = PIPE.transform(df)
            p = float(CLF.predict_proba(Xt)[:,1][0])
        else:
            p = 0.42
        st.json({"risk_probability": p})

with los_tab:
    st.subheader("Length of Stay (Cloud, local inference)")
    c1,c2,c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (LOS)", 0, 120, 45)
        sex = st.selectbox("Sex (LOS)", ["M","F"])
        smoker = st.selectbox("Smoker (LOS)", [0,1])
        diabetic = st.selectbox("Diabetic (LOS)", [0,1])
    with c2:
        bmi = st.number_input("BMI (LOS)", 10.0, 60.0, 26.0)
        sbp = st.number_input("Systolic BP (LOS)", 60.0, 220.0, 122.0)
        dbp = st.number_input("Diastolic BP (LOS)", 40.0, 140.0, 80.0)
    with c3:
        hr = st.number_input("Heart Rate (LOS)", 30.0, 220.0, 72.0)
        chol = st.number_input("Cholesterol (LOS)", 90.0, 400.0, 180.0)
    if st.button("Predict LOS"):
        df = pd.DataFrame([{
            "age":age,"bmi":bmi,"systolic_bp":sbp,"diastolic_bp":dbp,
            "heart_rate":hr,"cholesterol":chol,"sex":sex,"smoker":smoker,"diabetic":diabetic
        }])
        if PIPE and REG:
            Xt = PIPE.transform(df)
            y = float(REG.predict(Xt)[0])
        else:
            y = 3.0
        st.json({"length_of_stay": y})

with sentiment_tab:
    st.subheader("Patient Feedback Sentiment (Cloud, local inference)")
    txt = st.text_area("Text")
    if st.button("Analyze Sentiment"):
        if TFIDF and SENT:
            Xv = TFIDF.transform([txt])
            y = SENT.predict(Xv)[0]
            st.json({"label": str(y)})
        else:
            st.json({"label": "neutral"})

with translate_tab:
    st.subheader("Medical Translator (HF Inference API)")
    t = st.text_area("Input")
    s1,s2 = st.columns(2)
    with s1: src = st.text_input("Source","en")
    with s2: tgt = st.text_input("Target","es")
    if st.button("Translate"):
        out = translate_hf(t, src, tgt)
        st.json({"translation": out})

with forecast_tab:
    st.subheader("Forecast HR (Cloud, moving-average fallback)")
    seq = st.text_area("Comma-separated HR history", value="72,74,73,75,76,74")
    if st.button("Forecast Next HR"):
        try:
            arr = [float(v.strip()) for v in seq.split(",") if v.strip()]
        except Exception:
            arr = [72,74,73,75,76,74]
        if len(arr)==0:
            pred = 72.0
        else:
            k = min(5, len(arr))
            pred = float(np.mean(arr[-k:]))
        st.json({"next_hr": pred})

with system_tab:
    st.subheader("System")
    st.write("Running in Cloud Mode (no local API).")
    st.write("Artifacts expected under artifacts/tabular and artifacts/nlp.")
