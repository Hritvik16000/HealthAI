import os, json, io
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="HealthAI", layout="wide")
API_BASE = f"http://localhost:{os.getenv('API_PORT','8000')}"

def _safe_get(url, payload=None, files=None):
    try:
        if files is not None:
            return requests.post(url, files=files, timeout=6).json()
        if payload is None:
            return requests.get(url, timeout=2).json()
        return requests.post(url, json=payload, timeout=6).json()
    except Exception:
        return {"status":"down"}

dashboard_tab, risk_tab, los_tab, sentiment_tab, translate_tab, forecast_tab, imaging_tab, explain_tab, system_tab = st.tabs(
    ["Dashboard","Risk","LOS","Sentiment","Translate","Forecast","Imaging","Explain","System"]
)

with dashboard_tab:
    st.header("HealthAI Dashboard")
    c1,c2,c3 = st.columns(3)
    with c1:
        h = _safe_get(f"{API_BASE}/health")
        st.metric("API", h.get("status","down"))
    with c2:
        st.metric("Models", "Ready")
    with c3:
        st.metric("Tracking", "mlruns")

with risk_tab:
    st.subheader("Risk Prediction")
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
        r = _safe_get(f"{API_BASE}/predict/risk", {
            "age":age,"bmi":bmi,"systolic_bp":sbp,"diastolic_bp":dbp,
            "heart_rate":hr,"cholesterol":chol,"sex":sex,"smoker":smoker,"diabetic":diabetic
        })
        st.json(r)

with los_tab:
    st.subheader("Length of Stay")
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
        r = _safe_get(f"{API_BASE}/predict/los", {
            "age":age,"bmi":bmi,"systolic_bp":sbp,"diastolic_bp":dbp,
            "heart_rate":hr,"cholesterol":chol,"sex":sex,"smoker":smoker,"diabetic":diabetic
        })
        st.json(r)

with sentiment_tab:
    st.subheader("Patient Feedback Sentiment")
    txt = st.text_area("Text")
    if st.button("Analyze Sentiment"):
        r = _safe_get(f"{API_BASE}/analyze/sentiment", {"text":txt})
        st.json(r)

with translate_tab:
    st.subheader("Medical Translator")
    t = st.text_area("Input")
    c1,c2 = st.columns(2)
    with c1:
        s = st.text_input("Source","en")
    with c2:
        g = st.text_input("Target","es")
    if st.button("Translate"):
        r = _safe_get(f"{API_BASE}/translate", {"text":t,"src_lang":s,"tgt_lang":g})
        st.json(r)

with forecast_tab:
    st.subheader("Forecast HR")
    seq = st.text_area("Comma-separated HR history", value="72,74,73,75,76,74")
    if st.button("Forecast Next HR"):
        try:
            arr = [float(v.strip()) for v in seq.split(",") if v.strip()]
        except Exception:
            arr = [72,74,73,75,76,74]
        r = _safe_get(f"{API_BASE}/forecast/hr", {"series":arr})
        st.json(r)

with imaging_tab:
    st.subheader("Pneumonia Imaging Classifier")
    up = st.file_uploader("Upload chest X-ray (png/jpg)", type=["png","jpg","jpeg"])
    if up and st.button("Classify Image"):
        img = Image.open(up).convert("L")
        bio = io.BytesIO()
        img.save(bio, format="PNG"); bio.seek(0)
        files = {"file": ("upload.png", bio.getvalue(), "image/png")}
        r = _safe_get(f"{API_BASE}/classify/pneumonia", files=files)
        c1,c2 = st.columns([1,1])
        with c1: st.image(img, caption="Input", use_container_width=True)
        with c2: st.json(r)

with explain_tab:
    st.subheader("Model Interpretability (Tabular)")
    import pathlib
    st.image(f"{API_BASE}/artifacts/shap_summary_classification.png", caption="SHAP Summary – Classification", use_container_width=True)
    st.image(f"{API_BASE}/artifacts/shap_summary_regression.png", caption="SHAP Summary – Regression", use_container_width=True)

with system_tab:
    st.subheader("System")
    st.code(f"API {API_BASE}")
