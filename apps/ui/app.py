import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="HealthAI", layout="wide")
api = f"http://localhost:{os.getenv('API_PORT','8000')}"
tabs = st.tabs(["Dashboard", "Risk", "Sentiment", "Translate", "System"])
with tabs[0]:
    st.header("HealthAI Dashboard")
    c1, c2, c3 = st.columns(3)
    with c1:
        h = requests.get(f"{api}/health").json()
        st.metric("API", h.get("status", "down"))
    with c2:
        st.metric("Models", "Stub")
    with c3:
        st.metric("Tracking", "mlruns")
with tabs[1]:
    st.subheader("Risk Prediction")
    age = st.number_input("Age", 0, 120, 45)
    bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
    sbp = st.number_input("Systolic BP", 60.0, 220.0, 122.0)
    dbp = st.number_input("Diastolic BP", 40.0, 140.0, 80.0)
    hr = st.number_input("Heart Rate", 30.0, 220.0, 72.0)
    if st.button("Predict Risk"):
        r = requests.post(
            f"{api}/predict/risk",
            json={
                "age": age,
                "bmi": bmi,
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "heart_rate": hr,
            },
        ).json()
        st.json(r)
with tabs[2]:
    st.subheader("Patient Feedback Sentiment")
    txt = st.text_area("Text")
    if st.button("Analyze"):
        r = requests.post(f"{api}/analyze/sentiment", json={"text": txt}).json()
        st.json(r)
with tabs[3]:
    st.subheader("Medical Translator")
    t = st.text_area("Input")
    c1, c2 = st.columns(2)
    with c1:
        s = st.text_input("Source", "en")
    with c2:
        g = st.text_input("Target", "es")
    if st.button("Translate"):
        r = requests.post(f"{api}/translate", json={"text": t, "src_lang": s, "tgt_lang": g}).json()
        st.json(r)
with tabs[4]:
    st.subheader("System")
    st.code(f"API {api}")
