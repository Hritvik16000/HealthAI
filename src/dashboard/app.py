from pathlib import Path
import sys

# --- ensure project root is importable ---
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import json
import joblib
import pandas as pd
import streamlit as st

from src.utils.sentiment_engine import HybridSentimentEngine

TAB_DIR = BASE_DIR / "artifacts" / "tabular"
CLUSTER_DIR = BASE_DIR / "artifacts" / "cluster"
ASSOC_DIR = BASE_DIR / "artifacts" / "association"
DATA_DIR = BASE_DIR / "data" / "processed"

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

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")
st.title("HealthAI Suite Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Prediction",
    "Length of Stay",
    "Patient Clustering",
    "Association Rules",
    "Sentiment Analysis"
])

with st.sidebar:
    st.header("Patient Input")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
    blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=250.0, value=130.0)
    glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=110.0)
    cholesterol = st.number_input("Cholesterol", min_value=50.0, max_value=500.0, value=200.0)
    heart_rate = st.number_input("Heart Rate", min_value=30.0, max_value=220.0, value=80.0)
    smoker = st.selectbox("Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    diabetes_history = st.selectbox("Diabetes History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

patient_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "glucose": glucose,
    "cholesterol": cholesterol,
    "heart_rate": heart_rate,
    "smoker": smoker,
    "diabetes_history": diabetes_history
}])

with tab1:
    if st.button("Predict Risk"):
        pred = risk_model.predict(patient_df)[0]
        label = risk_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Risk Category: {label}")

with tab2:
    if st.button("Predict Length of Stay"):
        pred = los_model.predict(patient_df)[0]
        st.info(f"Predicted Length of Stay: {pred:.2f} days")

with tab3:
    if st.button("Assign Cluster"):
        cluster = int(cluster_model.predict(patient_df)[0])
        cluster_name = cluster_names.get(str(cluster), cluster_names.get(cluster, f"Cluster {cluster}"))
        st.warning(f"Cluster ID: {cluster}")
        st.success(f"Cluster Meaning: {cluster_name}")

with tab4:
    rules_path = ASSOC_DIR / "association_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path)
        st.dataframe(rules_df, use_container_width=True)

with tab5:
    review = st.text_area("Enter patient review")
    if st.button("Analyze Sentiment"):
        result = sentiment_engine.predict(review)
        st.success(f"Sentiment: {result['label']}")

st.markdown("---")

patients_path = DATA_DIR / "patients_clean.csv"
if patients_path.exists():
    df = pd.read_csv(patients_path)
    st.dataframe(df.head(), use_container_width=True)
