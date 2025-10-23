# HealthAI
Setup
1. source .venv/bin/activate
2. uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
3. streamlit run apps/ui/app.py --server.port 8501
4. jupyter lab
Training
python -m src.classical_ml.train_tasks
python -m src.cnn.train_cnn
python -m src.time_series.train_lstm
python -m src.nlp.train_clinical_nlp
Artifacts
mlruns for tracking, artifacts for outputs, reports for documentation
https://healthai-hxhickyvbxzztpnxydathv.streamlit.app - Access the App using this link.
# 🩺 HealthAI – End-to-End Healthcare Analytics Platform

**HealthAI** is an integrated AI-driven healthcare system that unifies classical machine learning, natural language processing, deep learning, and predictive analytics within an interactive web interface.

Built entirely from scratch and deployed across cloud services, HealthAI demonstrates full-stack MLOps — from data generation and model training to explainability and web deployment.

---

## 🚀 Live Demo

🌐 **Streamlit UI:** [https://your-streamlit-subdomain.streamlit.app](https://your-streamlit-subdomain.streamlit.app)  
⚙️ **FastAPI Backend:** [https://your-render-service.onrender.com](https://your-render-service.onrender.com)

---

## 🧠 Key Features

### 🧩 Predictive Models
- **Risk Prediction:** ML classification on patient vitals and lifestyle data.
- **Length of Stay Forecast:** Regression models predicting hospitalization duration.

### 💬 Natural Language Processing
- **Patient Sentiment Analysis:** Classifies clinical text as *positive* or *negative*.
- **Medical Translator:** English ↔ Spanish translation via MarianMT / HF API.

### 🫁 Medical Imaging
- **Pneumonia Classifier:** CNN model trained on chest X-rays for pneumonia detection.

### ⏱️ Time-Series Forecasting
- **Heart Rate Prediction:** LSTM-based model for vital signal forecasting.

### 🔍 Explainable AI
- **SHAP Interpretability Reports** for tabular ML models, enabling transparent decision-making.

---

## 🏗️ Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend/UI** | Streamlit, Plotly |
| **Backend/API** | FastAPI, Uvicorn |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch, TensorFlow |
| **NLP** | Hugging Face Transformers (MarianMT, TF-IDF) |
| **Forecasting** | LSTM (Keras) |
| **Explainability** | SHAP, LIME |
| **Data Handling** | Pandas, NumPy, MLflow |
| **Deployment** | Render (API), Streamlit Cloud (UI) |

---

## 🧪 Local Development

```bash
git clone https://github.com/Hritvik16000/HealthAI.git
cd HealthAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run locally:
# Start backend
uvicorn apps.api.main:app --reload --port 8000

# In a new terminal
streamlit run apps/ui/app.py
Then visit:
👉 http://localhost:8501 (UI)
👉 http://localhost:8000/docs (API Docs)
☁️ Cloud Deployment
FastAPI on Render
Build Command: pip install -r requirements.txt
Start Command: uvicorn apps.api.main:app --host 0.0.0.0 --port 10000
Add env var:
CORS_ORIGINS = https://your-streamlit-subdomain.streamlit.app
