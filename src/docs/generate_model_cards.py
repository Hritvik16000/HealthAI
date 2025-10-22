import json, os, pickle
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
ART = ROOT / "artifacts"
MC = ROOT / "reports" / "model_cards"
MC.mkdir(parents=True, exist_ok=True)

def write_card(name, content):
    (MC / f"{name}.md").write_text(content)

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def card_header(title):
    return f"# {title}\n\nGenerated: {now()}\n\n---\n"

def tabular_cards():
    tab = ART / "tabular"
    inter = ART / "interpretability" / "tabular"
    met = {}
    if (tab / "metrics_tabular.json").exists():
        met = json.loads((tab / "metrics_tabular.json").read_text())
    inter_meta = {}
    if (inter / "meta.json").exists():
        inter_meta = json.loads((inter / "meta.json").read_text())
    clf = met.get("clf", {})
    reg = met.get("reg", {})
    clf_md = card_header("Tabular Classification (Readmission)") + \
f"""## Overview
GradientBoostingClassifier trained on synthetic patient features.

## Data
- Source: data/processed/patients_*.csv
- Features: artifacts/tabular/preprocess_meta.json

## Metrics (Validation)
- Accuracy: {clf.get('val_accuracy','N/A')}
- F1: {clf.get('val_f1','N/A')}
- ROC AUC: {clf.get('val_roc_auc','N/A')}

## Interpretability
- SHAP summary: {inter_meta.get('classification',{}).get('shap_summary','(not generated)')}
- SHAP bar: {inter_meta.get('classification',{}).get('shap_bar','(not generated)')}
- LIME samples: {', '.join(inter_meta.get('classification',{}).get('lime_html_samples', [])) or '(not generated)'} 

## Ethics & Limitations
Synthetic data only. No PHI. Model is illustrative and not for clinical use. Bias may exist from synthetic generation rules.

## Artifacts
- Model: artifacts/tabular/clf_gb.pkl
- Preprocess: artifacts/tabular/preprocess_pipeline.pkl
- Metrics: artifacts/tabular/metrics_tabular.json
"""
    reg_md = card_header("Tabular Regression (Length of Stay)") + \
f"""## Overview
GradientBoostingRegressor trained on synthetic patient features.

## Data
- Source: data/processed/patients_*.csv
- Features: artifacts/tabular/preprocess_meta.json

## Metrics (Validation)
- RMSE: {reg.get('val_rmse','N/A')}
- R²: {reg.get('val_r2','N/A')}

## Interpretability
- SHAP summary: {inter_meta.get('regression',{}).get('shap_summary','(not generated)')}
- SHAP bar: {inter_meta.get('regression',{}).get('shap_bar','(not generated)')}
- LIME samples: {', '.join(inter_meta.get('regression',{}).get('lime_html_samples', [])) or '(not generated)'} 

## Ethics & Limitations
Synthetic data only. Not for clinical decision-making.

## Artifacts
- Model: artifacts/tabular/reg_gb.pkl
- Preprocess: artifacts/tabular/preprocess_pipeline.pkl
- Metrics: artifacts/tabular/metrics_tabular.json
"""
    write_card("tabular_classification", clf_md)
    write_card("tabular_regression", reg_md)

def cnn_card():
    p = ART / "cnn"
    met = {}
    if (p / "metrics_cnn.json").exists():
        met = json.loads((p / "metrics_cnn.json").read_text())
    md = card_header("CNN – Pneumonia (Synthetic)") + \
f"""## Overview
Simple CNN trained on synthetic grayscale images with simulated lesions.

## Metrics (Validation)
- Accuracy: {met.get('val_accuracy','N/A')}

## Data
- Images: data/raw/images/ (normal vs pneumonia)
- Labels: data/raw/image_labels.csv

## Ethics & Limitations
Synthetic task; not representative of medical imaging complexity.

## Artifacts
- Model: artifacts/cnn/pneumonia_cnn.pt
- Metrics: artifacts/cnn/metrics_cnn.json
"""
    write_card("cnn_pneumonia", md)

def ts_card():
    p = ART / "timeseries"
    window = "(unknown)"
    if (p / "meta.json").exists():
        window = json.loads((p / "meta.json").read_text()).get("window","(unknown)")
    md = card_header("LSTM – Heart Rate Forecasting") + \
f"""## Overview
Univariate LSTM to forecast next-step heart rate from synthetic vitals.

## Config
- Window: {window}

## Data
- Source: data/raw/vitals_timeseries.csv

## Ethics & Limitations
Synthetic series; only for demonstration.

## Artifacts
- Weights: artifacts/timeseries/lstm_hr.pt
- Meta: artifacts/timeseries/meta.json
"""
    write_card("timeseries_lstm", md)

def nlp_cards():
    p = ART / "nlp"
    met = {}
    if (p / "metrics_sentiment.json").exists():
        met = json.loads((p / "metrics_sentiment.json").read_text())
    md = card_header("NLP – Patient Feedback Sentiment (TF-IDF + LR)") + \
f"""## Overview
Classic TF-IDF + Logistic Regression on synthetic patient feedback.

## Metrics (Validation)
- F1 (macro): {met.get('val_f1','N/A')}
- Accuracy: {met.get('val_accuracy','N/A')}

## Data
- Source: data/raw/patient_feedback.csv

## Ethics & Limitations
Synthetic sentences; vocabulary limited.

## Artifacts
- Model bundle: artifacts/nlp/sentiment_tfidf.pkl
- Metrics: artifacts/nlp/metrics_sentiment.json
"""
    write_card("nlp_sentiment", md)

def main():
    tabular_cards()
    cnn_card()
    ts_card()
    nlp_cards()
    print("Model cards written to reports/model_cards")

if __name__ == "__main__":
    main()
