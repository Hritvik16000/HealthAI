import json, os
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def md_h1(t): return f"# {t}\n\n"
def md_h2(t): return f"## {t}\n\n"
def md_p(t): return f"{t}\n\n"
def md_li(items): return "".join([f"- {i}\n" for i in items]) + "\n"

def load_tabular_metrics():
    p = ROOT / "artifacts" / "tabular" / "metrics_tabular.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

def main():
    met_tab = load_tabular_metrics()
    clf = met_tab.get("clf", {})
    reg = met_tab.get("reg", {})
    inter_dir = "artifacts/interpretability/tabular"
    shap_cls = f"{inter_dir}/shap_summary_classification.png"
    shap_reg = f"{inter_dir}/shap_summary_regression.png"

    md = []
    md.append(md_h1("HealthAI – Final Report"))
    md.append(md_p(f"Generated: {now()}"))
    md.append(md_h2("1. Project Overview"))
    md.append(md_p("End-to-end synthetic healthcare ML system: tabular risk/LOS models, CNN imaging, LSTM forecasting, NLP sentiment + translation. Deployed via FastAPI + Streamlit, tracked with MLflow."))

    md.append(md_h2("2. Data & Preprocessing"))
    md.append(md_li([
        "Tabular: data/raw/patients.csv → train/valid/test splits, preprocessing pipeline (impute + scale + one-hot).",
        "Images: data/raw/images (synthetic normal vs pneumonia).",
        "Timeseries: data/raw/vitals_timeseries.csv (HR, SBP, SpO2).",
        "Text: patient_feedback.csv and doctor_notes.csv."
    ]))

    md.append(md_h2("3. Models & Metrics"))
    md.append(md_p("**Tabular Classification (Readmission)**"))
    md.append(md_li([
        f"Accuracy: {clf.get('val_accuracy','N/A')}",
        f"F1: {clf.get('val_f1','N/A')}",
        f"ROC AUC: {clf.get('val_roc_auc','N/A')}",
    ]))
    md.append(md_p("**Tabular Regression (Length of Stay)**"))
    md.append(md_li([
        f"RMSE: {reg.get('val_rmse','N/A')}",
        f"R²: {reg.get('val_r2','N/A')}",
    ]))
    md.append(md_p("CNN Pneumonia: see reports/model_cards/cnn_pneumonia.md"))
    md.append(md_p("LSTM HR Forecasting: see reports/model_cards/timeseries_lstm.md"))
    md.append(md_p("NLP Sentiment: see reports/model_cards/nlp_sentiment.md"))

    md.append(md_h2("4. Interpretability & Ethics"))
    md.append(md_p("SHAP and LIME explanations for tabular models."))
    if Path(shap_cls).exists():
        md.append(f"![SHAP Classification]({shap_cls})\n\n")
    if Path(shap_reg).exists():
        md.append(f"![SHAP Regression]({shap_reg})\n\n")
    md.append(md_p("Ethical considerations: synthetic-only data, non-clinical use, potential biases from generation rules, transparency via model cards."))

    md.append(md_h2("5. Deployment & Usage"))
    md.append(md_li([
        "API: uvicorn apps.api.main:app --host 0.0.0.0 --port 8000",
        "UI: streamlit run apps/ui/app.py --server.port 8501",
        "Docker: docker-compose up --build",
    ]))

    md.append(md_h2("6. Reproducibility"))
    md.append(md_li([
        "Experiment tracking: MLflow stored under mlruns/",
        "Artifacts: artifacts/",
        "Data generators: src/data/",
        "Training: scripts/train_all.sh",
        "Docs: reports/model_cards/, reports/final/FINAL_REPORT.md",
    ]))

    out_md = OUT / "FINAL_REPORT.md"
    out_md.write_text("".join(md))
    print("Wrote", out_md)

if __name__ == "__main__":
    main()
