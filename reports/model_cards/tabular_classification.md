# Tabular Classification (Readmission)

Generated: 2025-10-22 23:33 UTC

---
## Overview
GradientBoostingClassifier trained on synthetic patient features.

## Data
- Source: data/processed/patients_*.csv
- Features: artifacts/tabular/preprocess_meta.json

## Metrics (Validation)
- Accuracy: 0.9373333333333334
- F1: 0.0
- ROC AUC: 0.6868141318930819

## Interpretability
- SHAP summary: shap_summary_classification.png
- SHAP bar: shap_bar_classification.png
- LIME samples: lime_clf_0.html, lime_clf_1.html, lime_clf_2.html 

## Ethics & Limitations
Synthetic data only. No PHI. Model is illustrative and not for clinical use. Bias may exist from synthetic generation rules.

## Artifacts
- Model: artifacts/tabular/clf_gb.pkl
- Preprocess: artifacts/tabular/preprocess_pipeline.pkl
- Metrics: artifacts/tabular/metrics_tabular.json
