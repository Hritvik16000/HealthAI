# Tabular Regression (Length of Stay)

Generated: 2025-10-22 23:33 UTC

---
## Overview
GradientBoostingRegressor trained on synthetic patient features.

## Data
- Source: data/processed/patients_*.csv
- Features: artifacts/tabular/preprocess_meta.json

## Metrics (Validation)
- RMSE: 1.1144895807049833
- R²: 0.43019144663930864

## Interpretability
- SHAP summary: shap_summary_regression.png
- SHAP bar: shap_bar_regression.png
- LIME samples: lime_reg_0.html, lime_reg_1.html, lime_reg_2.html 

## Ethics & Limitations
Synthetic data only. Not for clinical decision-making.

## Artifacts
- Model: artifacts/tabular/reg_gb.pkl
- Preprocess: artifacts/tabular/preprocess_pipeline.pkl
- Metrics: artifacts/tabular/metrics_tabular.json
