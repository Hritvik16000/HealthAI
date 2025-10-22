# Submission Checklist

- ✅ Codebase (apps/, src/, configs/, scripts/)
- ✅ Requirements and environment (.env, requirements.txt)
- ✅ Trained model artifacts (artifacts/tabular/*.pkl, artifacts/cnn/metrics_cnn.json, artifacts/timeseries/meta.json, artifacts/nlp/metrics_sentiment.json)
- ✅ Interpretability artifacts (artifacts/interpretability/)
- ✅ Documentation
  - reports/model_cards/*.md
  - reports/final/FINAL_REPORT.md
  - reports/final/FINAL_REPORT.pdf
  - reports/slides/FINAL_PRESENTATION.pptx
  - reports/demo/*.json (after running scripts/demo_run.sh)
- ✅ Reproducibility
  - MLflow runs (local mlruns/ kept locally; not required in archive)
  - scripts/generate_data.sh
  - scripts/train_all.sh
  - scripts/serve_all.sh / scripts/stop_all.sh
- ✅ Demo Video (record screen: open UI, run each tab once, show SHAP/LIME images, briefly show MLflow UI)

