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
