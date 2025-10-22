#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.classical_ml.train_tabular_models
python -m src.cnn.train_pneumonia_cnn
python -m src.time_series.train_lstm_forecaster
python -m src.nlp.train_sentiment_tfidf
python -m src.nlp.load_marianmt
