#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.data.generate_tabular
python -m src.data.preprocess_tabular
python -m src.data.generate_timeseries
python -m src.data.generate_images
python -m src.data.generate_text
