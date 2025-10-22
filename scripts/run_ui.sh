#!/usr/bin/env bash
set -e
source .venv/bin/activate
streamlit run apps/ui/app.py --server.port ${UI_PORT:-8501} --server.address 0.0.0.0
