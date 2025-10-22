#!/usr/bin/env bash
set -e
source .venv/bin/activate
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
nohup uvicorn apps.api.main:app --host 0.0.0.0 --port ${API_PORT:-8000} >/tmp/healthai_api.log 2>&1 &
nohup streamlit run apps/ui/app.py --server.port ${UI_PORT:-8501} --server.address 0.0.0.0 >/tmp/healthai_ui.log 2>&1 &
sleep 2
echo "API:    http://localhost:${API_PORT:-8000}"
echo "UI:     http://localhost:${UI_PORT:-8501}"
