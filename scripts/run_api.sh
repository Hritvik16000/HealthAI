#!/usr/bin/env bash
set -e
source .venv/bin/activate
uvicorn apps.api.main:app --host 0.0.0.0 --port ${API_PORT:-8000} --reload
