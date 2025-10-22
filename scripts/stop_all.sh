#!/usr/bin/env bash
set -e
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
echo "Stopped API/UI."
