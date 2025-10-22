#!/usr/bin/env bash
set -e
NAME="HealthAI_submission_$(date +%Y%m%d_%H%M%S)"
OUTDIR="/tmp/$NAME"
mkdir -p "$OUTDIR"

cp -R \
  apps \
  src \
  configs \
  scripts \
  reports \
  artifacts/interpretability \
  artifacts/tabular/*.pkl \
  artifacts/tabular/metrics_tabular.json \
  artifacts/cnn/metrics_cnn.json \
  artifacts/timeseries/meta.json \
  artifacts/nlp/metrics_sentiment.json \
  README.md \
  .env \
  docker-compose.yml \
  Dockerfile.api \
  Dockerfile.ui \
  requirements.txt \
  "$OUTDIR"

find "$OUTDIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUTDIR" -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
rm -rf "$OUTDIR"/scripts/*.log 2>/dev/null || true

cd "$(dirname "$OUTDIR")"
tar -czf "$NAME.tar.gz" "$NAME"
echo "$(pwd)/$NAME.tar.gz"
