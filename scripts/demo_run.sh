#!/usr/bin/env bash
set -e
API="http://localhost:${API_PORT:-8000}"
OUT="reports/demo"
mkdir -p "$OUT"

curl -s "$API/health" | tee "$OUT/00_health.json" >/dev/null

curl -s -X POST "$API/predict/risk" \
  -H "Content-Type: application/json" \
  -d '{"age":45,"bmi":27.2,"systolic_bp":122,"diastolic_bp":78,"heart_rate":72,"cholesterol":180,"sex":"M","smoker":0,"diabetic":0}' \
  | tee "$OUT/01_risk.json" >/dev/null

curl -s -X POST "$API/predict/los" \
  -H "Content-Type: application/json" \
  -d '{"age":62,"bmi":31.5,"systolic_bp":138,"diastolic_bp":86,"heart_rate":88,"cholesterol":210,"sex":"F","smoker":1,"diabetic":1}' \
  | tee "$OUT/02_los.json" >/dev/null

curl -s -X POST "$API/analyze/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text":"Nurse was attentive and kind, very satisfied with the care."}' \
  | tee "$OUT/03_sentiment.json" >/dev/null

curl -s -X POST "$API/translate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Please take your medication twice daily after meals.","src_lang":"en","tgt_lang":"es"}' \
  | tee "$OUT/04_translate.json" >/dev/null

curl -s -X POST "$API/forecast/hr" \
  -H "Content-Type: application/json" \
  -d '{"series":[72,74,73,75,76,74,73,75,77,76,74,73,74,75,76,77,78,77,76,75,74,73,72,73]}' \
  | tee "$OUT/05_forecast.json" >/dev/null

echo "Demo outputs saved to $OUT"
