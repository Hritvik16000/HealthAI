.PHONY: setup lint test train-tabular train-sentiment run-api run-dashboard docker-api docker-app

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
\tpython -c "import nltk; nltk.download('punkt', quiet=True)"
\tpython - <<PY
import spacy, sys
try:
    spacy.load('en_core_web_sm')
except Exception:
    import subprocess
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
PY

lint:
\t@echo "Add ruff/black if desired (pyproject already configured)."

test:
\tpytest -q

train-tabular:
\tpython -m scripts.train_tabular --config configs/tabular.yaml

train-sentiment:
\tpython -m scripts.train_sentiment --config configs/nlp.yaml

run-api:
\tuvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
\tstreamlit run apps/dashboard/Home.py
