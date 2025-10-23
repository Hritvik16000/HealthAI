from pathlib import Path
import torch
from transformers import MarianTokenizer, MarianMTModel
from src.common.mlflow_utils import init_mlflow

def ensure_models():
    out = Path("artifacts/nlp"); out.mkdir(parents=True, exist_ok=True)
    for mid in ["Helsinki-NLP/opus-mt-en-es","Helsinki-NLP/opus-mt-es-en"]:
        MarianTokenizer.from_pretrained(mid).save_pretrained(out/mid.replace("/","__"))
        MarianMTModel.from_pretrained(mid).save_pretrained(out/mid.replace("/","__"))

if __name__=="__main__":
    init_mlflow()
    ensure_models()
