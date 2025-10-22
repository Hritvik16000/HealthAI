from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.common.mlflow_utils import init_mlflow


def run():
    mlflow = init_mlflow()
    model_id = "emilyalsentzer/Bio_ClinicalBERT"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    clf = pipeline("text-classification", model=mdl, tokenizer=tok)
    r = clf("Patient reports mild chest pain.")
    mlflow.start_run(run_name="clinical_nlp")
    mlflow.log_param("model", model_id)
    mlflow.log_metric("dummy_score", float(r[0]["score"]))
    mlflow.end_run()


if __name__ == "__main__":
    run()
