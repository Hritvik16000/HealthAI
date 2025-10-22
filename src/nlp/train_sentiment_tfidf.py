import json, pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from src.common.mlflow_utils import init_mlflow

def main():
    mlflow = init_mlflow()
    df = pd.read_csv("data/raw/patient_feedback.csv")
    Xtr, Xva, ytr, yva = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)
    Xtrv = tfidf.fit_transform(Xtr)
    Xvav = tfidf.transform(Xva)
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(Xtrv, ytr)
    ypred = clf.predict(Xvav)
    f1 = f1_score(yva, ypred, average="macro")
    acc = accuracy_score(yva, ypred)
    out = Path("artifacts/nlp"); out.mkdir(parents=True, exist_ok=True)
    with open(out/"sentiment_tfidf.pkl","wb") as f: pickle.dump({"vectorizer":tfidf,"model":clf}, f)
    with open(out/"metrics_sentiment.json","w") as f: json.dump({"val_f1":float(f1),"val_accuracy":float(acc)}, f, indent=2)
    mlflow.start_run(run_name="sentiment_tfidf")
    mlflow.log_metric("sentiment_val_f1", float(f1))
    mlflow.log_metric("sentiment_val_accuracy", float(acc))
    mlflow.log_artifact(str(out/"sentiment_tfidf.pkl"))
    mlflow.log_artifact(str(out/"metrics_sentiment.json"))
    mlflow.end_run()

if __name__=="__main__":
    main()
