import os

import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.common.metrics import compute_classification_metrics


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    base = load_cfg(cfg.get("defaults", "configs/defaults.yaml"))
    os.makedirs(base["paths"]["mlflow_uri"], exist_ok=True)
    mlflow.set_tracking_uri(base["paths"]["mlflow_uri"])
    mlflow.set_experiment(f"tabular-{cfg['task']}")

    df = pd.read_csv(cfg["dataset"]["csv"])
    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"]])

    num = cfg["features"]["numeric"]
    cat = cfg["features"]["categorical"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )

    if cfg["task"] == "classification":
        model = XGBClassifier(**cfg["model"]["params"])
    else:
        model = XGBRegressor(**cfg["model"]["params"])

    pipe = Pipeline([("pre", pre), ("model", model)])
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if cfg["task"] == "classification" else None,
    )

    with mlflow.start_run():
        pipe.fit(Xtr, ytr)
        if cfg["task"] == "classification":
            ypred = pipe.predict(Xte)
            yprob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe[-1], "predict_proba") else None
            metrics = compute_classification_metrics(yte, ypred, yprob)
        else:
            ypred = pipe.predict(Xte)
            mae = mean_absolute_error(yte, ypred)
            metrics = {"mae": mae}

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print("Metrics:", metrics)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
