import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from src.common.mlflow_utils import init_mlflow

def load_arrays():
    p = Path("artifacts/tabular")
    Xtr = np.load(p/"X_train.npy", allow_pickle=True)
    Xva = np.load(p/"X_valid.npy", allow_pickle=True)
    Xte = np.load(p/"X_test.npy", allow_pickle=True)
    ytr_c = np.load(p/"y_train_clf.npy", allow_pickle=True)
    yva_c = np.load(p/"y_valid_clf.npy", allow_pickle=True)
    yte_c = np.load(p/"y_test_clf.npy", allow_pickle=True)
    ytr_r = np.load(p/"y_train_reg.npy", allow_pickle=True)
    yva_r = np.load(p/"y_valid_reg.npy", allow_pickle=True)
    yte_r = np.load(p/"y_test_reg.npy", allow_pickle=True)
    return (Xtr,Xva,Xte,ytr_c,yva_c,yte_c,ytr_r,yva_r,yte_r)

def main():
    mlflow = init_mlflow()
    Xtr,Xva,Xte,ytr_c,yva_c,yte_c,ytr_r,yva_r,yte_r = load_arrays()

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr_c)
    pva = clf.predict_proba(Xva)[:,1]
    yva = (pva>=0.5).astype(int)
    metrics_clf = {
        "val_accuracy": float(accuracy_score(yva_c, yva)),
        "val_f1": float(f1_score(yva_c, yva)),
        "val_roc_auc": float(roc_auc_score(yva_c, pva)),
    }

    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(Xtr, ytr_r)
    yhat = reg.predict(Xva)
    metrics_reg = {
        "val_rmse": float(mean_squared_error(yva_r, yhat, squared=False)),
        "val_r2": float(r2_score(yva_r, yhat)),
    }

    out_dir = Path("artifacts/tabular")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"clf_gb.pkl","wb") as f: pickle.dump(clf,f)
    with open(out_dir/"reg_gb.pkl","wb") as f: pickle.dump(reg,f)
    with open(out_dir/"metrics_tabular.json","w") as f: json.dump({"clf":metrics_clf,"reg":metrics_reg}, f, indent=2)

    mlflow.start_run(run_name="tabular_models")
    mlflow.log_metrics(metrics_clf | metrics_reg)
    mlflow.log_artifact(str(out_dir/"clf_gb.pkl"))
    mlflow.log_artifact(str(out_dir/"reg_gb.pkl"))
    mlflow.log_artifact(str(out_dir/"metrics_tabular.json"))
    mlflow.end_run()

if __name__=="__main__":
    main()
