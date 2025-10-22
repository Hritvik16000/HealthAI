import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.common.mlflow_utils import init_mlflow

def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return ct

def main():
    mlflow = init_mlflow()
    proc_dir = Path("data/processed")
    train_csv = proc_dir / "patients_train.csv"
    valid_csv = proc_dir / "patients_valid.csv"
    test_csv  = proc_dir / "patients_test.csv"

    train = pd.read_csv(train_csv)
    valid = pd.read_csv(valid_csv)
    test  = pd.read_csv(test_csv)

    target_clf = "readmit_30d"
    target_reg = "length_of_stay"

    num_cols = ["age","bmi","systolic_bp","diastolic_bp","heart_rate","cholesterol"]
    cat_cols = ["sex","smoker","diabetic"]

    pipe = build_pipeline(num_cols, cat_cols)

    X_train = train[num_cols + cat_cols]
    X_valid = valid[num_cols + cat_cols]
    X_test  = test[num_cols + cat_cols]

    Xt_train = pipe.fit_transform(X_train)
    Xt_valid = pipe.transform(X_valid)
    Xt_test  = pipe.transform(X_test)

    y_train_clf = train[target_clf].values
    y_valid_clf = valid[target_clf].values
    y_test_clf  = test[target_clf].values

    y_train_reg = train[target_reg].values
    y_valid_reg = valid[target_reg].values
    y_test_reg  = test[target_reg].values

    out_dir = Path("artifacts/tabular")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save numpy arrays
    np.save(out_dir / "X_train.npy", Xt_train)
    np.save(out_dir / "X_valid.npy", Xt_valid)
    np.save(out_dir / "X_test.npy", Xt_test)
    np.save(out_dir / "y_train_clf.npy", y_train_clf)
    np.save(out_dir / "y_valid_clf.npy", y_valid_clf)
    np.save(out_dir / "y_test_clf.npy", y_test_clf)
    np.save(out_dir / "y_train_reg.npy", y_train_reg)
    np.save(out_dir / "y_valid_reg.npy", y_valid_reg)
    np.save(out_dir / "y_test_reg.npy", y_test_reg)

    with open(out_dir / "preprocess_pipeline.pkl","wb") as f:
        pickle.dump(pipe, f)

    meta = {
        "targets":{"classification":target_clf,"regression":target_reg},
        "numeric": num_cols,
        "categorical": cat_cols,
        "n_features_out": int(Xt_train.shape[1])
    }
    with open(out_dir / "preprocess_meta.json","w") as f:
        json.dump(meta, f, indent=2)

    mlflow.start_run(run_name="data_preprocess_tabular")
    mlflow.log_param("n_features_out", int(Xt_train.shape[1]))
    for fn in [
        "X_train.npy","X_valid.npy","X_test.npy",
        "y_train_clf.npy","y_valid_clf.npy","y_test_clf.npy",
        "y_train_reg.npy","y_valid_reg.npy","y_test_reg.npy",
        "preprocess_pipeline.pkl","preprocess_meta.json"
    ]:
        mlflow.log_artifact(str(out_dir / fn))
    mlflow.end_run()

if __name__ == "__main__":
    main()
