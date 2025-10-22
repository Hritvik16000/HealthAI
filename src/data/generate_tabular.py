import os, json, math, random, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.common.mlflow_utils import init_mlflow

RNG = np.random.default_rng(42)

def synth_tabular(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["M","F"], size=n)
    bmi = rng.normal(27.5, 5.0, size=n).clip(14, 60)
    sbp = rng.normal(122, 18, size=n).clip(80, 220)
    dbp = rng.normal(78, 12, size=n).clip(40, 140)
    hr  = rng.normal(75, 12, size=n).clip(40, 190)
    chol = rng.normal(190, 45, size=n).clip(90, 400)
    smoker = rng.choice([0,1], size=n, p=[0.72,0.28])
    diabetic = rng.choice([0,1], size=n, p=[0.86,0.14])
    # risk logit
    logit = (
        -4.0
        + 0.03*(age-50)
        + 0.05*(bmi-27)
        + 0.01*(sbp-120)
        + 0.008*(hr-70)
        + 0.004*(chol-180)
        + 0.9*smoker
        + 1.1*diabetic
    )
    prob = 1/(1+np.exp(-logit))
    readmit_30d = (rng.random(n) < prob).astype(int)

    # LOS regression with noise
    los = (
        2.5
        + 0.04*(age-50)
        + 0.06*(bmi-27)
        + 0.03*(sbp>140)
        + 0.7*smoker
        + 0.9*diabetic
        + rng.normal(0, 1.2, size=n)
    )
    los = np.clip(los, 0.5, 21.0)

    # introduce some missingness
    mask = rng.random(n) < 0.05
    bmi[mask] = np.nan
    mask = rng.random(n) < 0.03
    chol[mask] = np.nan

    df = pd.DataFrame({
        "patient_id": np.arange(1, n+1),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": hr,
        "cholesterol": chol,
        "smoker": smoker,
        "diabetic": diabetic,
        "readmit_30d": readmit_30d,
        "length_of_stay": los,
    })
    return df

def main():
    mlflow = init_mlflow()
    params = {"n": 5000, "seed": 42, "version": f"synth_tabular_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"}
    df = synth_tabular(n=params["n"], seed=params["seed"])
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / "patients.csv"
    df.to_csv(raw_csv, index=False)

    # simple split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(0.7*n)]
    valid_df = df.iloc[int(0.7*n): int(0.85*n)]
    test_df  = df.iloc[int(0.85*n):]

    train_df.to_csv(proc_dir / "patients_train.csv", index=False)
    valid_df.to_csv(proc_dir / "patients_valid.csv", index=False)
    test_df.to_csv(proc_dir / "patients_test.csv", index=False)

    with open(proc_dir / "dataset_meta.json","w") as f:
        json.dump(params, f, indent=2)

    mlflow.start_run(run_name="data_generate_tabular")
    mlflow.log_params(params)
    mlflow.log_artifact(str(raw_csv))
    mlflow.log_artifact(str(proc_dir / "patients_train.csv"))
    mlflow.log_artifact(str(proc_dir / "patients_valid.csv"))
    mlflow.log_artifact(str(proc_dir / "patients_test.csv"))
    mlflow.end_run()

if __name__ == "__main__":
    main()
