from pathlib import Path
import numpy as np
import pandas as pd
from src.common.mlflow_utils import init_mlflow

def synth_timeseries(n_patients=500, timesteps=96, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients+1):
        base_hr = rng.normal(72, 6)
        base_bp = rng.normal(120, 10)
        for t in range(timesteps):
            hour = t
            hr = base_hr + 6*np.sin(2*np.pi*(hour/24)) + rng.normal(0, 2)
            sbp = base_bp + 10*np.sin(2*np.pi*(hour/24 + 0.25)) + rng.normal(0, 3)
            spo2 = 96 + 1.0*np.sin(2*np.pi*(hour/24 + 0.5)) + rng.normal(0, 0.3)
            rows.append((pid, t, hr, sbp, spo2))
    return pd.DataFrame(rows, columns=["patient_id","t","heart_rate","systolic_bp","spo2"])

def main():
    mlflow = init_mlflow()
    df = synth_timeseries()
    out = Path("data/raw/vitals_timeseries.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    mlflow.start_run(run_name="data_generate_timeseries")
    mlflow.log_artifact(str(out))
    mlflow.end_run()

if __name__ == "__main__":
    main()
