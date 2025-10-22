from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

age = rng.integers(18, 90, N)
bmi = rng.normal(27, 5, N).clip(16, 55)
hr = rng.normal(78, 12, N).clip(40, 160)
sbp = rng.normal(125, 18, N).clip(85, 220)
dbp = rng.normal(78, 12, N).clip(45, 140)
glucose = rng.normal(105, 25, N).clip(60, 300)

sex = rng.choice(["Male", "Female"], N, p=[0.48, 0.52])
smoker = rng.choice(["No", "Yes"], N, p=[0.7, 0.3])
diabetic = rng.choice(["No", "Yes"], N, p=[0.8, 0.2])

# Logistic-ish ground truth for 30d readmission probability
logit = (
    -5.0
    + 0.02 * (age - 50)
    + 0.04 * (bmi - 27)
    + 0.015 * (hr - 78)
    + 0.01 * (sbp - 125)
    + 0.012 * (dbp - 78)
    + 0.01 * (glucose - 105)
    + 0.6 * (smoker == "Yes")
    + 0.9 * (diabetic == "Yes")
    + 0.2 * (sex == "Male")
)
p = 1 / (1 + np.exp(-logit))
readmission = (rng.random(N) < p).astype(int)

df = pd.DataFrame(
    {
        "age": age,
        "bmi": bmi.round(1),
        "hr": hr.round(0),
        "sbp": sbp.round(0),
        "dbp": dbp.round(0),
        "glucose": glucose.round(0),
        "sex": sex,
        "smoker": smoker,
        "diabetic": diabetic,
        "readmission_30d": readmission,
    }
)

out = Path("data/processed/tabular_train.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out} with shape {df.shape}")
