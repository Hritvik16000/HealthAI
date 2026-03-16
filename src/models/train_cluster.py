from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "cluster"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "age", "gender", "bmi", "blood_pressure",
    "glucose", "cholesterol", "heart_rate",
    "smoker", "diabetes_history"
]

X = df[feature_cols].copy()

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=20))
])

clusters = pipeline.fit_predict(X)

score = silhouette_score(StandardScaler().fit_transform(X), clusters)
print("Silhouette Score:", round(float(score), 4))

df["cluster"] = clusters

profiles = df.groupby("cluster")[feature_cols].mean().round(2)

# create human-friendly names
cluster_names = {}
for cluster_id, row in profiles.iterrows():
    risk_points = 0
    if row["age"] >= 55:
        risk_points += 1
    if row["bmi"] >= 29:
        risk_points += 1
    if row["blood_pressure"] >= 135:
        risk_points += 1
    if row["glucose"] >= 120:
        risk_points += 1
    if row["cholesterol"] >= 220:
        risk_points += 1
    if row["smoker"] >= 0.5:
        risk_points += 1
    if row["diabetes_history"] >= 0.5:
        risk_points += 1

    if risk_points >= 5:
        cluster_names[int(cluster_id)] = "High-Risk Metabolic"
    elif risk_points >= 3:
        cluster_names[int(cluster_id)] = "Moderate-Risk General"
    else:
        cluster_names[int(cluster_id)] = "Lower-Risk Stable"

joblib.dump(pipeline, ARTIFACT_DIR / "kmeans_pipeline.pkl")
profiles.to_csv(ARTIFACT_DIR / "cluster_profiles.csv")

with open(ARTIFACT_DIR / "cluster_names.json", "w") as f:
    json.dump(cluster_names, f, indent=2)

print("[OK] Cluster pipeline saved")
print("[OK] Cluster profiles saved")
print("[OK] Cluster names saved")
print(profiles)
print(cluster_names)
