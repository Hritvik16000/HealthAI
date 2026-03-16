from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "cluster"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["risk_label", "length_of_stay"], errors="ignore")

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X)

score = silhouette_score(X, clusters)
print("Silhouette Score:", score)

joblib.dump(model, ARTIFACT_DIR / "kmeans.pkl")
print("[OK] Cluster model saved")
