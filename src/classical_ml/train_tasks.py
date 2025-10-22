import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.common.mlflow_utils import init_mlflow


def run():
    mlflow = init_mlflow()
    with mlflow.start_run(run_name="classical"):
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
        yp = clf.predict(Xte)
        mlflow.log_metric("clf_accuracy", accuracy_score(yte, yp))
        mlflow.log_metric("clf_f1", f1_score(yte, yp))
        Xr, yr = make_regression(n_samples=1000, n_features=10, random_state=42, noise=10)
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(random_state=42).fit(Xtr2, ytr2)
        yp2 = reg.predict(Xte2)
        mlflow.log_metric("reg_rmse", mean_squared_error(yte2, yp2, squared=False))
        mlflow.log_metric("reg_r2", r2_score(yte2, yp2))
        Xc = np.random.rand(500, 5)
        km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xc)
        sil = pd.Series(km.labels_).value_counts().mean()
        mlflow.log_metric("cluster_label_mean", sil)
        basket = pd.DataFrame(
            np.random.randint(0, 2, (200, 6)), columns=[f"item_{i}" for i in range(6)]
        )
        freq = apriori(basket, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        rules.to_csv("artifacts/association_rules.csv", index=False)
        mlflow.log_artifact("artifacts/association_rules.csv")


if __name__ == "__main__":
    run()
