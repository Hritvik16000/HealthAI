import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

TAB_DIR = Path("artifacts/tabular")
OUT_DIR = Path("artifacts/interpretability/tabular")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(TAB_DIR / "preprocess_pipeline.pkl", "rb") as f:
    PIPE = pickle.load(f)
with open(TAB_DIR / "clf_gb.pkl", "rb") as f:
    CLF = pickle.load(f)
with open(TAB_DIR / "reg_gb.pkl", "rb") as f:
    REG = pickle.load(f)

Xtr = np.load(TAB_DIR / "X_train.npy", allow_pickle=True)
Xva = np.load(TAB_DIR / "X_valid.npy", allow_pickle=True)
ytr_c = np.load(TAB_DIR / "y_train_clf.npy", allow_pickle=True)
yva_c = np.load(TAB_DIR / "y_valid_clf.npy", allow_pickle=True)
ytr_r = np.load(TAB_DIR / "y_train_reg.npy", allow_pickle=True)
yva_r = np.load(TAB_DIR / "y_valid_reg.npy", allow_pickle=True)

try:
    feat_names = PIPE.get_feature_names_out()
except Exception:
    feat_names = [f"f{i}" for i in range(Xtr.shape[1])]

def ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

Xtr_d = ensure_dense(Xtr)
Xva_d = ensure_dense(Xva)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# SHAP for Classification
try:
    background_idx = np.random.default_rng(42).choice(Xtr_d.shape[0], size=min(500, Xtr_d.shape[0]), replace=False)
    explainer_clf = shap.TreeExplainer(CLF)
    shap_values_clf = explainer_clf.shap_values(Xva_d[background_idx])
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_clf, Xva_d[background_idx], feature_names=feat_names, show=False)
    save_fig(OUT_DIR / "shap_summary_classification.png")
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_clf, Xva_d[background_idx], feature_names=feat_names, plot_type="bar", show=False)
    save_fig(OUT_DIR / "shap_bar_classification.png")
except Exception as e:
    (OUT_DIR / "shap_classification_error.txt").write_text(str(e))

# SHAP for Regression
try:
    background_idx = np.random.default_rng(7).choice(Xtr_d.shape[0], size=min(500, Xtr_d.shape[0]), replace=False)
    explainer_reg = shap.TreeExplainer(REG)
    shap_values_reg = explainer_reg.shap_values(Xva_d[background_idx])
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_reg, Xva_d[background_idx], feature_names=feat_names, show=False)
    save_fig(OUT_DIR / "shap_summary_regression.png")
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_reg, Xva_d[background_idx], feature_names=feat_names, plot_type="bar", show=False)
    save_fig(OUT_DIR / "shap_bar_regression.png")
except Exception as e:
    (OUT_DIR / "shap_regression_error.txt").write_text(str(e))

# LIME for Classification
try:
    expl = LimeTabularExplainer(
        training_data=Xtr_d,
        feature_names=feat_names,
        class_names=["no_readmit","readmit"],
        discretize_continuous=True,
        random_state=42,
        verbose=False,
        mode="classification",
    )
    def predict_proba_clf(X):
        return CLF.predict_proba(X)
    for i, idx in enumerate(np.random.default_rng(99).choice(Xva_d.shape[0], size=min(3, Xva_d.shape[0]), replace=False)):
        exp = expl.explain_instance(Xva_d[idx], predict_proba_clf, num_features=10)
        (OUT_DIR / f"lime_clf_{i}.html").write_text(exp.as_html())
except Exception as e:
    (OUT_DIR / "lime_classification_error.txt").write_text(str(e))

# LIME for Regression
try:
    expl_r = LimeTabularExplainer(
        training_data=Xtr_d,
        feature_names=feat_names,
        discretize_continuous=True,
        random_state=123,
        verbose=False,
        mode="regression",
    )
    def predict_reg(X):
        return REG.predict(X)
    for i, idx in enumerate(np.random.default_rng(123).choice(Xva_d.shape[0], size=min(3, Xva_d.shape[0]), replace=False)):
        exp = expl_r.explain_instance(Xva_d[idx], predict_reg, num_features=10)
        (OUT_DIR / f"lime_reg_{i}.html").write_text(exp.as_html())
except Exception as e:
    (OUT_DIR / "lime_regression_error.txt").write_text(str(e))

meta = {
    "classification": {
        "shap_summary": "shap_summary_classification.png",
        "shap_bar": "shap_bar_classification.png",
        "lime_html_samples": [f"lime_clf_{i}.html" for i in range(3)]
    },
    "regression": {
        "shap_summary": "shap_summary_regression.png",
        "shap_bar": "shap_bar_regression.png",
        "lime_html_samples": [f"lime_reg_{i}.html" for i in range(3)]
    },
    "feature_names_count": int(len(feat_names))
}
(Path("artifacts/interpretability/tabular/meta.json")).write_text(json.dumps(meta, indent=2))
print("Interpretability artifacts saved to", OUT_DIR)
