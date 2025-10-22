from sklearn import metrics


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    out = {}
    out["f1"] = metrics.f1_score(y_true, y_pred)
    out["precision"] = metrics.precision_score(y_true, y_pred)
    out["recall"] = metrics.recall_score(y_true, y_pred)
    if y_prob is not None:
        out["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
    return out
