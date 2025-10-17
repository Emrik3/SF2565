from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from utils import dprint
import numpy as np


def evaluate_model(model, X, y, X_test, y_test):
    # ensure numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    dprint("fitting model in evaluate_model()...")
    if X is not None and y is not None:
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y)
    dprint("predicting probabiliites...")
    try:
        proba = model.predict_proba(X_test)
        pos_proba = proba[:, 1]
    except Exception:
        from scipy.special import expit

        scores = model.decision_function(X_test)
        pos_proba = expit(scores)

    pred = (pos_proba >= 0.5).astype(int)

    metrics = {
        "log_loss": log_loss(y_test, np.vstack([1 - pos_proba, pos_proba]).T),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, pos_proba),
        "pr_auc": average_precision_score(y_test, pos_proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "pos_rate": pred.mean(),
        "confusion": confusion_matrix(y_test, pred),
        "pos_proba_std": pos_proba.std(),
    }

    return metrics


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Evaluate model predictions using common classification metrics."""
    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    results["recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    results["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Compute ROC AUC only if we have probabilities and more than one class
    if y_proba is not None and y_proba.shape[1] > 1:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            results["roc_auc"] = np.nan

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    print("=== Evaluation Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:\n", results["confusion_matrix"])
    print("=========================")

    return results
