import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"model": name, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def export_model_metrics(results, save_path):
    """Export one or more model results (as returned by evaluate_model) to a JSON file."""
    if isinstance(results, dict):
        results = [results]

    metrics_out = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results}

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_out, f, indent=2)