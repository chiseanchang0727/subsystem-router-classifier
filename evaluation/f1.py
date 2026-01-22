from typing import List, Dict
import numpy as np
from sklearn.metrics import f1_score, classification_report


def evaluate_multilabel_f1(
    test_data: List[Dict],
    predictions: List[Dict],
    labels: List[str],
    verbose: bool = True,
):
    """
    Compute multi-label F1 scores for subsystem classification.

    Args:
        test_data: list of test samples, each containing `subsystems` ground truth
        predictions: list of prediction dicts (output of predict_batch)
        labels: ordered list of label names
        verbose: whether to print detailed report

    Returns:
        dict with f1_micro, f1_macro, f1_samples
    """
    y_true = []
    y_pred = []

    for item, pred in zip(test_data, predictions):
        y_true.append([item["subsystems"][label] for label in labels])
        y_pred.append([int(pred[label]["predicted"]) for label in labels])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_samples": f1_score(y_true, y_pred, average="samples"),
    }

    if verbose:
        print("\n===== F1 Scores =====")
        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")

        print("\n===== Per-label Report =====")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=labels,
                digits=4,
                zero_division=0,
            )
        )

    return metrics
