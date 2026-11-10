"""Shared scores for comparing the group's classifiers."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_predictions(y_true, y_pred, class_names):
    """Score paired food labels using the supplied class order.

    Pass one correct label and one predicted label per photo, in the same
    photo order. Use the group's full class list for every model. The matrix
    uses that order for both its rows (actual) and columns (predicted).
    """
    class_names = list(class_names)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Provide a class list with one entry per food.")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Provide one correct label and one prediction per photo.")

    unknown_labels = set(y_true).union(y_pred).difference(class_names)
    if unknown_labels:
        raise ValueError(f"These labels are missing from class_names: {unknown_labels}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names),
    }
