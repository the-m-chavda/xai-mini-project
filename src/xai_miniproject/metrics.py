from __future__ import annotations


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def macro_f1(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    if not y_true:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0
