"""Калиброванная модель, abstention и метрики."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import FeatureUnion, Pipeline


def build_model(*, random_state: int = 42) -> CalibratedClassifierCV:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    base_pipeline = Pipeline([("features", features), ("classifier", estimator)])
    return CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    k: int,
) -> float:
    class_count = probabilities.shape[1]
    if not 1 <= k <= class_count:
        raise ValueError(f"k должен быть в диапазоне [1, {class_count}], получено: {k}")
    indices = np.argsort(probabilities, axis=1)[:, -k:]
    labels = classes[indices]
    return float(np.mean([truth in row for truth, row in zip(y_true, labels, strict=True)]))


def select_abstention_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    minimum_coverage: float = 0.6,
) -> float:
    """Выбрать порог по validation, не опускаясь ниже заданного покрытия."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage должен быть в интервале (0, 1]")
    confidences = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    candidates = np.unique(np.concatenate(([0.0], confidences)))
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        accepted = confidences >= threshold
        coverage = float(accepted.mean())
        if coverage >= minimum_coverage and accepted.any():
            accuracy = float(np.mean(predictions[accepted] == y_true[accepted]))
            feasible.append((accuracy, threshold, coverage))
    if not feasible:
        return 0.0
    _, threshold, _ = max(feasible, key=lambda item: (item[0], item[1], item[2]))
    return float(threshold)


def classification_metrics(
    model: CalibratedClassifierCV,
    texts: Any,
    labels: Any,
    *,
    abstention_threshold: float,
    top_k: int = 3,
) -> dict[str, Any]:
    y_true = np.asarray(labels)
    probabilities = model.predict_proba(texts)
    classes = model.classes_
    predictions = classes[probabilities.argmax(axis=1)]
    confidences = probabilities.max(axis=1)
    accepted = confidences >= abstention_threshold
    selective_accuracy = (
        float(np.mean(predictions[accepted] == y_true[accepted])) if accepted.any() else None
    )
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        f"top_{top_k}_accuracy": top_k_accuracy(y_true, probabilities, classes, k=top_k),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "abstention_threshold": float(abstention_threshold),
        "coverage": float(accepted.mean()),
        "abstention_rate": float(1 - accepted.mean()),
        "selective_accuracy": selective_accuracy,
        "labels": [str(label) for label in classes],
    }
