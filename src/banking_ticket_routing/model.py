"""Калиброванная модель, abstention и метрики."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import FeatureUnion, Pipeline

FeatureMode = Literal["word", "char", "union"]
CalibrationMethod = Literal["sigmoid", "isotonic"]


def build_model(
    *,
    random_state: int = 42,
    regularization_c: float = 1.0,
    feature_mode: FeatureMode = "union",
    word_ngram_max: int = 2,
    char_ngram_max: int = 5,
    calibration_method: CalibrationMethod = "sigmoid",
    calibration_cv: int = 3,
) -> CalibratedClassifierCV:
    if regularization_c <= 0:
        raise ValueError("regularization_c должен быть положительным")
    if feature_mode not in {"word", "char", "union"}:
        raise ValueError("feature_mode должен быть word, char или union")
    if word_ngram_max < 1:
        raise ValueError("word_ngram_max должен быть не меньше 1")
    if char_ngram_max < 3:
        raise ValueError("char_ngram_max должен быть не меньше 3")
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method должен быть sigmoid или isotonic")
    if calibration_cv < 2:
        raise ValueError("calibration_cv должен быть не меньше 2")

    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, word_ngram_max),
        min_df=1,
        sublinear_tf=True,
    )
    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, char_ngram_max),
        min_df=1,
        sublinear_tf=True,
    )
    if feature_mode == "word":
        features: TfidfVectorizer | FeatureUnion = word_features
    elif feature_mode == "char":
        features = char_features
    else:
        features = FeatureUnion(
            [
                ("word", word_features),
                ("char", char_features),
            ]
        )
    estimator = LogisticRegression(
        class_weight="balanced",
        C=regularization_c,
        max_iter=1_000,
        random_state=random_state,
    )
    base_pipeline = Pipeline([("features", features), ("classifier", estimator)])
    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calibration_method,
        cv=calibration_cv,
    )


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


def calibration_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Посчитать multiclass Brier score и confidence-based ECE."""
    if n_bins < 2:
        raise ValueError("Для ECE нужно минимум два интервала")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Размерности probabilities и classes не согласованы")
    class_to_index = {label: index for index, label in enumerate(classes)}
    try:
        true_indices = np.asarray([class_to_index[label] for label in y_true])
    except KeyError as error:
        raise ValueError(f"Неизвестная истинная метка: {error.args[0]}") from error

    targets = np.zeros_like(probabilities, dtype=float)
    targets[np.arange(len(true_indices)), true_indices] = 1.0
    brier_score = float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
    confidences = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    correct = predictions == y_true
    bin_indices = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    bins: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for index in range(n_bins):
        selected = bin_indices == index
        if not selected.any():
            continue
        count = int(selected.sum())
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidences[selected].mean())
        expected_calibration_error += count / len(y_true) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "multiclass_brier_score": brier_score,
        "expected_calibration_error": float(expected_calibration_error),
        "mean_confidence": float(confidences.mean()),
        "bins": bins,
    }


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
    diagnostics = calibration_diagnostics(y_true, probabilities, classes)
    report = classification_report(
        y_true,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
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
        "selective_risk": None if selective_accuracy is None else 1 - selective_accuracy,
        "calibration": diagnostics,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=classes).tolist(),
        "per_class": {str(label): report[str(label)] for label in classes},
        "labels": [str(label) for label in classes],
    }
