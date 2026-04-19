import numpy as np
import pytest

from banking_ticket_routing.model import (
    build_model,
    calibration_diagnostics,
    per_intent_routing_diagnostics,
    select_abstention_threshold,
    top_k_accuracy,
)


def test_abstention_threshold_respects_minimum_coverage() -> None:
    classes = np.asarray(["a", "b"])
    labels = np.asarray(["a", "a", "b", "b"])
    probabilities = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.55, 0.45]])

    threshold = select_abstention_threshold(labels, probabilities, classes, minimum_coverage=0.5)
    coverage = float((probabilities.max(axis=1) >= threshold).mean())

    assert 0.0 <= threshold <= 1.0
    assert coverage >= 0.5


@pytest.mark.parametrize("invalid_k", [0, 3])
def test_top_k_accuracy_rejects_k_outside_class_count(invalid_k: int) -> None:
    labels = np.asarray(["a", "b"])
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    classes = np.asarray(["a", "b"])

    with pytest.raises(ValueError, match=r"k должен быть в диапазоне \[1, 2\]"):
        top_k_accuracy(labels, probabilities, classes, k=invalid_k)


@pytest.mark.parametrize("feature_mode", ["word", "char", "union"])
def test_model_supports_feature_ablation(feature_mode: str) -> None:
    model = build_model(
        feature_mode=feature_mode,
        regularization_c=0.5,
        calibration_method="isotonic",
        calibration_cv=2,
    )

    assert model.method == "isotonic"
    assert model.cv == 2
    assert model.estimator.named_steps["classifier"].C == 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"regularization_c": 0}, "положительным"),
        ({"feature_mode": "invalid"}, "word, char или union"),
        ({"word_ngram_max": 0}, "не меньше 1"),
        ({"char_ngram_max": 2}, "не меньше 3"),
        ({"calibration_method": "invalid"}, "sigmoid или isotonic"),
        ({"calibration_cv": 1}, "не меньше 2"),
    ],
)
def test_model_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_model(**kwargs)


def test_calibration_diagnostics_match_known_probabilities() -> None:
    classes = np.asarray(["a", "b"])
    labels = np.asarray(["a", "b"])
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6]])

    diagnostics = calibration_diagnostics(labels, probabilities, classes, n_bins=5)

    assert diagnostics["multiclass_brier_score"] == pytest.approx(0.2)
    assert diagnostics["expected_calibration_error"] == pytest.approx(0.3)
    assert sum(item["count"] for item in diagnostics["bins"]) == 2


def test_calibration_diagnostics_reject_unknown_label() -> None:
    with pytest.raises(ValueError, match="Неизвестная истинная метка"):
        calibration_diagnostics(
            np.asarray(["unknown"]),
            np.asarray([[0.5, 0.5]]),
            np.asarray(["a", "b"]),
        )


def test_per_intent_diagnostics_expose_coverage_gap() -> None:
    classes = np.asarray(["a", "b"])
    labels = np.asarray(["a", "a", "b", "b"])
    probabilities = np.asarray(
        [[0.9, 0.1], [0.55, 0.45], [0.2, 0.8], [0.4, 0.6]]
    )

    diagnostics = per_intent_routing_diagnostics(
        labels,
        probabilities,
        classes,
        abstention_threshold=0.7,
    )

    assert diagnostics["a"]["coverage"] == pytest.approx(0.5)
    assert diagnostics["b"]["coverage"] == pytest.approx(0.5)
    assert diagnostics["a"]["selective_accuracy"] == 1.0
    assert diagnostics["b"]["accepted_rows"] == 1
