import numpy as np
import pytest

from banking_ticket_routing.model import select_abstention_threshold, top_k_accuracy


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
