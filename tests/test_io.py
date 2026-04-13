from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pytest

from banking_ticket_routing.io import MODEL_SCHEMA_VERSION, load_bundle
from banking_ticket_routing.service import RoutingService


class CompatibleModel:
    def __init__(self) -> None:
        self.classes_ = np.asarray(["card_arrival", "transfer_pending"])

    def predict_proba(self, texts) -> np.ndarray:
        return np.tile([0.7, 0.3], (len(texts), 1))


class ModelWithoutClasses:
    def predict_proba(self, texts) -> np.ndarray:
        return np.tile([0.7, 0.3], (len(texts), 1))


def make_bundle() -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": CompatibleModel(),
        "intent_classes": ["card_arrival", "transfer_pending"],
        "abstention_threshold": 0.6,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda bundle: bundle.update(schema_version=2), "Неподдерживаемая schema_version"),
        (lambda bundle: bundle.update(abstention_threshold=np.nan), "конечным значением"),
        (lambda bundle: bundle.update(model=object()), "реализовывать predict_proba"),
        (lambda bundle: bundle.update(model=ModelWithoutClasses()), "содержать classes_"),
        (
            lambda bundle: bundle.update(intent_classes=["transfer_pending", "card_arrival"]),
            "Порядок intent_classes",
        ),
    ],
)
def test_load_bundle_rejects_incompatible_contract(
    tmp_path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    bundle = make_bundle()
    mutate(bundle)
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)

    with pytest.raises(ValueError, match=message):
        load_bundle(model_path)


def test_routing_service_validates_direct_bundle() -> None:
    bundle = make_bundle()
    bundle["intent_classes"] = ["transfer_pending", "card_arrival"]

    with pytest.raises(ValueError, match="Порядок intent_classes"):
        RoutingService(bundle)
