"""Сериализация артефактов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODEL_SCHEMA_VERSION = 1


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_bundle(bundle: Any) -> dict[str, Any]:
    """Проверить версию, интерфейс модели и порядок intent-классов."""
    if not isinstance(bundle, dict):
        raise ValueError("Файл не содержит bundle-словарь модели")
    required = {"schema_version", "model", "abstention_threshold", "intent_classes"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"В bundle отсутствуют обязательные поля: {sorted(missing)}")
    if bundle["schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Неподдерживаемая schema_version bundle: "
            f"{bundle['schema_version']}; ожидается {MODEL_SCHEMA_VERSION}"
        )

    try:
        threshold = float(bundle["abstention_threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError("abstention_threshold должен быть числом") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("abstention_threshold должен быть конечным значением от 0 до 1")

    model = bundle["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Модель bundle должна реализовывать predict_proba")
    if not hasattr(model, "classes_"):
        raise ValueError("Модель bundle должна содержать classes_")
    model_classes = np.asarray(model.classes_, dtype=object)
    if model_classes.ndim != 1 or model_classes.size < 2:
        raise ValueError("model.classes_ должен быть одномерным и содержать минимум два класса")

    intent_classes = bundle["intent_classes"]
    if (
        not isinstance(intent_classes, list)
        or not all(isinstance(label, str) and label for label in intent_classes)
        or len(set(intent_classes)) != len(intent_classes)
    ):
        raise ValueError("intent_classes должен быть списком уникальных непустых строк")
    if model_classes.tolist() != intent_classes:
        raise ValueError("Порядок intent_classes не совпадает с model.classes_")

    validated = dict(bundle)
    validated["abstention_threshold"] = threshold
    return validated


def load_bundle(path: str | Path) -> dict[str, Any]:
    return validate_bundle(joblib.load(path))
