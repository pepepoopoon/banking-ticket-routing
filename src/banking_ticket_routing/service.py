"""Независимый от веб-фреймворка сервисный слой."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_bundle


class RoutingService:
    """Загрузить модель один раз и маршрутизировать отдельные обращения."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.model = bundle["model"]
        self.abstention_threshold = float(bundle["abstention_threshold"])

    @classmethod
    def from_path(cls, path: str | Path) -> RoutingService:
        return cls(load_bundle(path))

    def route(self, text: str, *, top_k: int = 3) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("Текст обращения не может быть пустым")
        if top_k < 1:
            raise ValueError("top_k должен быть положительным")
        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        indices = np.argsort(probabilities)[::-1][: min(top_k, len(classes))]
        candidates = [
            {"intent": str(classes[index]), "probability": float(probabilities[index])}
            for index in indices
        ]
        confidence = candidates[0]["probability"]
        abstained = confidence < self.abstention_threshold
        return {
            "intent": None if abstained else candidates[0]["intent"],
            "suggested_intent": candidates[0]["intent"],
            "confidence": confidence,
            "abstained": abstained,
            "abstention_threshold": self.abstention_threshold,
            "candidates": candidates,
        }
