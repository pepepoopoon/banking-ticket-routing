"""Детерминированные искажения для проверки устойчивости маршрутизатора."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_label_noise(
    frame: pd.DataFrame,
    *,
    rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, int]:
    """Заменить часть train-меток случайным отличающимся intent."""
    if not 0 <= rate <= 0.5:
        raise ValueError("rate шума меток должен быть в диапазоне [0, 0.5]")
    result = frame.copy()
    labels = sorted(result["intent"].unique())
    if len(labels) < 2:
        raise ValueError("Для шума меток нужны минимум два intent")
    rng = np.random.default_rng(random_state)
    changed = rng.random(len(result)) < rate
    for index in np.flatnonzero(changed):
        current = result.iloc[index]["intent"]
        alternatives = [label for label in labels if label != current]
        result.iat[index, result.columns.get_loc("intent")] = rng.choice(alternatives)
    return result, int(changed.sum())


def _corrupt_token(token: str, rng: np.random.Generator) -> str:
    if len(token) < 4:
        return token + "x"
    position = int(rng.integers(1, len(token) - 1))
    characters = list(token)
    characters[position - 1], characters[position] = (
        characters[position],
        characters[position - 1],
    )
    return "".join(characters)


def inject_token_noise(
    frame: pd.DataFrame,
    *,
    rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, int]:
    """Внести опечатки в долю токенов validation/test текстов."""
    if not 0 <= rate <= 0.8:
        raise ValueError("rate токенного шума должен быть в диапазоне [0, 0.8]")
    result = frame.copy()
    rng = np.random.default_rng(random_state)
    changed_tokens = 0
    noisy_texts: list[str] = []
    for text in result["text"]:
        noisy_tokens: list[str] = []
        for token in text.split():
            if rng.random() < rate:
                token = _corrupt_token(token, rng)
                changed_tokens += 1
            noisy_tokens.append(token)
        noisy_texts.append(" ".join(noisy_tokens))
    result["text"] = noisy_texts
    return result, changed_tokens


def downsample_intent(
    frame: pd.DataFrame,
    *,
    intent: str,
    keep_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    """Уменьшить один intent-класс, сохранив минимум восемь строк для split."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction должен быть в диапазоне (0, 1]")
    selected = frame[frame["intent"] == intent]
    if selected.empty:
        raise ValueError(f"Intent {intent!r} отсутствует в данных")
    keep_rows = min(len(selected), max(8, round(len(selected) * keep_fraction)))
    kept = selected.sample(n=keep_rows, random_state=random_state)
    others = frame[frame["intent"] != intent]
    return (
        pd.concat([others, kept], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
