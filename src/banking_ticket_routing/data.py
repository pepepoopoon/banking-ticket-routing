"""Контракт и разбиение данных обращений."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("ticket_id", "text", "intent")


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing)}")
    data = frame.copy()
    for column in REQUIRED_COLUMNS:
        if data[column].isna().any():
            raise ValueError(f"Столбец {column!r} содержит пропуски")
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq("").any():
            raise ValueError(f"Столбец {column!r} содержит пустые значения")
    if data["ticket_id"].duplicated().any():
        raise ValueError("ticket_id должен быть уникальным")
    if data["text"].str.casefold().duplicated().any():
        raise ValueError("Повторяющиеся тексты необходимо удалить до split")
    return data


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path))


def stratified_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = validate_frame(frame)
    if data["intent"].nunique() < 2:
        raise ValueError("Для обучения нужны как минимум два intent-класса")
    counts = data["intent"].value_counts()
    scarce = counts[counts < 8]
    if not scarce.empty:
        raise ValueError(
            "Для split и трёхфолдовой калибровки нужно минимум 8 строк на intent: "
            f"{scarce.to_dict()}"
        )
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Доли validation и test должны быть положительными")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Сумма долей validation и test должна быть меньше 1")
    train_validation, test = train_test_split(
        data,
        test_size=test_fraction,
        random_state=random_state,
        stratify=data["intent"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction / (1 - test_fraction),
        random_state=random_state,
        stratify=train_validation["intent"],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    manifest = pd.concat(
        [
            train[["ticket_id", "intent"]].assign(split="train"),
            validation[["ticket_id", "intent"]].assign(split="validation"),
            test[["ticket_id", "intent"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    return train, validation, test, manifest
