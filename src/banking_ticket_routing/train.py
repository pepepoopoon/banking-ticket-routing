"""CLI обучения маршрутизатора обращений."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from .data import load_csv, stratified_split
from .io import write_json
from .model import build_model, classification_metrics, select_abstention_threshold

LOGGER = logging.getLogger(__name__)


def train(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    minimum_coverage: float = 0.6,
) -> dict:
    data = load_csv(data_path)
    train_frame, validation_frame, test_frame, manifest = stratified_split(
        data, random_state=random_state
    )
    model = build_model(random_state=random_state)
    model.fit(train_frame["text"], train_frame["intent"])
    validation_probabilities = model.predict_proba(validation_frame["text"])
    classes = model.classes_
    threshold = select_abstention_threshold(
        validation_frame["intent"].to_numpy(),
        validation_probabilities,
        classes,
        minimum_coverage=minimum_coverage,
    )
    validation_metrics = classification_metrics(
        model,
        validation_frame["text"],
        validation_frame["intent"],
        abstention_threshold=threshold,
    )
    bundle = {
        "schema_version": 1,
        "random_state": random_state,
        "model": model,
        "abstention_threshold": threshold,
        "minimum_validation_coverage": minimum_coverage,
        "calibration": "sigmoid, 3-fold CV inside train",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "model.joblib")
    manifest.to_csv(destination / "split_manifest.csv", index=False)
    validation_frame.to_csv(destination / "validation.csv", index=False)
    test_frame.to_csv(destination / "test.csv", index=False)
    metadata = {
        "schema_version": 1,
        "random_state": random_state,
        "split_strategy": "stratified by intent",
        "calibration": bundle["calibration"],
        "minimum_validation_coverage": minimum_coverage,
        "abstention_threshold": threshold,
        "split_rows": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "validation_metrics": validation_metrics,
    }
    write_json(destination / "metadata.json", metadata)
    LOGGER.info("Артефакты сохранены в %s", destination)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-coverage", type=float, default=0.6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(
        args.data,
        args.output_dir,
        random_state=args.seed,
        minimum_coverage=args.minimum_coverage,
    )


if __name__ == "__main__":
    main()
