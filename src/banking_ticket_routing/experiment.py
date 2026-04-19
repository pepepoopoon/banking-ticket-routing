"""Воспроизводимые эксперименты чувствительности маршрутизатора обращений."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import stratified_split, validate_frame
from .demo_data import make_smoke_data
from .io import write_json
from .model import build_model, classification_metrics, select_abstention_threshold
from .stress import downsample_intent, inject_label_noise, inject_token_noise


def _feature_count(model: Any) -> int:
    calibrated = model.calibrated_classifiers_[0]
    features = calibrated.estimator.named_steps["features"]
    return int(len(features.get_feature_names_out()))


def run_experiment(
    *,
    hypothesis: str,
    samples_per_intent: int = 20,
    data_seed: int = 42,
    split_seed: int = 42,
    model_seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    minimum_coverage: float = 0.6,
    regularization_c: float = 1.0,
    feature_mode: str = "union",
    word_ngram_max: int = 2,
    char_ngram_max: int = 5,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    imbalanced_intent: str = "transfer_pending",
    intent_keep_fraction: float = 1.0,
    label_noise_rate: float = 0.0,
    text_noise_rate: float = 0.0,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not hypothesis.strip():
        raise ValueError("hypothesis не может быть пустой")
    frame = make_smoke_data(
        samples_per_intent=samples_per_intent,
        random_state=data_seed,
    )
    if intent_keep_fraction < 1:
        frame = downsample_intent(
            frame,
            intent=imbalanced_intent,
            keep_fraction=intent_keep_fraction,
            random_state=data_seed + 101,
        )
    frame = validate_frame(frame)
    train, validation, test, _ = stratified_split(
        frame,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        random_state=split_seed,
    )
    train, noisy_train_labels = inject_label_noise(
        train,
        rate=label_noise_rate,
        random_state=data_seed + split_seed + 211,
    )
    validation, noisy_validation_tokens = inject_token_noise(
        validation,
        rate=text_noise_rate,
        random_state=data_seed + split_seed + 307,
    )
    test, noisy_test_tokens = inject_token_noise(
        test,
        rate=text_noise_rate,
        random_state=data_seed + split_seed + 401,
    )
    model = build_model(
        random_state=model_seed,
        regularization_c=regularization_c,
        feature_mode=feature_mode,
        word_ngram_max=word_ngram_max,
        char_ngram_max=char_ngram_max,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
    )
    model.fit(train["text"], train["intent"])
    validation_probabilities = model.predict_proba(validation["text"])
    threshold = select_abstention_threshold(
        validation["intent"].to_numpy(),
        validation_probabilities,
        model.classes_,
        minimum_coverage=minimum_coverage,
    )
    validation_metrics = classification_metrics(
        model,
        validation["text"],
        validation["intent"],
        abstention_threshold=threshold,
    )
    test_metrics = classification_metrics(
        model,
        test["text"],
        test["intent"],
        abstention_threshold=threshold,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "synthetic_banking_ticket_sensitivity",
        "hypothesis": hypothesis.strip(),
        "parameters": {
            "samples_per_intent": samples_per_intent,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "minimum_coverage": minimum_coverage,
            "regularization_c": regularization_c,
            "feature_mode": feature_mode,
            "word_ngram_max": word_ngram_max,
            "char_ngram_max": char_ngram_max,
            "calibration_method": calibration_method,
            "calibration_cv": calibration_cv,
            "imbalanced_intent": imbalanced_intent,
            "intent_keep_fraction": intent_keep_fraction,
            "label_noise_rate": label_noise_rate,
            "text_noise_rate": text_noise_rate,
        },
        "dataset": {
            "mode": "synthetic",
            "rows": len(frame),
            "class_counts": {
                str(label): int(count) for label, count in frame["intent"].value_counts().items()
            },
            "split_rows": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "feature_count": _feature_count(model),
            "noisy_train_labels": noisy_train_labels,
            "noisy_evaluation_tokens": noisy_validation_tokens + noisy_test_tokens,
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    if baseline is not None:
        baseline_test = baseline.get("test")
        baseline_dataset = baseline.get("dataset")
        if not isinstance(baseline_test, dict) or not isinstance(baseline_dataset, dict):
            raise ValueError("baseline не соответствует схеме эксперимента")
        baseline_calibration = baseline_test.get("calibration")
        if not isinstance(baseline_calibration, dict):
            raise ValueError("В baseline отсутствует диагностика калибровки")
        result["comparison"] = {
            "test_macro_f1_delta": test_metrics["macro_f1"] - baseline_test["macro_f1"],
            "test_top_3_accuracy_delta": (
                test_metrics["top_3_accuracy"] - baseline_test["top_3_accuracy"]
            ),
            "test_log_loss_delta": test_metrics["log_loss"] - baseline_test["log_loss"],
            "test_coverage_delta": test_metrics["coverage"] - baseline_test["coverage"],
            "test_brier_score_delta": (
                test_metrics["calibration"]["multiclass_brier_score"]
                - baseline_calibration["multiclass_brier_score"]
            ),
            "test_ece_delta": (
                test_metrics["calibration"]["expected_calibration_error"]
                - baseline_calibration["expected_calibration_error"]
            ),
            "threshold_delta": (
                test_metrics["abstention_threshold"] - baseline_test["abstention_threshold"]
            ),
            "feature_count_delta": _feature_count(model) - int(baseline_dataset["feature_count"]),
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--samples-per-intent", type=int, default=20)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-coverage", type=float, default=0.6)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["word", "char", "union"], default="union")
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
    )
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--imbalanced-intent", default="transfer_pending")
    parser.add_argument("--intent-keep-fraction", type=float, default=1.0)
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--text-noise-rate", type=float, default=0.0)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = run_experiment(
        hypothesis=args.hypothesis,
        samples_per_intent=args.samples_per_intent,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        minimum_coverage=args.minimum_coverage,
        regularization_c=args.regularization_c,
        feature_mode=args.feature_mode,
        word_ngram_max=args.word_ngram_max,
        char_ngram_max=args.char_ngram_max,
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv,
        imbalanced_intent=args.imbalanced_intent,
        intent_keep_fraction=args.intent_keep_fraction,
        label_noise_rate=args.label_noise_rate,
        text_noise_rate=args.text_noise_rate,
        baseline=baseline,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
