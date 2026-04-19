from banking_ticket_routing.experiment import run_experiment


def test_experiment_records_reproducible_metrics() -> None:
    result = run_experiment(
        hypothesis="Проверить тестовый сценарий",
        samples_per_intent=12,
        data_seed=3,
        split_seed=5,
        model_seed=7,
        calibration_cv=2,
        label_noise_rate=0.1,
        text_noise_rate=0.1,
    )

    assert result["dataset"]["rows"] == 60
    assert sum(result["dataset"]["split_rows"].values()) == 60
    assert result["dataset"]["feature_count"] > 0
    assert result["dataset"]["noisy_train_labels"] > 0
    assert 0 <= result["test"]["macro_f1"] <= 1
    assert 0 <= result["test"]["coverage"] <= 1
    assert result["test"]["calibration"]["multiclass_brier_score"] >= 0


def test_experiment_compares_metrics_with_baseline() -> None:
    baseline = run_experiment(
        hypothesis="Baseline",
        samples_per_intent=12,
        calibration_cv=2,
    )
    candidate = run_experiment(
        hypothesis="Сценарий с шумом",
        samples_per_intent=12,
        calibration_cv=2,
        text_noise_rate=0.2,
        baseline=baseline,
    )

    assert set(candidate["comparison"]) == {
        "test_macro_f1_delta",
        "test_top_3_accuracy_delta",
        "test_log_loss_delta",
        "test_coverage_delta",
        "test_brier_score_delta",
        "test_ece_delta",
        "threshold_delta",
        "feature_count_delta",
    }
