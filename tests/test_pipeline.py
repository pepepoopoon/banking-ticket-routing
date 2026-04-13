import joblib

from banking_ticket_routing.demo_data import make_smoke_data
from banking_ticket_routing.evaluate import evaluate
from banking_ticket_routing.io import MODEL_SCHEMA_VERSION
from banking_ticket_routing.service import RoutingService
from banking_ticket_routing.train import train


def test_end_to_end_core_service(tmp_path) -> None:
    data_path = tmp_path / "tickets.csv"
    artifact_dir = tmp_path / "artifacts"
    make_smoke_data().to_csv(data_path, index=False)

    metadata = train(data_path, artifact_dir, random_state=42, minimum_coverage=0.6)
    bundle = joblib.load(artifact_dir / "model.joblib")
    metrics = evaluate(artifact_dir / "model.joblib", artifact_dir / "test.csv")
    service = RoutingService.from_path(artifact_dir / "model.joblib")
    result = service.route("Where is the card that should be delivered", top_k=3)
    service.abstention_threshold = 1.0
    abstained_result = service.route("unclear request", top_k=3)

    assert metadata["calibration"].startswith("sigmoid")
    assert bundle["schema_version"] == MODEL_SCHEMA_VERSION
    assert bundle["intent_classes"] == bundle["model"].classes_.tolist()
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["top_3_accuracy"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert len(result["candidates"]) == 3
    assert result["intent"] is None or isinstance(result["intent"], str)
    assert abstained_result["abstained"] is True
    assert abstained_result["intent"] is None
    assert (artifact_dir / "split_manifest.csv").exists()
