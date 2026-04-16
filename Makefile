.PHONY: install install-api lint test smoke train evaluate predict experiment serve

install:
	python -m pip install -e ".[dev]"

install-api:
	python -m pip install -e ".[dev,api]"

lint:
	ruff check .

test:
	pytest -q

smoke:
	banking-generate-smoke --output data/smoke.csv

train:
	banking-train --data data/smoke.csv --output-dir artifacts/smoke

evaluate:
	banking-evaluate --model artifacts/smoke/model.joblib --data artifacts/smoke/test.csv --output artifacts/smoke/test_metrics.json

predict:
	banking-predict --model artifacts/smoke/model.joblib --text "Where is my new card?"

experiment:
	banking-experiment --output experiments/results/example.json --hypothesis "Проверить сценарий"

serve:
	BANKING_MODEL_PATH=artifacts/smoke/model.joblib uvicorn banking_ticket_routing.api:create_app --factory
