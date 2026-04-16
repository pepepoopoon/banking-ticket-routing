# Маршрутизация банковских обращений

## Описание задачи

Многоклассовая классификация текста обращения в один из fine-grained intent-классов
BANKING77 с возможностью отказаться от автоматической маршрутизации.

## Цель проекта

Создать воспроизводимый классический NLP baseline, который показывает top-k подсказки,
калиброванную уверенность и передаёт сомнительные обращения человеку.

## Архитектура решения

После проверки схемы данные стратифицированно делятся на train/validation/test.
`FeatureUnion` объединяет word TF-IDF (1–2-граммы) и char TF-IDF (3–5-граммы).
Логистическая регрессия калибруется sigmoid-методом только внутри train; порог abstention
выбирается на validation при минимальном покрытии. Test не участвует в настройке.
Экспериментальный API позволяет независимо менять регуляризацию, семейство признаков,
границы n-грамм, метод калибровки и число калибровочных фолдов.

## Структура каталогов

```text
src/banking_ticket_routing/  # данные, модель, сервис и CLI
data/                        # схема и синтетический smoke CSV
tests/                       # unit и интеграционные тесты
.github/workflows/ci.yml     # CI без загрузки данных
```

## Используемые технологии

Python 3.11, pandas, NumPy, scikit-learn, joblib, pytest и Ruff. FastAPI/uvicorn доступны
только в optional extra `api`; ядро и сервисный слой от них не зависят.

## Требования к окружению

Python 3.11 или 3.12. Для core CLI сетевой доступ не требуется.

## Установка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# опционально для HTTP: python -m pip install -e ".[dev,api]"
```

## Подготовка данных

Контракт и источник приведены в [`data/README.md`](data/README.md). Smoke-набор:
`banking-generate-smoke --output data/smoke.csv`.
Для learning curve генератор принимает `--samples-per-intent` и `--seed`; базовые
формулировки дополняются детерминированными нейтральными контекстами обращения.

## Запуск обучения

```bash
banking-train --data data/smoke.csv --output-dir artifacts/smoke
```

## Запуск оценки

```bash
banking-evaluate --model artifacts/smoke/model.joblib \
  --data artifacts/smoke/test.csv --output artifacts/smoke/test_metrics.json
```

## Запуск инференса

```bash
banking-predict --model artifacts/smoke/model.joblib --text "Where is my new card?" --top-k 3
```

## Инженерные эксперименты

`make experiment` обучает модель на отдельной синтетической выборке, выбирает порог
только по validation и сохраняет параметры, размерность TF-IDF, калибровочные метрики,
per-intent качество и test-результат в JSON.

Опциональный HTTP-слой после установки extra:

```bash
BANKING_MODEL_PATH=artifacts/smoke/model.joblib \
  uvicorn banking_ticket_routing.api:create_app --factory
```

## Метрики

Сохраняются macro-F1, top-3 accuracy, multiclass log-loss и Brier score, ECE,
confusion matrix, per-intent метрики, coverage, abstention rate и selective accuracy
на принятых ответах. Порог выбирается только по validation.

## Тестирование

`pytest -q` проверяет схему, split, калибровку/abstention и полный core-сценарий;
`ruff check .` — линтинг. FastAPI и интернет для тестов не нужны.

## Ограничения

Уверенность не является гарантией корректности, а калибровка может деградировать при
дрейфе. Abstention означает ручную очередь, не отказ клиенту. Модель не принимает
финансовых решений и требует мониторинга качества по каждому intent.
Для контролируемой проверки устойчивости доступны дисбаланс одного intent, шум train-меток
и детерминированные опечатки только на validation/test.

## Полученные результаты

Реальные метрики BANKING77 не заявляются. CLI рассчитывает значения на переданном
наборе; smoke-результат подтверждает лишь исполнимость конвейера.

## Источник и лицензия данных

[BANKING77 от PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
доступен по CC BY 4.0 и требует атрибуции. Репозиторий не содержит его строк.

## Статус проекта

Завершён core baseline и необязательная FastAPI-обёртка; production rollout не заявлен.
