# MLOps ЛР1 — вариант 5, California Housing

## Постановка

Регрессия: предсказание медианной стоимости жилья (MedHouseVal, единицы 100 000 USD).

- v1: северные районы (Latitude >= 35.0) — обучающий срез.
- v2: все районы.
- Поток для ЛР4: южные районы (Latitude < 35.0).

## Развёртывание

    pip install uv
    uv lock
    uv export --frozen --no-hashes -o requirements.lock.txt
    pip install -r requirements.lock.txt
    dvc pull
    dvc repro

## Пайплайн

prepare -> featurize -> train -> evaluate, описан в dvc.yaml.

- Граф: dvc dag
- Метрики: reports/metrics.json

## Результаты

| Версия | R2 | MAE | RMSE | n_test |
|---|---|---|---|---|
| v1 (north) | 0.8529 | 0.292 | 0.454 | 1870 |
| v2 (all) | 0.8300 | 0.318 | 0.478 | 4128 |

R2 на v2 ниже, потому что в полный набор входят южные районы с другим распределением цен; модель, обученная на северном срезе, на них обобщается хуже.

## Версии данных

- data-v1 — северные районы
- data-v2 — все районы

Возврат к v1:

    git checkout data-v1 && dvc checkout && dvc repro

## Окружение

Зафиксировано в uv.lock, разворачивается через uv export и pip install.

## Сценарий смены версии данных

1. v1 (north): `python -m src.ingest --version north`, `dvc add`, `dvc push`, `dvc repro` → r2 0.8529, n_test 1870. Зафиксировано тегом `data-v1`.
2. v2 (all): `python -m src.ingest --version all`, `dvc add`, `dvc push`, `dvc repro` → r2 0.8300, n_test 4128. Зафиксировано тегом `data-v2`.
3. Возврат: `git checkout data-v1 && dvc checkout && dvc repro` → снова r2 0.8529, n_test 1870.

## Сравнение моделей (дополнительное задание)

Модель задаётся в `params.yaml` полем `train.model`. Код `src/train.py` не менялся — `make_model` выбирает класс по имени.

| Модель | R2 | MAE | RMSE | n_test |
|---|---|---|---|---|
| gradient_boosting (базовая) | 0.8300 | 0.318 | 0.478 | 4128 |
| random_forest | 0.5995 | 0.540 | 0.734 | 4128 |

RandomForest с `n_estimators=500`, `max_depth=4`, `min_samples_leaf=2` показал R2 заметно ниже, чем GradientBoosting: бустинг последовательно исправляет ошибки предыдущих деревьев и лучше улавливает нелинейности, тогда как случайный лес с малой глубиной усредняет независимые деревья и на этом наборе обобщает слабее. При этом RandomForest не прошёл порог `evaluate.min_r2: 0.60` (R2 = 0.5995), и этап `evaluate` завершился с ошибкой — это ожидаемое поведение проверки качества.

Переключение модели выполняется только правкой `params.yaml`; `dvc repro` пересчитывает `train` и `evaluate`, не трогая `prepare` и `featurize`.
