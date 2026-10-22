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
