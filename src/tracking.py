"""Работа с MLflow. Трекинг включается только при заданном MLFLOW_TRACKING_URI.

В ЛР1 переменная не задана, и этапы работают без трекинга; начиная с ЛР2 она
указывает на сервер MLflow, встроенный в репозиторий DagsHub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def start_run(experiment: str, params: dict[str, Any]):
    """Открывает запуск, записывает параметры и сохраняет его идентификатор."""
    if not enabled():
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    run = mlflow.start_run()
    mlflow.log_params(params)
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.info.run_id, encoding="utf-8")
    return run


def log_train_metrics(metrics: dict[str, float]) -> None:
    """Дописывает метрики обучения в запуск, открытый в этом же процессе."""
    if not enabled():
        return
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)


def log_metrics(metrics: dict[str, float]) -> None:
    """Дописывает метрики в запуск, открытый на этапе train."""
    if not enabled() or not RUN_ID_FILE.exists():
        return
    import mlflow

    run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)


def log_model(model, features: list[str], example) -> None:
    """Сохраняет модель вместе с сигнатурой входа и запоминает её адрес.

    Адрес модели нужен реестру: в MLflow 3 он выглядит как models:/m-...,
    в MLflow 2 - как runs:/<run_id>/model. Запись в файл избавляет от
    различий между версиями сервера.
    """
    if not enabled():
        return
    import cloudpickle
    import mlflow
    import pandas
    import sklearn
    from mlflow.models import infer_signature

    mlflow.set_tag("features", ",".join(features))
    signature = infer_signature(example, model.predict(example))
    # Окружение модели задаётся явно - по тому же принципу, что и lock-файл
    # проекта: тот, кто загрузит модель, получит те же версии библиотек.
    requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ]
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example.head(3),
        # Деревья решений MLflow по умолчанию сохраняет через skops и требует
        # явно перечислить доверенные типы; cloudpickle работает одинаково
        # на всех версиях сервера.
        serialization_format="cloudpickle",
        pip_requirements=requirements,
    )
    MODEL_URI_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_URI_FILE.write_text(info.model_uri, encoding="utf-8")
