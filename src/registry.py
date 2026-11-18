"""Реестр моделей MLflow: регистрация кандидата и продвижение в champion.

    python -m src.registry register     # версия из последнего запуска -> challenger
    python -m src.registry promote      # challenger лучше champion -> меняем алиас
    python -m src.registry status       # версии и их алиасы

Алиас champion указывает на версию, которую использует пакетное предсказание;
challenger - на кандидата, который ещё не прошёл сравнение.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import get_logger, load_params

log = get_logger("registry")

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def _client():
    from mlflow import MlflowClient

    return MlflowClient()


def _metric(client, version, name: str) -> float | None:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(name)


def register(model_name: str) -> str:
    import mlflow

    run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    candidates = [f"runs:/{run_id}/model"]
    if MODEL_URI_FILE.exists():
        candidates.insert(0, MODEL_URI_FILE.read_text(encoding="utf-8").strip())

    version, last_error = None, None
    for model_uri in candidates:
        try:
            version = mlflow.register_model(model_uri, model_name)
            break
        except Exception as exc:
            last_error = exc
            log.warning("не удалось зарегистрировать %s: %s", model_uri, exc)
    if version is None:
        raise RuntimeError("модель не зарегистрирована") from last_error

    _client().set_registered_model_alias(model_name, "challenger", version.version)
    log.info("зарегистрирована версия %s, алиас challenger", version.version)
    return version.version


def promote(model_name: str, metric: str, greater_is_better: bool = True) -> bool:
    client = _client()
    challenger = client.get_model_version_by_alias(model_name, "challenger")
    new_value = _metric(client, challenger, metric)
    if new_value is None:
        raise ValueError(f"у challenger v{challenger.version} нет метрики {metric}")

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        champion = None

    if champion is None:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
        log.info(
            "champion не был задан: версия %s становится champion", challenger.version
        )
        return True

    old_value = _metric(client, champion, metric)
    if old_value is None:
        better = True
    elif greater_is_better:
        better = new_value > old_value
    else:
        better = new_value < old_value
    log.info(
        "%s: challenger v%s = %.4f, champion v%s = %s -> %s",
        metric,
        challenger.version,
        new_value,
        champion.version,
        "нет" if old_value is None else f"{old_value:.4f}",
        "продвигаем" if better else "оставляем прежнего",
    )
    if better:
        client.set_registered_model_alias(model_name, "champion", challenger.version)
    return better


def status(model_name: str, metric: str) -> list[dict]:
    """Версии модели с алиасами и значением основной метрики."""
    client = _client()
    aliases = dict(client.get_registered_model(model_name).aliases)
    versions = client.search_model_versions(f"name='{model_name}'")
    rows = []
    for version in sorted(versions, key=lambda v: int(v.version)):
        names = [a for a, v in aliases.items() if str(v) == str(version.version)]
        rows.append(
            {
                "version": int(version.version),
                "aliases": ", ".join(sorted(names)) or "-",
                metric: _metric(client, version, metric),
            }
        )
    return rows


def main() -> None:
    params = load_params()["mlflow"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "promote", "status"])
    parser.add_argument("--model-name", default=params["model_name"])
    parser.add_argument("--metric", default=params["primary_metric"])
    args = parser.parse_args()

    if args.action == "register":
        register(args.model_name)
    elif args.action == "promote":
        promote(args.model_name, args.metric)
    else:
        for row in status(args.model_name, args.metric):
            log.info("%s", row)


if __name__ == "__main__":
    main()
