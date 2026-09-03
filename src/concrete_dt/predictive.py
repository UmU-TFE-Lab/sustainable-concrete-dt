"""Multi-output RF/XGBoost/DNN benchmarking and compute-use accounting."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BenchmarkResult:
    """Aggregate benchmark tables and fitted estimators kept in memory."""

    metrics: pd.DataFrame
    resources: pd.DataFrame
    models: Dict[str, BaseEstimator]
    train_indices: np.ndarray
    test_indices: np.ndarray


def _random_forest(config: Mapping[str, object], seed: int) -> BaseEstimator:
    base = RandomForestRegressor(
        n_estimators=int(config.get("n_estimators", 100)),
        max_depth=config.get("max_depth"),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        max_features=config.get("max_features", 1.0),
        random_state=seed,
        n_jobs=int(config.get("n_jobs", 1)),
    )
    # The wrapper matches the independent-output formulation stated in the paper.
    return MultiOutputRegressor(base, n_jobs=1)


def _xgboost(config: Mapping[str, object], seed: int) -> BaseEstimator:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised in a full environment
        raise ImportError(
            "XGBoost was requested. Install the package's 'full' optional dependencies."
        ) from exc
    base = XGBRegressor(
        n_estimators=int(config.get("n_estimators", 100)),
        objective="reg:squarederror",
        max_depth=int(config.get("max_depth", 6)),
        learning_rate=float(config.get("learning_rate", 0.1)),
        subsample=float(config.get("subsample", 1.0)),
        colsample_bytree=float(config.get("colsample_bytree", 1.0)),
        reg_alpha=float(config.get("reg_alpha", 0.0)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=int(config.get("n_jobs", 1)),
        verbosity=0,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def _dnn(config: Mapping[str, object], seed: int) -> BaseEstimator:
    hidden = tuple(int(value) for value in config.get("hidden_layer_sizes", [64, 64]))
    network = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=float(config.get("learning_rate", 0.001)),
        beta_1=float(config.get("beta_1", 0.9)),
        beta_2=float(config.get("beta_2", 0.999)),
        epsilon=float(config.get("epsilon", 1e-8)),
        batch_size=int(config.get("batch_size", 16)),
        max_iter=int(config.get("epochs", 1000)),
        early_stopping=bool(config.get("early_stopping", False)),
        random_state=seed,
    )
    return Pipeline([("feature_scaler", StandardScaler()), ("dnn", network)])


def build_predictive_model(
    name: str,
    config: Mapping[str, object],
    seed: int,
) -> BaseEstimator:
    """Construct one of the three predictive models described in the paper."""
    normalized = name.lower()
    if normalized == "random_forest":
        return _random_forest(config, seed)
    if normalized == "xgboost":
        return _xgboost(config, seed)
    if normalized in {"dnn", "deep_neural_network"}:
        return _dnn(config, seed)
    raise ValueError(f"Unsupported predictive model: {name}")


def _start_codecarbon(enabled: bool, country_iso_code: str):
    if not enabled:
        return None, "disabled"
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        return None, "codecarbon_not_installed"
    tracker = OfflineEmissionsTracker(
        country_iso_code=country_iso_code,
        save_to_file=False,
        log_level="error",
    )
    tracker.start()
    return tracker, "measured"


def _stop_codecarbon(tracker) -> tuple[float, float]:
    if tracker is None:
        return float("nan"), float("nan")
    # Some CodeCarbon releases return ``None`` when no emission sample has
    # been finalized (for example, after a very short fit).  Keep the missing
    # value explicit rather than failing or manufacturing a zero.
    stopped = tracker.stop()
    emissions_kg = float(stopped) if stopped is not None else float("nan")
    data = getattr(tracker, "final_emissions_data", None)
    energy_kwh = float(getattr(data, "energy_consumed", float("nan")))
    return energy_kwh, emissions_kg


def _fit_with_measurement(
    estimator: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    track_energy: bool,
    country_iso_code: str,
) -> Dict[str, float | str]:
    tracker, status = _start_codecarbon(track_energy, country_iso_code)
    started = perf_counter()
    try:
        estimator.fit(x_train, y_train)
    finally:
        elapsed = perf_counter() - started
        energy_kwh, emissions_kg = _stop_codecarbon(tracker)

    if status == "measured" and not (
        np.isfinite(energy_kwh) and np.isfinite(emissions_kg)
    ):
        status = "measurement_incomplete"
    energy_j = energy_kwh * 3.6e6 if np.isfinite(energy_kwh) else float("nan")
    average_power = energy_j / elapsed if energy_j > 0.0 else float("nan")
    samples_per_joule = len(x_train) / energy_j if energy_j > 0.0 else float("nan")
    power_only_ratio = 1.0 / average_power if average_power > 0.0 else float("nan")
    return {
        "measurement_status": status,
        "training_seconds": float(elapsed),
        "energy_kwh": energy_kwh,
        "emissions_kgco2e": emissions_kg,
        "average_power_w": average_power,
        "training_samples_per_joule": samples_per_joule,
        "power_only_ratio_per_w": power_only_ratio,
    }


def benchmark_multioutput_models(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    config: Mapping[str, object],
    seed: int,
) -> BenchmarkResult:
    """Fit and compare the manuscript's three multi-output regressors.

    Only aggregate metrics and resource measurements are returned as tables.
    Predictions remain transient and are intentionally not part of the result.
    """
    x = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    y = frame.loc[:, target_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    if len(frame) < 3:
        raise ValueError("Predictive benchmarking requires at least three observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Predictive features and targets must be finite")
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(config.get("test_fraction", 0.2)),
        random_state=seed,
        shuffle=True,
    )
    requested = list(config.get("models", ["random_forest", "xgboost", "dnn"]))
    strict = bool(config.get("require_all_models", True))
    model_configs = config.get("model_parameters", {})
    track_energy = bool(config.get("track_energy", True))
    country = str(config.get("country_iso_code", "SWE"))

    models: Dict[str, BaseEstimator] = {}
    metric_rows = []
    resource_rows = []
    for offset, name in enumerate(requested):
        try:
            estimator = build_predictive_model(
                name,
                model_configs.get(name, {}),
                seed + 1009 * offset,
            )
        except ImportError:
            if strict:
                raise
            warnings.warn(f"Skipping unavailable optional model: {name}", stacklevel=2)
            resource_rows.append(
                {
                    "model": name,
                    "measurement_status": "model_dependency_not_installed",
                    "training_seconds": float("nan"),
                    "energy_kwh": float("nan"),
                    "emissions_kgco2e": float("nan"),
                    "average_power_w": float("nan"),
                    "training_samples_per_joule": float("nan"),
                    "power_only_ratio_per_w": float("nan"),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
            continue

        measurement = _fit_with_measurement(
            estimator,
            x[train_idx],
            y[train_idx],
            track_energy,
            country,
        )
        predicted = np.asarray(estimator.predict(x[test_idx]), dtype=float)
        models[name] = estimator
        for column_index, target in enumerate(target_columns):
            actual_column = y[test_idx, column_index]
            predicted_column = predicted[:, column_index]
            mse = float(mean_squared_error(actual_column, predicted_column))
            metric_rows.append(
                {
                    "model": name,
                    "output": target,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "r2": float(r2_score(actual_column, predicted_column)),
                    "n_test": int(len(test_idx)),
                }
            )
        resource_rows.append(
            {
                "model": name,
                **measurement,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    return BenchmarkResult(
        metrics=pd.DataFrame(metric_rows),
        resources=pd.DataFrame(resource_rows),
        models=models,
        train_indices=train_idx,
        test_indices=test_idx,
    )
