"""Strength-surrogate helpers and group-aware uncertainty utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .data import FEATURE_COLUMNS, STRENGTH_COLUMN


def make_strength_model(
    seed: int, model_config: Dict[str, object]
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(model_config["n_estimators"]),
        max_depth=model_config.get("max_depth"),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        max_features=float(model_config["max_features"]),
        n_jobs=int(model_config.get("n_jobs", -1)),
        random_state=int(seed),
    )


def group_folds(groups: Iterable[str], n_splits: int, seed: int):
    groups = np.asarray(list(groups))
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(groups))
    return splitter.split(dummy, groups=groups)


def cross_fitted_predictions(
    frame: pd.DataFrame,
    model_config: Dict[str, object],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(
        group_folds(frame["mix_id"], n_splits=n_splits, seed=seed)
    ):
        model = make_strength_model(seed + 1009 * (fold + 1), model_config)
        model.fit(
            frame.iloc[train_idx][FEATURE_COLUMNS],
            frame.iloc[train_idx][STRENGTH_COLUMN],
        )
        predictions[test_idx] = model.predict(frame.iloc[test_idx][FEATURE_COLUMNS])
    if np.isnan(predictions).any():
        raise RuntimeError("Cross-fitted predictions are incomplete")
    return predictions


def conformal_quantile(errors: Iterable[float], alpha: float) -> float:
    values = np.sort(np.abs(np.asarray(list(errors), dtype=float)))
    if len(values) == 0:
        raise ValueError("At least one calibration error is required")
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(values[rank - 1])


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Tuple[float, float, float]:
    error = np.asarray(actual) - np.asarray(predicted)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(predicted - actual))
    return rmse, mae, bias
