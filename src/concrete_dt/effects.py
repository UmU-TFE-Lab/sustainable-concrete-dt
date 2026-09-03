"""Adjusted associations, path models, DML, and model-based interventions.

All estimates use observational mix-design data. They are therefore labelled
model-based conditional effects under stated identification assumptions, not
experimentally identified causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, train_test_split


IDENTIFICATION_NOTE = (
    "Observational model-based conditional effect; a causal interpretation "
    "requires consistency, positivity, correct temporal ordering, and no "
    "unmeasured confounding after adjustment."
)


@dataclass
class OLSResult:
    coefficients: pd.DataFrame
    predictions: np.ndarray
    residuals: np.ndarray
    r2: float


def fit_ols(
    frame: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    confidence: float = 0.95,
) -> OLSResult:
    """Fit OLS with classical standard errors and a Moore-Penrose inverse."""
    columns = list(predictors) + [outcome]
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").dropna()
    x = data.loc[:, predictors].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS inputs must be finite after missing-value removal")
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ beta
    residual = y - predicted
    rank = int(np.linalg.matrix_rank(design))
    dof = len(y) - rank
    if dof <= 0:
        raise ValueError("OLS has no residual degrees of freedom")
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistic = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0.0,
    )
    p_value = 2.0 * stats.t.sf(np.abs(t_statistic), dof)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, dof))
    total_sum_squares = float(np.square(y - y.mean()).sum())
    if total_sum_squares <= np.finfo(float).eps:
        raise ValueError(f"OLS outcome '{outcome}' is constant")
    r2 = 1.0 - float(np.square(residual).sum()) / total_sum_squares
    terms = ["intercept"] + list(predictors)
    coefficient_table = pd.DataFrame(
        {
            "outcome": outcome,
            "term": terms,
            "estimate": beta,
            "standard_error": standard_error,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "ci_lower": beta - critical * standard_error,
            "ci_upper": beta + critical * standard_error,
            "n": int(len(y)),
            "residual_dof": dof,
            "r2": r2,
            "interpretation": "adjusted_linear_association",
        }
    )
    return OLSResult(coefficient_table, predicted, residual, r2)


def fit_mlr_models(
    frame: pd.DataFrame,
    outcomes: Sequence[str],
    predictors: Sequence[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Fit one adjusted multiple-linear-regression model per output."""
    tables = [
        fit_ols(frame, outcome, predictors, confidence).coefficients
        for outcome in outcomes
    ]
    result = pd.concat(tables, ignore_index=True)
    result["identification_note"] = IDENTIFICATION_NOTE
    return result


def fit_observed_path_model(
    frame: pd.DataFrame,
    equations: Mapping[str, Sequence[str]],
    covariance_pairs: Sequence[Sequence[str]] = (),
    confidence: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the manuscript's observed-variable path equations.

    This is structured path regression without latent variables or a measurement
    model. Consequently, latent-SEM fit indices such as CFI/TLI/RMSEA are not
    manufactured or reported by this implementation.
    """
    path_tables = []
    for outcome, predictors in equations.items():
        table = fit_ols(frame, outcome, predictors, confidence).coefficients
        table["model_type"] = "observed_variable_path_regression"
        path_tables.append(table)

    covariance_rows = []
    for pair in covariance_pairs:
        if len(pair) != 2:
            raise ValueError("Each covariance pair must contain exactly two variables")
        left, right = pair
        values = frame[[left, right]].apply(pd.to_numeric, errors="raise").dropna()
        correlation = stats.pearsonr(values[left], values[right])
        covariance_rows.append(
            {
                "variable_x": left,
                "variable_y": right,
                "covariance": float(values[left].cov(values[right], ddof=1)),
                "pearson_r": float(correlation.statistic),
                "association_p_value": float(correlation.pvalue),
                "n": int(len(values)),
                "interpretation": "modeled_exogenous_covariance",
            }
        )
    return pd.concat(path_tables, ignore_index=True), pd.DataFrame(covariance_rows)


def _nuisance_model(
    config: Mapping[str, object],
    seed: int,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(config.get("n_estimators", 100)),
        max_depth=config.get("max_depth"),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        max_features=config.get("max_features", 1.0),
        n_jobs=int(config.get("n_jobs", 1)),
        random_state=seed,
    )


def partially_linear_dml(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: Sequence[str],
    nuisance_config: Mapping[str, object],
    folds: int = 2,
    seed: int = 17,
    group_column: Optional[str] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate a constant partially-linear effect by cross-fitted orthogonalization."""
    columns = list(dict.fromkeys(list(controls) + [treatment, outcome]))
    if group_column:
        columns.append(group_column)
    data = frame.loc[:, columns].dropna().reset_index(drop=True)
    x = data.loc[:, controls].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    treatment_values = pd.to_numeric(data[treatment], errors="raise").to_numpy(
        dtype=float
    )
    outcome_values = pd.to_numeric(data[outcome], errors="raise").to_numpy(dtype=float)
    if treatment in controls:
        raise ValueError("The focal treatment cannot also be a control")
    if not controls:
        raise ValueError("DML requires at least one observed control variable")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(treatment_values).all()
        or not np.isfinite(outcome_values).all()
    ):
        raise ValueError("DML inputs must be finite after missing-value removal")
    if not 2 <= int(folds) <= len(data):
        raise ValueError("DML folds must satisfy 2 <= folds <= n_observations")

    if group_column:
        groups = data[group_column].to_numpy()
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x, groups=groups)
        split_type = "group_kfold"
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x)
        split_type = "kfold"

    predicted_outcome = np.full(len(data), np.nan)
    predicted_treatment = np.full(len(data), np.nan)
    for fold, (train_index, test_index) in enumerate(splits):
        outcome_model = clone(
            _nuisance_model(nuisance_config, seed + 1009 * (fold + 1))
        )
        treatment_model = clone(
            _nuisance_model(nuisance_config, seed + 2003 * (fold + 1))
        )
        outcome_model.fit(x[train_index], outcome_values[train_index])
        treatment_model.fit(x[train_index], treatment_values[train_index])
        predicted_outcome[test_index] = outcome_model.predict(x[test_index])
        predicted_treatment[test_index] = treatment_model.predict(x[test_index])
    if np.isnan(predicted_outcome).any() or np.isnan(predicted_treatment).any():
        raise RuntimeError("Cross-fitting did not produce a prediction for every row")

    residual_outcome = outcome_values - predicted_outcome
    residual_treatment = treatment_values - predicted_treatment
    denominator = float(np.sum(np.square(residual_treatment)))
    if denominator <= np.finfo(float).eps:
        raise ValueError(f"No residual treatment variation remains for {treatment}")
    estimate = float(np.sum(residual_treatment * residual_outcome) / denominator)

    score = residual_treatment * (residual_outcome - estimate * residual_treatment)
    jacobian = float(np.mean(np.square(residual_treatment)))
    standard_error = float(np.sqrt(np.mean(np.square(score)) / len(data)) / jacobian)
    critical = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    z_statistic = estimate / standard_error if standard_error > 0.0 else float("nan")
    p_value = 2.0 * stats.norm.sf(abs(z_statistic))
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "treatment": treatment,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_lower": estimate - critical * standard_error,
                "ci_upper": estimate + critical * standard_error,
                "z_statistic": z_statistic,
                "p_value": p_value,
                "n": int(len(data)),
                "folds": int(folds),
                "split_type": split_type,
                "nuisance_model": "random_forest",
                "interpretation": "orthogonalized_conditional_effect",
                "identification_note": IDENTIFICATION_NOTE,
            }
        ]
    )


def fit_dml_grid(
    frame: pd.DataFrame,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    nuisance_config: Mapping[str, object],
    folds: int,
    seed: int,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Treat each material/age variable in turn and adjust for the remainder."""
    rows = []
    for outcome in outcomes:
        for treatment in treatments:
            controls = [value for value in treatments if value != treatment]
            rows.append(
                partially_linear_dml(
                    frame=frame,
                    outcome=outcome,
                    treatment=treatment,
                    controls=controls,
                    nuisance_config=nuisance_config,
                    folds=folds,
                    seed=seed,
                    group_column=group_column,
                )
            )
    return pd.concat(rows, ignore_index=True)


def intervention_scenarios(
    variables: Sequence[str],
    relative_changes: Sequence[float],
) -> list[Dict[str, object]]:
    """Create the manuscript's one-at-a-time -30% to +30% scenarios."""
    return [
        {
            "id": f"{variable}_{change:+.0%}",
            "changes": {variable: float(change)},
        }
        for variable in variables
        for change in relative_changes
    ]


def model_based_interventions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    outcome_columns: Sequence[str],
    scenarios: Sequence[Mapping[str, object]],
    subgroup_features: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 17,
    bounds: Optional[Mapping[str, Sequence[float]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate ATE and explicit subgroup CATE summaries with linear models.

    The comparison is intervention prediction minus model-predicted baseline for
    the same test profile. A linear model is additive, so combined scenarios do
    not estimate statistical interactions unless interaction terms are supplied
    explicitly in ``feature_columns``.
    """
    indices = np.arange(len(frame))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )
    x = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
    y = frame.loc[:, outcome_columns].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(x.to_numpy(dtype=float)).all()
        or not np.isfinite(y.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Intervention features and outcomes must be finite")
    model = LinearRegression().fit(x.iloc[train_index], y.iloc[train_index])
    test_x = x.iloc[test_index].reset_index(drop=True)
    baseline = np.asarray(model.predict(test_x), dtype=float)
    ate_rows = []
    cate_rows = []

    subgroup_codes: Dict[str, pd.Series] = {}
    for feature in subgroup_features:
        if feature not in frame.columns:
            raise ValueError(f"Unknown subgroup feature: {feature}")
        values = pd.to_numeric(
            frame.iloc[test_index][feature], errors="raise"
        ).reset_index(drop=True)
        ranked = values.rank(method="first")
        subgroup_codes[feature] = pd.qcut(
            ranked,
            q=3,
            labels=["low", "medium", "high"],
        )

    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        changes = dict(scenario["changes"])
        unknown = set(changes).difference(feature_columns)
        if unknown:
            raise ValueError(
                f"Intervention references unknown features: {sorted(unknown)}"
            )
        modified = test_x.copy()
        clipped = np.zeros(len(modified), dtype=bool)
        for feature, relative_change in changes.items():
            modified[feature] *= 1.0 + float(relative_change)
            if bounds and feature in bounds:
                lower, upper = bounds[feature]
                before = modified[feature].to_numpy(copy=True)
                modified[feature] = modified[feature].clip(float(lower), float(upper))
                clipped |= before != modified[feature].to_numpy()
        intervention_prediction = np.asarray(model.predict(modified), dtype=float)
        individual_effect = intervention_prediction - baseline

        for output_index, outcome in enumerate(outcome_columns):
            values = individual_effect[:, output_index]
            standard_error = (
                float(stats.sem(values)) if len(values) > 1 else float("nan")
            )
            critical = (
                float(stats.t.ppf(0.975, len(values) - 1))
                if len(values) > 1
                else float("nan")
            )
            mean_effect = float(np.mean(values))
            ate_rows.append(
                {
                    "scenario": scenario_id,
                    "changes": ";".join(
                        f"{key}={value:+.3f}" for key, value in sorted(changes.items())
                    ),
                    "outcome": outcome,
                    "ate": mean_effect,
                    "standard_error": standard_error,
                    "ci_lower": mean_effect - critical * standard_error,
                    "ci_upper": mean_effect + critical * standard_error,
                    "n_test": int(len(values)),
                    "clipped_fraction": float(clipped.mean()),
                    "baseline": "model_prediction_same_profile",
                    "interaction_capability": (
                        "additive_only_without_explicit_interaction_terms"
                    ),
                    "identification_note": IDENTIFICATION_NOTE,
                }
            )
            for feature, groups in subgroup_codes.items():
                for group_name in ("low", "medium", "high"):
                    mask = np.asarray(groups == group_name)
                    group_values = values[mask]
                    cate_rows.append(
                        {
                            "scenario": scenario_id,
                            "outcome": outcome,
                            "subgroup_feature": feature,
                            "subgroup": group_name,
                            "cate": float(np.mean(group_values)),
                            "n": int(len(group_values)),
                            "identification_note": IDENTIFICATION_NOTE,
                        }
                    )
    return pd.DataFrame(ate_rows), pd.DataFrame(cate_rows)
