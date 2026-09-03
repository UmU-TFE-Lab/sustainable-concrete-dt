"""Retrospective empirical-Bayes material-state updating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .data import (
    ENERGY_COLUMN,
    FEATURE_COLUMNS,
    GWP_COLUMN,
    MATERIAL_TOTAL_COLUMN,
    STRENGTH_COLUMN,
)
from .modeling import (
    conformal_quantile,
    cross_fitted_predictions,
    group_folds,
    make_strength_model,
    regression_metrics,
)


@dataclass(frozen=True)
class VarianceComponents:
    tau2: float
    sigma2: float

    def shrinkage(self, count: int) -> float:
        if count <= 0 or self.tau2 <= 0.0:
            return 0.0
        return float(self.tau2 / (self.tau2 + self.sigma2 / count))


def estimate_variance_components(residual_frame: pd.DataFrame) -> VarianceComponents:
    grouped = residual_frame.groupby("mix_id")["residual"]
    sizes = grouped.size().to_numpy(dtype=float)
    means = grouped.mean().to_numpy(dtype=float)
    n_groups = len(sizes)
    n_total = int(sizes.sum())
    if n_groups < 2 or n_total <= n_groups:
        variance = float(np.var(residual_frame["residual"], ddof=1))
        return VarianceComponents(tau2=0.0, sigma2=max(variance, 1e-9))

    grand_mean = float(np.average(means, weights=sizes))
    within_ss = 0.0
    for _, group in residual_frame.groupby("mix_id"):
        within_ss += float(
            np.square(group["residual"] - group["residual"].mean()).sum()
        )
    ms_within = within_ss / (n_total - n_groups)
    between_ss = float(np.sum(sizes * np.square(means - grand_mean)))
    ms_between = between_ss / (n_groups - 1)
    n0 = (n_total - float(np.sum(np.square(sizes))) / n_total) / (n_groups - 1)
    tau2 = max((ms_between - ms_within) / max(n0, 1e-12), 0.0)
    return VarianceComponents(tau2=float(tau2), sigma2=float(max(ms_within, 1e-9)))


def _trajectory_records(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    early_ages: Sequence[int],
    target_ages: Sequence[int],
    components: VarianceComponents,
) -> List[Dict[str, float]]:
    work = frame[["mix_id", "age", STRENGTH_COLUMN]].copy()
    work["prediction"] = predictions
    work["residual"] = work[STRENGTH_COLUMN] - work["prediction"]
    records: List[Dict[str, float]] = []
    for mix_id, group in work.groupby("mix_id"):
        early = group[group["age"].isin(early_ages)]
        if early.empty:
            continue
        raw_offset = float(early["residual"].mean())
        shrinkage = components.shrinkage(len(early))
        eb_offset = shrinkage * raw_offset
        for target_age in target_ages:
            target = group[group["age"] == target_age]
            if target.empty:
                continue
            row = target.iloc[0]
            records.append(
                {
                    "mix_id": mix_id,
                    "target_age": int(target_age),
                    "actual": float(row[STRENGTH_COLUMN]),
                    "static_prediction": float(row["prediction"]),
                    "raw_prediction": float(row["prediction"] + raw_offset),
                    "eb_prediction": float(row["prediction"] + eb_offset),
                    "early_observation_count": int(len(early)),
                    "raw_offset": raw_offset,
                    "eb_offset": eb_offset,
                    "shrinkage": shrinkage,
                }
            )
    return records


def _calibration_quantiles(
    inner_frame: pd.DataFrame,
    inner_predictions: np.ndarray,
    components: VarianceComponents,
    early_ages: Sequence[int],
    target_ages: Sequence[int],
    alpha: float,
) -> Dict[Tuple[str, int], float]:
    records = pd.DataFrame(
        _trajectory_records(
            inner_frame, inner_predictions, early_ages, target_ages, components
        )
    )
    quantiles: Dict[Tuple[str, int], float] = {}
    residuals = inner_frame[STRENGTH_COLUMN].to_numpy() - inner_predictions
    for target_age in target_ages:
        target_mask = inner_frame["age"].to_numpy() == target_age
        static_errors = residuals[target_mask]
        if len(static_errors) < 10:
            static_errors = residuals
        quantiles[("static", target_age)] = conformal_quantile(static_errors, alpha)
        subset = (
            records[records["target_age"] == target_age]
            if not records.empty
            else records
        )
        for method in ("raw", "eb"):
            if subset.empty:
                quantiles[(method, target_age)] = quantiles[("static", target_age)]
            else:
                errors = subset["actual"] - subset[f"{method}_prediction"]
                quantiles[(method, target_age)] = conformal_quantile(errors, alpha)
    return quantiles


def evaluate_state_updates(
    frame: pd.DataFrame,
    seeds: Sequence[int],
    model_config: Dict[str, object],
    outer_folds: int,
    inner_folds: int,
    early_ages: Sequence[int],
    target_ages: Sequence[int],
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, float]] = []
    prediction_rows: List[Dict[str, float]] = []
    uncertainty_rows: List[Dict[str, float]] = []

    for seed in seeds:
        seed_records: List[Dict[str, float]] = []
        for fold, (train_idx, test_idx) in enumerate(
            group_folds(frame["mix_id"], outer_folds, seed)
        ):
            train = frame.iloc[train_idx].reset_index(drop=True)
            test = frame.iloc[test_idx].reset_index(drop=True)
            model = make_strength_model(seed + 7919 * (fold + 1), model_config)
            model.fit(train[FEATURE_COLUMNS], train[STRENGTH_COLUMN])
            test_predictions = model.predict(test[FEATURE_COLUMNS])

            inner_predictions = cross_fitted_predictions(
                train,
                model_config=model_config,
                n_splits=inner_folds,
                seed=seed + 104729 * (fold + 1),
            )
            inner_residuals = train[["mix_id"]].copy()
            inner_residuals["residual"] = (
                train[STRENGTH_COLUMN].to_numpy() - inner_predictions
            )
            components = estimate_variance_components(inner_residuals)
            quantiles = _calibration_quantiles(
                train,
                inner_predictions,
                components,
                early_ages,
                target_ages,
                alpha,
            )
            fold_records = _trajectory_records(
                test,
                test_predictions,
                early_ages,
                target_ages,
                components,
            )
            for record in fold_records:
                record.update(
                    {
                        "seed": int(seed),
                        "fold": int(fold),
                        "tau2": components.tau2,
                        "sigma2": components.sigma2,
                    }
                )
                for method in ("static", "raw", "eb"):
                    q = quantiles[(method, int(record["target_age"]))]
                    record[f"{method}_interval_half_width"] = q
                    prediction = record[f"{method}_prediction"]
                    record[f"{method}_covered"] = float(
                        prediction - q <= record["actual"] <= prediction + q
                    )
                seed_records.append(record)
                prediction_rows.append(record.copy())

            for target_age in target_ages:
                uncertainty_rows.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold),
                        "target_age": int(target_age),
                        "tau2": components.tau2,
                        "sigma2": components.sigma2,
                        "static_half_width": quantiles[("static", target_age)],
                        "raw_half_width": quantiles[("raw", target_age)],
                        "eb_half_width": quantiles[("eb", target_age)],
                    }
                )

        seed_frame = pd.DataFrame(seed_records)
        for target_age in target_ages:
            subset = seed_frame[seed_frame["target_age"] == target_age]
            for method in ("static", "raw", "eb"):
                actual = subset["actual"].to_numpy(dtype=float)
                predicted = subset[f"{method}_prediction"].to_numpy(dtype=float)
                rmse, mae, bias = regression_metrics(actual, predicted)
                metric_rows.append(
                    {
                        "seed": int(seed),
                        "target_age": int(target_age),
                        "method": method,
                        "n_groups": int(len(subset)),
                        "rmse_mpa": rmse,
                        "mae_mpa": mae,
                        "bias_mpa": bias,
                        "picp": float(subset[f"{method}_covered"].mean()),
                        "mean_interval_width_mpa": float(
                            2.0 * subset[f"{method}_interval_half_width"].mean()
                        ),
                        "mean_shrinkage": float(subset["shrinkage"].mean()),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    return metrics, predictions, uncertainty


def summarize_state_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["target_age", "method"], sort=True)
    summary = grouped.agg(
        n_groups=("n_groups", "first"),
        rmse_mean=("rmse_mpa", "mean"),
        rmse_sd=("rmse_mpa", "std"),
        mae_mean=("mae_mpa", "mean"),
        mae_sd=("mae_mpa", "std"),
        picp_mean=("picp", "mean"),
        picp_sd=("picp", "std"),
        interval_width_mean=("mean_interval_width_mpa", "mean"),
        shrinkage_mean=("mean_shrinkage", "mean"),
    ).reset_index()
    static_rmse = summary[summary["method"] == "static"][
        ["target_age", "rmse_mean"]
    ].rename(columns={"rmse_mean": "static_rmse"})
    summary = summary.merge(static_rmse, on="target_age", how="left")
    summary["rmse_change_percent_vs_static"] = (
        100.0 * (summary["rmse_mean"] - summary["static_rmse"]) / summary["static_rmse"]
    )
    return summary.drop(columns=["static_rmse"])


def _non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    mask = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominates = np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        mask[index] = not bool(np.any(dominates))
    return mask


def evaluate_decision_impacts(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    required_strength_mpa: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure admission and Pareto-ranking changes caused by state updating."""
    attributes = (
        frame.groupby(["mix_id", "age"], as_index=False)[
            [GWP_COLUMN, ENERGY_COLUMN, MATERIAL_TOTAL_COLUMN]
        ]
        .first()
        .rename(columns={"age": "target_age"})
    )
    decisions = predictions.merge(
        attributes,
        on=["mix_id", "target_age"],
        how="left",
        validate="many_to_one",
    ).copy()
    if decisions[[GWP_COLUMN, ENERGY_COLUMN, MATERIAL_TOTAL_COLUMN]].isna().any().any():
        raise ValueError("Decision replay could not recover environmental attributes")

    decisions["required_strength_mpa"] = float(required_strength_mpa)
    decisions["actual_accepted"] = decisions["actual"] >= required_strength_mpa
    for method in ("static", "eb"):
        decisions[f"{method}_lower_bound"] = (
            decisions[f"{method}_prediction"]
            - decisions[f"{method}_interval_half_width"]
        )
        decisions[f"{method}_accepted"] = (
            decisions[f"{method}_lower_bound"] >= required_strength_mpa
        )
        decisions[f"{method}_decision_error"] = (
            decisions[f"{method}_accepted"] != decisions["actual_accepted"]
        )
    decisions["admission_changed"] = (
        decisions["static_accepted"] != decisions["eb_accepted"]
    )
    decisions["decision_corrected"] = (
        decisions["static_decision_error"] & ~decisions["eb_decision_error"]
    )
    decisions["decision_error_introduced"] = (
        ~decisions["static_decision_error"] & decisions["eb_decision_error"]
    )

    metric_rows: List[Dict[str, float]] = []
    for (seed, target_age), group in decisions.groupby(
        ["seed", "target_age"], sort=True
    ):
        index = group.index
        truth = group["actual_accepted"].to_numpy(dtype=bool)
        static_decision = group["static_accepted"].to_numpy(dtype=bool)
        eb_decision = group["eb_accepted"].to_numpy(dtype=bool)
        negative_count = int((~truth).sum())
        positive_count = int(truth.sum())

        environmental = group[
            [GWP_COLUMN, ENERGY_COLUMN, MATERIAL_TOTAL_COLUMN]
        ].to_numpy(dtype=float)
        static_objectives = np.column_stack(
            [
                environmental,
                -group["static_lower_bound"].to_numpy(dtype=float),
            ]
        )
        eb_objectives = np.column_stack(
            [
                environmental,
                -group["eb_lower_bound"].to_numpy(dtype=float),
            ]
        )
        static_pareto = _non_dominated_mask(static_objectives)
        eb_pareto = _non_dominated_mask(eb_objectives)

        combined = np.vstack([static_objectives, eb_objectives])
        lower = combined.min(axis=0)
        scale = np.maximum(combined.max(axis=0) - lower, 1e-12)
        static_score = ((static_objectives - lower) / scale).mean(axis=1)
        eb_score = ((eb_objectives - lower) / scale).mean(axis=1)
        static_rank = pd.Series(static_score).rank(method="average").to_numpy()
        eb_rank = pd.Series(eb_score).rank(method="average").to_numpy()
        rank_tau = float(kendalltau(static_rank, eb_rank).statistic)
        top_count = min(10, len(group))
        static_top = set(group.iloc[np.argsort(static_score)[:top_count]]["mix_id"])
        eb_top = set(group.iloc[np.argsort(eb_score)[:top_count]]["mix_id"])
        static_pareto_ids = set(group.loc[static_pareto, "mix_id"])
        eb_pareto_ids = set(group.loc[eb_pareto, "mix_id"])
        pareto_union = static_pareto_ids | eb_pareto_ids
        pareto_intersection = static_pareto_ids & eb_pareto_ids

        decisions.loc[index, "static_pareto"] = static_pareto
        decisions.loc[index, "eb_pareto"] = eb_pareto
        decisions.loc[index, "static_equal_weight_score"] = static_score
        decisions.loc[index, "eb_equal_weight_score"] = eb_score
        decisions.loc[index, "static_rank"] = static_rank
        decisions.loc[index, "eb_rank"] = eb_rank

        metric_rows.append(
            {
                "seed": int(seed),
                "target_age": int(target_age),
                "n_groups": int(len(group)),
                "actual_below_target": negative_count,
                "actual_at_or_above_target": positive_count,
                "admission_change_rate": float(np.mean(static_decision != eb_decision)),
                "corrected_error_rate": float(group["decision_corrected"].mean()),
                "introduced_error_rate": float(
                    group["decision_error_introduced"].mean()
                ),
                "false_acceptance_rate_static": float(
                    np.sum(static_decision & ~truth) / negative_count
                )
                if negative_count
                else float("nan"),
                "false_acceptance_rate_eb": float(
                    np.sum(eb_decision & ~truth) / negative_count
                )
                if negative_count
                else float("nan"),
                "false_rejection_rate_static": float(
                    np.sum(~static_decision & truth) / positive_count
                )
                if positive_count
                else float("nan"),
                "false_rejection_rate_eb": float(
                    np.sum(~eb_decision & truth) / positive_count
                )
                if positive_count
                else float("nan"),
                "static_pareto_count": int(static_pareto.sum()),
                "eb_pareto_count": int(eb_pareto.sum()),
                "pareto_membership_change_count": int(
                    np.sum(static_pareto != eb_pareto)
                ),
                "pareto_jaccard": float(
                    len(pareto_intersection) / max(len(pareto_union), 1)
                ),
                "rank_kendall_tau": rank_tau,
                "top10_overlap": float(len(static_top & eb_top) / top_count),
                "compromise_changed": float(
                    group.iloc[int(np.argmin(static_score))]["mix_id"]
                    != group.iloc[int(np.argmin(eb_score))]["mix_id"]
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("target_age", sort=True)
        .agg(
            n_groups=("n_groups", "first"),
            admission_change_rate_mean=("admission_change_rate", "mean"),
            admission_change_rate_sd=("admission_change_rate", "std"),
            corrected_error_rate_mean=("corrected_error_rate", "mean"),
            introduced_error_rate_mean=("introduced_error_rate", "mean"),
            false_acceptance_rate_static_mean=("false_acceptance_rate_static", "mean"),
            false_acceptance_rate_eb_mean=("false_acceptance_rate_eb", "mean"),
            false_rejection_rate_static_mean=("false_rejection_rate_static", "mean"),
            false_rejection_rate_eb_mean=("false_rejection_rate_eb", "mean"),
            pareto_membership_change_count_mean=(
                "pareto_membership_change_count",
                "mean",
            ),
            pareto_jaccard_mean=("pareto_jaccard", "mean"),
            pareto_jaccard_sd=("pareto_jaccard", "std"),
            rank_kendall_tau_mean=("rank_kendall_tau", "mean"),
            rank_kendall_tau_sd=("rank_kendall_tau", "std"),
            top10_overlap_mean=("top10_overlap", "mean"),
            compromise_change_rate=("compromise_changed", "mean"),
        )
        .reset_index()
    )
    return decisions, metrics, summary


def bootstrap_rmse_changes(
    predictions: pd.DataFrame,
    iterations: int = 10000,
    seed: int = 20260805,
) -> pd.DataFrame:
    """Paired group bootstrap using seed-averaged squared errors per mixture."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for target_age, target in predictions.groupby("target_age", sort=True):
        group_rows = []
        for mix_id, group in target.groupby("mix_id"):
            record = {"mix_id": mix_id}
            for method in ("static", "raw", "eb"):
                error = group["actual"].to_numpy(dtype=float) - group[
                    f"{method}_prediction"
                ].to_numpy(dtype=float)
                record[f"{method}_mse"] = float(np.mean(np.square(error)))
            group_rows.append(record)
        grouped = pd.DataFrame(group_rows)
        count = len(grouped)
        sample_indices = rng.integers(0, count, size=(iterations, count))
        static_mse = grouped["static_mse"].to_numpy()[sample_indices].mean(axis=1)
        static_rmse = np.sqrt(static_mse)
        for method in ("raw", "eb"):
            method_mse = (
                grouped[f"{method}_mse"].to_numpy()[sample_indices].mean(axis=1)
            )
            method_rmse = np.sqrt(method_mse)
            change = 100.0 * (method_rmse - static_rmse) / static_rmse
            rows.append(
                {
                    "target_age": int(target_age),
                    "method": method,
                    "n_groups": int(count),
                    "rmse_change_percent": float(np.median(change)),
                    "rmse_change_ci_lower": float(np.quantile(change, 0.025)),
                    "rmse_change_ci_upper": float(np.quantile(change, 0.975)),
                    "probability_rmse_improvement": float(np.mean(change < 0.0)),
                    "bootstrap_iterations": int(iterations),
                }
            )
    return pd.DataFrame(rows)
