"""Scenario-based baseline, constrained, and robust NSGA-III experiments."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from .constraints import ApplicabilityDomain, EngineeringScreen, RULE_NAMES
from .data import FEATURE_COLUMNS, MATERIAL_COLUMNS, STRENGTH_COLUMN
from .lca import FactorSamples, ScreeningLCA
from .modeling import make_strength_model


STAGES = ["baseline_data_domain", "knowledge_deterministic", "knowledge_robust"]


def center_distance(
    objectives: np.ndarray, scales: Optional[np.ndarray] = None
) -> float:
    """Mean Euclidean distance from the origin in a declared scaled space."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("objectives must be a non-empty two-dimensional array")
    if scales is not None:
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (values.shape[1],) or np.any(scales <= 0.0):
            raise ValueError("scales must contain one positive value per objective")
        values = values / scales
    return float(np.mean(np.linalg.norm(values, axis=1)))


def spacing_metric(objectives: np.ndarray, normalize: bool = True) -> float:
    """Standard deviation of nearest-neighbor distances on a Pareto set."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    if normalize and len(values):
        values = _normalize(values, values.min(axis=0), values.max(axis=0))
    return _spacing(values)


def _candidate_id(seed: int, scenario: str, stage: str, values: np.ndarray) -> str:
    payload = f"{seed}|{scenario}|{stage}|" + "|".join(f"{v:.8f}" for v in values)
    return "candidate_" + hashlib.sha1(payload.encode("ascii")).hexdigest()[:14]


def _predict_strength(model, quantities: np.ndarray, age_days: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(quantities, dtype=float))
    features = pd.DataFrame(x, columns=MATERIAL_COLUMNS)
    features["age"] = float(age_days)
    return model.predict(features[FEATURE_COLUMNS])


class ConcreteOptimizationProblem(Problem):
    def __init__(
        self,
        stage: str,
        scenario: Mapping[str, object],
        strength_model,
        strength_half_width: float,
        engineering_screen: EngineeringScreen,
        lca: ScreeningLCA,
        lca_moments: Mapping[str, np.ndarray],
        risk_lambda: float,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        if stage not in STAGES:
            raise ValueError(f"Unknown optimization stage: {stage}")
        self.stage = stage
        self.scenario = scenario
        self.strength_model = strength_model
        self.strength_half_width = float(strength_half_width)
        self.engineering_screen = engineering_screen
        self.lca = lca
        self.lca_moments = lca_moments
        self.risk_lambda = float(risk_lambda)
        super().__init__(
            n_var=len(MATERIAL_COLUMNS),
            n_obj=4,
            n_ieq_constr=0 if stage == "baseline_data_domain" else len(RULE_NAMES),
            xl=lower_bounds,
            xu=upper_bounds,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        strength = _predict_strength(
            self.strength_model, x, int(self.scenario["age_days"])
        )
        material = np.asarray(x).sum(axis=1)
        if self.stage == "knowledge_robust":
            risk = self.lca.risk_objectives(x, self.lca_moments, self.risk_lambda)
            gwp = risk["gwp_risk"]
            energy = risk["energy_risk"]
        else:
            gwp, energy = self.lca.deterministic(x)
        out["F"] = np.column_stack([gwp, energy, material, -strength])
        if self.stage != "baseline_data_domain":
            constraints, _ = self.engineering_screen.constraint_values(
                x,
                strength,
                self.strength_half_width,
                float(self.scenario["required_strength_mpa"]),
            )
            out["G"] = constraints


def _non_dominated_indices(objectives: np.ndarray) -> np.ndarray:
    if len(objectives) == 0:
        return np.array([], dtype=int)
    return NonDominatedSorting().do(objectives, only_non_dominated_front=True)


def _normalize(
    objectives: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    scale = np.maximum(upper - lower, 1e-12)
    return (objectives - lower) / scale


def _spacing(normalized_objectives: np.ndarray) -> float:
    if len(normalized_objectives) < 2:
        return float("nan")
    distances = np.linalg.norm(
        normalized_objectives[:, None, :] - normalized_objectives[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest, ddof=1)) if len(nearest) > 1 else 0.0


def _choice_stability(
    quantities: np.ndarray,
    strength: np.ndarray,
    samples: FactorSamples,
) -> Tuple[float, int]:
    if len(quantities) == 0:
        return float("nan"), 0
    gwp = quantities @ samples.gwp.T
    energy = quantities @ samples.energy.T
    material = quantities.sum(axis=1)
    negative_strength = -np.asarray(strength)

    def normalize_columns(values: np.ndarray) -> np.ndarray:
        low = values.min(axis=0, keepdims=True)
        span = np.maximum(values.max(axis=0, keepdims=True) - low, 1e-12)
        return (values - low) / span

    gwp_n = normalize_columns(gwp)
    energy_n = normalize_columns(energy)
    material_n = normalize_columns(material[:, None])
    strength_n = normalize_columns(negative_strength[:, None])
    scores = (gwp_n + energy_n + material_n + strength_n) / 4.0
    winners = np.argmin(scores, axis=0)
    counts = np.bincount(winners, minlength=len(quantities))
    return float(counts.max() / len(winners)), int(np.count_nonzero(counts))


def _evaluate_candidates(
    x: np.ndarray,
    seed: int,
    scenario: Mapping[str, object],
    stage: str,
    strength_model,
    strength_half_width: float,
    screen: EngineeringScreen,
    lca: ScreeningLCA,
    samples: FactorSamples,
    moments: Mapping[str, np.ndarray],
    risk_lambda: float,
) -> pd.DataFrame:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    strength = _predict_strength(strength_model, x, int(scenario["age_days"]))
    screening = screen.evaluate(
        x,
        strength,
        strength_half_width,
        float(scenario["required_strength_mpa"]),
    )
    deterministic_gwp, deterministic_energy = lca.deterministic(x)
    risk = lca.risk_objectives(x, moments, risk_lambda)
    distributions = lca.candidate_distributions(x, samples)

    result = pd.DataFrame(x, columns=MATERIAL_COLUMNS)
    result.insert(
        0,
        "candidate_id",
        [_candidate_id(seed, str(scenario["id"]), stage, row) for row in x],
    )
    result.insert(1, "seed", int(seed))
    result.insert(2, "scenario", str(scenario["id"]))
    result.insert(3, "stage", stage)
    result["age_days"] = int(scenario["age_days"])
    result["required_strength_mpa"] = float(scenario["required_strength_mpa"])
    result["strength_prediction_mpa"] = strength
    result["strength_interval_half_width_mpa"] = float(strength_half_width)
    result["strength_lower_bound_mpa"] = strength - float(strength_half_width)
    result["gwp_deterministic_kgco2e_m3"] = deterministic_gwp
    result["energy_deterministic_mj_m3"] = deterministic_energy
    result["gwp_risk_kgco2e_m3"] = risk["gwp_risk"]
    result["gwp_mean_kgco2e_m3"] = risk["gwp_mean"]
    result["gwp_sd_kgco2e_m3"] = risk["gwp_sd"]
    result["gwp_p05_kgco2e_m3"] = distributions["gwp_p05"]
    result["gwp_p95_kgco2e_m3"] = distributions["gwp_p95"]
    result["energy_risk_mj_m3"] = risk["energy_risk"]
    result["energy_mean_mj_m3"] = risk["energy_mean"]
    result["energy_sd_mj_m3"] = risk["energy_sd"]
    result["energy_p05_mj_m3"] = distributions["energy_p05"]
    result["energy_p95_mj_m3"] = distributions["energy_p95"]
    for column in screening.columns:
        result[column] = screening[column].to_numpy()
    return result


def run_optimization_experiments(
    frame: pd.DataFrame,
    run_config: Mapping[str, object],
    engineering_config: Mapping[str, object],
    lca_config: Mapping[str, object],
    uncertainty: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    optimization = run_config["optimization"]
    material_order = list(engineering_config["material_order"])
    if material_order != MATERIAL_COLUMNS:
        raise ValueError(
            "engineering_config.material_order must match the canonical model order"
        )
    quantities = frame[material_order].to_numpy(dtype=float)
    applicability = ApplicabilityDomain.fit(
        quantities,
        k_neighbors=int(optimization["k_neighbors"]),
        quantile=float(optimization["applicability_quantile"]),
    )
    screen = EngineeringScreen(engineering_config, material_order, applicability)
    lca = ScreeningLCA(lca_config, material_order)
    bounds = engineering_config["material_bounds_kg_m3"]
    lower = np.array([bounds[name][0] for name in material_order], dtype=float)
    upper = np.array([bounds[name][1] for name in material_order], dtype=float)
    ref_dirs = get_reference_directions(
        "das-dennis", 4, n_partitions=int(optimization["reference_partitions"])
    )

    candidate_frames: List[pd.DataFrame] = []
    sample_cache: Dict[int, FactorSamples] = {}
    factor_summary = None
    for seed in run_config["seeds"]:
        model = make_strength_model(int(seed), run_config["model"])
        model.fit(frame[FEATURE_COLUMNS], frame[STRENGTH_COLUMN])
        samples = lca.sample(
            int(optimization["monte_carlo_samples"]), int(seed) + 50021
        )
        moments = lca.moments(samples)
        sample_cache[int(seed)] = samples
        if factor_summary is None:
            factor_summary = lca.factor_summary(samples)

        for scenario in engineering_config["scenarios"]:
            subset = uncertainty[
                (uncertainty["seed"] == int(seed))
                & (uncertainty["target_age"] == int(scenario["age_days"]))
            ]
            if subset.empty:
                raise ValueError(
                    "Missing strength-uncertainty calibration for "
                    f"seed {seed}, age {scenario['age_days']}"
                )
            strength_half_width = float(subset["static_half_width"].mean())
            if not np.isfinite(strength_half_width) or strength_half_width < 0.0:
                raise ValueError(
                    "Strength interval half-width must be finite and non-negative"
                )
            for stage in STAGES:
                problem = ConcreteOptimizationProblem(
                    stage=stage,
                    scenario=scenario,
                    strength_model=model,
                    strength_half_width=strength_half_width,
                    engineering_screen=screen,
                    lca=lca,
                    lca_moments=moments,
                    risk_lambda=float(optimization["risk_lambda"]),
                    lower_bounds=lower,
                    upper_bounds=upper,
                )
                algorithm = NSGA3(
                    pop_size=int(optimization["population_size"]),
                    ref_dirs=ref_dirs,
                    eliminate_duplicates=True,
                )
                result = minimize(
                    problem,
                    algorithm,
                    termination=("n_gen", int(optimization["generations"])),
                    seed=int(seed),
                    verbose=False,
                )
                if result.X is None:
                    raise RuntimeError(
                        f"NSGA-III found no feasible solution for {scenario['id']} / {stage} / seed {seed}"
                    )
                candidate_frames.append(
                    _evaluate_candidates(
                        result.X,
                        int(seed),
                        scenario,
                        stage,
                        model,
                        strength_half_width,
                        screen,
                        lca,
                        samples,
                        moments,
                        float(optimization["risk_lambda"]),
                    )
                )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    run_metrics: List[Dict[str, object]] = []
    for (seed, scenario), comparison in candidates.groupby(
        ["seed", "scenario"], sort=True
    ):
        feasible_all = comparison[comparison["is_feasible"]]
        objective_columns = [
            "gwp_risk_kgco2e_m3",
            "energy_risk_mj_m3",
            "total_material",
            "strength_prediction_mpa",
        ]
        if feasible_all.empty:
            raise RuntimeError(
                f"No engineering-feasible candidates for seed {seed}, scenario {scenario}"
            )
        common = feasible_all[objective_columns].to_numpy(dtype=float)
        common[:, 3] *= -1.0
        lower_common = common.min(axis=0)
        upper_common = common.max(axis=0)

        for stage, stage_frame in comparison.groupby("stage", sort=True):
            feasible = stage_frame[stage_frame["is_feasible"]].copy()
            objectives = feasible[objective_columns].to_numpy(dtype=float)
            if len(objectives):
                objectives[:, 3] *= -1.0
                nd_idx = _non_dominated_indices(objectives)
                feasible = feasible.iloc[nd_idx].copy()
                objectives = objectives[nd_idx]
                normalized = _normalize(objectives, lower_common, upper_common)
                hypervolume = float(HV(ref_point=np.full(4, 1.1))(normalized))
                spacing = _spacing(normalized)
                stability, winner_count = _choice_stability(
                    feasible[MATERIAL_COLUMNS].to_numpy(dtype=float),
                    feasible["strength_prediction_mpa"].to_numpy(dtype=float),
                    sample_cache[int(seed)],
                )
            else:
                hypervolume = 0.0
                spacing = float("nan")
                stability = float("nan")
                winner_count = 0
            run_metrics.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "stage": stage,
                    "candidate_count": int(len(stage_frame)),
                    "feasible_candidate_count": int(stage_frame["is_feasible"].sum()),
                    "infeasible_ratio": float(1.0 - stage_frame["is_feasible"].mean()),
                    "common_nondominated_feasible_count": int(len(feasible)),
                    "feasible_hypervolume": hypervolume,
                    "spacing": spacing,
                    "modal_choice_stability": stability,
                    "unique_monte_carlo_winners": int(winner_count),
                }
            )

    run_metrics_frame = pd.DataFrame(run_metrics)
    summary = (
        run_metrics_frame.groupby(["scenario", "stage"], sort=True)
        .agg(
            candidate_count_mean=("candidate_count", "mean"),
            infeasible_ratio_mean=("infeasible_ratio", "mean"),
            infeasible_ratio_sd=("infeasible_ratio", "std"),
            feasible_hypervolume_mean=("feasible_hypervolume", "mean"),
            feasible_hypervolume_sd=("feasible_hypervolume", "std"),
            spacing_mean=("spacing", "mean"),
            modal_choice_stability_mean=("modal_choice_stability", "mean"),
            modal_choice_stability_sd=("modal_choice_stability", "std"),
        )
        .reset_index()
    )
    ad_summary = {
        "k_neighbors": int(applicability.k_neighbors),
        "distance_threshold": float(applicability.threshold),
        "empirical_quantile": float(optimization["applicability_quantile"]),
    }
    if factor_summary is None:
        raise RuntimeError("No LCA factor samples were generated")
    return candidates, run_metrics_frame, summary, factor_summary, ad_summary


class ScaledGaussianMutation(Mutation):
    """Bounded Gaussian mutation matching the legacy scale/probability controls."""

    def __init__(self, probability: float, scale: float):
        super().__init__()
        self.probability = float(probability)
        self.scale = float(scale)

    def _do(self, problem, x, **kwargs):
        mutated = np.asarray(x, dtype=float).copy()
        mask = np.random.random(mutated.shape) < self.probability
        perturbation = np.random.normal(size=mutated.shape)
        perturbation *= self.scale * (problem.xu - problem.xl)
        mutated[mask] += perturbation[mask]
        return np.clip(mutated, problem.xl, problem.xu)


class LegacyDataDomainProblem(Problem):
    """Original eight-variable, four-surrogate data-domain formulation."""

    def __init__(
        self,
        surrogate,
        target_columns: Sequence[str],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        self.surrogate = surrogate
        self.target_columns = list(target_columns)
        required = {
            "concrete_compressive_strength",
            "Embodied_CO2 (kg)",
            "Energy_Use (MJ)",
            "Total_Material_Use (kg)",
        }
        if set(self.target_columns) != required:
            raise ValueError("Legacy optimization requires the four manuscript outputs")
        super().__init__(
            n_var=len(FEATURE_COLUMNS),
            n_obj=4,
            n_ieq_constr=0,
            xl=np.asarray(lower_bounds, dtype=float),
            xu=np.asarray(upper_bounds, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        features = pd.DataFrame(np.asarray(x, dtype=float), columns=FEATURE_COLUMNS)
        prediction = np.asarray(self.surrogate.predict(features), dtype=float)
        index = {name: self.target_columns.index(name) for name in self.target_columns}
        out["F"] = np.column_stack(
            [
                prediction[:, index["Embodied_CO2 (kg)"]],
                prediction[:, index["Energy_Use (MJ)"]],
                prediction[:, index["Total_Material_Use (kg)"]],
                -prediction[:, index["concrete_compressive_strength"]],
            ]
        )


class ParetoConvergenceTrace(Callback):
    """Record center distance and spacing without retaining population rows."""

    def __init__(self, center_scales: Optional[Sequence[float]] = None):
        super().__init__()
        self.center_scales = (
            None if center_scales is None else np.asarray(center_scales, dtype=float)
        )
        self.rows: List[Dict[str, float]] = []

    def notify(self, algorithm):
        objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
        nondominated = objectives[_non_dominated_indices(objectives)]
        self.rows.append(
            {
                "generation": int(algorithm.n_gen),
                "nondominated_count": int(len(nondominated)),
                "center_distance": center_distance(
                    nondominated, scales=self.center_scales
                ),
                "spacing": spacing_metric(nondominated, normalize=True),
            }
        )


def run_legacy_data_domain_optimization(
    surrogate,
    target_columns: Sequence[str],
    bounds: np.ndarray,
    config: Mapping[str, object],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the original unconstrained eight-variable NSGA-III analysis.

    This reproduces the manuscript's baseline formulation only. Its candidates
    are data-domain model outputs and must not be labelled construction-ready.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(FEATURE_COLUMNS), 2):
        raise ValueError("bounds must contain lower/upper values for eight inputs")
    problem = LegacyDataDomainProblem(
        surrogate,
        target_columns,
        lower_bounds=bounds[:, 0],
        upper_bounds=bounds[:, 1],
    )
    ref_dirs = get_reference_directions(
        "das-dennis",
        4,
        n_partitions=int(config.get("reference_partitions", 6)),
    )
    callback = ParetoConvergenceTrace(config.get("center_distance_scales"))
    algorithm = NSGA3(
        pop_size=int(config.get("population_size", 400)),
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(
            prob=float(config.get("crossover_probability", 0.9)),
            eta=float(config.get("crossover_eta", 15.0)),
        ),
        mutation=ScaledGaussianMutation(
            probability=float(config.get("mutation_probability", 0.2)),
            scale=float(config.get("mutation_scale", 0.2)),
        ),
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", int(config.get("generations", 1200))),
        seed=int(seed),
        callback=callback,
        verbose=False,
    )
    if result.X is None or result.F is None:
        raise RuntimeError("Legacy NSGA-III did not return a Pareto set")
    candidates = pd.DataFrame(result.X, columns=FEATURE_COLUMNS)
    objectives = np.asarray(result.F, dtype=float)
    candidates["predicted_gwp_kgco2e_m3"] = objectives[:, 0]
    candidates["predicted_energy_mj_m3"] = objectives[:, 1]
    candidates["predicted_total_material_kg_m3"] = objectives[:, 2]
    candidates["predicted_strength_mpa"] = -objectives[:, 3]
    candidates["candidate_status"] = "data_domain_model_candidate"
    return candidates, pd.DataFrame(callback.rows)
