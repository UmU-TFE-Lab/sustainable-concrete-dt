#!/usr/bin/env python3
"""Run the statistical, ML, and conditional-effect methods in the manuscript.

The command writes aggregate tables only. It deliberately does not export the
train/test rows, row-level predictions, PCA scores, or cluster assignments.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scipy
import sklearn

from concrete_dt.config import load_json, load_run_config, write_json
from concrete_dt import __version__ as code_version
from concrete_dt.data import file_sha256, load_dataset
from concrete_dt.effects import (
    fit_dml_grid,
    fit_mlr_models,
    fit_observed_path_model,
    intervention_scenarios,
    model_based_interventions,
)
from concrete_dt.exploratory import (
    clustering_analysis,
    correlation_analysis,
    descriptive_statistics,
    principal_component_analysis,
    selected_cluster_summary,
)
from concrete_dt.predictive import benchmark_multioutput_models, build_predictive_model
from concrete_dt.probabilistic import fit_concrete_bayesian_network, query_table
from concrete_dt.sensitivity import sobol_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="Local trusted CSV override; the file is read but never copied.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-missing-optional-models",
        action="store_true",
        help="Continue without XGBoost if its optional dependency is unavailable.",
    )
    return parser.parse_args()


def _write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> str:
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _installed_version(distribution: str) -> str:
    """Return a package version without making optional dependencies mandatory."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def main() -> None:
    args = parse_args()
    run_config = load_run_config(args.config)
    methods = load_json(args.methods_config)
    engineering = load_json(Path(run_config["engineering_config"]))
    data_path = args.data.resolve() if args.data else Path(run_config["data_path"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(run_config["output_dir"]) / "manuscript_methods"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path, engineering)

    features = list(methods["feature_columns"])
    targets = list(methods["target_columns"])
    analysis_columns = list(methods["analysis_columns"])
    seed = int(run_config["seeds"][0])
    outputs: Dict[str, str] = {}

    outputs["descriptive_statistics"] = _write_table(
        descriptive_statistics(frame, analysis_columns),
        output_dir,
        "descriptive_statistics",
    )
    outputs["correlations"] = _write_table(
        correlation_analysis(
            frame,
            analysis_columns,
            methods=methods["correlation"]["methods"],
        ),
        output_dir,
        "correlations",
    )

    clustering_config = methods["clustering"]
    clustering = clustering_analysis(
        frame,
        analysis_columns,
        k_values=clustering_config["k_values"],
        selected_k=int(clustering_config["selected_k"]),
        seed=seed,
        n_init=int(clustering_config["n_init"]),
    )
    outputs["clustering_diagnostics"] = _write_table(
        clustering.diagnostics, output_dir, "clustering_diagnostics"
    )
    outputs["kmeans_cluster_summary"] = _write_table(
        selected_cluster_summary(
            frame,
            analysis_columns,
            clustering.kmeans_labels,
            "kmeans_cluster",
        ),
        output_dir,
        "kmeans_cluster_summary",
    )
    outputs["agglomerative_cluster_summary"] = _write_table(
        selected_cluster_summary(
            frame,
            analysis_columns,
            clustering.agglomerative_labels,
            "agglomerative_cluster",
        ),
        output_dir,
        "agglomerative_cluster_summary",
    )

    pca_config = methods["pca"]
    pca = principal_component_analysis(
        frame,
        analysis_columns,
        n_components=int(pca_config["n_components"]),
        include_scores=bool(pca_config.get("export_row_scores", False)),
    )
    if pca.scores is not None:
        raise ValueError("Row-level PCA-score export is disabled for this release")
    outputs["pca_variance"] = _write_table(pca.variance, output_dir, "pca_variance")
    outputs["pca_loadings"] = _write_table(pca.loadings, output_dir, "pca_loadings")

    predictive_config = dict(methods["predictive"])
    if args.allow_missing_optional_models:
        predictive_config["require_all_models"] = False
    benchmark = benchmark_multioutput_models(
        frame,
        features,
        targets,
        predictive_config,
        seed,
    )
    outputs["predictive_metrics"] = _write_table(
        benchmark.metrics, output_dir, "predictive_metrics"
    )
    outputs["predictive_resources"] = _write_table(
        benchmark.resources, output_dir, "predictive_resources"
    )

    sobol_config = methods["sobol"]
    surrogate_name = str(sobol_config["surrogate_model"])
    if surrogate_name not in benchmark.models:
        raise RuntimeError(f"Sobol surrogate was not fitted: {surrogate_name}")
    feature_bounds = np.column_stack(
        [frame[features].min(axis=0).to_numpy(), frame[features].max(axis=0).to_numpy()]
    )
    outputs["sobol_indices"] = _write_table(
        sobol_indices(
            benchmark.models[surrogate_name].predict,
            variable_names=features,
            bounds=feature_bounds,
            base_sample_count=int(sobol_config["base_sample_count"]),
            seed=seed,
            output_names=targets,
            numerical_zero_tolerance=float(sobol_config["numerical_zero_tolerance"]),
        ),
        output_dir,
        "sobol_indices",
    )

    legacy_config = methods.get("legacy_data_domain_optimization", {})
    if bool(legacy_config.get("enabled", False)):
        # Imported lazily so exploratory-only runs do not require pymoo.
        from concrete_dt.optimization import run_legacy_data_domain_optimization

        legacy_model_name = str(legacy_config["surrogate_model"])
        legacy_model = build_predictive_model(
            legacy_model_name,
            predictive_config["model_parameters"][legacy_model_name],
            seed,
        )
        legacy_model.fit(frame[features], frame[targets])
        legacy_candidates, legacy_trace = run_legacy_data_domain_optimization(
            legacy_model,
            target_columns=targets,
            bounds=feature_bounds,
            config=legacy_config,
            seed=seed,
        )
        outputs["legacy_optimization_trace"] = _write_table(
            legacy_trace, output_dir, "legacy_optimization_trace"
        )
        # Keep candidate-level decisions private; release only aggregate ranges.
        legacy_summary = (
            legacy_candidates.select_dtypes(include="number")
            .agg(["min", "mean", "max"])
            .T.reset_index(names="variable")
        )
        outputs["legacy_optimization_summary"] = _write_table(
            legacy_summary, output_dir, "legacy_optimization_summary"
        )

    bn_config = methods["bayesian_network"]
    network, edges, thresholds = fit_concrete_bayesian_network(
        frame,
        feature_columns=features,
        target_columns=targets,
        parent_map=bn_config["parent_map"],
        smoothing=float(bn_config["smoothing"]),
    )
    outputs["dag_edges"] = _write_table(edges, output_dir, "dag_edges")
    outputs["bayesian_thresholds"] = _write_table(
        thresholds, output_dir, "bayesian_discretization_thresholds"
    )
    outputs["bayesian_queries"] = _write_table(
        query_table(network, bn_config["queries"]),
        output_dir,
        "bayesian_queries",
    )

    outputs["mlr_coefficients"] = _write_table(
        fit_mlr_models(frame, targets, features),
        output_dir,
        "mlr_coefficients",
    )
    path_config = methods["path_model"]
    path_coefficients, path_covariances = fit_observed_path_model(
        frame,
        equations=path_config["equations"],
        covariance_pairs=path_config["covariance_pairs"],
    )
    outputs["path_coefficients"] = _write_table(
        path_coefficients, output_dir, "path_coefficients"
    )
    outputs["path_covariances"] = _write_table(
        path_covariances, output_dir, "path_covariances"
    )

    dml_config = methods["dml"]
    dml_partition = str(dml_config.get("fit_partition", "predictive_training"))
    if dml_partition != "predictive_training":
        raise ValueError(
            "dml.fit_partition must be 'predictive_training' so that the DML "
            "sample matches the manuscript's retained 80/20 partition"
        )
    # Reusing the benchmark's split indices makes the DML sample exactly the
    # same 80% training partition described in the methods section.
    dml_frame = frame.iloc[benchmark.train_indices].copy()
    outputs["dml_effects"] = _write_table(
        fit_dml_grid(
            dml_frame,
            outcomes=targets,
            treatments=features,
            nuisance_config=dml_config["nuisance_model"],
            folds=int(dml_config["folds"]),
            seed=seed,
            group_column=dml_config.get("group_column"),
        ),
        output_dir,
        "dml_effects",
    )

    intervention_config = methods["interventions"]
    observed_bounds = {
        feature: [float(frame[feature].min()), float(frame[feature].max())]
        for feature in features
    }
    bounds = (
        observed_bounds
        if bool(intervention_config["clip_to_observed_bounds"])
        else None
    )
    ate, cate = model_based_interventions(
        frame,
        feature_columns=features,
        outcome_columns=targets,
        scenarios=intervention_config["reported_scenarios"],
        subgroup_features=intervention_config["subgroup_features"],
        test_fraction=float(intervention_config["test_fraction"]),
        seed=seed,
        bounds=bounds,
    )
    outputs["intervention_ate"] = _write_table(ate, output_dir, "intervention_ate")
    outputs["intervention_cate"] = _write_table(cate, output_dir, "intervention_cate")
    robustness = intervention_scenarios(
        intervention_config["sensitivity_variables"],
        intervention_config["sensitivity_changes"],
    )
    robustness_ate, robustness_cate = model_based_interventions(
        frame,
        feature_columns=features,
        outcome_columns=targets,
        scenarios=robustness,
        subgroup_features=intervention_config["subgroup_features"],
        test_fraction=float(intervention_config["test_fraction"]),
        seed=seed,
        bounds=bounds,
    )
    outputs["intervention_robustness_ate"] = _write_table(
        robustness_ate, output_dir, "intervention_robustness_ate"
    )
    outputs["intervention_robustness_cate"] = _write_table(
        robustness_cate, output_dir, "intervention_robustness_cate"
    )

    manifest = {
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "code_version": code_version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": _installed_version("xgboost"),
            "codecarbon": _installed_version("codecarbon"),
        },
        "seed": seed,
        "dml_fit_partition": dml_partition,
        "dml_training_rows": int(len(dml_frame)),
        "dataset_sha256": file_sha256(data_path),
        "config_sha256": {
            "reproducibility": file_sha256(args.config.resolve()),
            "manuscript_methods": file_sha256(args.methods_config.resolve()),
        },
        "row_level_exports": False,
        "fitted_models_exported": False,
        "outputs": outputs,
    }
    write_json(output_dir / "methods_run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
