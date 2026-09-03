#!/usr/bin/env python3
"""Run the complete reproducible analysis and regenerate all reported outputs."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pymoo
import pyshacl
import rdflib
import sklearn

from concrete_dt.config import load_json, load_run_config, write_json
from concrete_dt import __version__ as code_version
from concrete_dt.data import (
    FEATURE_COLUMNS,
    STRENGTH_COLUMN,
    dataset_audit,
    file_sha256,
    load_dataset,
)
from concrete_dt.kg import build_and_validate_graph
from concrete_dt.modeling import make_strength_model
from concrete_dt.optimization import run_optimization_experiments
from concrete_dt.reporting import generate_all_reports
from concrete_dt.twin import (
    bootstrap_rmse_changes,
    evaluate_decision_impacts,
    evaluate_state_updates,
    summarize_state_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="Local trusted CSV override; the file is read but never copied.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--export-row-level",
        action="store_true",
        help="Write prediction, decision, and candidate rows to the local results folder.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted joblib models locally (excluded from source archives).",
    )
    return parser.parse_args()


def config_digest(path: Path) -> str:
    return file_sha256(path)


def _installed_version(distribution: str) -> str:
    """Return a package version for the machine-readable run manifest."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    if args.data:
        config["data_path"] = str(args.data.resolve())
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.resolve())
    engineering = load_json(Path(config["engineering_config"]))
    lca = load_json(Path(config["lca_config"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(Path(config["data_path"]), engineering)
    cv = config["cross_validation"]
    audit = dataset_audit(frame, cv["early_ages"], cv["target_ages"])
    write_json(output_dir / "dataset_audit.json", audit)

    metrics, predictions, uncertainty = evaluate_state_updates(
        frame=frame,
        seeds=config["seeds"],
        model_config=config["model"],
        outer_folds=int(cv["outer_folds"]),
        inner_folds=int(cv["inner_folds"]),
        early_ages=cv["early_ages"],
        target_ages=cv["target_ages"],
        alpha=float(cv["alpha"]),
    )
    twin_summary = summarize_state_metrics(metrics)
    twin_bootstrap = bootstrap_rmse_changes(predictions)
    required_strengths = {
        float(scenario["required_strength_mpa"])
        for scenario in engineering["scenarios"]
    }
    if len(required_strengths) != 1:
        raise ValueError("Decision-impact replay requires a common strength target")
    decision_records, decision_metrics, decision_summary = evaluate_decision_impacts(
        predictions,
        frame,
        required_strength_mpa=required_strengths.pop(),
    )
    metrics.to_csv(output_dir / "twin_metrics_by_seed.csv", index=False)
    uncertainty.to_csv(output_dir / "twin_uncertainty_calibration.csv", index=False)
    twin_summary.to_csv(output_dir / "twin_metrics_summary.csv", index=False)
    twin_bootstrap.to_csv(output_dir / "twin_bootstrap_rmse_changes.csv", index=False)
    decision_metrics.to_csv(
        output_dir / "twin_decision_impacts_by_seed.csv", index=False
    )
    decision_summary.to_csv(
        output_dir / "twin_decision_impacts_summary.csv", index=False
    )
    if args.export_row_level:
        predictions.to_csv(output_dir / "twin_predictions.csv", index=False)
        decision_records.to_csv(
            output_dir / "twin_decision_impacts_by_mix.csv", index=False
        )

    if args.save_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for seed in config["seeds"]:
            model = make_strength_model(int(seed), config["model"])
            model.fit(frame[FEATURE_COLUMNS], frame[STRENGTH_COLUMN])
            joblib.dump(model, models_dir / f"strength_rf_seed_{seed}.joblib")

    candidates, optimization_runs, optimization_summary, factor_summary, ad_summary = (
        run_optimization_experiments(frame, config, engineering, lca, uncertainty)
    )
    if args.export_row_level:
        candidates.to_csv(output_dir / "pareto_candidates_all.csv", index=False)
    optimization_runs.to_csv(
        output_dir / "optimization_metrics_by_seed.csv", index=False
    )
    optimization_summary.to_csv(
        output_dir / "optimization_metrics_summary.csv", index=False
    )
    factor_summary.to_csv(
        output_dir / "lca_factor_monte_carlo_summary.csv", index=False
    )
    write_json(output_dir / "applicability_domain.json", ad_summary)

    kg_summary = build_and_validate_graph(
        candidates, engineering, lca, ad_summary, output_dir
    )
    generate_all_reports(
        twin_summary,
        decision_summary,
        optimization_summary,
        candidates,
        engineering,
        lca,
        ad_summary,
        kg_summary,
        output_dir,
        representative_seed=int(config["seeds"][0]),
    )

    manifest = {
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "code_version": code_version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "pymoo": pymoo.__version__,
            "rdflib": rdflib.__version__,
            "pyshacl": pyshacl.__version__,
            "scipy": _installed_version("scipy"),
            "joblib": _installed_version("joblib"),
            "owlrl": _installed_version("owlrl"),
        },
        "seeds": config["seeds"],
        "dataset_sha256": file_sha256(Path(config["data_path"])),
        "configuration_sha256": {
            "reproducibility": config_digest(args.config.resolve()),
            "engineering": config_digest(Path(config["engineering_config"])),
            "lca": config_digest(Path(config["lca_config"])),
        },
        "analysis_contract": {
            "pre_observation": "Static strength surrogate plus conformal lower bound",
            "post_observation": "Mix-specific empirical-Bayes residual update",
            "decision_replay": "40 MPa held-out admission and Pareto re-ranking using method-specific conformal lower bounds",
            "environmental_outputs": "Direct screening-LCA calculation",
            "material_output": "Exact sum of constituent masses",
            "deterministic_parallelism": "Random Forest n_jobs fixed at 1 to avoid machine-precision prediction drift",
            "row_level_exports": bool(args.export_row_level),
            "fitted_models_exported": bool(args.save_models),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Generated Results\n\n"
        "Every file in this directory is regenerated by `scripts/run_pipeline.py`.\n"
        "The run manifest records package versions, configuration hashes, data checksum, and seeds.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "twin_summary": str(output_dir / "twin_metrics_summary.csv"),
                "optimization_summary": str(
                    output_dir / "optimization_metrics_summary.csv"
                ),
                "kg_summary": kg_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
