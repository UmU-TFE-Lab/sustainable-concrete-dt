from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from concrete_dt.config import load_json
from concrete_dt.constraints import ApplicabilityDomain, EngineeringScreen
from concrete_dt.data import MATERIAL_COLUMNS, dataset_audit, load_dataset
from concrete_dt.kg import build_and_validate_graph
from concrete_dt.lca import ScreeningLCA
from concrete_dt.optimization import center_distance, spacing_metric
from concrete_dt.probabilistic import topological_order
from concrete_dt.twin import VarianceComponents, evaluate_decision_impacts


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = load_json(ROOT / "config" / "engineering_scenarios.json")
LCA_CONFIG = load_json(ROOT / "config" / "lca_factors.json")
DATA_PATH = ROOT / "data" / "SwedishSustainableConcrete.csv"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="trusted dataset is not distributed")
def test_dataset_grouping_and_identities():
    frame = load_dataset(DATA_PATH, ENGINEERING)
    audit = dataset_audit(frame, [3, 7], [28, 56, 90])
    assert audit["rows"] == 1030
    assert audit["unique_mix_groups"] == 427
    assert audit["eligible_groups_by_target_age"] == {"28": 171, "56": 82, "90": 49}
    assert audit["maximum_total_material_identity_error"] < 1e-9
    assert audit["embedded_factor_audit"]["gwp"]["max_absolute_residual"] < 1e-8


def test_lca_reproduces_embedded_coefficients():
    lca = ScreeningLCA(LCA_CONFIG, MATERIAL_COLUMNS)
    x = np.array([[100.0, 20.0, 10.0, 180.0, 5.0, 1000.0, 800.0]])
    gwp, energy = lca.deterministic(x)
    assert np.isclose(gwp[0], x[0] @ lca.baseline_gwp)
    assert np.isclose(energy[0], x[0] @ lca.baseline_energy)
    assert x.sum() == 2115.0


def test_lca_rejects_invalid_uncertainty_bounds():
    invalid = deepcopy(LCA_CONFIG)
    invalid["materials"]["cement"]["gwp_kgco2e_per_kg"]["low"] = 0.90
    lca = ScreeningLCA(invalid, MATERIAL_COLUMNS)
    with pytest.raises(ValueError, match="Invalid triangular"):
        lca.sample(10, seed=17)


def test_applicability_domain_rejects_invalid_neighbor_count():
    quantities = np.ones((3, len(MATERIAL_COLUMNS)))
    with pytest.raises(ValueError, match="k_neighbors"):
        ApplicabilityDomain.fit(quantities, k_neighbors=3, quantile=0.95)


def test_dag_rejects_duplicate_edges():
    with pytest.raises(ValueError, match="duplicate"):
        topological_order(["x", "y"], [("x", "y"), ("x", "y")])


def test_empirical_bayes_shrinkage_is_bounded():
    components = VarianceComponents(tau2=4.0, sigma2=9.0)
    assert 0.0 < components.shrinkage(1) < components.shrinkage(2) < 1.0
    assert VarianceComponents(tau2=0.0, sigma2=1.0).shrinkage(2) == 0.0


def test_pareto_diagnostics_use_declared_geometry():
    objectives = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert np.isclose(center_distance(objectives), 5.0)
    # Equally spaced points have identical nearest-neighbor distances.
    assert np.isclose(spacing_metric(objectives, normalize=False), 0.0)


def test_shacl_validation_matches_numeric_screening(tmp_path):
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_rejected",
                "scenario": "28-day, 40 MPa",
                "stage": "baseline_data_domain",
                "seed": 17,
                "is_feasible": False,
                "water_binder_ratio": 0.40,
                "binder": 400.0,
                "scm_replacement_ratio": 0.30,
                "absolute_volume": 0.84,
                "strength_lower_bound_mpa": 42.0,
                "applicability_distance": 2.0,
                "violated_rules": "absolute_volume;applicability_domain",
            },
            {
                "candidate_id": "candidate_accepted",
                "scenario": "28-day, 40 MPa",
                "stage": "knowledge_robust",
                "seed": 17,
                "is_feasible": True,
                "water_binder_ratio": 0.38,
                "binder": 402.0,
                "scm_replacement_ratio": 0.46,
                "absolute_volume": 1.00,
                "strength_lower_bound_mpa": 43.0,
                "applicability_distance": 0.90,
                "violated_rules": "",
            },
        ]
    )
    summary = build_and_validate_graph(
        candidates,
        ENGINEERING,
        LCA_CONFIG,
        {"distance_threshold": 1.042},
        tmp_path,
    )
    # A graph containing an intentionally rejected candidate does not conform
    # globally, while the SHACL and numerical decisions must still agree.
    assert not summary["combined_graph_conforms"]
    assert summary["decision_shacl_exact_match"]
    assert summary["decision_shacl_agreement"] == 1.0
    assert summary["robust_candidate_conformance_rate"] == 1.0

    second_dir = tmp_path / "second_run"
    second_summary = build_and_validate_graph(
        candidates.iloc[::-1].reset_index(drop=True),
        ENGINEERING,
        LCA_CONFIG,
        {"distance_threshold": 1.042},
        second_dir,
    )
    assert second_summary == summary
    for name in (
        "decision_graph.ttl",
        "decision_graph.jsonld",
        "constraints.shacl.ttl",
        "shacl_report.ttl",
        "shacl_report.txt",
        "validation_summary.json",
    ):
        assert (tmp_path / "knowledge" / name).read_bytes() == (
            second_dir / "knowledge" / name
        ).read_bytes()


@pytest.mark.skipif(not DATA_PATH.exists(), reason="trusted dataset is not distributed")
def test_engineering_screen_flags_clear_violation():
    frame = load_dataset(DATA_PATH, ENGINEERING)
    quantities = frame[MATERIAL_COLUMNS].to_numpy()
    ad = ApplicabilityDomain.fit(quantities, k_neighbors=5, quantile=0.95)
    screen = EngineeringScreen(ENGINEERING, MATERIAL_COLUMNS, ad)
    candidate = np.array([[102.0, 0.0, 0.0, 247.0, 0.0, 801.0, 594.0]])
    result = screen.evaluate(candidate, np.array([20.0]), 5.0, 40.0)
    assert not bool(result.loc[0, "is_feasible"])
    assert "maximum_water_binder_ratio" in result.loc[0, "violated_rules"]


def test_decision_impact_tracks_corrected_admissions():
    predictions = pd.DataFrame(
        {
            "mix_id": ["a", "b", "c", "d"],
            "target_age": [28, 28, 28, 28],
            "seed": [17, 17, 17, 17],
            "actual": [45.0, 42.0, 35.0, 30.0],
            "static_prediction": [43.0, 46.0, 47.0, 30.0],
            "static_interval_half_width": [5.0, 5.0, 5.0, 5.0],
            "eb_prediction": [47.0, 45.0, 38.0, 35.0],
            "eb_interval_half_width": [4.0, 4.0, 4.0, 4.0],
        }
    )
    frame = pd.DataFrame(
        {
            "mix_id": ["a", "b", "c", "d"],
            "age": [28, 28, 28, 28],
            "Embodied_CO2 (kg)": [100.0, 110.0, 120.0, 130.0],
            "Energy_Use (MJ)": [900.0, 950.0, 1000.0, 1050.0],
            "Total_Material_Use (kg)": [2200.0, 2210.0, 2220.0, 2230.0],
        }
    )
    decisions, metrics, summary = evaluate_decision_impacts(
        predictions,
        frame,
        required_strength_mpa=40.0,
    )
    assert int(decisions["decision_corrected"].sum()) == 2
    assert int(decisions["decision_error_introduced"].sum()) == 0
    assert np.isclose(metrics.loc[0, "false_acceptance_rate_static"], 0.5)
    assert np.isclose(metrics.loc[0, "false_acceptance_rate_eb"], 0.0)
    assert np.isclose(metrics.loc[0, "false_rejection_rate_static"], 0.5)
    assert np.isclose(metrics.loc[0, "false_rejection_rate_eb"], 0.0)
    assert int(summary.loc[0, "n_groups"]) == 4
