import numpy as np
import pandas as pd

from concrete_dt.effects import (
    model_based_interventions,
    partially_linear_dml,
)
from concrete_dt.exploratory import (
    correlation_analysis,
    principal_component_analysis,
)
from concrete_dt.predictive import benchmark_multioutput_models
from concrete_dt.probabilistic import DiscreteBayesianNetwork
from concrete_dt.sensitivity import sobol_indices


def test_correlation_and_full_pca_are_well_formed():
    frame = pd.DataFrame(
        {
            "x": np.arange(1.0, 11.0),
            "y": np.arange(1.0, 11.0) * 2.0,
            "z": np.array([0.0, 1.0] * 5),
        }
    )
    correlations = correlation_analysis(frame, ["x", "y", "z"])
    xy = correlations[
        (correlations["method"] == "pearson")
        & (correlations["variable_x"] == "x")
        & (correlations["variable_y"] == "y")
    ].iloc[0]
    assert np.isclose(xy["coefficient"], 1.0)

    pca = principal_component_analysis(frame, ["x", "y", "z"])
    assert np.isclose(pca.variance["explained_variance_ratio"].sum(), 1.0)
    assert pca.scores is None


def test_saltelli_jansen_indices_recover_additive_linear_importance():
    def predictor(values):
        return values[:, 0] + 2.0 * values[:, 1]

    result = sobol_indices(
        predictor,
        variable_names=["x1", "x2"],
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        base_sample_count=4096,
        seed=17,
        output_names=["y"],
    ).set_index("variable")
    assert abs(result.loc["x1", "first_order"] - 0.2) < 0.03
    assert abs(result.loc["x2", "first_order"] - 0.8) < 0.03
    assert abs(result.loc["x1", "total_order"] - 0.2) < 0.03
    assert abs(result.loc["x2", "total_order"] - 0.8) < 0.03


def test_bayesian_network_query_is_normalized_and_responds_to_evidence():
    data = pd.DataFrame(
        {
            "x": [0] * 30 + [1] * 30 + [2] * 30,
            "y": [0] * 25 + [1] * 5 + [0] * 5 + [1] * 20 + [2] * 5 + [1] * 5 + [2] * 25,
        }
    )
    network = DiscreteBayesianNetwork(
        nodes=["x", "y"], edges=[("x", "y")], smoothing=1.0
    ).fit(data)
    marginal = network.query("y")
    given_high = network.query("y", {"x": 2})
    assert np.isclose(marginal.sum(), 1.0)
    assert np.isclose(given_high.sum(), 1.0)
    assert given_high[2] > marginal[2]


def test_cross_fitted_dml_recovers_known_partial_linear_effect():
    rng = np.random.default_rng(29)
    n = 700
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treatment = 0.8 * x1 - 0.4 * x2 + rng.normal(scale=0.8, size=n)
    outcome = 2.0 * treatment + np.sin(x1) + 0.5 * x2 + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(
        {"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome}
    )
    result = partially_linear_dml(
        frame,
        outcome="outcome",
        treatment="treatment",
        controls=["x1", "x2"],
        nuisance_config={"n_estimators": 60, "min_samples_leaf": 4, "n_jobs": 1},
        folds=3,
        seed=17,
    ).iloc[0]
    assert abs(result["estimate"] - 2.0) < 0.2
    assert result["ci_lower"] < 2.0 < result["ci_upper"]


def test_intervention_uses_model_predicted_baseline():
    rng = np.random.default_rng(43)
    x = rng.uniform(10.0, 20.0, size=300)
    context = rng.normal(size=300)
    y = 3.0 * x + context
    frame = pd.DataFrame({"x": x, "context": context, "y": y})
    ate, cate = model_based_interventions(
        frame,
        feature_columns=["x", "context"],
        outcome_columns=["y"],
        scenarios=[{"id": "x_plus_10", "changes": {"x": 0.1}}],
        subgroup_features=["context"],
        seed=17,
    )
    assert ate.loc[0, "baseline"] == "model_prediction_same_profile"
    assert 4.0 < ate.loc[0, "ate"] < 5.0
    assert set(cate["subgroup"]) == {"low", "medium", "high"}


def test_predictive_benchmark_does_not_return_prediction_rows():
    rng = np.random.default_rng(71)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
        }
    )
    frame["y1"] = 2.0 * frame["x1"] - frame["x2"]
    frame["y2"] = frame["x1"] + 0.5 * frame["x2"]
    result = benchmark_multioutput_models(
        frame,
        feature_columns=["x1", "x2"],
        target_columns=["y1", "y2"],
        config={
            "models": ["random_forest"],
            "test_fraction": 0.2,
            "track_energy": False,
            "model_parameters": {"random_forest": {"n_estimators": 20, "n_jobs": 1}},
        },
        seed=17,
    )
    assert len(result.metrics) == 2
    assert list(result.models) == ["random_forest"]
    assert not hasattr(result, "predictions")
