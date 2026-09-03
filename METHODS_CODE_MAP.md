# Manuscript-to-Code Map

This table maps every computational method in the manuscript to an executable
implementation. Functions return row-level quantities only in memory unless a
caller explicitly requests an export. The prepared source archive excludes all
CSV files, train/test extracts, predictions, and fitted models.

| Manuscript method | Implementation | Primary configuration |
|---|---|---|
| Data integrity and exact-composition grouping | `concrete_dt.data` | `config/reproducibility.json` |
| Descriptive statistics | `exploratory.descriptive_statistics` | `analysis_columns` |
| Pearson, Spearman, Kendall-Tau | `exploratory.correlation_analysis` | `correlation` |
| Saltelli sampling | `sensitivity.saltelli_base_matrices` | `sobol` |
| First-order and Jansen total-order Sobol indices | `sensitivity.sobol_indices` | `sobol` |
| K-means, elbow, and silhouette analysis | `exploratory.clustering_analysis` | `clustering` |
| Ward hierarchical clustering and linkage | `exploratory.clustering_analysis` | `clustering` |
| PCA, loadings, and explained variance | `exploratory.principal_component_analysis` | `pca` |
| Multi-output Random Forest | `predictive.build_predictive_model` | `predictive.model_parameters.random_forest` |
| Multi-output XGBoost | `predictive.build_predictive_model` | `predictive.model_parameters.xgboost` |
| Two-layer ReLU DNN with Adam | `predictive.build_predictive_model` | `predictive.model_parameters.dnn` |
| MSE, RMSE, and R-squared | `predictive.benchmark_multioutput_models` | `predictive` |
| Runtime and optional CodeCarbon accounting | `predictive.benchmark_multioutput_models` | `predictive.track_energy` |
| Multiple linear regression | `effects.fit_mlr_models` | `feature_columns`, `target_columns` |
| Hypothesized DAG | `probabilistic.concrete_dag` | `bayesian_network.parent_map` |
| Tercile Bayesian network and variable elimination | `probabilistic.fit_concrete_bayesian_network` | `bayesian_network` |
| Observed-variable path model | `effects.fit_observed_path_model` | `path_model` |
| Cross-fitted partially linear DML | `effects.partially_linear_dml` | `dml` |
| ATE, subgroup CATE, and robustness interventions | `effects.model_based_interventions` | `interventions` |
| Empirical-Bayes material-twin update | `twin.evaluate_state_updates` | `cross_validation`, `model` |
| Admission and Pareto re-ranking impact | `twin.evaluate_decision_impacts` | engineering scenarios |
| Engineering constraints and applicability domain | `constraints.EngineeringScreen` | `engineering_scenarios.json` |
| Deterministic and Monte Carlo screening LCA | `lca.ScreeningLCA` | `lca_factors.json` |
| Baseline/constrained/robust NSGA-III | `optimization.run_optimization_experiments` | `optimization` |
| Original eight-variable data-domain NSGA-III | `optimization.run_legacy_data_domain_optimization` | legacy control-parameter arguments |
| Center distance, spacing, and convergence trace | `optimization.center_distance`, `optimization.spacing_metric`, `optimization.ParetoConvergenceTrace` | declared objective scales |
| RDF/JSON-LD graph and SHACL validation | `kg.build_and_validate_graph` | engineering and LCA configurations |
| Optional RAG extension shown in the architecture | Not implemented or executed; no numerical result depends on it | not applicable |

## Important Estimand Notes

- The Sobol implementation uses exactly the manuscript convention
  `A_B^(i) = A` with column `i` replaced by column `i` of `B`. It reports the
  Saltelli covariance estimator for first-order effects and the Jansen
  squared-difference estimator for total-order effects. The sampled rectangular
  domain assumes independent inputs; indices are global surrogate sensitivities,
  not physical causal effects under correlated empirical mix proportions.
- `partially_linear_dml` implements the constant-effect equation printed in the
  manuscript. The top-level runner fits it on the same 80% training partition
  used by the predictive benchmark and performs two-fold cross-fitting within
  that partition. Historical files in the working directory appear to have been
  generated with `econml.LinearDML` and row-varying effect estimates. Those old
  numbers should not be presented as a reproduction of the printed constant-
  effect formula without retaining the original fitting script and settings.
- Intervention ATE is computed as intervention prediction minus the model-
  predicted baseline for the same test profile. CATE is an explicit average
  within low/medium/high subgroups of the configured feature. Values with
  different output units are never pooled.
- The path model contains observed variables only. The code does not invent
  latent-variable SEM fit indices such as CFI, TLI, or RMSEA.
- The legacy eight-variable optimizer is disabled in the default configuration
  because it is a costly historical baseline and its old center-distance plot
  did not preserve the objective scaling vector. Set `enabled` to `true` to
  rerun NSGA-III, and declare `center_distance_scales` before claiming numerical
  reproduction of that diagnostic. The constrained fixed-age experiments do
  not depend on this legacy trace.
- GWP, energy, and total material are retained as predictive targets only for
  reproducing the manuscript's model-comparison section. In the decision layer,
  they are calculated directly from constituent quantities and LCA factors.
- The package makes no network, language-model, or retrieval call. The RAG path
  in the conceptual figure remains a human-verified future extension and has no
  authority over candidate optimization or acceptance.

## Entry Points

`scripts/run_manuscript_methods.py` executes the statistical, unsupervised,
predictive, probabilistic, path, DML, and intervention sections. It writes only
aggregate tables. `scripts/run_pipeline.py` executes the digital-twin,
engineering-screening, LCA, NSGA-III, knowledge-graph, and SHACL sections.
`scripts/run_all.py` runs both stages in sequence.
