# Changelog

## 0.2.0 - 2026-09-03

- Added executable coverage for the manuscript's exploratory statistics,
  correlations, Sobol analysis, clustering, PCA, RF/XGBoost/DNN benchmark,
  DAG, Bayesian network, MLR, observed-variable path model, DML, and
  intervention sections.
- Retained and documented the empirical-Bayes material-state update,
  engineering screening, uncertain LCA, NSGA-III, knowledge graph, and SHACL
  decision pipeline.
- Standardized DML on the printed constant-effect partially linear estimand,
  input-only controls, the shared 80% predictive-training partition, and
  two-fold cross-fitting.
- Added explicit subgroup CATE definitions and model-predicted counterfactual
  baselines for intervention summaries.
- Added input/configuration validation, dependency/version manifests,
  deterministic archive generation, and synthetic tests.
- Pinned optional XGBoost and CodeCarbon versions and added deterministic
  RDF/JSON-LD/SHACL serialization across independent processes.
- Changed default exports so fitted models, train/test rows, row-level
  predictions, and complete candidate tables are not written unless explicitly
  requested for private diagnostics.
- Added a source-only archive policy that excludes datasets, generated results,
  binary models, caches, and local environment artifacts.
- Adopted the MIT License for the source-code release.

## 0.1.0 - 2026-08-18

- Initial reproducibility workflow for grouped strength forecasting,
  empirical-Bayes updating, engineering constraints, screening LCA,
  multi-objective optimization, and knowledge-graph validation.
