# Sustainable Concrete: Complete Methods Code

This source package implements the computational methods used in the
sustainable-concrete manuscript. Coverage includes descriptive statistics;
Pearson, Spearman, and Kendall-Tau correlations; Saltelli-Jansen Sobol analysis;
K-means and Ward clustering; PCA; multi-output Random Forest, XGBoost, and DNN
benchmarks; MLR; a hypothesized DAG and discrete Bayesian network; observed-
variable path regression; cross-fitted DML; model-based interventions; the
empirical-Bayes material twin; engineering constraints; screening LCA;
constrained robust NSGA-III; and RDF/JSON-LD plus SHACL validation.

See `METHODS_CODE_MAP.md` for a section-by-section map from the manuscript to
the implementing function and configuration block.

The implementation is a retrospective decision-support prototype. It does not
claim a live sensor connection, automated batching control, measured workability
or durability verification, or a complete service-life LCA.

## Install

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-full.txt
.venv/bin/python -m pip install -e .
```

`requirements-full.txt` adds XGBoost and CodeCarbon. The DNN is implemented
with scikit-learn's `MLPRegressor` using the stated two 64-unit ReLU layers,
Adam settings, batch size, and epoch limit, avoiding a second deep-learning
runtime solely for this compact architecture.

## Reproduce

Keep the trusted CSV outside the publishable source tree and pass it explicitly:

```bash
.venv/bin/python scripts/run_all.py \
  --data /local/path/SwedishSustainableConcrete.csv
```

The statistical/ML stage can be run separately:

```bash
.venv/bin/python scripts/run_manuscript_methods.py \
  --config config/reproducibility.json \
  --methods-config config/manuscript_methods.json \
  --data /local/path/SwedishSustainableConcrete.csv
```

The digital-twin and decision stage can likewise be run separately:

```bash
.venv/bin/python scripts/run_pipeline.py \
  --config config/reproducibility.json \
  --data /local/path/SwedishSustainableConcrete.csv
```

Both commands verify the trusted file's SHA-256 checksum before analysis. The
default interfaces do not save fitted models or row-level prediction files.
Use `--export-row-level` or `--save-models` only for private local diagnostics;
those outputs are excluded from the source archive.

Run the checks with:

```bash
.venv/bin/python -m pytest -q
```

All stochastic routines use the seeds declared in the configuration files.
The Random Forest worker count is fixed at one because parallel floating-point
reduction can change predictions at machine precision and perturb tie-breaking
inside NSGA-III. This setting favors byte-stable analytical outputs over
training speed; the manifest timestamp changes on each execution by design.

## Analysis Contract

The design and state-update stages are deliberately separated:

1. Before a new mixture has early-age observations, NSGA-III uses the static
   strength surrogate, its group-cross-validated lower prediction bound,
   engineering constraints, and screening LCA uncertainty.
2. After 3- or 7-day strength is observed for a selected mixture, the
   mix-specific residual state updates that mixture's later-age forecast and
   acceptance probability. The offset is not transferred to arbitrary unseen
   mixtures.

Environmental and total-material indicators are calculated directly from
material quantities. They are not fitted as machine-learning targets because
the supplied dataset encodes them as deterministic linear functions.

## Repository Layout

- `config/`: versioned model, engineering, and LCA assumptions.
- `data/`: schema and checksum documentation; no observations are distributed.
- `src/concrete_dt/`: reusable analysis modules.
- `scripts/`: complete, staged, and source-archive entry points.
- `tests/`: synthetic unit tests plus optional trusted-data integration checks.
- `results/`: local generated outputs; contents are not included in the ZIP.

## Data and LCA Boundaries

The analysis reads `SwedishSustainableConcrete.csv` as a trusted reference
dataset. The data-embedded environmental coefficients are preserved
as a deterministic reproduction baseline. The uncertainty configuration is a
transparent screening envelope, not a region-specific verified EPD model. Each
factor records its scope and evidence status in `config/lca_factors.json`.

The source archive intentionally excludes the CSV, train/test subsets,
row-level predictions, fitted models, and generated result tables. Build and
verify it with:

```bash
.venv/bin/python scripts/build_source_archive.py
```

The command performs a CRC check, rejects blocked data/model suffixes, and
writes a SHA-256 sidecar next to the ZIP.

For a versioned repository upload, set both the output file and internal root
directory explicitly:

```bash
.venv/bin/python scripts/build_source_archive.py \
  --output dist/sustainable-concrete-dt-v0.2.0-source.zip \
  --root-name sustainable-concrete-dt-v0.2.0
```

## Release Status

Version 0.2.0 is a source-only release licensed under the MIT License; see
`LICENSE`. It may be placed in an unpublished repository draft for metadata and
file validation. Public release still requires confirmation that the numerical
results align with the submitted manuscript and that public creator metadata is
compatible with the journal's review policy.
