# Analysis Data

`SwedishSustainableConcrete.csv` is the frozen input used by the pipeline, but
it is intentionally **not included** in this source-only release.

- Rows: 1,030 plus header
- SHA-256: `d6db56a6ca7c5fd1722844004c3eac0490ad5155286af06a5ac64bbec7382bd6`
- Analysis grain: one mixture composition observed at one curing age
- Grouping key: exact equality of the seven material quantities

The pipeline verifies the checksum, required columns, deterministic total-mass
identity, and the embedded linear environmental factors before fitting models.
These checks verify computational integrity, not experimental provenance.

Confirm redistribution permission before publishing this copied file. If the
file cannot be redistributed, provide an acquisition link and retain the
checksum and `schema.json` in this directory. Do not add train/test extracts,
row-level predictions, or fitted model binaries to the public source archive.
