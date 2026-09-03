"""Deterministic reproduction and uncertainty-aware screening LCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSamples:
    gwp: np.ndarray
    energy: np.ndarray

    @property
    def size(self) -> int:
        return int(self.gwp.shape[0])


class ScreeningLCA:
    def __init__(self, config: Mapping[str, object], material_order: Sequence[str]):
        self.config = config
        self.material_order = list(material_order)
        materials = config["materials"]
        missing = [name for name in self.material_order if name not in materials]
        if missing:
            raise ValueError(f"LCA configuration is missing materials: {missing}")
        self.baseline_gwp = np.array(
            [
                materials[name]["gwp_kgco2e_per_kg"]["baseline"]
                for name in self.material_order
            ],
            dtype=float,
        )
        self.baseline_energy = np.array(
            [
                materials[name]["energy_mj_per_kg"]["baseline"]
                for name in self.material_order
            ],
            dtype=float,
        )
        if (
            not np.isfinite(self.baseline_gwp).all()
            or not np.isfinite(self.baseline_energy).all()
        ):
            raise ValueError("Baseline LCA factors must be finite")

    def sample(self, sample_count: int, seed: int) -> FactorSamples:
        if int(sample_count) < 2:
            raise ValueError("At least two Monte Carlo samples are required")
        rng = np.random.default_rng(seed)
        gwp = np.empty((sample_count, len(self.material_order)), dtype=float)
        energy = np.empty_like(gwp)
        for column, material in enumerate(self.material_order):
            record = self.config["materials"][material]
            for target, key in (
                (gwp, "gwp_kgco2e_per_kg"),
                (energy, "energy_mj_per_kg"),
            ):
                factor = record[key]
                low = float(factor["low"])
                mode = float(factor["mode"])
                high = float(factor["high"])
                if not low <= mode <= high or low == high:
                    raise ValueError(
                        f"Invalid triangular {key} bounds for material '{material}'"
                    )
                target[:, column] = rng.triangular(
                    low,
                    mode,
                    high,
                    size=sample_count,
                )
        return FactorSamples(gwp=gwp, energy=energy)

    def _validate_quantities(self, quantities: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(quantities, dtype=float))
        if values.shape[1] != len(self.material_order):
            raise ValueError(
                "Material quantities do not match the configured LCA order"
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Material quantities must be finite and non-negative")
        return values

    def deterministic(self, quantities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        quantities = self._validate_quantities(quantities)
        return quantities @ self.baseline_gwp, quantities @ self.baseline_energy

    @staticmethod
    def moments(samples: FactorSamples) -> Dict[str, np.ndarray]:
        return {
            "gwp_mean": samples.gwp.mean(axis=0),
            "gwp_cov": np.cov(samples.gwp, rowvar=False),
            "energy_mean": samples.energy.mean(axis=0),
            "energy_cov": np.cov(samples.energy, rowvar=False),
        }

    @staticmethod
    def risk_adjusted(
        quantities: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
        risk_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        quantities = np.atleast_2d(np.asarray(quantities, dtype=float))
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if quantities.ndim != 2 or quantities.shape[1] != len(mean):
            raise ValueError(
                "Risk-adjusted LCA quantities and factor means are incompatible"
            )
        if covariance.shape != (len(mean), len(mean)):
            raise ValueError("Risk-adjusted LCA covariance has an incompatible shape")
        if (
            not np.isfinite(quantities).all()
            or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("Risk-adjusted LCA inputs must be finite")
        if np.any(quantities < 0.0) or float(risk_lambda) < 0.0:
            raise ValueError(
                "Quantities and the LCA risk multiplier must be non-negative"
            )
        expected = quantities @ mean
        variance = np.einsum("ij,jk,ik->i", quantities, covariance, quantities)
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        objective = expected + risk_lambda * standard_deviation
        return objective, expected, standard_deviation

    def risk_objectives(
        self,
        quantities: np.ndarray,
        moments: Mapping[str, np.ndarray],
        risk_lambda: float,
    ) -> Dict[str, np.ndarray]:
        gwp, gwp_mean, gwp_sd = self.risk_adjusted(
            quantities, moments["gwp_mean"], moments["gwp_cov"], risk_lambda
        )
        energy, energy_mean, energy_sd = self.risk_adjusted(
            quantities, moments["energy_mean"], moments["energy_cov"], risk_lambda
        )
        return {
            "gwp_risk": gwp,
            "gwp_mean": gwp_mean,
            "gwp_sd": gwp_sd,
            "energy_risk": energy,
            "energy_mean": energy_mean,
            "energy_sd": energy_sd,
        }

    @staticmethod
    def candidate_distributions(
        quantities: np.ndarray, samples: FactorSamples
    ) -> Dict[str, np.ndarray]:
        quantities = np.atleast_2d(np.asarray(quantities, dtype=float))
        if quantities.shape[1] != samples.gwp.shape[1]:
            raise ValueError(
                "Candidate quantities and sampled factors are incompatible"
            )
        if samples.gwp.shape != samples.energy.shape or samples.size < 2:
            raise ValueError(
                "GWP and energy factor samples must have equal non-trivial shape"
            )
        if not np.isfinite(quantities).all() or np.any(quantities < 0.0):
            raise ValueError("Candidate quantities must be finite and non-negative")
        gwp = quantities @ samples.gwp.T
        energy = quantities @ samples.energy.T
        return {
            "gwp_mean": gwp.mean(axis=1),
            "gwp_p05": np.quantile(gwp, 0.05, axis=1),
            "gwp_p95": np.quantile(gwp, 0.95, axis=1),
            "energy_mean": energy.mean(axis=1),
            "energy_p05": np.quantile(energy, 0.05, axis=1),
            "energy_p95": np.quantile(energy, 0.95, axis=1),
        }

    def factor_summary(self, samples: FactorSamples) -> pd.DataFrame:
        rows = []
        for index, material in enumerate(self.material_order):
            record = self.config["materials"][material]
            for metric, matrix, unit in (
                ("gwp", samples.gwp, "kg CO2e/kg"),
                ("energy", samples.energy, "MJ/kg"),
            ):
                values = matrix[:, index]
                rows.append(
                    {
                        "material": material,
                        "metric": metric,
                        "unit": unit,
                        "baseline": float(
                            self.baseline_gwp[index]
                            if metric == "gwp"
                            else self.baseline_energy[index]
                        ),
                        "sample_mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)),
                        "sample_p05": float(np.quantile(values, 0.05)),
                        "sample_p95": float(np.quantile(values, 0.95)),
                        "source": record["source"],
                        "geography": record["geography"],
                        "reference_year": record["reference_year"],
                        "allocation": record["allocation"],
                    }
                )
        return pd.DataFrame(rows)
