"""Engineering screening rules and empirical applicability-domain checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RULE_NAMES: List[str] = [
    "minimum_water_binder_ratio",
    "maximum_water_binder_ratio",
    "minimum_binder",
    "maximum_binder",
    "minimum_scm_ratio",
    "maximum_scm_ratio",
    "absolute_volume",
    "strength_lower_bound",
    "applicability_domain",
]


@dataclass
class ApplicabilityDomain:
    scaler: StandardScaler
    neighbors: NearestNeighbors
    threshold: float
    k_neighbors: int

    @classmethod
    def fit(
        cls,
        quantities: np.ndarray,
        k_neighbors: int,
        quantile: float,
    ) -> "ApplicabilityDomain":
        quantities = np.asarray(quantities, dtype=float)
        if quantities.ndim != 2 or not np.isfinite(quantities).all():
            raise ValueError(
                "Applicability-domain quantities must be a finite 2D array"
            )
        if not 1 <= int(k_neighbors) < len(quantities):
            raise ValueError("k_neighbors must satisfy 1 <= k_neighbors < n_samples")
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                "Applicability-domain quantile must lie strictly between 0 and 1"
            )
        scaler = StandardScaler().fit(quantities)
        standardized = scaler.transform(quantities)
        neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(standardized)
        distances, _ = neighbors.kneighbors(standardized)
        mean_distance = distances[:, 1:].mean(axis=1)
        threshold = float(np.quantile(mean_distance, quantile))
        return cls(scaler, neighbors, threshold, k_neighbors)

    def distance(self, quantities: np.ndarray) -> np.ndarray:
        quantities = np.atleast_2d(np.asarray(quantities, dtype=float))
        if quantities.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                "Applicability-domain query has the wrong number of columns"
            )
        if not np.isfinite(quantities).all():
            raise ValueError("Applicability-domain query contains non-finite values")
        standardized = self.scaler.transform(quantities)
        distances, _ = self.neighbors.kneighbors(
            standardized, n_neighbors=self.k_neighbors
        )
        return distances.mean(axis=1)


class EngineeringScreen:
    def __init__(
        self,
        config: Mapping[str, object],
        material_order: Sequence[str],
        applicability: ApplicabilityDomain,
    ):
        self.config = config
        self.material_order = list(material_order)
        required_materials = {
            "cement",
            "blast_furnace_slag",
            "fly_ash",
            "water",
        }
        missing = sorted(required_materials.difference(self.material_order))
        if missing:
            raise ValueError(f"Engineering material order is missing: {missing}")
        self.index = {name: i for i, name in enumerate(self.material_order)}
        self.rules = config["screening_rules"]
        self.densities = np.array(
            [config["density_kg_m3"][name] for name in self.material_order], dtype=float
        )
        if not np.isfinite(self.densities).all() or np.any(self.densities <= 0.0):
            raise ValueError(
                "All configured material densities must be finite and positive"
            )
        self.applicability = applicability

    def derived(self, quantities: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.atleast_2d(np.asarray(quantities, dtype=float))
        if x.shape[1] != len(self.material_order):
            raise ValueError(
                "Candidate quantities do not match the configured material order"
            )
        if not np.isfinite(x).all() or np.any(x < 0.0):
            raise ValueError(
                "Candidate material quantities must be finite and non-negative"
            )
        binder = (
            x[:, self.index["cement"]]
            + x[:, self.index["blast_furnace_slag"]]
            + x[:, self.index["fly_ash"]]
        )
        if np.any(binder <= 0.0):
            raise ValueError("Candidate binder content must be positive")
        scm = x[:, self.index["blast_furnace_slag"]] + x[:, self.index["fly_ash"]]
        water_binder = x[:, self.index["water"]] / binder
        scm_ratio = scm / binder
        volume = x @ (1.0 / self.densities) + float(self.rules["air_volume_m3_m3"])
        distance = self.applicability.distance(x)
        return {
            "binder": binder,
            "water_binder_ratio": water_binder,
            "scm_replacement_ratio": scm_ratio,
            "absolute_volume": volume,
            "applicability_distance": distance,
        }

    def constraint_values(
        self,
        quantities: np.ndarray,
        strength_prediction: np.ndarray,
        strength_half_width: float,
        required_strength: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        d = self.derived(quantities)
        wb_low, wb_high = self.rules["water_binder_ratio"]
        binder_low, binder_high = self.rules["binder_kg_m3"]
        scm_low, scm_high = self.rules["scm_replacement_ratio"]
        volume_tolerance = float(self.rules["absolute_volume_tolerance_m3_m3"])
        strength_lower = np.asarray(strength_prediction) - float(strength_half_width)

        values = np.column_stack(
            [
                (float(wb_low) - d["water_binder_ratio"])
                / max(float(wb_high) - float(wb_low), 1e-9),
                (d["water_binder_ratio"] - float(wb_high))
                / max(float(wb_high) - float(wb_low), 1e-9),
                (float(binder_low) - d["binder"])
                / max(float(binder_high) - float(binder_low), 1e-9),
                (d["binder"] - float(binder_high))
                / max(float(binder_high) - float(binder_low), 1e-9),
                (float(scm_low) - d["scm_replacement_ratio"])
                / max(float(scm_high) - float(scm_low), 1e-9),
                (d["scm_replacement_ratio"] - float(scm_high))
                / max(float(scm_high) - float(scm_low), 1e-9),
                (np.abs(d["absolute_volume"] - 1.0) - volume_tolerance)
                / volume_tolerance,
                (float(required_strength) - strength_lower)
                / max(float(required_strength), 1.0),
                (d["applicability_distance"] - self.applicability.threshold)
                / max(self.applicability.threshold, 1e-9),
            ]
        )
        details = dict(d)
        details["strength_prediction"] = np.asarray(strength_prediction, dtype=float)
        details["strength_lower_bound"] = strength_lower
        details["total_material"] = np.atleast_2d(quantities).sum(axis=1)
        return values, details

    def evaluate(
        self,
        quantities: np.ndarray,
        strength_prediction: np.ndarray,
        strength_half_width: float,
        required_strength: float,
    ) -> pd.DataFrame:
        values, details = self.constraint_values(
            quantities, strength_prediction, strength_half_width, required_strength
        )
        result = pd.DataFrame(details)
        for index, name in enumerate(RULE_NAMES):
            result[f"violation_{name}"] = values[:, index] > 0.0
        result["is_feasible"] = ~(
            result[[f"violation_{name}" for name in RULE_NAMES]].any(axis=1)
        )
        result["violated_rules"] = [
            ";".join(name for i, name in enumerate(RULE_NAMES) if row[i] > 0.0)
            for row in values
        ]
        return result
