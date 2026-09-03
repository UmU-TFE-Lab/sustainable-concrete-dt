"""Dataset loading, exact-composition grouping, and integrity audits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


MATERIAL_COLUMNS: List[str] = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
]
AGE_COLUMN = "age"
STRENGTH_COLUMN = "concrete_compressive_strength"
GWP_COLUMN = "Embodied_CO2 (kg)"
ENERGY_COLUMN = "Energy_Use (MJ)"
MATERIAL_TOTAL_COLUMN = "Total_Material_Use (kg)"
FEATURE_COLUMNS = MATERIAL_COLUMNS + [AGE_COLUMN]
EXPECTED_SHA256 = "d6db56a6ca7c5fd1722844004c3eac0490ad5155286af06a5ac64bbec7382bd6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mix_identifier(values: Iterable[float]) -> str:
    key = "|".join(f"{float(value):.12g}" for value in values)
    return "mix_" + hashlib.sha1(key.encode("ascii")).hexdigest()[:12]


def load_dataset(path: Path, engineering: Mapping[str, object]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trusted dataset not found: {path}")
    checksum = file_sha256(path)
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"Unexpected dataset checksum: {checksum}")

    frame = pd.read_csv(path)
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicated}")
    required = FEATURE_COLUMNS + [
        STRENGTH_COLUMN,
        GWP_COLUMN,
        ENERGY_COLUMN,
        MATERIAL_TOTAL_COLUMN,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame[required].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(
            "Required analysis columns contain missing or non-finite values"
        )
    if (frame[MATERIAL_COLUMNS] < 0.0).any().any():
        raise ValueError("Material quantities must be non-negative")
    if (frame[AGE_COLUMN] <= 0.0).any():
        raise ValueError("Curing ages must be positive")

    frame["mix_id"] = [
        _mix_identifier(row)
        for row in frame[MATERIAL_COLUMNS].itertuples(index=False, name=None)
    ]
    frame["binder"] = frame[["cement", "blast_furnace_slag", "fly_ash"]].sum(axis=1)
    if (frame["binder"] <= 0.0).any():
        raise ValueError("Every mixture must have positive binder content")
    frame["water_binder_ratio"] = frame["water"] / frame["binder"]
    frame["scm_replacement_ratio"] = (
        frame["blast_furnace_slag"] + frame["fly_ash"]
    ) / frame["binder"]

    densities = engineering["density_kg_m3"]
    air = engineering["screening_rules"]["air_volume_m3_m3"]
    frame["absolute_volume"] = air
    for material in MATERIAL_COLUMNS:
        frame["absolute_volume"] += frame[material] / float(densities[material])
    return frame


def estimate_embedded_factors(frame: pd.DataFrame) -> Dict[str, object]:
    x = frame[MATERIAL_COLUMNS].to_numpy(dtype=float)
    outputs = {
        "gwp": frame[GWP_COLUMN].to_numpy(dtype=float),
        "energy": frame[ENERGY_COLUMN].to_numpy(dtype=float),
        "material": frame[MATERIAL_TOTAL_COLUMN].to_numpy(dtype=float),
    }
    result: Dict[str, object] = {}
    for name, y in outputs.items():
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        prediction = np.einsum("ij,j->i", x, coefficients)
        residual = y - prediction
        result[name] = {
            "coefficients": {
                material: float(value)
                for material, value in zip(MATERIAL_COLUMNS, coefficients)
            },
            "max_absolute_residual": float(np.max(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        }
    return result


def dataset_audit(
    frame: pd.DataFrame, early_ages: Sequence[int], target_ages: Sequence[int]
) -> Dict[str, object]:
    grouped_ages = frame.groupby("mix_id")[AGE_COLUMN].apply(
        lambda s: set(int(v) for v in s)
    )
    repeated = int(sum(len(ages) >= 2 for ages in grouped_ages))
    early_late = int(
        sum(
            any(age in ages for age in early_ages)
            and any(age >= min(target_ages) for age in ages)
            for ages in grouped_ages
        )
    )
    eligible = {
        str(target): int(
            sum(
                target in ages and any(age in ages for age in early_ages)
                for ages in grouped_ages
            )
        )
        for target in target_ages
    }
    mass_error = (
        frame[MATERIAL_COLUMNS].sum(axis=1) - frame[MATERIAL_TOTAL_COLUMN]
    ).abs()
    return {
        "rows": int(len(frame)),
        "unique_mix_groups": int(frame["mix_id"].nunique()),
        "groups_with_two_or_more_ages": repeated,
        "groups_with_early_and_late_observations": early_late,
        "eligible_groups_by_target_age": eligible,
        "age_counts": {
            str(int(age)): int(count)
            for age, count in frame[AGE_COLUMN].value_counts().sort_index().items()
        },
        "maximum_total_material_identity_error": float(mass_error.max()),
        "embedded_factor_audit": estimate_embedded_factors(frame),
    }


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float)
