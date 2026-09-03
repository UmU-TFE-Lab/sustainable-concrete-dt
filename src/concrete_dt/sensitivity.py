"""Saltelli sampling and first-/total-order Sobol sensitivity estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc


@dataclass(frozen=True)
class SobolSample:
    """Paired base matrices using the manuscript's hybrid-matrix convention."""

    a: np.ndarray
    b: np.ndarray

    def hybrid(self, variable_index: int) -> np.ndarray:
        """Return A_B^(i): matrix A with column i replaced by column i of B."""
        if self.a.ndim != 2 or self.a.shape != self.b.shape:
            raise ValueError("Sobol base matrices A and B must have equal 2D shape")
        if not 0 <= variable_index < self.a.shape[1]:
            raise IndexError("variable_index is outside the sample dimension")
        hybrid = self.a.copy()
        hybrid[:, variable_index] = self.b[:, variable_index]
        return hybrid


def saltelli_base_matrices(
    bounds: np.ndarray,
    base_sample_count: int,
    seed: int,
) -> SobolSample:
    """Generate paired low-discrepancy A and B matrices over independent bounds.

    A power-of-two base size preserves the balance properties of Sobol sequences.
    The resulting indices describe the surrogate over an independent rectangular
    domain; they do not recover sensitivity under correlated empirical inputs.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_variables, 2)")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Each upper bound must exceed its lower bound")
    if base_sample_count < 2 or base_sample_count & (base_sample_count - 1):
        raise ValueError("base_sample_count must be a power of two")

    dimension = bounds.shape[0]
    exponent = int(np.log2(base_sample_count))
    unit = qmc.Sobol(d=2 * dimension, scramble=True, seed=seed).random_base2(exponent)
    lower, upper = bounds[:, 0], bounds[:, 1]
    a = qmc.scale(unit[:, :dimension], lower, upper)
    b = qmc.scale(unit[:, dimension:], lower, upper)
    return SobolSample(a=a, b=b)


def _as_output_matrix(values: np.ndarray, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            "Predictor must return shape (n_samples,) or (n_samples, n_outputs)"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Predictor returned non-finite values")
    return matrix


def sobol_indices(
    predict: Callable[[np.ndarray], np.ndarray],
    variable_names: Sequence[str],
    bounds: np.ndarray,
    base_sample_count: int = 1024,
    seed: int = 17,
    output_names: Optional[Sequence[str]] = None,
    numerical_zero_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Estimate Saltelli first-order and Jansen total-order Sobol indices.

    The implementation uses A_B^(i) = A with column i replaced by B. With that
    convention, the covariance numerator is E[f(B)(f(A_B^i)-f(A))], while the
    Jansen total-order numerator is E[(f(A)-f(A_B^i))^2]/2.
    """
    if len(variable_names) != np.asarray(bounds).shape[0]:
        raise ValueError("variable_names and bounds have different dimensions")
    if numerical_zero_tolerance < 0.0:
        raise ValueError("numerical_zero_tolerance cannot be negative")
    samples = saltelli_base_matrices(bounds, base_sample_count, seed)
    y_a = _as_output_matrix(predict(samples.a), base_sample_count)
    y_b = _as_output_matrix(predict(samples.b), base_sample_count)
    n_outputs = y_a.shape[1]
    if y_b.shape[1] != n_outputs:
        raise ValueError("Predictor returned inconsistent output dimensions")
    if output_names is None:
        output_names = [f"output_{i + 1}" for i in range(n_outputs)]
    if len(output_names) != n_outputs:
        raise ValueError("output_names does not match predictor output dimension")

    # Pool A and B to stabilize the common Monte Carlo variance denominator.
    variance = np.var(np.vstack([y_a, y_b]), axis=0, ddof=1)
    if np.any(variance <= np.finfo(float).eps):
        raise ValueError("Sobol indices are undefined for a constant model output")
    rows = []
    for index, variable in enumerate(variable_names):
        y_hybrid = _as_output_matrix(predict(samples.hybrid(index)), base_sample_count)
        first_raw = np.mean(y_b * (y_hybrid - y_a), axis=0) / variance
        total_raw = np.mean(np.square(y_a - y_hybrid), axis=0) / (2.0 * variance)
        for output_index, output in enumerate(output_names):
            first = float(first_raw[output_index])
            total = float(total_raw[output_index])
            first_reported = 0.0 if abs(first) < numerical_zero_tolerance else first
            total_reported = 0.0 if abs(total) < numerical_zero_tolerance else total
            rows.append(
                {
                    "output": output,
                    "variable": variable,
                    "first_order_raw": first,
                    "first_order": first_reported,
                    "total_order_raw": total,
                    "total_order": total_reported,
                    "output_variance": float(variance[output_index]),
                    "base_sample_count": int(base_sample_count),
                    "sampling_assumption": "independent_uniform_bounds",
                }
            )
    return pd.DataFrame(rows)
