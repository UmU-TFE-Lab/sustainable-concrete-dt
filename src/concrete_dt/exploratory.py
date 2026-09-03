"""Descriptive statistics, correlations, clustering, and PCA.

The functions in this module mirror the exploratory methods in the manuscript.
They return tidy aggregate tables by default. Row-level cluster labels and PCA
scores remain available in memory but are not written to the source archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CORRELATION_METHODS = ("pearson", "spearman", "kendall")


def descriptive_statistics(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return manuscript-ready descriptive statistics and missing-value counts."""
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    summary = numeric.describe(percentiles=[0.25, 0.50, 0.75]).T
    summary = summary.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    summary["missing"] = numeric.isna().sum()
    summary.index.name = "variable"
    return summary.reset_index()


def _correlation_pair(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if method == "pearson":
        result = pearsonr(x, y)
    elif method == "spearman":
        result = spearmanr(x, y)
    elif method == "kendall":
        # SciPy's tau-b implementation handles tied concrete quantities.
        result = kendalltau(x, y, variant="b")
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def correlation_analysis(
    frame: pd.DataFrame,
    columns: Sequence[str],
    methods: Sequence[str] = CORRELATION_METHODS,
) -> pd.DataFrame:
    """Compute coefficients and two-sided p-values for all variable pairs.

    Pearson quantifies linear association; Spearman and Kendall quantify
    monotonic/rank association. None of the three establishes causality.
    """
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    rows = []
    for method in methods:
        if method not in CORRELATION_METHODS:
            raise ValueError(f"Unknown correlation method: {method}")
        for i, left in enumerate(columns):
            for right in columns[i:]:
                pair = pd.concat(
                    [numeric[left].rename("x"), numeric[right].rename("y")],
                    axis=1,
                ).dropna()
                coefficient, p_value = _correlation_pair(
                    pair["x"].to_numpy(dtype=float),
                    pair["y"].to_numpy(dtype=float),
                    method,
                )
                rows.append(
                    {
                        "method": method,
                        "variable_x": left,
                        "variable_y": right,
                        "n": int(len(pair)),
                        "coefficient": coefficient,
                        "p_value": p_value,
                    }
                )
                if left != right:
                    rows.append({**rows[-1], "variable_x": right, "variable_y": left})
    return pd.DataFrame(rows)


@dataclass
class ClusteringResult:
    """Aggregate diagnostics plus in-memory labels for downstream plotting."""

    diagnostics: pd.DataFrame
    kmeans_labels: np.ndarray
    agglomerative_labels: np.ndarray
    linkage_matrix: np.ndarray
    scaler: StandardScaler


def clustering_analysis(
    frame: pd.DataFrame,
    columns: Sequence[str],
    k_values: Sequence[int],
    selected_k: Optional[int] = None,
    seed: int = 17,
    n_init: int = 20,
) -> ClusteringResult:
    """Run standardized K-means diagnostics and Ward hierarchical clustering."""
    if not k_values:
        raise ValueError("At least one cluster count is required")
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Clustering input contains missing or non-finite values")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)

    diagnostics = []
    fitted: Dict[int, KMeans] = {}
    for k in sorted(set(int(value) for value in k_values)):
        if k < 2 or k >= len(standardized):
            raise ValueError("Each k must satisfy 2 <= k < number of observations")
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        labels = model.fit_predict(standardized)
        fitted[k] = model
        diagnostics.append(
            {
                "k": k,
                "within_cluster_sum_of_squares": float(model.inertia_),
                "silhouette_score": float(silhouette_score(standardized, labels)),
            }
        )

    if selected_k is None:
        selected_k = int(max(diagnostics, key=lambda row: row["silhouette_score"])["k"])
    if selected_k not in fitted:
        raise ValueError("selected_k must be included in k_values")

    kmeans_labels = fitted[selected_k].labels_.copy()
    agglomerative = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    agglomerative_labels = agglomerative.fit_predict(standardized)
    # SciPy linkage preserves the complete merge history required for a dendrogram.
    linkage_matrix = linkage(standardized, method="ward", metric="euclidean")
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame["selected"] = diagnostic_frame["k"] == selected_k
    return ClusteringResult(
        diagnostics=diagnostic_frame,
        kmeans_labels=kmeans_labels,
        agglomerative_labels=agglomerative_labels,
        linkage_matrix=linkage_matrix,
        scaler=scaler,
    )


@dataclass
class PCAResult:
    """PCA aggregate outputs and optional row-level scores."""

    variance: pd.DataFrame
    loadings: pd.DataFrame
    scores: Optional[pd.DataFrame]
    scaler: StandardScaler
    model: PCA


def principal_component_analysis(
    frame: pd.DataFrame,
    columns: Sequence[str],
    n_components: Optional[int] = None,
    include_scores: bool = False,
) -> PCAResult:
    """Fit PCA to standardized variables.

    Explained-variance ratios are normalized by the sum of all retained input
    eigenvalues, as required by the definition used in the manuscript.
    """
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains missing or non-finite values")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    maximum = min(standardized.shape)
    if n_components is None:
        n_components = maximum
    if not 1 <= int(n_components) <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}")

    model = PCA(n_components=int(n_components), svd_solver="full")
    score_values = model.fit_transform(standardized)
    component_names = [f"PC{i + 1}" for i in range(model.n_components_)]
    variance = pd.DataFrame(
        {
            "component": component_names,
            "eigenvalue": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_),
        }
    )
    loadings = pd.DataFrame(
        model.components_.T,
        index=list(columns),
        columns=component_names,
    )
    loadings.index.name = "variable"
    score_frame = (
        pd.DataFrame(score_values, columns=component_names, index=frame.index)
        if include_scores
        else None
    )
    return PCAResult(variance, loadings.reset_index(), score_frame, scaler, model)


def selected_cluster_summary(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: np.ndarray,
    label_name: str,
) -> pd.DataFrame:
    """Summarize cluster size and feature means without exporting row identities."""
    work = frame.loc[:, columns].copy()
    if len(work) != len(labels):
        raise ValueError("Cluster labels and data have different lengths")
    work[label_name] = np.asarray(labels, dtype=int)
    means = work.groupby(label_name, sort=True)[list(columns)].mean()
    means.insert(0, "n", work.groupby(label_name).size())
    return means.reset_index()
