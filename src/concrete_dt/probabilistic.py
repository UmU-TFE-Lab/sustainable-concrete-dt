"""Hypothesized DAG utilities and a small discrete Bayesian network engine.

The DAG records domain assumptions; it is not learned as proof of causality.
Continuous variables are discretized by empirical terciles, conditional
probability tables are estimated with optional Laplace smoothing, and queries
are answered by exact variable elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STATE_LABELS = {0: "low", 1: "medium", 2: "high"}


def topological_order(
    nodes: Sequence[str], edges: Sequence[Tuple[str, str]]
) -> List[str]:
    """Return a deterministic topological order and reject cyclic graphs."""
    ordered_nodes = list(dict.fromkeys(nodes))
    if len(set(edges)) != len(edges):
        raise ValueError("A DAG cannot contain duplicate directed edges")
    unknown = {value for edge in edges for value in edge}.difference(ordered_nodes)
    if unknown:
        raise ValueError(f"Edges reference unknown nodes: {sorted(unknown)}")
    children = {node: [] for node in ordered_nodes}
    indegree = {node: 0 for node in ordered_nodes}
    for parent, child in edges:
        if parent == child:
            raise ValueError("A DAG cannot contain self-loops")
        children[parent].append(child)
        indegree[child] += 1
    queue = [node for node in ordered_nodes if indegree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(result) != len(ordered_nodes):
        raise ValueError("The proposed dependency graph contains a directed cycle")
    return result


def concrete_dag(
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    parent_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Build and validate the manuscript's hypothesized input-output DAG."""
    if parent_map is None:
        parent_map = {target: list(feature_columns) for target in target_columns}
    edges = [
        (parent, target) for target, parents in parent_map.items() for parent in parents
    ]
    topological_order(list(feature_columns) + list(target_columns), edges)
    return pd.DataFrame(edges, columns=["parent", "child"])


def quantile_discretize(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discretize continuous variables into low/medium/high using terciles."""
    discrete = pd.DataFrame(index=frame.index)
    threshold_rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing or non-finite values")
        threshold_low, threshold_high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        discrete[column] = np.where(
            values <= threshold_low,
            0,
            np.where(values <= threshold_high, 1, 2),
        ).astype(int)
        threshold_rows.append(
            {
                "variable": column,
                "low_upper_bound": float(threshold_low),
                "medium_upper_bound": float(threshold_high),
            }
        )
    return discrete, pd.DataFrame(threshold_rows)


@dataclass
class Factor:
    variables: Tuple[str, ...]
    table: pd.DataFrame

    def restrict(self, variable: str, state: int) -> "Factor":
        if variable not in self.variables:
            return self
        restricted = self.table[self.table[variable] == state].drop(columns=variable)
        variables = tuple(value for value in self.variables if value != variable)
        return Factor(variables, restricted.reset_index(drop=True))

    def multiply(self, other: "Factor") -> "Factor":
        shared = [value for value in self.variables if value in other.variables]
        if shared:
            merged = self.table.merge(other.table, on=shared, how="inner")
        else:
            merged = (
                self.table.assign(_join_key=1)
                .merge(other.table.assign(_join_key=1), on="_join_key")
                .drop(columns="_join_key")
            )
        merged["probability"] = merged.pop("probability_x") * merged.pop(
            "probability_y"
        )
        variables = tuple(dict.fromkeys(self.variables + other.variables))
        return Factor(variables, merged[list(variables) + ["probability"]])

    def sum_out(self, variable: str) -> "Factor":
        if variable not in self.variables:
            return self
        variables = tuple(value for value in self.variables if value != variable)
        if variables:
            table = self.table.groupby(list(variables), as_index=False, sort=False)[
                "probability"
            ].sum()
        else:
            table = pd.DataFrame({"probability": [self.table["probability"].sum()]})
        return Factor(variables, table)


class DiscreteBayesianNetwork:
    """Discrete Bayesian network with dense, Laplace-smoothed CPTs."""

    def __init__(
        self,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
        n_states: int = 3,
        smoothing: float = 1.0,
    ):
        self.nodes = topological_order(nodes, edges)
        self.edges = list(edges)
        self.n_states = int(n_states)
        self.smoothing = float(smoothing)
        if self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if self.smoothing < 0.0:
            raise ValueError("smoothing cannot be negative")
        self.parents = {
            node: [parent for parent, child in self.edges if child == node]
            for node in self.nodes
        }
        self.factors: Dict[str, Factor] = {}

    def fit(self, data: pd.DataFrame) -> "DiscreteBayesianNetwork":
        missing = set(self.nodes).difference(data.columns)
        if missing:
            raise ValueError(
                f"Bayesian-network data are missing nodes: {sorted(missing)}"
            )
        values = data[self.nodes].to_numpy(dtype=int)
        if np.any(values < 0) or np.any(values >= self.n_states):
            raise ValueError("Bayesian-network states are outside the configured range")
        states = range(self.n_states)
        for node in self.nodes:
            parents = self.parents[node]
            rows = []
            parent_assignments = product(states, repeat=len(parents))
            for parent_values in parent_assignments:
                mask = np.ones(len(data), dtype=bool)
                for parent, value in zip(parents, parent_values):
                    mask &= data[parent].to_numpy(dtype=int) == value
                node_values = data.loc[mask, node].to_numpy(dtype=int)
                counts = np.bincount(node_values, minlength=self.n_states).astype(float)
                denominator = counts.sum() + self.smoothing * self.n_states
                if denominator == 0.0:
                    probabilities = np.full(self.n_states, 1.0 / self.n_states)
                else:
                    probabilities = (counts + self.smoothing) / denominator
                for state, probability in enumerate(probabilities):
                    row = {
                        parent: value for parent, value in zip(parents, parent_values)
                    }
                    row.update({node: state, "probability": float(probability)})
                    rows.append(row)
            variables = tuple(parents + [node])
            self.factors[node] = Factor(
                variables,
                pd.DataFrame(rows, columns=list(variables) + ["probability"]),
            )
        return self

    def query(
        self,
        variable: str,
        evidence: Optional[Mapping[str, int]] = None,
    ) -> np.ndarray:
        """Return P(variable | evidence) using exact variable elimination."""
        if not self.factors:
            raise RuntimeError("Fit the Bayesian network before querying it")
        if variable not in self.nodes:
            raise ValueError(f"Unknown query variable: {variable}")
        evidence = dict(evidence or {})
        unknown = set(evidence).difference(self.nodes)
        if unknown:
            raise ValueError(f"Evidence references unknown nodes: {sorted(unknown)}")
        for node, state in evidence.items():
            if not 0 <= int(state) < self.n_states:
                raise ValueError(
                    f"Evidence state for {node} is outside the valid range"
                )
        if variable in evidence:
            answer = np.zeros(self.n_states)
            answer[int(evidence[variable])] = 1.0
            return answer

        factors = []
        for factor in self.factors.values():
            reduced = factor
            for node, state in evidence.items():
                reduced = reduced.restrict(node, int(state))
            factors.append(reduced)

        # Reverse topological order is efficient for the input-to-output DAG used here.
        hidden = [
            node
            for node in reversed(self.nodes)
            if node != variable and node not in evidence
        ]
        for node in hidden:
            related = [factor for factor in factors if node in factor.variables]
            if not related:
                continue
            factors = [factor for factor in factors if node not in factor.variables]
            combined = related[0]
            for factor in related[1:]:
                combined = combined.multiply(factor)
            factors.append(combined.sum_out(node))

        combined = factors[0]
        for factor in factors[1:]:
            combined = combined.multiply(factor)
        for node in list(combined.variables):
            if node != variable:
                combined = combined.sum_out(node)
        probabilities = np.zeros(self.n_states)
        for _, row in combined.table.iterrows():
            probabilities[int(row[variable])] += float(row["probability"])
        total = probabilities.sum()
        if total <= 0.0:
            raise RuntimeError("Evidence has zero probability under the fitted network")
        return probabilities / total


def fit_concrete_bayesian_network(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    parent_map: Optional[Mapping[str, Sequence[str]]] = None,
    smoothing: float = 1.0,
) -> tuple[DiscreteBayesianNetwork, pd.DataFrame, pd.DataFrame]:
    """Discretize data, fit the hypothesized network, and return its audit tables."""
    edge_frame = concrete_dag(feature_columns, target_columns, parent_map)
    nodes = list(dict.fromkeys(list(feature_columns) + list(target_columns)))
    discrete, thresholds = quantile_discretize(frame, nodes)
    edges = list(edge_frame.itertuples(index=False, name=None))
    network = DiscreteBayesianNetwork(
        nodes=nodes,
        edges=edges,
        n_states=3,
        smoothing=smoothing,
    ).fit(discrete)
    return network, edge_frame, thresholds


def query_table(
    network: DiscreteBayesianNetwork,
    queries: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Evaluate configured marginal or conditional queries as a tidy table."""
    label_to_state = {label: state for state, label in STATE_LABELS.items()}
    rows = []
    for index, query in enumerate(queries):
        variable = str(query["variable"])
        raw_evidence = query.get("evidence", {})
        evidence = {
            name: label_to_state[str(value).lower()]
            if isinstance(value, str)
            else int(value)
            for name, value in raw_evidence.items()
        }
        probabilities = network.query(variable, evidence)
        evidence_text = (
            ";".join(
                f"{name}={STATE_LABELS[state]}"
                for name, state in sorted(evidence.items())
            )
            or "none"
        )
        for state, probability in enumerate(probabilities):
            rows.append(
                {
                    "query_id": str(query.get("id", f"query_{index + 1}")),
                    "variable": variable,
                    "evidence": evidence_text,
                    "state": STATE_LABELS[state],
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows)
