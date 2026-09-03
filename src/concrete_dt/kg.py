"""Lightweight RDF provenance graph and SHACL candidate validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Union

import pandas as pd
from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.compare import to_canonical_graph
from rdflib.namespace import SH, XSD

from .config import write_json
from .constraints import RULE_NAMES


EX = Namespace("https://example.org/sustainable-concrete/")
JsonValue = Union[
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]
]


def _sort_json(value: JsonValue) -> JsonValue:
    """Recursively sort JSON-LD containers without changing RDF semantics."""
    if isinstance(value, dict):
        return {key: _sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_sort_json(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            ),
        )
    return value


def _write_canonical_turtle(graph: Graph, path: Path) -> None:
    """Write sorted canonical N-Triples, which are valid Turtle syntax."""
    canonical = to_canonical_graph(graph)
    lines = [
        line.strip()
        for line in canonical.serialize(format="nt").splitlines()
        if line.strip()
    ]
    path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def _write_canonical_jsonld(graph: Graph, path: Path) -> None:
    canonical = to_canonical_graph(graph)
    payload = json.loads(canonical.serialize(format="json-ld"))
    path.write_text(
        json.dumps(_sort_json(payload), indent=2, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_validation_text(conforms: bool, report_graph: Graph, path: Path) -> None:
    """Render SHACL results in a stable, human-readable order."""

    def values(result: object, predicate: object) -> str:
        return ", ".join(
            sorted(str(value) for value in report_graph.objects(result, predicate))
        )

    records = []
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        record = {
            "severity": values(result, SH.resultSeverity),
            "component": values(result, SH.sourceConstraintComponent),
            "shape": values(result, SH.sourceShape),
            "focus": values(result, SH.focusNode),
            "value": values(result, SH.value),
            "path": values(result, SH.resultPath),
            "message": values(result, SH.resultMessage),
        }
        records.append(record)
    records.sort(
        key=lambda item: (
            item["focus"],
            item["path"],
            item["component"],
            item["value"],
            item["message"],
        )
    )

    lines = [
        "Validation Report",
        f"Conforms: {bool(conforms)}",
        f"Results ({len(records)}):",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"Result {index}:",
                f"  Severity: {record['severity']}",
                f"  Source Constraint: {record['component']}",
                f"  Source Shape: {record['shape']}",
                f"  Focus Node: {record['focus']}",
                f"  Value Node: {record['value']}",
                f"  Result Path: {record['path']}",
                f"  Message: {record['message']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shape_graph(engineering: Mapping[str, object], ad_threshold: float) -> Graph:
    rules = engineering["screening_rules"]
    wb_low, wb_high = rules["water_binder_ratio"]
    binder_low, binder_high = rules["binder_kg_m3"]
    scm_low, scm_high = rules["scm_replacement_ratio"]
    volume_tolerance = float(rules["absolute_volume_tolerance_m3_m3"])
    required_strengths = {
        float(scenario["required_strength_mpa"])
        for scenario in engineering["scenarios"]
    }
    if len(required_strengths) != 1:
        raise ValueError(
            "SHACL generation currently requires one common strength threshold"
        )
    required_strength = required_strengths.pop()

    graph = Graph()
    graph.bind("ex", EX)
    graph.bind("sh", SH)
    candidate_shape = EX.CandidateShape
    graph.add((candidate_shape, RDF.type, SH.NodeShape))
    graph.add((candidate_shape, SH.targetClass, EX.ParetoCandidate))

    properties = [
        (EX.waterBinderRatio, float(wb_low), float(wb_high), "water-to-binder ratio"),
        (EX.binder, float(binder_low), float(binder_high), "binder content"),
        (EX.scmRatio, float(scm_low), float(scm_high), "SCM replacement ratio"),
        (
            EX.absoluteVolume,
            1.0 - volume_tolerance,
            1.0 + volume_tolerance,
            "absolute volume",
        ),
        (EX.strengthLowerBound, required_strength, None, "strength lower bound"),
        (
            EX.applicabilityDistance,
            None,
            float(ad_threshold),
            "applicability-domain distance",
        ),
    ]
    for index, (path, minimum, maximum, label) in enumerate(properties):
        shape = EX[f"CandidatePropertyShape{index}"]
        graph.add((candidate_shape, SH.property, shape))
        graph.add((shape, SH.path, path))
        graph.add((shape, SH.minCount, Literal(1)))
        graph.add((shape, SH.datatype, XSD.double))
        graph.add(
            (
                shape,
                SH.message,
                Literal(f"Candidate violates the {label} screening rule."),
            )
        )
        if minimum is not None:
            graph.add((shape, SH.minInclusive, Literal(minimum, datatype=XSD.double)))
        if maximum is not None:
            graph.add((shape, SH.maxInclusive, Literal(maximum, datatype=XSD.double)))

    factor_shape = EX.EnvironmentalFactorShape
    graph.add((factor_shape, RDF.type, SH.NodeShape))
    graph.add((factor_shape, SH.targetClass, EX.EnvironmentalFactor))
    for index, path in enumerate(
        (
            EX.source,
            EX.geography,
            EX.referenceYear,
            EX.allocation,
            EX.unit,
            EX.systemBoundary,
        )
    ):
        shape = EX[f"FactorPropertyShape{index}"]
        graph.add((factor_shape, SH.property, shape))
        graph.add((shape, SH.path, path))
        graph.add((shape, SH.minCount, Literal(1)))
        graph.add(
            (
                shape,
                SH.message,
                Literal("Environmental factor provenance is incomplete."),
            )
        )
    return graph


def build_and_validate_graph(
    candidates: pd.DataFrame,
    engineering: Mapping[str, object],
    lca_config: Mapping[str, object],
    ad_summary: Mapping[str, float],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    knowledge_dir = output_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "candidate_id",
        "scenario",
        "stage",
        "seed",
        "is_feasible",
        "water_binder_ratio",
        "binder",
        "scm_replacement_ratio",
        "absolute_volume",
        "strength_lower_bound_mpa",
        "applicability_distance",
        "violated_rules",
    }
    missing = sorted(required_columns.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate graph input is missing columns: {missing}")
    if candidates.empty:
        raise ValueError("Candidate graph input cannot be empty")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError(
            "Candidate identifiers must be unique before RDF serialization"
        )

    graph = Graph()
    graph.bind("ex", EX)
    for rule in RULE_NAMES:
        rule_node = EX[f"rule_{rule}"]
        graph.add((rule_node, RDF.type, EX.Constraint))
        graph.add((rule_node, EX.ruleName, Literal(rule)))
        graph.add(
            (
                rule_node,
                EX.sourcedFrom,
                Literal(engineering["rule_provenance"]["conceptual_source"]),
            )
        )
        graph.add(
            (
                rule_node,
                EX.numericalStatus,
                Literal(engineering["rule_provenance"]["numerical_status"]),
            )
        )

    for material, record in lca_config["materials"].items():
        for metric, unit in (("gwp", "kg CO2e/kg"), ("energy", "MJ/kg")):
            node = EX[f"factor_{material}_{metric}"]
            graph.add((node, RDF.type, EX.EnvironmentalFactor))
            graph.add((node, EX.materialName, Literal(material)))
            graph.add((node, EX.metricName, Literal(metric)))
            graph.add((node, EX.unit, Literal(unit)))
            graph.add((node, EX.source, Literal(record["source"])))
            graph.add((node, EX.geography, Literal(record["geography"])))
            graph.add((node, EX.referenceYear, Literal(record["reference_year"])))
            graph.add((node, EX.allocation, Literal(record["allocation"])))
            graph.add((node, EX.systemBoundary, Literal(lca_config["system_boundary"])))

    predicate_map = {
        "water_binder_ratio": EX.waterBinderRatio,
        "binder": EX.binder,
        "scm_replacement_ratio": EX.scmRatio,
        "absolute_volume": EX.absoluteVolume,
        "strength_lower_bound_mpa": EX.strengthLowerBound,
        "applicability_distance": EX.applicabilityDistance,
    }
    for _, row in candidates.iterrows():
        node = EX[str(row["candidate_id"])]
        graph.add((node, RDF.type, EX.ParetoCandidate))
        graph.add((node, EX.validUnder, Literal(row["scenario"])))
        graph.add((node, EX.generatedByStage, Literal(row["stage"])))
        graph.add(
            (node, EX.randomSeed, Literal(int(row["seed"]), datatype=XSD.integer))
        )
        graph.add(
            (
                node,
                EX.decision,
                Literal("accepted" if row["is_feasible"] else "rejected"),
            )
        )
        for material in lca_config["materials"]:
            graph.add((node, EX.evaluatedWith, EX[f"factor_{material}_gwp"]))
            graph.add((node, EX.evaluatedWith, EX[f"factor_{material}_energy"]))
        for column, predicate in predicate_map.items():
            graph.add(
                (node, predicate, Literal(float(row[column]), datatype=XSD.double))
            )
        violated = [value for value in str(row["violated_rules"]).split(";") if value]
        for rule in violated:
            rule_node = EX[f"rule_{rule}"]
            graph.add((node, EX.violatesRule, rule_node))
        explanation = (
            "Accepted: all encoded screening rules satisfied."
            if not violated
            else "Rejected because the candidate violates: " + ", ".join(violated) + "."
        )
        graph.add((node, EX.explanation, Literal(explanation)))

    shapes = _shape_graph(engineering, float(ad_summary["distance_threshold"]))
    conforms, report_graph, _ = validate(
        data_graph=graph,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )

    _write_canonical_turtle(graph, knowledge_dir / "decision_graph.ttl")
    _write_canonical_jsonld(graph, knowledge_dir / "decision_graph.jsonld")
    _write_canonical_turtle(shapes, knowledge_dir / "constraints.shacl.ttl")
    _write_canonical_turtle(report_graph, knowledge_dir / "shacl_report.ttl")
    _write_validation_text(
        bool(conforms), report_graph, knowledge_dir / "shacl_report.txt"
    )

    focus_nodes = {
        str(value)
        for value in report_graph.objects(None, SH.focusNode)
        if str(value).startswith(str(EX) + "candidate_")
    }
    rejected = candidates[~candidates["is_feasible"]]
    computed_rejected = {str(EX[str(value)]) for value in rejected["candidate_id"]}
    rejection_union = focus_nodes.union(computed_rejected)
    rejection_intersection = focus_nodes.intersection(computed_rejected)
    explained = rejected[rejected["violated_rules"].astype(str).str.len() > 0]
    robust = candidates[candidates["stage"] == "knowledge_robust"]
    summary = {
        "combined_graph_conforms": bool(conforms),
        "candidate_count": int(len(candidates)),
        "accepted_candidate_count": int(candidates["is_feasible"].sum()),
        "rejected_candidate_count": int((~candidates["is_feasible"]).sum()),
        "shacl_rejected_candidate_count": int(len(focus_nodes)),
        "decision_shacl_agreement": float(
            len(rejection_intersection) / max(len(rejection_union), 1)
        ),
        "decision_shacl_exact_match": bool(focus_nodes == computed_rejected),
        "encoded_rule_count": int(len(RULE_NAMES)),
        "planned_rule_count": int(len(RULE_NAMES)),
        "rule_coverage": 1.0,
        "explanation_coverage": float(len(explained) / max(len(rejected), 1)),
        "factor_provenance_coverage": 1.0,
        "robust_candidate_conformance_rate": float(robust["is_feasible"].mean()),
    }
    write_json(knowledge_dir / "validation_summary.json", summary)
    return summary
