"""Figures and compact manuscript tables generated from saved results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0057B8"
METHOD_LABELS = {
    "static": "Static surrogate",
    "raw": "Raw residual update",
    "eb": "Empirical-Bayes update",
}
STAGE_LABELS = {
    "baseline_data_domain": "Baseline",
    "knowledge_deterministic": "Engineering-screened",
    "knowledge_robust": "Robust LCA + screening",
}


def _style_axes(ax):
    ax.tick_params(colors=BLUE)
    ax.xaxis.label.set_color(BLUE)
    ax.yaxis.label.set_color(BLUE)
    ax.title.set_color(BLUE)
    for spine in ax.spines.values():
        spine.set_color(BLUE)
    ax.grid(axis="y", color="#D9E5F2", linewidth=0.7)


def plot_twin_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    ages = sorted(summary["target_age"].unique())
    methods = ["static", "raw", "eb"]
    x = np.arange(len(ages))
    width = 0.24
    colors = ["#777777", "#4C9F70", "#E69F00"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for index, (method, color) in enumerate(zip(methods, colors)):
        subset = summary[summary["method"] == method].set_index("target_age").loc[ages]
        ax.bar(
            x + (index - 1) * width,
            subset["rmse_mean"],
            width,
            yerr=subset["rmse_sd"].fillna(0.0),
            capsize=3,
            label=METHOD_LABELS[method],
            color=color,
            edgecolor="white",
        )
    ax.set_xticks(x, [f"{age} days" for age in ages])
    ax.set_ylabel("Group-held-out RMSE (MPa)")
    ax.set_xlabel("Forecast age")
    ax.legend(frameon=False, labelcolor=BLUE, ncol=1)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_decision_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    scenarios = list(dict.fromkeys(summary["scenario"]))
    stages = ["baseline_data_domain", "knowledge_deterministic", "knowledge_robust"]
    x = np.arange(len(scenarios))
    width = 0.24
    colors = ["#777777", "#4C9F70", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.25))
    for index, (stage, color) in enumerate(zip(stages, colors)):
        subset = summary[summary["stage"] == stage].set_index("scenario").loc[scenarios]
        offset = x + (index - 1) * width
        axes[0].bar(
            offset,
            100.0 * subset["infeasible_ratio_mean"],
            width,
            color=color,
            label=STAGE_LABELS[stage],
        )
        axes[1].bar(
            offset,
            subset["feasible_hypervolume_mean"],
            width,
            color=color,
            label=STAGE_LABELS[stage],
        )
    labels = [scenario.replace("target40_age", "") + " d" for scenario in scenarios]
    axes[0].set_xticks(x, labels)
    axes[1].set_xticks(x, labels)
    axes[0].set_ylabel("Candidates rejected by screening (%)")
    axes[1].set_ylabel("Feasible hypervolume")
    axes[0].set_xlabel("40 MPa target scenario")
    axes[1].set_xlabel("40 MPa target scenario")
    for ax in axes:
        _style_axes(ax)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        labelcolor=BLUE,
        fontsize=8,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robust_pareto(candidates: pd.DataFrame, output_path: Path, seed: int) -> None:
    subset = candidates[
        (candidates["seed"] == seed)
        & (candidates["scenario"] == "target40_age28")
        & (candidates["stage"] == "knowledge_robust")
        & candidates["is_feasible"]
    ]
    if subset.empty:
        raise ValueError("No feasible 28-day robust candidates are available to plot")
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    scatter = ax.scatter(
        subset["gwp_risk_kgco2e_m3"],
        subset["strength_prediction_mpa"],
        c=subset["energy_risk_mj_m3"],
        s=38,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Risk-adjusted GWP (kg CO$_2$e/m$^3$)")
    ax.set_ylabel("Predicted 28-day strength (MPa)")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Risk-adjusted energy (MJ/m$^3$)", color=BLUE)
    colorbar.ax.tick_params(colors=BLUE)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write(path: Path, content: str) -> None:
    Path(path).write_text(content.rstrip() + "\n", encoding="utf-8")


def _format_or_dash(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def write_twin_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.sort_values(["target_age", "method"]).iterrows():
        rows.append(
            f"{int(row.target_age)} & {METHOD_LABELS[row.method]} & {int(row.n_groups)} & "
            f"{row.rmse_mean:.2f} $\\pm$ {row.rmse_sd:.2f} & "
            f"{row.mae_mean:.2f} $\\pm$ {row.mae_sd:.2f} & "
            f"{row.picp_mean:.3f} & {row.rmse_change_percent_vs_static:+.1f}\\% \\\\"
        )
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Group-held-out ablation of retrospective strength-state updating across five declared seeds.}",
                "\\label{tab:dt_replay_ablation}",
                "\\small",
                "\\begin{tabular}{llrrrrr}",
                "\\toprule",
                "Age & Method & Groups & RMSE (MPa) & MAE (MPa) & PICP & $\\Delta$RMSE \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        ),
    )


def write_twin_decision_impact_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.sort_values("target_age").iterrows():
        rows.append(
            f"{int(row.target_age)} & {int(row.n_groups)} & "
            f"{100.0 * row.admission_change_rate_mean:.1f} & "
            f"{100.0 * row.false_acceptance_rate_static_mean:.1f} $\\rightarrow$ "
            f"{100.0 * row.false_acceptance_rate_eb_mean:.1f} & "
            f"{100.0 * row.false_rejection_rate_static_mean:.1f} $\\rightarrow$ "
            f"{100.0 * row.false_rejection_rate_eb_mean:.1f} & "
            f"{row.pareto_jaccard_mean:.3f} & "
            f"{row.rank_kendall_tau_mean:.3f} \\\\"
        )
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Decision impact of empirical-Bayes material-state updating across five group-held-out seeds. Rates are averaged across seeds.}",
                "\\label{tab:dt_decision_impact}",
                "\\small",
                "\\begin{tabular}{rrrrrrr}",
                "\\toprule",
                "Age & Groups & Changed (\\%) & False accept. (\\%) & False reject. (\\%) & $J_{\\mathrm{Pareto}}$ & Rank $\\tau$ \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        ),
    )


def write_decision_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.sort_values(["scenario", "stage"]).iterrows():
        age = int(str(row.scenario).replace("target40_age", ""))
        rows.append(
            f"{age} & {STAGE_LABELS[row.stage]} & "
            f"{100.0 * row.infeasible_ratio_mean:.1f} & "
            f"{_format_or_dash(row.feasible_hypervolume_mean)} & "
            f"{_format_or_dash(row.spacing_mean)} & "
            f"{_format_or_dash(row.modal_choice_stability_mean)} \\\\"
        )
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Decision-level ablation across five NSGA-III seeds for the 40 MPa screening scenarios.}",
                "\\label{tab:decision_ablation}",
                "\\small",
                "\\begin{tabular}{llrrrr}",
                "\\toprule",
                "Age & Stage & Rejected (\\%) & Feasible HV & Spacing & Choice stability \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        ),
    )


def write_constraint_table(
    engineering: Mapping[str, object], ad_summary: Mapping[str, float], path: Path
) -> None:
    rules = engineering["screening_rules"]
    rows = [
        f"Water-to-binder ratio & {rules['water_binder_ratio'][0]:.2f}--{rules['water_binder_ratio'][1]:.2f} & Engineering screening assumption \\\\",
        f"Binder content & {rules['binder_kg_m3'][0]:.0f}--{rules['binder_kg_m3'][1]:.0f} kg/m$^3$ & Engineering screening assumption \\\\",
        f"SCM replacement & {rules['scm_replacement_ratio'][0]:.2f}--{rules['scm_replacement_ratio'][1]:.2f} & Engineering screening assumption \\\\",
        f"Absolute volume & $1.000\\pm{rules['absolute_volume_tolerance_m3_m3']:.3f}$ m$^3$/m$^3$ & Representative densities and {rules['air_volume_m3_m3']:.2f} air volume \\\\",
        "Strength requirement & 40 MPa at 28, 56, or 90 days & Scenario-specific one-sided conformal lower bound \\\\",
        f"Applicability domain & $d_{{5NN}}\\leq{ad_summary['distance_threshold']:.3f}$ & {100.0 * ad_summary['empirical_quantile']:.0f}th percentile of empirical 5-NN distance \\\\",
    ]
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Numerical engineering feasibility-screening configuration used in the reproducible experiments.}",
                "\\label{tab:engineering_screening_configuration}",
                "\\small",
                "\\begin{tabularx}{0.96\\linewidth}{l l X}",
                "\\toprule",
                "Rule & Numerical setting & Status/source \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabularx}",
                "\\end{table}",
            ]
        ),
    )


def write_kg_table(summary: Mapping[str, object], path: Path) -> None:
    rows = [
        f"Encoded engineering rules & {summary['encoded_rule_count']}/{summary['planned_rule_count']} \\\\",
        f"Candidate nodes & {summary['candidate_count']} \\\\",
        f"Accepted/rejected candidates & {summary['accepted_candidate_count']}/{summary['rejected_candidate_count']} \\\\",
        f"Rule coverage & {100.0 * summary['rule_coverage']:.1f}\\% \\\\",
        f"Rejection explanation coverage & {100.0 * summary['explanation_coverage']:.1f}\\% \\\\",
        f"Environmental-factor provenance coverage & {100.0 * summary['factor_provenance_coverage']:.1f}\\% \\\\",
        f"Robust-candidate conformance & {100.0 * summary['robust_candidate_conformance_rate']:.1f}\\% \\\\",
    ]
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Knowledge-graph and SHACL validation audit.}",
                "\\label{tab:kg_validation_audit}",
                "\\small",
                "\\begin{tabular}{lr}",
                "\\toprule",
                "Audit item & Result \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        ),
    )


def select_representative_candidates(
    candidates: pd.DataFrame, seed: int
) -> pd.DataFrame:
    subset = candidates[
        (candidates["seed"] == seed)
        & (candidates["scenario"] == "target40_age28")
        & (candidates["stage"] == "knowledge_robust")
        & candidates["is_feasible"]
    ].copy()
    if subset.empty:
        raise ValueError(
            "No feasible 28-day robust candidates are available for selection"
        )
    objective_columns = [
        "gwp_risk_kgco2e_m3",
        "energy_risk_mj_m3",
        "total_material",
        "strength_prediction_mpa",
    ]
    values = subset[objective_columns].to_numpy(dtype=float)
    values[:, 3] *= -1.0
    normalized = (values - values.min(axis=0)) / np.maximum(
        values.max(axis=0) - values.min(axis=0), 1e-12
    )
    selections = [
        ("Minimum GWP", int(np.argmin(values[:, 0]))),
        ("Minimum energy", int(np.argmin(values[:, 1]))),
        ("Minimum material", int(np.argmin(values[:, 2]))),
        ("Maximum strength", int(np.argmin(values[:, 3]))),
        ("Balanced", int(np.argmin(normalized.mean(axis=1)))),
    ]
    labels = {}
    for label, position in selections:
        candidate_id = subset.iloc[position]["candidate_id"]
        labels.setdefault(candidate_id, []).append(label)
    selected = subset[subset["candidate_id"].isin(labels)].copy()
    selected.insert(
        0,
        "selection",
        [" / ".join(labels[value]) for value in selected["candidate_id"]],
    )
    return selected.sort_values("selection").reset_index(drop=True)


def write_representative_table(selected: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            f"{row.selection} & {row.cement:.1f} & {row.blast_furnace_slag:.1f} & "
            f"{row.fly_ash:.1f} & {row.water:.1f} & {row.superplasticizer:.1f} & "
            f"{row.coarse_aggregate:.1f} & {row.fine_aggregate:.1f} & "
            f"{row.strength_lower_bound_mpa:.1f} & {row.gwp_p95_kgco2e_m3:.1f} & "
            f"{row.energy_p95_mj_m3:.0f} & {row.total_material:.1f} \\\\"
        )
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Representative feasible 28-day robust Pareto candidates for seed 17. Material quantities and total material use are in kg/m$^3$.}",
                "\\label{tab:robust_representative_candidates}",
                "\\small",
                "\\resizebox{\\linewidth}{!}{%",
                "\\begin{tabular}{lrrrrrrrrrrr}",
                "\\toprule",
                "Selection & C & S & F & W & SP & CA & FA & $f_{c,L}$ & GWP$_{95}$ & Energy$_{95}$ & Material \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "\\end{table}",
            ]
        ),
    )


def write_lca_factor_table(lca_config: Mapping[str, object], path: Path) -> None:
    labels = {
        "cement": "Cement",
        "blast_furnace_slag": "Slag",
        "fly_ash": "Fly ash",
        "water": "Water",
        "superplasticizer": "Superplasticizer",
        "coarse_aggregate": "Coarse aggregate",
        "fine_aggregate": "Fine aggregate",
    }
    rows = []
    for material, record in lca_config["materials"].items():
        gwp = record["gwp_kgco2e_per_kg"]
        energy = record["energy_mj_per_kg"]
        rows.append(
            f"{labels[material]} & {gwp['baseline']:.4g} & "
            f"[{gwp['low']:.4g}, {gwp['mode']:.4g}, {gwp['high']:.4g}] & "
            f"{energy['baseline']:.4g} & "
            f"[{energy['low']:.4g}, {energy['mode']:.4g}, {energy['high']:.4g}] \\\\"
        )
    _write(
        path,
        "\n".join(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Dataset-embedded baseline factors and triangular screening distributions used for Monte Carlo propagation.}",
                "\\label{tab:lca_factor_uncertainty}",
                "\\small",
                "\\resizebox{0.96\\linewidth}{!}{%",
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "Material & GWP baseline & GWP [low, mode, high] (kg CO$_2$e/kg) & Energy baseline & Energy [low, mode, high] (MJ/kg) \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "\\end{table}",
            ]
        ),
    )


def generate_all_reports(
    twin_summary: pd.DataFrame,
    twin_decision_summary: pd.DataFrame,
    optimization_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    engineering: Mapping[str, object],
    lca_config: Mapping[str, object],
    ad_summary: Mapping[str, float],
    kg_summary: Mapping[str, object],
    output_dir: Path,
    representative_seed: int,
) -> None:
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plot_twin_ablation(twin_summary, figures / "twin_ablation.png")
    plot_decision_ablation(optimization_summary, figures / "decision_ablation.png")
    plot_robust_pareto(
        candidates, figures / "robust_pareto_28day.png", representative_seed
    )
    write_twin_table(twin_summary, tables / "manuscript_twin_table.tex")
    write_twin_decision_impact_table(
        twin_decision_summary,
        tables / "manuscript_twin_decision_impact_table.tex",
    )
    write_decision_table(optimization_summary, tables / "manuscript_decision_table.tex")
    write_constraint_table(
        engineering, ad_summary, tables / "manuscript_constraints_table.tex"
    )
    write_kg_table(kg_summary, tables / "manuscript_kg_table.tex")
    selected = select_representative_candidates(candidates, representative_seed)
    selected.to_csv(output_dir / "representative_robust_candidates.csv", index=False)
    write_representative_table(
        selected, tables / "manuscript_representative_candidates.tex"
    )
    write_lca_factor_table(lca_config, tables / "manuscript_lca_factors_table.tex")
