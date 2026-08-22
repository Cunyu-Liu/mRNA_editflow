#!/usr/bin/env python3
"""Build provisional Route 2 V3.3.2 manuscript figures from frozen evidence.

The builder deliberately excludes Development TEST and new final Evaluation
outcomes.  It renders Development generation evidence, terminal Critic V2
diagnostics, and the explicitly outcome-exposed GSE232572 historical transfer
summary.  Publisher-specific dimensions remain pending until a venue and
submission phase are selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv"
)
DEFAULT_CRITIC_TABLE = (
    ROOT / "docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv"
)
DEFAULT_HISTORICAL_SUMMARY = (
    ROOT / "audits/route_a_v3_route2_gse232572_zero_shot_summary_v1.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)

REQUIRED_METHOD_IDS = (
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
    "unguided_learned_base_flow_g0",
)
FIGURE_FORMATS = ("png", "pdf", "svg")

BLUE = "#0072B2"
VERMILLION = "#D55E00"
AMBER = "#E6AB5F"
BLACK = "#000000"
DARK_GRAY = "#4D4D4D"
LIGHT_GRAY = "#D9D9D9"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "lines.linewidth": 1.2,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

METHOD_LABELS = {
    "genetic": "Genetic",
    "generate_then_rerank": "Generate + rerank",
    "unguided_learned_base_flow_g0": "Unguided Base Flow",
    "random_legal": "Random legal",
    "local_search": "Local search",
    "greedy": "Greedy",
    "beam": "Beam",
}

TASK_LABELS = {
    "MEAN_RIBOSOME_LOAD::region=0": "Mean ribosome load, r0",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1": "MPRA allelic skew, r1",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1": "Proximal poly(A), r1",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1": "Published ref-vs-alt, r1",
    "RNA_HALF_LIFE_MINUTES::region=0": "RNA half-life, r0",
    "RNA_HALF_LIFE_MINUTES::region=1": "RNA half-life, r1",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1": "Polysome TE, r1",
    "te_log2_polysome_over_totalrna::region=0": "TE polysome/total, r0",
    "transcript_log2_totalrna_over_dna::region=0": "Transcript total/DNA, r0",
}


class FigureInputError(RuntimeError):
    """Frozen figure input does not match the declared Route 2 evidence shape."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureInputError(message)


def _validate_inputs(
    generation_rows: Sequence[Mapping[str, str]],
    critic_rows: Sequence[Mapping[str, str]],
    historical: Mapping[str, Any],
) -> None:
    _require(
        {row["method_id"] for row in generation_rows} == set(REQUIRED_METHOD_IDS),
        "generation table does not contain the exact frozen seven-method set",
    )
    _require(len(critic_rows) == 9, "critic diagnostic table must contain nine tasks")
    _require(
        {row["task_id"] for row in critic_rows} == set(TASK_LABELS),
        "critic diagnostic task set changed",
    )
    _require(
        historical.get("study_unit_id") == "GSE232572",
        "historical summary is not GSE232572",
    )
    _require(
        historical.get("preregistered_pass") is False,
        "historical summary no longer records the frozen negative adjudication",
    )
    paired = historical.get("paired_results", [])
    _require(
        [int(row["seed"]) for row in paired] == [20260816, 20260817, 20260818],
        "historical summary seed set or order changed",
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
        ha="left",
    )


def _method_style(method_id: str) -> tuple[str, str, int]:
    if method_id == "genetic":
        return VERMILLION, "D", 36
    if method_id == "unguided_learned_base_flow_g0":
        return BLUE, "s", 36
    return DARK_GRAY, "o", 24


def _render_generation_figure(
    generation_rows: Sequence[Mapping[str, str]],
) -> tuple[plt.Figure, dict[str, Any]]:
    ordered = sorted(
        generation_rows,
        key=lambda row: float(
            row["source_macro_independent_evaluator_max_uplift_over_source"]
        ),
        reverse=True,
    )
    labels = [METHOD_LABELS[row["method_id"]] for row in ordered]
    y = list(range(len(ordered)))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.4),
        layout="constrained",
    )
    fig.suptitle(
        "Route 2 Development generation benchmark\n"
        "Open generated support; computational evidence, not biological validation",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0, 0]
    uplift = [
        float(row["source_macro_independent_evaluator_max_uplift_over_source"])
        for row in ordered
    ]
    for index, (row, value) in enumerate(zip(ordered, uplift)):
        color, marker, size = _method_style(row["method_id"])
        ax.hlines(index, 0.0, value, color=LIGHT_GRAY, linewidth=1.2, zorder=1)
        ax.scatter(value, index, color=color, marker=marker, s=size, zorder=2)
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Max independent-evaluator uplift over source",
        xlim=(0.0, 1.16),
        title="Frozen Development evaluator",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "A")

    ax = axes[0, 1]
    candidate_recovery = [
        float(row["source_macro_candidate_recovery_rate"]) for row in ordered
    ]
    topk_recovery = [
        float(row["source_macro_measured_top_k_recovery_at_k"]) for row in ordered
    ]
    ax.scatter(
        candidate_recovery,
        y,
        color=BLUE,
        marker="o",
        s=28,
        label="Candidate recovery",
        zorder=2,
    )
    ax.scatter(
        topk_recovery,
        y,
        facecolors="white",
        edgecolors=VERMILLION,
        marker="D",
        s=30,
        linewidths=1.1,
        label="Measured top-k recovery",
        zorder=2,
    )
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Source-macro recovery rate",
        xlim=(0.0, 0.22),
        title="Sparse measured neighborhood",
    )
    ax.set_title("Sparse measured neighborhood", pad=27)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    _panel_label(ax, "B")

    ax = axes[1, 0]
    stop = [float(row["explicit_stop_rate"]) for row in ordered]
    exhausted = [float(row["budget_exhausted_rate"]) for row in ordered]
    ax.barh(
        y,
        stop,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=0.5,
        hatch="///",
        label="Explicit STOP",
    )
    ax.barh(
        y,
        exhausted,
        left=stop,
        color=AMBER,
        edgecolor=BLACK,
        linewidth=0.5,
        hatch="\\\\",
        label="Budget exhausted",
    )
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Fraction of terminal candidates",
        xlim=(0.0, 1.0),
        title="Terminal mechanism (other causes = 0)",
    )
    ax.set_title("Terminal mechanism (other causes = 0)", pad=27)
    ax.invert_yaxis()
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    _panel_label(ax, "C")

    ax = axes[1, 1]
    means = [float(row["source_candidate_count_mean"]) for row in ordered]
    minimums = [float(row["source_candidate_count_min"]) for row in ordered]
    maximums = [float(row["source_candidate_count_max"]) for row in ordered]
    for index, (row, mean, minimum, maximum) in enumerate(
        zip(ordered, means, minimums, maximums)
    ):
        color, marker, size = _method_style(row["method_id"])
        ax.hlines(index, minimum, maximum, color=DARK_GRAY, linewidth=1.2)
        ax.scatter(mean, index, color=color, marker=marker, s=size, zorder=2)
        duplicates = int(row["duplicate_candidate_count"])
        if duplicates:
            ax.annotate(
                f"{duplicates:,} duplicates",
                (mean, index),
                xytext=(-5, 7),
                textcoords="offset points",
                ha="right",
                fontsize=6.5,
                color=DARK_GRAY,
            )
    ax.axvline(32.0, color=BLACK, linestyle="--", linewidth=0.9)
    ax.text(
        31.7,
        -0.35,
        "Cap = 32",
        ha="right",
        va="bottom",
        fontsize=7,
        color=BLACK,
    )
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Candidates per source (min-mean-max)",
        xlim=(0.0, 34.0),
        title="Candidate realization and duplication",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "D")

    provenance = {
        "figure_id": "route2_v332_figure1_generation_benchmark_v1",
        "title": "Route 2 Development generation benchmark",
        "width_inches": 7.2,
        "height_inches": 6.4,
        "analysis_unit": "SOURCE for source-macro metrics; candidate for terminal fractions",
        "transformations": [
            "Methods sorted by frozen independent-evaluator point estimate",
            "Terminal fractions copied from exact terminal candidate counts",
            "Candidate panel shows per-source minimum, mean and maximum",
        ],
        "uncertainty": "No per-method uncertainty is available in this figure; only point estimates are plotted.",
        "missing_data": [
            "Closed measured NDCG is undefined under open generated support and is not plotted.",
            "Search-method generation wall time and per-candidate algorithmic STOP time were not retained and are not imputed.",
        ],
        "claim_boundary": "Development computational evidence only; generated candidates do not grant canonical or biological credit.",
    }
    return fig, provenance


def _task_marker(value: float) -> tuple[str, str]:
    return (BLUE, "o") if value > 0.0 else (VERMILLION, "X")


def _render_limits_figure(
    critic_rows: Sequence[Mapping[str, str]],
    historical: Mapping[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    labels = [
        f"{TASK_LABELS[row['task_id']]} (n={int(row['record_count']):,})"
        for row in critic_rows
    ]
    y = list(range(len(critic_rows)))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.8),
        layout="constrained",
    )
    fig.suptitle(
        "Predictor and historical-transfer limits\n"
        "Development Validation and outcome-exposed historical diagnostic",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0, 0]
    spearman_margin = [
        float(row["full_minus_strongest_baseline_spearman"]) for row in critic_rows
    ]
    for index, value in enumerate(spearman_margin):
        color, marker = _task_marker(value)
        ax.scatter(value, index, color=color, marker=marker, s=28, zorder=2)
    ax.axvline(0.0, color=BLACK, linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Spearman margin (full - baseline)",
        title="Critic V2 rank margins",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "A")

    ax = axes[0, 1]
    mae_margin = [
        float(row["full_minus_strongest_baseline_standardized_mae"])
        for row in critic_rows
    ]
    ax.scatter(mae_margin, y, color=VERMILLION, marker="X", s=28, zorder=2)
    ax.axvline(0.0, color=BLACK, linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=[],
        xlabel="Standardized MAE margin (full - baseline)",
        xlim=(-0.04, 1.38),
        title="Critic V2 MAE margins\nPositive = full worse",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "B")

    paired = historical["paired_results"]
    seeds = [str(row["seed"]) for row in paired]
    seed_y = list(range(len(paired)))

    ax = axes[1, 0]
    rank = [float(row["task_macro_spearman_improvement"]) for row in paired]
    rank_low = [float(row["task_macro_spearman_improvement_ci_95"][0]) for row in paired]
    rank_high = [float(row["task_macro_spearman_improvement_ci_95"][1]) for row in paired]
    rank_error = [
        [value - lower for value, lower in zip(rank, rank_low)],
        [upper - value for value, upper in zip(rank, rank_high)],
    ]
    ax.errorbar(
        rank,
        seed_y,
        xerr=rank_error,
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
        markersize=4.5,
        linewidth=1.1,
    )
    ax.axvline(0.0, color=BLACK, linewidth=0.8)
    ax.set(
        yticks=seed_y,
        yticklabels=seeds,
        xlabel="Model - baseline task-macro Spearman",
        ylabel="Frozen seed",
        xlim=(-0.02, 0.09),
        title="Historical rank transfer (paired 95% CI)",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "C")

    ax = axes[1, 1]
    mae = [float(row["baseline_mae_minus_model_mae"]) for row in paired]
    mae_low = [float(row["baseline_mae_minus_model_mae_ci_95"][0]) for row in paired]
    mae_high = [float(row["baseline_mae_minus_model_mae_ci_95"][1]) for row in paired]
    mae_error = [
        [value - lower for value, lower in zip(mae, mae_low)],
        [upper - value for value, upper in zip(mae, mae_high)],
    ]
    ax.errorbar(
        mae,
        seed_y,
        xerr=mae_error,
        fmt="X",
        color=VERMILLION,
        ecolor=VERMILLION,
        capsize=3,
        markersize=5,
        linewidth=1.1,
    )
    ax.axvline(0.0, color=BLACK, linewidth=0.8)
    ax.set(
        yticks=seed_y,
        yticklabels=[],
        xlabel="MAE margin (baseline - model)",
        xlim=(-0.21, 0.01),
        title="Historical MAE transfer (paired 95% CI)",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "D")

    provenance = {
        "figure_id": "route2_v332_figure2_predictor_transfer_limits_v1",
        "title": "Predictor and historical-transfer limits",
        "width_inches": 7.2,
        "height_inches": 6.8,
        "analysis_unit": "Task for Critic V2 margins; paired source bootstrap within each historical frozen seed",
        "transformations": [
            "Critic V2 margins copied without filtering from the nine-task diagnostic table",
            "Historical interval errors reconstructed only as point-to-recorded-bound distances",
        ],
        "uncertainty": "GSE232572 panels show recorded paired 95% confidence intervals; Critic V2 task panels have no intervals and show point margins only.",
        "missing_data": "No missing values were imputed. The two n=48 tasks remain included and labeled.",
        "claim_boundary": "GSE232572 is outcome-exposed historical transfer evidence, not final independent confirmation.",
    }
    return fig, provenance


def _export_figure(
    fig: plt.Figure,
    output_directory: Path,
    stem: str,
    formats: Iterable[str],
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for format_name in formats:
        _require(format_name in FIGURE_FORMATS, f"unsupported figure format: {format_name}")
        path = output_directory / f"{stem}.{format_name}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing figure: {path}")
        if format_name == "pdf":
            metadata = {
                "Title": stem,
                "Author": "mRNA-EditFlow Route 2 evidence builder",
                "Subject": "Provisional general-manuscript scientific figure",
            }
        elif format_name == "svg":
            metadata = {
                "Title": stem,
                "Creator": "mRNA-EditFlow Route 2 evidence builder",
                "Description": "Provisional general-manuscript scientific figure",
            }
        else:
            metadata = {
                "Title": stem,
                "Author": "mRNA-EditFlow Route 2 evidence builder",
                "Description": "Provisional general-manuscript scientific figure",
            }
        save_options: dict[str, Any] = {
            "format": format_name,
            "facecolor": "white",
            "transparent": False,
            "metadata": metadata,
        }
        if format_name == "png":
            save_options["dpi"] = dpi
        fig.savefig(path, **save_options)
        outputs[format_name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "dpi": dpi if format_name == "png" else None,
        }
    return outputs


def build_figures(
    *,
    generation_table: Path = DEFAULT_GENERATION_TABLE,
    critic_table: Path = DEFAULT_CRITIC_TABLE,
    historical_summary: Path = DEFAULT_HISTORICAL_SUMMARY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FIGURE_FORMATS,
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    generation_table = generation_table.resolve()
    critic_table = critic_table.resolve()
    historical_summary = historical_summary.resolve()
    output_directory = output_directory.resolve()

    generation_rows = _read_csv(generation_table)
    critic_rows = _read_csv(critic_table)
    historical = _read_json(historical_summary)
    _validate_inputs(generation_rows, critic_rows, historical)
    _require(dpi >= 150, "raster DPI must be at least 150 for manuscript review")

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "route2_v332_figure_manifest_v1.json"
    alt_text_path = output_directory / "route2_v332_figure_alt_text_v1.md"
    for path in (manifest_path, alt_text_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing figure artifact: {path}")

    with matplotlib.rc_context(STYLE):
        generation_figure, generation_provenance = _render_generation_figure(
            generation_rows
        )
        limits_figure, limits_provenance = _render_limits_figure(
            critic_rows, historical
        )
        try:
            generation_outputs = _export_figure(
                generation_figure,
                output_directory,
                generation_provenance["figure_id"],
                formats,
                dpi=dpi,
                overwrite=overwrite,
            )
            limits_outputs = _export_figure(
                limits_figure,
                output_directory,
                limits_provenance["figure_id"],
                formats,
                dpi=dpi,
                overwrite=overwrite,
            )
        finally:
            plt.close(generation_figure)
            plt.close(limits_figure)

    alt_text = """# Route 2 V3.3.2 provisional figure descriptions

## Figure 1 — Development generation benchmark

Four-panel comparison of seven matched-budget generation/search methods across
891 Development sources. Genetic search has the largest frozen independent-
evaluator uplift, whereas unguided Base Flow has the largest sparse measured-
neighborhood recovery. Terminal STOP versus budget-exhaustion fractions vary
substantially by method. Local search returns fewer than the 32-candidate cap for
some sources, while Base Flow contains 3,339 duplicate candidate rows. All
candidates are legal and within budget. These are computational Development
results under open generated support, not measured biological validation.

## Figure 2 — Predictor and historical-transfer limits

Four-panel diagnostic. Critic V2 beats the strongest same-information baseline
on four of nine task Spearman margins but has worse standardized MAE on all nine
tasks; the two n=48 tasks remain visible. In the outcome-exposed GSE232572
historical diagnostic, all three rank point estimates favor the model, but one
paired 95% interval crosses zero, and all three MAE intervals favor the baseline.
The historical study is not an independent final confirmation.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")

    manifest = {
        "schema_version": "route_a_v3_route2_v332_figure_manifest.v1",
        "status": "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED",
        "target_journal": "PENDING_SELECTION",
        "article_type": "PENDING_SELECTION",
        "submission_phase": "INTERNAL_EVIDENCE_REVIEW",
        "publisher_compliance_claimed": False,
        "matplotlib_version": matplotlib.__version__,
        "python_version": platform.python_version(),
        "raster_dpi": dpi,
        "background": "OPAQUE_WHITE",
        "palette": {
            "blue": BLUE,
            "vermillion": VERMILLION,
            "amber": AMBER,
            "black": BLACK,
            "dark_gray": DARK_GRAY,
            "light_gray": LIGHT_GRAY,
            "color_is_redundant_with": [
                "marker shape",
                "hatching",
                "direct category labels",
                "panel separation",
            ],
            "screening_note": [
                "Amber fill and light-gray guides are below 3:1 contrast against white; amber is bounded and hatched in black, while light gray is non-semantic scaffolding.",
                "Blue and vermillion have limited grayscale separation; every comparison also uses distinct marker shape, fill state, hatching, or direct labels.",
            ],
        },
        "source_data": {
            "generation_table": str(generation_table),
            "critic_table": str(critic_table),
            "historical_summary": str(historical_summary),
        },
        "figures": [
            {**generation_provenance, "outputs": generation_outputs},
            {**limits_provenance, "outputs": limits_outputs},
        ],
        "alt_text_path": str(alt_text_path),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "historical_outcome_exposed_gse232572_read": True,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-table", type=Path, default=DEFAULT_GENERATION_TABLE)
    parser.add_argument("--critic-table", type=Path, default=DEFAULT_CRITIC_TABLE)
    parser.add_argument(
        "--historical-summary", type=Path, default=DEFAULT_HISTORICAL_SUMMARY
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--formats",
        default=",".join(FIGURE_FORMATS),
        help="Comma-separated subset of png,pdf,svg",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_figures(
        generation_table=args.generation_table,
        critic_table=args.critic_table,
        historical_summary=args.historical_summary,
        output_directory=args.output_directory,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
