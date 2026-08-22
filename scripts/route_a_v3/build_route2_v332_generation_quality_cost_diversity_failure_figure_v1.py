#!/usr/bin/env python3
"""Build a provisional Generation quality-cost/diversity/failure figure.

The figure uses the frozen seven-method Development aggregate table and action-
space audit.  It does not read candidate payloads, Development TEST, new final
Evaluation outcomes, or any live training state.  Publisher-specific export
requirements remain pending until a venue, article type and phase are selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv"
)
DEFAULT_GEOMETRY_AUDIT = (
    ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)
FIGURE_ID = "route2_v332_generation_quality_cost_diversity_failure_figure_v1"
MANIFEST_NAME = f"{FIGURE_ID}_manifest.json"
ALT_TEXT_NAME = f"{FIGURE_ID}_alt_text.md"
FORMATS = ("png", "pdf", "svg")

METHOD_ORDER = (
    "genetic",
    "generate_then_rerank",
    "unguided_learned_base_flow_g0",
    "random_legal",
    "local_search",
    "greedy",
    "beam",
)
METHOD_LABELS = {
    "genetic": "Genetic",
    "generate_then_rerank": "Generate + rerank",
    "unguided_learned_base_flow_g0": "Unguided Base Flow",
    "random_legal": "Random legal",
    "local_search": "Local search",
    "greedy": "Greedy",
    "beam": "Beam",
}

BLUE = "#0072B2"
VERMILLION = "#D55E00"
AMBER = "#E6AB5F"
DARK_GRAY = "#4D4D4D"
LIGHT_GRAY = "#D8D8D8"
BLACK = "#111111"

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
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.7,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "grid.color": LIGHT_GRAY,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


class GenerationFigureInputError(RuntimeError):
    """Frozen inputs do not support the declared Generation figure."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationFigureInputError(message)


def _float(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    _require(math.isfinite(value), f"{row['method_id']} {field} is not finite")
    return value


def _int(row: Mapping[str, str], field: str) -> int:
    return int(row[field])


def _validate_inputs(
    rows: Sequence[Mapping[str, str]], audit: Mapping[str, Any]
) -> None:
    _require(len(rows) == 7, "generation table must contain seven methods")
    _require(
        {row["method_id"] for row in rows} == set(METHOD_ORDER),
        "generation table method set changed",
    )
    for row in rows:
        method_id = row["method_id"]
        _require(_int(row, "source_count") == 891, f"{method_id} source count changed")
        _require(_float(row, "hard_legality_rate") == 1.0,
                 f"{method_id} hard legality changed")
        _require(_int(row, "edit_budget_violation_count") == 0,
                 f"{method_id} edit-budget violation appeared")
        _require(_int(row, "candidate_budget_violation_count") == 0,
                 f"{method_id} candidate-cap violation appeared")
        _require(_int(row, "no_legal_action_count") == 0,
                 f"{method_id} no-legal-action failure appeared")
        _require(_int(row, "numerical_failure_count") == 0,
                 f"{method_id} numerical failure appeared")
        _require(_float(row, "mean_total_forward_equivalents_per_source") > 0.0,
                 f"{method_id} forward-equivalent cost is not positive")
        _require(0.0 <= _float(row, "source_macro_unique_candidate_rate") <= 1.0,
                 f"{method_id} unique rate is outside [0,1]")
        _require(_float(row, "source_macro_pairwise_hamming_diversity") >= 0.0,
                 f"{method_id} Hamming diversity is negative")
        _require(_int(row, "closed_measured_ndcg_defined_source_count") == 0,
                 f"{method_id} closed measured NDCG boundary changed")
    boundary = audit["protocol_boundary"]
    _require(boundary["candidate_support_mode"] == "OPEN_GENERATED_SUPPORT",
             "candidate support is no longer open generated support")
    _require(boundary["generated_candidates_grant_canonical_credit"] is False,
             "generated candidates unexpectedly grant canonical credit")
    _require(boundary["unknown_generated_candidates_are_zero_gain"] is False,
             "unknown generated candidates were relabeled as zero gain")
    _require(boundary["guided_xeditflow_run"] is False,
             "guided XEditFlow execution boundary changed")
    _require(boundary["development_test_outcomes_read"] == 0,
             "Development TEST was opened")
    _require(boundary["new_final_evaluation_outcomes_read"] == 0,
             "new final Evaluation was opened")
    cross = audit["cross_method_geometry"]
    _require(cross["total_no_legal_action_terminals"] == 0,
             "terminal no-legal-action total changed")
    _require(cross["total_numerical_failure_terminals"] == 0,
             "terminal numerical-failure total changed")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _style_for(method_id: str) -> tuple[str, str, int]:
    if method_id == "genetic":
        return VERMILLION, "s", 42
    if method_id == "unguided_learned_base_flow_g0":
        return BLUE, "D", 42
    return DARK_GRAY, "o", 28


def _annotate_points(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, str]],
    x_field: str,
    y_field: str,
    offsets: Mapping[str, tuple[int, int]],
) -> None:
    for row in rows:
        method_id = row["method_id"]
        x = _float(row, x_field)
        y = _float(row, y_field)
        color, marker, size = _style_for(method_id)
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=size,
            edgecolor=BLACK,
            linewidth=0.45,
            zorder=3,
        )
        dx, dy = offsets[method_id]
        ax.annotate(
            METHOD_LABELS[method_id],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=6.4,
            color=BLACK,
        )


def _render_figure(
    rows: Sequence[Mapping[str, str]], audit: Mapping[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_method = {row["method_id"]: row for row in rows}
    ordered = [by_method[method_id] for method_id in METHOD_ORDER]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 7.0),
        layout="constrained",
    )
    fig.suptitle(
        "Route 2 Development generation quality, cost and failure geometry\n"
        "Open generated support; computational evidence, not biological validation",
        fontsize=11,
        fontweight="bold",
    )

    offsets_a = {
        "genetic": (5, 3),
        "generate_then_rerank": (-5, 4),
        "unguided_learned_base_flow_g0": (5, -4),
        "random_legal": (-5, 4),
        "local_search": (-5, -5),
        "greedy": (-5, 4),
        "beam": (-5, -5),
    }
    ax = axes[0, 0]
    _annotate_points(
        ax,
        ordered,
        "mean_total_forward_equivalents_per_source",
        "source_macro_independent_evaluator_max_uplift_over_source",
        offsets_a,
    )
    ax.set(
        xlabel="Mean forward-equivalents per source",
        ylabel="Independent-evaluator max uplift",
        xlim=(0.0, 290.0),
        ylim=(0.70, 1.14),
        title="Independent-evaluator quality–cost",
    )
    ax.grid(True)
    _panel_label(ax, "A")

    offsets_b = {
        "genetic": (5, 4),
        "generate_then_rerank": (-5, -16),
        "unguided_learned_base_flow_g0": (5, 4),
        "random_legal": (5, 4),
        "local_search": (-5, 4),
        "greedy": (-5, 7),
        "beam": (-5, 9),
    }
    ax = axes[0, 1]
    _annotate_points(
        ax,
        ordered,
        "mean_total_forward_equivalents_per_source",
        "source_macro_candidate_recovery_rate",
        offsets_b,
    )
    ax.set(
        xlabel="Mean forward-equivalents per source",
        ylabel="Candidate recovery rate",
        xlim=(0.0, 290.0),
        ylim=(0.0, 0.22),
        title="Sparse measured recovery–cost",
    )
    ax.grid(True)
    _panel_label(ax, "B")

    ax = axes[1, 0]
    y = list(range(len(ordered)))
    diversity = [
        _float(row, "source_macro_pairwise_hamming_diversity") for row in ordered
    ]
    unique = [_float(row, "source_macro_unique_candidate_rate") for row in ordered]
    for index, (row, value, unique_rate) in enumerate(zip(ordered, diversity, unique)):
        color, marker, size = _style_for(row["method_id"])
        ax.hlines(index, 0.0, value, color=LIGHT_GRAY, linewidth=1.2, zorder=1)
        ax.scatter(
            value,
            index,
            color=color,
            marker=marker,
            s=size,
            edgecolor=BLACK,
            linewidth=0.45,
            zorder=2,
        )
        ax.text(
            0.003,
            index,
            f"u={unique_rate:.3f}",
            ha="left",
            va="center",
            fontsize=6.3,
            color=DARK_GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    ax.set(
        yticks=y,
        yticklabels=[METHOD_LABELS[row["method_id"]] for row in ordered],
        xlabel="Source-macro pairwise Hamming diversity",
        xlim=(0.0, 0.086),
        title="Candidate diversity and uniqueness",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    _panel_label(ax, "C")

    ax = axes[1, 1]
    candidate_cap_total = 891 * 32
    duplicate_rate = [
        _int(row, "duplicate_candidate_count") / _int(row, "candidate_count")
        for row in ordered
    ]
    cap_shortfall_rate = [
        (candidate_cap_total - _int(row, "candidate_count")) / candidate_cap_total
        for row in ordered
    ]
    height = 0.36
    ax.barh(
        [value - height / 2 for value in y],
        duplicate_rate,
        height=height,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=0.5,
        hatch="///",
        label="Within-source duplicate fraction",
    )
    ax.barh(
        [value + height / 2 for value in y],
        cap_shortfall_rate,
        height=height,
        color=AMBER,
        edgecolor=BLACK,
        linewidth=0.5,
        hatch="\\",
        label="Candidate-cap shortfall fraction",
    )
    ax.set(
        yticks=y,
        yticklabels=[METHOD_LABELS[row["method_id"]] for row in ordered],
        xlabel="Candidate-row fraction",
        xlim=(0.0, 0.29),
        title="Collapse/coverage geometry\nOther recorded failures = 0",
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=False,
        handletextpad=0.5,
    )
    _panel_label(ax, "D")

    provenance = {
        "figure_id": FIGURE_ID,
        "title": "Route 2 Development generation quality, cost and failure geometry",
        "status": "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURE_RENDERED",
        "width_inches": 7.2,
        "height_inches": 7.0,
        "analysis_unit": "SOURCE for source-macro quality/diversity/cost; candidate rows for duplicate and cap-shortfall fractions",
        "transformations": [
            "No row filtering; all seven frozen terminal methods are shown.",
            "Quality-cost panels plot frozen source-macro point estimates against mean total forward-equivalents per source.",
            "Duplicate fraction equals duplicate_candidate_count divided by candidate_count.",
            "Candidate-cap shortfall fraction equals (891*32-candidate_count)/(891*32).",
            "Methods use direct labels; genetic and unguided Base Flow also have distinct marker shapes and colors.",
        ],
        "uncertainty": "No per-method uncertainty interval is available for the plotted quality-cost or diversity aggregates; all panels show point estimates.",
        "missing_data": [
            "Closed measured NDCG is undefined for all seven open-support methods and is not plotted.",
            "Generation wall time is absent for six search methods; forward-equivalents, not wall time, define the plotted cost axis.",
            "Guided methods were closed by Critic V2 NO-GO and are not presented as executed points.",
        ],
        "zero_and_axis_policy": [
            "The cost and measured-recovery axes include zero.",
            "The independent-evaluator point-position panel uses y limits 0.70-1.14 to show the observed positive range; it is not a bar/area encoding.",
            "The diversity lollipop axis starts at zero; duplicate and shortfall bars start at zero.",
        ],
        "failure_boundary": {
            "hard_legality_rate_all_methods": audit["cross_method_geometry"][
                "all_method_hard_legality_rate"
            ],
            "total_edit_budget_violations": audit["cross_method_geometry"][
                "total_edit_budget_violations"
            ],
            "total_candidate_budget_violations": audit["cross_method_geometry"][
                "total_candidate_budget_violations"
            ],
            "total_no_legal_action_terminals": audit["cross_method_geometry"][
                "total_no_legal_action_terminals"
            ],
            "total_numerical_failure_terminals": audit["cross_method_geometry"][
                "total_numerical_failure_terminals"
            ],
        },
        "claim_boundary": "Development computational prioritization, diversity and efficiency only; no biological, guided-generation or external-generalization improvement is established.",
        "target_journal": "PENDING_SELECTION",
        "article_type": "PENDING_SELECTION",
        "submission_phase": "PENDING_SELECTION",
        "publisher_compliance_claimed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "palette": {
            "blue": BLUE,
            "vermillion": VERMILLION,
            "amber": AMBER,
            "dark_gray": DARK_GRAY,
            "background": "#FFFFFF",
            "color_is_redundant_with": [
                "marker shape",
                "hatching",
                "direct labels",
                "panel separation",
            ],
            "screening_note": "Amber is bounded and hatched in black; all method points are directly labeled, so color is never the only encoding.",
        },
    }
    return fig, provenance


def _export_figure(
    fig: plt.Figure,
    output_directory: Path,
    formats: Sequence[str],
    *,
    dpi: int,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for file_format in formats:
        path = output_directory / f"{FIGURE_ID}.{file_format}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing figure artifact: {path}")
        save_kwargs: dict[str, Any] = {
            "format": file_format,
            "facecolor": "white",
            "transparent": False,
        }
        if file_format == "png":
            save_kwargs["dpi"] = dpi
            save_kwargs["metadata"] = {"Software": "Matplotlib"}
        fig.savefig(path, **save_kwargs)
        outputs[file_format] = {"path": str(path), "bytes": path.stat().st_size}
    return outputs


def build_figure(
    *,
    generation_table: Path = DEFAULT_GENERATION_TABLE,
    geometry_audit: Path = DEFAULT_GEOMETRY_AUDIT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
) -> dict[str, Any]:
    generation_table = generation_table.resolve()
    geometry_audit = geometry_audit.resolve()
    output_directory = output_directory.resolve()
    rows = _read_csv(generation_table)
    audit = _read_json(geometry_audit)
    _validate_inputs(rows, audit)
    _require(dpi >= 150, "raster DPI must be at least 150")
    _require(formats and set(formats) <= set(FORMATS), "unsupported figure format")

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / MANIFEST_NAME
    alt_text_path = output_directory / ALT_TEXT_NAME
    for path in (manifest_path, alt_text_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing figure artifact: {path}")

    with matplotlib.rc_context(STYLE):
        fig, provenance = _render_figure(rows, audit)
        try:
            outputs = _export_figure(fig, output_directory, formats, dpi=dpi)
        finally:
            plt.close(fig)

    alt_text = """# Generation quality-cost, diversity and failure figure

Four-panel Development comparison of seven frozen generation/search methods.
Panel A plots independent-evaluator maximum uplift against mean forward-
equivalents; genetic search has the highest evaluator uplift, while random legal
search has the lowest cost. Panel B plots sparse measured candidate recovery
against the same cost axis; unguided Base Flow has the highest recovery at lower
cost than genetic and the three 266.6-forward methods. Panel C shows source-
macro pairwise Hamming diversity and annotates unique-candidate rate; Base Flow
has the highest Hamming diversity but a lower unique rate because of duplicates.
Panel D shows duplicate and candidate-cap-shortfall fractions: duplicates occur
only for Base Flow, while cap shortfall occurs only for local search. All methods
have hard legality 1.0 and zero edit-budget, candidate-budget, no-legal-action or
numerical failures. These are Development computational results under open
generated support, not measured biological validation.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")
    manifest = {
        "schema_version": "route2_v332_generation_quality_cost_diversity_failure_figure.v1",
        **provenance,
        "outputs": outputs,
        "raster_dpi": dpi,
        "formats": list(formats),
        "alt_text_path": str(alt_text_path),
        "source_data": {
            "generation_table": str(generation_table),
            "geometry_audit": str(geometry_audit),
        },
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "generated_candidate_payload_read": False,
            "guided_xeditflow_run": False,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-table", type=Path, default=DEFAULT_GENERATION_TABLE)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    manifest = build_figure(
        generation_table=args.generation_table,
        geometry_audit=args.geometry_audit,
        output_directory=args.output_directory,
        formats=formats,
        dpi=args.dpi,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
