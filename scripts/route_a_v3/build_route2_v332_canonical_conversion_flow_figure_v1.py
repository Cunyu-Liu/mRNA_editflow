#!/usr/bin/env python3
"""Render the provisional Route 2 V3.3.2 canonical conversion flow figure."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
DEFAULT_SPLIT_PROTOCOL = ROOT / "configs/route_a_v3_route2_method_repair_protocol_v2.json"
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)
STEM = "route2_v332_canonical_conversion_flow_figure_v1"
FORMATS = ("png", "pdf", "svg")

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#007A5A"
DARK_GRAY = "#4D4D4D"
BLACK = "#000000"
LIGHT_BLUE = "#E8F2F8"
LIGHT_ORANGE = "#FBEFE8"
LIGHT_GREEN = "#E8F4EF"
LIGHT_GRAY = "#F1F1F1"
WHITE = "#FFFFFF"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "savefig.facecolor": WHITE,
    "figure.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


class ConversionFlowInputError(RuntimeError):
    """The frozen table or split protocol changed unexpectedly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionFlowInputError(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _derive_evidence(rows: Sequence[Mapping[str, str]], split: Mapping[str, int]) -> dict[str, Any]:
    _require(len(rows) == 14, "dataset table must contain 14 study units")
    _require(len({row["study_unit_id"] for row in rows}) == 14, "study IDs must be unique")

    development = [row for row in rows if row["current_analysis_role_v332"] == "DEVELOPMENT"]
    historical = [
        row
        for row in rows
        if row["current_analysis_role_v332"]
        == "HISTORICAL_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
    ]
    other = [row for row in rows if row not in development and row not in historical]

    strata = {
        "qualified": [row for row in development if row["qualification_stratum"] == "QUALIFIED"],
        "relaxed": [
            row
            for row in development
            if row["qualification_stratum"] == "DEVELOPMENT_RELAXED_NOT_QUALIFIED"
        ],
        "listwise": [
            row
            for row in development
            if row["qualification_stratum"] == "DEVELOPMENT_LISTWISE_NOT_QUALIFIED"
        ],
        "unconvertible": [
            row for row in development if row["qualification_stratum"] == "UNCONVERTIBLE"
        ],
    }

    evidence = {
        "registered_study_count": len(rows),
        "development_study_count": len(development),
        "development_record_count": sum(int(row["development_canonical_records"]) for row in rows),
        "historical_study_count": len(historical),
        "historical_record_count": sum(
            int(row["historical_transfer_canonical_records"]) for row in rows
        ),
        "other_study_count": len(other),
        "other_record_count": sum(int(row["canonical_records"]) for row in other),
        "strata": {
            name: {
                "study_count": len(group),
                "record_count": sum(int(row["development_canonical_records"]) for row in group),
            }
            for name, group in strata.items()
        },
        "split": {key: int(value) for key, value in split.items()},
        "zero_record_study_count": sum(int(row["canonical_records"]) == 0 for row in rows),
        "new_final_evaluation_record_count": sum(
            int(row["final_evaluation_unexposed_canonical_records"]) for row in rows
        ),
    }
    _require(evidence["development_study_count"] == 8, "Development study count changed")
    _require(evidence["development_record_count"] == 126165, "Development record total changed")
    _require(evidence["historical_study_count"] == 1, "historical study count changed")
    _require(evidence["historical_record_count"] == 8068, "historical record total changed")
    _require(evidence["other_study_count"] == 5 and evidence["other_record_count"] == 0, "other-role geometry changed")
    _require(evidence["strata"] == {
        "qualified": {"study_count": 1, "record_count": 6547},
        "relaxed": {"study_count": 5, "record_count": 88652},
        "listwise": {"study_count": 1, "record_count": 30966},
        "unconvertible": {"study_count": 1, "record_count": 0},
    }, "Development qualification strata changed")
    _require(evidence["split"] == {"TRAIN": 89580, "VALIDATION": 18293, "TEST": 18292}, "frozen split counts changed")
    _require(sum(evidence["split"].values()) == 126165, "split records do not conserve Development total")
    _require(evidence["zero_record_study_count"] == 6, "zero-record study count changed")
    _require(evidence["new_final_evaluation_record_count"] == 0, "new final Evaluation records are no longer zero")
    return evidence


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.1,
    fontsize: float = 7.3,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=BLACK,
        linespacing=1.25,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK_GRAY,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.01, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="left", va="bottom")


def _render(evidence: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), layout="constrained", gridspec_kw={"height_ratios": [0.85, 1.15]})
    fig.suptitle(
        "Route 2 canonical study-to-evidence flow",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Study-unit disposition and permissible evidence use", pad=5)
    _panel_label(ax, "A")

    _box(ax, (0.02, 0.36), 0.20, 0.28, "Registered inventory\n14 study units\nterminal conversion facts", facecolor=WHITE, edgecolor=BLACK, fontsize=8)
    role_boxes = [
        ((0.34, 0.68), "Development\n8 study units | 126,165 records", LIGHT_BLUE, BLUE),
        ((0.34, 0.41), "Historical transfer/diagnostic\n1 study | 8,068 records", LIGHT_ORANGE, VERMILLION),
        ((0.34, 0.10), "Other explicit terminal roles\n5 studies | 0 canonical records", LIGHT_GRAY, DARK_GRAY),
    ]
    use_boxes = [
        ((0.72, 0.68), "TRAIN/VALIDATION benchmark\nTEST withheld", LIGHT_BLUE, BLUE),
        ((0.72, 0.41), "Outcome-exposed diagnostic\nnot final confirmation", LIGHT_ORANGE, VERMILLION),
        ((0.72, 0.10), "Conversion failure 1 | auxiliary 1\nnegative controls 2 | sealed 1", LIGHT_GRAY, DARK_GRAY),
    ]
    for (xy, text, fill, edge), (use_xy, use_text, use_fill, use_edge) in zip(role_boxes, use_boxes):
        _box(ax, xy, 0.28, 0.18, text, facecolor=fill, edgecolor=edge)
        _box(ax, use_xy, 0.26, 0.18, use_text, facecolor=use_fill, edgecolor=use_edge)
        _arrow(ax, (0.22, 0.50), (xy[0], xy[1] + 0.09), color=edge)
        _arrow(ax, (xy[0] + 0.28, xy[1] + 0.09), (use_xy[0], use_xy[1] + 0.09), color=edge)

    ax = axes[1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Development qualification strata and grouped record split", pad=5)
    _panel_label(ax, "B")

    _box(ax, (0.01, 0.39), 0.18, 0.22, "Development pool\n8 study units\n126,165 records", facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=8)
    strata_specs = [
        ((0.28, 0.75), "Qualified\n1 study | 6,547 records\ncredit = 1/1/0", LIGHT_GREEN, GREEN),
        ((0.28, 0.54), "Development-relaxed\n5 studies | 88,652 records\nno qualified credit", LIGHT_BLUE, BLUE),
        ((0.28, 0.33), "Development listwise\n1 study | 30,966 records\nno qualified credit", LIGHT_BLUE, BLUE),
        ((0.28, 0.12), "Unconvertible in V1\n1 study | 0 records", LIGHT_GRAY, DARK_GRAY),
    ]
    for xy, text, fill, edge in strata_specs:
        _box(ax, xy, 0.25, 0.14, text, facecolor=fill, edgecolor=edge, fontsize=7)
        _arrow(ax, (0.19, 0.50), (xy[0], xy[1] + 0.07), color=edge)

    _box(
        ax,
        (0.59, 0.42),
        0.16,
        0.20,
        "Grouped split\nsource/gene/family/\nnear-duplicate components\nseed 20260816",
        facecolor=WHITE,
        edgecolor=BLACK,
        fontsize=6.8,
    )
    for index, (_, _, _, edge) in enumerate(strata_specs[:3]):
        y = strata_specs[index][0][1] + 0.07
        _arrow(ax, (0.53, y), (0.59, 0.52), color=edge)

    split_specs = [
        ((0.82, 0.71), "TRAIN\n89,580", LIGHT_GREEN, GREEN),
        ((0.82, 0.44), "VALIDATION\n18,293", LIGHT_BLUE, BLUE),
        ((0.82, 0.17), "TEST withheld\n18,292", LIGHT_ORANGE, VERMILLION),
    ]
    for xy, text, fill, edge in split_specs:
        _box(ax, xy, 0.16, 0.15, text, facecolor=fill, edgecolor=edge, fontsize=7.5)
        _arrow(ax, (0.75, 0.52), (xy[0], xy[1] + 0.075), color=edge)

    ax.text(
        0.01,
        0.015,
        "Arrows encode workflow only; widths are not proportional. Generated candidates add zero canonical credit. New unexposed final Evaluation records = 0.",
        fontsize=6.5,
        color=DARK_GRAY,
        ha="left",
        va="bottom",
    )
    return fig


def _metadata(format_name: str) -> dict[str, str]:
    if format_name == "pdf":
        return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder", "Subject": "Provisional canonical conversion flow figure"}
    if format_name == "svg":
        return {"Title": STEM, "Creator": "mRNA-EditFlow Route 2 evidence builder", "Description": "Provisional canonical conversion flow figure"}
    return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder", "Description": "Provisional canonical conversion flow figure"}


def build_figure(
    *,
    dataset_table: Path = DEFAULT_DATASET_TABLE,
    split_protocol: Path = DEFAULT_SPLIT_PROTOCOL,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_table = dataset_table.resolve()
    split_protocol = split_protocol.resolve()
    output_directory = output_directory.resolve()
    _require(dpi >= 150, "raster DPI must be at least 150")
    _require(set(formats) <= set(FORMATS) and len(formats) > 0, "formats must be a nonempty subset of png,pdf,svg")

    rows = _read_rows(dataset_table)
    protocol = json.loads(split_protocol.read_text(encoding="utf-8"))
    split = protocol["development_independence_units"]["fixed_split_record_counts"]
    evidence = _derive_evidence(rows, split)

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / f"{STEM}_manifest.json"
    alt_text_path = output_directory / f"{STEM}_alt_text.md"
    targets = [output_directory / f"{STEM}.{format_name}" for format_name in formats] + [manifest_path, alt_text_path]
    for path in targets:
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing conversion-flow artifact: {path}")

    with matplotlib.rc_context(STYLE):
        fig = _render(evidence)
        outputs: dict[str, dict[str, Any]] = {}
        try:
            for format_name in formats:
                path = output_directory / f"{STEM}.{format_name}"
                options: dict[str, Any] = {
                    "format": format_name,
                    "facecolor": WHITE,
                    "transparent": False,
                    "metadata": _metadata(format_name),
                }
                if format_name == "png":
                    options["dpi"] = dpi
                fig.savefig(path, **options)
                outputs[format_name] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "dpi": dpi if format_name == "png" else None,
                }
        finally:
            plt.close(fig)

    alt_text = """# Route 2 V3.3.2 canonical conversion flow figure

Two-panel workflow diagram. Panel A routes 14 registered study units to eight
Development studies with 126,165 records, one outcome-exposed historical study
with 8,068 records, and five auxiliary, negative-control, conversion-failure or
sealed studies with zero canonical records. Panel B separates the Development
pool into one qualified study with 6,547 records, five Development-relaxed
studies with 88,652 records, one listwise study with 30,966 records and one
unconvertible study with zero records. Materialized Development records undergo
a grouped split into 89,580 TRAIN, 18,293 VALIDATION and 18,292 withheld TEST
records. Arrow widths are not quantitative. Generated candidates add no
canonical credit, and no new unexposed final Evaluation records are present.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")
    manifest = {
        "schema_version": "route_a_v3_route2_v332_canonical_conversion_flow_figure.v1",
        "status": "PROVISIONAL_CANONICAL_CONVERSION_FLOW_FIGURE_RENDERED",
        "target_journal": "PENDING_SELECTION",
        "article_type": "PENDING_SELECTION",
        "submission_phase": "INTERNAL_EVIDENCE_REVIEW",
        "publisher_compliance_claimed": False,
        "matplotlib_version": matplotlib.__version__,
        "python_version": platform.python_version(),
        "width_inches": 7.2,
        "height_inches": 6.2,
        "raster_dpi": dpi,
        "background": "OPAQUE_WHITE",
        "source_data": {
            "dataset_table": str(dataset_table),
            "split_protocol": str(split_protocol),
        },
        "transformations": [
            "Exact study and record counts grouped by declared V3.3.2 role and qualification stratum",
            "Workflow arrows deliberately rendered at constant width rather than encoding mixed study/record units",
        ],
        "evidence": evidence,
        "outputs": outputs,
        "alt_text_path": str(alt_text_path),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "emtab10902_outcome_read": False,
            "sealed_gse246381_read": False,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-table", type=Path, default=DEFAULT_DATASET_TABLE)
    parser.add_argument("--split-protocol", type=Path, default=DEFAULT_SPLIT_PROTOCOL)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_figure(
        dataset_table=args.dataset_table,
        split_protocol=args.split_protocol,
        output_directory=args.output_directory,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
