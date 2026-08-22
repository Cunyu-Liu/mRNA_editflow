#!/usr/bin/env python3
"""Render the provisional Route 2 V3.3.2 Development/Evaluation architecture figure."""

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
DEFAULT_METHOD_PROTOCOL = ROOT / "configs/route_a_v3_route2_method_repair_protocol_v2.json"
DEFAULT_READINESS_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json"
)
DEFAULT_PACKAGE_AUDIT = ROOT / "audits/route_a_v3_route2_v332_minimum_benchmark_package_v1.json"
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)
STEM = "route2_v332_development_evaluation_architecture_figure_v1"
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


class ArchitectureFigureInputError(RuntimeError):
    """A frozen Development/Evaluation boundary changed unexpectedly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchitectureFigureInputError(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_by_id(rows: Sequence[Mapping[str, str]], study_id: str) -> Mapping[str, str]:
    matches = [row for row in rows if row["study_unit_id"] == study_id]
    _require(len(matches) == 1, f"expected exactly one row for {study_id}")
    return matches[0]


def _derive_evidence(
    rows: Sequence[Mapping[str, str]],
    method_protocol: Mapping[str, Any],
    readiness_protocol: Mapping[str, Any],
    package_audit: Mapping[str, Any],
) -> dict[str, Any]:
    _require(len(rows) == 14, "dataset table must contain 14 study units")
    split = method_protocol["development_independence_units"]["fixed_split_record_counts"]
    split = {key: int(value) for key, value in split.items()}
    _require(
        split == {"TRAIN": 89580, "VALIDATION": 18293, "TEST": 18292},
        "frozen Development split changed",
    )
    _require(sum(split.values()) == 126165, "Development split does not conserve 126,165 records")
    _require(
        method_protocol["screen_selection"]["development_test_role"] == "NOT_ACCESSED",
        "method protocol no longer keeps Development TEST closed",
    )
    _require(method_protocol["legacy_best_observed_validation_reference"]["development_test_used"] is False,
             "legacy validation reference used Development TEST")
    _require(method_protocol["legacy_best_observed_validation_reference"]["evaluation_used"] is False,
             "legacy validation reference used Evaluation")

    post_exposure = method_protocol["post_exposure_boundary"]
    _require(
        post_exposure == {
            "GSE232572": "OUTCOME_EXPOSED_DO_NOT_USE_FOR_MODEL_SELECTION",
            "E-MTAB-10902": "EVALUATION_UNMATERIALIZED_DO_NOT_USE_FOR_MODEL_SELECTION",
            "GSE246381": "SEALED_EXCLUDED",
            "new_external_confirmation_required_for_new_method_claim": True,
        },
        "post-exposure boundary changed",
    )

    gse232572 = _row_by_id(rows, "GSE232572")
    emtab10902 = _row_by_id(rows, "E-MTAB-10902")
    gse246381 = _row_by_id(rows, "GSE246381")
    _require(int(gse232572["historical_transfer_canonical_records"]) == 8068,
             "historical GSE232572 record count changed")
    _require(
        gse232572["outcome_exposure"] == "OUTCOME_EXPOSED_BY_EXISTING_ZERO_SHOT",
        "historical GSE232572 exposure label changed",
    )
    _require(emtab10902["terminal_conversion_status"] == "UNCONVERTIBLE_FOR_ROUTE2_V1",
             "E-MTAB-10902 conversion status changed")
    _require(emtab10902["outcome_exposure"] == "OUTCOME_NOT_READ",
             "E-MTAB-10902 outcome-read boundary changed")
    _require(gse246381["terminal_conversion_status"] == "SEALED_EXCLUDED",
             "GSE246381 sealed status changed")
    _require(gse246381["outcome_exposure"] == "SEALED_NOT_READ",
             "GSE246381 read boundary changed")

    _require(
        readiness_protocol["single_test_metric_policy"]
        == "REPORT_ONLY_NO_STRUCTURE_LOSS_SEED_EPOCH_THRESHOLD_OR_POLICY_SELECTION",
        "single TEST use is no longer report-only",
    )
    _require(readiness_protocol["loso_gate"]["evaluation_studies_included"] == 0,
             "readiness protocol unexpectedly includes Evaluation studies")
    _require(readiness_protocol["development_test_outcomes_accessed_at_protocol_freeze"] is False,
             "Development TEST was already accessed at readiness freeze")
    _require(readiness_protocol["evaluation_outcomes_accessed"] is False,
             "Evaluation was already accessed at readiness freeze")
    _require(readiness_protocol["guided_generation_authorized"] is False,
             "readiness protocol unexpectedly authorizes guided generation")

    external = package_audit["external_evaluation"]
    guided = package_audit["guided_generation"]
    _require(external["evaluation_unexposed_canonical_records"] == 0,
             "new outcome-unexposed final Evaluation records are no longer zero")
    _require(external["replacement_study_registered"] is False,
             "a replacement Evaluation study is now registered")
    _require(external["new_final_evaluation_opened"] is False,
             "new final Evaluation has been opened")
    _require(external["gse232572_final_confirmation_eligible"] is False,
             "historical GSE232572 was promoted to final confirmation")
    _require(external["emtab10902_outcome_read"] is False,
             "E-MTAB-10902 outcome was read")
    _require(guided["critic_ready_for_guidance"] is False,
             "critic is no longer in the terminal not-ready state")
    _require(
        guided["critic_v2_control_status"]
        == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 terminal control status changed",
    )
    _require(guided["frozen_critic_xeditflow_run"] is False,
             "guided XEditFlow was run")

    return {
        "development_record_count": 126165,
        "development_split": split,
        "development_test_role": "WITHHELD_SINGLE_REPORT_ONLY_IF_GATE_PASSES",
        "critic_v2_gate_status": guided["critic_v2_control_status"],
        "critic_ready_for_guidance": False,
        "guided_xeditflow_run": False,
        "historical_gse232572": {
            "record_count": 8068,
            "role": "HISTORICALLY_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION",
            "final_confirmation_eligible": False,
        },
        "emtab10902": {
            "status": "UNCONVERTIBLE_FOR_ROUTE2_V1",
            "outcome_read": False,
            "permitted_evidence": "CONVERSION_FAILURE_ONLY",
        },
        "gse246381": {
            "status": "SEALED_EXCLUDED",
            "outcome_read": False,
        },
        "replacement_evaluation": {
            "registered": False,
            "outcome_unexposed_canonical_records": 0,
            "opened": False,
            "execution_order": [
                "FREEZE_PREDICTOR_GENERATOR_BASELINES_METRICS_AND_ADAPTATION_POLICY",
                "RUN_AND_PERMANENTLY_RECORD_ONE_NEW_STUDY_ZERO_SHOT",
                "ONLY_THEN_ALLOW_CALIBRATION_OR_FEW_SHOT_ADAPTATION",
                "ZERO_SHOT_REMAINS_HEADLINE",
            ],
        },
    }


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
    linestyle: str = "-",
    fontsize: float = 7.1,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
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
        linespacing=1.22,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK_GRAY,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.01, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=10,
            ha="left", va="bottom")


def _render(evidence: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.8),
        gridspec_kw={"height_ratios": [1.02, 0.98]},
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.925, hspace=0.29)
    fig.suptitle("Route 2 Development and external Evaluation firewall", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Frozen Development split and conditional downstream access", pad=5)
    _panel_label(ax, "A")

    _box(ax, (0.015, 0.39), 0.18, 0.22, "Development pool\n126,165 records\nfixed grouped split",
         facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=8.0)
    _box(ax, (0.27, 0.72), 0.18, 0.15, "TRAIN\n89,580",
         facecolor=LIGHT_GREEN, edgecolor=GREEN, fontsize=7.8)
    _box(ax, (0.27, 0.48), 0.18, 0.15, "VALIDATION\n18,293",
         facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=7.8)
    _box(ax, (0.27, 0.16), 0.23, 0.18,
         "Critic V2 three-seed gate\nCURRENT: NO-GO\nTEST remains closed",
         facecolor=LIGHT_ORANGE, edgecolor=VERMILLION, fontsize=7.0)
    _box(ax, (0.57, 0.72), 0.19, 0.15, "Parameter fitting\nper frozen protocol",
         facecolor=LIGHT_GREEN, edgecolor=GREEN)
    _box(ax, (0.57, 0.48), 0.19, 0.15, "Architecture/loss/control\nselection only",
         facecolor=LIGHT_BLUE, edgecolor=BLUE)
    _box(ax, (0.59, 0.17), 0.17, 0.16, "TEST withheld\n18,292\nsingle report-only read",
         facecolor=LIGHT_ORANGE, edgecolor=VERMILLION, linestyle="--", fontsize=6.8)
    _box(ax, (0.82, 0.14), 0.16, 0.22, "Conditional downstream\nall-record refit → LOSO\n→ readiness → guidance\nCURRENTLY CLOSED",
         facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linestyle="--", fontsize=6.4)

    _arrow(ax, (0.195, 0.50), (0.27, 0.795), color=GREEN)
    _arrow(ax, (0.195, 0.50), (0.27, 0.555), color=BLUE)
    _arrow(ax, (0.45, 0.795), (0.57, 0.795), color=GREEN)
    _arrow(ax, (0.45, 0.555), (0.57, 0.555), color=BLUE)
    _arrow(ax, (0.57, 0.72), (0.48, 0.34), color=DARK_GRAY, connectionstyle="arc3,rad=0.16")
    _arrow(ax, (0.57, 0.48), (0.48, 0.34), color=DARK_GRAY, connectionstyle="arc3,rad=0.08")
    _arrow(ax, (0.50, 0.25), (0.59, 0.25), color=VERMILLION, linestyle="--")
    _arrow(ax, (0.76, 0.25), (0.82, 0.25), color=DARK_GRAY, linestyle="--")
    ax.text(0.545, 0.365, "only if gate passes", color=VERMILLION, fontsize=6.0,
            ha="center", va="bottom")
    ax.text(0.01, 0.015,
            "Solid arrows: permissible Development use. Dashed arrows: prospective only. Arrow widths are not quantitative.",
            fontsize=6.4, color=DARK_GRAY, ha="left", va="bottom")

    ax = axes[1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Registered external-study roles and permanently ordered future Evaluation", pad=5)
    _panel_label(ax, "B")

    current_specs = [
        ((0.015, 0.70), "GSE232572\n8,068 records\noutcome exposed", LIGHT_ORANGE, VERMILLION),
        ((0.015, 0.43), "E-MTAB-10902\nunconvertible in V1\noutcome not read", LIGHT_GRAY, DARK_GRAY),
        ((0.015, 0.16), "GSE246381\nsealed and excluded\nnot read", LIGHT_GRAY, DARK_GRAY),
    ]
    use_specs = [
        ((0.32, 0.70), "Historical transfer/diagnostic only\nnot selection or final confirmation", LIGHT_ORANGE, VERMILLION),
        ((0.32, 0.43), "Conversion-failure evidence only\nno outcome evaluation", LIGHT_GRAY, DARK_GRAY),
        ((0.32, 0.16), "No current evidence use\nremains sealed", LIGHT_GRAY, DARK_GRAY),
    ]
    for (xy, text, fill, edge), (use_xy, use_text, use_fill, use_edge) in zip(current_specs, use_specs):
        _box(ax, xy, 0.23, 0.17, text, facecolor=fill, edgecolor=edge, fontsize=6.8)
        _box(ax, use_xy, 0.29, 0.17, use_text, facecolor=use_fill, edgecolor=use_edge, fontsize=6.6)
        _arrow(ax, (xy[0] + 0.23, xy[1] + 0.085), (use_xy[0], use_xy[1] + 0.085), color=edge)

    future_x = 0.69
    future_specs = [
        ((future_x, 0.76), "New convertible replacement\nCURRENT: absent | records = 0", LIGHT_GRAY, DARK_GRAY),
        ((future_x, 0.56), "Freeze predictor, generator,\nbaselines, metrics + adaptation policy", WHITE, BLUE),
        ((future_x, 0.36), "Run once: new-study zero-shot\npermanently record result", WHITE, BLUE),
        ((future_x, 0.12), "Only then: calibration/few-shot\nzero-shot remains headline", WHITE, GREEN),
    ]
    heights = [0.14, 0.14, 0.14, 0.17]
    for (xy, text, fill, edge), height in zip(future_specs, heights):
        _box(ax, xy, 0.29, height, text, facecolor=fill, edgecolor=edge,
             linestyle="--", fontsize=6.5)
    _arrow(ax, (0.835, 0.76), (0.835, 0.70), color=DARK_GRAY, linestyle="--")
    _arrow(ax, (0.835, 0.56), (0.835, 0.50), color=BLUE, linestyle="--")
    _arrow(ax, (0.835, 0.36), (0.835, 0.29), color=GREEN, linestyle="--")
    ax.text(0.835, 0.015,
            "Future chain is not executed.\nHistorical exposure cannot replace final Evaluation.",
            fontsize=5.8, color=DARK_GRAY, ha="center", va="bottom", linespacing=1.15)
    return fig


def _metadata(format_name: str) -> dict[str, str]:
    if format_name == "pdf":
        return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder",
                "Subject": "Provisional Development/Evaluation architecture figure"}
    if format_name == "svg":
        return {"Title": STEM, "Creator": "mRNA-EditFlow Route 2 evidence builder",
                "Description": "Provisional Development/Evaluation architecture figure"}
    return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder",
            "Description": "Provisional Development/Evaluation architecture figure"}


def build_figure(
    *,
    dataset_table: Path = DEFAULT_DATASET_TABLE,
    method_protocol: Path = DEFAULT_METHOD_PROTOCOL,
    readiness_protocol: Path = DEFAULT_READINESS_PROTOCOL,
    package_audit: Path = DEFAULT_PACKAGE_AUDIT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_table = dataset_table.resolve()
    method_protocol = method_protocol.resolve()
    readiness_protocol = readiness_protocol.resolve()
    package_audit = package_audit.resolve()
    output_directory = output_directory.resolve()
    _require(dpi >= 150, "raster DPI must be at least 150")
    _require(set(formats) <= set(FORMATS) and len(formats) > 0,
             "formats must be a nonempty subset of png,pdf,svg")

    evidence = _derive_evidence(
        _read_rows(dataset_table),
        json.loads(method_protocol.read_text(encoding="utf-8")),
        json.loads(readiness_protocol.read_text(encoding="utf-8")),
        json.loads(package_audit.read_text(encoding="utf-8")),
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / f"{STEM}_manifest.json"
    alt_text_path = output_directory / f"{STEM}_alt_text.md"
    targets = [output_directory / f"{STEM}.{format_name}" for format_name in formats]
    targets.extend([manifest_path, alt_text_path])
    for path in targets:
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing architecture artifact: {path}")

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

    alt_text = """# Route 2 V3.3.2 Development/Evaluation architecture figure

Two-panel workflow diagram. Panel A separates the fixed 126,165-record
Development pool into 89,580 TRAIN records for fitting and 18,293 VALIDATION
records for architecture, loss and control selection. The current Critic V2
three-seed gate is NO-GO, so the 18,292-record Development TEST remains closed.
The diagram marks a single report-only TEST read and all-record refit, LOSO,
readiness and guidance as conditional downstream steps that have not run.

Panel B shows three distinct registered external-study roles. GSE232572 has
8,068 historically outcome-exposed records and is limited to transfer or
diagnostic reporting, not model selection or final confirmation. E-MTAB-10902
is unconvertible under Route 2 V1 and contributes conversion-failure evidence
without an outcome read. GSE246381 remains sealed, excluded and unread. A
dashed future chain shows that no replacement Evaluation study is registered
and there are zero outcome-unexposed final Evaluation records. A new convertible
replacement must be registered, all methods and policies frozen, and one
new-study zero-shot result permanently recorded before calibration or few-shot
adaptation; the zero-shot result remains the headline. Arrow widths are not
quantitative, and dashed paths are prospective rather than completed.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")
    manifest = {
        "schema_version": "route_a_v3_route2_v332_development_evaluation_architecture_figure.v1",
        "status": "PROVISIONAL_DEVELOPMENT_EVALUATION_ARCHITECTURE_FIGURE_RENDERED",
        "target_journal": "PENDING_SELECTION",
        "article_type": "PENDING_SELECTION",
        "submission_phase": "INTERNAL_EVIDENCE_REVIEW",
        "publisher_compliance_claimed": False,
        "matplotlib_version": matplotlib.__version__,
        "python_version": platform.python_version(),
        "width_inches": 7.2,
        "height_inches": 6.8,
        "raster_dpi": dpi,
        "background": "OPAQUE_WHITE",
        "source_data": {
            "dataset_table": str(dataset_table),
            "method_protocol": str(method_protocol),
            "readiness_protocol": str(readiness_protocol),
            "minimum_package_audit": str(package_audit),
        },
        "transformations": [
            "Exact frozen Development counts and current terminal gate state rendered as labeled workflow nodes",
            "Registered external studies separated by permissible evidence role rather than legacy inventory role",
            "Future final Evaluation order rendered as dashed prospective arrows of constant width",
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
    parser.add_argument("--method-protocol", type=Path, default=DEFAULT_METHOD_PROTOCOL)
    parser.add_argument("--readiness-protocol", type=Path, default=DEFAULT_READINESS_PROTOCOL)
    parser.add_argument("--package-audit", type=Path, default=DEFAULT_PACKAGE_AUDIT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_figure(
        dataset_table=args.dataset_table,
        method_protocol=args.method_protocol,
        readiness_protocol=args.readiness_protocol,
        package_audit=args.package_audit,
        output_directory=args.output_directory,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
