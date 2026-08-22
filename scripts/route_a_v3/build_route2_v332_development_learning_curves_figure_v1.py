#!/usr/bin/env python3
"""Render Route 2 V3.3.2 learning curves from frozen Development histories.

This builder does not train or monitor models.  It reads terminal Development
histories only.  Predictor histories expose pooled Validation Spearman per
epoch, whereas the architecture audit records the separate task-macro
selection statistic; the two metrics are deliberately not conflated.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HPO_AUDIT = ROOT / "audits/route_a_v3_route2_neural_hpo_selection_v1.json"
DEFAULT_CRITIC_AUDIT = (
    ROOT / "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json"
)
DEFAULT_FRESHNESS_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_freshness_and_critic_v2_freeze_v1.json"
)
DEFAULT_CRITIC_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/"
    "task_study_macro_screen_seed20260825_v1"
)
DEFAULT_EVALUATOR_SUMMARY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/"
    "independent_generation_evaluator/"
    "neural_medium_siamese_task_scaled_seed20260816_"
    "frozen_development_validation_gpu2_v3/training_summary.json"
)
DEFAULT_FLOW_SUMMARY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/base_flow_g0/"
    "position_progress_gpu_v2/training_summary.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)

STEM = "route2_v332_development_learning_curves_figure_v1"
FORMATS = ("png", "pdf", "svg")

PREDICTOR_PROFILES = (
    "DELTA_DIAGNOSTIC_0_08M",
    "DELTA_MAIN_2M",
    "NEURAL_MAIN_2M_CANDIDATE_CNN",
    "NEURAL_MAIN_2M_FULL_PAIR_CNN",
    "NEURAL_MAIN_2M_SIAMESE_CNN",
    "NEURAL_MAIN_2M_SMALL_TRANSFORMER",
)
PREDICTOR_LABELS = {
    "DELTA_DIAGNOSTIC_0_08M": "Delta diagnostic 0.08M",
    "DELTA_MAIN_2M": "Delta 2M",
    "NEURAL_MAIN_2M_CANDIDATE_CNN": "Candidate CNN 2M",
    "NEURAL_MAIN_2M_FULL_PAIR_CNN": "Full-pair CNN 2M",
    "NEURAL_MAIN_2M_SIAMESE_CNN": "Siamese CNN 2M",
    "NEURAL_MAIN_2M_SMALL_TRANSFORMER": "Small Transformer 2M",
}
CRITIC_ARMS = ("full", "candidate_permutation", "source_only", "source_edit_metadata")
CRITIC_LABELS = {
    "full": "Full",
    "candidate_permutation": "Candidate permutation",
    "source_only": "Source only",
    "source_edit_metadata": "Source + edit metadata",
}

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
AMBER = "#E69F00"
BLACK = "#000000"
DARK_GRAY = "#4D4D4D"
LIGHT_GRAY = "#D9D9D9"
COLORS = (BLUE, VERMILLION, GREEN, PURPLE, SKY, AMBER)
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)))
MARKERS = ("o", "s", "D", "^", "v", "P")

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "grid.color": LIGHT_GRAY,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.65,
    "lines.linewidth": 1.25,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


class LearningCurveInputError(RuntimeError):
    """Frozen learning-curve evidence changed or crossed a protected boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearningCurveInputError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _history(summary: Mapping[str, Any], expected_epochs: int, label: str) -> list[Mapping[str, Any]]:
    history = summary.get("history")
    _require(isinstance(history, list), f"{label} has no terminal history list")
    _require(len(history) == expected_epochs, f"{label} history must contain {expected_epochs} epochs")
    _require(
        [row.get("epoch") for row in history] == list(range(1, expected_epochs + 1)),
        f"{label} epoch sequence changed",
    )
    return history


def _protected_predictor_summary(summary: Mapping[str, Any], label: str) -> None:
    _require(summary.get("development_test_outcomes_evaluated") is False,
             f"{label} accessed Development TEST")
    _require(summary.get("evaluation_outcomes_read") == 0,
             f"{label} accessed Evaluation")


def _selected_trial(selection: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    trial_id = selection.get("selected_trial_id")
    matches = [row for row in selection.get("all_trials_ranked", []) if row.get("trial_id") == trial_id]
    _require(len(matches) == 1, f"{profile} selected trial is not uniquely resolved")
    return matches[0]


def _derive_inputs(
    *,
    hpo_audit: Mapping[str, Any],
    critic_audit: Mapping[str, Any],
    freshness_audit: Mapping[str, Any],
    predictor_summaries: Mapping[str, Mapping[str, Any]],
    critic_summaries: Mapping[str, Mapping[str, Any]],
    evaluator_summary: Mapping[str, Any],
    flow_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _require(hpo_audit.get("selection_pool") == "DEVELOPMENT_VALIDATION",
             "predictor HPO selection pool changed")
    _require(hpo_audit.get("development_test_outcomes_accessed") is False,
             "predictor HPO audit accessed Development TEST")
    _require(hpo_audit.get("evaluation_outcomes_accessed") is False,
             "predictor HPO audit accessed Evaluation")
    selections = hpo_audit.get("selections", {})
    _require(all(profile in selections for profile in PREDICTOR_PROFILES),
             "a required predictor profile is missing")
    _require(set(predictor_summaries) == set(PREDICTOR_PROFILES),
             "predictor summary set must match the declared six-profile panel")

    predictors: list[dict[str, Any]] = []
    for profile in PREDICTOR_PROFILES:
        selection = selections[profile]
        _require(
            selection.get("selection_primary_metric")
            == "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN",
            f"{profile} selection metric changed",
        )
        trial = _selected_trial(selection, profile)
        summary = predictor_summaries[profile]
        history = _history(summary, 8, profile)
        _protected_predictor_summary(summary, profile)
        pooled = [float(row["validation"]["spearman"]) for row in history]
        _require(all(math.isfinite(value) for value in pooled),
                 f"{profile} pooled Validation Spearman is non-finite")
        task_macro = trial.get("task_macro_spearman")
        _require(task_macro is not None and math.isfinite(float(task_macro)),
                 f"{profile} selected task-macro Spearman is undefined")
        predictors.append({
            "profile_id": profile,
            "label": PREDICTOR_LABELS[profile],
            "selected_trial_id": trial["trial_id"],
            "model_kind": trial["model_kind"],
            "parameter_count": int(trial["parameter_count"]),
            "epochs": [int(row["epoch"]) for row in history],
            "pooled_validation_spearman": pooled,
            "selected_task_macro_spearman": float(task_macro),
        })

    screen = critic_audit.get("control_screen", {})
    _require(
        screen.get("status") == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 terminal NO-GO status changed",
    )
    _require(screen.get("supports_three_frozen_seeds") is False,
             "Critic V2 terminal screen now supports confirmation seeds")
    _require(set(critic_summaries) == set(CRITIC_ARMS),
             "Critic V2 summary set changed")
    protected = critic_audit.get("protected_outcomes", {})
    _require(protected.get("development_test_outcomes_accessed") is False,
             "Critic V2 audit accessed Development TEST")
    _require(protected.get("evaluation_outcomes_accessed") is False,
             "Critic V2 audit accessed Evaluation")

    critic_rows: list[dict[str, Any]] = []
    for arm in CRITIC_ARMS:
        terminal = screen["arms"][arm]
        summary = critic_summaries[arm]
        history = _history(summary, 100, f"Critic V2 {arm}")
        _protected_predictor_summary(summary, f"Critic V2 {arm}")
        selected_epoch = int(terminal["selected_epoch"])
        _require(summary.get("selected_epoch") == selected_epoch,
                 f"Critic V2 {arm} selected epoch changed")
        values = [float(row["validation"]["task_macro_spearman"]) for row in history]
        _require(_close(values[selected_epoch - 1], terminal["task_macro_spearman"]),
                 f"Critic V2 {arm} selected metric does not match terminal audit")
        critic_rows.append({
            "arm_id": arm,
            "label": CRITIC_LABELS[arm],
            "epochs": list(range(1, 101)),
            "task_macro_spearman": values,
            "selected_epoch": selected_epoch,
            "selected_task_macro_spearman": float(terminal["task_macro_spearman"]),
        })
    hurdle = float(screen["strongest_same_information_baseline"]["task_macro_spearman"])

    evaluator = freshness_audit.get("independent_evaluator", {})
    _require(evaluator.get("adjudication_status") == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
             "independent evaluator terminal qualification changed")
    _require(evaluator.get("development_test_outcomes_accessed") is False,
             "independent evaluator audit accessed Development TEST")
    _require(evaluator.get("evaluation_outcomes_accessed") is False,
             "independent evaluator audit accessed Evaluation")
    evaluator_history = _history(evaluator_summary, 8, "independent evaluator")
    _protected_predictor_summary(evaluator_summary, "independent evaluator")
    _require(evaluator_summary.get("checkpoint_selection") == "FINAL_EPOCH",
             "independent evaluator checkpoint policy changed")
    _require(evaluator_summary.get("selected_epoch") == 8,
             "independent evaluator selected epoch changed")
    evaluator_values = [
        float(row["validation"]["task_macro_spearman"]) for row in evaluator_history
    ]
    _require(_close(evaluator_values[-1], evaluator["task_macro_spearman"]),
             "independent evaluator final metric does not match freshness audit")

    flow_history = _history(flow_summary, 30, "Base Flow G0")
    _require(flow_summary.get("status") == "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
             "Base Flow terminal engineering status changed")
    _require(flow_summary.get("selected_epoch") == 1,
             "Base Flow selected epoch changed")
    _require(flow_summary.get("development_test_outcomes_evaluated") is False,
             "Base Flow accessed Development TEST")
    _require(flow_summary.get("evaluation_records_read") == 0,
             "Base Flow accessed Evaluation")
    _require(flow_summary.get("guided_critic_used") is False,
             "Base Flow unexpectedly used guided critic")
    _require(flow_summary.get("biological_optimization_established") is False,
             "Base Flow summary now claims biological optimization")

    return {
        "predictors": predictors,
        "critic": critic_rows,
        "critic_hurdle": hurdle,
        "critic_status": screen["status"],
        "evaluator": {
            "epochs": list(range(1, 9)),
            "task_macro_spearman": evaluator_values,
            "selected_epoch": 8,
            "selected_task_macro_spearman": float(evaluator["task_macro_spearman"]),
            "exclusive_threshold": float(evaluator["exclusive_threshold"]),
            "margin": float(evaluator["margin"]),
            "status": evaluator["adjudication_status"],
        },
        "flow": {
            "epochs": list(range(1, 31)),
            "train_nll": [float(row["train_nll"]) for row in flow_history],
            "validation_nll": [float(row["validation_nll"]) for row in flow_history],
            "selected_epoch": 1,
            "status": flow_summary["status"],
            "guided_critic_used": False,
            "biological_optimization_established": False,
        },
    }


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom", ha="left")


def _render(evidence: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4), layout="constrained")
    fig.suptitle(
        "Route 2 frozen Development learning curves\n"
        "Raw terminal histories; metrics are panel-specific and not cross-comparable",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0, 0]
    for index, row in enumerate(evidence["predictors"]):
        ax.plot(
            row["epochs"], row["pooled_validation_spearman"],
            color=COLORS[index], linestyle=LINESTYLES[index], marker=MARKERS[index],
            markersize=3.0,
            label=f'{row["label"]} (selected task-macro {row["selected_task_macro_spearman"]:.3f})',
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Pooled Validation Spearman",
        xticks=range(1, 9),
        title="Predictor HPO\nCurves: pooled; legend values: separate selection metric",
    )
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, columnspacing=0.8, handlelength=2.3)
    _panel_label(ax, "A")

    ax = axes[0, 1]
    for index, row in enumerate(evidence["critic"]):
        ax.plot(
            row["epochs"], row["task_macro_spearman"],
            color=COLORS[index], linestyle=LINESTYLES[index],
            marker=MARKERS[index], markevery=10, markersize=2.7,
            label=row["label"],
        )
        ax.scatter(
            [row["selected_epoch"]], [row["selected_task_macro_spearman"]],
            color=COLORS[index], marker=MARKERS[index], s=34,
            edgecolor=BLACK, linewidth=0.5, zorder=4,
        )
    ax.axhline(evidence["critic_hurdle"], color=BLACK, linestyle="--", linewidth=1.0,
               label="Strongest same-information baseline")
    ax.set(
        xlabel="Epoch",
        ylabel="Validation task-macro Spearman",
        title="Critic V2 control screen\nSelected checkpoints marked; terminal NO-GO",
    )
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, columnspacing=0.8, handlelength=2.3)
    _panel_label(ax, "B")

    ax = axes[1, 0]
    evaluator = evidence["evaluator"]
    ax.plot(
        evaluator["epochs"], evaluator["task_macro_spearman"],
        color=BLUE, marker="o", markersize=4, label="Independent evaluator",
    )
    ax.axhline(evaluator["exclusive_threshold"], color=VERMILLION, linestyle="--",
               label="Exclusive qualification threshold")
    ax.scatter(
        [evaluator["selected_epoch"]], [evaluator["selected_task_macro_spearman"]],
        color=BLUE, edgecolor=BLACK, linewidth=0.6, s=42, zorder=4,
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation task-macro Spearman",
        xticks=range(1, 9),
        title="Independent evaluator\nQualified for Development method selection only",
    )
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False)
    _panel_label(ax, "C")

    ax = axes[1, 1]
    flow = evidence["flow"]
    ax.plot(flow["epochs"], flow["train_nll"], color=BLUE, linestyle="-",
            label="Train NLL")
    ax.plot(flow["epochs"], flow["validation_nll"], color=VERMILLION,
            linestyle="--", label="Validation NLL")
    ax.scatter(
        [flow["selected_epoch"]], [flow["validation_nll"][0]],
        color=VERMILLION, marker="D", edgecolor=BLACK, linewidth=0.6, s=38,
        zorder=4, label="Selected epoch 1",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Negative log-likelihood",
        title="Base Flow G0\nEngineering-only fit; validation loss worsens",
    )
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, columnspacing=0.8)
    _panel_label(ax, "D")

    return fig


def _export(
    fig: plt.Figure,
    output_directory: Path,
    formats: Sequence[str],
    *,
    dpi: int,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for format_name in formats:
        _require(format_name in FORMATS, f"unsupported figure format: {format_name}")
        path = output_directory / f"{STEM}.{format_name}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing learning-curve artifact: {path}")
        if format_name == "pdf":
            metadata = {
                "Title": STEM,
                "Author": "mRNA-EditFlow Route 2 evidence builder",
                "Subject": "Frozen Development learning curves; provisional general-manuscript figure",
            }
        elif format_name == "svg":
            metadata = {
                "Title": STEM,
                "Creator": "mRNA-EditFlow Route 2 evidence builder",
                "Description": "Frozen Development learning curves; provisional general-manuscript figure",
            }
        else:
            metadata = {
                "Title": STEM,
                "Author": "mRNA-EditFlow Route 2 evidence builder",
                "Description": "Frozen Development learning curves; provisional general-manuscript figure",
            }
        options: dict[str, Any] = {
            "format": format_name,
            "facecolor": "white",
            "transparent": False,
            "metadata": metadata,
        }
        if format_name == "png":
            options["dpi"] = dpi
        fig.savefig(path, **options)
        outputs[format_name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "dpi": dpi if format_name == "png" else None,
        }
    return outputs


def build_figure(
    *,
    hpo_audit_path: Path = DEFAULT_HPO_AUDIT,
    critic_audit_path: Path = DEFAULT_CRITIC_AUDIT,
    freshness_audit_path: Path = DEFAULT_FRESHNESS_AUDIT,
    predictor_summary_paths: Mapping[str, Path] | None = None,
    critic_summary_paths: Mapping[str, Path] | None = None,
    evaluator_summary_path: Path = DEFAULT_EVALUATOR_SUMMARY,
    flow_summary_path: Path = DEFAULT_FLOW_SUMMARY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    hpo_audit_path = hpo_audit_path.resolve()
    critic_audit_path = critic_audit_path.resolve()
    freshness_audit_path = freshness_audit_path.resolve()
    evaluator_summary_path = evaluator_summary_path.resolve()
    flow_summary_path = flow_summary_path.resolve()
    output_directory = output_directory.resolve()
    hpo = _load(hpo_audit_path)
    critic_audit = _load(critic_audit_path)
    freshness = _load(freshness_audit_path)

    if predictor_summary_paths is None:
        predictor_summary_paths = {
            profile: Path(_selected_trial(hpo["selections"][profile], profile)["training_summary_path"])
            for profile in PREDICTOR_PROFILES
        }
    if critic_summary_paths is None:
        critic_summary_paths = {
            arm: DEFAULT_CRITIC_ROOT / arm / "training_summary.json" for arm in CRITIC_ARMS
        }
    predictor_paths = {key: Path(value).resolve() for key, value in predictor_summary_paths.items()}
    critic_paths = {key: Path(value).resolve() for key, value in critic_summary_paths.items()}
    predictor_summaries = {key: _load(path) for key, path in predictor_paths.items()}
    critic_summaries = {key: _load(path) for key, path in critic_paths.items()}
    evidence = _derive_inputs(
        hpo_audit=hpo,
        critic_audit=critic_audit,
        freshness_audit=freshness,
        predictor_summaries=predictor_summaries,
        critic_summaries=critic_summaries,
        evaluator_summary=_load(evaluator_summary_path),
        flow_summary=_load(flow_summary_path),
    )
    _require(dpi >= 150, "raster DPI must be at least 150 for manuscript review")

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / f"{STEM}_manifest.json"
    alt_text_path = output_directory / f"{STEM}_alt_text.md"
    for path in (manifest_path, alt_text_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing learning-curve artifact: {path}")

    with matplotlib.rc_context(STYLE):
        figure = _render(evidence)
        try:
            outputs = _export(
                figure, output_directory, formats, dpi=dpi, overwrite=overwrite
            )
        finally:
            plt.close(figure)

    alt_text = """# Route 2 V3.3.2 Development learning curves

Four-panel figure drawn from raw, unsmoothed terminal Development histories.
Panel A shows pooled Validation Spearman over eight epochs for six selected
predictor profiles. Its legend separately reports final task-macro Spearman
used for architecture selection; those values are not points on the plotted
curves. Panel B shows Validation task-macro Spearman over 100 epochs for the
four Critic V2 control arms, with selected checkpoints marked and the strongest
same-information baseline shown as a dashed hurdle. The full arm remains below
that hurdle, so the screen is terminal NO-GO and no guided run is authorized.
Panel C shows the independent evaluator crossing its exclusive Development
qualification threshold at the final selected epoch; qualification is only for
Development generation-method selection, not biological validation. Panel D
shows Base Flow G0 train NLL decreasing while validation NLL worsens after the
selected first epoch, an explicit overfitting pattern. Base Flow is an
engineering-only unguided component, and biological optimization is not
established. Metrics have different semantics across panels and must not be
ranked against one another. Development TEST and new final Evaluation outcomes
were not read.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")

    manifest = {
        "schema_version": "route_a_v3_route2_v332_development_learning_curves_figure.v1",
        "status": "PROVISIONAL_DEVELOPMENT_LEARNING_CURVES_FIGURE_RENDERED",
        "target_journal": "PENDING_SELECTION",
        "article_type": "PENDING_SELECTION",
        "submission_phase": "INTERNAL_EVIDENCE_REVIEW",
        "publisher_compliance_claimed": False,
        "matplotlib_version": matplotlib.__version__,
        "python_version": platform.python_version(),
        "width_inches": 8.0,
        "height_inches": 7.4,
        "raster_dpi": dpi,
        "background": "OPAQUE_WHITE",
        "raw_unsmoothed_histories": True,
        "cross_panel_metric_comparison_allowed": False,
        "predictor_curve_metric": "POOLED_DEVELOPMENT_VALIDATION_SPEARMAN",
        "predictor_selection_metric": "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN",
        "source_data": {
            "hpo_audit": str(hpo_audit_path),
            "critic_terminal_audit": str(critic_audit_path),
            "freshness_audit": str(freshness_audit_path),
            "predictor_training_summaries": {key: str(path) for key, path in predictor_paths.items()},
            "critic_training_summaries": {key: str(path) for key, path in critic_paths.items()},
            "independent_evaluator_training_summary": str(evaluator_summary_path),
            "base_flow_training_summary": str(flow_summary_path),
        },
        "panel_evidence": evidence,
        "transformations": [
            "Terminal history rows plotted in recorded epoch order without smoothing or interpolation",
            "Selected checkpoint coordinates matched to frozen terminal audits",
            "Predictor per-epoch pooled metric kept distinct from final task-macro selection statistic",
        ],
        "outputs": outputs,
        "alt_text_path": str(alt_text_path),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-audit", type=Path, default=DEFAULT_HPO_AUDIT)
    parser.add_argument("--critic-audit", type=Path, default=DEFAULT_CRITIC_AUDIT)
    parser.add_argument("--freshness-audit", type=Path, default=DEFAULT_FRESHNESS_AUDIT)
    parser.add_argument("--critic-root", type=Path, default=DEFAULT_CRITIC_ROOT)
    parser.add_argument("--evaluator-summary", type=Path, default=DEFAULT_EVALUATOR_SUMMARY)
    parser.add_argument("--flow-summary", type=Path, default=DEFAULT_FLOW_SUMMARY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    critic_paths = {
        arm: args.critic_root / arm / "training_summary.json" for arm in CRITIC_ARMS
    }
    manifest = build_figure(
        hpo_audit_path=args.hpo_audit,
        critic_audit_path=args.critic_audit,
        freshness_audit_path=args.freshness_audit,
        critic_summary_paths=critic_paths,
        evaluator_summary_path=args.evaluator_summary,
        flow_summary_path=args.flow_summary,
        output_directory=args.output_directory,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
