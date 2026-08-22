#!/usr/bin/env python3
"""Render the provisional Route 2 predictor–XEditFlow–evaluator architecture figure."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRITIC_CONFIG = ROOT / "configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json"
DEFAULT_FLOW_CONFIG = ROOT / "configs/route_a_v3_route2_base_flow_g0_position_progress_gpu_v2.json"
DEFAULT_EVALUATOR_CONFIG = (
    ROOT / "configs/route_a_v3_route2_independent_evaluator_neural_medium_task_scaled_gpu2_v3.json"
)
DEFAULT_EVALUATOR_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_mrnabert_independent_evaluator_qualification_v1.json"
)
DEFAULT_REWARD_POLICY = ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"
DEFAULT_FRESHNESS_AUDIT = ROOT / "audits/route_a_v3_route2_v332_freshness_and_critic_v2_freeze_v1.json"
DEFAULT_GEOMETRY_AUDIT = ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
DEFAULT_PACKAGE_AUDIT = ROOT / "audits/route_a_v3_route2_v332_minimum_benchmark_package_v1.json"
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1"
)
STEM = "route2_v332_predictor_xeditflow_evaluator_architecture_figure_v1"
FORMATS = ("png", "pdf", "svg")

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#007A5A"
PURPLE = "#6F4C9B"
DARK_GRAY = "#4D4D4D"
BLACK = "#000000"
LIGHT_BLUE = "#E8F2F8"
LIGHT_ORANGE = "#FBEFE8"
LIGHT_GREEN = "#E8F4EF"
LIGHT_PURPLE = "#F0ECF6"
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


class ArchitectureInputError(RuntimeError):
    """A frozen component or information-flow boundary changed unexpectedly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchitectureInputError(message)


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_evidence(
    critic: Mapping[str, Any],
    flow: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    evaluator_protocol: Mapping[str, Any],
    reward: Mapping[str, Any],
    freshness: Mapping[str, Any],
    geometry: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    _require(critic["model_kind"] == "delta_pretrained_mrnabert_edit_centered_antisymmetric",
             "Delta critic model kind changed")
    _require(critic["expected_frozen_pretrained_parameter_count"] == 113389056,
             "frozen mRNABERT parameter count changed")
    _require(critic["expected_trainable_parameter_count"] == 9342914,
             "Delta critic trainable parameter count changed")
    _require(critic["hidden_dim"] == 384 and critic["depth"] == 10,
             "Delta critic head architecture changed")
    _require(critic["critic_position_features"] == "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED",
             "Delta critic position features changed")
    _require(critic["development_test_outcomes_accessed"] is False,
             "Delta critic config accessed Development TEST")
    _require(critic["evaluation_outcomes_accessed"] is False,
             "Delta critic config accessed Evaluation")

    _require(flow["model_kind"] == "route2_base_flow_sub_stop_position_progress",
             "Base Flow model kind changed")
    _require(flow["loss_kind"] == "next_legal_action_cross_entropy",
             "Base Flow loss changed")
    _require(flow["generation_action_space"] == "SUB_PLUS_STOP",
             "Base Flow action space changed")
    _require(flow["allowed_edit_budgets"] == [1, 3, 5],
             "Base Flow edit budgets changed")
    _require(flow["position_progress_features"] is True,
             "Base Flow position/progress features are disabled")
    _require(flow["algorithmic_time_feature"] == "CONSUMED_EDIT_BUDGET_FRACTION",
             "Base Flow progress feature changed")
    _require(flow["guided_critic_used"] is False,
             "Base Flow training config unexpectedly used guided critic")
    _require(flow["evaluation_outcomes_accessed"] is False,
             "Base Flow training config accessed Evaluation")

    _require(evaluator["scientific_role"] == "INDEPENDENT_GENERATION_EVALUATOR_NOT_GUIDING_CRITIC",
             "evaluator scientific role changed")
    _require(evaluator["model_kind"] == "siamese_cnn",
             "independent evaluator model kind changed")
    _require(evaluator["hidden_dim"] == 103 and evaluator["depth"] == 7,
             "independent evaluator architecture changed")
    _require(evaluator["checkpoint_selection"] == "FINAL_EPOCH",
             "independent evaluator checkpoint policy changed")
    _require(evaluator["evaluation_outcomes_accessed"] is False,
             "independent evaluator config accessed Evaluation")

    qualification = evaluator_protocol["independent_evaluator_qualification"]
    _require(evaluator_protocol["guide_evaluator_architecture_distinct"] is True,
             "guide and evaluator architectures are no longer declared distinct")
    _require(qualification["selection_pool"] == "DEVELOPMENT_VALIDATION",
             "independent evaluator qualification pool changed")
    _require(qualification["development_test_outcomes_accessed"] == 0,
             "independent evaluator qualification accessed TEST")
    _require(qualification["evaluation_outcomes_accessed"] == 0,
             "independent evaluator qualification accessed Evaluation")

    _require(reward["critic_parameter_update_during_generation"] is False,
             "reward policy updates critic during generation")
    _require(reward["generator_gradient_into_critic"] is False,
             "reward policy sends generator gradient into critic")
    _require(reward["evaluation_model_gradient_into_generator"] is False,
             "reward policy sends evaluator gradient into generator")
    _require(reward["reward_signal"] == "STANDARDIZED_PREDICTED_MEAN_DELTA",
             "guidance reward signal changed")
    _require(reward["uncertainty_in_guidance"] == "DISABLED_DIAGNOSTIC_ONLY",
             "uncertainty is no longer diagnostic-only")
    _require(reward["transition_rule"] == "BASE_TRANSITION_RATE_TIMES_EXP_POTENTIAL_DIFFERENCE",
             "guided transition rule changed")
    _require(reward["action_space"] == "SUB_PLUS_STOP",
             "reward-policy action space changed")
    _require(reward["evaluation_records_used_for_training_hpo_threshold_or_reward"] == 0,
             "Evaluation records entered reward calibration")
    _require(reward["generated_candidates_add_to_canonical_records"] is False,
             "generated candidates now add canonical credit")

    evaluator_terminal = freshness["independent_evaluator"]
    _require(evaluator_terminal["adjudication_status"] == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
             "independent evaluator terminal qualification changed")
    _require(evaluator_terminal["model_kind"] == "siamese_cnn",
             "terminal evaluator model kind changed")
    _require(evaluator_terminal["trainable_parameter_count"] == 509845,
             "terminal evaluator actual parameter count changed")
    _require(evaluator_terminal["development_test_outcomes_accessed"] is False,
             "terminal evaluator accessed Development TEST")
    _require(evaluator_terminal["evaluation_outcomes_accessed"] is False,
             "terminal evaluator accessed Evaluation")

    protocol_boundary = geometry["protocol_boundary"]
    _require(protocol_boundary["action_types_in_scope"] == ["SUB", "STOP"],
             "terminal action types in scope changed")
    _require(protocol_boundary["action_types_out_of_scope"] == ["INS", "DEL"],
             "terminal action types out of scope changed")
    _require(protocol_boundary["candidate_support_mode"] == "OPEN_GENERATED_SUPPORT",
             "generated candidate support mode changed")
    _require(protocol_boundary["guided_xeditflow_run"] is False,
             "geometry audit says guided XEditFlow ran")
    _require(protocol_boundary["generated_candidates_grant_canonical_credit"] is False,
             "geometry audit grants generated candidates canonical credit")
    _require(geometry["cross_method_geometry"]["all_method_hard_legality_rate"] == 1.0,
             "terminal matched generation no longer has 100% hard legality")

    guided = package["guided_generation"]
    _require(guided["flow_g0_ready"] is True,
             "Base Flow G0 is no longer ready")
    _require(guided["critic_ready_for_guidance"] is False,
             "critic is no longer terminal not-ready")
    _require(guided["critic_v2_control_status"]
             == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
             "Critic V2 terminal status changed")
    _require(guided["first_order_guidance_run"] is False,
             "first-order guidance was run")
    _require(guided["frozen_critic_xeditflow_run"] is False,
             "frozen-critic XEditFlow was run")

    return {
        "delta_critic": {
            "model_kind": critic["model_kind"],
            "frozen_encoder": "mRNABERT",
            "frozen_encoder_parameter_count": 113389056,
            "trainable_head_parameter_count": 9342914,
            "total_effective_parameter_count": 122731970,
            "hidden_dim": 384,
            "depth": 10,
            "position_features": critic["critic_position_features"],
            "reward_signal": reward["reward_signal"],
            "current_guidance_status": guided["critic_v2_control_status"],
            "critic_ready_for_guidance": False,
        },
        "legal_xeditflow": {
            "base_flow_model_kind": flow["model_kind"],
            "engineering_status": "FLOW_G0_READY",
            "guided_xeditflow_run": False,
            "action_space": "SUB_PLUS_STOP",
            "action_types_in_scope": ["SUB", "STOP"],
            "action_types_out_of_scope": ["INS", "DEL"],
            "allowed_edit_budgets": [1, 3, 5],
            "position_progress_features": True,
            "terminal_matched_generation_hard_legality_rate": 1.0,
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "generated_candidates_add_canonical_credit": False,
        },
        "independent_evaluator": {
            "model_kind": "siamese_cnn",
            "hidden_dim": 103,
            "depth": 7,
            "terminal_actual_trainable_parameter_count": 509845,
            "architecture_distinct_from_guide": True,
            "qualification_pool": "DEVELOPMENT_VALIDATION",
            "terminal_status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
            "role": "DEVELOPMENT_GENERATION_METHOD_SELECTION_NOT_BIOLOGICAL_VALIDATION",
        },
        "frozen_feedback_boundaries": {
            "critic_parameter_update_during_generation": False,
            "generator_gradient_into_critic": False,
            "evaluation_model_gradient_into_generator": False,
            "evaluation_records_used_for_reward": 0,
        },
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
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
    fontsize: float = 7.0,
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
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=BLACK, linespacing=1.20)


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK_GRAY,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(FancyArrowPatch(
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
    ))


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.01, 1.02, label, transform=ax.transAxes, fontweight="bold", fontsize=10,
            ha="left", va="bottom")


def _render(evidence: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), gridspec_kw={"height_ratios": [1.06, 0.94]})
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.925, hspace=0.30)
    fig.suptitle("Route 2 predictor–generator–evaluator separation", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Component architectures and scientific roles", pad=5)
    _panel_label(ax, "A")

    columns = [0.015, 0.345, 0.675]
    widths = 0.31
    headings = [
        ("Delta predictor / critic\nCURRENT: NOT READY FOR GUIDANCE", LIGHT_ORANGE, VERMILLION),
        ("Legal XEditFlow generator\nBase Flow ready | guided NOT RUN", LIGHT_BLUE, BLUE),
        ("Independent evaluator\nQUALIFIED for method selection", LIGHT_GREEN, GREEN),
    ]
    for x, (text, fill, edge) in zip(columns, headings):
        _box(ax, (x, 0.80), widths, 0.14, text, facecolor=fill, edgecolor=edge,
             fontsize=7.1)

    critic_nodes = [
        ((columns[0] + 0.02, 0.59), "Source + candidate sequence\nedit identity/position + context", LIGHT_GRAY, DARK_GRAY),
        ((columns[0] + 0.02, 0.36), "Frozen mRNABERT encoder\n113,389,056 parameters", LIGHT_PURPLE, PURPLE),
        ((columns[0] + 0.02, 0.10), "Edit-centered local + global head\nantisymmetric Δ | 9,342,914 trainable\nstandardized predicted mean Δ", LIGHT_ORANGE, VERMILLION),
    ]
    flow_nodes = [
        ((columns[1] + 0.02, 0.59), "Source state + edit budget\nallowed budgets: 1 / 3 / 5", LIGHT_GRAY, DARK_GRAY),
        ((columns[1] + 0.02, 0.36), "Position/progress Base Flow\nnext-legal-action rates\nFLOW_G0_READY", LIGHT_BLUE, BLUE),
        ((columns[1] + 0.02, 0.10), "Legal action graph: SUB + STOP\nINS/DEL out of scope\nopen-support candidates", LIGHT_BLUE, BLUE),
    ]
    evaluator_nodes = [
        ((columns[2] + 0.02, 0.59), "Generated source–candidate pair\nfull Development context", LIGHT_GRAY, DARK_GRAY),
        ((columns[2] + 0.02, 0.36), "Distinct Siamese CNN\nhidden 103 | depth 7\n509,845 actual parameters", LIGHT_GREEN, GREEN),
        ((columns[2] + 0.02, 0.10), "Independent method-selection score\nDevelopment only\nnot biological validation", LIGHT_GREEN, GREEN),
    ]
    for nodes in (critic_nodes, flow_nodes, evaluator_nodes):
        for xy, text, fill, edge in nodes:
            _box(ax, xy, 0.27, 0.16 if xy[1] != 0.10 else 0.19, text,
                 facecolor=fill, edgecolor=edge, fontsize=6.5)
        _arrow(ax, (nodes[0][0][0] + 0.135, 0.59), (nodes[1][0][0] + 0.135, 0.52),
               color=nodes[1][3])
        _arrow(ax, (nodes[1][0][0] + 0.135, 0.36), (nodes[2][0][0] + 0.135, 0.29),
               color=nodes[2][3])

    _arrow(ax, (0.305, 0.195), (0.365, 0.445), color=VERMILLION,
           linestyle="--", connectionstyle="arc3,rad=-0.18")
    _arrow(ax, (0.635, 0.195), (0.695, 0.67), color=GREEN,
           connectionstyle="arc3,rad=-0.18")
    ax.text(0.01, 0.015,
            "Solid arrows: executed/currently permissible flow. Dashed critic guidance: frozen design, not executed. Arrow widths are not quantitative.",
            fontsize=6.2, color=DARK_GRAY, ha="left", va="bottom")

    ax = axes[1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title("Allowed information flow, blocked feedback and current claim boundary", pad=5)
    _panel_label(ax, "B")

    _box(ax, (0.015, 0.58), 0.18, 0.17, "Source + context\nlegal mask + budget", facecolor=LIGHT_GRAY,
         edgecolor=DARK_GRAY, fontsize=7.0)
    _box(ax, (0.27, 0.58), 0.20, 0.17, "Base Flow proposals\nSUB + STOP candidates", facecolor=LIGHT_BLUE,
         edgecolor=BLUE, fontsize=7.0)
    _box(ax, (0.55, 0.72), 0.20, 0.15, "Frozen critic potential\nCURRENTLY BLOCKED", facecolor=LIGHT_ORANGE,
         edgecolor=VERMILLION, linestyle="--", fontsize=6.8)
    _box(ax, (0.55, 0.48), 0.20, 0.15, "Independent evaluator\npost-generation scoring", facecolor=LIGHT_GREEN,
         edgecolor=GREEN, fontsize=6.8)
    _box(ax, (0.82, 0.69), 0.16, 0.16, "Critic self-score\nnot measured outcome", facecolor=LIGHT_ORANGE,
         edgecolor=VERMILLION, linestyle="--", fontsize=6.3)
    _box(ax, (0.82, 0.45), 0.16, 0.16, "Method selection score\nno generator feedback", facecolor=LIGHT_GREEN,
         edgecolor=GREEN, fontsize=6.3)
    _box(ax, (0.82, 0.21), 0.16, 0.16, "Measured outcome\nnew external: unavailable", facecolor=LIGHT_GRAY,
         edgecolor=DARK_GRAY, linestyle=":", fontsize=6.3)

    _arrow(ax, (0.195, 0.665), (0.27, 0.665), color=BLUE)
    _arrow(ax, (0.47, 0.665), (0.55, 0.795), color=VERMILLION, linestyle="--")
    _arrow(ax, (0.47, 0.665), (0.55, 0.555), color=GREEN)
    _arrow(ax, (0.75, 0.795), (0.82, 0.77), color=VERMILLION, linestyle="--")
    _arrow(ax, (0.75, 0.555), (0.82, 0.53), color=GREEN)
    _arrow(ax, (0.47, 0.62), (0.82, 0.29), color=DARK_GRAY, linestyle=":",
           connectionstyle="arc3,rad=0.30")
    ax.text(0.66, 0.405, "NO evaluator → generator gradient\nNO critic parameter update during generation",
            fontsize=6.0, color=VERMILLION, ha="center", va="center")

    status_specs = [
        ((0.015, 0.04), "FLOW_G0_READY\nengineering only", LIGHT_BLUE, BLUE),
        ((0.215, 0.04), "Critic V2 NO-GO\nfor guidance", LIGHT_ORANGE, VERMILLION),
        ((0.415, 0.04), "Evaluator QUALIFIED\nmethod selection only", LIGHT_GREEN, GREEN),
        ((0.615, 0.04), "Guided XEditFlow\nNOT RUN", LIGHT_GRAY, DARK_GRAY),
        ((0.815, 0.04), "Biological/external\nNOT ESTABLISHED", LIGHT_GRAY, BLACK),
    ]
    for xy, text, fill, edge in status_specs:
        _box(ax, xy, 0.17, 0.11, text, facecolor=fill, edgecolor=edge, fontsize=5.8)
    return fig


def _metadata(format_name: str) -> dict[str, str]:
    description = "Provisional predictor–XEditFlow–evaluator architecture figure"
    if format_name == "pdf":
        return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder", "Subject": description}
    if format_name == "svg":
        return {"Title": STEM, "Creator": "mRNA-EditFlow Route 2 evidence builder", "Description": description}
    return {"Title": STEM, "Author": "mRNA-EditFlow Route 2 evidence builder", "Description": description}


def build_figure(
    *,
    critic_config: Path = DEFAULT_CRITIC_CONFIG,
    flow_config: Path = DEFAULT_FLOW_CONFIG,
    evaluator_config: Path = DEFAULT_EVALUATOR_CONFIG,
    evaluator_protocol: Path = DEFAULT_EVALUATOR_PROTOCOL,
    reward_policy: Path = DEFAULT_REWARD_POLICY,
    freshness_audit: Path = DEFAULT_FRESHNESS_AUDIT,
    geometry_audit: Path = DEFAULT_GEOMETRY_AUDIT,
    package_audit: Path = DEFAULT_PACKAGE_AUDIT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    formats: Sequence[str] = FORMATS,
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    inputs = {
        "critic_config": critic_config.resolve(),
        "flow_config": flow_config.resolve(),
        "evaluator_config": evaluator_config.resolve(),
        "evaluator_protocol": evaluator_protocol.resolve(),
        "reward_policy": reward_policy.resolve(),
        "freshness_audit": freshness_audit.resolve(),
        "geometry_audit": geometry_audit.resolve(),
        "minimum_package_audit": package_audit.resolve(),
    }
    output_directory = output_directory.resolve()
    _require(dpi >= 150, "raster DPI must be at least 150")
    _require(set(formats) <= set(FORMATS) and len(formats) > 0,
             "formats must be a nonempty subset of png,pdf,svg")
    evidence = _derive_evidence(*(_load(path) for path in inputs.values()))

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

    alt_text = """# Route 2 V3.3.2 predictor–XEditFlow–evaluator architecture figure

Two-panel architecture diagram. Panel A separates three components. The Delta
predictor uses source and candidate sequence, edit and context features, a
frozen 113,389,056-parameter mRNABERT encoder and a 9,342,914-parameter
edit-centered antisymmetric head. Its current Critic V2 state is not ready for
guidance. The generator uses a position/progress Base Flow and a legal SUB plus
STOP action graph with edit budgets 1, 3 or 5. Base Flow engineering is ready,
but frozen-critic guided XEditFlow has not run. The independent evaluator is an
architecturally distinct Siamese CNN with hidden dimension 103, depth 7 and
509,845 actual trainable parameters. It is qualified only for Development
generation-method selection, not biological validation.

Panel B shows source and context entering Base Flow proposals. A dashed frozen
critic-potential branch is prospective and currently blocked, whereas generated
candidates may be scored by the qualified independent evaluator. The critic
self-score, independent method-selection score and measured outcome are kept in
separate boxes. There is no evaluator-to-generator gradient, no generator
gradient into the critic and no critic parameter update during generation.
Measured external outcome is unavailable, generated candidates add no canonical
credit, and biological or external success is not established. Arrow widths are
not quantitative.
"""
    alt_text_path.write_text(alt_text, encoding="utf-8")
    manifest = {
        "schema_version": "route_a_v3_route2_v332_predictor_xeditflow_evaluator_architecture_figure.v1",
        "status": "PROVISIONAL_PREDICTOR_XEDITFLOW_EVALUATOR_ARCHITECTURE_FIGURE_RENDERED",
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
        "source_data": {key: str(path) for key, path in inputs.items()},
        "transformations": [
            "Frozen component configuration and terminal role fields rendered as separate architecture lanes",
            "Executed/current information flow rendered solid and prospective critic guidance rendered dashed",
            "Critic self-score, independent evaluator score and unavailable measured outcome kept separate",
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
    parser.add_argument("--critic-config", type=Path, default=DEFAULT_CRITIC_CONFIG)
    parser.add_argument("--flow-config", type=Path, default=DEFAULT_FLOW_CONFIG)
    parser.add_argument("--evaluator-config", type=Path, default=DEFAULT_EVALUATOR_CONFIG)
    parser.add_argument("--evaluator-protocol", type=Path, default=DEFAULT_EVALUATOR_PROTOCOL)
    parser.add_argument("--reward-policy", type=Path, default=DEFAULT_REWARD_POLICY)
    parser.add_argument("--freshness-audit", type=Path, default=DEFAULT_FRESHNESS_AUDIT)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--package-audit", type=Path, default=DEFAULT_PACKAGE_AUDIT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_figure(
        critic_config=args.critic_config,
        flow_config=args.flow_config,
        evaluator_config=args.evaluator_config,
        evaluator_protocol=args.evaluator_protocol,
        reward_policy=args.reward_policy,
        freshness_audit=args.freshness_audit,
        geometry_audit=args.geometry_audit,
        package_audit=args.package_audit,
        output_directory=args.output_directory,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
