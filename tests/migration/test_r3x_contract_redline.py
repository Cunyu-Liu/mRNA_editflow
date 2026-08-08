from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
REDLINE = ROOT / "docs/contracts/mrna_xeditflow_v2_1_C3_redline_PROPOSAL.md"
MATRIX = ROOT / "docs/execution/xeditflow_claim_matrix_v2_1_PROPOSAL.yaml"
PARENT_SHA = "9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e"
ACTIVE_SHA = "fc9c1c882efbaa4c1e86f4da2e1be64e219755fb9c5941da4b4309793d3d8c2f"


def test_redline_is_explicitly_inactive_and_parent_bound() -> None:
    text = REDLINE.read_text(encoding="utf-8")
    assert "PROPOSED_REDLINE_NOT_ACTIVE" in text
    assert PARENT_SHA in text
    assert ACTIVE_SHA in text
    assert "SOURCE_PROPOSAL_NOT_ACTIVE" in text
    assert "active_contract_mutation: NOT_AUTHORIZED" in text
    assert "macro sign accuracy >= 0.60" in text
    assert "sealed" in text.lower()
    assert "DGM/Doob-derived" in text


def test_claim_matrix_keeps_effect_and_sealed_gates() -> None:
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert data["revision_authority"]["status"] == "PROPOSED_REDLINE_NOT_ACTIVE"
    assert data["revision_authority"]["parent_contract_sha256"] == PARENT_SHA
    assert data["revision_authority"]["active_contract_mutation"] is False
    assert data["selected_route"] == "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY"
    assert data["revision_authority"]["parent_contract_authority_role"] == (
        "SOURCE_PROPOSAL_NOT_ACTIVE"
    )
    assert data["revision_authority"]["current_active_executable_contract_sha256"] == ACTIVE_SHA
    assert data["revision_authority"]["successor_must_explicitly_supersede_active_v1_1"] is True
    assert data["effect_gate"]["conjunctive"] is True
    assert data["effect_gate"]["macro_sign_accuracy"] == ">=0.60"
    assert data["sealed_policy"]["development_access"] == "PROHIBITED"
    assert data["sealed_policy"]["primary_access"] == "ONE_TIME_AFTER_ALL_ORDINARY_GATES_FROZEN"
    assert data["sealed_policy"]["post_unblinding_retuning"] == "PROHIBITED"


def test_c3_is_adaptation_and_new_theory_stays_locked() -> None:
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    c3 = data["claims"]["C3_GUIDANCE_ADAPTATION"]
    theory = data["claims"]["C3_NEW_GUIDANCE_THEORY"]
    assert c3["status"] == "LOCKED_PENDING_ORDINARY_EMPIRICAL_GATE"
    assert c3["allowed_wording"] == "DGM/Doob-derived learned rate guidance"
    assert theory["status"] == "PROHIBITED_UNLESS_G0_NEW_THEORY_PASS"
    assert data["bare_exact_label"] == "PROHIBITED"


def test_guidance_claim_axes_are_orthogonal_and_required() -> None:
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    axes = data["guidance_claim_axes"]
    required = {
        "computational_exactness",
        "target_epistemic_status",
        "method_provenance",
    }
    assert axes["relationship"] == "ORTHOGONAL_ALL_THREE_REQUIRED"
    assert set(axes["required_fields"]) == required
    assert "guidance_exactness_labels" not in data
    assert "BIOLOGICAL_TARGET_VALIDITY" in axes["computational_exactness"]["does_not_assert"]
    assert "COMPUTATIONAL_EXACTNESS" in axes["target_epistemic_status"]["does_not_assert"]
    assert "COMPUTATIONAL_EXACTNESS" in axes["method_provenance"]["does_not_assert"]

    for artifact in data["guidance_artifact_defaults"].values():
        assert required.issubset(artifact)
        for field in required:
            assert artifact[field] in axes[field]["allowed_values"]


def test_current_r2_is_historical_and_r2_style_adaptation_is_not_overclaimed() -> None:
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    current = data["critic_candidates"]["current_R2"]
    adaptation = data["critic_candidates"]["R2_style_no_explicit_study_adaptation"]
    future = data["critic_candidates"]["future_faithful_R2_no_study_comparator"]

    assert current["status"] == "HISTORICAL_DIAGNOSTIC_ONLY"
    assert current["information_set"] == "PAIRED_SEQUENCE_PLUS_STUDY_PROXY"
    assert current["contains_study_proxy"] is True
    assert current["same_information_set_comparator_eligible"] is False
    assert current["strongest_baseline_gate_eligible"] is False

    assert adaptation["information_set"] == "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    assert adaptation["contains_explicit_study_id_input_or_embedding"] is False
    assert adaptation["semantic_study_proxy_exclusion_established"] is False
    assert adaptation["historical_r2_single_variable_ablation"] is False
    assert adaptation["strongest_baseline_gate_eligibility"] is False
    assert "FROZEN_HISTORICAL_R2_ARCHITECTURE_CROSSWALK" in (
        adaptation["missing_qualification_evidence"]
    )
    assert future["status"] == "NOT_IMPLEMENTED"
    assert "SEMANTIC_STUDY_PROXY_AUDIT_PASS" in future["qualification_requirements"]
    assert data["r2_same_information_set_comparator_policy"] == (
        "CURRENT_R2_STYLE_ADAPTATION_NOT_STRONGEST_BASELINE_ELIGIBLE"
    )


def test_claim_layers_cannot_backdoor_effect_failure() -> None:
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    claims = data["claims"]
    assert "GO_EFFECT" in claims["BENCHMARK_CRITIC"]["prerequisites"]
    assert "GO_EFFECT" in claims["MEASURED_NEIGHBORHOOD_OPTIMIZATION"]["prerequisites"]
    assert "GO_EFFECT" in claims["C3_GUIDANCE_ADAPTATION"]["prerequisites"]
    assert data["effect_gate"]["substitutions_prohibited"]
