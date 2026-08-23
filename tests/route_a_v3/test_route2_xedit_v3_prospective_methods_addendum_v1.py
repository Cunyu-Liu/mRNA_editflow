from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = ROOT / "docs/paper/route2_xedit_v3_prospective_methods_addendum_v1.md"


def test_prospective_methods_addendum_retains_result_and_external_claim_boundaries() -> None:
    text = ADDENDUM.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("> ", "").split())
    assert "not a Results section" in text
    assert "No statement in this document asserts that a V3 performance gate has passed" in normalized
    assert "Needs terminal evidence" in text
    assert "no independent external confirmation is currently available" in text
    assert "submission-ready" in text


def test_prospective_methods_addendum_covers_both_models_and_benchmark_estimands() -> None:
    text = ADDENDUM.read_text(encoding="utf-8")
    for marker in (
        "XEditCritic V3",
        "XEditSetFlow V3",
        "set-marginal",
        "scalar soft value-to-go",
        "Sequential Monte Carlo",
        "closed measured-neighborhood",
        "Open-support evaluation",
        "paired, task-stratified bootstrap",
        "random, greedy, beam, genetic, and local-search",
    ):
        assert marker in text


def test_prospective_methods_addendum_records_realized_frozen_capacities() -> None:
    text = ADDENDUM.read_text(encoding="utf-8")
    for parameter_count in (
        "29,489,049",
        "30,472,089",
        "16,178,790",
        "42,196,934",
    ):
        assert parameter_count in text
    assert "no fourth seed is permitted" in text
    assert "unknown study retains scale one" in text
