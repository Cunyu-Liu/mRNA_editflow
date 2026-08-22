from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/paper/route2_xedit_v3_prospective_experiments_protocol_v1.md"


def test_experiments_protocol_preserves_terminal_and_protected_result_boundaries() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "not a Results section" in text
    assert "does not assert that any running V3 arm" in text
    assert "a failed V3 gate remains a reportable terminal result" in text
    assert "Development TEST rows are not fully decoded" in text
    assert "not externally confirmed" in text


def test_experiments_protocol_covers_frozen_models_baselines_and_seeds() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for marker in (
        "C0, C1, C2, C3",
        "20260831, 20260901, and 20260902",
        "F0 replay, F1, F2, F3",
        "20260904, 20260905, and 20260906",
        "ordinary first-order guidance",
        "one-step critic-rate guidance",
        "generate-then-rerank",
        "random, greedy, beam, genetic, and local-search",
        "320 forward-equivalents per source",
    ):
        assert marker in text


def test_experiments_protocol_separates_estimands_and_statistical_units() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for marker in (
        "task-macro Spearman",
        "resample source groups within task",
        "Source macro by training seed",
        "Closed NDCG is undefined",
        "never filled with zero",
        "never merged into a composite score",
        "Decoder streams describe sampling variability",
    ):
        assert marker in text


def test_experiments_protocol_maps_claims_to_decisive_evidence() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for marker in (
        "## Claim-to-experiment map",
        "C-Confirm, then C-Test",
        "F-Screen and F-Confirm",
        "G-Closed across three training seeds",
        "Needs terminal evidence",
        "Protocol implemented; result artifacts pending",
    ):
        assert marker in text
