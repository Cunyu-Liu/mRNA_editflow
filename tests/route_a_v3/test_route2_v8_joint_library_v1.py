"""Unit tests for core/route2_v8_joint_library_v1.py (V8 Stage 1 joint pipeline)."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import core.route2_v8_joint_library_v1 as jl  # noqa: E402
from core.route2_v8_joint_library_v1 import (  # noqa: E402
    DomainBalancedSampler,
    audit_leak_flags,
    build_protected_index,
    format_sequence,
    load_cms_library,
    load_mrl_library,
    load_polya_library,
    prepare_domain_library,
    resolve_libraries,
    standardize,
    _blocks,
)

MRNABERT_PATH = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
    "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
)
requires_assets = pytest.mark.skipif(not MRNABERT_PATH.exists(), reason="mRNABERT assets not mounted")


def test_format_sequence() -> None:
    assert format_sequence("AUGC") == "A T G C"
    assert format_sequence("acgn") == "A C G N"


def test_mrl_loader_merges_replicates(tmp_path: Path) -> None:
    for name, values in (("GSM3130435_egfp_unmod_1.csv.gz", {"AAAA": 1.0, "CCCC": 3.0}),
                         ("GSM3130436_egfp_unmod_2.csv.gz", {"AAAA": 3.0, "TTTT": 5.0})):
        with gzip.open(tmp_path / name, "wt") as handle:
            handle.write("utr,rl\n")
            for utr, rl in values.items():
                handle.write(f"{utr},{rl}\n")
    sequences, activities = load_mrl_library(tmp_path)
    merged = dict(zip(sequences, activities))
    assert merged["AAAA"] == pytest.approx(2.0)  # replicate mean
    assert merged["CCCC"] == pytest.approx(3.0)
    assert merged["TTTT"] == pytest.approx(5.0)


def test_polya_loader_count_filter_and_log2odds(tmp_path: Path) -> None:
    lib = tmp_path / "lib.csv.gz"
    with gzip.open(lib, "wt") as handle:
        handle.write("seq,proximal_count,total_count_vs_distal\n")
        handle.write("AAAA,90,100\n")     # p=0.9
        handle.write("CCCC,10,100\n")     # p=0.1
        handle.write("GGGG,1,5\n")        # total < 10 -> dropped
        handle.write("TTTT,0,20\n")       # p clipped to 1e-4
    sequences, activities = load_polya_library(lib, min_total_count=10, p_clip=1e-4)
    assert sequences == ["AAAA", "CCCC", "TTTT"]
    assert activities[0] == pytest.approx(np.log2(0.9 / 0.1))
    assert activities[1] == pytest.approx(np.log2(0.1 / 0.9))
    assert activities[2] == pytest.approx(np.log2(1e-4 / (1 - 1e-4)))


def test_cms_stub_missing_data_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CMS array data not yet downloaded"):
        load_cms_library(tmp_path / "cms_array_activity.csv")


def test_cms_loader_parses_frozen_schema(tmp_path: Path) -> None:
    path = tmp_path / "cms_array_activity.csv"
    path.write_text("sequence,activity,cell_context\nACGT,1.5,0\nTGCA,-0.5,2\n")
    sequences, activities, contexts = load_cms_library(path)
    assert sequences == ["ACGT", "TGCA"]
    assert activities.tolist() == [1.5, -0.5]
    assert contexts == [0, 2]
    path.write_text("sequence,activity\nACGT,1.0\n")
    sequences, activities, contexts = load_cms_library(path)
    assert contexts == [0]  # cell_context optional


def test_resolve_libraries_skips_missing_cms(monkeypatch) -> None:
    monkeypatch.setattr(jl, "cms_library_available", lambda path=None: False)
    active, skipped = resolve_libraries(["mrl", "cms"])
    assert active == ["mrl"]
    assert [s["domain"] for s in skipped] == ["cms"]
    assert "stub" in skipped[0]["reason"] or "not yet downloaded" in skipped[0]["reason"]
    monkeypatch.setattr(jl, "cms_library_available", lambda path=None: True)
    active, skipped = resolve_libraries(["mrl", "cms"])
    assert active == ["mrl", "cms"] and skipped == []
    with pytest.raises(ValueError):
        resolve_libraries(["nosuch"])


def _protected_50mer() -> str:
    return "ACGT" * 12 + "AC"


def test_audit_flags_near_duplicates_consecutive_thirds() -> None:
    protected = _protected_50mer()
    scheme = jl.STUDY_BLOCK_SCHEMES["GSE114002"]
    index = {"GSE114002": {}}
    for block in _blocks(protected, scheme):
        index["GSE114002"].setdefault(block, set()).add(protected)
    one_mismatch = protected[:10] + ("T" if protected[10] != "T" else "A") + protected[11:]
    many_mismatch = "".join("A" if c != "A" else "T" for c in protected)
    flags = audit_leak_flags([protected, one_mismatch, many_mismatch, "TTTT" * 12 + "TT"], index)
    assert flags["GSE114002"].tolist() == [True, True, False, False]


def test_audit_flags_near_duplicates_first_mid_last() -> None:
    protected = "ACGT" * 15  # 60-mer
    scheme = jl.STUDY_BLOCK_SCHEMES["GSE269595"]
    index = {"GSE269595": {}}
    for block in _blocks(protected, scheme):
        index["GSE269595"].setdefault(block, set()).add(protected)
    mid = len(protected) // 2
    one_mismatch = protected[:mid] + ("T" if protected[mid] != "T" else "A") + protected[mid + 1 :]
    flags = audit_leak_flags([protected, one_mismatch, "G" * 60], index)
    assert flags["GSE269595"].tolist() == [True, True, False]


def test_standardize() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    z, mean, std = standardize(values)
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(values.std())
    assert z.mean() == pytest.approx(0.0, abs=1e-7)
    assert z.std() == pytest.approx(1.0, abs=1e-7)
    with pytest.raises(ValueError):
        standardize(np.asarray([7.0, 7.0, 7.0]))


def test_sampler_equal_proportions_two_domains() -> None:
    sampler = DomainBalancedSampler({"mrl": 100, "polya": 300}, batch_size=10, seed=1)
    assert sampler.steps_per_epoch == 40
    counts = {"mrl": np.zeros(100, dtype=int), "polya": np.zeros(300, dtype=int)}
    for batch in sampler.epoch_batches(epoch=0):
        assert set(batch) == {"mrl", "polya"}
        assert len(batch["mrl"]) == 5 and len(batch["polya"]) == 5  # exact 50/50
        for domain, idx in batch.items():
            counts[domain][idx] += 1
    # smaller domain cycles exactly twice per epoch (200 draws / 100 rows)
    assert counts["mrl"].tolist() == [2] * 100
    assert counts["polya"].sum() == 200


def test_sampler_remainder_rotation_three_domains() -> None:
    sampler = DomainBalancedSampler({"mrl": 50, "polya": 50, "cms": 50}, batch_size=10, seed=3)
    totals = {"mrl": 0, "polya": 0, "cms": 0}
    for batch in sampler.epoch_batches(epoch=0):
        quotas = {d: len(idx) for d, idx in batch.items()}
        assert sum(quotas.values()) == 10
        assert all(3 <= q <= 4 for q in quotas.values())  # base 3, at most one domain at 4
        for d, q in quotas.items():
            totals[d] += q
    assert max(totals.values()) - min(totals.values()) <= 1  # remainder rotated fairly


def _batches_equal(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    for x, y in zip(left, right):
        if set(x) != set(y):
            return False
        if not all(np.array_equal(x[d], y[d]) for d in x):
            return False
    return True


def test_sampler_seed_reproducibility() -> None:
    sampler_a = DomainBalancedSampler({"mrl": 20, "polya": 30}, batch_size=8, seed=7)
    sampler_b = DomainBalancedSampler({"mrl": 20, "polya": 30}, batch_size=8, seed=7)
    batches_a = list(sampler_a.epoch_batches(epoch=0))
    batches_b = list(sampler_b.epoch_batches(epoch=0))
    assert _batches_equal(batches_a, batches_b)
    batches_epoch1 = list(sampler_a.epoch_batches(epoch=1))
    assert not _batches_equal(batches_a, batches_epoch1)  # per-epoch reshuffle


def test_sampler_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        DomainBalancedSampler({}, batch_size=8, seed=1)
    with pytest.raises(ValueError):
        DomainBalancedSampler({"mrl": 10, "polya": 10, "cms": 10}, batch_size=2, seed=1)


def test_sampler_domain_draws_per_epoch_matches_batches() -> None:
    sampler = DomainBalancedSampler({"mrl": 100, "polya": 300}, batch_size=10, seed=1)
    draws = sampler.domain_draws_per_epoch()
    assert draws == {"mrl": 200, "polya": 200}
    assert sum(draws.values()) == sampler.steps_per_epoch * 10


@requires_assets
def test_prepare_domain_library_end_to_end() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    sequences = ["ACGT" * 10, "TTTT" * 10, "GGGG" * 10]
    activities = np.asarray([1.0, 2.0, 3.0])
    flags = {"GSE114002": np.asarray([False, True, False]), "GSE269595": np.asarray([False, False, False])}
    library = prepare_domain_library("mrl", sequences, activities, flags, tokenizer)
    assert library.domain_id == 0
    assert library.n_raw == 3 and library.n_clean == 2  # flagged row excluded
    assert library.input_ids.shape == (2, 42)  # 40 nt + CLS + SEP
    assert library.input_ids[0, 0].item() == 2 and library.input_ids[0, -1].item() == 3
    assert bool((library.attention_mask == 1).all())
    # standardised on the CLEAN subset only
    assert library.target_mean == pytest.approx(activities[[0, 2]].mean())
    assert library.targets.tolist() == pytest.approx([-1.0, 1.0], abs=1e-6)
    summary = library.audit_summary()
    assert summary["n_flagged"] == 1 and summary["per_study_flagged"] == {"GSE114002": 1, "GSE269595": 0}


def test_build_protected_index_reads_canonical(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        '{"canonical_record_id": 1, "source_sequence": "%s", "candidate_sequence": "%s"}\n'
        % (_protected_50mer(), "TTTT" * 12 + "TT")
    )
    index = build_protected_index({"GSE114002": canonical})
    assert set(index) == {"GSE114002"}
    flags = audit_leak_flags([_protected_50mer(), "CCCC" * 12 + "CC"], index)
    assert flags["GSE114002"].tolist() == [True, False]
