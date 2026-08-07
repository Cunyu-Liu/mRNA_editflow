"""X0-X unit tests: CDS synonymous-codon state machine + 3'UTR region adapter.

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT PREPARATION tests.  They
do NOT touch the frozen 5' primary model, do NOT access sealed labels, and do
NOT trigger the formal X0-X gate (which is gated on frozen 5' primary model +
threshold + sealed results).

These tests verify the pure, data-free design cores in:
  * `scripts/x0x/codon.py`  : synonymous-codon CDS state, atomic synonymous
    codon substitution, and the protein-identity / frame / start / stop hard
    invariants, plus protein-family listwise metric + family split helpers.
  * `scripts/x0x/region.py` : 3'UTR region adapter with independent endpoint
    heads (5' MRL and 3' stability never pooled) and study/context transfer
    structure.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.x0x import codon
from scripts.x0x import region


# ---------------------------------------------------------------------------
# codon.py: genetic code + synonymous classes
# ---------------------------------------------------------------------------

def test_translate_basic_protein():
    # ATG (M) - AAA (K) - UUU (F) - UAA (stop)
    assert codon.translate("AUGAAAUUUUAA") == "MKF"


def test_translate_rejects_internal_stop():
    # AUG AAA UAA UUU -> internal stop at position 2 -> invalid
    assert codon.translate("AUGAAAUAAUUU") is None


def test_translate_rejects_non_multiple_of_3():
    assert codon.translate("AUGAAA") is None  # 6 nt, no stop -> incomplete
    assert codon.translate("AUGAAAUUUUA") is None  # 11 nt not %3==0


def test_synonymous_codons_consistent():
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        cods = codon.synonymous_codons(aa)
        assert len(cods) >= 1
        for c in cods:
            assert codon.GENETIC_CODE[c] == aa
    # stop has no synonymous codons
    assert codon.synonymous_codons("*") == []


def test_build_synonymous_classes_has_expected_sizes():
    classes = codon.build_synonymous_classes()
    # Leu (L) has 6 codons, Ser (S) has 6, Met (M) has 1, Trp (W) has 1
    assert len(classes["L"]) == 6
    assert len(classes["S"]) == 6
    assert len(classes["M"]) == 1
    assert len(classes["W"]) == 1
    assert "*" not in classes


# ---------------------------------------------------------------------------
# codon.py: CDS state + atomic synonymous-codon substitution invariants
# ---------------------------------------------------------------------------

# A CDS: AUG (M) - UUU (F) - AAA (K) - UUC (F) - UAA (stop)  => protein "MFKF"
_CDS = "AUGUUUAAAUUCUAA"


def test_build_cds_state_valid():
    st = codon.build_cds_state(_CDS)
    assert st.protein == "MFKF"
    assert st.n_codons == 5


def test_build_cds_state_rejects_bad_start():
    with pytest.raises(ValueError):
        codon.build_cds_state("UUUUUUAAAUUCUAA")  # no AUG start


def test_build_cds_state_rejects_bad_stop():
    with pytest.raises(ValueError):
        codon.build_cds_state("AUGUUUAAAUUCUUU")  # ends in UUU not stop


def test_enumerate_synonymous_edits_preserves_protein():
    st = codon.build_cds_state(_CDS)
    edits = codon.enumerate_synonymous_edits(st)
    assert len(edits) > 0
    for e in edits:
        assert e.is_synonymous
        # start (idx 0) and stop (idx 4) never editable by default
        assert e.codon_idx not in (0, 4)
        new = codon.apply_edit(st, e)
        assert new.protein == st.protein
        assert new.seq[0:3] == "AUG"
        assert new.seq.endswith(("UAA", "UAG", "UGA"))


def test_apply_edit_rejects_non_synonymous():
    st = codon.build_cds_state(_CDS)
    bad = codon.CodonEdit(codon_idx=1, new_codon="AAA", old_codon="UUU")
    # UUU -> AAA is Phe->Lys (non-synonymous)
    assert bad.is_synonymous is False
    with pytest.raises(ValueError):
        codon.apply_edit(st, bad)


def test_apply_edit_rejects_start_or_stop_edit():
    st = codon.build_cds_state(_CDS)
    # start codon edit -> must reject
    with pytest.raises(ValueError):
        codon.apply_edit(st, codon.CodonEdit(0, "AUU", "AUG"))
    # stop codon edit (last index 4 = UAA -> UAG, still a stop, but forbidden)
    with pytest.raises(ValueError):
        codon.apply_edit(st, codon.CodonEdit(4, "UAG", "UAA"))


def test_apply_edit_changes_sequence_but_not_protein():
    st = codon.build_cds_state(_CDS)
    e = codon.enumerate_synonymous_edits(st)[0]
    new = codon.apply_edit(st, e)
    assert new.seq != st.seq
    assert new.protein == st.protein
    # frame preserved: length unchanged
    assert new.length == st.length


# ---------------------------------------------------------------------------
# codon.py: protein-family listwise metric + family split helpers
# ---------------------------------------------------------------------------

def test_family_members_groups_by_protein():
    recs = [{"protein": "MKFF"}, {"protein": "MKFF"}, {"protein": "AAAA"}]
    fam = codon.family_members(recs)
    assert set(fam.keys()) == {"prot_MKFF", "prot_AAAA"}
    assert fam["prot_MKFF"] == [0, 1]
    assert fam["prot_AAAA"] == [2]


def test_listwise_ndcg_perfect_ranking_is_1():
    score = [3.0, 2.0, 1.0]
    gain = [10.0, 5.0, 0.0]
    assert codon.listwise_ndcg(score, gain) == pytest.approx(1.0)


def test_listwise_ndcg_reversed_ranking_is_below_1():
    score = [1.0, 2.0, 3.0]  # model thinks 3rd is best but it is worst
    gain = [10.0, 5.0, 0.0]
    ndcg = codon.listwise_ndcg(score, gain)
    assert 0.0 <= ndcg < 1.0


def test_listwise_ndcg_handles_signed_gains():
    # min-max normalization must rank negative gains correctly
    score = [2.0, 1.0, 0.0]
    gain = [-5.0, 5.0, 0.0]  # best is +5 (idx1), worst is -5 (idx0)
    ndcg = codon.listwise_ndcg(score, gain)
    assert ndcg < 1.0  # model ranks idx0 first (bad) -> not perfect


def test_macro_listwise_ndcg_by_family_counts_singletons():
    recs = [{"protein": "MKFF"}, {"protein": "MKFF"},
            {"protein": "AAAA"}, {"protein": "AAAA"},
            {"protein": "SINGLE"}]
    score = [2.0, 1.0, 3.0, 2.5, 1.0]
    gain = [10.0, 0.0, 5.0, 2.0, 1.0]
    out = codon.macro_listwise_ndcg_by_family(recs, score, gain)
    assert out["n_families"] == 3
    assert out["n_rankable_families"] == 2
    assert out["n_singleton_families"] == 1
    assert 0.0 <= out["macro_ndcg"] <= 1.0


# ---------------------------------------------------------------------------
# region.py: 3'UTR adapter with independent endpoint heads
# ---------------------------------------------------------------------------

def test_region_adapter_has_independent_heads():
    cfg = region.RegionConfig(hidden=16)
    adapter = region.RegionAdapter(cfg)
    # 5' and 3' mean heads must be structurally separate tensors
    assert not torch.equal(adapter.mean_5u.weight, adapter.mean_3u.weight)
    assert adapter.mean_5u.weight.shape == adapter.mean_3u.weight.shape


def test_region_adapter_forward_routes_by_region():
    torch.manual_seed(0)
    cfg = region.RegionConfig(hidden=16)
    adapter = region.RegionAdapter(cfg)
    z = torch.randn(4, 16)
    region_ids = torch.tensor([1, 1, 0, 0])  # 2x 3UTR, 2x 5UTR
    ep3 = torch.tensor([0, 1, 0, 1])
    out = adapter(z, region_ids, ep3)
    assert out["mean"].shape == (4,)
    assert out["logvar"].shape == (4,)
    assert out["rank"].shape == (4,)
    # 3' rows (indices 0,1) get nonzero mean from mean_3u head
    assert torch.isfinite(out["mean"]).all()


def test_region_adapter_3u_endpoint_emb_only_on_3u():
    torch.manual_seed(0)
    cfg = region.RegionConfig(hidden=16, n_3u_endpoints=4)
    adapter = region.RegionAdapter(cfg)
    z = torch.randn(2, 16)
    region_ids = torch.tensor([1, 1])  # both 3UTR
    ep_a = torch.tensor([0, 0])
    ep_b = torch.tensor([1, 1])
    # different 3' endpoint ids must change the 3' route output
    out_a = adapter(z, region_ids, ep_a)
    out_b = adapter(z, region_ids, ep_b)
    assert not torch.allclose(out_a["mean"], out_b["mean"])


def test_independent_endpoint_head_guard():
    cfg = region.RegionConfig()
    assert region.independent_endpoint_head_guard(cfg) is True


def test_build_region_config():
    cfg = region.build_region_config(n_3u_endpoints=7, region="3UTR", hidden=32)
    assert cfg.n_3u_endpoints == 7
    assert cfg.region == "3UTR"
    assert cfg.hidden == 32
    assert cfg.n_3u_endpoints == 7
