"""P3-01 local-edit benchmark: unit + integration + determinism tests.

Covers (per phase contract):
    * unit tests      : schema / legality / neighborhood / split / tier builders
    * integration     : driver smoke build (--skip-proxy) end-to-end on server
                        data (skipped locally when data assets are absent)
    * CPU smoke       : proxy tier with a deterministic fake ensemble (numpy only)
    * deterministic   : identical inputs -> identical record_ids / split / hashes
    * resume          : driver re-run reproduces byte-identical tier JSONLs
                        (idempotent build == safe resume)

No torch dependency at import time; the proxy smoke test uses a fake ensemble
so the module never loads real checkpoints in tests.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import bootstrap: the repo directory IS the ``mrna_editflow`` package.
# On the server the repo dir is literally named ``mrna_editflow``; elsewhere we
# create a shim symlink so the absolute import path resolves identically.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_package_importable() -> None:
    parent = REPO_ROOT.parent
    if REPO_ROOT.name == "mrna_editflow":
        sys.path.insert(0, str(parent))
        return
    if (parent / "mrna_editflow").is_dir():
        sys.path.insert(0, str(parent))
        return
    shim = Path(tempfile.gettempdir()) / "p3_01_test_shim"
    shim.mkdir(exist_ok=True)
    link = shim / "mrna_editflow"
    if not link.exists():
        os.symlink(REPO_ROOT, link, target_is_directory=True)
    sys.path.insert(0, str(shim))


_ensure_package_importable()

from mrna_editflow.data.p3_legality import (  # noqa: E402
    creates_cryptic_splice,
    creates_homopolymer_ge6,
    creates_premature_stop_codon,
    creates_upstream_in_frame_start_codon,
    has_upstream_in_frame_start_codon,
    is_valid_cds,
    legal_cds_synonymous_single_subs,
    legal_five_utr_single_subs,
    motif_policy_v1_guarded_risk_flags,
    motif_policy_v1_hard_forbidden_triggered,
    normalize_rna,
    protein_identical,
    translate,
)
from mrna_editflow.data.p3_local_edit_schema import (  # noqa: E402
    ALL_FIELDS,
    REQUIRED_FIELDS,
    BenchmarkRecord,
    BenchmarkSchemaError,
    edits_from_alignment,
    hamming_distance,
    read_benchmark_jsonl,
    write_benchmark_jsonl,
)
from mrna_editflow.data.p3_neighborhood import (  # noqa: E402
    NeighborhoodConfig,
    apply_edits,
    assemble_doubles,
    cds_singles,
    five_utr_singles,
    joint_doubles,
    structure_disruption,
)
from mrna_editflow.data.p3_split import (  # noqa: E402
    SourceGroup,
    SplitConfig,
    assign_split_roles,
    audit_split,
    flag_ood_groups,
    resolve_cross_role_sequence_collisions,
    split_assignment_sha256,
)
from mrna_editflow.data.p3_benchmark_builder import (  # noqa: E402
    BuildConfig,
    build_measured_tier,
    build_unlabeled_cds_tier,
    filter_eligible_sources,
    select_proxy_sources,
)
import random  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

UTR50 = "ACGUGGCAUC" * 5  # 50 nt, valid RNA, no homopolymer, no splice consensus
CDS = "AUG" + "GCU" * 60 + "UAA"  # 62 codons + stop; poly-Ala, valid


def _make_record(**overrides) -> BenchmarkRecord:
    src = UTR50
    cand = src[:11] + "A" + src[12:]
    assert src[11] != "A"
    kwargs = dict(
        source_id="src:1", cargo_id="prot:x", cell_context="HEK293T",
        source_sequence=src, candidate_sequence=cand,
        edit_list=[{"region": "five_utr", "pos": 11, "ref": src[11], "alt": "A"}],
        edit_count=1, edited_region="five_utr", protein_identity=True,
        measured_or_proxy_source_value=1.0, measured_or_proxy_candidate_value=2.0,
        delta=1.0, data_source="test", assay_type="test_assay",
        confidence="proxy", edit_type="all_legal_single",
        task_eligibility="task_a_active", value_qualifier="predicted/internal proxy",
    )
    kwargs.update(overrides)
    return BenchmarkRecord(**kwargs).finalize()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_required_fields_cover_prompt_16(self):
        for f in ("record_id", "source_id", "cargo_id", "cell_context",
                  "source_sequence", "candidate_sequence", "edit_list",
                  "edit_count", "edited_region", "protein_identity",
                  "measured_or_proxy_source_value",
                  "measured_or_proxy_candidate_value", "delta",
                  "data_source", "assay_type", "confidence"):
            assert f in REQUIRED_FIELDS

    def test_valid_record_finalizes(self):
        rec = _make_record()
        assert rec.record_id.startswith("p3b:")
        d = rec.to_dict()
        assert set(ALL_FIELDS) <= set(d.keys())

    def test_record_id_deterministic(self):
        assert _make_record().record_id == _make_record().record_id

    def test_unlabeled_must_have_null_values(self):
        with pytest.raises(BenchmarkSchemaError, match="unlabeled"):
            _make_record(confidence="unlabeled")

    def test_unlabeled_with_values_ok(self):
        rec = _make_record(
            confidence="unlabeled",
            measured_or_proxy_source_value=None,
            measured_or_proxy_candidate_value=None, delta=None,
        )
        assert rec.delta is None

    def test_labeled_record_requires_values(self):
        with pytest.raises(BenchmarkSchemaError):
            _make_record(measured_or_proxy_candidate_value=None, delta=None)

    def test_delta_consistency(self):
        with pytest.raises(BenchmarkSchemaError, match="delta"):
            _make_record(delta=0.5)

    def test_edit_application_mismatch_rejected(self):
        with pytest.raises(BenchmarkSchemaError, match="reproduce"):
            _make_record(candidate_sequence=UTR50[:10] + "C" + UTR50[11:])

    def test_edit_ref_mismatch_rejected(self):
        src = UTR50
        with pytest.raises(BenchmarkSchemaError, match="ref mismatch"):
            _make_record(edit_list=[{"region": "five_utr", "pos": 10,
                                     "ref": "A" if src[10] != "A" else "C",
                                     "alt": "G" if src[10] != "G" else "C"}])

    def test_over_budget_rejected(self):
        src = UTR50
        edits, seq = [], list(src)
        alts = {"A": "C", "C": "G", "G": "U", "U": "A"}
        for i in range(11):
            edits.append({"region": "five_utr", "pos": i, "ref": seq[i],
                          "alt": alts[seq[i]]})
            seq[i] = alts[seq[i]]
        with pytest.raises(BenchmarkSchemaError, match="budget"):
            _make_record(edit_list=edits, edit_count=11,
                         candidate_sequence="".join(seq))

    def test_region_task_mismatch_rejected(self):
        with pytest.raises(BenchmarkSchemaError, match="task_eligibility"):
            _make_record(task_eligibility="task_b_frozen_fallback")

    def test_jsonl_roundtrip(self, tmp_path):
        rec = _make_record()
        p = tmp_path / "t.jsonl"
        assert write_benchmark_jsonl([rec], str(p)) == 1
        back = list(read_benchmark_jsonl(str(p)))
        assert len(back) == 1
        assert back[0].to_dict() == rec.to_dict()

    def test_edits_from_alignment_and_hamming(self):
        a, b = "ACGU", "ACGA"
        assert hamming_distance(a, b) == 1
        edits = edits_from_alignment(a, b, "five_utr")
        assert edits == [{"region": "five_utr", "pos": 3, "ref": "U", "alt": "A"}]
        assert hamming_distance("ACGU", "ACGUA") == -1


# ---------------------------------------------------------------------------
# Legality tests
# ---------------------------------------------------------------------------

class TestLegality:
    def test_translate_and_protein_identity(self):
        assert translate("AUGGCUUAA") == "MA*"
        assert protein_identical("AUGGCUUAA", "AUGGCCUAA")
        assert not protein_identical("AUGGCUUAA", "AUGAAAUAA")

    def test_is_valid_cds(self):
        assert is_valid_cds(CDS)
        assert not is_valid_cds("AUGGCU")          # no stop
        assert not is_valid_cds("GCGGCUUAA")       # bad start
        assert not is_valid_cds("AUGUAA")          # internal stop == immediate
        assert not is_valid_cds("AUGGCUUAAA")      # len % 3 != 0

    def test_upstream_in_frame_start_codon(self):
        # UTR of length 12: in-frame offsets are 0,3,6,9 (12%3==0).
        assert has_upstream_in_frame_start_codon("AUG" + "ACU" * 3)
        # AUG at offset 0 of a 13nt UTR is OUT of frame (offsets 1,4,7,10).
        assert not has_upstream_in_frame_start_codon("AUGCUACUACUAC")
        assert not has_upstream_in_frame_start_codon("AUG" + "UAA" + "ACU" * 2)  # stop intervenes
        assert creates_upstream_in_frame_start_codon("ACU" * 4, "AUG" + "ACU" * 3)

    def test_premature_stop_fail_safe(self):
        assert creates_premature_stop_codon("AUGGCUUAA", "AUGUAAUAA")
        assert not creates_premature_stop_codon("AUGGCUUAA", "AUGGCCUAA")

    def test_homopolymer_and_splice_detection(self):
        assert creates_homopolymer_ge6("ACGUGG", "ACAAAAAA")
        donorless = "ACUG" * 6
        donor = "ACUG" * 4 + "CAGGUAAGU" + "CU"  # canonical donor motif
        assert creates_cryptic_splice(donorless, donor)

    def test_hard_forbidden_triggered_region_scoped(self):
        t = motif_policy_v1_hard_forbidden_triggered(
            "five_utr", "ACU" * 4, "AUG" + "ACU" * 3)
        assert "creates_upstream_in_frame_start_codon" in t
        t2 = motif_policy_v1_hard_forbidden_triggered(
            "five_utr", "ACUG" * 6, "ACUG" * 6)
        assert t2 == []

    def test_guarded_risk_flags_tracked_not_legal(self):
        old = "ACGUGGCA" * 6
        new = old[:5] + "GATAC" + old[10:]  # introduces GATAC? need DRACH: [AGU][GA]AC[ACU]
        flags = motif_policy_v1_guarded_risk_flags(old, "GGACA" + old[5:])
        assert "m6a_motif_gain_or_loss" in flags

    def test_legal_five_utr_subs_excludes_hard_forbidden(self):
        seq = "AAAAAACGU" + "ACUG" * 10  # existing homopolymer -> edits cannot CREATE new
        subs = legal_five_utr_single_subs(UTR50)
        assert subs, "expected some legal subs"
        for pos, ref, alt in subs:
            assert UTR50[pos] == ref and alt != ref
            cand = UTR50[:pos] + alt + UTR50[pos + 1:]
            assert not creates_upstream_in_frame_start_codon(UTR50, cand)
            assert not creates_cryptic_splice(UTR50, cand)
            assert not creates_homopolymer_ge6(UTR50, cand)

    def test_legal_cds_subs_synonymous_and_start_stop_untouched(self):
        subs = legal_cds_synonymous_single_subs(CDS, (0, 62))
        assert subs
        for nt_pos, ref, alt, ci, new_codon in subs:
            assert 1 <= ci < 61  # start (0) and stop (61) excluded
            cand = CDS[:nt_pos] + alt + CDS[nt_pos + 1:]
            assert protein_identical(CDS, cand)


# ---------------------------------------------------------------------------
# Neighborhood tests
# ---------------------------------------------------------------------------

class TestNeighborhood:
    def test_five_utr_singles_positions(self):
        singles = five_utr_singles(UTR50, (0, 10))
        assert singles and all(0 <= e.pos < 10 for e in singles)
        assert all(e.region == "five_utr" for e in singles)

    def test_cds_singles_scope_coords(self):
        edits, scope = cds_singles(CDS, "cds_first30", (0, 30))
        assert scope == CDS[3:90]  # codons 1..29
        assert edits and all(0 <= e.pos < len(scope) for e in edits)
        for e in edits:
            cand_scope = apply_edits(scope, [e])
            off = 3  # scope starts at codon 1
            full = CDS[:off + e.pos] + e.alt + CDS[off + e.pos + 1:]
            assert protein_identical(CDS, full)

    def test_assemble_doubles_counts_and_distinct_positions(self):
        singles = five_utr_singles(UTR50)
        scores = {e: float(e.pos) for e in singles}
        cfg = NeighborhoodConfig()
        gen = assemble_doubles(singles, scores, UTR50, cfg, random.Random(1))
        assert len(gen["random_double"]) <= cfg.n_random_double
        assert len(gen["structure_guided_double"]) <= cfg.n_structure_double
        assert len(gen["topranked_double"]) <= cfg.n_topranked_double
        assert len(gen["matched_negative_single"]) == cfg.n_matched_negative
        for pair in gen["random_double"] + gen["topranked_double"]:
            assert len(pair) == 2 and pair[0].pos != pair[1].pos
        # matched negatives = lowest-score singles (deterministic tie-break)
        expected = sorted(singles, key=lambda e: (scores[e], e.pos, e.alt))[
            : cfg.n_matched_negative]
        assert gen["matched_negative_single"] == [(e,) for e in expected]

    def test_assemble_doubles_deterministic(self):
        singles = five_utr_singles(UTR50)
        scores = {e: float(e.pos % 7) for e in singles}
        cfg = NeighborhoodConfig()
        g1 = assemble_doubles(singles, scores, UTR50, cfg, random.Random(42))
        g2 = assemble_doubles(singles, scores, UTR50, cfg, random.Random(42))
        assert [[e.to_dict() for e in grp] for grp in g1["random_double"]] == \
               [[e.to_dict() for e in grp] for grp in g2["random_double"]]

    def test_assemble_doubles_cds_excludes_same_codon_pairs(self):
        """Regression: two synonymous singles in ONE codon can combine into a
        nonsynonymous codon (CUG->UUG + CUG->CUU give CUG->UUU = Phe)."""
        edits, scope = cds_singles(CDS, "cds_first50", (0, 50))
        scores = {e: float(e.pos) for e in edits}
        cfg = NeighborhoodConfig()
        gen = assemble_doubles(edits, scores, scope, cfg, random.Random(7))
        for etype in ("random_double", "structure_guided_double", "topranked_double"):
            assert gen[etype], f"{etype} unexpectedly empty"
            for a, b in gen[etype]:
                assert a.pos != b.pos
                assert a.pos // 3 != b.pos // 3, f"same-codon pair in {etype}"
        # paired edits applied together must preserve the protein (full CDS)
        for a, b in gen["random_double"]:
            full = list(CDS)
            for e in (a, b):
                pos = 3 + e.pos  # scope starts at codon 1
                assert full[pos] == e.ref
                full[pos] = e.alt
            assert protein_identical(CDS, "".join(full))

    def test_joint_doubles_shift(self):
        utr = "ACUG" * 10  # 40 nt
        utr_s = five_utr_singles(utr)
        cds_s, scope = cds_singles(CDS, "cds_first50", (0, 50))
        cfg = NeighborhoodConfig()
        gen = joint_doubles(utr_s, cds_s, utr, scope, cfg, random.Random(7))
        assert gen["random_double"]
        for u, c in gen["random_double"]:
            assert 0 <= u.pos < len(utr)
            assert len(utr) <= c.pos < len(utr) + len(scope)
        # apply via shifted joint scope reproduces both edits
        joint_scope = utr + scope
        u, c = gen["random_double"][0]
        cand = apply_edits(joint_scope, [u, c])
        assert cand[u.pos] == u.alt
        assert cand[c.pos] == c.alt

    def test_structure_disruption_deterministic(self):
        e = five_utr_singles(UTR50)[0]
        assert structure_disruption(UTR50, e) == structure_disruption(UTR50, e)
        assert structure_disruption(UTR50, e) >= 0.0


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplit:
    def _groups(self, n=20):
        return [
            SourceGroup(group_id=f"g{i}", family_id=f"f{i // 4}", n_records=10,
                        gc=0.3 + 0.02 * i, length=50 + 10 * i,
                        cohort="c" + str(i % 2))
            for i in range(n)
        ]

    def test_assign_roles_deterministic_and_disjoint(self):
        g = self._groups()
        r1, m1 = assign_split_roles(g, SplitConfig(seed=1))
        r2, m2 = assign_split_roles(g, SplitConfig(seed=1))
        assert r1 == r2
        assert split_assignment_sha256(r1) == split_assignment_sha256(r2)
        assert set(r1.values()) <= {"train", "val", "test", "ood"}
        assert set(r1.keys()) == {x.group_id for x in g}

    def test_per_cohort_ood_not_global_length_bias(self):
        # cohort A all length 50 (like MPRA mothers); cohort B varied.
        groups = [SourceGroup(f"a{i}", "fa", 5, 0.5, 50, "A") for i in range(10)] + \
                 [SourceGroup(f"b{i}", "fb", 5, 0.5, 100 + i, "B") for i in range(10)]
        flagged = flag_ood_groups(groups, SplitConfig())
        a_flags = [gid for gid in flagged if gid.startswith("a")]
        assert not any("length_shift" in flagged[gid] for gid in a_flags)

    def test_ood_boundary_tie_mass_not_wholesale_flagged(self):
        """Regression: 60% tie-mass at the modal length (128nt UTR spike) must
        NOT be swallowed by an inclusive boundary comparison."""
        groups = [SourceGroup(f"m{i}", f"f{i}", 5, 0.5, 128, "R") for i in range(60)] + \
                 [SourceGroup(f"v{i}", f"f{60 + i}", 5, 0.5, 60 + i, "R") for i in range(40)]
        flagged = flag_ood_groups(groups, SplitConfig())
        length_flagged = [g for g, rs in flagged.items() if "length_shift" in rs]
        assert all(g.startswith("v") for g in length_flagged)
        assert len(length_flagged) <= 8  # ~5% low tail of 100, not the modal 60

    def test_test_role_family_disjoint_from_ood(self):
        """Regression: a family with an OOD member must never contribute test
        groups (test is cargo/family-disjoint from ALL roles)."""
        normals = [
            SourceGroup(f"g{i}", f"f{i // 4}", 10, 0.40 + 0.01 * i, 100 + i, "R")
            for i in range(20)
        ]
        hot = SourceGroup("hot", "f0", 5, 0.95, 500, "R")  # extreme -> OOD
        groups = normals + [hot]
        roles, _ = assign_split_roles(groups, SplitConfig(seed=5))
        assert roles["hot"] == "ood"
        fam_roles: Dict[str, set] = {}
        for g in groups:
            fam_roles.setdefault(g.family_id, set()).add(roles[g.group_id])
        for fam, rs in fam_roles.items():
            assert not ("test" in rs and "ood" in rs), fam

    def test_audit_passes_clean_assignment(self):
        groups = self._groups(12)
        roles, _ = assign_split_roles(groups, SplitConfig(seed=3))
        records = []
        for g in groups:
            for k in range(g.n_records):
                records.append({
                    "source_id": g.group_id, "family_cluster_id": g.family_id,
                    "candidate_sequence": f"{g.group_id}:cand{k}",
                    "split_role": roles[g.group_id],
                })
        audit = audit_split(records, "unit")
        assert audit["passed"], audit
        assert audit["n_source_role_violations"] == 0

    def test_audit_catches_source_leak(self):
        records = [
            {"source_id": "s1", "family_cluster_id": "f1",
             "candidate_sequence": "AAAA", "split_role": "train"},
            {"source_id": "s1", "family_cluster_id": "f1",
             "candidate_sequence": "AAAC", "split_role": "test"},
        ]
        audit = audit_split(records, "unit")
        assert not audit["passed"]
        assert audit["n_source_role_violations"] == 1

    def test_audit_catches_test_family_leak(self):
        records = [
            {"source_id": "s1", "family_cluster_id": "fX",
             "candidate_sequence": "AAAA", "split_role": "test"},
            {"source_id": "s2", "family_cluster_id": "fX",
             "candidate_sequence": "AAAC", "split_role": "train"},
        ]
        audit = audit_split(records, "unit")
        assert not audit["passed"]
        assert audit["n_test_family_violations"] == 1


class _Rec:
    """Minimal record stub for collision resolution (attrs, not dict)."""

    def __init__(self, record_id: str, candidate_sequence: str, split_role: str):
        self.record_id = record_id
        self.candidate_sequence = candidate_sequence
        self.split_role = split_role


class TestCollisionResolution:
    def test_drops_lower_priority_role_keeps_test(self):
        # same candidate sequence emitted by two different sources: one train,
        # one test -> the train copy must be dropped.
        recs = {
            "proxy": [
                _Rec("r_train", "SEQ_X", "train"),
                _Rec("r_test", "SEQ_X", "test"),
                _Rec("r_uniq", "SEQ_Y", "train"),
            ]
        }
        dropped, stats = resolve_cross_role_sequence_collisions(recs)
        assert dropped == {"r_train"}
        assert stats["n_collision_sequences"] == 1
        assert stats["n_dropped_records"] == 1
        assert stats["drops_by_role"] == {"train": 1, "val": 0, "ood": 0, "test": 0}

    def test_same_role_duplicates_untouched(self):
        recs = {"proxy": [_Rec("a", "SEQ", "train"), _Rec("b", "SEQ", "train")]}
        dropped, stats = resolve_cross_role_sequence_collisions(recs)
        assert dropped == set()
        assert stats["n_collision_sequences"] == 0

    def test_ood_beats_val_and_train(self):
        recs = {"unlabeled": [
            _Rec("v", "SEQ", "val"), _Rec("t", "SEQ", "train"), _Rec("o", "SEQ", "ood"),
        ]}
        dropped, stats = resolve_cross_role_sequence_collisions(recs)
        assert dropped == {"v", "t"}
        assert stats["drops_by_role"]["val"] == 1
        assert stats["drops_by_role"]["train"] == 1
        assert stats["drops_by_role"]["ood"] == 0

    def test_deterministic_and_audit_passes_after_drop(self):
        recs = {"proxy": [
            _Rec("r1", "X", "train"), _Rec("r2", "X", "test"),
            _Rec("r3", "Y", "val"), _Rec("r4", "Y", "ood"),
            _Rec("r5", "Z", "train"),
        ]}
        d1, s1 = resolve_cross_role_sequence_collisions(recs)
        d2, s2 = resolve_cross_role_sequence_collisions(recs)
        assert d1 == d2 and s1 == s2
        kept = [r for r in recs["proxy"] if r.record_id not in d1]
        audit = audit_split(
            [{"source_id": f"src:{r.record_id}", "family_cluster_id": None,
              "candidate_sequence": r.candidate_sequence,
              "split_role": r.split_role} for r in kept], "unit")
        assert audit["passed"], audit
        assert audit["cross_role_exact_sequence_collisions"] == 0


# ---------------------------------------------------------------------------
# Measured tier (pure-python builder on synthetic snv rows)
# ---------------------------------------------------------------------------

def _snv_row(utr: str, mother: str, rl: float, vid: str = "v") -> dict:
    return {"utr": utr, "mother": mother, "rl": str(rl), "id": vid,
            "info1": "", "info2": "", "info3": "", "info4": "", "library": "snv"}


class TestMeasuredTier:
    MOTHER = "ACGUGGCAUC" * 5

    def test_anchor_and_delta(self):
        var = self.MOTHER[:5] + "A" + self.MOTHER[6:]
        assert self.MOTHER[5] != "A"
        rows = [
            _snv_row(self.MOTHER, self.MOTHER, 2.0, "wt1"),
            _snv_row(self.MOTHER, self.MOTHER, 4.0, "wt2"),  # upper-median WT -> 4.0
            _snv_row(var, self.MOTHER, 5.5, "v1"),
        ]
        records, stats = build_measured_tier(rows)
        anchors = [r for r in records if r.edit_count == 0]
        variants = [r for r in records if r.edit_count == 1]
        assert len(anchors) == 1 and len(variants) == 1
        assert anchors[0].measured_or_proxy_source_value == 4.0
        assert variants[0].delta == pytest.approx(1.5)
        assert variants[0].confidence == "measured"
        assert variants[0].task_eligibility == "task_a_active"
        assert variants[0].edit_type == "measured_single"

    def test_unanchored_variants_excluded(self):
        var = self.MOTHER[:5] + "A" + self.MOTHER[6:]
        rows = [_snv_row(var, self.MOTHER, 5.0)]  # no WT row for this mother
        records, stats = build_measured_tier(rows)
        assert records == []
        assert stats["n_excluded_unanchored_variants"] == 1

    def test_over_budget_excluded(self):
        alts = {"A": "C", "C": "G", "G": "U", "U": "A"}
        var = "".join(alts[c] for c in self.MOTHER[:11]) + self.MOTHER[11:]
        rows = [_snv_row(self.MOTHER, self.MOTHER, 2.0),
                _snv_row(var, self.MOTHER, 3.0)]
        records, stats = build_measured_tier(rows)
        assert [r.edit_count for r in records] == [0]
        assert stats["n_excluded_over_budget_variants"] == 1

    def test_deterministic_record_ids(self):
        var = self.MOTHER[:5] + "A" + self.MOTHER[6:]
        rows = [_snv_row(self.MOTHER, self.MOTHER, 2.0), _snv_row(var, self.MOTHER, 5.0)]
        r1, _ = build_measured_tier(rows)
        r2, _ = build_measured_tier(rows)
        assert [r.record_id for r in r1] == [r.record_id for r in r2]


# ---------------------------------------------------------------------------
# Unlabeled CDS tier (pure python)
# ---------------------------------------------------------------------------

class TestUnlabeledCdsTier:
    def _source(self, tid="tx1"):
        return {"transcript_id": tid, "five_utr": UTR50, "cds": CDS,
                "three_utr": "", "family_cluster_id": "fam1"}

    def test_null_values_and_protein_identity(self):
        cfg = BuildConfig(smoke=True).effective()
        records, stats = build_unlabeled_cds_tier([self._source()], cfg)
        assert records and stats["protein_identity_failures"] == 0
        regions = {r.edited_region for r in records}
        assert "cds_first30" in regions and "cds_first50" in regions
        assert "joint_5utr_cds" in regions
        for r in records:
            assert r.confidence == "unlabeled"
            assert r.delta is None and r.measured_or_proxy_candidate_value is None
            assert r.protein_identity
            assert r.edit_count <= 10

    def test_cds_tier_deterministic(self):
        cfg = BuildConfig(smoke=True).effective()
        r1, _ = build_unlabeled_cds_tier([self._source()], cfg)
        r2, _ = build_unlabeled_cds_tier([self._source()], cfg)
        assert [r.record_id for r in r1] == [r.record_id for r in r2]


# ---------------------------------------------------------------------------
# Proxy tier CPU smoke (fake ensemble; numpy only, no torch)
# ---------------------------------------------------------------------------

class _FakeCNN:
    """Deterministic stand-in for CNN50merPredictor (same .predict API)."""

    def predict(self, seqs):
        np = pytest.importorskip("numpy")
        return np.array([(sum(ord(c) for c in s) % 997) / 100.0 for s in seqs])


class TestProxyTierSmoke:
    def _source(self, tid="tx1", utr=UTR50):
        return {"transcript_id": tid, "five_utr": utr, "cds": CDS,
                "three_utr": "", "family_cluster_id": "fam1"}

    def test_proxy_records_carry_proxy_qualifier_and_std(self):
        np = pytest.importorskip("numpy")
        from mrna_editflow.data.p3_benchmark_builder import build_proxy_tier
        cfg = BuildConfig(smoke=True).effective()
        models = [_FakeCNN(), _FakeCNN()]
        proxy, unlab, stats = build_proxy_tier([self._source()], models, cfg)
        assert proxy and stats["n_scored_sequences"] > 0
        for r in proxy:
            assert r.confidence == "proxy"
            assert "predicted/internal proxy" in r.value_qualifier
            assert r.delta is not None
            assert r.value_std is not None
            assert r.edit_count <= 2  # singles + doubles only
        # 50nt source: no out-of-window unlabeled byproduct
        assert unlab == []

    def test_out_of_window_edits_never_get_proxy_values(self):
        np = pytest.importorskip("numpy")
        from mrna_editflow.data.p3_benchmark_builder import build_proxy_tier
        long_utr = UTR50 + "ACGUGGCAUCGG" * 5  # 110 nt
        cfg = BuildConfig(smoke=True).effective()
        proxy, unlab, _ = build_proxy_tier(
            [self._source(utr=long_utr)], [_FakeCNN()], cfg)
        assert unlab, "expected out-of-window unlabeled records"
        for r in unlab:
            assert r.confidence == "unlabeled"
            assert r.delta is None
            assert all(e["pos"] >= 50 for e in r.edit_list)
        for r in proxy:
            assert all(e["pos"] < 50 for e in r.edit_list)

    def test_proxy_delta_is_ensemble_mean_difference(self):
        np = pytest.importorskip("numpy")
        from mrna_editflow.data.p3_benchmark_builder import build_proxy_tier
        cfg = BuildConfig(smoke=True).effective()
        proxy, _, _ = build_proxy_tier([self._source()], [_FakeCNN()], cfg)
        fake = _FakeCNN()
        for r in proxy[:5]:
            src_v = float(fake.predict([r.source_sequence[:50]])[0])
            cand_v = float(fake.predict([r.candidate_sequence[:50]])[0])
            assert r.measured_or_proxy_source_value == pytest.approx(src_v)
            assert r.delta == pytest.approx(cand_v - src_v)

    def test_proxy_tier_deterministic(self):
        np = pytest.importorskip("numpy")
        from mrna_editflow.data.p3_benchmark_builder import build_proxy_tier
        cfg = BuildConfig(smoke=True).effective()
        p1, u1, _ = build_proxy_tier([self._source()], [_FakeCNN()], cfg)
        p2, u2, _ = build_proxy_tier([self._source()], [_FakeCNN()], cfg)
        assert [r.record_id for r in p1] == [r.record_id for r in p2]


# ---------------------------------------------------------------------------
# Eligibility + selection
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_filter_fail_closed(self):
        good = {"transcript_id": "g", "five_utr": UTR50, "cds": CDS,
                "three_utr": "", "family_cluster_id": "f"}
        bad_utr = {**good, "transcript_id": "b1", "five_utr": "ACGU"}
        bad_cds = {**good, "transcript_id": "b2", "cds": "AUGGCU"}
        up_aug = {**good, "transcript_id": "b3",
                  "five_utr": "A" + "AUG" + "ACU" * 16}  # in-frame AUG at offset 1
        eligible, reasons = filter_eligible_sources(
            [good, bad_utr, bad_cds, up_aug])
        assert [s["transcript_id"] for s in eligible] == ["g"]
        assert reasons["five_utr_shorter_than_50nt"] == 1
        assert reasons["invalid_cds"] == 1
        assert reasons["source_has_upstream_in_frame_start_codon"] == 1

    def test_select_proxy_sources_stratified_deterministic(self):
        pool = []
        for i in range(40):
            utr = UTR50 if i < 10 else UTR50 + "ACGUGGCA" * 10  # 10 short, 30 long
            pool.append({"transcript_id": f"t{i}", "five_utr": utr, "cds": CDS,
                         "three_utr": "", "family_cluster_id": "f"})
        cfg = BuildConfig(seed=9, proxy_sources=20, short_utr_fraction=0.25)
        s1 = select_proxy_sources(pool, cfg)
        s2 = select_proxy_sources(pool, cfg)
        assert [s["transcript_id"] for s in s1] == [s["transcript_id"] for s in s2]
        n_short = sum(1 for s in s1 if len(s["five_utr"]) < 128)
        assert n_short == 5  # 25% of 20


# ---------------------------------------------------------------------------
# Driver integration + resume (server data only; skips locally)
# ---------------------------------------------------------------------------

REPO_DATA_MARKERS = [
    "data/raw/sample2019_mpra/GSM3130443_designed_library.csv.gz",
    "data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl",
]
HAS_REPO_DATA = all((REPO_ROOT / m).exists() for m in REPO_DATA_MARKERS)


@pytest.mark.skipif(not HAS_REPO_DATA, reason="repo data assets not present locally")
class TestDriverIntegration:
    def _run_driver(self, out_dir: Path) -> int:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import p3_01_build_benchmark as driver
            return driver.main([
                "--repo-root", str(REPO_ROOT),
                "--out-dir", str(out_dir),
                "--audit-out", str(out_dir / "p3_01_split_audit.json"),
                "--smoke", "--skip-proxy",
            ])
        finally:
            sys.path.remove(str(REPO_ROOT / "scripts"))
            sys.modules.pop("p3_01_build_benchmark", None)

    def test_smoke_build_end_to_end(self, tmp_path):
        out = tmp_path / "p3"
        rc = self._run_driver(out)
        assert rc == 0
        for tier in ("measured", "proxy", "unlabeled"):
            assert (out / "benchmark" / f"{tier}_tier.jsonl").exists()
        manifest = json.loads((out / "manifests" / "p3_01_benchmark_manifest.json").read_text())
        assert manifest["split"]["audit_passed"]
        audit = json.loads((out / "p3_01_split_audit.json").read_text())
        assert audit["passed"]
        # all roles populated (regression: boundary tie-mass once sent ~90% of
        # records to OOD, leaving train empty)
        assert audit["split_metadata"]["role_record_counts"]["train"] > 0
        # every record validates against the frozen schema
        for rec in read_benchmark_jsonl(str(out / "benchmark" / "unlabeled_tier.jsonl")):
            rec.validate()

    def test_resume_reproduces_identical_hashes(self, tmp_path):
        out1, out2 = tmp_path / "p3a", tmp_path / "p3b"
        assert self._run_driver(out1) == 0
        assert self._run_driver(out2) == 0
        m1 = json.loads((out1 / "manifests" / "p3_01_benchmark_manifest.json").read_text())
        m2 = json.loads((out2 / "manifests" / "p3_01_benchmark_manifest.json").read_text())
        assert m1["tiers"]["measured"]["sha256"] == m2["tiers"]["measured"]["sha256"]
        assert m1["tiers"]["unlabeled"]["sha256"] == m2["tiers"]["unlabeled"]["sha256"]
        assert m1["split"]["assignment_sha256"] == m2["split"]["assignment_sha256"]
