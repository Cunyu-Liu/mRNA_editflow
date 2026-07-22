"""Unit tests for P2-09 OOD Robustness Stress Test."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Ensure the script is importable. The script lives at
# scripts/run_p2_09_ood_stress_test.py (repo root is parent of tests/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_p2_09_ood_stress_test as mod  # noqa: E402


# ---------------------------------------------------------------------------
# GC content / gene ID parsing
# ---------------------------------------------------------------------------


class TestGcContent:
    def test_empty(self):
        assert mod._gc_content("") == 0.0

    def test_all_gc(self):
        assert mod._gc_content("GCGCGC") == 1.0

    def test_no_gc(self):
        assert mod._gc_content("AATAAT") == 0.0

    def test_half(self):
        assert mod._gc_content("GCAATT") == pytest.approx(1.0 / 3.0)

    def test_case_insensitive(self):
        assert mod._gc_content("gc") == 1.0


class TestGeneIdFromTranscript:
    def test_ensembl_human(self):
        assert mod._gene_id_from_transcript("gencode_v45:ENST00000328596.10") == "ENSG00000328596"

    def test_ensembl_mouse(self):
        assert mod._gene_id_from_transcript("ENSMUST00000000001.5") == "ENSMUSG00000000001"

    def test_ensembl_zebrafish(self):
        assert mod._gene_id_from_transcript("ENSDART00000000001.5") == "ENSDARG00000000001"

    def test_refseq(self):
        # refseq falls back to stripping version.
        assert mod._gene_id_from_transcript("refseq:NM_001005484.1") == "NM_001005484"

    def test_empty(self):
        assert mod._gene_id_from_transcript("") == ""

    def test_no_version(self):
        assert mod._gene_id_from_transcript("ENST00000328596") == "ENSG00000328596"


# ---------------------------------------------------------------------------
# Source metadata extraction
# ---------------------------------------------------------------------------


class TestExtractSourceMetadata:
    def test_basic(self):
        sources = [
            {
                "five_utr": "GCC",
                "cds": "GCCAUG",
                "three_utr": "GGG",
                "transcript_id": "gencode_v45:ENST000001.1",
                "species": "human",
            },
            {
                "five_utr": "",
                "cds": "AUG",
                "three_utr": "",
                "transcript_id": "refseq:NM_001.2",
                "species": "mouse",
            },
        ]
        meta = mod.extract_source_metadata(sources)
        assert len(meta) == 2
        assert meta[0].index == 0
        assert meta[0].total_length == 12
        assert meta[0].five_utr_length == 3
        assert meta[0].cds_length == 6
        assert meta[0].three_utr_length == 3
        assert meta[0].gc_total == pytest.approx(10.0 / 12.0)
        assert meta[0].gene_id == "ENSG000001"
        assert meta[0].species == "human"
        assert meta[1].total_length == 3
        assert meta[1].gc_total == pytest.approx(1.0 / 3.0)  # "AUG" has 1 G

    def test_missing_fields(self):
        sources = [{}]
        meta = mod.extract_source_metadata(sources)
        assert meta[0].total_length == 0
        assert meta[0].gene_id == ""

    def test_cluster_assignments_lookup(self):
        """When cluster_assignments + test_idx provided, cluster_id is populated."""
        sources = [
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST1.1", "species": "human"},
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST2.1", "species": "human"},
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST3.1", "species": "human"},
        ]
        # Full dataset has 10 records; test split = [2, 5, 7].
        cluster_assignments = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        test_idx = [2, 5, 7]
        meta = mod.extract_source_metadata(
            sources,
            cluster_assignments=cluster_assignments,
            test_idx=test_idx,
        )
        assert meta[0].cluster_id == 1  # cluster_assignments[2]
        assert meta[1].cluster_id == 2  # cluster_assignments[5]
        assert meta[2].cluster_id == 2  # cluster_assignments[7]

    def test_cluster_assignments_optional(self):
        """Without cluster_assignments, cluster_id remains -1."""
        sources = [
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST1.1", "species": "human"},
        ]
        meta = mod.extract_source_metadata(sources)
        assert meta[0].cluster_id == -1

    def test_cluster_assignments_short_test_idx(self):
        """When test_idx is shorter than sources, fall back to -1."""
        sources = [
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST1.1", "species": "human"},
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST2.1", "species": "human"},
        ]
        # test_idx too short -> use_clusters = False.
        meta = mod.extract_source_metadata(
            sources,
            cluster_assignments=[0, 1, 2],
            test_idx=[0],  # only 1, but sources has 2.
        )
        assert all(m.cluster_id == -1 for m in meta)


# ---------------------------------------------------------------------------
# OOD subset computation
# ---------------------------------------------------------------------------


class TestComputeOodSubsets:
    def _make_meta(self, n: int = 100) -> List[mod.SourceMetadata]:
        rng = np.random.default_rng(0)
        return [
            mod.SourceMetadata(
                index=i,
                total_length=int(rng.integers(100, 5000)),
                five_utr_length=int(rng.integers(0, 200)),
                cds_length=int(rng.integers(100, 3000)),
                three_utr_length=int(rng.integers(50, 1000)),
                gc_total=float(rng.uniform(0.3, 0.7)),
                gene_id=f"ENSG{ i // 3:06d}",  # ~3 per family
                species="human",
            )
            for i in range(n)
        ]

    def test_returns_8_subsets(self):
        meta = self._make_meta()
        subsets = mod.compute_ood_subsets(meta)
        assert len(subsets) == 8
        expected = {
            "length_total_p10",
            "length_total_p90",
            "length_5utr_p90",
            "length_3utr_p90",
            "gc_total_p10",
            "gc_total_p90",
            "family_rare",
            "cds_long_p90",
        }
        assert set(subsets.keys()) == expected

    def test_p10_is_bottom(self):
        meta = self._make_meta(100)
        subsets = mod.compute_ood_subsets(meta)
        p10_idx = set(subsets["length_total_p10"]["indices"])
        # The minimum should be in p10.
        lens = [m.total_length for m in meta]
        min_idx = lens.index(min(lens))
        assert min_idx in p10_idx

    def test_p90_is_top(self):
        meta = self._make_meta(100)
        subsets = mod.compute_ood_subsets(meta)
        p90_idx = set(subsets["length_total_p90"]["indices"])
        lens = [m.total_length for m in meta]
        max_idx = lens.index(max(lens))
        assert max_idx in p90_idx

    def test_p10_and_p90_disjoint(self):
        meta = self._make_meta(100)
        subsets = mod.compute_ood_subsets(meta)
        p10 = set(subsets["length_total_p10"]["indices"])
        p90 = set(subsets["length_total_p90"]["indices"])
        assert p10.isdisjoint(p90)

    def test_family_rare_finds_small_families(self):
        # Build metadata with 3 gene families of size 1, 1, 5.
        meta = []
        for i in range(7):
            meta.append(
                mod.SourceMetadata(
                    index=i,
                    total_length=100,
                    five_utr_length=10,
                    cds_length=50,
                    three_utr_length=40,
                    gc_total=0.5,
                    gene_id=f"G{ i // 5 }",  # i=0..4 -> G0, i=5,6 -> G1, G2
                    species="human",
                )
            )
        # Actually with i//5: i=0..4 -> G0 (size 5), i=5,6 -> G1 (size 2).
        # Family G1 has 2 members -> <=2 -> rare.
        # Family G0 has 5 members -> not rare.
        subsets = mod.compute_ood_subsets(meta)
        rare = set(subsets["family_rare"]["indices"])
        # G1 members are indices 5, 6.
        assert 5 in rare
        assert 6 in rare
        # G0 members are 0..4.
        for i in range(5):
            assert i not in rare

    def test_subset_size_field(self):
        meta = self._make_meta(100)
        subsets = mod.compute_ood_subsets(meta)
        for name, s in subsets.items():
            assert s["size"] == len(s["indices"])

    def test_family_rare_uses_cluster_id_when_available(self):
        """When cluster_id is populated, family_rare uses cluster counts."""
        meta = []
        # Cluster 0: 5 members (not rare).
        for i in range(5):
            meta.append(mod.SourceMetadata(
                index=i, total_length=100, five_utr_length=10,
                cds_length=50, three_utr_length=40, gc_total=0.5,
                gene_id=f"G{i}",  # all distinct gene_ids.
                species="human", cluster_id=0,
            ))
        # Cluster 1: 2 members (rare, <=2).
        for i in range(5, 7):
            meta.append(mod.SourceMetadata(
                index=i, total_length=100, five_utr_length=10,
                cds_length=50, three_utr_length=40, gc_total=0.5,
                gene_id=f"G{i}",
                species="human", cluster_id=1,
            ))
        # Cluster 2: 1 member (rare).
        meta.append(mod.SourceMetadata(
            index=7, total_length=100, five_utr_length=10,
            cds_length=50, three_utr_length=40, gc_total=0.5,
            gene_id="G7",
            species="human", cluster_id=2,
        ))
        subsets = mod.compute_ood_subsets(meta)
        rare = set(subsets["family_rare"]["indices"])
        # Cluster 1 members (5, 6) and cluster 2 member (7) are rare.
        assert 5 in rare
        assert 6 in rare
        assert 7 in rare
        # Cluster 0 members (0..4) are not rare.
        for i in range(5):
            assert i not in rare
        # Definition should mention "split contract".
        assert "split contract" in subsets["family_rare"]["definition"]

    def test_family_rare_falls_back_to_gene_id(self):
        """When cluster_id is -1 for all, family_rare falls back to gene_id."""
        meta = []
        for i in range(7):
            meta.append(mod.SourceMetadata(
                index=i, total_length=100, five_utr_length=10,
                cds_length=50, three_utr_length=40, gc_total=0.5,
                # i=0..4 -> G0 (size 5), i=5,6 -> G1 (size 2, rare).
                gene_id=f"G{ i // 5 }",
                species="human", cluster_id=-1,
            ))
        subsets = mod.compute_ood_subsets(meta)
        rare = set(subsets["family_rare"]["indices"])
        assert 5 in rare
        assert 6 in rare
        for i in range(5):
            assert i not in rare
        # Definition should mention "approximated" (gene_id approximation).
        assert "approximat" in subsets["family_rare"]["definition"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_identical(self):
        assert mod.cohens_d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_positive(self):
        # a has higher mean than b.
        d = mod.cohens_d([5, 6, 7], [1, 2, 3])
        assert d > 0

    def test_negative(self):
        d = mod.cohens_d([1, 2, 3], [5, 6, 7])
        assert d < 0

    def test_too_short(self):
        assert mod.cohens_d([1], [2]) == 0.0


class TestPairedPermutationTest:
    def test_identical(self):
        # Identical samples -> diff 0, p_value ~ 1.
        r = mod.paired_permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n_perm=200)
        assert r["diff"] == pytest.approx(0.0)
        assert r["p_value"] > 0.5

    def test_large_difference(self):
        # Large positive difference -> significant.
        r = mod.paired_permutation_test([10, 20, 30, 40, 50], [1, 2, 3, 4, 5], n_perm=500)
        assert r["diff"] > 0
        assert r["p_value"] < 0.1  # should be small

    def test_too_short(self):
        r = mod.paired_permutation_test([1], [2], n_perm=100)
        assert r["p_value"] == 1.0

    def test_n_perm_recorded(self):
        r = mod.paired_permutation_test([1, 2, 3], [2, 3, 4], n_perm=42)
        assert r["n_perm"] == 42


class TestBootstrapCi:
    def test_mean(self):
        m, lo, hi = mod.bootstrap_ci([1, 2, 3, 4, 5], n_boot=100)
        assert m == pytest.approx(3.0)

    def test_ci_contains_mean(self):
        m, lo, hi = mod.bootstrap_ci([1, 2, 3, 4, 5], n_boot=500)
        assert lo <= m <= hi

    def test_empty(self):
        m, lo, hi = mod.bootstrap_ci([], n_boot=10)
        assert m == 0.0 and lo == 0.0 and hi == 0.0

    def test_single(self):
        m, lo, hi = mod.bootstrap_ci([42.0], n_boot=10)
        assert m == 42.0 and lo == 42.0 and hi == 42.0


# ---------------------------------------------------------------------------
# Per-record delta computation
# ---------------------------------------------------------------------------


class TestComputeDeltaPerRecord:
    def test_happy_path(self):
        prm = {
            "oracle_ensemble_te": [1.0, 2.0, 3.0],
            "source_oracle_ensemble_te": [0.5, 1.0, 1.5],
        }
        delta = mod.compute_delta_per_record(prm)
        assert delta is not None
        assert list(delta) == [0.5, 1.0, 1.5]

    def test_missing_key(self):
        prm = {"oracle_ensemble_te": [1.0, 2.0]}
        assert mod.compute_delta_per_record(prm) is None

    def test_length_mismatch(self):
        prm = {
            "oracle_ensemble_te": [1.0, 2.0],
            "source_oracle_ensemble_te": [0.5],
        }
        assert mod.compute_delta_per_record(prm) is None


# ---------------------------------------------------------------------------
# Baseline analysis
# ---------------------------------------------------------------------------


class TestAnalyzeBaselineSubset:
    def _make_seed_deltas(self, n_sources: int = 50, n_seeds: int = 3):
        rng = np.random.default_rng(0)
        return [
            rng.normal(loc=0.5, scale=0.1, size=n_sources).astype(float)
            for _ in range(n_seeds)
        ]

    def test_basic(self):
        seed_deltas = self._make_seed_deltas(50, 3)
        subset_idx = set(range(5))  # first 5 sources
        r = mod.analyze_baseline_subset(
            baseline="test",
            subset_name="test_subset",
            subset_indices=subset_idx,
            seed_deltas=seed_deltas,
            n_bootstrap=100,
            n_perm=100,
        )
        assert r is not None
        assert r.baseline == "test"
        assert r.subset == "test_subset"
        assert r.n_ood == 5
        assert r.n_id == 45
        # OOD mean ~ 0.5
        assert 0.3 < r.ood_mean < 0.7
        # p_value should be high (no real difference)
        assert r.p_value > 0.05

    def test_empty_subset(self):
        seed_deltas = self._make_seed_deltas(50, 3)
        r = mod.analyze_baseline_subset(
            baseline="test",
            subset_name="empty",
            subset_indices=set(),
            seed_deltas=seed_deltas,
            n_bootstrap=100,
            n_perm=100,
        )
        assert r is None

    def test_mismatched_lengths(self):
        seed_deltas = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),  # mismatched
        ]
        r = mod.analyze_baseline_subset(
            baseline="test",
            subset_name="test",
            subset_indices={0},
            seed_deltas=seed_deltas,
            n_bootstrap=10,
            n_perm=10,
        )
        assert r is None


class TestRobustnessScore:
    def test_empty(self):
        assert mod.robustness_score([]) == 0.0

    def test_negative_when_ood_degrades(self):
        results = [
            mod.SubsetResult(
                baseline="b", subset="s", n_ood=10, n_id=100,
                ood_mean=0.1, ood_ci_low=0.0, ood_ci_high=0.2,
                id_mean=0.5, id_ci_low=0.4, id_ci_high=0.6,
                diff_ood_minus_id=-0.4, cohens_d=-1.0,
                p_value=0.001, n_perm=1000,
                significant_after_bonferroni=True,
            )
        ]
        assert mod.robustness_score(results) == pytest.approx(-0.4)

    def test_zero_when_balanced(self):
        results = [
            mod.SubsetResult(
                baseline="b", subset=f"s{i}", n_ood=10, n_id=100,
                ood_mean=0.5, ood_ci_low=0.4, ood_ci_high=0.6,
                id_mean=0.5, id_ci_low=0.4, id_ci_high=0.6,
                diff_ood_minus_id=0.0, cohens_d=0.0,
                p_value=1.0, n_perm=1000,
                significant_after_bonferroni=False,
            )
            for i in range(4)
        ]
        assert mod.robustness_score(results) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end (synthetic data)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_p2_09_synthetic(self, tmp_path: Path):
        """Build synthetic P2-03 results and run the full pipeline."""
        p2_03_root = tmp_path / "p2_03"
        p2_03_root.mkdir()
        n_sources = 60
        rng = np.random.default_rng(42)

        for baseline in ["te_only", "scalar"]:
            bdir = p2_03_root / f"{baseline}_seed0"
            bdir.mkdir()
            # Write sources.jsonl
            sources: List[Dict[str, Any]] = []
            for i in range(n_sources):
                # Make first 4 sources have rare families (unique gene_id).
                # Remaining 56 sources share gene_id by i//5 -> ~11 families of 5.
                if i < 4:
                    tid = f"ENST{i + 1000:06d}.1"  # unique
                else:
                    tid = f"ENST{(i - 4) // 5:06d}.1"
                sources.append({
                    "five_utr": "GCC" * (1 + i % 3),
                    "cds": "AUG" * (10 + i),
                    "three_utr": "GGG" * (5 + i % 4),
                    "transcript_id": tid,
                    "species": "human",
                })
            (bdir / "sources.jsonl").write_text(
                "\n".join(json.dumps(s) for s in sources) + "\n"
            )
            # Write 2 seed dirs.
            for sid in range(2):
                sdir = bdir / f"seed_{sid:03d}"
                sdir.mkdir()
                # OOD subset (first 6 sources) has lower delta.
                deltas = rng.normal(loc=0.5, scale=0.1, size=n_sources).tolist()
                for i in range(6):
                    deltas[i] -= 0.3  # OOD degradation.
                te = [0.5 + d for d in deltas]
                src_te = [0.5 for _ in deltas]
                summary = {
                    "per_record_metrics": {
                        "oracle_ensemble_te": te,
                        "source_oracle_ensemble_te": src_te,
                    },
                    "n_sources": n_sources,
                    "n_candidates": n_sources,
                }
                (sdir / "eval_summary.json").write_text(json.dumps(summary))

        out_dir = tmp_path / "out"
        summary = mod.run_p2_09(
            p2_03_root=p2_03_root,
            baselines=["te_only", "scalar"],
            out_dir=out_dir,
            n_bootstrap=100,
            n_perm=100,
        )
        assert summary["n_baselines"] == 2
        assert summary["n_sources"] == n_sources
        assert (out_dir / "ood_subsets.json").exists()
        assert (out_dir / "robustness_summary.json").exists()
        assert (out_dir / "robustness_summary.md").exists()
        # Check that some per-baseline per-subset files were written.
        files = list(out_dir.glob("te_only_*.json"))
        assert len(files) == 8  # 8 subsets.

    def test_missing_baseline_skipped(self, tmp_path: Path):
        p2_03_root = tmp_path / "p2_03"
        p2_03_root.mkdir()
        # Only create one baseline.
        bdir = p2_03_root / "te_only_seed0"
        bdir.mkdir()
        sources = [
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": "ENST1.1", "species": "human"}
            for _ in range(40)
        ]
        (bdir / "sources.jsonl").write_text(
            "\n".join(json.dumps(s) for s in sources) + "\n"
        )
        for sid in range(2):
            sdir = bdir / f"seed_{sid:03d}"
            sdir.mkdir()
            te = [1.0] * 40
            src_te = [0.5] * 40
            (sdir / "eval_summary.json").write_text(json.dumps({
                "per_record_metrics": {
                    "oracle_ensemble_te": te,
                    "source_oracle_ensemble_te": src_te,
                },
                "n_sources": 40,
            }))
        out_dir = tmp_path / "out"
        # scalar is missing - should be skipped.
        summary = mod.run_p2_09(
            p2_03_root=p2_03_root,
            baselines=["te_only", "scalar"],
            out_dir=out_dir,
            n_bootstrap=50,
            n_perm=50,
        )
        assert summary["n_baselines"] == 1  # only te_only.


class TestMainCli:
    """Tests for the main() CLI entry point."""

    def _setup_p2_03(self, tmp_path: Path, n_sources: int = 30) -> Path:
        p2_03_root = tmp_path / "p2_03"
        p2_03_root.mkdir()
        bdir = p2_03_root / "te_only_seed0"
        bdir.mkdir()
        sources = [
            {"five_utr": "GC", "cds": "AUG", "three_utr": "GG",
             "transcript_id": f"ENST{i}.1", "species": "human"}
            for i in range(n_sources)
        ]
        (bdir / "sources.jsonl").write_text(
            "\n".join(json.dumps(s) for s in sources) + "\n"
        )
        for sid in range(2):
            sdir = bdir / f"seed_{sid:03d}"
            sdir.mkdir()
            te = [1.0] * n_sources
            src_te = [0.5] * n_sources
            (sdir / "eval_summary.json").write_text(json.dumps({
                "per_record_metrics": {
                    "oracle_ensemble_te": te,
                    "source_oracle_ensemble_te": src_te,
                },
                "n_sources": n_sources,
            }))
        return p2_03_root

    def test_main_without_cluster_assignments(self, tmp_path: Path):
        """main() works without --cluster-assignments (backward compat)."""
        p2_03_root = self._setup_p2_03(tmp_path)
        out_dir = tmp_path / "out"
        rc = mod.main([
            "--p2-03-root", str(p2_03_root),
            "--baselines", "te_only",
            "--out-dir", str(out_dir),
            "--n-bootstrap", "50",
            "--n-perm", "50",
        ])
        assert rc == 0
        assert (out_dir / "robustness_summary.json").exists()

    def test_main_with_cluster_assignments(self, tmp_path: Path):
        """main() accepts --cluster-assignments and --test-idx."""
        n_sources = 30
        p2_03_root = self._setup_p2_03(tmp_path, n_sources=n_sources)
        # Build cluster_assignments.json: 100 records in full dataset.
        cluster_assignments = list(range(100))  # each in its own cluster.
        ca_path = tmp_path / "cluster_assignments.json"
        ca_path.write_text(json.dumps(cluster_assignments))
        # test.idx: first n_sources records are the test split.
        test_idx = list(range(n_sources))
        ti_path = tmp_path / "test.idx"
        ti_path.write_text("\n".join(str(i) for i in test_idx) + "\n")
        out_dir = tmp_path / "out"
        rc = mod.main([
            "--p2-03-root", str(p2_03_root),
            "--baselines", "te_only",
            "--out-dir", str(out_dir),
            "--n-bootstrap", "50",
            "--n-perm", "50",
            "--cluster-assignments", str(ca_path),
            "--test-idx", str(ti_path),
        ])
        assert rc == 0
        # All sources are in their own cluster (size 1, <=2), so family_rare
        # should capture all of them.
        import json as _json
        subsets = _json.loads((out_dir / "ood_subsets.json").read_text())
        assert subsets["family_rare"]["size"] == n_sources
        assert "split contract" in subsets["family_rare"]["definition"]

    def test_main_cluster_assignments_requires_test_idx(self, tmp_path: Path):
        """--cluster-assignments without --test-idx should error."""
        p2_03_root = self._setup_p2_03(tmp_path)
        ca_path = tmp_path / "cluster_assignments.json"
        ca_path.write_text("[0, 1, 2]")
        out_dir = tmp_path / "out"
        with pytest.raises(SystemExit):
            mod.main([
                "--p2-03-root", str(p2_03_root),
                "--baselines", "te_only",
                "--out-dir", str(out_dir),
                "--cluster-assignments", str(ca_path),
                # --test-idx deliberately omitted.
            ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
