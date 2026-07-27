"""P0-02: result-artifact integrity audit — unit tests on synthetic fixtures.

Each check in scripts/audit_result_artifacts.py must detect a planted
discrepancy, and a clean fixture must produce zero FAIL findings.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_result_artifacts import (
    SEVERITY_FAIL,
    apply_dispositions,
    check_checkpoint_correspondence,
    check_empty_json,
    check_frozen_overwrite,
    check_gate_status_conflict,
    check_hash_sidecars,
    check_missing_references,
    check_record_counts,
    check_summary_vs_records,
    AuditReport,
    run_audit,
)


def _report() -> AuditReport:
    return AuditReport(generated_at="t0", repo_root="")


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _gate_shard(seeds, verdict="PASS", finite=1.0):
    n = len(seeds)
    return {
        "gate": "B",
        "n_seeds": n,
        "n_seeds_completed": n,
        "n_seeds_failed": 0,
        "verdict": verdict,
        "criteria": {
            "no_collapse": finite >= 0.99,
            "finite_fractions": [finite] * n,
            "hard_constraints_100": True,
            "constraint_validities": [1.0] * n,
            "two_thirds_beat_warm": True,
            "beat_warm_start": [True] * n,
            "no_reward_hacking": True,
            "positive_rates": [0.7] * n,
            "stop_not_collapsed": True,
            "stop_rates": [0.0] * n,
        },
        "results": [{"seed": s, "final_validation": {
            "mean_reward": -0.04, "positive_improvement_rate": 0.7,
            "stop_at_root_rate": 0.0, "constraint_validity": 1.0},
            "warm_start_validation": {"mean_reward": -0.1}}
            for s in seeds],
    }


# ---------------------------------------------------------------------------
# empty_json
# ---------------------------------------------------------------------------

class TestEmptyJson:
    def test_zero_byte_detected(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "x.json").write_text("")
        rep = _report()
        check_empty_json(tmp_path, rep)
        assert any(f.severity == SEVERITY_FAIL for f in rep.findings)

    def test_truncated_json_detected(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "x.json").write_text('{"a": [1, 2,')
        rep = _report()
        check_empty_json(tmp_path, rep)
        assert any("invalid JSON" in f.message for f in rep.findings)

    def test_valid_json_clean(self, tmp_path):
        _write_json(tmp_path / "docs" / "x.json", {"a": 1})
        rep = _report()
        check_empty_json(tmp_path, rep)
        assert not rep.findings


# ---------------------------------------------------------------------------
# missing_references
# ---------------------------------------------------------------------------

class TestMissingReferences:
    def test_missing_result_asset_is_fail(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "r.md").write_text("see `benchmark/paper/x.json`")
        rep = _report()
        check_missing_references(tmp_path, rep)
        fails = [f for f in rep.findings if f.severity == SEVERITY_FAIL]
        assert len(fails) == 1

    def test_missing_md_ref_is_warn(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "r.md").write_text("see `docs/nonexistent.md`")
        rep = _report()
        check_missing_references(tmp_path, rep)
        assert rep.n_fail == 0 and rep.n_warn == 1

    def test_package_prefix_resolves(self, tmp_path):
        (tmp_path / "benchmark").mkdir(parents=True)
        (tmp_path / "benchmark" / "x.json").write_text("{}")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "r.md").write_text("see `mrna_editflow/benchmark/x.json`")
        rep = _report()
        check_missing_references(tmp_path, rep)
        assert not rep.findings

    def test_external_repo_ref_is_warn(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "r.md").write_text("see `other_repo/flow.py`")
        rep = _report()
        check_missing_references(tmp_path, rep)
        assert rep.n_fail == 0


# ---------------------------------------------------------------------------
# checkpoint_correspondence
# ---------------------------------------------------------------------------

class TestCheckpointCorrespondence:
    def _setup(self, tmp_path, extra_orphan_seed=None, drop_seed=None):
        seeds = [42, 123, 456]
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateA.json",
                    _gate_shard(seeds))
        # gateB shard fixtures so their checkpoint specs resolve too
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB_gpu1.json",
                    _gate_shard([1]))
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB_gpu6.json",
                    _gate_shard([2]))
        for gdir, gseeds in (("p3_08_gateB_gpu1", [1]), ("p3_08_gateB_gpu6", [2])):
            d = tmp_path / "checkpoints" / gdir
            d.mkdir(parents=True)
            for s in gseeds:
                for step in [1000, 2000, 3000, 4000, 5000]:
                    (d / f"grpo_seed{s}_step{step}.pt").write_bytes(b"x")
        ck = tmp_path / "checkpoints" / "p3_08_gateA"
        ck.mkdir(parents=True)
        for s in seeds:
            if s == drop_seed:
                continue
            for step in [200, 400, 600, 800, 1000]:
                (ck / f"grpo_seed{s}_step{step}.pt").write_bytes(b"x")
        if extra_orphan_seed is not None:
            for step in [200, 400, 600, 800, 1000]:
                (ck / f"grpo_seed{extra_orphan_seed}_step{step}.pt").write_bytes(b"x")

    def test_missing_checkpoint_is_fail(self, tmp_path):
        self._setup(tmp_path, drop_seed=456)
        rep = _report()
        check_checkpoint_correspondence(tmp_path, rep)
        fails = [f for f in rep.findings if f.severity == SEVERITY_FAIL]
        assert any("456" in f.message for f in fails)

    def test_full_correspondence_clean(self, tmp_path):
        self._setup(tmp_path)
        rep = _report()
        check_checkpoint_correspondence(tmp_path, rep)
        assert rep.n_fail == 0

    def test_orphan_checkpoint_is_warn(self, tmp_path):
        self._setup(tmp_path, extra_orphan_seed=999)
        rep = _report()
        check_checkpoint_correspondence(tmp_path, rep)
        assert rep.n_fail == 0 and rep.n_warn >= 1


# ---------------------------------------------------------------------------
# gate_status_conflict
# ---------------------------------------------------------------------------

class TestGateStatusConflict:
    def _write_gate_set(self, tmp_path, merged_verdict, shard_verdicts,
                        top_verdict=None):
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB_gpu1.json",
                    _gate_shard([1, 2], verdict=shard_verdicts[0]))
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB_gpu6.json",
                    _gate_shard([3, 4], verdict=shard_verdicts[1]))
        merged = _gate_shard([1, 2, 3, 4], verdict=merged_verdict)
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB.json", merged)
        if top_verdict is not None:
            _write_json(tmp_path / "docs" / "p3_08_grpo_results.json",
                        {"gate_b_verdict": top_verdict})

    def test_verdict_conflict_detected(self, tmp_path):
        self._write_gate_set(tmp_path, "FAIL", ["PASS", "PASS"], top_verdict="PASS")
        rep = _report()
        check_gate_status_conflict(tmp_path, rep)
        assert rep.n_fail >= 2  # merged-vs-shards and top-vs-merged

    def test_consistent_set_clean(self, tmp_path):
        self._write_gate_set(tmp_path, "PASS", ["PASS", "PASS"], top_verdict="PASS")
        rep = _report()
        check_gate_status_conflict(tmp_path, rep)
        assert rep.n_fail == 0

    def test_shard_order_insensitive(self, tmp_path):
        # merged arrays in gpu6-first order must not conflict with gpu1-first shards
        self._write_gate_set(tmp_path, "PASS", ["PASS", "PASS"])
        rep = _report()
        check_gate_status_conflict(tmp_path, rep)
        assert not any("positive_rates" in f.message for f in rep.findings)


# ---------------------------------------------------------------------------
# hash_mismatch
# ---------------------------------------------------------------------------

class TestHashSidecars:
    def test_mismatch_detected(self, tmp_path):
        payload = _write_json(tmp_path / "docs" / "m.json", {"a": 1})
        (tmp_path / "docs" / "m.json.sha256").write_text("0" * 64 + "  m.json\n")
        rep = _report()
        check_hash_sidecars(tmp_path, rep)
        assert rep.n_fail == 1

    def test_match_clean(self, tmp_path):
        payload = _write_json(tmp_path / "docs" / "m.json", {"a": 1})
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (tmp_path / "docs" / "m.json.sha256").write_text(f"{digest}  m.json\n")
        rep = _report()
        check_hash_sidecars(tmp_path, rep)
        assert not rep.findings


# ---------------------------------------------------------------------------
# record_count_mismatch
# ---------------------------------------------------------------------------

class TestRecordCounts:
    def test_n_seeds_mismatch(self, tmp_path):
        d = _gate_shard([1, 2, 3])
        d["n_seeds"] = 10
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateA.json", d)
        rep = _report()
        check_record_counts(tmp_path, rep)
        assert any("n_seeds" in f.message for f in rep.findings)

    def test_n_pairs_mismatch(self, tmp_path):
        _write_json(tmp_path / "docs" / "p3_09_oracle_transfer.json",
                    {"n_pairs": 5, "per_pair_records": [{"a": 1}]})
        rep = _report()
        check_record_counts(tmp_path, rep)
        assert rep.n_fail == 1

    def test_consistent_clean(self, tmp_path):
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateA.json",
                    _gate_shard([1, 2, 3]))
        _write_json(tmp_path / "docs" / "p3_09_oracle_transfer.json",
                    {"n_pairs": 1, "per_pair_records": [{"a": 1}]})
        rep = _report()
        check_record_counts(tmp_path, rep)
        assert rep.n_fail == 0


# ---------------------------------------------------------------------------
# summary_vs_records
# ---------------------------------------------------------------------------

class TestSummaryVsRecords:
    def test_summary_mean_mismatch(self, tmp_path):
        d = _gate_shard([1, 2])
        d["summary_stats"] = {"pos_rate": {"mean": 0.99, "values": [0.7, 0.7]}}
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB.json", d)
        rep = _report()
        check_summary_vs_records(tmp_path, rep)
        assert any("pos_rate" in f.message for f in rep.findings)

    def test_summary_consistent(self, tmp_path):
        d = _gate_shard([1, 2])
        d["summary_stats"] = {"pos_rate": {"mean": 0.7, "values": [0.7, 0.7]}}
        _write_json(tmp_path / "docs" / "p3_08_grpo_results_gateB.json", d)
        rep = _report()
        check_summary_vs_records(tmp_path, rep)
        assert rep.n_fail == 0


# ---------------------------------------------------------------------------
# frozen_artifact_overwrite
# ---------------------------------------------------------------------------

class TestFrozenOverwrite:
    def _manifest(self, tmp_path, files):
        m = {"artifacts": {"g": {"files": [
            {"path": rel, "sha256": hashlib.sha256(
                (tmp_path / rel).read_bytes()).hexdigest()}
            for rel in files]}}}
        return _write_json(tmp_path / "artifacts" / "nmi_phase0_freeze_manifest.json", m)

    def test_modified_after_freeze_detected(self, tmp_path):
        _write_json(tmp_path / "docs" / "x.json", {"v": 1})
        manifest = self._manifest(tmp_path, ["docs/x.json"])
        _write_json(tmp_path / "docs" / "x.json", {"v": 2})  # overwrite
        rep = _report()
        check_frozen_overwrite(tmp_path, rep, manifest)
        assert any("modified after freeze" in f.message for f in rep.findings)

    def test_deleted_after_freeze_detected(self, tmp_path):
        _write_json(tmp_path / "docs" / "x.json", {"v": 1})
        manifest = self._manifest(tmp_path, ["docs/x.json"])
        (tmp_path / "docs" / "x.json").unlink()
        rep = _report()
        check_frozen_overwrite(tmp_path, rep, manifest)
        assert any("deleted after freeze" in f.message for f in rep.findings)

    def test_unmodified_clean(self, tmp_path):
        _write_json(tmp_path / "docs" / "x.json", {"v": 1})
        manifest = self._manifest(tmp_path, ["docs/x.json"])
        rep = _report()
        check_frozen_overwrite(tmp_path, rep, manifest)
        assert rep.n_fail == 0


# ---------------------------------------------------------------------------
# dispositions
# ---------------------------------------------------------------------------

class TestDispositions:
    def test_disposition_downgrades_fail(self, tmp_path):
        _write_json(tmp_path / "docs" / "nmi_artifact_dispositions.json", {
            "dispositions": [{
                "check": "missing_references",
                "reference": "benchmark/x.json",
                "disposition": "lost_accepted",
                "rationale": "superseded",
                "impact": "no paper-candidate number affected",
            }]})
        (tmp_path / "docs" / "r.md").write_text("see `benchmark/x.json`")
        rep = _report()
        check_missing_references(tmp_path, rep)
        assert rep.n_fail == 1
        from scripts.audit_result_artifacts import load_dispositions
        apply_dispositions(rep, load_dispositions(tmp_path))
        assert rep.n_fail == 0 and rep.n_warn == 1


# ---------------------------------------------------------------------------
# end-to-end on a clean synthetic repo
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_clean_repo_zero_fail(self, tmp_path):
        TestCheckpointCorrespondence()._setup(tmp_path)
        rep = run_audit(tmp_path, manifest_path=tmp_path / "nonexistent.json")
        fails = [f for f in rep.findings if f.severity == SEVERITY_FAIL]
        assert fails == [], f"unexpected FAILs: {[vars(f) for f in fails]}"


# ---------------------------------------------------------------------------
# real repository smoke test (skipped when repo assets unavailable)
# ---------------------------------------------------------------------------

class TestRealRepo:
    def test_real_repo_audit_runs(self):
        repo = Path(__file__).resolve().parent.parent
        if not (repo / "docs" / "p3_08_grpo_results_gateB.json").exists():
            pytest.skip("result assets not present in this checkout")
        rep = run_audit(repo)
        # the audit must run all 8 checks and produce a traceability section
        assert len(rep.checks_run) == 8
        assert rep.traceability, "traceability section must not be empty"
        # Gate A/B verdicts must be consistent after P0-02 remediation
        conflicts = [f for f in rep.findings
                     if f.check == "gate_status_conflict"
                     and f.severity == SEVERITY_FAIL]
        assert conflicts == [], f"gate status conflict: {[vars(f) for f in conflicts]}"
