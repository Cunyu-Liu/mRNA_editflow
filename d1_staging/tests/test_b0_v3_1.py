#!/usr/bin/env python
"""Unit tests for the B0-R (v3.1) seven-stage transaction (builder/validator/freezer).

These tests run against synthetic D1-shaped data so they are fast and
deterministic. They cover, at minimum:

  * RFC8785/JCS canonicalization and self-hash
  * frozen definition/component-set hashes
  * builder Stage 1..5 artifact generation (dual-store isolation)
  * validator schema / self-hash / transition-chain / conservation / FK / isolation
  * freezer Stage 6 PREPARED + Stage 7 root commit
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "b0"))

# Import the shared modules under test. They are deployed under d1_staging/scripts/b0/.
if (Path(__file__).resolve().parent.parent / "scripts" / "b0" / "b0_v3_1_common.py").exists():
    MOD_DIR = Path(__file__).resolve().parent.parent / "scripts" / "b0"
else:
    MOD_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(MOD_DIR))

import b0_v3_1_common as common  # noqa: E402
import b0_v3_1_builder as builder  # noqa: E402
import b0_v3_1_validator as validator  # noqa: E402
import b0_v3_1_freezer as freezer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: synthetic D1-shaped data + registry YAML
# ---------------------------------------------------------------------------

def _write_registry_files(worktree: Path) -> None:
    exec_dir = worktree / "docs" / "execution"
    cfg_dir = worktree / "configs"
    exec_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # task registry (12 tasks, minimal fields)
    tasks = []
    for tid in common.REQUIRED_TASK_IDS:
        kind = "AUXILIARY_TRAINING" if tid.startswith("F") else "BENCHMARK_EVALUATION"
        tasks.append({
            "task_id": tid, "task_kind": kind, "object_type": "OBSERVATION" if tid.startswith("F") else "PAIR",
            "scientific_track": "F" if tid.startswith("F") else "E",
        })
    (exec_dir / "task_registry_v3_1.yaml").write_text(
        "tasks:\n" + "".join(
            f"  - task_id: {t['task_id']}\n    task_kind: {t['task_kind']}\n"
            f"    object_type: {t['object_type']}\n    scientific_track: {t['scientific_track']}\n"
            for t in tasks))

    # split registry (10 splits)
    splits = [{"split_contract_id": sid} for sid in common.REQUIRED_SPLIT_IDS]
    (exec_dir / "split_registry_v3_1.yaml").write_text(
        "splits:\n" + "".join(
            f"  - split_contract_id: {s['split_contract_id']}\n" for s in splits))

    # matrix: 120 rows, ALLOWED only for E-PAIR / F-OBSERVATION matching track
    rows = []
    for t in tasks:
        for s in splits:
            if t["scientific_track"] == "E" and t["object_type"] == "PAIR":
                mapping = "ALLOWED"
            elif t["scientific_track"] == "F" and t["object_type"] == "OBSERVATION":
                mapping = "ALLOWED"
            else:
                mapping = "NOT_ALLOWED"
            rows.append({
                "task_id": t["task_id"], "split_contract_id": s["split_contract_id"],
                "contract_mapping": mapping,
                "object_type": t["object_type"], "scientific_track": t["scientific_track"],
            })
    (exec_dir / "task_split_contract_matrix_v3_1.yaml").write_text(
        "rows:\n" + "".join(
            f"  - task_id: {r['task_id']}\n    split_contract_id: {r['split_contract_id']}\n"
            f"    contract_mapping: {r['contract_mapping']}\n    object_type: {r['object_type']}\n"
            f"    scientific_track: {r['scientific_track']}\n" for r in rows))

    # viability rule
    (exec_dir / "resource_viability_rule_v3_1.yaml").write_text(
        "thresholds:\n  repeated_context_min_groups: 10\n  min_calibration_components_for_activation: 5\n")

    # config (frozen hashes)
    cfg = {
        "contract_id": "utr_editflow_goal_v3.1_benchmark_first",
        "frozen_hashes": common.FROZEN_HASHES,
    }
    (cfg_dir / "utr_editflow_contract_v3_1.yaml").write_text(
        "contract_id: utr_editflow_goal_v3.1_benchmark_first\n"
        "frozen_hashes:\n" + "".join(f"  {k}: \"{v}\"\n" for k, v in common.FROZEN_HASHES.items()))


def _make_d1(out_dir: Path, n_ord_pairs=5, n_res_pairs=3, n_ord_obs=4, n_res_obs=2):
    od = out_dir / "ordinary"
    rd = out_dir / "restricted"
    od.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)

    with open(od / "utr_edit_pairs.jsonl", "w") as fh:
        for i in range(n_ord_pairs):
            fh.write(json.dumps({
                "pair_id": f"gse114002_pair_{i}",
                "candidate_id": f"gse114002_rel_{i}",
                "source_sequence_id": f"GSE114002_NC_..._{i}__src",
                "candidate_sequence_id": f"GSE114002_NC_..._{i}__cand",
                "design_relation_group_id": f"gse114002_design_{i}",
                "scientific_track": "E",
                "relation_type": "SOURCE_CONDITIONED_EDIT",
                "immutable_base_future_use_role": "AWAITING_B0_GLOBAL_DISPOSITION",
            }) + "\n")
    with open(rd / "utr_edit_pairs.jsonl", "w") as fh:
        for i in range(n_res_pairs):
            fh.write(json.dumps({
                "pair_id": f"gse246381_pair_{i}",
                "candidate_id": f"gse246381_rel_{i}",
                "source_sequence_id": f"GSE246381_ENST_{i}__src",
                "candidate_sequence_id": f"GSE246381_ENST_{i}__cand",
                "design_relation_group_id": f"gse246381_design_{i}",
                "scientific_track": "E",
                "relation_type": "SOURCE_CONDITIONED_EDIT",
                "immutable_base_future_use_role": "SEALED_EXTERNAL_FINAL_CANDIDATE",
            }) + "\n")
    with open(od / "functional_observations.jsonl", "w") as fh:
        for i in range(n_ord_obs):
            fh.write(json.dumps({
                "observation_id": f"gse114002_obs_{i}",
                "sequence_id": f"GSE114002_seq_{i}",
                "endpoint_id": "ep_rl",
                "context_id": "ctx_gse114002",
                "value": float(i), "unit": "rl",
            }) + "\n")
    with open(rd / "functional_observations.jsonl", "w") as fh:
        for i in range(n_res_obs):
            fh.write(json.dumps({
                "observation_id": f"gse246381_obs_{i}",
                "sequence_id": f"GSE246381_seq_{i}",
                "endpoint_id": "ep_hek_mean_umi_ref",
                "context_id": "ctx_gse246381",
                "value": float(i), "unit": "hek_mean_umi_ref",
            }) + "\n")
    return od, rd


@pytest.fixture()
def env(tmp_path):
    worktree = tmp_path / "worktree"
    _write_registry_files(worktree)
    od, rd = _make_d1(tmp_path / "d1")
    out = tmp_path / "out"
    res_out = tmp_path / "res_out"
    out.mkdir(parents=True, exist_ok=True)
    res_out.mkdir(parents=True, exist_ok=True)
    return {
        "worktree": worktree, "od": od, "rd": rd, "out": out, "res_out": res_out,
    }


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------

def test_jcs_canonicalization():
    assert common.jcs_dumps({"b": 1, "a": [{"d": True, "c": "x"}, None]}) == \
        '{"a":[{"c":"x","d":true},null],"b":1}'


def test_frozen_hashes_match_contract():
    assert common.set_sha256(common.REQUIRED_TASK_IDS) == \
        common.FROZEN_HASHES["task_id_set_sha256"]
    assert common.set_sha256(common.REQUIRED_SPLIT_IDS) == \
        common.FROZEN_HASHES["split_id_set_sha256"]
    assert common.set_sha256(common.SEALED_COHORT_IDS) == \
        common.FROZEN_HASHES["sealed_cohort_set_sha256"]
    assert common.set_sha256(common.ORDINARY_PREPARED_COMPONENTS) == \
        common.FROZEN_HASHES["b0_ordinary_prepared_component_set_sha256"]
    assert common.set_sha256(common.RESTRICTED_PREPARED_COMPONENTS) == \
        common.FROZEN_HASHES["b0_restricted_prepared_component_set_sha256"]


def test_jcs_self_hash_excludes_named_field():
    obj = {"a": 1, "sha": "x"}
    assert common.jcs_sha256(obj, exclude=["sha"]) == \
        common.jcs_sha256({"a": 1}, exclude=["sha"])


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def test_verify_definitions(env):
    errs = builder.verify_definitions(env["worktree"])
    # structural checks pass on synthetic data (frozen allowlist hash is bound to
    # the real contract matrix, so it legitimately differs here)
    assert errs["_task_count"] == 12
    assert errs["_split_count"] == 10
    assert errs["_matrix_count"] == 120


def test_builder_stages(env):
    wt, out, res_out = env["worktree"], env["out"], env["res_out"]
    od, rd = env["od"], env["rd"]
    c = Counter()
    c.update(builder.build_stage1(od / "utr_edit_pairs.jsonl", rd / "utr_edit_pairs.jsonl",
                                  out / "B0_ROLE_DECISION_EVIDENCE.jsonl",
                                  res_out / "B0_ROLE_DECISION_EVIDENCE.jsonl"))
    c.update(builder.build_stage2(od / "utr_edit_pairs.jsonl", rd / "utr_edit_pairs.jsonl",
                                  out / "EFFECTIVE_ROLE_PROJECTION.jsonl",
                                  res_out / "RELATION_ROLE_TRANSITIONS.jsonl"))
    c.update(builder.build_stage3(od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                                  rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                                  out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                  out / "ELIGIBILITY_MANIFEST.jsonl",
                                  res_out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                  res_out / "ELIGIBILITY_MANIFEST.jsonl"))
    c.update(builder.build_stage4(wt, out, od / "functional_observations.jsonl"))
    c.update(builder.build_stage5(wt, out, res_out,
                                  od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                                  rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                                  out / "ELIGIBILITY_MANIFEST.jsonl",
                                  res_out / "ELIGIBILITY_MANIFEST.jsonl"))
    from b0_v3_1_common import write_jsonl
    write_jsonl(out / "RELATION_ROLE_TRANSITIONS.jsonl", [])

    assert c["role_evidence_total"] == 5 + 3
    assert c["ordinary_eligibility_evidence"] == 5 + 4
    assert c["restricted_eligibility_evidence"] == 3 + 2
    assert c["task_decisions"] == 12
    assert c["split_decisions"] == 10
    assert c["applicability_decisions"] == 120


def test_builder_dual_store_isolation(env):
    wt, out, res_out = env["worktree"], env["out"], env["res_out"]
    od, rd = env["od"], env["rd"]
    builder.build_stage3(od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                         rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                         out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                         out / "ELIGIBILITY_MANIFEST.jsonl",
                         res_out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                         res_out / "ELIGIBILITY_MANIFEST.jsonl")
    builder.build_stage5(wt, out, res_out,
                         od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                         rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                         out / "ELIGIBILITY_MANIFEST.jsonl",
                         res_out / "ELIGIBILITY_MANIFEST.jsonl")
    ord_objs = {r["object_id"] for r in common.iter_jsonl(out / "ELIGIBILITY_MANIFEST.jsonl")}
    res_objs = {r["object_id"] for r in common.iter_jsonl(res_out / "ELIGIBILITY_MANIFEST.jsonl")}
    assert ord_objs.isdisjoint(res_objs)
    # restricted cells must live only in the restricted store
    res_cells = list(common.iter_jsonl(res_out / "TASK_ELIGIBILITY_UNIVERSE.jsonl"))
    assert all(r["object_id"].startswith("gse246381") for r in res_cells)


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def _run_full_pipeline(env):
    """Run builder -> validator -> freezer on synthetic data."""
    wt, out, res_out = env["worktree"], env["out"], env["res_out"]
    od, rd = env["od"], env["rd"]
    c = Counter()
    c.update(builder.verify_definitions(wt))
    c.update(builder.build_stage1(od / "utr_edit_pairs.jsonl", rd / "utr_edit_pairs.jsonl",
                                  out / "B0_ROLE_DECISION_EVIDENCE.jsonl",
                                  res_out / "B0_ROLE_DECISION_EVIDENCE.jsonl"))
    c.update(builder.build_stage2(od / "utr_edit_pairs.jsonl", rd / "utr_edit_pairs.jsonl",
                                  out / "EFFECTIVE_ROLE_PROJECTION.jsonl",
                                  res_out / "RELATION_ROLE_TRANSITIONS.jsonl"))
    c.update(builder.build_stage3(od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                                  rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                                  out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                  out / "ELIGIBILITY_MANIFEST.jsonl",
                                  res_out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                  res_out / "ELIGIBILITY_MANIFEST.jsonl"))
    c.update(builder.build_stage4(wt, out, od / "functional_observations.jsonl"))
    c.update(builder.build_stage5(wt, out, res_out,
                                  od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
                                  rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
                                  out / "ELIGIBILITY_MANIFEST.jsonl",
                                  res_out / "ELIGIBILITY_MANIFEST.jsonl"))
    from b0_v3_1_common import write_jsonl
    write_jsonl(out / "RELATION_ROLE_TRANSITIONS.jsonl", [])
    return c


def test_validator_passes(env):
    _run_full_pipeline(env)
    errs = Counter()
    # NOTE: validate_definition_hashes is omitted here because its frozen
    # allowlist hash is bound to the real contract matrix (synthetic data
    # legitimately differs). It is exercised on the real remote worktree.
    for key, fname in [
        ("B0_ROLE_DECISION_EVIDENCE", "B0_ROLE_DECISION_EVIDENCE.jsonl"),
        ("GLOBAL_ELIGIBILITY_DECISION_EVIDENCE", "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl"),
        ("ELIGIBILITY_MANIFEST", "ELIGIBILITY_MANIFEST.jsonl"),
        ("RELATION_ROLE_TRANSITIONS", "RELATION_ROLE_TRANSITIONS.jsonl"),
        ("EFFECTIVE_ROLE_PROJECTION", "EFFECTIVE_ROLE_PROJECTION.jsonl"),
        ("TASK_ELIGIBILITY_UNIVERSE", "TASK_ELIGIBILITY_UNIVERSE.jsonl"),
        ("SPLIT_ASSIGNMENTS", "SPLIT_ASSIGNMENTS.jsonl"),
    ]:
        errs.update(validator.validate_artifact_schema(env["out"] / fname, key))
        errs.update(validator.validate_artifact_schema(env["res_out"] / fname, key))
    errs.update(validator.validate_self_hash(env["out"] / "ELIGIBILITY_MANIFEST.jsonl",
                                             "eligibility_manifest_sha256", "object_id"))
    errs.update(validator.validate_self_hash(env["res_out"] / "ELIGIBILITY_MANIFEST.jsonl",
                                             "eligibility_manifest_sha256", "object_id"))
    errs.update(validator.validate_transition_chain(env["res_out"] / "RELATION_ROLE_TRANSITIONS.jsonl"))
    errs.update(validator.validate_conservation(
        env["od"] / "utr_edit_pairs.jsonl", env["od"] / "functional_observations.jsonl",
        env["rd"] / "utr_edit_pairs.jsonl", env["rd"] / "functional_observations.jsonl",
        env["out"], env["res_out"]))
    errs.update(validator.validate_fk_and_isolation(env["out"], env["res_out"]))
    errs.update(validator.validate_decision_counts(env["out"]))
    total = sum(v for k, v in errs.items() if not k.startswith("_"))
    assert total == 0, dict(errs)


def test_validator_detects_self_hash_mutation(env):
    _run_full_pipeline(env)
    man = env["out"] / "ELIGIBILITY_MANIFEST.jsonl"
    rows = list(common.iter_jsonl(man))
    rows[0]["eligibility_manifest_sha256"] = "0" * 64
    common.write_jsonl(man, rows)
    errs = validator.validate_self_hash(man, "eligibility_manifest_sha256", "object_id")
    assert errs["self_hash_mismatch"] >= 1


def test_validator_detects_transition_chain_break(env):
    _run_full_pipeline(env)
    chain = env["res_out"] / "RELATION_ROLE_TRANSITIONS.jsonl"
    rows = list(common.iter_jsonl(chain))
    if len(rows) >= 2:
        # break the link between row 0 and row 1
        rows[1]["prev_event_sha256"] = "broken"
        common.write_jsonl(chain, rows)
        errs = validator.validate_transition_chain(chain)
        assert errs["transition_predecessor_hash_mismatch"] >= 1


def test_validator_detects_cross_store_overlap(env):
    _run_full_pipeline(env)
    # force overlap by injecting a restricted object into the ordinary manifest
    res_man = list(common.iter_jsonl(env["res_out"] / "ELIGIBILITY_MANIFEST.jsonl"))
    ord_man_path = env["out"] / "ELIGIBILITY_MANIFEST.jsonl"
    ord_rows = list(common.iter_jsonl(ord_man_path))
    ord_rows.append(res_man[0])
    common.write_jsonl(ord_man_path, ord_rows)
    errs = validator.validate_fk_and_isolation(env["out"], env["res_out"])
    assert errs["cross_store_object_overlap"] >= 1


# ---------------------------------------------------------------------------
# freezer
# ---------------------------------------------------------------------------

def test_freezer_prepared_and_commit(env):
    _run_full_pipeline(env)
    wt, out, res_out = env["worktree"], env["out"], env["res_out"]
    od, rd = env["od"], env["rd"]

    freezer.materialize_ordinary(out, od / "utr_edit_pairs.jsonl",
                                 od / "functional_observations.jsonl", wt)
    freezer.materialize_restricted(res_out, rd / "functional_observations.jsonl")

    ord_prepared = freezer.build_prepared_manifest(
        "ORDINARY", freezer.ORDINARY_COMPONENT_FILES, out, None)
    res_prepared = freezer.build_prepared_manifest(
        "RESTRICTED_GSE246381", freezer.RESTRICTED_COMPONENT_FILES, res_out, None)

    assert ord_prepared["component_set_sha256"] == \
        common.FROZEN_HASHES["b0_ordinary_prepared_component_set_sha256"]
    assert res_prepared["component_set_sha256"] == \
        common.FROZEN_HASHES["b0_restricted_prepared_component_set_sha256"]
    # every logical ID maps to exactly one physical path + hash
    assert len(ord_prepared["component_paths_and_sha256s"]) == len(common.ORDINARY_PREPARED_COMPONENTS)
    assert len(res_prepared["component_paths_and_sha256s"]) == len(common.RESTRICTED_PREPARED_COMPONENTS)
    # self-hash correctness
    assert ord_prepared["prepared_manifest_sha256"] == \
        common.jcs_sha256(ord_prepared, exclude=["prepared_manifest_sha256"])

    commit = freezer.build_root_commit(
        ord_prepared, res_prepared, out, res_out,
        ord_prepared["component_set_sha256"], res_prepared["component_set_sha256"],
        "executable_sha", common.GENESIS_SENTINEL, 1)
    assert commit["commit_record_sha256"] == common.jcs_sha256(commit, exclude=["commit_record_sha256"])
    assert commit["ordinary_prepared_manifest_sha256"] == ord_prepared["prepared_manifest_sha256"]
    assert commit["restricted_prepared_manifest_sha256"] == res_prepared["prepared_manifest_sha256"]


def test_freezer_rejects_missing_component(env):
    _run_full_pipeline(env)
    out, res_out = env["out"], env["res_out"]
    freezer.materialize_ordinary(out, env["od"] / "utr_edit_pairs.jsonl",
                                 env["od"] / "functional_observations.jsonl", env["worktree"])
    freezer.materialize_restricted(res_out, env["rd"] / "functional_observations.jsonl")
    # delete one component to force a FAIL
    (res_out / "B0_ROLE_DECISION_EVIDENCE.jsonl").unlink()
    with pytest.raises(SystemExit):
        freezer.build_prepared_manifest("RESTRICTED_GSE246381",
                                        freezer.RESTRICTED_COMPONENT_FILES, res_out, None)


def test_freezer_rejects_component_set_hash_drift(env):
    _run_full_pipeline(env)
    out, res_out = env["out"], env["res_out"]
    freezer.materialize_ordinary(out, env["od"] / "utr_edit_pairs.jsonl",
                                 env["od"] / "functional_observations.jsonl", env["worktree"])
    freezer.materialize_restricted(res_out, env["rd"] / "functional_observations.jsonl")
    # drop a component from the mapping -> set hash drift
    files = list(freezer.RESTRICTED_COMPONENT_FILES)[:-1]
    with pytest.raises(SystemExit):
        freezer.build_prepared_manifest("RESTRICTED_GSE246381", files, res_out, None)