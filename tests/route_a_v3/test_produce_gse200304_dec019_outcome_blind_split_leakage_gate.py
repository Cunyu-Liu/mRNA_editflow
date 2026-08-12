from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/produce_gse200304_dec019_outcome_blind_split_leakage_gate.py"
)
CONFIG = (
    ROOT
    / "configs/route_a_v3_gse200304_dec019_outcome_blind_split_leakage_gate_v1.json"
)
CONSUMER_CONFIG = (
    ROOT
    / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
CONSUMER_SCRIPT = (
    ROOT
    / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
if not CONSUMER_CONFIG.is_file() or not CONSUMER_SCRIPT.is_file():
    consumer_root = WORK / "g200_split_consumer_commitment_upgrade_staging"
    CONSUMER_CONFIG = (
        consumer_root
        / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
    )
    CONSUMER_SCRIPT = (
        consumer_root
        / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
    )

AUTHORITY_CONFIG = json.loads(CONFIG.read_text(encoding="utf-8"))
PUBLIC_ROOT = Path(AUTHORITY_CONFIG["source_authority"]["public_data_root"])
if not PUBLIC_ROOT.is_dir():
    PUBLIC_ROOT = WORK / "gse200304_public_assets_20260810T143731P0800"
GROUP_PUBLICATION = Path(
    AUTHORITY_CONFIG["source_authority"]["biological_group_publication"][
        "absolute_path"
    ]
)
if not GROUP_PUBLICATION.is_dir():
    GROUP_PUBLICATION = WORK / "g200_split_real_input_fixture/group_exact4_bound"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load(SCRIPT, "gse200304_split_producer")
CONSUMER = _load(CONSUMER_SCRIPT, "gse200304_split_consumer")


def producer_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
            "consumer_upgrade_binding_commit": "4" * 40,
            "consumer_upgrade_config_sha256": PRODUCER.sha256(
                CONSUMER_CONFIG.read_bytes()
            ),
            "consumer_upgrade_script_sha256": PRODUCER.sha256(
                CONSUMER_SCRIPT.read_bytes()
            ),
        }
    )
    config["repository_authority"]["implementation_base_commit"] = "5" * 40
    return config


def _producer_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    config = producer_config()
    payloads = PRODUCER.produce(
        config,
        public_root=PUBLIC_ROOT,
        group_publication=GROUP_PUBLICATION,
        consumer_config_path=CONSUMER_CONFIG,
        consumer_script_path=CONSUMER_SCRIPT,
    )
    return config, payloads


@pytest.fixture(scope="module")
def real_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("g200-split-real")
    config, payloads = _producer_fixture(root)
    output = config["output_contract"]
    return {
        "config": config,
        "payloads": payloads,
        "private": json.loads(payloads[output["private_assignment_basename"]]),
        "audit": json.loads(payloads[output["aggregate_audit_basename"]]),
        "gate": json.loads(payloads[output["allowed_gate_basename"]]),
    }


def test_real_data_split_is_outcome_blind_nonempty_zero_leakage_go(
    real_result: Mapping[str, Any],
) -> None:
    private = real_result["private"]
    audit = real_result["audit"]
    assert audit["status"] == "GO_PASS_CONDITIONS_MET"
    assert audit["outcome_columns_read"] == []
    assert (
        audit["record_count"],
        audit["biological_group_node_count"],
        audit["connected_component_count"],
        audit["largest_component_group_count"],
    ) == (6547, 6544, 1936, 26)
    assert audit["edge_counts_by_reason"] == {
        PRODUCER.EDGE_GENE: 11505,
        PRODUCER.EDGE_HAMMING: 0,
        PRODUCER.EDGE_JACCARD: 279,
        "UNION_DISTINCT_EDGE_COUNT": 11505,
    }
    assert [item["group_count"] for item in audit["outer_fold_counts"]] == [
        1309,
        1309,
        1309,
        1309,
        1308,
    ]
    assert audit["all_outer_folds_nonempty"] is True
    assert audit["all_outer_train_inner_folds_nonempty"] is True
    assert audit["all_required_cross_fold_leakage_counts_zero"] is True
    assert all(
        value == 0
        for key, value in audit["outer_cross_fold_leakage_counts"].items()
        if key.endswith("_cross_fold_count")
    )
    assert all(
        value == 0
        for item in audit["inner_cross_fold_leakage_counts_by_outer"]
        for key, value in item.items()
        if key.endswith("_cross_fold_count")
    )
    assert private["assignment_commitment_sha256"] == audit[
        "assignment_commitment_sha256"
    ]
    assert private["group_count"] == 6544
    assert len(private["assignments"]) == 6544


def test_pass_gate_binds_assignment_root_and_actual_consumer_accepts(
    real_result: Mapping[str, Any],
) -> None:
    config = real_result["config"]
    gate = real_result["gate"]
    root = real_result["private"]["assignment_commitment_sha256"]
    assert gate["status"] == "PASS"
    assert gate["provenance"][PRODUCER.SPLIT_COMMITMENT_KEY] == root
    assert real_result["audit"]["consumer_actual_acceptance"] is True
    consumer_config = json.loads(CONSUMER_CONFIG.read_text(encoding="utf-8"))
    slot = next(
        item
        for item in consumer_config["evidence_contract"]["slots"]
        if item["slot_id"] == PRODUCER.GATE_ID
    )
    accepted = CONSUMER._validate_gate_record(
        PRODUCER.json_bytes(gate), slot, consumer_config
    )
    assert CONSUMER._slot_gate_pass(slot["slot_id"], accepted["facts"]) is True
    forbidden = {
        key.casefold()
        for key in config["output_contract"]["aggregate_forbidden_keys"]
    }
    assert not (
        forbidden
        & {
            key.casefold()
            for payload in (real_result["audit"], gate)
            for key in PRODUCER._walk_keys(payload)
        }
    )


def test_fixed_protocol_stops_when_required_fold_is_empty() -> None:
    config = producer_config()
    nodes = {
        str(index): {"record_count": 1, "sequence": "A" * 201, "gene": str(index)}
        for index in range(4)
    }
    with pytest.raises(PRODUCER.ProtocolStop) as error:
        PRODUCER.build_split(nodes, {}, config)
    assert error.value.audit["status"] == "STOP_PASS_CONDITIONS_NOT_MET"
    assert error.value.audit["all_outer_folds_nonempty"] is False


def test_publication_marker_is_last_exact_and_idempotent(
    tmp_path: Path,
    real_result: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = real_result["config"]
    payloads = real_result["payloads"]
    output = config["output_contract"]
    calls: list[str] = []
    real_write = PRODUCER._write_new

    def recording_write(path: Path, payload: bytes) -> None:
        calls.append(path.name)
        real_write(path, payload)

    monkeypatch.setattr(PRODUCER, "_write_new", recording_write)
    target = tmp_path / "published"
    assert PRODUCER.write_outputs(target, payloads, config) == "PUBLISHED"
    assert calls == output["exact_final_member_names"]
    assert sorted(path.name for path in target.iterdir()) == sorted(
        output["exact_final_member_names"]
    )
    marker = json.loads(
        (target / output["terminal_commit_marker"]).read_text(encoding="utf-8")
    )
    assert [item["name"] for item in marker["members"]] == output[
        "data_member_names"
    ]
    calls.clear()
    assert PRODUCER.write_outputs(target, payloads, config) == "EXISTING_EXACT"
    assert calls == []


def test_partial_or_mismatched_publication_stops_and_preserves(
    tmp_path: Path,
    real_result: Mapping[str, Any],
) -> None:
    config = real_result["config"]
    payloads = real_result["payloads"]
    output = config["output_contract"]
    target = tmp_path / "partial"
    target.mkdir()
    first = output["data_member_names"][0]
    sentinel = b"preserve this partial evidence\n"
    (target / first).write_bytes(sentinel)
    with pytest.raises(PRODUCER.ProducerError, match="partial or mismatched"):
        PRODUCER.write_outputs(target, payloads, config)
    assert (target / first).read_bytes() == sentinel
    assert [path.name for path in target.iterdir()] == [first]
