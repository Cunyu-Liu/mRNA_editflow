from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest


STAGING = Path(__file__).resolve().parents[2]
SCRIPT = STAGING / "scripts" / "route_a_v3" / "qualify_gse200304_a1.py"
CONFIG = STAGING / "configs" / "route_a_v3_gse200304_a1_qualification.json"
SPEC = importlib.util.spec_from_file_location("qualify_gse200304_a1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALIFY
SPEC.loader.exec_module(QUALIFY)


def _gzip_tsv(header: Iterable[str], rows: Iterable[Iterable[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(list(header))
    writer.writerows(rows)
    return gzip.compress(stream.getvalue().encode("utf-8"), mtime=0)


def _sequence_pair(index: int) -> tuple[str, str]:
    if index < 6:
        source_group = index // 2
        candidate_variant = index % 2
    else:
        source_group = index - 3
        candidate_variant = 0
    alphabet = "ACGT"
    prefix: list[str] = []
    encoded = source_group
    for _ in range(8):
        prefix.append(alphabet[encoded % 4])
        encoded //= 4
    source = "".join(prefix) + "A" * 193
    position = 100 + candidate_variant
    replacement = "C" if candidate_variant == 0 else "G"
    candidate = source[:position] + replacement + source[position + 1 :]
    return source, candidate


def _full_design_payload() -> bytes:
    header = QUALIFY.EXPECTED_DESIGN_CONTRACT["exact_header"]

    def rows() -> Iterable[list[str]]:
        for index in range(6885):
            identifier = f"PAIR{index:05d}"
            wt, mutant = _sequence_pair(index)
            yield [identifier, "WT", wt, "AAA", "TTT", "FULL", f"{identifier}_WT"]
            yield [
                identifier,
                "Mutant",
                mutant,
                "AAA",
                "TTT",
                "FULL",
                f"{identifier}_Mutant",
            ]
        for index in range(66):
            identifier = f"CONTROL{index:03d}"
            yield [identifier, "Control", "A" * 201, "AAA", "TTT", "FULL", identifier]

    return _gzip_tsv(header, rows())


def _full_processed_payload() -> bytes:
    header = QUALIFY.expected_processed_header()

    def rows() -> Iterable[list[str]]:
        for index in range(6772):
            yield [f"PAIR{index:05d}", *("1" for _ in range(60))]

    return _gzip_tsv(header, rows())


def _full_small_payload() -> bytes:
    def rows() -> Iterable[list[str]]:
        for index in range(6120):
            identifier = f"PAIR{index:05d}"
            yield [f"{identifier}_WT", "1"]
            yield [f"{identifier}_Mutant", "1"]
        for index in range(6120, 6312):
            yield [f"PAIR{index:05d}_WT", "1"]
        for index in range(6312, 6537):
            yield [f"PAIR{index:05d}_Mutant", "1"]
        for index in range(47):
            yield [f"CONTROL{index:03d}", "1"]

    return _gzip_tsv(["Barcode", "Freq"], rows())


def _full_ivt_payload() -> bytes:
    header = QUALIFY.expected_ivt_header()

    def rows() -> Iterable[list[str]]:
        for index in range(6774):
            identifier = f"PAIR{index:05d}"
            yield [f"{identifier}_WT", *("1" for _ in range(30))]
            yield [f"{identifier}_Mutant", *("1" for _ in range(30))]

    return _gzip_tsv(header, rows())


def _tar_payload(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o640
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _make_two_commit_binding_repo(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "binding_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "qualifier-test@example.invalid")
    _git(repo, "config", "user.name", "Qualifier Test")
    authority_payloads = {
        "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md": b"contract authority\n",
        "docs/execution/route_a_v3_data_role_registry.yaml": b"data-role: frozen\n",
        "docs/execution/route_a_v3_decision_log.yaml": b"decision: frozen\n",
        "configs/route_a_v3_a1_qualification.json": b'{"a1":"frozen"}\n',
    }
    for relative, payload in authority_payloads.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    qualifier = repo / QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN["qualifier_path"]
    test_file = repo / QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN["test_path"]
    qualifier.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    qualifier.write_bytes(b"print('authority version')\n")
    test_file.write_bytes(b"def test_authority(): pass\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "authority")
    authority_commit = _git(repo, "rev-parse", "HEAD")

    staging_marker = repo / "docs" / "execution" / "STAGING_PARENT.txt"
    staging_marker.write_text("staging parent\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "staging parent")
    staging_parent = _git(repo, "rev-parse", "HEAD")

    qualifier_bytes = b"print('implementation version')\n"
    test_bytes = b"def test_implementation(): assert True\n"
    qualifier.write_bytes(qualifier_bytes)
    test_file.write_bytes(test_bytes)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    binding_config = repo / "configs" / QUALIFY.PROTOCOL_BASENAME
    binding_config.parent.mkdir(parents=True, exist_ok=True)
    binding_config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "config-only binding")
    current_head = _git(repo, "rev-parse", "HEAD")
    authority = {
        **QUALIFY.EXPECTED_AUTHORITY,
        "active_authority_commit": authority_commit,
        "staging_parent_head": staging_parent,
        "contract_sha256": _sha256(
            authority_payloads["docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md"]
        ),
        "data_role_registry_sha256": _sha256(
            authority_payloads[
                "docs/execution/route_a_v3_data_role_registry.yaml"
            ]
        ),
        "decision_log_sha256": _sha256(
            authority_payloads["docs/execution/route_a_v3_decision_log.yaml"]
        ),
        "a1_qualification_sha256": _sha256(
            authority_payloads["configs/route_a_v3_a1_qualification.json"]
        ),
    }
    binding = {
        **QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN,
        "status": "BOUND",
        "implementation_commit": implementation_commit,
        "qualifier_blob_sha256": _sha256(qualifier_bytes),
        "test_blob_sha256": _sha256(test_bytes),
    }
    return {
        "repo": repo,
        "authority": authority,
        "binding": binding,
        "qualifier": qualifier,
        "binding_config": binding_config,
        "implementation_commit": implementation_commit,
        "authority_commit": authority_commit,
        "staging_parent": staging_parent,
        "current_head": current_head,
    }


@pytest.fixture(scope="module")
def full_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("gse200304_full")
    root = base / "GSE200304"
    root.mkdir()
    (root / "GSE200302").mkdir()
    (root / "GSE200303").mkdir()
    (root / "GSE217530").mkdir()

    design = _full_design_payload()
    processed = _full_processed_payload()
    small = _full_small_payload()
    ivt = _full_ivt_payload()
    raw_tar = _tar_payload(
        {
            "GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz": design,
            "GSM6030637_log2_cpm_small_seq_on_plasmid.txt.gz": small,
        }
    )
    payload_by_id = {
        "GSE200302_DESIGN": design,
        "GSE200302_PROCESSED": processed,
        "GSE200303_RAW_TAR": raw_tar,
        "GSE200303_DESIGN": design,
        "GSE200303_SMALL_PLASMID": small,
        "GSE217530_IVT": ivt,
    }
    assets = []
    for production in QUALIFY.EXPECTED_ASSETS:
        asset = dict(production)
        payload = payload_by_id[asset["asset_id"]]
        asset["bytes"] = len(payload)
        asset["sha256"] = _sha256(payload)
        target = root / asset["relative_path"]
        target.write_bytes(payload)
        assets.append(asset)

    direct = {asset["asset_id"]: asset for asset in assets}
    manifest = {
        "files": [
            {
                "name": Path(direct["GSE200303_DESIGN"]["relative_path"]).name,
                "bytes": direct["GSE200303_DESIGN"]["bytes"],
                "sha256": direct["GSE200303_DESIGN"]["sha256"],
            },
            {
                "name": Path(direct["GSE200303_SMALL_PLASMID"]["relative_path"]).name,
                "bytes": direct["GSE200303_SMALL_PLASMID"]["bytes"],
                "sha256": direct["GSE200303_SMALL_PLASMID"]["sha256"],
            },
        ]
    }
    manifest_payload = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    (root / "manifest.json").write_bytes(manifest_payload)

    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol["implementation_binding"] = copy.deepcopy(
        QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN
    )
    protocol["unresolved_blockers"] = QUALIFY._expected_blockers_for_binding(
        protocol["implementation_binding"]
    )
    protocol["input_contract"]["data_root"] = str(root)
    protocol["input_contract"]["assets"] = assets
    protocol["input_contract"]["manifest_expected_bytes"] = len(manifest_payload)
    protocol["input_contract"]["manifest_expected_sha256"] = _sha256(manifest_payload)
    protocol_dir = base / "configs"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / QUALIFY.PROTOCOL_BASENAME
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    return {
        "base": base,
        "root": root,
        "assets": tuple(assets),
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": _sha256(manifest_payload),
    }


def _bind_fixture(monkeypatch: pytest.MonkeyPatch, fixture: Mapping[str, Any]) -> None:
    monkeypatch.setattr(QUALIFY, "EXPECTED_DATA_ROOT", Path(fixture["root"]))
    monkeypatch.setattr(QUALIFY, "EXPECTED_ASSETS", tuple(fixture["assets"]))
    monkeypatch.setattr(QUALIFY, "EXPECTED_MANIFEST_BYTES", fixture["manifest_bytes"])
    monkeypatch.setattr(QUALIFY, "EXPECTED_MANIFEST_SHA256", fixture["manifest_sha256"])


def _clone_fixture(
    tmp_path: Path, fixture: Mapping[str, Any]
) -> dict[str, Any]:
    root = tmp_path / "GSE200304"
    shutil.copytree(fixture["root"], root)
    protocol = copy.deepcopy(fixture["protocol"])
    protocol["input_contract"]["data_root"] = str(root)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    protocol_path = config_dir / QUALIFY.PROTOCOL_BASENAME
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "base": tmp_path,
        "root": root,
        "assets": fixture["assets"],
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "manifest_bytes": fixture["manifest_bytes"],
        "manifest_sha256": fixture["manifest_sha256"],
    }


def _success_payloads_for_fixture(fixture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, bytes] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for asset in fixture["assets"]:
        payload = (Path(fixture["root"]) / asset["relative_path"]).read_bytes()
        payloads[asset["asset_id"]] = payload
        provenance[asset["asset_id"]] = {
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
    protocol_path = Path(fixture["protocol_path"])
    return QUALIFY._build_success_payloads(
        protocol=fixture["protocol"],
        protocol_provenance={
            "sha256": fixture["protocol_sha256"],
            "bytes": len(protocol_path.read_bytes()),
            "launch_expected_sha256": fixture["protocol_sha256"],
        },
        manifest_payload=(Path(fixture["root"]) / "manifest.json").read_bytes(),
        asset_payloads=payloads,
        asset_provenance=provenance,
        implementation_binding_audit=QUALIFY._verify_implementation_binding(
            fixture["protocol"]["implementation_binding"],
            fixture["protocol"]["authority"],
            Path(fixture["base"]),
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_production_protocol_is_fixed_gap_only_and_records_current_public_bundle() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    QUALIFY._validate_protocol(protocol)
    assert protocol["protocol_status"] == "GO_QUALIFICATION_ONLY"
    assert protocol["activation_status"] == "NO_GO_ACTIVATION_NOW"
    assert protocol["qualification_status"] == "BLOCKED_NOT_QUALIFIED"
    assert protocol["scope"]["qualified"] is False
    assert protocol["scope"]["canonical_record_count"] == 0
    assert protocol["scope"]["training_allowed"] is False
    assert protocol["scope"]["model_selection_allowed"] is False
    assert protocol["a1_gate"] == QUALIFY.EXPECTED_A1_GATE
    binding = protocol["implementation_binding"]
    assert binding["status"] in {"UNKNOWN_NOT_ASSERTED", "BOUND"}
    assert protocol["unresolved_blockers"] == QUALIFY._expected_blockers_for_binding(binding)
    assert "PMC_TABLE_S2_S3_ABSENT" not in protocol["unresolved_blockers"]
    assert "ZENODO_V1_2_ARCHIVE_ABSENT_AND_EXPLICIT_CODE_LICENSE_UNKNOWN" not in protocol["unresolved_blockers"]
    assert all(
        item["production_server_status"] == "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE"
        and item["used_by_current_qualifier"] is False
        for item in protocol["expected_future_assets"]
    )
    lineage = protocol["paper_and_external_evidence"]["public_asset_bundle_lineage"]
    assert lineage["acquisition_manifest_sha256"] == "8318990d9e3b6a0e6265bf9d1e8bc20f56f0ecfd994e83d279e733258642100c"
    assert lineage["sha256sums_sha256"] == "20da85cd34f0574829392b5de1d7c48cc9782219847f56ccc07dffd579d79f15"
    assert lineage["publication_commit_sha256"] == "4742508195f28bf8c7ab1f7cb8bb0b68c32304f31b19c8f8979d098fa75786a5"
    ena = protocol["paper_and_external_evidence"]["ena_fastq_manifest_bundle_lineage"]
    assert ena["canonical_tsv_sha256"] == "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b"
    assert ena["summary_sha256"] == "f92f944c825a255f3f1fb50f48cbf0e701980b7895101c1a2a6699d4b190e1e4"
    assert ena["terminal_marker_sha256"] == "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b"
    assert ena["fastq_body_download_count"] == 0
    assert ena["fastq_md5_local_recomputation_status"] == "NOT_RUN"
    assert "FASTQ_CONTENT_INTEGRITY_NOT_VERIFIED" in protocol["unresolved_blockers"]
    assert "RAW_COUNTS_ABSENT" in protocol["unresolved_blockers"]
    assert "RAW_24_RUN_48_FASTQ_COUNTS_ABSENT" not in protocol["unresolved_blockers"]
    assert protocol["input_contract"]["manifest_expected_bytes"] == 1115
    assert protocol["input_contract"]["manifest_expected_sha256"] == "4a9a0b162f0731df6a5c15441b8984505e2ebaee260ad4e46f62636621125a8c"
    assert "access_log" in protocol["input_contract"]["forbidden_path_tokens"]


def test_protocol_structure_accepts_unknown_and_future_config_only_bound_binding() -> None:
    production = json.loads(CONFIG.read_text(encoding="utf-8"))
    unknown = copy.deepcopy(production)
    unknown["implementation_binding"] = copy.deepcopy(
        QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN
    )
    unknown["unresolved_blockers"] = QUALIFY._expected_blockers_for_binding(
        unknown["implementation_binding"]
    )
    QUALIFY._validate_protocol(unknown)

    bound = copy.deepcopy(production)
    bound["implementation_binding"] = {
        **QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN,
        "status": "BOUND",
        "implementation_commit": "1" * 40,
        "qualifier_blob_sha256": "2" * 64,
        "test_blob_sha256": "3" * 64,
    }
    bound["unresolved_blockers"] = QUALIFY._expected_blockers_for_binding(
        bound["implementation_binding"]
    )
    QUALIFY._validate_protocol(bound)
    assert QUALIFY.IMPLEMENTATION_BINDING_BLOCKER not in bound["unresolved_blockers"]


def test_two_commit_implementation_binding_success_and_fail_closed_variants(
    tmp_path: Path,
) -> None:
    fixture = _make_two_commit_binding_repo(tmp_path)
    verified = QUALIFY._verify_implementation_binding(
        fixture["binding"],
        fixture["authority"],
        fixture["repo"],
        running_script_path=fixture["qualifier"],
    )
    assert verified == {
        "status": "PASS_IMPLEMENTATION_BINDING",
        "verified": True,
        "implementation_commit": fixture["implementation_commit"],
        "current_head": fixture["current_head"],
        "clean_worktree": True,
        "active_authority_ancestor": True,
        "staging_parent_ancestor": True,
        "current_head_strict_descendant": True,
        "authority_blob_hashes_match": True,
        "implementation_blob_hashes_match": True,
        "running_script_matches_bound_blob": True,
        "post_implementation_change_set_is_config_only": True,
    }

    original_config = fixture["binding_config"].read_bytes()
    fixture["binding_config"].write_bytes(original_config + b"dirty")
    with pytest.raises(QUALIFY.ProtocolError, match="clean worktree"):
        QUALIFY._verify_implementation_binding(
            fixture["binding"],
            fixture["authority"],
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )
    fixture["binding_config"].write_bytes(original_config)
    assert _git(fixture["repo"], "status", "--porcelain=v1") == ""

    wrong_blob = copy.deepcopy(fixture["binding"])
    wrong_blob["test_blob_sha256"] = "0" * 64
    with pytest.raises(QUALIFY.ProtocolError, match="blob hash"):
        QUALIFY._verify_implementation_binding(
            wrong_blob,
            fixture["authority"],
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )

    wrong_a1_authority = copy.deepcopy(fixture["authority"])
    wrong_a1_authority["a1_qualification_sha256"] = "0" * 64
    with pytest.raises(QUALIFY.ProtocolError, match="authority blob hash"):
        QUALIFY._verify_implementation_binding(
            fixture["binding"],
            wrong_a1_authority,
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )

    implementation_tree = _git(
        fixture["repo"], "rev-parse", f"{fixture['implementation_commit']}^{{tree}}"
    )
    sibling_implementation = _git(
        fixture["repo"],
        "commit-tree",
        implementation_tree,
        "-p",
        fixture["authority_commit"],
        "-m",
        "implementation sibling without staging parent",
    )
    sibling_binding = copy.deepcopy(fixture["binding"])
    sibling_binding["implementation_commit"] = sibling_implementation
    with pytest.raises(QUALIFY.ProtocolError, match="staging parent is not a strict ancestor"):
        QUALIFY._verify_implementation_binding(
            sibling_binding,
            fixture["authority"],
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )

    broken_authority = copy.deepcopy(fixture["authority"])
    broken_authority["active_authority_commit"] = fixture["current_head"]
    with pytest.raises(QUALIFY.ProtocolError, match="not a strict ancestor"):
        QUALIFY._verify_implementation_binding(
            fixture["binding"],
            broken_authority,
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )

    not_strict = copy.deepcopy(fixture["binding"])
    not_strict["implementation_commit"] = fixture["current_head"]
    with pytest.raises(QUALIFY.ProtocolError, match="strict descendant"):
        QUALIFY._verify_implementation_binding(
            not_strict,
            fixture["authority"],
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )

    drifted_running_script = tmp_path / "drifted_running_qualifier.py"
    drifted_running_script.write_bytes(b"print('drift')\n")
    with pytest.raises(QUALIFY.ProtocolError, match="running qualifier bytes"):
        QUALIFY._verify_implementation_binding(
            fixture["binding"],
            fixture["authority"],
            fixture["repo"],
            running_script_path=drifted_running_script,
        )

    current_test = fixture["repo"] / fixture["binding"]["test_path"]
    current_test.write_bytes(current_test.read_bytes() + b"# forbidden later test drift\n")
    _git(fixture["repo"], "add", fixture["binding"]["test_path"])
    _git(fixture["repo"], "commit", "-m", "forbidden post-binding test drift")
    with pytest.raises(QUALIFY.ProtocolError, match="not config-only"):
        QUALIFY._verify_implementation_binding(
            fixture["binding"],
            fixture["authority"],
            fixture["repo"],
            running_script_path=fixture["qualifier"],
        )


def test_exact_generated_headers_and_frozen_table_counts() -> None:
    processed = QUALIFY.expected_processed_header()
    assert len(processed) == 61
    assert processed[0] == "barcode"
    assert processed[1:7] == [
        "80S_RNA_1_S2_WT",
        "80S_RNA_2_S6_WT",
        "80S_RNA_3_S10_WT",
        "80S_RNA_4_S14_WT",
        "80S_RNA_5_S18_WT",
        "80S_RNA_6_S22_WT",
    ]
    assert processed[-6:] == [
        "Total_RNA_1_S1_Mutant",
        "Total_RNA_2_S5_Mutant",
        "Total_RNA_3_S9_Mutant",
        "Total_RNA_4_S13_Mutant",
        "Total_RNA_5_S17_Mutant",
        "Total_RNA_6_S21_Mutant",
    ]
    ivt = QUALIFY.expected_ivt_header()
    assert len(ivt) == 31
    assert ivt[:7] == [
        "ids",
        "IVT_12hr_1_S43",
        "IVT_12hr_2_S44",
        "IVT_12hr_3_S45",
        "IVT_12hr_4_S46",
        "IVT_12hr_5_S47",
        "IVT_12hr_6_S48",
    ]
    assert ivt[-1] == "IVT_6hr_6_S42"
    assert QUALIFY.EXPECTED_DESIGN_CONTRACT["type_counts"] == {
        "WT": 6885,
        "Mutant": 6885,
        "Control": 66,
    }
    assert QUALIFY.EXPECTED_DESIGN_CONTRACT["distinct_wt_source_group_count"] == 6882
    assert QUALIFY.EXPECTED_DESIGN_CONTRACT["singleton_source_pool_count"] == 6879
    assert QUALIFY.EXPECTED_DESIGN_CONTRACT["two_candidate_source_pool_count"] == 3
    assert QUALIFY.EXPECTED_DESIGN_CONTRACT["ndcg_eligible_source_pool_count"] == 0
    assert QUALIFY.EXPECTED_SMALL_CONTRACT["complete_pair_count"] == 6120
    assert QUALIFY.EXPECTED_IVT_CONTRACT["complete_pair_count"] == 6774


def test_full_aggregate_gap_run_exact_counts_manifest_attrition_no_raw_payload_and_terminal_last(
    full_fixture: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _bind_fixture(monkeypatch, full_fixture)
    writes: list[str] = []
    original_write = QUALIFY._write_exclusive_at

    def track_write(
        directory_fd: int, name: str, payload: bytes, **kwargs: Any
    ) -> list[str]:
        writes.append(name)
        return original_write(directory_fd, name, payload, **kwargs)

    monkeypatch.setattr(QUALIFY, "_write_exclusive_at", track_write)
    output = Path(full_fixture["base"]) / "positive_bundle"
    result = QUALIFY.qualify_gse200304_a1(
        protocol_path=Path(full_fixture["protocol_path"]),
        protocol_sha256=str(full_fixture["protocol_sha256"]),
        data_root=Path(full_fixture["root"]),
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.SUCCESS_OUTCOME
    assert result["qualified"] is False
    assert result["canonical_record_count"] == 0
    assert result["terminal_marker_validated"] is True
    assert result["publication_state"] == "COMMITTED_ACCEPTED"
    assert result["committed"] is True
    assert result["accepted"] is True
    assert writes[-1] == QUALIFY.PUBLICATION_MARKER

    integrity = _read_json(output / "INPUT_INTEGRITY_AUDIT.json")
    mechanical = _read_json(output / "MECHANICAL_AUDIT.json")
    report = _read_json(output / "QUALIFICATION_REPORT.json")
    assert integrity["manifest"]["declared_entry_count"] == 2
    assert integrity["manifest"]["full_superseries_binding_complete"] is False
    assert integrity["manifest"]["trusted_as_complete"] is False
    assert integrity["unique_input_asset_count"] == 6
    assert integrity["tar_equivalence"]["byte_identical_direct_member_count"] == 2
    assert integrity["tar_equivalence"]["additional_unique_asset_count"] == 0
    assert mechanical["primary_design"]["row_count"] == 13836
    assert mechanical["primary_design"]["wt_row_count"] == 6885
    assert mechanical["primary_design"]["mutant_row_count"] == 6885
    assert mechanical["primary_design"]["control_row_count"] == 66
    assert mechanical["primary_design"]["unique_pair_count"] == 6885
    assert mechanical["primary_design"]["distinct_candidate_count"] == 6885
    assert mechanical["primary_design"]["distinct_wt_source_group_count"] == 6882
    assert mechanical["primary_design"]["singleton_source_pool_count"] == 6879
    assert mechanical["primary_design"]["two_candidate_source_pool_count"] == 3
    assert mechanical["primary_design"]["three_or_more_candidate_source_pool_count"] == 0
    assert mechanical["primary_design"]["ndcg_eligible_source_pool_count"] == 0
    assert mechanical["processed"]["row_count"] == 6772
    assert mechanical["processed"]["measurement_column_count"] == 60
    assert mechanical["processed"]["finite_nonmissing_numeric_cell_count"] == 406320
    assert mechanical["processed"]["outcome_blind_attrition_count"] == 113
    assert mechanical["processed"]["legacy_freq_used_as_endpoint"] is False
    assert mechanical["small_plasmid"]["row_count"] == 12704
    assert mechanical["small_plasmid"]["complete_pair_count"] == 6120
    assert mechanical["small_plasmid"]["wt_only_pair_count"] == 192
    assert mechanical["small_plasmid"]["mutant_only_pair_count"] == 225
    assert mechanical["small_plasmid"]["neither_pair_count"] == 348
    assert mechanical["small_plasmid"]["control_row_count"] == 47
    assert mechanical["small_plasmid"]["legacy_freq_used_as_endpoint"] is False
    assert mechanical["ivt"]["row_count"] == 13548
    assert mechanical["ivt"]["complete_pair_count"] == 6774
    assert mechanical["ivt"]["missing_both_pair_count"] == 111
    assert mechanical["join_audit"]["three_modal_auxiliary_join_pair_count"] == 6120
    assert mechanical["scope_audit"]["maximum_independent_study_count"] == 1
    assert report["qualified"] is False
    assert report["ordinary_study_contribution"] == 0
    assert report["a1_study_contribution"] == 0
    assert report["true_a2_study_contribution"] == 0
    assert report["canonical_record_count"] == 0
    assert report["unresolved_blockers"] == QUALIFY._expected_blockers_for_binding(
        full_fixture["protocol"]["implementation_binding"]
    )
    assert report["current_qualifier_integrates_public_asset_bundle"] is False
    assert report["current_qualifier_integrates_ena_fastq_manifest_bundle"] is False
    master = report["a1_master_report"]
    assert master["required_report_fields"] == QUALIFY.EXPECTED_A1_MASTER_REPORT_CONTRACT[
        "required_report_fields"
    ]
    assert master["study_recovery_method"] == "SOURCE_CANDIDATE_ENDPOINT_CONTEXT_AND_GROUP_AUDIT"
    assert master["metadata_only_recovery_allowed"] is False
    assert master["eligible_multi_candidate_pools"]["ndcg_eligible_pool_count"] == 0
    assert master["eligible_multi_candidate_pools"]["pairwise_only_pool_count"] == 3
    assert master["gene_groups"]["value"] is None
    assert master["post_dedup_effective_n"]["effective_n"] is None
    assert master["identification_audit"]["context"] == "UNKNOWN_NOT_ASSERTED"

    all_payloads = [integrity, mechanical, report]
    forbidden_exact_keys = {
        "row_id",
        "barcode",
        "sequence",
        "raw_row",
        "measurement_value",
        "label",
        "canonical_record",
    }
    assert not (set().union(*(set(_walk_keys(value)) for value in all_payloads)) & forbidden_exact_keys)
    combined = "\n".join(json.dumps(value, sort_keys=True) for value in all_payloads)
    assert "PAIR00000" not in combined
    assert "A" * 201 not in combined

    marker = _read_json(output / QUALIFY.PUBLICATION_MARKER)
    assert marker["committed"] is True
    assert marker["execution_outcome"] == QUALIFY.SUCCESS_OUTCOME
    expected_target = QUALIFY._absolute_without_resolving(output)
    assert marker["final_output_directory_name_sha256"] == _sha256(
        expected_target.name.encode("utf-8")
    )
    assert marker["final_output_target_sha256"] == _sha256(
        os.fspath(expected_target).encode("utf-8")
    )
    assert marker["bundle_member_names"] == sorted(
        [*QUALIFY.SUCCESS_JSON_FILES, QUALIFY.SHA256SUMS_FILENAME]
    )
    payloads = {
        name: _read_json(output / name) for name in QUALIFY.SUCCESS_JSON_FILES
    }
    accepted = QUALIFY.validate_published_bundle(output)
    assert accepted["publication_state"] == "COMMITTED_ACCEPTED"
    with pytest.raises(QUALIFY.PublicationContention):
        QUALIFY._publish_closed_bundle(output, payloads, outcome=QUALIFY.SUCCESS_OUTCOME)


@pytest.mark.parametrize("forbidden_token", ["restricted", "access_log"])
def test_forbidden_scope_fails_before_protocol_or_input_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbidden_token: str
) -> None:
    forbidden_root = tmp_path / forbidden_token / "GSE200304"
    monkeypatch.setattr(QUALIFY, "EXPECTED_DATA_ROOT", forbidden_root)
    calls: list[str] = []

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        calls.append("read")
        raise AssertionError("payload read occurred")

    monkeypatch.setattr(QUALIFY, "_read_path_verified_snapshot", forbidden_read)
    with pytest.raises(QUALIFY.ScopeViolation, match="forbidden"):
        QUALIFY.qualify_gse200304_a1(
            protocol_path=tmp_path / QUALIFY.PROTOCOL_BASENAME,
            protocol_sha256="0" * 64,
            data_root=forbidden_root,
            output_directory=tmp_path / "out",
        )
    assert calls == []


def test_hash_drift_publishes_only_closed_failure_bundle(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _clone_fixture(tmp_path, full_fixture)
    _bind_fixture(monkeypatch, fixture)
    processed = next(
        asset for asset in fixture["assets"] if asset["asset_id"] == "GSE200302_PROCESSED"
    )
    path = Path(fixture["root"]) / processed["relative_path"]
    path.write_bytes(path.read_bytes() + b"drift")
    output = tmp_path / "hash_drift_failure"
    result = QUALIFY.execute_qualification(
        protocol_path=Path(fixture["protocol_path"]),
        protocol_sha256=str(fixture["protocol_sha256"]),
        data_root=Path(fixture["root"]),
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    failure = _read_json(output / "FAILURE_REPORT.json")
    assert failure["failure_code"] == "INPUT_INTEGRITY_FAILED"
    assert failure["qualified"] is False
    assert (output / QUALIFY.PUBLICATION_MARKER).is_file()
    assert not (output / "QUALIFICATION_REPORT.json").exists()


def test_root_manifest_hash_drift_fails_before_manifest_parse(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _clone_fixture(tmp_path, full_fixture)
    _bind_fixture(monkeypatch, fixture)
    manifest = Path(fixture["root"]) / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    parse_calls: list[str] = []
    original = QUALIFY._audit_root_manifest

    def track_parse(payload: bytes, protocol: Mapping[str, Any]) -> dict[str, Any]:
        parse_calls.append("parse")
        return original(payload, protocol)

    monkeypatch.setattr(QUALIFY, "_audit_root_manifest", track_parse)
    output = tmp_path / "manifest_drift_failure"
    result = QUALIFY.execute_qualification(
        protocol_path=Path(fixture["protocol_path"]),
        protocol_sha256=str(fixture["protocol_sha256"]),
        data_root=Path(fixture["root"]),
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    assert parse_calls == []
    assert _read_json(output / "FAILURE_REPORT.json")["failure_code"] == "INPUT_INTEGRITY_FAILED"


def test_symlink_asset_fails_closed_without_following_target(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _clone_fixture(tmp_path, full_fixture)
    _bind_fixture(monkeypatch, fixture)
    design = next(asset for asset in fixture["assets"] if asset["asset_id"] == "GSE200302_DESIGN")
    candidate = Path(fixture["root"]) / design["relative_path"]
    target = Path(full_fixture["root"]) / design["relative_path"]
    candidate.unlink()
    candidate.symlink_to(target)
    output = tmp_path / "symlink_failure"
    result = QUALIFY.execute_qualification(
        protocol_path=Path(fixture["protocol_path"]),
        protocol_sha256=str(fixture["protocol_sha256"]),
        data_root=Path(fixture["root"]),
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    assert _read_json(output / "FAILURE_REPORT.json")["failure_code"] == "INPUT_INTEGRITY_FAILED"


def test_input_root_path_replacement_after_verified_snapshots_fails_closed(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _clone_fixture(tmp_path, full_fixture)
    _bind_fixture(monkeypatch, fixture)
    root = Path(fixture["root"])
    displaced = tmp_path / "displaced_ordinary_root"

    def replace_root() -> None:
        root.rename(displaced)
        shutil.copytree(displaced, root)

    monkeypatch.setattr(QUALIFY, "_POST_VERIFIED_INPUT_SNAPSHOT_HOOK", replace_root)
    output = tmp_path / "replacement_failure"
    result = QUALIFY.execute_qualification(
        protocol_path=Path(fixture["protocol_path"]),
        protocol_sha256=str(fixture["protocol_sha256"]),
        data_root=root,
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    assert _read_json(output / "FAILURE_REPORT.json")["failure_code"] == "INPUT_INTEGRITY_FAILED"


def test_same_descriptor_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "ordinary_root"
    root_path.mkdir()
    candidate = root_path / "asset.bin"
    candidate.write_bytes(b"A" * (2 << 20))
    binding = QUALIFY._open_directory_no_symlinks(root_path, label="mutation fixture")
    original_read = QUALIFY.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, count)
        if block and not mutated:
            mutated = True
            with candidate.open("ab") as handle:
                handle.write(b"mutation")
        return block

    monkeypatch.setattr(QUALIFY.os, "read", mutate_after_first_read)
    try:
        with pytest.raises(QUALIFY.ScopeViolation, match="changed"):
            QUALIFY._read_relative_verified_snapshot(
                binding, "asset.bin", label="same descriptor mutation fixture"
            )
    finally:
        os.close(binding.fd)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unknown_type"])
def test_design_duplicate_missing_pair_and_unknown_type_fail_closed(kind: str) -> None:
    wt, mutant = _sequence_pair(0)
    rows = [
        ["P0", "WT", wt, "A", "T", "F", "P0_WT"],
        ["P0", "Mutant", mutant, "A", "T", "F", "P0_Mutant"],
        ["C0", "Control", wt, "A", "T", "F", "C0"],
    ]
    contract = copy.deepcopy(QUALIFY.EXPECTED_DESIGN_CONTRACT)
    if kind == "duplicate":
        rows.insert(1, ["P0", "WT", wt, "A", "T", "F", "P0_WT"])
        contract.update(
            row_count=4,
            type_counts={"WT": 2, "Mutant": 1, "Control": 1},
            unique_pair_count=1,
            control_id_count=1,
        )
    elif kind == "missing":
        rows.insert(2, ["P1", "WT", wt, "A", "T", "F", "P1_WT"])
        contract.update(
            row_count=4,
            type_counts={"WT": 2, "Mutant": 1, "Control": 1},
            unique_pair_count=2,
            control_id_count=1,
        )
    else:
        rows[0][1] = "Unknown"
        contract.update(
            row_count=3,
            type_counts={"WT": 0, "Mutant": 1, "Control": 1},
            unique_pair_count=1,
            control_id_count=1,
        )
    payload = _gzip_tsv(contract["exact_header"], rows)
    with pytest.raises(QUALIFY.TableAuditError):
        QUALIFY._audit_design(payload, contract, label=f"{kind} design fixture")


@pytest.mark.parametrize("bad_value", ["", "nan", "inf", "not-a-number"])
def test_processed_missing_or_nonfinite_value_fails_closed(bad_value: str) -> None:
    contract = copy.deepcopy(QUALIFY.EXPECTED_PROCESSED_CONTRACT)
    contract["row_count"] = 1
    contract["outcome_blind_attrition_count"] = 0
    values = ["1"] * 60
    values[17] = bad_value
    payload = _gzip_tsv(QUALIFY.expected_processed_header(contract), [["P0", *values]])
    with pytest.raises(QUALIFY.TableAuditError):
        QUALIFY._audit_processed(payload, contract, {"P0"}, label="bad processed fixture")


def test_processed_freq_cannot_become_an_endpoint() -> None:
    contract = copy.deepcopy(QUALIFY.EXPECTED_PROCESSED_CONTRACT)
    contract["row_count"] = 1
    contract["outcome_blind_attrition_count"] = 0
    header = QUALIFY.expected_processed_header(contract)
    header[-1] = "Freq"
    payload = _gzip_tsv(header, [["P0", *("1" for _ in range(60))]])
    with pytest.raises(QUALIFY.TableAuditError):
        QUALIFY._audit_processed(payload, contract, {"P0"}, label="Freq endpoint fixture")


def test_single_auxiliary_arm_cannot_masquerade_as_complete_pair() -> None:
    contract = copy.deepcopy(QUALIFY.EXPECTED_SMALL_CONTRACT)
    contract.update(
        row_count=1,
        complete_pair_count=0,
        wt_only_pair_count=1,
        mutant_only_pair_count=0,
        neither_pair_count=0,
        control_row_count=0,
    )
    payload = _gzip_tsv(["Barcode", "Freq"], [["P0_WT", "1"]])
    state = QUALIFY._audit_small_plasmid(
        payload, contract, {"P0"}, set(), label="single-arm fixture"
    )
    assert state.aggregate["complete_pair_count"] == 0
    assert state.aggregate["wt_only_pair_count"] == 1
    falsely_complete = copy.deepcopy(contract)
    falsely_complete.update(complete_pair_count=1, wt_only_pair_count=0)
    with pytest.raises(QUALIFY.TableAuditError, match="arm completeness"):
        QUALIFY._audit_small_plasmid(
            payload, falsely_complete, {"P0"}, set(), label="false pair fixture"
        )


def test_corrupt_gzip_and_tar_and_tar_byte_mismatch_fail_closed() -> None:
    with pytest.raises(QUALIFY.TableAuditError, match="corrupt gzip"):
        QUALIFY._parse_gzip_tsv(b"not-gzip", label="corrupt fixture")
    direct = {
        "GSE200303_DESIGN": b"design",
        "GSE200303_SMALL_PLASMID": b"small",
    }
    with pytest.raises(QUALIFY.QualificationError, match="corrupt"):
        QUALIFY._audit_tar_equivalence(
            b"not-a-tar", direct, QUALIFY.EXPECTED_TAR_EQUIVALENCE
        )
    mismatched_tar = _tar_payload(
        {
            "GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz": b"different",
            "GSM6030637_log2_cpm_small_seq_on_plasmid.txt.gz": b"small",
        }
    )
    with pytest.raises(QUALIFY.QualificationError, match="byte-identical"):
        QUALIFY._audit_tar_equivalence(
            mismatched_tar, direct, QUALIFY.EXPECTED_TAR_EQUIVALENCE
        )


def test_extra_blocker_is_rejected_before_data_root_open(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _clone_fixture(tmp_path, full_fixture)
    _bind_fixture(monkeypatch, fixture)
    protocol = copy.deepcopy(fixture["protocol"])
    protocol["unresolved_blockers"].append("FREEFORM_EXTRA_BLOCKER")
    path = Path(fixture["protocol_path"])
    path.write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    opened: list[Path] = []
    original_open = QUALIFY._open_directory_no_symlinks

    def track_open(candidate: Path, *, label: str) -> Any:
        opened.append(Path(candidate))
        return original_open(candidate, label=label)

    monkeypatch.setattr(QUALIFY, "_open_directory_no_symlinks", track_open)
    with pytest.raises(QUALIFY.ProtocolError):
        QUALIFY.qualify_gse200304_a1(
            protocol_path=path,
            protocol_sha256=digest,
            data_root=Path(fixture["root"]),
            output_directory=tmp_path / "must_not_publish",
        )
    assert Path(fixture["root"]) not in opened


def test_integrity_asset_semantic_tampering_is_rejected_before_output_creation(
    tmp_path: Path,
    full_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_fixture(monkeypatch, full_fixture)
    valid = _success_payloads_for_fixture(full_fixture)
    mutations = {
        "accession": "GSE999999",
        "relative_locator": "GSE200302/not-the-frozen-asset.gz",
        "role": "UNFROZEN_ROLE",
        "format": "UNFROZEN_FORMAT",
        "bytes": valid["INPUT_INTEGRITY_AUDIT.json"]["assets"][0]["bytes"] + 1,
        "sha256": "0" * 64,
    }
    for index, (field, value) in enumerate(mutations.items()):
        tampered = copy.deepcopy(valid)
        tampered["INPUT_INTEGRITY_AUDIT.json"]["assets"][0][field] = value
        output = tmp_path / f"tampered_asset_{index}"
        with pytest.raises(QUALIFY.PublicationError, match="frozen semantics"):
            QUALIFY._publish_closed_bundle(
                output, tampered, outcome=QUALIFY.SUCCESS_OUTCOME
            )
        assert not output.exists()


def test_publisher_reports_partial_precommit_without_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fault(phase: str) -> None:
        if phase == "precommit_output_fsync":
            raise OSError("injected precommit fsync failure")

    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", fault)
    output = tmp_path / "partial_precommit"
    with pytest.raises(QUALIFY.PartialPrecommitError) as captured:
        QUALIFY._publish_closed_bundle(
            output,
            QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
            outcome=QUALIFY.FAILURE_OUTCOME,
        )
    assert captured.value.publication_state == "PARTIAL_PRECOMMIT"
    assert output.is_dir()
    assert not (output / QUALIFY.PUBLICATION_MARKER).exists()


@pytest.mark.parametrize(
    "phase", ["post_marker_validation", "post_marker_stat"]
)
def test_persistent_post_marker_acceptance_fault_is_committed_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    def fault(observed: str) -> None:
        if observed == phase:
            raise OSError("injected persistent post-marker acceptance fault")

    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", fault)
    output = tmp_path / f"committed_not_accepted_{phase}"
    with pytest.raises(QUALIFY.CommittedNotAcceptedError) as captured:
        QUALIFY._publish_closed_bundle(
            output,
            QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
            outcome=QUALIFY.FAILURE_OUTCOME,
        )
    error = captured.value
    assert error.publication_state == "COMMITTED_NOT_ACCEPTED"
    assert error.outcome == QUALIFY.FAILURE_OUTCOME
    assert (output / QUALIFY.PUBLICATION_MARKER).is_file()
    assert "POST_MARKER_ACCEPTANCE_PERSISTENT_FAILURE" in error.durability_warnings
    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", None)
    assert QUALIFY.validate_published_bundle(output)["accepted"] is True


def test_transient_post_marker_validation_fault_returns_committed_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fault(phase: str) -> None:
        nonlocal calls
        if phase == "post_marker_validation":
            calls += 1
            if calls == 1:
                raise OSError("injected transient validation fault")

    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", fault)
    output = tmp_path / "committed_validation_warning"
    result = QUALIFY._publish_closed_bundle(
        output,
        QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
        outcome=QUALIFY.FAILURE_OUTCOME,
    )
    assert result["publication_state"] == "COMMITTED_WITH_DURABILITY_WARNING"
    assert result["committed"] is True
    assert result["accepted"] is True
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    assert "POST_MARKER_ACCEPTANCE_RETRIED" in result["durability_warning_codes"]


@pytest.mark.parametrize(
    "phase",
    ["post_marker_output_fsync", "post_marker_parent_fsync", "terminal_marker_fsync"],
)
def test_post_marker_fsync_fault_returns_committed_durability_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    def fault(observed: str) -> None:
        if observed == phase:
            raise OSError("injected post-marker fsync fault")

    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", fault)
    output = tmp_path / f"committed_fsync_warning_{phase}"
    result = QUALIFY._publish_closed_bundle(
        output,
        QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
        outcome=QUALIFY.FAILURE_OUTCOME,
    )
    assert result["publication_state"] == "COMMITTED_WITH_DURABILITY_WARNING"
    assert result["committed"] is True
    assert result["accepted"] is True
    assert result["durability_warning"] is True


def test_post_marker_output_close_fault_does_not_skip_parent_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_phases: list[str] = []

    def fault(phase: str) -> None:
        if phase in {"post_marker_close_output", "post_marker_close_parent"}:
            close_phases.append(phase)
        if phase == "post_marker_close_output":
            raise OSError("injected output close fault")

    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", fault)
    output = tmp_path / "committed_close_warning"
    result = QUALIFY._publish_closed_bundle(
        output,
        QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
        outcome=QUALIFY.FAILURE_OUTCOME,
    )
    assert close_phases == ["post_marker_close_output", "post_marker_close_parent"]
    assert result["publication_state"] == "COMMITTED_WITH_DURABILITY_WARNING"
    assert "POST_MARKER_OUTPUT_CLOSE_WARNING" in result["durability_warning_codes"]


@pytest.mark.parametrize("operation", ["rename", "copy"])
def test_default_consumer_rejects_renamed_or_copied_committed_directory(
    tmp_path: Path, operation: str
) -> None:
    original = tmp_path / f"original_{operation}"
    QUALIFY._publish_closed_bundle(
        original,
        QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED"),
        outcome=QUALIFY.FAILURE_OUTCOME,
    )
    assert QUALIFY.validate_published_bundle(original)["accepted"] is True
    relocated = tmp_path / f"relocated_{operation}"
    if operation == "rename":
        original.rename(relocated)
    else:
        shutil.copytree(original, relocated)
    with pytest.raises(QUALIFY.PublicationError, match="exactly bind"):
        QUALIFY.validate_published_bundle(relocated)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_status",), "GO_ACTIVATION"),
        (("activation_status",), "GO"),
        (("qualification_status",), "QUALIFIED"),
        (("scope", "qualified"), True),
        (("scope", "training_allowed"), True),
        (("scope", "model_selection_allowed"), True),
        (("scope", "canonical_materialization_allowed"), True),
        (("scope", "canonical_record_count"), 1),
        (("a1_gate", "this_protocol_ordinary_study_contribution"), 1),
        (("model_results_may_change_this_protocol",), True),
    ],
)
def test_production_hard_block_cannot_be_toggled(
    path: tuple[str, ...], value: Any
) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    target: dict[str, Any] = protocol
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(QUALIFY.ProtocolError):
        QUALIFY._validate_protocol(protocol)


def test_closed_failure_schema_rejects_raw_or_freeform_fields() -> None:
    payloads = QUALIFY._failure_payload("INPUT_INTEGRITY_FAILED")
    payloads["FAILURE_REPORT.json"]["row_id"] = "PAIR00000"
    with pytest.raises(QUALIFY.ProtocolError):
        QUALIFY._validate_closed_payloads(payloads, outcome=QUALIFY.FAILURE_OUTCOME)
