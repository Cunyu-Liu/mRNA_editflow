from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import pytest


STAGING = Path(__file__).resolve().parents[2]
SCRIPT = STAGING / "scripts" / "route_a_v3" / "acquire_gse200304_fastq.py"
CONFIG = STAGING / "configs" / "route_a_v3_gse200304_fastq_acquisition.json"
LOCAL_AUTHORITY = STAGING.parent / "gse200304_ena_manifest"
SPEC = importlib.util.spec_from_file_location("acquire_gse200304_fastq", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ACQUIRE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ACQUIRE
SPEC.loader.exec_module(ACQUIRE)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        url: str,
        status: int,
        headers: Mapping[str, str],
        backend: "FakeBackend",
        fail_after: int | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = dict(headers)
        self._url = url
        self._backend = backend
        self._fail_after = fail_after
        self._delivered = 0
        self._closed_once = False

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int = -1) -> bytes:
        time.sleep(0.0005)
        if self._fail_after is not None:
            if self._delivered >= self._fail_after:
                raise OSError("synthetic interrupted transfer")
            remaining_before_failure = self._fail_after - self._delivered
            if amount < 0 or amount > remaining_before_failure:
                amount = remaining_before_failure
        chunk = super().read(amount)
        self._delivered += len(chunk)
        return chunk

    def close(self) -> None:
        if not self._closed_once:
            self._closed_once = True
            self._backend.response_closed()
        super().close()


class FakeTransport:
    def __init__(self, backend: "FakeBackend") -> None:
        self._backend = backend

    def open(self, url: str, *, offset: int, timeout_seconds: int) -> FakeResponse:
        return self._backend.open(url, offset=offset, timeout_seconds=timeout_seconds)


class FakeBackend:
    def __init__(self, payload_by_url: Mapping[str, bytes]) -> None:
        self.payload_by_url = dict(payload_by_url)
        self.fail_after_by_url: dict[str, int] = {}
        self.wrong_content_range_urls: set[str] = set()
        self.corrupt_urls: set[str] = set()
        self.calls: list[tuple[str, int, int]] = []
        self.active_responses = 0
        self.maximum_active_responses = 0
        self._lock = threading.Lock()

    def factory(self) -> FakeTransport:
        return FakeTransport(self)

    def open(self, url: str, *, offset: int, timeout_seconds: int) -> FakeResponse:
        payload = self.payload_by_url[url]
        body = payload[offset:]
        if url in self.corrupt_urls and body:
            body = bytes([body[0] ^ 1]) + body[1:]
        status = 206 if offset else 200
        headers = {"Content-Length": str(len(payload) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
            if url in self.wrong_content_range_urls:
                headers["Content-Range"] = f"bytes {offset + 1}-{len(payload) - 1}/{len(payload)}"
        with self._lock:
            self.calls.append((url, offset, timeout_seconds))
            self.active_responses += 1
            self.maximum_active_responses = max(
                self.maximum_active_responses, self.active_responses
            )
        return FakeResponse(
            body,
            url=url,
            status=status,
            headers=headers,
            backend=self,
            fail_after=self.fail_after_by_url.get(url),
        )

    def response_closed(self) -> None:
        with self._lock:
            self.active_responses -= 1


@pytest.fixture
def synthetic_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    output_root = tmp_path / "mnt" / "cunyuliu" / "mrna_xeditflow_routea_v3" / "data" / "A1" / "GSE200304"
    output_root.mkdir(parents=True)
    bundle_root = output_root / ACQUIRE.EXPECTED_MANIFEST_BUNDLE_BASENAME
    bundle_root.mkdir()

    payload_by_url: dict[str, bytes] = {}
    manifest_lines = ["\t".join(ACQUIRE.EXPECTED_MANIFEST_HEADER)]
    total_bytes = 0
    for index, run in enumerate(ACQUIRE.EXPECTED_RUN_ACCESSIONS):
        for mate in (1, 2):
            payload = (f"{run}:mate={mate}:payload={index}:" + "ACGT" * (index % 5 + 1)).encode()
            url = ACQUIRE._expected_url(run, mate)
            payload_by_url[url] = payload
            total_bytes += len(payload)
            manifest_lines.append(
                "\t".join(
                    (
                        run,
                        str(mate),
                        url,
                        url.removeprefix("https://"),
                        str(len(payload)),
                        _md5(payload),
                    )
                )
            )
    manifest_payload = ("\n".join(manifest_lines) + "\n").encode()
    manifest_sha = _sha256(manifest_payload)
    (bundle_root / ACQUIRE.EXPECTED_MANIFEST_FILENAME).write_bytes(manifest_payload)

    source_report_payload = b"synthetic descriptor-bound ENA source report\n"
    source_report_sha = _sha256(source_report_payload)
    summary = {
        "schema_version": "route_a_v3_ena_fastq_manifest_summary.v1",
        "dataset_accession": ACQUIRE.DATASET_ACCESSION,
        "bioproject_accession": ACQUIRE.BIOPROJECT_ACCESSION,
        "canonical_manifest": {
            "path": ACQUIRE.EXPECTED_MANIFEST_FILENAME,
            "bytes": len(manifest_payload),
            "sha256": manifest_sha,
        },
        "aggregate": {
            "run_count": ACQUIRE.EXPECTED_RUN_COUNT,
            "paired_fastq_file_count": ACQUIRE.EXPECTED_FILE_COUNT,
            "total_fastq_bytes": total_bytes,
        },
        "verification": {
            "ena_two_files_per_run": True,
            "filename_run_and_mate_binding": True,
            "fastq_file_bodies_downloaded": 0,
            "repository_md5_recomputed_from_fastq_bodies": False,
        },
    }
    summary_payload = _json_bytes(summary)
    summary_sha = _sha256(summary_payload)
    source_sums_payload = (
        f"{manifest_sha}  {ACQUIRE.EXPECTED_MANIFEST_FILENAME}\n"
        f"{source_report_sha}  {ACQUIRE.EXPECTED_SOURCE_REPORT_FILENAME}\n"
        f"{summary_sha}  {ACQUIRE.EXPECTED_SUMMARY_FILENAME}\n"
    ).encode("ascii")
    source_sums_sha = _sha256(source_sums_payload)
    (bundle_root / ACQUIRE.EXPECTED_SOURCE_REPORT_FILENAME).write_bytes(
        source_report_payload
    )
    (bundle_root / ACQUIRE.EXPECTED_SUMMARY_FILENAME).write_bytes(summary_payload)
    (bundle_root / ACQUIRE.EXPECTED_SOURCE_SHA256SUMS_FILENAME).write_bytes(
        source_sums_payload
    )
    source_marker = {
        "schema_version": "route_a_v3_publication_commit.v1",
        "record_type": "GSE200304_ENA_FASTQ_MANIFEST_PUBLICATION_COMMIT",
        "dataset_accession": ACQUIRE.DATASET_ACCESSION,
        "bioproject_accession": ACQUIRE.BIOPROJECT_ACCESSION,
        "publication_status": "MANIFEST_COMMITTED_FASTQ_NOT_DOWNLOADED",
        "member_set": sorted(
            (
                ACQUIRE.EXPECTED_MANIFEST_FILENAME,
                ACQUIRE.EXPECTED_SOURCE_REPORT_FILENAME,
                ACQUIRE.EXPECTED_SUMMARY_FILENAME,
                ACQUIRE.EXPECTED_SOURCE_SHA256SUMS_FILENAME,
            )
        ),
        "member_sha256": {
            ACQUIRE.EXPECTED_MANIFEST_FILENAME: manifest_sha,
            ACQUIRE.EXPECTED_SOURCE_REPORT_FILENAME: source_report_sha,
            ACQUIRE.EXPECTED_SUMMARY_FILENAME: summary_sha,
            ACQUIRE.EXPECTED_SOURCE_SHA256SUMS_FILENAME: source_sums_sha,
        },
        "run_count": ACQUIRE.EXPECTED_RUN_COUNT,
        "paired_fastq_file_count": ACQUIRE.EXPECTED_FILE_COUNT,
        "total_fastq_bytes": total_bytes,
        "fastq_bodies_downloaded": 0,
        "repository_md5_recomputed_count": 0,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }
    marker_payload = _json_bytes(source_marker)
    marker_sha = _sha256(marker_payload)
    (bundle_root / ACQUIRE.EXPECTED_SOURCE_COMMIT_FILENAME).write_bytes(marker_payload)

    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol["input_authority"]["bundle_directory"] = str(bundle_root)
    protocol["input_authority"]["terminal_marker"]["bytes"] = len(marker_payload)
    protocol["input_authority"]["terminal_marker"]["sha256"] = marker_sha
    protocol["input_authority"]["canonical_manifest"]["bytes"] = len(manifest_payload)
    protocol["input_authority"]["canonical_manifest"]["sha256"] = manifest_sha
    protocol["input_authority"]["canonical_manifest"]["total_fastq_bytes"] = total_bytes
    protocol["input_authority"]["manifest_summary"]["bytes"] = len(summary_payload)
    protocol["input_authority"]["manifest_summary"]["sha256"] = summary_sha
    protocol["input_authority"]["source_file_report"]["bytes"] = len(
        source_report_payload
    )
    protocol["input_authority"]["source_file_report"]["sha256"] = source_report_sha
    protocol["input_authority"]["source_sha256sums"]["bytes"] = len(
        source_sums_payload
    )
    protocol["input_authority"]["source_sha256sums"]["sha256"] = source_sums_sha
    protocol["implementation_binding"]["status"] = "BOUND"
    protocol["implementation_binding"]["implementation_commit"] = "1" * 40
    protocol["implementation_binding"]["implementation_script_sha256"] = "2" * 64
    protocol["output_contract"]["base_directory"] = str(output_root)
    protocol["integrity_gate"]["required_verified_total_bytes"] = total_bytes
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / ACQUIRE.PROTOCOL_BASENAME
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    monkeypatch.setattr(ACQUIRE, "EXPECTED_MANIFEST_BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(ACQUIRE, "EXPECTED_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ACQUIRE, "EXPECTED_SOURCE_COMMIT_BYTES", len(marker_payload))
    monkeypatch.setattr(ACQUIRE, "EXPECTED_SOURCE_COMMIT_SHA256", marker_sha)
    monkeypatch.setattr(ACQUIRE, "EXPECTED_MANIFEST_BYTES", len(manifest_payload))
    monkeypatch.setattr(ACQUIRE, "EXPECTED_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(
        ACQUIRE, "EXPECTED_SOURCE_REPORT_BYTES", len(source_report_payload)
    )
    monkeypatch.setattr(ACQUIRE, "EXPECTED_SOURCE_REPORT_SHA256", source_report_sha)
    monkeypatch.setattr(ACQUIRE, "EXPECTED_SUMMARY_BYTES", len(summary_payload))
    monkeypatch.setattr(ACQUIRE, "EXPECTED_SUMMARY_SHA256", summary_sha)
    monkeypatch.setattr(
        ACQUIRE, "EXPECTED_SOURCE_SHA256SUMS_BYTES", len(source_sums_payload)
    )
    monkeypatch.setattr(
        ACQUIRE, "EXPECTED_SOURCE_SHA256SUMS_SHA256", source_sums_sha
    )
    monkeypatch.setattr(ACQUIRE, "EXPECTED_TOTAL_BYTES", total_bytes)
    return {
        "output_root": output_root,
        "bundle_root": bundle_root,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "manifest_payload": manifest_payload,
        "manifest_sha": manifest_sha,
        "marker_sha": marker_sha,
        "implementation_commit": "1" * 40,
        "implementation_script_sha256": "2" * 64,
        "binding_commit": "3" * 40,
        "payload_by_url": payload_by_url,
        "total_bytes": total_bytes,
    }


def _output(case: Mapping[str, Any], suffix: str) -> Path:
    return case["output_root"] / f"GSE200304_FASTQ_ACQUISITION_{suffix}"


def _run(
    case: Mapping[str, Any],
    output: Path,
    backend: FakeBackend,
    *,
    resume: bool = False,
    workers: int = 2,
    chunk_bytes: int = 7,
    capacity_bytes: int | None = None,
) -> dict[str, Any]:
    def fake_implementation_verifier(
        protocol_path: Path,
        protocol_sha256: str,
        protocol: Mapping[str, Any],
    ) -> dict[str, Any]:
        del protocol_path, protocol
        return {
            "status": "BOUND",
            "binding_mode": "TWO_COMMIT_NON_SELF_REFERENTIAL",
            "implementation_commit": case["implementation_commit"],
            "implementation_script_sha256": case[
                "implementation_script_sha256"
            ],
            "binding_commit": case["binding_commit"],
            "protocol_sha256": protocol_sha256,
            "worktree_and_index_clean": True,
        }

    available = (
        capacity_bytes
        if capacity_bytes is not None
        else case["total_bytes"] + ACQUIRE.MINIMUM_CAPACITY_SAFETY_MARGIN_BYTES + 1
    )
    return ACQUIRE.execute(
        case["protocol_path"],
        output,
        resume=resume,
        workers=workers,
        timeout_seconds=17,
        chunk_bytes=chunk_bytes,
        transport_factory=backend.factory,
        implementation_verifier=fake_implementation_verifier,
        capacity_probe=lambda directory_fd: available,
    )


def test_frozen_production_authority_hashes_and_config_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    configured_authority = Path(config["input_authority"]["bundle_directory"])
    authority = configured_authority if configured_authority.is_dir() else LOCAL_AUTHORITY
    canonical = authority / ACQUIRE.EXPECTED_MANIFEST_FILENAME
    marker = authority / ACQUIRE.EXPECTED_SOURCE_COMMIT_FILENAME
    assert _sha256(canonical.read_bytes()) == ACQUIRE.EXPECTED_MANIFEST_SHA256
    assert canonical.stat().st_size == ACQUIRE.EXPECTED_MANIFEST_BYTES
    assert _sha256(marker.read_bytes()) == ACQUIRE.EXPECTED_SOURCE_COMMIT_SHA256
    assert marker.stat().st_size == ACQUIRE.EXPECTED_SOURCE_COMMIT_BYTES
    assert (
        config["input_authority"]["bundle_directory"]
        == str(ACQUIRE.EXPECTED_MANIFEST_BUNDLE_ROOT)
    )
    assert config["download_policy"]["default_workers"] == 2
    assert config["download_policy"]["maximum_workers"] == 2
    assert config["integrity_gate"]["training_allowed"] is False
    assert config["integrity_gate"]["next_phase_authorized"] is False


def test_success_verifies_all_48_and_writes_terminal_marker_last(
    synthetic_case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "SUCCESS0001")
    write_order: list[str] = []
    original = ACQUIRE._create_exclusive_at

    def recording_create(directory_fd: int, name: str, payload: bytes, *, mode: int = 0o640) -> None:
        original(directory_fd, name, payload, mode=mode)
        write_order.append(name)

    monkeypatch.setattr(ACQUIRE, "_create_exclusive_at", recording_create)
    result = _run(synthetic_case, output, backend, workers=2)

    assert result["success"] is True
    assert result["publication_status"] == "FASTQ_ACQUISITION_COMMITTED"
    assert result["verified_files"] == 48
    assert result["verified_bytes"] == synthetic_case["total_bytes"]
    assert backend.maximum_active_responses <= 2
    assert backend.maximum_active_responses == 2
    assert write_order[-1] == ACQUIRE.TERMINAL_MARKER_FILENAME
    assert write_order.count(ACQUIRE.TERMINAL_MARKER_FILENAME) == 1
    assert not list(output.glob("*.part"))

    marker = json.loads((output / ACQUIRE.TERMINAL_MARKER_FILENAME).read_text())
    assert marker["verified_file_count"] == 48
    assert marker["repository_md5_verified_count"] == 48
    assert marker["training_allowed"] is False
    integrity = json.loads((output / ACQUIRE.INTEGRITY_MANIFEST_FILENAME).read_text())
    assert len(integrity["files"]) == 48
    for row in integrity["files"]:
        payload = synthetic_case["payload_by_url"][row["source_url"]]
        assert (output / row["filename"]).read_bytes() == payload
        assert row["ena_repository_md5"] == _md5(payload)
        assert row["local_sha256"] == _sha256(payload)


def test_failure_keeps_bound_partial_and_status_without_commit(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    first_url = next(iter(synthetic_case["payload_by_url"]))
    backend.fail_after_by_url[first_url] = 5
    output = _output(synthetic_case, "FAILPART01")
    result = _run(synthetic_case, output, backend, workers=1, chunk_bytes=3)

    assert result["success"] is False
    assert result["publication_status"] == "FAILED_NOT_COMMITTED"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()
    part = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz.part"
    binding = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz.transfer.json"
    assert part.read_bytes() == synthetic_case["payload_by_url"][first_url][:5]
    transfer = json.loads(binding.read_text())
    assert transfer["url"] == first_url
    assert transfer["repository_md5"] == _md5(
        synthetic_case["payload_by_url"][first_url]
    )
    status = json.loads((output / ACQUIRE.PROGRESS_STATUS_FILENAME).read_text())
    assert status["current_status"] == "FAILED_NOT_COMMITTED"
    assert status["terminal_commit_present"] is False
    assert status["training_allowed"] is False


def test_exact_range_resume_recovers_and_preserves_attempt_history(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    first_url = next(iter(synthetic_case["payload_by_url"]))
    backend.fail_after_by_url[first_url] = 6
    output = _output(synthetic_case, "RESUME0001")
    first = _run(synthetic_case, output, backend, workers=1, chunk_bytes=4)
    assert first["success"] is False
    del backend.fail_after_by_url[first_url]

    second = _run(
        synthetic_case, output, backend, resume=True, workers=2, chunk_bytes=5
    )
    assert second["success"] is True
    assert any(url == first_url and offset == 6 for url, offset, _ in backend.calls)
    assert not list(output.glob("*.part"))
    status = json.loads((output / ACQUIRE.PROGRESS_STATUS_FILENAME).read_text())
    assert len(status["attempts"]) == 2
    assert status["attempts"][0]["status"] == "FAILED_NOT_COMMITTED"
    assert (
        status["attempts"][1]["status"]
        == "VERIFIED_ALL_FILES_READY_FOR_TERMINAL_COMMIT"
    )


def test_wrong_content_range_fails_closed_and_keeps_partial(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    first_url = next(iter(synthetic_case["payload_by_url"]))
    backend.fail_after_by_url[first_url] = 4
    output = _output(synthetic_case, "BADRANGE01")
    assert _run(synthetic_case, output, backend, workers=1)["success"] is False
    del backend.fail_after_by_url[first_url]
    backend.wrong_content_range_urls.add(first_url)
    part = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz.part"
    before = part.read_bytes()

    resumed = _run(synthetic_case, output, backend, resume=True, workers=1)
    assert resumed["success"] is False
    assert part.read_bytes() == before
    assert resumed["failures"][0]["error_code"] == "ResumeError"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()


def test_tampered_transfer_binding_disables_resume_before_network(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    first_url = next(iter(synthetic_case["payload_by_url"]))
    backend.fail_after_by_url[first_url] = 3
    output = _output(synthetic_case, "BADBIND001")
    assert _run(synthetic_case, output, backend, workers=1)["success"] is False
    del backend.fail_after_by_url[first_url]
    binding_path = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz.transfer.json"
    binding = json.loads(binding_path.read_text())
    binding["url"] = "https://example.org/forbidden.fastq.gz"
    binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    calls_before = len(backend.calls)

    resumed = _run(synthetic_case, output, backend, resume=True, workers=1)
    assert resumed["success"] is False
    assert len(backend.calls) == calls_before
    assert resumed["publication_status"] == "PREFLIGHT_FAILED_NOT_COMMITTED"
    assert resumed["error_code"] == "ResumeError"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()


def test_existing_corrupt_completed_file_is_never_overwritten(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    urls = list(synthetic_case["payload_by_url"])
    backend.fail_after_by_url[urls[1]] = 2
    output = _output(synthetic_case, "NOOVERW001")
    first = _run(synthetic_case, output, backend, workers=1)
    assert first["success"] is False
    completed = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz"
    original_size = completed.stat().st_size
    corrupt = b"X" * original_size
    completed.write_bytes(corrupt)
    del backend.fail_after_by_url[urls[1]]
    calls_before = len(backend.calls)

    resumed = _run(synthetic_case, output, backend, resume=True, workers=1)
    assert resumed["success"] is False
    assert completed.read_bytes() == corrupt
    assert len(backend.calls) == calls_before
    assert resumed["publication_status"] == "PREFLIGHT_FAILED_NOT_COMMITTED"
    assert resumed["error_code"] == "IntegrityError"


def test_committed_resume_is_idempotent_and_uses_no_transport(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "IDEMPOT001")
    assert _run(synthetic_case, output, backend)["success"] is True
    calls_before = list(backend.calls)
    second = _run(synthetic_case, output, backend, resume=True)
    assert second["success"] is True
    assert second["publication_status"] == "ALREADY_COMMITTED_VERIFIED"
    assert backend.calls == calls_before


@pytest.mark.parametrize(
    ("field", "value"),
    (("verified_total_bytes", 0), ("member_set", []), ("member_sha256", {})),
)
def test_committed_validation_rejects_terminal_marker_self_report(
    synthetic_case: Mapping[str, Any], field: str, value: Any
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, f"BADMARK{field[:3].upper()}01")
    assert _run(synthetic_case, output, backend)["success"] is True
    marker_path = output / ACQUIRE.TERMINAL_MARKER_FILENAME
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    calls_before = list(backend.calls)
    with pytest.raises(ACQUIRE.AcquisitionError):
        _run(synthetic_case, output, backend, resume=True)
    assert backend.calls == calls_before


@pytest.mark.parametrize("member_kind", ("transfer", "integrity", "sums"))
def test_committed_validation_rejects_semantically_tampered_members(
    synthetic_case: Mapping[str, Any], member_kind: str
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, f"BADSEM{member_kind[:3].upper()}1")
    assert _run(synthetic_case, output, backend)["success"] is True
    if member_kind == "transfer":
        path = output / (
            f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz.transfer.json"
        )
        document = json.loads(path.read_text())
        document["expected_bytes"] += 1
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    elif member_kind == "integrity":
        path = output / ACQUIRE.INTEGRITY_MANIFEST_FILENAME
        document = json.loads(path.read_text())
        document["verified_total_bytes"] = 0
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    else:
        path = output / ACQUIRE.SHA256SUMS_FILENAME
        lines = path.read_text().splitlines()
        lines[0] = "0" * 64 + lines[0][64:]
        path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ACQUIRE.AcquisitionError):
        _run(synthetic_case, output, backend, resume=True)


def test_committed_validation_rehashes_fastq_bytes_md5_and_sha256(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "BADFASTQ01")
    assert _run(synthetic_case, output, backend)["success"] is True
    target = output / f"{ACQUIRE.EXPECTED_RUN_ACCESSIONS[0]}_1.fastq.gz"
    target.write_bytes(b"Q" * target.stat().st_size)
    calls_before = list(backend.calls)
    with pytest.raises(ACQUIRE.IntegrityError, match="MD5"):
        _run(synthetic_case, output, backend, resume=True)
    assert backend.calls == calls_before


def test_post_promotion_final_name_is_rehashed(
    synthetic_case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "POSTLINK01")
    original = ACQUIRE._promote_verified_part
    mutated = False

    def promote_then_mutate(directory_fd: int, entry: Any) -> None:
        nonlocal mutated
        original(directory_fd, entry)
        if not mutated:
            mutated = True
            target = output / entry.filename
            target.write_bytes(b"X" * entry.expected_bytes)

    monkeypatch.setattr(ACQUIRE, "_promote_verified_part", promote_then_mutate)
    result = _run(synthetic_case, output, backend, workers=1)
    assert result["success"] is False
    assert result["failures"][0]["error_code"] == "IntegrityError"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()


def test_reused_final_is_rehashed_after_capacity_preflight(
    synthetic_case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    urls = list(synthetic_case["payload_by_url"])
    backend.fail_after_by_url[urls[1]] = 2
    output = _output(synthetic_case, "REUSETOC01")
    assert _run(synthetic_case, output, backend, workers=1)["success"] is False
    del backend.fail_after_by_url[urls[1]]
    original = ACQUIRE._remove_duplicate_part_after_link_crash
    mutated = False

    def cleanup_then_mutate(directory_fd: int, entry: Any) -> None:
        nonlocal mutated
        original(directory_fd, entry)
        if not mutated:
            mutated = True
            target = output / entry.filename
            target.write_bytes(b"Y" * entry.expected_bytes)

    monkeypatch.setattr(
        ACQUIRE, "_remove_duplicate_part_after_link_crash", cleanup_then_mutate
    )
    result = _run(synthetic_case, output, backend, resume=True, workers=1)
    assert result["success"] is False
    assert result["failures"][0]["error_code"] == "IntegrityError"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()


def test_terminal_precommit_rehash_detects_late_mutation(
    synthetic_case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "LATEHASH01")
    original = ACQUIRE._reverify_final_results

    def mutate_then_reverify(directory_fd: int, results: Any) -> Any:
        first = results[0].entry
        (output / first.filename).write_bytes(b"Z" * first.expected_bytes)
        return original(directory_fd, results)

    monkeypatch.setattr(ACQUIRE, "_reverify_final_results", mutate_then_reverify)
    result = _run(synthetic_case, output, backend)
    assert result["success"] is False
    assert result["publication_status"] == "PUBLICATION_FAILED_NOT_COMMITTED"
    assert result["error_code"] == "IntegrityError"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()


def test_publication_failure_is_statused_and_never_claims_commit(
    synthetic_case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "PUBFAIL001")
    original = ACQUIRE._create_exclusive_at

    def injected_failure(
        directory_fd: int, name: str, payload: bytes, *, mode: int = 0o640
    ) -> None:
        if name == ACQUIRE.SHA256SUMS_FILENAME:
            raise ACQUIRE.PublicationError("synthetic SHA256SUMS publication failure")
        original(directory_fd, name, payload, mode=mode)

    monkeypatch.setattr(ACQUIRE, "_create_exclusive_at", injected_failure)
    result = _run(synthetic_case, output, backend)
    assert result["success"] is False
    assert result["publication_status"] == "PUBLICATION_FAILED_NOT_COMMITTED"
    assert not (output / ACQUIRE.TERMINAL_MARKER_FILENAME).exists()
    status = json.loads((output / ACQUIRE.PROGRESS_STATUS_FILENAME).read_text())
    assert status["current_status"] == "PUBLICATION_FAILED_NOT_COMMITTED"
    assert status["terminal_commit_present"] is False
    integrity_before = (output / ACQUIRE.INTEGRITY_MANIFEST_FILENAME).read_bytes()
    final_status_before = (output / ACQUIRE.FINAL_STATUS_FILENAME).read_bytes()

    monkeypatch.setattr(ACQUIRE, "_create_exclusive_at", original)
    resumed = _run(synthetic_case, output, backend, resume=True)
    assert resumed["success"] is True
    assert resumed["publication_status"] == "FASTQ_ACQUISITION_COMMITTED"
    assert (output / ACQUIRE.INTEGRITY_MANIFEST_FILENAME).read_bytes() == integrity_before
    assert (output / ACQUIRE.FINAL_STATUS_FILENAME).read_bytes() == final_status_before


def test_capacity_gate_fails_before_network_and_records_truth(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "NOSPACE001")
    result = _run(
        synthetic_case,
        output,
        backend,
        capacity_bytes=synthetic_case["total_bytes"],
    )
    assert result["success"] is False
    assert result["publication_status"] == "PREFLIGHT_FAILED_NOT_COMMITTED"
    assert result["error_code"] == "CapacityError"
    assert backend.calls == []
    capacity = result["capacity_gate"]
    assert capacity["remaining_bytes"] == synthetic_case["total_bytes"]
    assert capacity["required_available_bytes"] > capacity["available_bytes"]
    progress = json.loads((output / ACQUIRE.PROGRESS_STATUS_FILENAME).read_text())
    assert progress["attempts"][-1]["capacity_gate"] == capacity


def test_source_bundle_exact_member_set_and_summary_hash_are_enforced(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    extra = synthetic_case["bundle_root"] / "UNEXPECTED.txt"
    extra.write_text("unexpected\n")
    output = _output(synthetic_case, "BADBUNDL01")
    with pytest.raises(ACQUIRE.ProtocolError, match="member set"):
        _run(synthetic_case, output, backend)
    assert not output.exists()
    assert backend.calls == []


def test_source_summary_is_descriptor_hash_bound(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    summary_path = synthetic_case["bundle_root"] / ACQUIRE.EXPECTED_SUMMARY_FILENAME
    summary_path.write_bytes(summary_path.read_bytes() + b"tamper")
    output = _output(synthetic_case, "BADSUMM001")
    with pytest.raises(ACQUIRE.ProtocolError, match="source bundle member"):
        _run(synthetic_case, output, backend)
    assert not output.exists()
    assert backend.calls == []


def test_resume_rejects_malformed_progress_schema_before_network(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    first_url = next(iter(synthetic_case["payload_by_url"]))
    backend.fail_after_by_url[first_url] = 3
    output = _output(synthetic_case, "BADPROG001")
    assert _run(synthetic_case, output, backend, workers=1)["success"] is False
    del backend.fail_after_by_url[first_url]
    progress_path = output / ACQUIRE.PROGRESS_STATUS_FILENAME
    progress = json.loads(progress_path.read_text())
    progress["record_type"] = "UNBOUND_STATUS"
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    calls_before = list(backend.calls)
    with pytest.raises(ACQUIRE.ResumeError, match="progress status"):
        _run(synthetic_case, output, backend, resume=True, workers=1)
    assert backend.calls == calls_before


def test_unknown_implementation_binding_fails_before_git_or_network(
    synthetic_case: Mapping[str, Any]
) -> None:
    protocol = json.loads(synthetic_case["protocol_path"].read_text())
    protocol["implementation_binding"]["status"] = "UNKNOWN_NOT_ASSERTED"
    protocol["implementation_binding"]["implementation_commit"] = "UNKNOWN_NOT_ASSERTED"
    protocol["implementation_binding"]["implementation_script_sha256"] = (
        "UNKNOWN_NOT_ASSERTED"
    )
    with pytest.raises(ACQUIRE.ProtocolError, match="separate binding commit"):
        ACQUIRE.verify_implementation_binding(
            synthetic_case["protocol_path"], "a" * 64, protocol
        )


def test_manifest_url_allowlist_and_closed_shape_are_enforced(
    synthetic_case: Mapping[str, Any]
) -> None:
    text = synthetic_case["manifest_payload"].decode()
    malicious = text.replace(
        "https://ftp.sra.ebi.ac.uk/", "https://example.org/", 1
    ).encode()
    with pytest.raises(ACQUIRE.ManifestError, match="official ENA object path"):
        ACQUIRE.parse_manifest(malicious)

    missing_row = ("\n".join(text.splitlines()[:-1]) + "\n").encode()
    with pytest.raises(ACQUIRE.ManifestError, match="exactly 48 rows"):
        ACQUIRE.parse_manifest(missing_row)


def test_source_manifest_hash_mismatch_fails_before_output_or_network(
    synthetic_case: Mapping[str, Any]
) -> None:
    manifest_path = synthetic_case["bundle_root"] / ACQUIRE.EXPECTED_MANIFEST_FILENAME
    manifest_path.write_bytes(synthetic_case["manifest_payload"] + b"tamper")
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "BADHASH001")
    with pytest.raises(ACQUIRE.ManifestError, match="byte count|SHA256"):
        _run(synthetic_case, output, backend)
    assert not output.exists()
    assert backend.calls == []


def test_worker_cap_and_output_scope_reject_before_acquisition(
    synthetic_case: Mapping[str, Any]
) -> None:
    backend = FakeBackend(synthetic_case["payload_by_url"])
    output = _output(synthetic_case, "WORKERS001")
    with pytest.raises(ACQUIRE.ProtocolError, match="between 1 and 2"):
        _run(synthetic_case, output, backend, workers=3)
    assert not output.exists()
    assert backend.calls == []

    outside = synthetic_case["output_root"].parent / "GSE200304_FASTQ_ACQUISITION_OUTSIDE01"
    with pytest.raises(ACQUIRE.ScopeViolation, match="one direct child"):
        _run(synthetic_case, outside, backend)
    assert not outside.exists()


def test_no_subprocess_or_shell_execution_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "subprocess.run(" in source
    assert 'GIT_BINARY = "/usr/bin/git"' in source
    assert "urllib.request.ProxyHandler({})" in source
    assert "os.system(" not in source
    assert "shell=True" not in source
    assert "eval(" not in source
    assert "exec(" not in source
