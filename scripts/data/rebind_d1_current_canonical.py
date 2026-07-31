#!/usr/bin/env python3
"""Create an immutable D1 rebind view for a user-approved current canonical store."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.d1_builder import _candidate_record
from data.utr_benchmark_v2.d1_builder import candidate_store_label_paths
from data.utr_benchmark_v2.records import validate_canonical_record
from scripts.execution.acceptance_semantics import validate_phase_acceptance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.link(temporary, path)
    temporary.unlink()


def _build_current_candidate_store(
    *,
    canonical_path: Path,
    output_path: Path,
    expected_record_ids_sha256: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = hashlib.sha256()
    candidate_ids = hashlib.sha256()
    count = 0
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with canonical_path.open("r", encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8", newline=""
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"canonical line {line_number} is not an object")
                validated = validate_canonical_record(record)
                record_id = str(validated["record_id"])
                ids.update((record_id + "\n").encode("utf-8"))
                candidate = _candidate_record(validated)
                leaks = candidate_store_label_paths(candidate)
                if leaks:
                    raise ValueError(f"candidate projection leaks labels at line {line_number}")
                candidate_ids.update((str(candidate["record_id"]) + "\n").encode("utf-8"))
                target.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        observed_ids = ids.hexdigest()
        if observed_ids != expected_record_ids_sha256:
            raise ValueError("current canonical record IDs differ from the accepted D1 universe")
        if candidate_ids.hexdigest() != observed_ids:
            raise ValueError("candidate projection record IDs differ from canonical universe")
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    reference = _file_ref(output_path)
    reference.update({"records": count, "record_ids_sha256": observed_ids})
    return reference


def rebind(
    *,
    historical_acceptance: Path,
    current_canonical: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite rebind root: {output_root}")
    historical_acceptance = historical_acceptance.resolve(strict=True)
    current_canonical = current_canonical.resolve(strict=True)
    acceptance = _read_json(historical_acceptance)
    semantic_errors = validate_phase_acceptance("D1", acceptance, require_pass=True)
    if semantic_errors:
        raise ValueError("historical D1 acceptance is not a semantic PASS")
    historical_root = Path(str(acceptance["stage_d1_root"])).resolve(strict=True)
    historical_build = _read_json(historical_root / "build_manifest.json")
    stores = historical_build["global_stores"]
    historical_label = stores["canonical_label_store"]
    expected_ids = str(historical_label["record_ids_sha256"])
    original_label_ref = _file_ref(historical_root / str(historical_label["path"]))
    output_root.mkdir(parents=True)
    canonical_view = output_root / "canonical" / "records_with_labels.jsonl"
    canonical_view.parent.mkdir(parents=True)
    os.link(current_canonical, canonical_view)
    current_label_ref = _file_ref(canonical_view)
    candidate_ref = _build_current_candidate_store(
        canonical_path=canonical_view,
        output_path=output_root / "candidate_store" / "candidates.jsonl",
        expected_record_ids_sha256=expected_ids,
    )
    current_label_ref.update(
        {"records": candidate_ref["records"], "record_ids_sha256": expected_ids}
    )
    rebind_build = dict(historical_build)
    rebind_stores = dict(stores)
    rebind_stores["canonical_label_store"] = {
        "path": "canonical/records_with_labels.jsonl",
        "bytes": current_label_ref["bytes"],
        "sha256": current_label_ref["sha256"],
        "records": current_label_ref["records"],
        "record_ids_sha256": expected_ids,
    }
    rebind_stores["sealed_label_free_candidate_store"] = {
        "path": "candidate_store/candidates.jsonl",
        "bytes": candidate_ref["bytes"],
        "sha256": candidate_ref["sha256"],
        "records": candidate_ref["records"],
        "record_ids_sha256": expected_ids,
    }
    rebind_build["global_stores"] = rebind_stores
    rebind_build["current_canonical_rebind"] = {
        "historical_acceptance": _file_ref(historical_acceptance),
        "historical_canonical": original_label_ref,
        "current_canonical": current_label_ref,
        "record_identity_preserved": True,
        "labels_or_sequences_emitted": False,
    }
    build_path = output_root / "build_manifest.json"
    _write_json_exclusive(build_path, rebind_build)
    rebind_acceptance = dict(acceptance)
    rebind_acceptance["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    rebind_acceptance["stage_d1_root"] = str(output_root.resolve())
    rebind_acceptance["global_store_validation"] = {
        "passed": True,
        "label_store": str(canonical_view.resolve()),
        "candidate_store": str((output_root / "candidate_store" / "candidates.jsonl").resolve()),
        "rebind_mode": "CURRENT_CANONICAL_USER_CONFIRMED",
        "record_identity_preserved": True,
    }
    rebind_acceptance["current_canonical_rebind"] = rebind_build["current_canonical_rebind"]
    acceptance_path = output_root / "acceptance.json"
    _write_json_exclusive(acceptance_path, rebind_acceptance)
    manifest = {
        "artifact_type": "d1_current_canonical_rebind.v1",
        "status": "PASS",
        "scientific_result_claimed": False,
        "historical_acceptance": _file_ref(historical_acceptance),
        "acceptance": _file_ref(acceptance_path),
        "build_manifest": _file_ref(build_path),
        "current_canonical": current_label_ref,
        "candidate_store": candidate_ref,
        "record_identity_preserved": True,
        "historical_canonical_sha256": original_label_ref["sha256"],
    }
    _write_json_exclusive(output_root / "rebind_manifest.json", manifest)
    _write_json_exclusive(output_root / "terminal.json", {"status": "PASS", "scientific_result_claimed": False})
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-acceptance", type=Path, required=True)
    parser.add_argument("--current-canonical", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = rebind(
            historical_acceptance=args.historical_acceptance,
            current_canonical=args.current_canonical,
            output_root=args.output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED_WITH_EVIDENCE", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "manifest": result["acceptance"]["path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
