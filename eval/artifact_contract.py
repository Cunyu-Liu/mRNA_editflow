"""Shared run-mode, provenance, oracle, and paper-artifact contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.split_contract import (
    SPLIT_ROLES,
    SplitRoleError,
    VerifiedSplitContract,
    build_split_provenance,
    load_and_verify_split_manifest,
    select_role_records,
    sha256_file,
    transcript_id_digest,
)

RUN_MODES = ("development", "paper")
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"


class ArtifactContractError(ValueError):
    """Base class for scientific artifact contract failures."""


class RunModeError(ArtifactContractError):
    pass


class ArtifactNamespaceError(ArtifactContractError):
    pass


class ArtifactProvenanceError(ArtifactContractError):
    pass


class OracleContractError(ArtifactContractError):
    pass


def normalize_run_mode(run_mode: str) -> str:
    mode = str(run_mode).lower()
    if mode not in RUN_MODES:
        raise RunModeError(f"run_mode must be one of {RUN_MODES}, got {run_mode!r}")
    return mode


def require_paper_cli_inputs(
    *,
    run_mode: str,
    split_manifest: Optional[str],
    split_role: Optional[str],
    allowed_roles: Sequence[str],
    oracle_manifest: Optional[str] = None,
    require_oracle: bool = False,
) -> None:
    mode = normalize_run_mode(run_mode)
    if mode != "paper":
        return
    if not split_manifest:
        raise RunModeError("paper mode requires --split-manifest")
    if split_role not in allowed_roles:
        raise SplitRoleError(
            f"paper mode requires --split-role in {tuple(allowed_roles)}, got {split_role!r}"
        )
    if require_oracle and not oracle_manifest:
        raise OracleContractError("paper functional path requires --oracle-manifest")


def canonical_digest(value: object) -> str:
    if is_dataclass(value):
        value = asdict(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def records_identity(records: Sequence[MRNARecord]) -> dict[str, object]:
    canonical = [record.to_dict() for record in records]
    return {
        "records_sha256": canonical_digest(canonical),
        "records_count": len(records),
        "selected_count": len(records),
        "selected_transcript_id_digest": transcript_id_digest(
            [record.transcript_id for record in records]
        ),
        "split_manifest_sha256": None,
        "split_schema_version": None,
        "split_role": None,
        "role_idx_sha256": None,
        "dataset_id": "unrestricted_or_synthetic_records",
        "block_reasons": ["verified_split_manifest_missing"],
    }


def prepare_scientific_records(
    records: Sequence[MRNARecord],
    *,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    allowed_roles: Sequence[str] = SPLIT_ROLES,
) -> tuple[list[MRNARecord], dict[str, object]]:
    """Select a verified role in paper mode or label unrestricted development data."""
    mode = normalize_run_mode(run_mode)
    for role in allowed_roles:
        if role not in SPLIT_ROLES:
            raise SplitRoleError(f"invalid allowed split role {role!r}")
    if mode == "paper":
        if split_contract is None:
            raise RunModeError("paper mode requires --split-manifest / VerifiedSplitContract")
        if split_role not in allowed_roles:
            raise SplitRoleError(
                f"paper mode requires role in {tuple(allowed_roles)}, got {split_role!r}"
            )
        if not split_contract.paper_eligible:
            raise ArtifactProvenanceError(
                "paper mode requires a paper-eligible split contract; blockers: "
                + ", ".join(split_contract.block_reasons)
            )
        selected = select_role_records(records, split_contract, str(split_role))
        provenance = build_split_provenance(split_contract, str(split_role))
    elif split_contract is not None or split_role is not None:
        if split_contract is None or split_role not in SPLIT_ROLES:
            raise SplitRoleError("development role selection requires both contract and valid role")
        selected = select_role_records(records, split_contract, str(split_role))
        provenance = build_split_provenance(split_contract, str(split_role))
    else:
        selected = list(records)
        provenance = records_identity(selected)
    provenance = dict(provenance)
    provenance.update(
        {
            "run_mode": mode,
            "claim_tier": "paper" if mode == "paper" else "development_only",
            "paper_eligible": mode == "paper",
        }
    )
    if mode == "development":
        provenance["paper_eligible"] = False
    return selected, provenance


def _contains_namespace(path: Path, namespace: str) -> bool:
    parts = path.resolve().parts
    return any(
        parts[index] == "benchmark" and parts[index + 1] == namespace
        for index in range(len(parts) - 1)
    )


def validate_output_namespace(path: str, run_mode: str) -> str:
    mode = normalize_run_mode(run_mode)
    resolved = Path(path).resolve()
    if mode == "paper" and not _contains_namespace(resolved, "paper"):
        raise ArtifactNamespaceError("paper mode output must be under benchmark/paper/")
    if mode == "development" and _contains_namespace(resolved, "paper"):
        raise ArtifactNamespaceError("development output cannot use benchmark/paper/")
    return str(resolved)


def validate_report_output_namespaces(
    report: Mapping[str, object], paths: Sequence[Optional[str]]
) -> None:
    """Constrain only eligible paper outputs; blocked diagnostics may live in docs."""
    if report.get("claim_tier") == "development_only":
        mode = "development"
    elif report.get("paper_eligible") is True:
        mode = "paper"
    else:
        return
    for path in paths:
        if path:
            validate_output_namespace(path, mode)


def write_paper_report_sidecars(
    report: Mapping[str, object], paths: Sequence[Optional[str]]
) -> list[str]:
    if report.get("paper_eligible") is not True:
        return []
    metadata = report.get("scientific_validity")
    if not isinstance(metadata, Mapping):
        raise ArtifactProvenanceError("eligible paper report lacks scientific_validity")
    return [write_provenance_sidecar(path, metadata) for path in paths if path]


def load_and_verify_oracle_manifest(path: Optional[str], *, run_mode: str) -> dict[str, object]:
    mode = normalize_run_mode(run_mode)
    if path is None:
        if mode == "paper":
            raise OracleContractError("paper functional scoring requires --oracle-manifest")
        return {
            "oracle_type": "heuristic_development_oracle",
            "independent": False,
            "manifest_sha256": None,
            "artifact_sha256": None,
            "paper_permitted": False,
        }
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise OracleContractError("oracle manifest schema_version must equal 1")
    oracle_type = str(payload.get("oracle_type", ""))
    independent = payload.get("independent") is True
    source = payload.get("source")
    independence_statement = payload.get("independence_statement")
    raw_artifact = payload.get("artifact_path")
    if not oracle_type or not source or not isinstance(raw_artifact, str):
        raise OracleContractError("oracle manifest requires oracle_type, source, and artifact_path")
    artifact_path = Path(raw_artifact)
    if not artifact_path.is_absolute():
        manifest_root = manifest_path.parent.resolve()
        artifact_path = (manifest_root / artifact_path).resolve()
        try:
            artifact_path.relative_to(manifest_root)
        except ValueError as exc:
            raise OracleContractError(
                "relative oracle artifact path escapes the manifest directory"
            ) from exc
    artifact_path = artifact_path.resolve()
    expected_sha = str(payload.get("artifact_sha256", ""))
    if not artifact_path.is_file() or sha256_file(artifact_path) != expected_sha:
        raise OracleContractError("oracle artifact SHA mismatch")
    oracle_type_lower = oracle_type.lower()
    heuristic = "heuristic" in oracle_type_lower or "localtranslationoracle" in oracle_type_lower
    if mode == "paper" and not (
        isinstance(independence_statement, str) and independence_statement.strip()
    ):
        raise OracleContractError(
            "paper oracle manifest requires a non-empty independence_statement"
        )
    if mode == "paper" and (not independent or heuristic):
        raise OracleContractError("paper mode forbids heuristic or non-independent oracle contracts")
    return {
        "oracle_type": oracle_type,
        "independent": independent,
        "source": source,
        "independence_statement": independence_statement,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_sha,
        "paper_permitted": independent and not heuristic,
    }


def code_identity(paths: Iterable[str]) -> dict[str, object]:
    rows = []
    for raw in sorted(set(str(item) for item in paths)):
        path = Path(raw).resolve()
        if path.is_file():
            rows.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "git_commit": None,
        "source_digest": canonical_digest(rows),
        "source_files": rows,
    }


def build_run_metadata(
    *,
    run_mode: str,
    data_provenance: Mapping[str, object],
    config: Optional[object] = None,
    code_paths: Sequence[str] = (),
    training_seed: Optional[int] = None,
    decoder_seed: Optional[int] = None,
    oracle: Optional[Mapping[str, object]] = None,
    upstream: Optional[Mapping[str, object]] = None,
    extra_block_reasons: Sequence[str] = (),
    functional_claim: bool = False,
) -> dict[str, object]:
    mode = normalize_run_mode(run_mode)
    block_reasons = list(data_provenance.get("block_reasons", []))
    block_reasons.extend(str(reason) for reason in extra_block_reasons if reason)
    if mode == "development" and "development_mode" not in block_reasons:
        block_reasons.append("development_mode")
    paper_eligible = mode == "paper" and not block_reasons
    return {
        **dict(data_provenance),
        "run_mode": mode,
        "claim_tier": "paper" if mode == "paper" else "development_only",
        "paper_eligible": paper_eligible,
        "block_reasons": block_reasons,
        "config_sha256": canonical_digest(config) if config is not None else None,
        "code_identity": code_identity(code_paths),
        "oracle": dict(oracle) if oracle is not None else None,
        "training_seed": training_seed,
        "decoder_seed": decoder_seed,
        "upstream": dict(upstream or {}),
        "functional_claim": bool(functional_claim),
    }


def provenance_digest(metadata: Mapping[str, object]) -> str:
    return canonical_digest(metadata)


def write_provenance_sidecar(artifact_path: str, metadata: Mapping[str, object]) -> str:
    artifact = Path(artifact_path).resolve()
    if not artifact.is_file():
        raise ArtifactProvenanceError(
            f"cannot bind provenance before artifact exists: {artifact}"
        )
    sidecar = str(artifact) + PROVENANCE_SIDECAR_SUFFIX
    Path(sidecar).parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "artifact_binding": {
                    "path": str(artifact),
                    "sha256": sha256_file(artifact),
                },
                "scientific_validity": dict(metadata),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return sidecar


def load_artifact_provenance(path: str) -> dict[str, object]:
    artifact_path = Path(path).resolve()
    candidates = [Path(str(artifact_path) + PROVENANCE_SIDECAR_SUFFIX)]
    if artifact_path.suffix == ".json":
        candidates.append(artifact_path)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, Mapping):
            metadata = payload.get("scientific_validity")
            if isinstance(metadata, Mapping):
                return dict(metadata)
    raise ArtifactProvenanceError(f"artifact lacks scientific_validity provenance: {path}")


def verify_provenance_compatibility(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    require_same_role: bool,
) -> None:
    keys = ["records_sha256", "split_manifest_sha256"]
    if require_same_role:
        keys.extend(["split_role", "role_idx_sha256", "selected_transcript_id_digest"])
    mismatches = [key for key in keys if expected.get(key) != actual.get(key)]
    if mismatches:
        raise ArtifactProvenanceError(
            "artifact provenance mismatch for: " + ", ".join(mismatches)
        )


def upstream_data_provenance(
    paths: Sequence[str],
    *,
    run_mode: str,
    require_same_role: bool = True,
) -> dict[str, object]:
    mode = normalize_run_mode(run_mode)
    loaded = []
    missing = []
    for path in paths:
        try:
            loaded.append(
                verify_paper_artifact(path)
                if mode == "paper"
                else load_artifact_provenance(path)
            )
        except ArtifactProvenanceError:
            missing.append(path)
    if mode == "paper" and missing:
        raise ArtifactProvenanceError(
            "paper chained artifact lacks provenance: " + ", ".join(missing)
        )
    if mode == "paper" and any(
        item.get("run_mode") != "paper" or item.get("paper_eligible") is not True
        for item in loaded
    ):
        raise ArtifactProvenanceError("paper chained artifacts must all be paper eligible")
    if loaded:
        base = loaded[0]
        for other in loaded[1:]:
            verify_provenance_compatibility(base, other, require_same_role=require_same_role)
            if mode == "paper" and canonical_digest(base.get("oracle")) != canonical_digest(
                other.get("oracle")
            ):
                raise ArtifactProvenanceError("paper chained artifacts use different oracles")
        keys = (
            "dataset_id",
            "records_path",
            "records_sha256",
            "records_count",
            "split_manifest_sha256",
            "split_manifest_path",
            "split_schema_version",
            "split_role",
            "role_idx_sha256",
            "role_idx_path",
            "selected_count",
            "selected_transcript_id_digest",
            "cluster_assignment_path",
            "cluster_assignment_sha256",
            "selected_cluster_digest",
            "oracle",
            "block_reasons",
        )
        return {key: base.get(key) for key in keys}
    return {
        "dataset_id": "unverified_upstream_artifacts",
        "records_path": None,
        "records_sha256": None,
        "records_count": None,
        "split_manifest_sha256": None,
        "split_manifest_path": None,
        "split_schema_version": None,
        "split_role": None,
        "role_idx_sha256": None,
        "role_idx_path": None,
        "selected_count": None,
        "selected_transcript_id_digest": None,
        "cluster_assignment_path": None,
        "cluster_assignment_sha256": None,
        "selected_cluster_digest": None,
        "oracle": None,
        "block_reasons": ["upstream_provenance_missing"],
    }


def _verify_code_identity(metadata: Mapping[str, object]) -> None:
    identity = metadata.get("code_identity")
    if not isinstance(identity, Mapping) or not identity.get("source_digest"):
        raise ArtifactProvenanceError("paper evidence lacks deterministic code identity")
    source_files = identity.get("source_files")
    if not isinstance(source_files, Sequence) or isinstance(source_files, (str, bytes)) or not source_files:
        raise ArtifactProvenanceError("paper code identity requires source file hashes")
    rows = []
    for item in source_files:
        if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
            raise ArtifactProvenanceError("paper code identity contains a malformed source row")
        source_path = Path(str(item["path"])).resolve()
        if not source_path.is_file() or sha256_file(source_path) != item.get("sha256"):
            raise ArtifactProvenanceError(f"paper source identity changed: {source_path}")
        rows.append({"path": str(source_path), "sha256": str(item["sha256"])})
    if canonical_digest(sorted(rows, key=lambda row: row["path"])) != identity.get("source_digest"):
        raise ArtifactProvenanceError("paper source identity digest is inconsistent")


def _verify_split_metadata(metadata: Mapping[str, object]) -> VerifiedSplitContract:
    manifest_path = metadata.get("split_manifest_path")
    role = metadata.get("split_role")
    if not isinstance(manifest_path, str) or role not in SPLIT_ROLES:
        raise ArtifactProvenanceError("paper evidence lacks verifiable split manifest path/role")
    try:
        contract = load_and_verify_split_manifest(manifest_path)
    except Exception as exc:
        raise ArtifactProvenanceError(f"paper split contract no longer verifies: {exc}") from exc
    if not contract.paper_eligible:
        raise ArtifactProvenanceError("paper evidence references a non-paper split contract")
    expected = build_split_provenance(contract, str(role))
    verify_provenance_compatibility(expected, metadata, require_same_role=True)
    for key in (
        "records_path",
        "role_idx_path",
        "cluster_assignment_sha256",
        "selected_cluster_digest",
    ):
        if expected.get(key) != metadata.get(key):
            raise ArtifactProvenanceError(f"paper split provenance mismatch for: {key}")
    return contract


def _verify_oracle_metadata(metadata: Mapping[str, object]) -> None:
    if metadata.get("functional_claim") is not True:
        return
    oracle = metadata.get("oracle")
    if not isinstance(oracle, Mapping) or oracle.get("paper_permitted") is not True:
        raise OracleContractError("paper functional artifact lacks a permitted oracle")
    manifest_path = oracle.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise OracleContractError("paper oracle provenance lacks manifest_path")
    verified = load_and_verify_oracle_manifest(manifest_path, run_mode="paper")
    for key in (
        "manifest_sha256",
        "artifact_sha256",
        "oracle_type",
        "source",
        "independence_statement",
    ):
        if verified.get(key) != oracle.get(key):
            raise OracleContractError(f"paper oracle provenance mismatch for: {key}")


def verify_paper_checkpoint(path: str, test_provenance: Mapping[str, object]) -> dict[str, object]:
    import torch

    resolved = Path(path).resolve()
    if not _contains_namespace(resolved, "paper"):
        raise ArtifactNamespaceError("paper checkpoint must reside under benchmark/paper/")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ArtifactProvenanceError("checkpoint payload is not a mapping")
    metadata = payload.get("scientific_validity")
    if not isinstance(metadata, Mapping):
        raise ArtifactProvenanceError("paper evaluation requires checkpoint train provenance")
    if (
        metadata.get("run_mode") != "paper"
        or metadata.get("claim_tier") != "paper"
        or metadata.get("paper_eligible") is not True
        or metadata.get("split_role") != "train"
        or metadata.get("block_reasons")
    ):
        raise ArtifactProvenanceError("paper checkpoint must carry verified train-role provenance")
    if "config" not in payload or canonical_digest(payload.get("config")) != metadata.get(
        "config_sha256"
    ):
        raise ArtifactProvenanceError("paper checkpoint config digest does not match payload")
    _verify_split_metadata(metadata)
    _verify_code_identity(metadata)
    verify_provenance_compatibility(test_provenance, metadata, require_same_role=False)
    return dict(metadata)


def verify_paper_artifact(path: str) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not _contains_namespace(resolved, "paper"):
        raise ArtifactNamespaceError("paper artifact must reside under benchmark/paper/")
    sidecar = Path(str(resolved) + PROVENANCE_SIDECAR_SUFFIX)
    if not sidecar.is_file():
        raise ArtifactProvenanceError("paper artifact requires a bound provenance sidecar")
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            envelope = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactProvenanceError("paper artifact sidecar is invalid JSON") from exc
    if not isinstance(envelope, Mapping):
        raise ArtifactProvenanceError("paper artifact sidecar must be an object")
    binding = envelope.get("artifact_binding")
    metadata = envelope.get("scientific_validity")
    if not isinstance(binding, Mapping) or not isinstance(metadata, Mapping):
        raise ArtifactProvenanceError("paper artifact sidecar lacks binding or provenance")
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ArtifactProvenanceError("paper artifact sidecar is bound to a different path")
    if binding.get("sha256") != sha256_file(resolved):
        raise ArtifactProvenanceError("paper artifact content SHA changed after provenance binding")
    required = (
        "records_sha256",
        "split_manifest_sha256",
        "split_role",
        "role_idx_sha256",
        "selected_transcript_id_digest",
        "config_sha256",
        "code_identity",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ArtifactProvenanceError("paper artifact missing provenance: " + ", ".join(missing))
    if (
        metadata.get("run_mode") != "paper"
        or metadata.get("claim_tier") != "paper"
        or metadata.get("paper_eligible") is not True
    ):
        raise ArtifactProvenanceError("artifact is not marked as paper eligible")
    if metadata.get("block_reasons"):
        raise ArtifactProvenanceError("paper artifact has open block reasons")
    if "functional_claim" not in metadata:
        raise ArtifactProvenanceError("paper artifact must declare functional_claim")
    _verify_split_metadata(metadata)
    _verify_code_identity(metadata)
    _verify_oracle_metadata(metadata)
    return dict(metadata)


def paper_builder_gate(
    builder_id: str,
    project_root: str,
    artifact_paths: Optional[Sequence[str]] = None,
    builder_source_path: Optional[str] = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    candidates = (
        [Path(item).resolve() for item in artifact_paths]
        if artifact_paths is not None
        else sorted((root / "benchmark" / "paper").glob("**/*.json"))
    )
    accepted = []
    rejected = []
    accepted_metadata: Optional[dict[str, object]] = None
    paper_root = (root / "benchmark" / "paper").resolve()
    for candidate in candidates:
        try:
            try:
                candidate.relative_to(paper_root)
            except ValueError as exc:
                raise ArtifactNamespaceError(
                    f"paper builder input is outside project benchmark/paper: {candidate}"
                ) from exc
            metadata = verify_paper_artifact(str(candidate))
            if accepted_metadata is not None:
                verify_provenance_compatibility(
                    accepted_metadata, metadata, require_same_role=False
                )
            else:
                accepted_metadata = metadata
            accepted.append({"path": str(candidate), "provenance_digest": provenance_digest(metadata)})
        except ArtifactContractError as exc:
            rejected.append({"path": str(candidate), "reason": str(exc)})
    reasons = [] if accepted else ["no_verified_paper_artifacts"]
    report = {
        "artifact_kind": "paper_builder_gate",
        "builder_id": builder_id,
        "claim_tier": "paper",
        "paper_eligible": bool(accepted),
        "status": "verified_inputs_available" if accepted else "blocked",
        "accepted_artifacts": accepted,
        "rejected_artifacts": rejected,
        "block_reasons": reasons,
        "rows": [],
    }
    if accepted and accepted_metadata is not None:
        output_metadata = dict(accepted_metadata)
        output_metadata.update({
            "code_identity": code_identity(
                tuple(path for path in (__file__, builder_source_path) if path)
            ),
            "config_sha256": canonical_digest({
                "builder_id": builder_id,
                "accepted_artifacts": accepted,
            }),
            "upstream": {
                "paper_artifacts": accepted,
            },
        })
        report["scientific_validity"] = output_metadata
    return report


__all__ = [
    "RUN_MODES",
    "ArtifactContractError",
    "RunModeError",
    "ArtifactNamespaceError",
    "ArtifactProvenanceError",
    "OracleContractError",
    "normalize_run_mode",
    "require_paper_cli_inputs",
    "canonical_digest",
    "records_identity",
    "prepare_scientific_records",
    "validate_output_namespace",
    "validate_report_output_namespaces",
    "write_paper_report_sidecars",
    "load_and_verify_oracle_manifest",
    "code_identity",
    "build_run_metadata",
    "provenance_digest",
    "write_provenance_sidecar",
    "load_artifact_provenance",
    "verify_provenance_compatibility",
    "upstream_data_provenance",
    "verify_paper_checkpoint",
    "verify_paper_artifact",
    "paper_builder_gate",
]
