"""Shared, auditable GP0 data, model, and run-protocol helpers.

This module is intentionally conservative.  It binds a D1 paired record to a
split row and an exposure-ledger row before any sequence reaches a model.  It
never serialises source/candidate sequences, labels, or target-derived
features into run artifacts.  The only target-dependent object constructed by
the training path is the existing MK0 latent alignment/switch-clock oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_ID = "utr_editflow_goal_v2"
CONTRACT_SHA256 = "3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791"
SCIENTIFIC_QUESTION_ID = "RQ-UTR-EDITFLOW-V2"
PHASE_ID = "GP0"
TASK_ID = "GP0-01"
EXPECTED_D1_PAIRED = 134059
EXPECTED_D1_ACCESSIONS = 11
FORBIDDEN_NEW_LABEL_ACCESSIONS = frozenset({"GSE246381"})
ALPHABET = frozenset("ACGU")
SPLIT_VALUES = frozenset({"train", "val", "test"})


class GP0GateError(RuntimeError):
    """A contract/data/runtime condition that must stop the current run."""


def canonical_region(value: Any) -> str:
    """Map D1/B0 display forms to the frozen MK0 region enum strings."""

    if not isinstance(value, str):
        raise GP0GateError(f"region is not a string: {value!r}")
    normalised = value.strip().upper().replace("'", "").replace("′", "").replace("’", "")
    if normalised in {"5UTR", "3UTR"}:
        return normalised
    raise GP0GateError(f"unsupported UTR region spelling: {value!r}")


@dataclass(frozen=True)
class PairedRecord:
    """In-memory training row; sequence fields are never written to artifacts."""

    record_id: str
    accession: str
    region: str
    source: str
    candidate: str
    edit_distance: int
    split: str
    split_types: tuple[str, ...]


def canonical_sequence(value: Any, *, policy: str) -> tuple[str, int]:
    """Canonicalize sequence notation without modifying the D1 source file."""

    if not isinstance(value, str) or not value:
        raise GP0GateError("sequence is empty or not a string")
    if policy == "strict_rna":
        sequence = value.upper()
        converted = 0
    elif policy == "dna_t_to_rna_u":
        sequence = value.upper()
        converted = sequence.count("T")
        sequence = sequence.replace("T", "U")
    else:
        raise GP0GateError(f"unknown sequence alphabet policy: {policy}")
    if any(token not in ALPHABET for token in sequence):
        raise GP0GateError("sequence is outside the canonical RNA alphabet after policy")
    return sequence, converted


@dataclass(frozen=True)
class TrainingExample:
    state: Any
    oracle: Any
    alignment: Any
    clocks: dict[int, float]
    time: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path, *, supplied_sha256: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = supplied_sha256 or sha256_file(path)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "sha256_source": "supplied_prior_verified_binding"
        if supplied_sha256
        else "computed_during_run",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GP0GateError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise GP0GateError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def _require_string(row: Mapping[str, Any], key: str, *, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise GP0GateError(f"{path}: required non-empty string field {key!r} missing")
    return value


def load_split_binding(paths: Sequence[Path]) -> dict[str, Any]:
    """Read split metadata and reject conflicting roles before data selection."""

    if not paths:
        raise GP0GateError("at least one split manifest is required")
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    path_stats: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise GP0GateError(f"split manifest does not exist: {path}")
        count = 0
        split_counts: dict[str, int] = {}
        split_type_counts: dict[str, int] = {}
        for row in _jsonl(path):
            record_id = _require_string(row, "record_id", path=path)
            split = _require_string(row, "split", path=path)
            split_type = _require_string(row, "split_type", path=path)
            if split not in SPLIT_VALUES:
                raise GP0GateError(f"unsupported split value {split!r} in {path}")
            accession = _require_string(row, "accession", path=path)
            region = canonical_region(_require_string(row, "region", path=path))
            item = {
                "split": split,
                "split_type": split_type,
                "accession": accession,
                "region": region,
            }
            rows_by_id.setdefault(record_id, []).append(item)
            count += 1
            split_counts[split] = split_counts.get(split, 0) + 1
            split_type_counts[split_type] = split_type_counts.get(split_type, 0) + 1
        path_stats.append(
            {
                "binding": file_binding(path),
                "row_count": count,
                "split_counts": dict(sorted(split_counts.items())),
                "split_type_counts": dict(sorted(split_type_counts.items())),
            }
        )

    selected_by_split: dict[str, set[str]] = {value: set() for value in SPLIT_VALUES}
    role_by_id: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for record_id, rows in rows_by_id.items():
        split_values = {item["split"] for item in rows}
        if len(split_values) != 1:
            conflicts.append({"record_id_sha256": hashlib.sha256(record_id.encode()).hexdigest(), "splits": sorted(split_values)})
            continue
        split = next(iter(split_values))
        split_types = sorted({item["split_type"] for item in rows})
        accessions = {item["accession"] for item in rows}
        regions = {item["region"] for item in rows}
        if len(accessions) != 1 or len(regions) != 1:
            raise GP0GateError("split metadata disagrees on accession/region for a record")
        role_by_id[record_id] = {
            "split": split,
            "split_types": split_types,
            "accession": next(iter(accessions)),
            "region": next(iter(regions)),
        }
        selected_by_split[split].add(record_id)
    if conflicts:
        raise GP0GateError(
            "a record has conflicting train/val/test roles across supplied split manifests"
        )

    combined = hashlib.sha256()
    for item in path_stats:
        combined.update(item["binding"]["sha256"].encode("ascii"))
    return {
        "schema_version": "gp0_split_binding_v1",
        "paths": path_stats,
        "combined_sha256": combined.hexdigest(),
        "record_count": len(role_by_id),
        "split_counts": {key: len(value) for key, value in sorted(selected_by_split.items())},
        "record_roles": role_by_id,
        "selected_record_ids": {key: sorted(value) for key, value in selected_by_split.items()},
    }


def load_exposure_policy(path: Path, record_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load only ledger rows needed by the selected split; no full ledger export."""

    if not path.exists():
        raise GP0GateError(f"exposure ledger does not exist: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in record_ids:
            continue
        policy = {
            "record_id": record_id,
            "accession": _require_string(row, "accession", path=path),
            "region": canonical_region(_require_string(row, "region", path=path)),
            "data_role": row.get("data_role"),
            "evidence_grade": row.get("evidence_grade"),
            "exposure_status": row.get("exposure_status"),
            "historically_exposed": bool(row.get("historically_exposed")),
            "labels_allowed_for_new_training": bool(
                row.get("labels_allowed_for_new_training")
            ),
            "labels_allowed_for_new_hyperparameter_selection": bool(
                row.get("labels_allowed_for_new_hyperparameter_selection")
            ),
        }
        prior = result.get(record_id)
        if prior is not None and prior != policy:
            raise GP0GateError("exposure ledger contains conflicting rows for one record")
        result[record_id] = policy
    return result


def scan_d1_and_select(
    path: Path,
    *,
    split_binding: Mapping[str, Any],
    requested_split: str,
    exposure_policy: Mapping[str, Mapping[str, Any]],
    allow_forbidden_for_development: bool = False,
    max_records: int | None = None,
    sequence_alphabet_policy: str = "dna_t_to_rna_u",
) -> tuple[list[PairedRecord], dict[str, Any]]:
    """Scan D1, enforce pair/schema rules, and select one split without labels."""

    if requested_split not in SPLIT_VALUES:
        raise GP0GateError(f"invalid requested split: {requested_split}")
    all_ids = set(split_binding["record_roles"])
    selected_ids = set(split_binding["selected_record_ids"][requested_split])
    total = 0
    paired = 0
    malformed = 0
    accessions: set[str] = set()
    paired_accessions: dict[str, int] = {}
    paired_regions: dict[str, int] = {}
    selected: list[PairedRecord] = []
    dropped: dict[str, int] = {}
    alphabet_conversions = {"source_T_to_U": 0, "candidate_T_to_U": 0}
    seen_selected: set[str] = set()
    for row in _jsonl(path):
        total += 1
        record_id = row.get("record_id")
        accession = row.get("accession")
        raw_region = row.get("region")
        if isinstance(accession, str):
            accessions.add(accession)
        source = row.get("source_sequence")
        candidate = row.get("candidate_sequence")
        is_paired = isinstance(source, str) and isinstance(candidate, str)
        if is_paired:
            paired += 1
            paired_accessions[str(accession)] = paired_accessions.get(str(accession), 0) + 1
            paired_regions[str(raw_region)] = paired_regions.get(str(raw_region), 0) + 1
        else:
            if record_id in selected_ids:
                dropped["selected_record_not_paired"] = dropped.get("selected_record_not_paired", 0) + 1
            continue
        if record_id not in selected_ids:
            continue
        region = canonical_region(raw_region)
        if not isinstance(record_id, str) or record_id in seen_selected:
            raise GP0GateError("selected split record IDs are missing, invalid, or duplicated in D1")
        seen_selected.add(record_id)
        role = split_binding["record_roles"].get(record_id)
        if role is None or role["accession"] != accession or role["region"] != region:
            raise GP0GateError("split metadata does not bind to the D1 accession/region")
        if region not in ("5UTR", "3UTR"):
            raise GP0GateError(f"unsupported canonical UTR region in selected D1 row: {region!r}")
        canonical_source, source_converted = canonical_sequence(
            source, policy=sequence_alphabet_policy
        )
        canonical_candidate, candidate_converted = canonical_sequence(
            candidate, policy=sequence_alphabet_policy
        )
        alphabet_conversions["source_T_to_U"] += source_converted
        alphabet_conversions["candidate_T_to_U"] += candidate_converted
        try:
            edit_distance = int(row["edit_distance"])
        except (KeyError, TypeError, ValueError) as error:
            raise GP0GateError("selected D1 row has no integer edit_distance") from error
        if edit_distance < 0:
            raise GP0GateError("selected D1 edit_distance is negative")
        policy = exposure_policy.get(record_id)
        if policy is None:
            raise GP0GateError("selected D1 row has no exposure-ledger binding")
        if policy["accession"] != accession or policy["region"] != region:
            raise GP0GateError("D1 and exposure-ledger accession/region disagree")
        forbidden = (
            accession in FORBIDDEN_NEW_LABEL_ACCESSIONS
            or not policy["labels_allowed_for_new_training"]
            or not policy["labels_allowed_for_new_hyperparameter_selection"]
        )
        if forbidden and not allow_forbidden_for_development:
            dropped["contract_forbidden_or_label_not_admitted"] = dropped.get(
                "contract_forbidden_or_label_not_admitted", 0
            ) + 1
            continue
        if max_records is not None and len(selected) >= max_records:
            continue
        selected.append(
            PairedRecord(
                record_id=record_id,
                accession=accession,
                region=region,
                source=canonical_source,
                candidate=canonical_candidate,
                edit_distance=edit_distance,
                split=requested_split,
                split_types=tuple(role["split_types"]),
            )
        )
    if malformed:
        dropped["malformed"] = malformed
    missing_selected = selected_ids - seen_selected
    if missing_selected and max_records is None:
        raise GP0GateError(
            f"split references {len(missing_selected)} IDs absent from paired D1 records"
        )
    summary = {
        "schema_version": "gp0_d1_binding_v1",
        "path": str(path),
        "total_records": total,
        "paired_records": paired,
        "non_paired_records": total - paired,
        "accessions": len(accessions),
        "paired_by_accession": dict(sorted(paired_accessions.items())),
        "paired_by_region": dict(sorted(paired_regions.items())),
        "requested_split": requested_split,
        "split_candidate_records": len(selected_ids),
        "selected_records_after_policy": len(selected),
        "selected_accessions": dict(sorted({r.accession: sum(x.accession == r.accession for x in selected) for r in selected}.items())),
        "dropped": dropped,
        "selection_policy": "paired sequences only; labels field never read; target_condition fixed to maintain",
        "sequence_alphabet_policy": sequence_alphabet_policy,
        "sequence_alphabet_conversion_counts": alphabet_conversions,
    }
    return selected, summary


def load_b0_binding(card_path: Path | None, split_summary_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "card": file_binding(card_path) if card_path else {"path": None, "exists": False},
        "split_summary": file_binding(split_summary_path) if split_summary_path else {"path": None, "exists": False},
        "track_unique_records": {},
        "study_disjoint_n_total": None,
        "study_disjoint_by_accession": {},
    }
    if card_path and card_path.exists():
        payload = json.loads(card_path.read_text(encoding="utf-8"))
        result["track_unique_records"] = {
            str(track): card.get("counts", {}).get("unique_records")
            for track, card in payload.get("track_cards", {}).items()
        }
    if split_summary_path and split_summary_path.exists():
        payload = json.loads(split_summary_path.read_text(encoding="utf-8"))
        study = payload.get("study_disjoint", {})
        result["study_disjoint_n_total"] = study.get("n_total")
        result["study_disjoint_by_accession"] = {
            accession: sum(int(value) for value in splits.values())
            for accession, splits in study.get("by_accession_split", {}).items()
        }
    return result


def validate_formal_data_binding(
    *,
    d1_summary: Mapping[str, Any],
    b0_binding: Mapping[str, Any],
    selected_summary: Mapping[str, Any],
    exposure_policy: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return blockers; caller must preserve them and refuse formal training."""

    blockers: list[dict[str, Any]] = []
    if d1_summary.get("paired_records") != EXPECTED_D1_PAIRED or d1_summary.get("accessions") != EXPECTED_D1_ACCESSIONS:
        blockers.append({
            "id": "D1_CURRENT_COUNT_MISMATCH",
            "expected": {"paired_records": EXPECTED_D1_PAIRED, "accessions": EXPECTED_D1_ACCESSIONS},
            "observed": {"paired_records": d1_summary.get("paired_records"), "accessions": d1_summary.get("accessions")},
        })
    track_counts = b0_binding.get("track_unique_records", {})
    if not track_counts or any(value != EXPECTED_D1_PAIRED for value in track_counts.values()):
        blockers.append({
            "id": "B0_SPLIT_CARD_NOT_BOUND_TO_CURRENT_D1",
            "expected_unique_records": EXPECTED_D1_PAIRED,
            "observed_track_unique_records": track_counts,
            "observed_study_disjoint_n_total": b0_binding.get("study_disjoint_n_total"),
        })
    if b0_binding.get("study_disjoint_n_total") != EXPECTED_D1_PAIRED:
        blockers.append({
            "id": "B0_STUDY_SPLIT_TOTAL_NOT_BOUND_TO_CURRENT_D1",
            "expected": EXPECTED_D1_PAIRED,
            "observed": b0_binding.get("study_disjoint_n_total"),
        })
    forbidden = [
        row for row in exposure_policy.values()
        if row["accession"] in FORBIDDEN_NEW_LABEL_ACCESSIONS
        or not row["labels_allowed_for_new_training"]
        or not row["labels_allowed_for_new_hyperparameter_selection"]
    ]
    if forbidden:
        blockers.append({
            "id": "GSE246381_HIGH_LEVEL_POLICY_CONFLICT",
            "contract_rule": "historically exposed labels are not admitted for new training or hyperparameter selection",
            "selected_policy_row_count": len(forbidden),
        })
    if selected_summary.get("dropped", {}).get("contract_forbidden_or_label_not_admitted", 0):
        blockers.append({
            "id": "SELECTED_RECORDS_DROPPED_BY_EXPOSURE_POLICY",
            "count": selected_summary["dropped"]["contract_forbidden_or_label_not_admitted"],
        })
    return blockers


def _alignment_action(alignment: Any, augmented_current: Sequence[str], index: int):
    """Mirror the target-kernel coordinate mapping for completed switches."""

    from mrna_editflow.core.mk0.alignment_coupling import BLANK
    from mrna_editflow.core.mk0.types import ActionType, AtomicAction

    column = alignment.columns[index]
    current_token = augmented_current[index]
    if current_token != column.source_token:
        raise GP0GateError("completed switch was not in its source-side state")
    position = sum(token != BLANK for token in augmented_current[:index])
    if column.source_token == BLANK and column.target_token != BLANK:
        return AtomicAction(ActionType.INS, position, column.target_token)
    if column.target_token == BLANK and column.source_token != BLANK:
        return AtomicAction(ActionType.DEL, position)
    if column.source_token != BLANK and column.target_token != BLANK and column.source_token != column.target_token:
        return AtomicAction(ActionType.SUB, position, column.target_token)
    raise GP0GateError("alignment coordinate is not an atomic edit")


def make_training_example(
    record: PairedRecord,
    *,
    rng: random.Random,
    min_length: int,
    max_length: int,
    time_policy: str = "stochastic",
) -> TrainingExample:
    """Construct a legal MK0 training state and target oracle."""

    from mrna_editflow.core.mk0.alignment_coupling import (
        build_alignment,
        changed_indices,
        sample_switch_clocks,
        switched_alignment_state,
    )
    from mrna_editflow.core.mk0.state_action import apply_action
    from mrna_editflow.core.mk0.target_kernel import build_target_transition_oracle
    from mrna_editflow.core.mk0.types import EditState

    alignment = build_alignment(record.source, record.candidate)
    if alignment.cost != record.edit_distance:
        raise GP0GateError(
            "D1 edit_distance disagrees with the frozen canonical alignment"
        )
    if len(record.source) < min_length or len(record.candidate) < min_length:
        raise GP0GateError("paired record violates GP0 minimum length")
    if len(record.source) > max_length or len(record.candidate) > max_length:
        raise GP0GateError("paired record violates GP0 maximum length")
    if time_policy not in {"initial_only", "stochastic"}:
        raise GP0GateError(f"unknown GP0 time policy: {time_policy}")
    clocks = sample_switch_clocks(alignment, rng=rng, schedule="cubic")
    clocks = {index: max(1.0e-6, min(1.0 - 1.0e-6, value)) for index, value in clocks.items()}
    if time_policy == "initial_only":
        time = 0.0
    else:
        time = rng.uniform(0.0, 0.75)
    changed = changed_indices(alignment)
    completed = tuple(index for index in changed if clocks[index] <= time)
    augmented_current = switched_alignment_state(alignment, clocks, time)
    state = EditState.initial(
        record.source,
        region=record.region,
        context={
            "assay": "unspecified",
            "cell_or_tissue": "unspecified",
            "endpoint": "unspecified",
            "batch": None,
        },
        target_condition="maintain",
        budget=max(1, alignment.cost),
    )
    # Target condition/function labels are intentionally absent.  This is only
    # the latent path used to construct the GP0 edit-flow target.
    for index in sorted(completed):
        action = _alignment_action(alignment, augmented_current, index)
        state = apply_action(
            state,
            action,
            min_length=min_length,
            max_length=max_length,
        ).after
    observable_current = "".join(token for token in augmented_current if token != "ε")
    if state.current != observable_current:
        raise GP0GateError("constructed training state does not match switched alignment")
    oracle = build_target_transition_oracle(
        state,
        alignment,
        clocks,
        time,
        min_length=min_length,
        max_length=max_length,
        schedule="cubic",
    )
    return TrainingExample(state=state, oracle=oracle, alignment=alignment, clocks=clocks, time=time)


def require_cuda_device(device_name: str) -> dict[str, Any]:
    """Require real CUDA before importing/using the neural rate field."""

    import torch

    if not torch.cuda.is_available():
        raise GP0GateError("CUDA is unavailable; GP0 formal neural work is stopped")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise GP0GateError("GP0 formal neural work cannot use a CPU device")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise GP0GateError(f"CUDA device index is outside device_count: {index}")
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    uuid_result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + str(index),
            "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if uuid_result.returncode != 0 or not uuid_result.stdout.strip():
        raise GP0GateError("nvidia-smi could not bind the requested CUDA device")
    return {
        "device": str(device),
        "device_index": index,
        "device_count": torch.cuda.device_count(),
        "cuda_available": True,
        "torch_name": properties.name,
        "torch_total_memory_bytes": properties.total_memory,
        "nvidia_smi": uuid_result.stdout.strip(),
    }


class LowRankResidualAdapter:  # populated as an nn.Module at runtime
    """Marker for documentation; concrete class is defined with torch below."""


def build_rate_field(
    *,
    variant: str,
    snapshot_dir: Path,
    device: Any,
    seed: int,
    min_length: int,
    max_length: int,
    hidden_head_width: int = 128,
    adapter_rank: int = 8,
):
    """Instantiate one explicit GP0 comparison without changing the default EF0."""

    import torch
    from torch import nn

    from mrna_editflow.core.ef0.model import EF0ModelConfig, TrueUTREditFlowRateField
    from mrna_editflow.core.mk0.foundation_fusion import load_official_utrlm
    from mrna_editflow.core.mk0.types import ActionType

    variant_aliases = {
        "from-scratch": "from_scratch",
        "frozen-foundation": "frozen_foundation",
        "lora-adapter": "lora_adapter",
        "no-source": "no_source",
        "no-time": "no_time",
        "no-indel": "no_indel",
        "no-STOP": "no_stop",
        "fixed-length": "fixed_length",
        "no-region-adapter": "no_region_adapter",
    }
    if variant not in variant_aliases:
        raise GP0GateError(f"unsupported GP0 variant: {variant}")
    kind = variant_aliases[variant]
    foundation, tokenizer = load_official_utrlm(
        snapshot_dir,
        device=device,
        from_scratch=kind == "from_scratch",
        seed=seed,
    )
    field = TrueUTREditFlowRateField(
        foundation,
        tokenizer,
        device=device,
        config=EF0ModelConfig(
            min_length=min_length,
            max_length=max_length,
            hidden_head_width=hidden_head_width,
        ),
        train_foundation=kind == "from_scratch",
        cache_current_embeddings=kind == "frozen_foundation",
    )
    field.gp0_variant = variant
    field.gp0_kind = kind
    field.gp0_ablation_semantics = {
        "no-source": "remove explicit source-aligned/shared source features; current sequence remains observable",
        "no-time": "replace explicit external-time scalar with fixed 0.5",
        "no-indel": "set INS and DEL rates to zero; paired records must be same-length",
        "no-STOP": "set STOP rate to zero; termination is forced by the sampler horizon/budget",
        "fixed-length": "same-length training subset plus INS/DEL rates set to zero",
        "no-region-adapter": "replace learned region gate with unit gate",
    }.get(variant)

    if kind == "lora_adapter":
        if adapter_rank < 1 or adapter_rank >= field.hidden_size:
            raise GP0GateError("adapter rank must be positive and below foundation width")

        class ResidualAdapter(nn.Module):
            def __init__(self, width: int, rank: int) -> None:
                super().__init__()
                self.down = nn.Linear(width, rank, bias=False)
                self.up = nn.Linear(rank, width, bias=False)
                nn.init.xavier_uniform_(self.down.weight)
                nn.init.zeros_(self.up.weight)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return tokens + self.up(torch.tanh(self.down(tokens)))

        field.feature_adapter = ResidualAdapter(field.hidden_size, adapter_rank).to(
            device=device, dtype=torch.float32
        )
        field.gp0_adapter_rank = adapter_rank
    else:
        field.gp0_adapter_rank = None
    if kind == "no_region_adapter":
        field.region_adapter.requires_grad_(False)

    # A single ablation wrapper around the same production forward path.
    original_forward = field.forward
    original_encoded_state = field._encoded_state
    original_region_gate = field._region_gate

    def encoded_state(state, time):
        source_tokens, current_tokens, aligned_tokens, shared = original_encoded_state(state, time)
        if kind == "no_source":
            source_tokens = torch.zeros_like(source_tokens)
            aligned_tokens = torch.zeros_like(aligned_tokens)
            shared = shared.clone()
            shared[: field.hidden_size] = 0.0
            shared[2 * field.hidden_size : 3 * field.hidden_size] = 0.0
        if kind == "no_time":
            shared = shared.clone()
            shared[3 * field.hidden_size] = 0.5
        return source_tokens, current_tokens, aligned_tokens, shared

    def region_gate(state, action):
        if kind == "no_region_adapter":
            return torch.ones((), device=field.device, dtype=torch.float32)
        return original_region_gate(state, action)

    field._encoded_state = encoded_state  # type: ignore[method-assign]
    field._region_gate = region_gate  # type: ignore[method-assign]

    def forward(state, time, actions=None):
        result = original_forward(state, time, actions=actions)
        if kind in {"no_indel", "fixed_length"}:
            result = {
                action: (rate * 0.0 if action.kind in {ActionType.INS, ActionType.DEL} else rate)
                for action, rate in result.items()
            }
        if kind == "no_stop":
            result = {
                action: (rate * 0.0 if action.kind == ActionType.STOP else rate)
                for action, rate in result.items()
            }
        return result

    field.forward = forward  # type: ignore[method-assign]
    field.to(device=device, dtype=torch.float32)
    return field


def runtime_model_audit(field: Any) -> dict[str, Any]:
    import torch

    parameters = list(field.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    frozen = [parameter for parameter in parameters if not parameter.requires_grad]
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "field_device": str(field.device),
        "all_parameters_cuda": all(parameter.device.type == "cuda" for parameter in parameters),
        "trainable_parameters_cuda": all(parameter.device.type == "cuda" for parameter in trainable),
        "frozen_parameters_cuda": all(parameter.device.type == "cuda" for parameter in frozen),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "frozen_parameter_count": int(sum(parameter.numel() for parameter in frozen)),
        "foundation_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in field.foundation.parameters() if parameter.requires_grad)
        ),
        "foundation_requires_grad_count": int(
            sum(int(parameter.requires_grad) for parameter in field.foundation.parameters())
        ),
        "representation_mode": field.representation_mode(),
        "forward_calls": int(getattr(field, "_runtime_forward_calls", 0)),
    }


def assert_cuda_rates(rates: Mapping[Any, Any]) -> None:
    import torch

    for action, rate in rates.items():
        if not isinstance(rate, torch.Tensor) or rate.device.type != "cuda":
            raise GP0GateError(f"rate for {getattr(action, 'key', action)} used CPU fallback")
        if rate.numel() != 1 or not bool(torch.isfinite(rate)) or bool(rate < 0):
            raise GP0GateError(f"invalid CUDA rate for {getattr(action, 'key', action)}")


def artifact_checksums(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_checksums.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    return "\n".join(rows) + "\n"
