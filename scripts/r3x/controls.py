"""Preregistered training-only negative controls for R3X critics."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .features import normalize_sequence, validated_substitutions


CONTROL_SCHEMA = "R3X_NEGATIVE_CONTROL_MANIFEST_V1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ALIASES = ("normalized_source_sequence", "source_sequence")
CANDIDATE_ALIASES = ("normalized_candidate_sequence", "candidate_sequence")
CONTEXT_ALIASES = ("publication_context_id", "context_id_v2", "context_id")

# These fields may never be admitted to a transform's mutable-field allowlist.
# The per-transform comparison in ``_finish`` protects every other unlisted
# field as well, including project-specific outcome names not known here.
ABSOLUTELY_IMMUTABLE_FIELDS = frozenset(
    {
        "r3_input_row_sha256",
        "record_id",
        "partition",
        "delta",
        "label",
        "target",
        "y",
        "source_value",
        "candidate_value",
    }
)
CONTROL_METADATA_FIELDS = frozenset(
    {
        "control_name",
        "control_donor_record_id",
        "control_original_context_id",
        "control_original_edit_list",
    }
)


@dataclass(frozen=True)
class ControlResult:
    name: str
    status: str
    rows: list[dict] = field(default_factory=list)
    reason: str = ""
    manifest: Mapping = field(default_factory=dict)


def _resolve_alias_family(
    row: Mapping, aliases: Sequence[str], *, family: str, sequence: bool
) -> str:
    present = [(name, row[name]) for name in aliases if name in row and row[name] is not None]
    if not present:
        raise ValueError(f"negative-control rows require the {family} alias family")
    if sequence:
        normalized = [normalize_sequence(value) for _, value in present]
    else:
        normalized = [str(value) for _, value in present]
        if any(not value for value in normalized):
            raise ValueError(f"negative-control {family} aliases must be non-empty")
    if any(value != normalized[0] for value in normalized[1:]):
        names = ", ".join(name for name, _ in present)
        raise ValueError(f"conflicting negative-control {family} aliases: {names}")
    return normalized[0]


def _set_alias_family(row: dict, aliases: Sequence[str], value: str) -> None:
    for name in aliases:
        row[name] = value


def _canonicalize_row(row: Mapping) -> dict:
    """Resolve the three model-visible alias families exactly once.

    All aliases are populated in the returned row.  This prevents one model
    family from observing a perturbed legacy field while another continues to
    consume a stale normalized/publication alias.
    """

    canonical = copy.deepcopy(dict(row))
    source = _resolve_alias_family(
        canonical, SOURCE_ALIASES, family="source-sequence", sequence=True
    )
    candidate = _resolve_alias_family(
        canonical, CANDIDATE_ALIASES, family="candidate-sequence", sequence=True
    )
    context = _resolve_alias_family(
        canonical, CONTEXT_ALIASES, family="publication-context", sequence=False
    )
    _set_alias_family(canonical, SOURCE_ALIASES, source)
    _set_alias_family(canonical, CANDIDATE_ALIASES, candidate)
    _set_alias_family(canonical, CONTEXT_ALIASES, context)
    if (
        canonical.get("actual_edit_count") is not None
        and canonical.get("edit_count") is not None
        and canonical["actual_edit_count"] != canonical["edit_count"]
    ):
        raise ValueError("conflicting negative-control edit-count aliases")
    # Admission requires the same exact edit replay that downstream paired
    # feature maps and models require.
    validated_substitutions(canonical)
    return canonical


def _source(row: Mapping) -> str:
    return str(row["normalized_source_sequence"])


def _sorted_rows(rows: Sequence[Mapping]) -> list[dict]:
    raw = [copy.deepcopy(dict(row)) for row in rows]
    if not raw:
        raise ValueError("negative control refuses empty rows")
    # Partition and identity are checked before resolving any model-visible
    # fields.  A caller cannot use this training transform as a parser or
    # validator for internal-test rows.
    if any(row.get("partition") != "train" for row in raw):
        raise ValueError("negative controls accept the frozen train partition only")
    ids = [row.get("r3_input_row_sha256") for row in raw]
    if any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in ids):
        raise ValueError("negative-control rows require frozen r3_input_row_sha256 identities")
    if len(set(ids)) != len(ids):
        raise ValueError("negative-control row identities must be unique")
    record_ids = [row.get("record_id") for row in raw]
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise ValueError("negative-control rows require non-empty record_id identities")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("negative-control record_id identities must be unique")
    if any(CONTROL_METADATA_FIELDS.intersection(row) for row in raw):
        raise ValueError("negative controls require unperturbed train rows")
    copied = [_canonicalize_row(row) for row in raw]
    return sorted(copied, key=lambda row: row["r3_input_row_sha256"])


def _sha(rows: Sequence[Mapping]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cell(row: Mapping) -> tuple[str, str, str, str]:
    return (
        str(row.get("study") or ""),
        str(row.get("estimand_id") or ""),
        str(row.get("context_id_v2") or row.get("context_id") or ""),
        str(row.get("actual_edit_count") or row.get("edit_count") or ""),
    )


def _offset(seed: int, name: str, key: object, size: int) -> int:
    if size < 2:
        raise ValueError("derangement requires at least two items")
    payload = json.dumps(
        {"seed": int(seed), "name": name, "key": key},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % (size - 1) + 1


def _finish(
    name: str,
    seed: int,
    original: list[dict],
    output: list[dict],
    *,
    mutable_fields: frozenset[str],
) -> ControlResult:
    if mutable_fields & ABSOLUTELY_IMMUTABLE_FIELDS:
        raise ValueError("negative-control mutable fields include protected identity/target data")
    if any(row.get("partition") != "train" for row in original + output):
        raise ValueError("negative-control transform escaped the train partition")
    original_ids = [row["r3_input_row_sha256"] for row in original]
    output_ids = [row.get("r3_input_row_sha256") for row in output]
    if len(output_ids) != len(set(output_ids)) or sorted(output_ids) != original_ids:
        raise ValueError("negative-control transform changed the frozen row identity set")
    output = sorted(
        [_canonicalize_row(row) for row in output],
        key=lambda row: row["r3_input_row_sha256"],
    )
    if [row["r3_input_row_sha256"] for row in output] != original_ids:
        raise ValueError("negative-control transform changed canonical row order")
    allowed = mutable_fields | CONTROL_METADATA_FIELDS
    for before, after in zip(original, output):
        for field_name in set(before) | set(after):
            if field_name in allowed:
                continue
            if field_name not in before or field_name not in after:
                raise ValueError(
                    f"negative-control transform changed protected field presence: {field_name}"
                )
            if before[field_name] != after[field_name]:
                raise ValueError(
                    f"negative-control transform changed protected identity/target field: {field_name}"
                )
    return ControlResult(
        name=name,
        status="RUN",
        rows=output,
        manifest={
            "schema_version": CONTROL_SCHEMA,
            "name": name,
            "scope": "ORDINARY_TRAIN_PARTITION_ONLY",
            "seed": int(seed),
            "n_rows": len(original),
            "ordered_input_row_ids_sha256": hashlib.sha256(
                json.dumps(original_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _sha(original),
            "output_sha256": _sha(output),
            "partition_validation": "ALL_INPUT_AND_OUTPUT_ROWS_EQUAL_TRAIN",
            "test_partition_modified": False,
            "sealed_access": False,
        },
    )


def _not_run(
    name: str,
    seed: int,
    original: list[dict],
    reason: str,
    *,
    required_remediation: str = "FREEZE_A_REPLAYABLE_CONTROL_PROTOCOL_AND_RERUN",
) -> ControlResult:
    bounded_reason = (
        "PRE_TEST_FREEZE_HOLD; NOT_RUN_IS_NOT_PASS; "
        f"reason={reason}; required_remediation={required_remediation}"
    )
    return ControlResult(
        name=name,
        status="NOT_RUN_WITH_REASON",
        reason=bounded_reason,
        manifest={
            "schema_version": CONTROL_SCHEMA,
            "name": name,
            "scope": "ORDINARY_TRAIN_PARTITION_ONLY",
            "seed": int(seed),
            "n_rows": len(original),
            "ordered_input_row_ids_sha256": hashlib.sha256(
                json.dumps(
                    [row["r3_input_row_sha256"] for row in original],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "input_sha256": _sha(original),
            "partition_validation": "ALL_INPUT_ROWS_EQUAL_TRAIN",
            "status": "NOT_RUN_WITH_REASON",
            "reason": bounded_reason,
            "sealed_access": False,
        },
    )


def pairing_shuffle(rows: Sequence[Mapping], *, seed: int) -> ControlResult:
    """Fail closed until a replayable broken-pair tensorizer is preregistered.

    The current paired model interface requires ``candidate == replay(source,
    edit_list)``.  Swapping candidates across sources breaks that invariant;
    restricting swaps to rows where the invariant still holds makes the
    sequence pair observationally unchanged and is not a genuine broken-pair
    intervention.  Emitting either case as a completed control would therefore
    be scientifically misleading.
    """

    name = "SOURCE_CANDIDATE_PAIRING_SHUFFLE"
    original = _sorted_rows(rows)
    return _not_run(
        name,
        seed,
        original,
        (
            "the current replay-required model schema cannot represent a genuine "
            "source-candidate derangement: swapped pairs fail edit replay, while "
            "replay-compatible swaps are observationally unchanged and are not a "
            "scientifically identifiable broken-pair control"
        ),
        required_remediation=(
            "PREREGISTER_AND_FREEZE_A_BROKEN_PAIR_TENSORIZER_AND_PROTOCOL_REVISION_THEN_RERUN"
        ),
    )


def permute_context_labels(rows: Sequence[Mapping], *, seed: int) -> ControlResult:
    """Permute context identities within endpoint/region without touching labels."""

    name = "CONTEXT_LABEL_PERMUTATION"
    original = _sorted_rows(rows)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in original:
        key = (
            str(row.get("endpoint_id") or row.get("endpoint") or ""),
            str(row.get("region") or ""),
        )
        groups.setdefault(key, []).append(row)
    mappings: dict[tuple[str, str], dict[str, str]] = {}
    for key, group in groups.items():
        levels = sorted({str(row.get("context_id_v2") or row.get("context_id") or "") for row in group})
        if len(levels) < 2 or any(not value for value in levels):
            return _not_run(name, seed, original, "insufficient distinct context support within endpoint/region")
        shift = _offset(seed, name, key, len(levels))
        mappings[key] = {value: levels[(index + shift) % len(levels)] for index, value in enumerate(levels)}
    output = []
    for row in original:
        key = (
            str(row.get("endpoint_id") or row.get("endpoint") or ""),
            str(row.get("region") or ""),
        )
        changed = copy.deepcopy(row)
        old = str(row.get("context_id_v2") or row.get("context_id") or "")
        changed["control_original_context_id"] = old
        _set_alias_family(changed, CONTEXT_ALIASES, mappings[key][old])
        changed["control_name"] = name
        output.append(changed)
    return _finish(
        name,
        seed,
        original,
        output,
        mutable_fields=frozenset(CONTEXT_ALIASES),
    )


def _action_signature(row: Mapping) -> tuple[tuple[int, str, str], ...]:
    return tuple(validated_substitutions(row))


def _apply(source: str, edits: Sequence[tuple[int, str, str]]) -> str:
    sequence = list(normalize_sequence(source))
    for pos, ref, alt in edits:
        if sequence[pos] != ref:
            raise ValueError("control action cannot replay on donor source")
        sequence[pos] = alt
    return "".join(sequence)


def _standard_edit_list(edits: Sequence[tuple[int, str, str]]) -> list[dict]:
    return [{"op": "SUB", "pos": int(pos), "ref": ref, "alt": alt} for pos, ref, alt in edits]


def source_context_shuffle_within_action(rows: Sequence[Mapping], *, seed: int) -> ControlResult:
    """Shuffle source contexts while holding endpoint and exact edit action fixed."""

    name = "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION"
    original = _sorted_rows(rows)
    groups: dict[tuple, list[dict]] = {}
    for row in original:
        source = _source(row)
        key = (*_cell(row), len(source), _action_signature(row))
        groups.setdefault(key, []).append(row)
    if any(len(group) < 2 for group in groups.values()):
        return _not_run(name, seed, original, "insufficient independent source-context support per action stratum")
    output = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        shift = _offset(seed, name, repr(key), len(group))
        action = key[-1]
        for index, row in enumerate(group):
            donor = group[(index + shift) % len(group)]
            donor_source = _source(donor)
            if donor_source == _source(row):
                return _not_run(
                    name, seed, original, "source-context derangement is observationally null"
                )
            try:
                candidate = _apply(donor_source, action)
            except ValueError:
                return _not_run(name, seed, original, "donor source is incompatible with frozen action stratum")
            changed = copy.deepcopy(row)
            _set_alias_family(changed, SOURCE_ALIASES, donor_source)
            _set_alias_family(changed, CANDIDATE_ALIASES, candidate)
            changed["edit_list"] = _standard_edit_list(action)
            changed["control_donor_record_id"] = donor["record_id"]
            changed["control_name"] = name
            output.append(changed)
    return _finish(
        name,
        seed,
        original,
        output,
        mutable_fields=frozenset((*SOURCE_ALIASES, *CANDIDATE_ALIASES, "edit_list")),
    )


def position_shuffle_within_source(rows: Sequence[Mapping], *, seed: int) -> ControlResult:
    """Move each action to another compatible source position."""

    name = "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE"
    original = _sorted_rows(rows)
    output = []
    for row in original:
        source = _source(row)
        edits = validated_substitutions(row)
        original_positions = {pos for pos, _, _ in edits}
        chosen: set[int] = set()
        moved: list[tuple[int, str, str]] = []
        for edit_index, (old_pos, ref, alt) in enumerate(edits):
            candidates = [
                pos for pos, token in enumerate(source)
                if token == ref and pos not in original_positions and pos not in chosen
            ]
            if not candidates:
                return _not_run(name, seed, original, "insufficient compatible alternative-position support")
            key = (row["record_id"], edit_index, old_pos, ref, alt)
            pos = candidates[_offset(seed, name, key, len(candidates) + 1) - 1 if len(candidates) > 1 else 0]
            chosen.add(pos)
            moved.append((pos, ref, alt))
        moved.sort()
        changed = copy.deepcopy(row)
        _set_alias_family(changed, CANDIDATE_ALIASES, _apply(source, moved))
        changed["edit_list"] = _standard_edit_list(moved)
        changed["control_original_edit_list"] = copy.deepcopy(row["edit_list"])
        changed["control_name"] = name
        output.append(changed)
    return _finish(
        name,
        seed,
        original,
        output,
        mutable_fields=frozenset((*CANDIDATE_ALIASES, "edit_list")),
    )


def target_token_shuffle_within_cell(rows: Sequence[Mapping], *, seed: int) -> ControlResult:
    """Rotate target-token tuples across compatible action-position strata."""

    name = "TARGET_TOKEN_SHUFFLE_WITHIN_CELL"
    original = _sorted_rows(rows)
    groups: dict[tuple, list[dict]] = {}
    parsed: dict[str, list[tuple[int, str, str]]] = {}
    for row in original:
        edits = validated_substitutions(row)
        parsed[row["record_id"]] = edits
        key = (*_cell(row), tuple((pos, ref) for pos, ref, _ in edits))
        groups.setdefault(key, []).append(row)
    if any(len(group) < 2 for group in groups.values()):
        return _not_run(name, seed, original, "insufficient compatible target-token support")
    output = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        shift = _offset(seed, name, repr(key), len(group))
        for index, row in enumerate(group):
            donor = group[(index + shift) % len(group)]
            own = parsed[row["record_id"]]
            donor_edits = parsed[donor["record_id"]]
            changed_edits = [
                (pos, ref, donor_alt)
                for (pos, ref, _), (_, _, donor_alt) in zip(own, donor_edits)
            ]
            if any(ref == alt for _, ref, alt in changed_edits) or changed_edits == own:
                return _not_run(name, seed, original, "target-token rotation is identity or creates null edits")
            source = _source(row)
            changed = copy.deepcopy(row)
            _set_alias_family(
                changed, CANDIDATE_ALIASES, _apply(source, changed_edits)
            )
            changed["edit_list"] = _standard_edit_list(changed_edits)
            changed["control_donor_record_id"] = donor["record_id"]
            changed["control_name"] = name
            output.append(changed)
    return _finish(
        name,
        seed,
        original,
        output,
        mutable_fields=frozenset((*CANDIDATE_ALIASES, "edit_list")),
    )
