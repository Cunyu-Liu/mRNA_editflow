"""Post-preflight R3X example materialization."""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

from .information_sets import (
    FeatureView,
    FoundationQualification,
    InformationSet,
    InformationSetError,
    build_feature_view,
    canonical_feature_row,
    information_set_allowlist,
)
from .protocol import (
    ProtocolHoldError,
    assert_ordinary_record,
    require_staged_disclosure_before_outcome_access,
)


@dataclass(frozen=True)
class EffectExample:
    row_id: str
    source_component_id: str
    ranking_pool_id: str
    features: FeatureView
    target_field: str
    target: float


_SYNTHETIC_ENGINEERING_MATERIALIZATION = ContextVar(
    "r3x_synthetic_engineering_materialization", default=False
)


@contextmanager
def _synthetic_engineering_materialization_scope():
    """Private capability scope for label-bearing synthetic unit fixtures."""

    token = _SYNTHETIC_ENGINEERING_MATERIALIZATION.set(True)
    try:
        yield
    finally:
        _SYNTHETIC_ENGINEERING_MATERIALIZATION.reset(token)


def _assert_materialization_stage(row: Mapping, *, purpose: str) -> None:
    """Reject real internal-test rows until staged release APIs exist.

    Synthetic engineering fixtures remain usable for unit tests, but cannot be
    confused with authority-bound ordinary rows because all three explicit
    fixture markers must agree.
    """

    assert_ordinary_record(row, purpose=purpose)
    partition = row.get("partition")
    if partition in {"train", "development"}:
        return
    if partition == "internal_test":
        if _SYNTHETIC_ENGINEERING_MATERIALIZATION.get() is True:
            return
        raise ProtocolHoldError(
            "HOLD_TEST_OUTCOME_NOT_RELEASED",
            "real internal-test rows require a staged feature/outcome release API",
            details={
                "partition": "internal_test",
                "test_feature_release": False,
                "test_outcome_release": False,
            },
        )
    raise ProtocolHoldError(
        "HOLD_PARTITION_AUTHORITY_REQUIRED",
        "materialization requires an exact train/development partition binding",
        details={"partition": partition},
    )


def build_examples(
    rows: Sequence[Mapping],
    information_set: InformationSet | str,
    *,
    target_field: str = "delta",
    foundation_qualification: FoundationQualification | None = None,
) -> tuple[EffectExample, ...]:
    examples = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise InformationSetError("HOLD_INVALID_DATASET_ROW", "dataset row must be an object")
        _assert_materialization_stage(
            row, purpose="R3X example materialization row"
        )
        target = row.get(target_field)
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(float(target)):
            raise InformationSetError("HOLD_INVALID_TARGET", f"row target {target_field} must be finite")
        row_id = row.get("r3_input_row_sha256") or row.get("record_id")
        component = row.get("source_component_id")
        pool = row.get("source_group_id_v2")
        if not isinstance(row_id, str) or not row_id:
            raise InformationSetError("HOLD_INVALID_DATASET_ROW", "dataset row lacks stable ID")
        if not isinstance(component, str) or not component:
            raise InformationSetError("HOLD_INVALID_DATASET_ROW", "dataset row lacks source_component_id")
        if not isinstance(pool, str) or not pool:
            raise InformationSetError("HOLD_INVALID_DATASET_ROW", "dataset row lacks ranking pool")
        features = build_feature_view(
            row, information_set,
            foundation_qualification=foundation_qualification,
        )
        examples.append(EffectExample(
            row_id=row_id,
            source_component_id=component,
            ranking_pool_id=pool,
            features=features,
            target_field=target_field,
            target=float(target),
        ))
    if not examples:
        raise InformationSetError("HOLD_EMPTY_DATASET", "dataset materialization produced no examples")
    return tuple(examples)


def materialize_method_rows(
    rows: Sequence[Mapping],
    information_set: InformationSet | str,
    *,
    target_fields: Sequence[str] = ("delta",),
    foundation_qualification: FoundationQualification | None = None,
) -> tuple[list[dict], str]:
    """Create the only row mapping a model/baseline is permitted to consume.

    Targets and audit/grouping identifiers are carried separately from the
    immutable feature allowlist.  Non-allowlisted raw fields are never copied.
    """

    info = InformationSet(information_set)
    projected: list[dict] = []
    fingerprints = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise InformationSetError(
                "HOLD_INVALID_DATASET_ROW", "method row must be an object"
            )
        _assert_materialization_stage(
            raw, purpose="R3X method materialization row"
        )
        canonical = canonical_feature_row(raw)
        view = build_feature_view(
            canonical, info, foundation_qualification=foundation_qualification
        )
        fingerprints.add(view.fingerprint)
        item = {
            field: canonical[field]
            for field in information_set_allowlist(info)
        }
        for field in (
            "r3_input_row_sha256", "record_id", "source_component_id",
            "source_group_id_v2", "partition", "study",
            "estimand_id",
        ):
            if field in raw:
                item[field] = raw[field]
        for target in target_fields:
            value = raw.get(target)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise InformationSetError(
                        "HOLD_INVALID_TARGET", f"row target {target} must be finite"
                    )
                item[target] = float(value)
        projected.append(item)
    if not projected:
        raise InformationSetError("HOLD_EMPTY_DATASET", "method materialization produced no rows")
    if len(fingerprints) != 1:
        raise InformationSetError(
            "HOLD_UNEQUAL_INFORMATION_SET", "method rows do not share one feature contract"
        )
    return projected, fingerprints.pop()


def preflight_and_load_examples(
    *,
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    prospective_output_dir: Path,
    row_loader: Callable[[], Sequence[Mapping]],
    information_set: InformationSet | str,
    target_field: str = "delta",
    foundation_qualification: FoundationQualification | None = None,
) -> NoReturn:
    # This call is deliberately first.  It reads only the activation bytes and
    # fails closed before the legacy monolithic ready authority, row loader,
    # model targets, or output path can be opened/created.
    require_staged_disclosure_before_outcome_access(
        activation_manifest_path,
        activation_manifest_sha256,
        prospective_output_dir=prospective_output_dir,
    )
    raise ProtocolHoldError(
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED",
        "staged disclosure guard returned without a stage-specific row loader",
    )
