#!/usr/bin/env python
"""B0-R (v3.1) — shared constants, RFC8785/JCS helpers and streaming I/O.

This module is shared by the B0-R builder, validator and freezer. It implements
the frozen hashes, the immutable component-set orderings, RFC8785 JSON
Canonicalization Scheme (JCS) hashing used for transition events / access
events / root-commit records, and streaming JSONL helpers so that the
multi-million-row D1 artifacts are never loaded into memory at once.

No training, no GPU work.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen constants (authoritative contract §5.7, §14.7, config + C3)
# ---------------------------------------------------------------------------

CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"
SCHEMA_VERSION = "3.1"

# The 12 required task IDs (UTF-8, lexicographic, LF-terminated each).
REQUIRED_TASK_IDS = sorted([
    "CROSS_REGION_PROPERTY_F_OBSERVATION",
    "CROSS_REGION_RECONSTRUCT_E_PAIR",
    "F3_OUTCOME_AUX_OBSERVATION",
    "F5_OUTCOME_AUX_OBSERVATION",
    "T3_EFFECT_DELTA_E_PAIR",
    "T3_PROPERTY_E_PAIR",
    "T3_RANK_EXPLORATORY_E_PAIR",
    "T3_RECONSTRUCT_E_PAIR",
    "T5_CONTEXT_E_PAIR",
    "T5_CONTEXT_F_OBSERVATION",
    "T5_GEN_RECONSTRUCT_E_PAIR",
    "T5_RANK_CLOSED_SELECT_E_PAIR",
])

# The 10 required split contract IDs.
REQUIRED_SPLIT_IDS = sorted([
    "3utr_sequence_cluster_disjoint",
    "3utr_source_or_variant_disjoint",
    "3utr_study_disjoint",
    "5utr_sequence_cluster_disjoint",
    "5utr_source_disjoint",
    "5utr_study_disjoint",
    "cross_region_3_to_5",
    "cross_region_5_to_3",
    "heldout_context",
    "sealed_final_v1",
])

# Sealed cohort.
SEALED_COHORT_IDS = ["GSE246381"]

# Frozen hashes (from config + contract).
FROZEN_HASHES = {
    "task_id_set_sha256": "b0b43cb76f39b32009e3a6ef8ae6d05395d61bf7baa7480743587e6772447207",
    "task_descriptor_set_sha256": "8f42ef044d8de1a26b9b587587c2de99c6068f67f37e269e226e143333245ba3",
    "split_id_set_sha256": "b8c6fb2718875862da500c949481d04db08d1d21f94e3d13da49e3ace64ff487",
    "split_descriptor_set_sha256": "c8a6c82a9a1ab687ef2c3cb912ed96aae26c73a0662b0ae0911040c37e8ef1fa",
    "task_split_allowlist_sha256": "02b25e4717e4a7192b658d5e69cdbb198e5b696b3ea520b7a0a887fcf89097ab",
    "grouping_atom_rule_sha256": "bd8395ab0ec23d98d7c1b717e7fcb0bdd3df6d18002985624cd9eb41f8bd7983",
    "activation_calibration_rule_sha256": "b2652abda7a2dbb7001e7fb655db9b6ac19f2b8f80fbc65362dc1236fd9781e9",
    "diagnostic_registry_expected_set_sha256": "f25c0adc643f38ff26c5e08bf07e4175a4e2571eaae939d61daa91fc6f2aabb2",
    "sealed_cohort_set_sha256": "275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0",
    # ordinary / restricted B0 PREPARED logical-component-set hashes
    "b0_ordinary_prepared_component_set_sha256": "645042cc476710448f4f5b70c80c8cd624c4ea44177eea48d22233fd575545d8",
    "b0_restricted_prepared_component_set_sha256": "00ebb4bb9090ed74c2d37a424773edd2b4216e50fec084013d978469fcb9b3ff",
}

# Ordinary B0 PREPARED logical-component set (UTF-8 lexicographic, LF each).
ORDINARY_PREPARED_COMPONENTS = sorted([
    "ACTIVATION_CALIBRATION_MASK",
    "B0_ROLE_DECISION_EVIDENCE",
    "EFFECTIVE_EXPOSURE_PROJECTION",
    "EFFECTIVE_ROLE_PROJECTION",
    "ELIGIBILITY_MANIFEST",
    "FIVE_SCALE_DATA_CARD",
    "FOUNDATION_EXPOSURE_LEDGER_MANIFEST",
    "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE",
    "GSE246381_B0_AGGREGATE",
    "GSE246381_B0_COMMITMENT",
    "LEGACY_B0_INVALIDATION_MANIFEST",
    "ORDINARY_ACCESS_PREFIX_MANIFEST",
    "RELATION_ROLE_TRANSITIONS",
    "RESOURCE_VIABILITY_ASSESSMENT",
    "SPLIT_ACTIVATION_DECISIONS",
    "SPLIT_ASSIGNMENTS",
    "TASK_ACTIVATION_DECISIONS",
    "TASK_ELIGIBILITY_UNIVERSE",
    "TASK_SPLIT_APPLICABILITY_DECISIONS",
])

# Restricted B0 PREPARED logical-component set.
RESTRICTED_PREPARED_COMPONENTS = sorted([
    "ACCESS_PREFIX_MANIFEST",
    "B0_ROLE_DECISION_EVIDENCE",
    "EFFECTIVE_EXPOSURE_PROJECTION",
    "EFFECTIVE_ROLE_PROJECTION",
    "ELIGIBILITY_MANIFEST",
    "FOUNDATION_EXPOSURE_LEDGER",
    "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE",
    "RELATION_ROLE_TRANSITIONS",
    "SPLIT_ASSIGNMENTS",
    "TASK_ELIGIBILITY_UNIVERSE",
])

# GSE sealed evaluation-compatibility scope (frozen §B0-00 Stage1).
GSE_SEALED_SCOPE_TASKS = ["T5_GEN_RECONSTRUCT_E_PAIR", "T5_RANK_CLOSED_SELECT_E_PAIR"]

# Canonical role-transition matrix (contract §5.5.1).
ALLOWED_ROLE_TRANSITIONS = {
    "PENDING": ["GENERAL_DEVELOPMENT_POOL", "SEALED_EXTERNAL_FINAL_CANDIDATE",
                "EXTERNAL_STRESS_ONLY", "EXCLUDED"],
    "GENERAL_DEVELOPMENT_POOL": ["EXCLUDED"],
    "SEALED_EXTERNAL_FINAL_CANDIDATE": ["SEALED_EXTERNAL_FINAL", "EXCLUDED"],
    "EXTERNAL_STRESS_ONLY": ["EXCLUDED"],
    "SEALED_EXTERNAL_FINAL": [],
    "EXCLUDED": [],
}

# E PAIR global-role x partition-role allowed matrix (§5.7.5).
ROLE_PARTITION_MATRIX = {
    "GENERAL_DEVELOPMENT_POOL": {"TRAIN", "DEVELOPMENT", "INTERNAL_TEST"},
    "SEALED_EXTERNAL_FINAL": {"SEALED_FINAL"},
    "EXTERNAL_STRESS_ONLY": {"STRESS_ONLY"},
    "SEALED_EXTERNAL_FINAL_CANDIDATE": set(),
    "PENDING": set(),
    "EXCLUDED": set(),
}

GENESIS_SENTINEL = "GENESIS"

# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_utf8(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def set_sha256(ids, key="id") -> str:
    """SHA256 over UTF-8, lexicographic, LF-terminated each element."""
    lines = "".join(sorted(str(i) + "\n" for i in ids))
    return sha256_utf8(lines)


# ---------------------------------------------------------------------------
# RFC8785 JSON Canonicalization Scheme
# ---------------------------------------------------------------------------

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _jcs_escape_char(ch: str) -> str:
    if ch in _ESCAPES:
        return _ESCAPES[ch]
    cp = ord(ch)
    if cp < 0x20:
        return "\\u%04x" % cp
    # RFC8785 requires escaping U+007F..U+009F as \uXXXX.
    if 0x7F <= cp <= 0x9F:
        return "\\u%04x" % cp
    return ch


def _jcs_escape_string(s: str) -> str:
    return '"' + "".join(_jcs_escape_char(c) for c in s) + '"'


def _jcs_float_str(f: float) -> str:
    # RFC8785: shortest round-trippable, no exponent for integral, etc.
    if f != f:
        raise ValueError("NaN not allowed in JCS")
    if f == float("inf") or f == float("-inf"):
        raise ValueError("Infinity not allowed in JCS")
    if f == 0:
        return "0" if (1.0 / f) > 0 else "-0"
    s = repr(f)
    if "e" not in s and "E" not in s:
        if "." in s:
            return s
        return s + ".0"
    return s


def jcs_dumps(obj) -> str:
    """Serialize a Python object to RFC8785 canonical JSON string."""
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _jcs_float_str(obj)
    if isinstance(obj, str):
        return _jcs_escape_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(jcs_dumps(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return "{" + ",".join(
            _jcs_escape_string(k) + ":" + jcs_dumps(v) for k, v in items
        ) + "}"
    raise TypeError("Unsupported type for JCS: %r" % type(obj))


def jcs_sha256(obj, exclude: list | None = None) -> str:
    """RFC8785 canonify obj, optionally drop members named in `exclude`, then SHA256."""
    work = obj
    if exclude:
        work = {k: v for k, v in obj.items() if k not in exclude}
    canonical = jcs_dumps(work).encode("utf-8")
    return sha256_bytes(canonical)


# ---------------------------------------------------------------------------
# Streaming JSONL I/O
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path):
    """Yield parsed JSON objects line-by-line; empty/missing file -> no rows."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    """Stream rows to a JSONL file (one compact JSON per line, LF)."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def count_jsonl(path: Path) -> int:
    n = 0
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for _ in fh:
                n += 1
    return n


# ---------------------------------------------------------------------------
# Registry loading (streaming-free, small YAML files)
# ---------------------------------------------------------------------------


def load_tasks(path: Path) -> dict:
    try:
        import yaml
    except Exception:  # pragma: no cover
        raise RuntimeError("PyYAML required to load task/split registries")
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return {t["task_id"]: t for t in doc["tasks"]}


def load_splits(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return {s["split_contract_id"]: s for s in doc["splits"]}


def load_matrix(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc["rows"]


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_viability_rule(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_diagnostic_registry(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    # Self-test of JCS golden vector (RFC8785 example).
    # Note: RFC8785's canonical example intentionally uses escaped forms.
    obj = {"b": 1, "a": [{"d": True, "c": "x"}, None]}
    print("jcs_dumps:", jcs_dumps(obj))
    print("task_id_set_sha256:", set_sha256(REQUIRED_TASK_IDS))
    print("split_id_set_sha256:", set_sha256(REQUIRED_SPLIT_IDS))
    print("sealed_cohort_set_sha256:", set_sha256(SEALED_COHORT_IDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())