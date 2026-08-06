#!/usr/bin/env python3
"""M2: build schemas/xedit_v1_1/ namespace from schemas/v3_1/.

Preserves the 21 v3_1 schemas byte-for-byte and creates a new xedit_v1_1
namespace that:
  1. rebinds contract_id -> mrna_xeditflow_goal_v1_1, schema_version -> 1.1,
     $id path -> /schemas/xedit_v1_1/, title suffix;
  2. injects the orthogonal axes (scientific_track / intervention_evidence_grade
     / method_training_role / endpoint_role / critic_eligibility /
     flow_base_eligibility / guidance_training_eligibility /
     measured_optimization_eligibility / transfer_eligibility) as optional
     fields on the core data entities;
  3. emits a new orthogonal-axes reference schema and a putatively shared $defs.

Old v3_1 schemas are never modified. All new fields are optional so existing
sealed/frozen v3_1 rows remain valid under the new namespace.
"""

import hashlib
import json
import shutil
from pathlib import Path

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"
NEW_SCHEMA_VERSION = "1.1"
OLD_VERSION = "3.1"

SRC = Path("schemas/v3_1")
DST = Path("schemas/xedit_v1_1")

# Orthogonal axes injected onto core data entities (optional properties).
ORTHOGONAL_AXES = {
    "scientific_track": {
        "type": "string",
        "enum": ["E", "F", "AUX", "REFERENCE"],
        "description": "Retained old scientific track; A1/D never overwrite E/F.",
    },
    "intervention_evidence_grade": {
        "type": "string",
        "enum": ["A1", "A2", "B1", "B2", "C", "D"],
        "description": "Evidence ladder, orthogonal to scientific_track.",
    },
    "method_training_role": {
        "type": "string",
        "enum": [
            "EFFECT_PRIMARY",
            "FLOW_BASE",
            "CRITIC_AUX",
            "TRANSFER",
            "DIAGNOSTIC",
            "EXCLUDED",
        ],
    },
    "endpoint_role": {
        "type": "string",
        "enum": [
            "DELTA",
            "ABSOLUTE_PROPERTY",
            "FAMILY_RANK",
            "RECONSTRUCTION",
            "NOT_APPLICABLE",
        ],
    },
    "critic_eligibility": {
        "type": "string",
        "enum": ["YES", "NO"],
    },
    "flow_base_eligibility": {
        "type": "string",
        "enum": ["YES", "NO"],
    },
    "guidance_training_eligibility": {
        "type": "string",
        "enum": ["YES", "NO"],
    },
    "measured_optimization_eligibility": {
        "type": "string",
        "enum": ["YES", "NO"],
    },
    "transfer_eligibility": {
        "type": "string",
        "enum": ["YES", "NO"],
    },
}

# Core data entities that receive the orthogonal axes.
CORE_ENTITIES = {
    "dataset_asset.schema.json",
    "sequence_entity.schema.json",
    "functional_observation.schema.json",
    "utr_edit_relation_candidate.schema.json",
    "utr_edit_pair.schema.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebind(schema: dict, filename: str) -> dict:
    schema = json.loads(json.dumps(schema))
    schema["contract_id"] = NEW_CONTRACT_ID
    schema["schema_version"] = NEW_SCHEMA_VERSION
    schema["title"] = schema.get("title", filename).replace(
        f"(v{OLD_VERSION})", f"(xedit_v1_1)"
    ).replace("v3.1", "xedit_v1_1")
    if "$id" in schema:
        schema["$id"] = schema["$id"].replace("/v3_1/", "/xedit_v1_1/")
    return schema


def inject_axes(schema: dict) -> dict:
    schema = json.loads(json.dumps(schema))
    props = schema.setdefault("properties", {})
    for name, subschema in ORTHOGONAL_AXES.items():
        props.setdefault(name, subschema)
    return schema


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    # Clear stale generated files (do not delete the namespace dir itself).
    for old in DST.iterdir():
        if old.is_file():
            old.unlink()

    manifest_schemas = []
    for src in sorted(SRC.glob("*.schema.json")):
        data = json.loads(src.read_text(encoding="utf-8"))
        data = rebind(data, src.name)
        if src.name in CORE_ENTITIES:
            data = inject_axes(data)
        dst = DST / src.name
        # Deterministic, stable key order via sort_keys.
        ordered = json.loads(json.dumps(data, sort_keys=True))
        dst.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_schemas.append(
            {
                "$id": f"https://github.com/Cunyu-Liu/mRNA_editflow/schemas/xedit_v1_1/{src.name}",
                "contract_id": NEW_CONTRACT_ID,
                "filename": src.name,
                "schema_version": NEW_SCHEMA_VERSION,
                "sha256": sha256_file(dst),
            }
        )

    # Orthogonal-axes shared reference schema.
    axes_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Cunyu-Liu/mRNA_editflow/schemas/xedit_v1_1/xedit_orthogonal_axes.schema.json",
        "title": "OrthogonalAxes (xedit_v1_1)",
        "description": (
            "Shared orthogonal axes injected onto core data entities. "
            "Orthogonal to the retained scientific_track E/F/AUX/REFERENCE."
        ),
        "schema_version": NEW_SCHEMA_VERSION,
        "contract_id": NEW_CONTRACT_ID,
        "type": "object",
        "additionalProperties": False,
        "properties": ORTHOGONAL_AXES,
    }
    axes_path = DST / "xedit_orthogonal_axes.schema.json"
    axes_path.write_text(
        json.dumps(axes_schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_schemas.append(
        {
            "$id": axes_schema["$id"],
            "contract_id": NEW_CONTRACT_ID,
            "filename": axes_path.name,
            "schema_version": NEW_SCHEMA_VERSION,
            "sha256": sha256_file(axes_path),
        }
    )

    # Manifest.
    manifest = {
        "contract_id": NEW_CONTRACT_ID,
        "manifest_version": "1.1",
        "schema_count": len(manifest_schemas),
        "schema_version": NEW_SCHEMA_VERSION,
        "source_namespace": "v3_1",
        "source_manifest": "preserved",
        "schemas": sorted(manifest_schemas, key=lambda s: s["filename"]),
    }
    manifest_path = DST / "SCHEMA_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # SHA256SUMS over all schema files (deterministic sort).
    lines = []
    for p in sorted(DST.glob("*.schema.json")):
        lines.append(f"{sha256_file(p)}  {p.name}")
    (DST / "SCHEMA_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"namespace={DST} files={len(manifest_schemas)}")
    print(f"manifest_sha={sha256_file(manifest_path)}")
    for p in sorted(DST.iterdir()):
        print(f"  {p.name}  {sha256_file(p)}")


if __name__ == "__main__":
    main()