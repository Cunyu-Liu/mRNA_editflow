"""Draft 2020-12, closed-object and deterministic manifest tests."""

from __future__ import annotations

import json
import shutil


def test_six_public_schemas_and_manifest_validate(validator, repo_root):
    assert validator.validate_schema_manifest(repo_root) == []
    schema_dir = repo_root / validator.SCHEMA_DIR
    actual = sorted(path.name for path in schema_dir.glob("*.schema.json"))
    assert actual == sorted(validator.SCHEMA_FILES)


def test_every_explicit_object_schema_is_closed(validator, repo_root):
    for filename in validator.SCHEMA_FILES:
        document = json.loads((repo_root / validator.SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["additionalProperties"] is False
        assert validator._object_schemas_without_closed_properties(document) == []


def test_manifest_write_is_deterministic_and_scoped(validator, repo_root, tmp_path):
    target = tmp_path / "schemas" / "route_a_v3"
    target.mkdir(parents=True)
    for filename in validator.SCHEMA_FILES:
        shutil.copy2(repo_root / validator.SCHEMA_DIR / filename, target / filename)

    validator.write_schema_manifests(tmp_path)
    manifest_first = (target / "SCHEMA_MANIFEST.json").read_bytes()
    sums_first = (target / "SCHEMA_SHA256SUMS").read_bytes()
    validator.write_schema_manifests(tmp_path)
    assert (target / "SCHEMA_MANIFEST.json").read_bytes() == manifest_first
    assert (target / "SCHEMA_SHA256SUMS").read_bytes() == sums_first
    assert validator.validate_schema_manifest(tmp_path) == []


def test_default_manifest_validation_is_read_only(validator, repo_root):
    manifest = repo_root / validator.SCHEMA_MANIFEST
    sums = repo_root / validator.SCHEMA_SUMS
    before = (manifest.read_bytes(), sums.read_bytes())
    assert validator.validate_schema_manifest(repo_root) == []
    after = (manifest.read_bytes(), sums.read_bytes())
    assert after == before
