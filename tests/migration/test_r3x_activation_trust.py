"""Adversarial tests for the development-only activation trust v3."""

from __future__ import annotations

import copy
import hashlib
import hmac
import inspect
from pathlib import Path

import pytest

import scripts.r3x.activation_trust as trust_module
import scripts.r3x.staged_disclosure as staged_module
from scripts.r3x.activation_trust import (
    DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH,
    DEVELOPMENT_ACTIVATION_TRUST_SCHEMA,
    DEVELOPMENT_ACTIVATION_TRUST_STATUS,
    ActivationTrustError,
    canonical_development_activation_trust_bytes,
    load_development_activation_trust_exact,
    prepare_development_activation_trust_exclusive,
    validate_development_activation_trust_payload,
)
from scripts.r3x.stage_row_projections import (
    DEVELOPMENT_STAGE,
    FirewallReplayContext,
    ROW_AUTHORITIES,
    attest_pretraining_firewall,
    build_development_row_projections,
    build_internal_test_feature_row_projections,
    independent_firewall_verifier,
    register_trusted_firewall_verifier,
    stage_projection_hashes,
    verifier_code_sha256,
    write_pretraining_firewall_certificate_exclusive,
    write_stage_row_projection_exclusive,
)
from scripts.r3x.staged_disclosure import validate_development_stage
from scripts.r3x.typed_safe_authorities import (
    ANALYSIS_CODE_TREE_MANIFEST,
    SUPPORTED_AUTHORITIES,
    write_typed_safe_authority_exclusive,
)
from scripts.r3x.typed_staged_rows import (
    EXACT_FEATURE_FIELDS,
    EXACT_OUTCOME_FIELDS,
    normalize_development_disclosure,
)
from tests.migration import test_r3x_stage_row_projections as row_fixtures
from tests.migration import test_r3x_staged_disclosure as staged_fixtures
from tests.migration import test_r3x_typed_safe_authorities as typed_fixtures


def _ref(root: Path, path: Path, sha256: str) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256,
    }


def _all_strings(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _all_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_strings(nested)
    elif isinstance(value, str):
        yield value


def _build_integrated_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    # Reuse the already-adversarial fixture builders, but make the staged
    # train/development schema the exact primary paired-sequence schema.
    monkeypatch.setattr(
        staged_fixtures,
        "FEATURE_FIELDS",
        list(EXACT_FEATURE_FIELDS),
    )
    monkeypatch.setattr(
        staged_fixtures,
        "OUTCOME_FIELDS",
        list(EXACT_OUTCOME_FIELDS),
    )
    train_rows = [
        row_fixtures._staged_development_row(
            "train",
            component=row_fixtures.TRAIN_COMPONENT,
            study="GSE217518",
            context="hek293_arm_a",
            group="pool_train_a",
        ),
        row_fixtures._staged_development_row(
            "development",
            component=row_fixtures.DEVELOPMENT_COMPONENT,
            study="GSE217519",
            context="hela_arm_b",
            group="pool_development_a",
        ),
    ]
    monkeypatch.setattr(staged_fixtures, "TRAIN_ID", train_rows[0]["row_id"])
    monkeypatch.setattr(
        staged_fixtures,
        "DEVELOPMENT_ID",
        train_rows[1]["row_id"],
    )
    staged = staged_fixtures._build_fixture(
        tmp_path,
        train_development_rows=train_rows,
    )
    disclosure = validate_development_stage(staged.manifest_path, staged.manifest_sha)
    typed_development = normalize_development_disclosure(disclosure)

    sources = {
        name: staged.manifest["nondata_authorities"][name]["source_sha256"]
        for name in row_fixtures.SOURCES
    }
    train_sha = staged.manifest["authorities"]["train_development_outcomes"]["sha256"]
    test_feature_sha = staged.manifest["authorities"]["internal_test_features"][
        "sha256"
    ]
    development_projections = build_development_row_projections(
        typed_development,
        source_authority_sha256=sources,
        train_development_outcomes_sha256=train_sha,
        split_seed=row_fixtures.SPLIT_SEED,
    )
    typed_test = row_fixtures._typed_test()
    test_projections = build_internal_test_feature_row_projections(
        typed_test,
        source_authority_sha256=sources,
        internal_test_features_sha256=test_feature_sha,
        split_seed=row_fixtures.SPLIT_SEED,
    )
    context = FirewallReplayContext(
        development_projections=development_projections,
        internal_test_feature_projections=test_projections,
        typed_development_rows=typed_development,
        typed_internal_test_feature_rows=typed_test,
        source_authority_sha256=sources,
        train_development_outcomes_sha256=train_sha,
        internal_test_features_sha256=test_feature_sha,
        split_seed=row_fixtures.SPLIT_SEED,
    )
    verifier_sha = verifier_code_sha256(independent_firewall_verifier)
    registration = register_trusted_firewall_verifier(
        row_fixtures.VERIFIER_ID,
        independent_firewall_verifier,
        expected_code_sha256=verifier_sha,
    )
    certificate = attest_pretraining_firewall(
        context,
        verifier_id=row_fixtures.VERIFIER_ID,
        verifier_registry={row_fixtures.VERIFIER_ID: registration},
    )

    row_dir = tmp_path / "development" / "row_authorities"
    row_dir.mkdir()
    row_refs = {}
    for name in ROW_AUTHORITIES:
        path = row_dir / f"{name}.json"
        digest = write_stage_row_projection_exclusive(
            tmp_path,
            path,
            development_projections[name],
        )
        row_refs[name] = _ref(tmp_path, path, digest)

    firewall_dir = tmp_path / "authority_preparation" / "firewall"
    firewall_dir.mkdir(parents=True)
    firewall_path = firewall_dir / "pretraining-firewall.json"
    firewall_sha = write_pretraining_firewall_certificate_exclusive(
        tmp_path,
        firewall_path,
        certificate,
    )

    code_path = tmp_path / "scripts" / "r3x" / "train_model.py"
    code_path.parent.mkdir(parents=True)
    code_path.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    code_snapshot = typed_fixtures._trusted_code_snapshot(tmp_path)
    development_hashes = stage_projection_hashes(
        development_projections,
        stage=DEVELOPMENT_STAGE,
    )
    bundle = typed_fixtures._bundle(tmp_path)
    typed_refs = {}
    authorities_dir = tmp_path / "development" / "authorities"
    authorities_dir.mkdir()
    for name in SUPPORTED_AUTHORITIES:
        payload = bundle[name]
        payload["source_authority_sha256"] = staged.manifest["nondata_authorities"][
            name
        ]["source_sha256"]
        payload["train_development_outcomes_sha256"] = train_sha
        payload["development_projection_hashes"] = development_hashes
        path = authorities_dir / f"{name}.json"
        digest = write_typed_safe_authority_exclusive(
            tmp_path,
            path,
            payload,
            trusted_code_snapshot=(
                code_snapshot if name == ANALYSIS_CODE_TREE_MANIFEST else None
            ),
        )
        typed_refs[name] = _ref(tmp_path, path, digest)

    commitment_key_path = (
        tmp_path / trust_module.FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH
    )
    commitment_key_path.parent.mkdir(parents=True)
    commitment_key = hashlib.sha256(
        b"activation-trust-test-only-privileged-hmac-key"
    ).digest()
    commitment_key_path.write_bytes(commitment_key)
    commitment_key_path.chmod(0o600)
    commitment_key_sha = hashlib.sha256(commitment_key).hexdigest()

    trust_dir = tmp_path / DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.parent
    trust_dir.mkdir()
    return {
        "root": tmp_path,
        "staged": staged,
        "activation_ref": _ref(tmp_path, staged.manifest_path, staged.manifest_sha),
        "firewall_path": firewall_path,
        "firewall_ref": _ref(tmp_path, firewall_path, firewall_sha),
        "row_refs": row_refs,
        "typed_refs": typed_refs,
        "code_snapshot": code_snapshot,
        "development_hashes": development_hashes,
        "commitment_key_path": commitment_key_path,
        "commitment_key": commitment_key,
        "commitment_key_sha": commitment_key_sha,
    }


def _prepare(fixture: dict) -> dict:
    return prepare_development_activation_trust_exclusive(
        fixture["root"],
        activation_manifest_ref=fixture["activation_ref"],
        firewall_certificate_ref=fixture["firewall_ref"],
        typed_authority_refs=fixture["typed_refs"],
        development_row_projection_refs=fixture["row_refs"],
        trusted_code_snapshot=fixture["code_snapshot"],
        future_stage_commitment_key_sha256=fixture["commitment_key_sha"],
    )


def test_preparation_publishes_safe_v3_and_runtime_exact_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)
    payload = result["payload"]
    assert payload["schema_version"] == DEVELOPMENT_ACTIVATION_TRUST_SCHEMA
    assert payload["status"] == DEVELOPMENT_ACTIVATION_TRUST_STATUS
    assert result["development_activation_trust_binding_complete"] is True
    assert result["training_ready"] is False
    assert payload["training_ready"] is False
    assert payload["sign_accuracy_gate"] == 0.60
    assert payload["ordinary_only"] is True
    assert payload["sealed_access"] is False
    assert payload["downstream_unlock"] is False
    assert payload["internal_test_feature_access"] is False
    assert payload["internal_test_outcome_access"] is False
    assert payload["overwrite_authorized"] is False
    assert payload["generic_summary_used_as_rich_projection"] is False
    assert payload["legacy_full_staged_activation_training_ready"] is False
    assert payload["legacy_full_staged_activation_ref_exposed"] is False
    assert payload["development_projection_hashes"] == fixture["development_hashes"]
    assert (
        "FUTURE_STAGE_HMAC_KEY_PREBINDING_NOT_INDEPENDENTLY_ATTESTED"
        in payload["blockers"]
    )
    assert "FUTURE_STAGE_HMAC_KEY_SECRECY_NOT_OS_ENFORCED" in payload["blockers"]
    expected_commitment = hmac.new(
        fixture["commitment_key"],
        trust_module.FUTURE_STAGE_COMMITMENT_DOMAIN
        + bytes.fromhex(fixture["firewall_ref"]["sha256"]),
        digestmod=hashlib.sha256,
    ).hexdigest()
    assert payload["opaque_future_stage_commitment"] == {
        "scheme": trust_module.FUTURE_STAGE_COMMITMENT_SCHEME,
        "commitment": expected_commitment,
    }

    # Development v3 contains no old/full activation ref, firewall path, test
    # path/hash, HMAC-key path/hash, test data hash, or test projection support.
    assert "activation_manifest_ref" not in payload
    assert "firewall_certificate_ref" not in payload
    assert "internal_test_feature_projection_hashes" not in payload
    assert "opaque_future_stage_commitment_sha256" not in payload
    assert not any("internal_test" in value.lower() for value in _all_strings(payload))
    assert not any(
        value == fixture["staged"].manifest["authorities"][name]["sha256"]
        for value in _all_strings(payload)
        for name in ("internal_test_features", "internal_test_outcomes")
    )
    disclosed_strings = set(_all_strings(payload))
    assert fixture["firewall_ref"]["relative_path"] not in disclosed_strings
    assert fixture["firewall_ref"]["sha256"] not in disclosed_strings
    assert (
        trust_module.FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH.as_posix()
        not in disclosed_strings
    )
    assert fixture["commitment_key_sha"] not in disclosed_strings

    loaded = load_development_activation_trust_exact(
        tmp_path,
        result["sha256"],
        trusted_code_snapshot=fixture["code_snapshot"],
    )
    assert loaded == payload
    assert Path(result["path"]).read_bytes() == (
        canonical_development_activation_trust_bytes(payload)
    )


def test_runtime_does_not_need_or_return_old_activation_firewall_or_test_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)

    # Remove only temporary test-fixture material that a development process
    # must not need.  The v3 runtime replay must remain fully functional.
    fixture["staged"].manifest_path.unlink()
    fixture["firewall_path"].unlink()
    fixture["commitment_key_path"].unlink()
    fixture["staged"].feature_path.unlink()
    fixture["staged"].outcome_path.unlink()
    loaded = load_development_activation_trust_exact(
        tmp_path,
        result["sha256"],
        trusted_code_snapshot=fixture["code_snapshot"],
    )
    assert loaded == result["payload"]
    assert isinstance(loaded, dict)
    assert not isinstance(loaded, staged_module.StagedActivation)


def test_hmac_commitment_is_not_an_offline_low_entropy_candidate_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)
    disclosed = result["payload"]["opaque_future_stage_commitment"]
    observed = disclosed["commitment"]

    # A development caller knows the scheme/domain and may have a tiny list of
    # candidate certificate digests.  None can be checked with a bare SHA-256;
    # verification requires the privileged key that is absent from the
    # artifact and absent from the runtime API.
    candidate_hashes = [
        fixture["firewall_ref"]["sha256"],
        hashlib.sha256(b"low-entropy-firewall-candidate-a").hexdigest(),
        hashlib.sha256(b"low-entropy-firewall-candidate-b").hexdigest(),
    ]
    for candidate in candidate_hashes:
        candidate_bytes = bytes.fromhex(candidate)
        assert hashlib.sha256(candidate_bytes).hexdigest() != observed
        assert (
            hashlib.sha256(
                trust_module.FUTURE_STAGE_COMMITMENT_DOMAIN + candidate_bytes
            ).hexdigest()
            != observed
        )
    for guessed_key_label in (b"guess-a", b"guess-b", b"guess-c"):
        guessed_key = hashlib.sha256(guessed_key_label).digest()
        guessed_tag = hmac.new(
            guessed_key,
            trust_module.FUTURE_STAGE_COMMITMENT_DOMAIN
            + bytes.fromhex(fixture["firewall_ref"]["sha256"]),
            digestmod=hashlib.sha256,
        ).hexdigest()
        assert guessed_tag != observed

    assert set(disclosed) == {"scheme", "commitment"}
    runtime_parameters = inspect.signature(
        load_development_activation_trust_exact
    ).parameters
    assert "future_stage_commitment_key_sha256" not in runtime_parameters
    assert "firewall_certificate_ref" not in runtime_parameters


def test_preparation_exact_reads_one_fixed_32_byte_hmac_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    preparation_parameters = inspect.signature(
        prepare_development_activation_trust_exclusive
    ).parameters
    assert "future_stage_commitment_key_sha256" in preparation_parameters
    assert "future_stage_commitment_key_path" not in preparation_parameters

    with pytest.raises(ActivationTrustError) as exc_info:
        prepare_development_activation_trust_exclusive(
            tmp_path,
            activation_manifest_ref=fixture["activation_ref"],
            firewall_certificate_ref=fixture["firewall_ref"],
            typed_authority_refs=fixture["typed_refs"],
            development_row_projection_refs=fixture["row_refs"],
            trusted_code_snapshot=fixture["code_snapshot"],
            future_stage_commitment_key_sha256="0" * 64,
        )
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_HASH_MISMATCH"

    short_key = b"x" * 31
    fixture["commitment_key_path"].write_bytes(short_key)
    fixture["commitment_key_path"].chmod(0o600)
    with pytest.raises(ActivationTrustError) as exc_info:
        prepare_development_activation_trust_exclusive(
            tmp_path,
            activation_manifest_ref=fixture["activation_ref"],
            firewall_certificate_ref=fixture["firewall_ref"],
            typed_authority_refs=fixture["typed_refs"],
            development_row_projection_refs=fixture["row_refs"],
            trusted_code_snapshot=fixture["code_snapshot"],
            future_stage_commitment_key_sha256=hashlib.sha256(short_key).hexdigest(),
        )
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_COMMITMENT_KEY"


def test_preparation_and_runtime_never_open_internal_test_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    reads: list[Path] = []
    staged_reader = staged_module.read_exact_bytes
    runtime_reader = trust_module.read_exact_bytes

    def spy_staged(path, expected_sha256, *, purpose):
        reads.append(Path(path))
        return staged_reader(path, expected_sha256, purpose=purpose)

    def spy_runtime(path, expected_sha256, *, purpose):
        reads.append(Path(path))
        return runtime_reader(path, expected_sha256, purpose=purpose)

    monkeypatch.setattr(staged_module, "read_exact_bytes", spy_staged)
    monkeypatch.setattr(trust_module, "read_exact_bytes", spy_runtime)
    result = _prepare(fixture)
    load_development_activation_trust_exact(
        tmp_path,
        result["sha256"],
        trusted_code_snapshot=fixture["code_snapshot"],
    )
    resolved_reads = {path.resolve() for path in reads}
    assert fixture["staged"].outcome_path.resolve() not in resolved_reads
    assert fixture["staged"].feature_path.resolve() not in resolved_reads
    assert fixture["staged"].train_development_path.resolve() in resolved_reads


def test_generic_summary_cannot_substitute_for_rich_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    name = ROW_AUTHORITIES[0]
    generic_ref = fixture["staged"].manifest["nondata_authorities"][name][
        "development_projection"
    ]
    generic_path = tmp_path / generic_ref["path"]
    forged_path = tmp_path / "development" / "row_authorities" / "generic.json"
    forged_path.write_bytes(generic_path.read_bytes())
    refs = copy.deepcopy(fixture["row_refs"])
    refs[name] = _ref(
        tmp_path,
        forged_path,
        hashlib.sha256(forged_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ActivationTrustError) as exc_info:
        prepare_development_activation_trust_exclusive(
            tmp_path,
            activation_manifest_ref=fixture["activation_ref"],
            firewall_certificate_ref=fixture["firewall_ref"],
            typed_authority_refs=fixture["typed_refs"],
            development_row_projection_refs=refs,
            trusted_code_snapshot=fixture["code_snapshot"],
            future_stage_commitment_key_sha256=fixture["commitment_key_sha"],
        )
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_DEPENDENCY"


def test_hash_self_report_cannot_replace_exact_loader_and_output_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    bad_refs = copy.deepcopy(fixture["typed_refs"])
    bad_refs[SUPPORTED_AUTHORITIES[0]][
        "relative_path"
    ] = "development/authorities/missing.json"
    with pytest.raises(ActivationTrustError) as exc_info:
        prepare_development_activation_trust_exclusive(
            tmp_path,
            activation_manifest_ref=fixture["activation_ref"],
            firewall_certificate_ref=fixture["firewall_ref"],
            typed_authority_refs=bad_refs,
            development_row_projection_refs=fixture["row_refs"],
            trusted_code_snapshot=fixture["code_snapshot"],
            future_stage_commitment_key_sha256=fixture["commitment_key_sha"],
        )
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_DEPENDENCY"

    result = _prepare(fixture)
    with pytest.raises(ActivationTrustError) as exc_info:
        _prepare(fixture)
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_WRITE"
    with pytest.raises(ActivationTrustError) as exc_info:
        load_development_activation_trust_exact(
            tmp_path,
            "0" * 64,
            trusted_code_snapshot=fixture["code_snapshot"],
        )
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_HASH_MISMATCH"
    assert result["payload"]["training_ready"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sign_accuracy_gate", 0.59),
        ("sign_accuracy_gate", 0.61),
        ("sign_gate_change_authorized", True),
        ("training_ready", True),
        ("sealed_access", True),
        ("downstream_unlock", True),
        ("future_stage_access_authorized", True),
        ("internal_test_feature_access", True),
        ("internal_test_outcome_access", True),
        ("overwrite_authorized", True),
        ("generic_summary_used_as_rich_projection", True),
    ],
)
def test_gate_capability_and_generic_summary_mutations_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)
    tampered = copy.deepcopy(result["payload"])
    tampered[field] = value
    with pytest.raises(ActivationTrustError):
        validate_development_activation_trust_payload(tampered)


def test_unknown_old_activation_or_firewall_ref_field_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)
    for field, value in (
        ("activation_manifest_ref", fixture["activation_ref"]),
        ("firewall_certificate_ref", fixture["firewall_ref"]),
        ("internal_test_feature_projection_hashes", {"x": "0" * 64}),
        (
            "opaque_future_stage_commitment_sha256",
            fixture["firewall_ref"]["sha256"],
        ),
        (
            "future_stage_commitment_key_ref",
            {
                "relative_path": trust_module.FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH.as_posix(),
                "sha256": fixture["commitment_key_sha"],
            },
        ),
    ):
        tampered = copy.deepcopy(result["payload"])
        tampered[field] = value
        with pytest.raises(ActivationTrustError) as exc_info:
            validate_development_activation_trust_payload(tampered)
        assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_SCHEMA"


def test_future_stage_commitment_requires_exact_hmac_scheme_and_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_integrated_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture)

    tampered = copy.deepcopy(result["payload"])
    tampered["opaque_future_stage_commitment"]["scheme"] = "SHA256"
    with pytest.raises(ActivationTrustError) as exc_info:
        validate_development_activation_trust_payload(tampered)
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_SCHEMA"

    tampered = copy.deepcopy(result["payload"])
    tampered["opaque_future_stage_commitment"]["key_sha256"] = fixture[
        "commitment_key_sha"
    ]
    with pytest.raises(ActivationTrustError) as exc_info:
        validate_development_activation_trust_payload(tampered)
    assert exc_info.value.code == "HOLD_ACTIVATION_TRUST_SCHEMA"
